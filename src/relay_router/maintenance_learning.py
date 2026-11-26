"""Fit and freeze an existing trajectory predictor for paired maintenance runs.

Only the unchanged routing text enters X. Outcomes select operating points on
development after fit/calibration; this module never evaluates a holdout.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from .agent import AgentRequest, AgentStateV8
from .trajectory import TrajectoryValuePolicy
from .maintenance_comparison import select_reference_threshold
from routerbench.agent_trajectory_learning import TARGET, TrajectorySuccessPredictor, TrajectoryTrainingRow
from routerbench.features import normalize_for_dedup


MODELS = ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")
CONTINUATION_POLICY = "shared_inspection_20b_first_edit_candidate_max_one_repair_120b_v1"
COST_BASIS = "noncached_tariff_scenario_usd_including_shared_inspection_and_all_branch_calls"
GRID = (0., .05) + tuple(index / 10 for index in range(1, 11))
REFERENCE_GRID = tuple(index / 10 for index in range(11)) + (1.000001,)
NAME = "relay_maintenance_trajectory_v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Se requiere SHA-256 externo de confianza.")
    return value


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} debe ser texto no vacío.")
    return value


def _number(value, label, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{label} debe ser finito dentro del rango permitido.")
    return float(value)


def _request(text):
    return AgentRequest(_text(text, "routing_text"), (), AgentStateV8(
        step=1, attempt=0, previous_errors=(), available_tools=(), remaining_calls=2))


def _prompt_digest(text):
    """Dedup independent of inspection plans and transient test diagnostics."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict) and isinstance(value.get("task"), str):
        # File inputs distinguish different repositories sharing an instruction.
        # This projection is only for dedup; predictor text stays unchanged.
        text = _json({key: value.get(key) for key in ("task", "editable_files", "read_only_tests")})
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def _acquisition_contract(records):
    present = ["acquisition_contract" in row for row in records]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Todos los registros deben declarar el mismo acquisition_contract.")
    contract = records[0]["acquisition_contract"]
    fields = {"suite_sha256", "tasks_sha256", "tariffs_sha256", "sources_sha256", "scope", "models", "continuation"}
    if type(contract) is not dict or set(contract) != fields:
        raise ValueError("El acquisition_contract debe declarar suite, tarifas, fuentes, alcance y modelos.")
    _digest(contract["suite_sha256"])
    _digest(contract["tasks_sha256"])
    _digest(contract["tariffs_sha256"])
    if type(contract["sources_sha256"]) is not dict or not contract["sources_sha256"]:
        raise ValueError("Se requieren hashes de las fuentes de adquisición.")
    for name, sha in contract["sources_sha256"].items():
        _text(name, "source filename")
        _digest(sha)
    if _text(contract["scope"], "scope") == "diagnostic":
        raise ValueError("Una adquisición diagnostic no puede entrenar ni validar la política.")
    if contract["models"] != {"weak": MODELS[0], "strong": MODELS[1]} or not contract["continuation"]:
        raise ValueError("El contrato de adquisición requiere el catálogo y la continuación explícitos.")
    canonical = _json(contract)
    if any(_json(row["acquisition_contract"]) != canonical for row in records):
        raise ValueError("Los registros mezclan distintos contratos de adquisición.")
    return deepcopy(contract)


def _validate(records):
    if type(records) is not list or not records:
        raise ValueError("Se requieren registros observados no vacíos.")
    if any(type(row) is not dict for row in records):
        raise ValueError("Cada registro debe ser un objeto.")
    _acquisition_contract(records)
    seen_tasks, seen_prompts, groups = set(), set(), {}
    parts = {name: [] for name in ("fit", "calibration", "development")}
    for record in records:
        if type(record) is not dict or record.get("partition") not in parts:
            raise ValueError("Solo se admite fit, calibration y development; holdout está reservado.")
        task = _text(record.get("task_id"), "task_id")
        group = _text(record.get("group_id"), "group_id")
        text = _text(record.get("routing_text"), "routing_text")
        if group != group.strip() or task in seen_tasks:
            raise ValueError("Los task_id deben ser únicos y group_id exacto.")
        partition = record["partition"]
        if group in groups and groups[group] != partition:
            raise ValueError("Los grupos deben ser disjuntos entre particiones.")
        normalized = _prompt_digest(text)
        if normalized in seen_prompts:
            raise ValueError("Hay prompts normalizados duplicados; deduplicar antes de entrenar.")
        seen_tasks.add(task)
        seen_prompts.add(normalized)
        groups[group] = partition
        _number(record.get("reference_probability"), "reference_probability", 1.)
        for field, expected in (("model_catalog", list(MODELS)),
                                ("continuation_policy", CONTINUATION_POLICY), ("cost_basis", COST_BASIS)):
            if field in record and record[field] != expected:
                raise ValueError(f"{field} difiere del contrato de mantenimiento fijado.")
        branches = record.get("branches")
        if type(branches) is not dict or set(branches) != {"weak", "strong"}:
            raise ValueError("Cada tarea requiere las dos continuaciones weak y strong.")
        for name in ("weak", "strong"):
            branch = branches[name]
            if type(branch) is not dict or type(branch.get("success")) is not bool:
                raise ValueError("success debe ser un resultado booleano observado.")
            _number(branch.get("cost_usd"), "cost_usd")
        parts[partition].append(deepcopy(record))
    if any(not values for values in parts.values()):
        raise ValueError("Se requieren las tres particiones fit, calibration y development.")
    return parts


def _training_rows(records):
    return [TrajectoryTrainingRow(_request(record["routing_text"]), model,
            record["branches"][choice]["success"], record["branches"][choice]["cost_usd"],
            record["group_id"], CONTINUATION_POLICY, "paired_state", TARGET, COST_BASIS)
            for record in records for choice, model in zip(("weak", "strong"), MODELS)]


def _sources():
    source = Path(__file__)
    return {name: _hash(source.with_name(name)) for name in (
        "maintenance_learning.py", "trajectory.py", "v8.py")}


def _select_relay(predictor, development):
    strong_rate = sum(row["branches"]["strong"]["success"] for row in development) / len(development)
    candidates = []
    for threshold in GRID:
        policy = TrajectoryValuePolicy(predictor, threshold, "maintenance_development_selection")
        selected = [row["branches"]["weak" if policy.select(_request(row["routing_text"])) == MODELS[0]
                                    else "strong"] for row in development]
        rate = sum(branch["success"] for branch in selected) / len(selected)
        candidates.append({"min_probability": threshold, "success_rate": rate,
                           "total_cost_usd": math.fsum(branch["cost_usd"] for branch in selected),
                           "eligible": rate + 1e-12 >= strong_rate})
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise ValueError("Ningún punto Relay cumple el éxito del fuerte en desarrollo; recoger evidencia nueva.")
    selected = min(eligible, key=lambda row: (row["total_cost_usd"], -row["success_rate"], row["min_probability"]))
    return {"min_probability": selected["min_probability"], "strong_success_rate": strong_rate,
            "quality_loss_margin": 0., "selection_scope": "development_only_empirical_not_noninferiority",
            "tie_break": "minimum cost, maximum success, minimum threshold", "candidates": candidates}


def train_policy(records, output_dir) -> dict:
    """Train on fit, calibrate separately, select both thresholds on development.

    All branch costs must include the shared inspection plus the initial edit
    and any allowed repair, valued with the same noncached tariff scenario.
    These are scenario costs, not Groq Free invoices or full operating costs.
    The caller must retain the returned policy SHA outside the artifact bundle.
    """
    parts = _validate(records)
    acquisition_contract = _acquisition_contract(records)
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Usa una carpeta nueva o vacía para la política congelada.")
    started = time.perf_counter()
    predictor = TrajectorySuccessPredictor(MODELS).fit(
        _training_rows(parts["fit"]), _training_rows(parts["calibration"]))
    diagnostics = predictor.evaluate(_training_rows(parts["development"]))
    selection = _select_relay(predictor, parts["development"])
    reference_rows = [{key: row[key] for key in (
        "task_id", "group_id", "partition", "routing_text", "reference_probability")} |
        {"branches": {name: {"success": branch["success"], "cost_usd": branch["cost_usd"],
                              "total_tokens": branch.get("total_tokens")}
                      for name, branch in row["branches"].items()}} for row in parts["development"]]
    reference_selection = select_reference_threshold(reference_rows, quality_loss_margin=0., grid=REFERENCE_GRID)
    training_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "predictor.joblib"
    artifact_sha = predictor.save(artifact)
    manifest = {"schema_version": 1, "name": NAME, "frozen": True, "learned": True,
        "model_catalog": list(MODELS), "continuation_policy": CONTINUATION_POLICY, "cost_basis": COST_BASIS,
        "min_probability": selection["min_probability"], "reference_threshold": reference_selection["threshold"],
        "threshold_grid": list(GRID), "reference_threshold_grid": list(REFERENCE_GRID), "quality_loss_margin": 0.,
        "threshold_search_budget": {"relay": len(GRID), "reference": len(REFERENCE_GRID)},
        "reference_endpoint_note": "Threshold 1.000001 includes always-weak even when preference equals 1; twelve points per policy.",
        "selection": selection, "reference_selection": reference_selection,
        "development_diagnostics": diagnostics, "training_seconds": training_seconds,
        "artifact": artifact.name, "artifact_sha256": artifact_sha,
        "artifact_manifest_sha256": _hash(artifact.with_name(artifact.name + ".manifest.json")),
        "predictor_manifest": predictor.manifest(), "source_sha256": _sources(),
        "partition_counts": {name: len(values) for name, values in parts.items()},
        "dedup_prompt_sha256": {name: sorted(_prompt_digest(row["routing_text"]) for row in values)
                                for name, values in parts.items()},
        "dedup_rule": "NFKC/casefold/whitespace; JSON task and source/test files exclude plans and diagnostics",
        "holdout_used": False, "independent_validation": False,
        "acquisition_contract": acquisition_contract,
        "data_provenance": "acquisition_contract_bound_by_policy_hash" if acquisition_contract is not None else "unverified_injected_records",
        "request_contract": "routing_text only; unchanged prompt, empty history, constant public state",
        "cost_target_note": "Episode scenario cost includes common inspection already paid at the edit decision.",
        "scope": "learned first-edit choice with fixed repair continuation; pretrained BERT transfer comparison"}
    policy_path = output / "policy.json"
    policy_path.write_text(_json(manifest), encoding="utf-8")
    return {"policy_path": str(policy_path.resolve()), "policy_sha256": _hash(policy_path),
            "predictor_path": str(artifact.resolve()), "predictor_sha256": artifact_sha,
            "selection": selection, "reference_selection": reference_selection,
            "development_diagnostics": diagnostics}


class MaintenanceLearnedPolicy:
    """Frozen learned edit choice; load via externally hashed policy.json."""

    def __init__(self, predictor, manifest, policy_sha256):
        self._metadata = deepcopy(manifest)
        self._policy_sha256 = policy_sha256
        self._policy = TrajectoryValuePolicy(predictor, manifest["min_probability"],
                                             "maintenance_development_selection")

    def choose(self, routing_text) -> dict:
        request = _request(routing_text)
        started = time.perf_counter()
        selected = self._policy.select(request)
        seconds = time.perf_counter() - started
        audit = self._policy.last_audit
        return {"choice": "weak" if selected == MODELS[0] else "strong", "router_seconds": seconds,
                "forecasts": {model: {"success_probability": audit["probabilities"][model],
                    "remaining_provider_cost_usd": audit["expected_remaining_provider_cost_usd"][model]}
                              for model in MODELS}}

    def manifest(self) -> dict:
        self._policy.manifest()  # Enforce predictor and threshold immutability.
        return {**deepcopy(self._metadata), "policy_sha256": self._policy_sha256}

    def audit_records(self, records, *, evaluation_split="holdout") -> dict:
        if (self._metadata.get("acquisition_contract") is not None
                and _acquisition_contract(records) != self._metadata["acquisition_contract"]):
            raise ValueError("Los registros de evaluación no corresponden al acquisition_contract congelado.")
        protected = ("fit", "calibration", "development") if evaluation_split == "holdout" else ("fit", "calibration")
        existing = {digest for name in protected for digest in self._metadata["dedup_prompt_sha256"][name]}
        if any(_prompt_digest(_text(row.get("routing_text"), "routing_text")) in existing for row in records):
            raise ValueError("Evaluation prompts overlap normalized fit/calibration/development task inputs.")
        return self._policy.predictor.audit_task_groups(
            [{"source_group": row["group_id"], "prompt": row["routing_text"]} for row in records],
            evaluation_split=evaluation_split)


def load_policy(path, expected_sha256) -> MaintenanceLearnedPolicy:
    """Verify external policy hash, bound artifacts and contracts before joblib."""
    path = Path(path)
    expected_sha256 = _digest(expected_sha256)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Policy SHA-256 mismatch; no se deserializó el predictor.")
    manifest = json.loads(payload)
    expected = {"schema_version": 1, "name": NAME, "frozen": True, "learned": True, "model_catalog": list(MODELS),
                "continuation_policy": CONTINUATION_POLICY, "cost_basis": COST_BASIS,
                "threshold_grid": list(GRID), "reference_threshold_grid": list(REFERENCE_GRID),
                "threshold_search_budget": {"relay": len(GRID), "reference": len(REFERENCE_GRID)},
                "quality_loss_margin": 0., "holdout_used": False,
                "source_sha256": _sources(), "artifact": "predictor.joblib"}
    if type(manifest) is not dict or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("El contrato o fuente de la política congelada cambió.")
    for field, grid in (("min_probability", GRID), ("reference_threshold", REFERENCE_GRID)):
        if _number(manifest.get(field), field, max(grid)) not in grid:
            raise ValueError("Los puntos operativos deben pertenecer a la grilla fijada.")
    if (manifest["min_probability"] != manifest.get("selection", {}).get("min_probability")
            or manifest["reference_threshold"] != manifest.get("reference_selection", {}).get("threshold")):
        raise ValueError("Los puntos operativos difieren de la selección congelada.")
    contract = manifest.get("acquisition_contract")
    if contract is not None:
        _acquisition_contract([{"acquisition_contract": contract}])
    provenance = "acquisition_contract_bound_by_policy_hash" if contract is not None else "unverified_injected_records"
    if manifest.get("data_provenance") != provenance:
        raise ValueError("La procedencia de datos no coincide con el contrato congelado.")
    artifact = path.parent / manifest["artifact"]
    if _hash(artifact.with_name(artifact.name + ".manifest.json")) != _digest(manifest.get("artifact_manifest_sha256")):
        raise ValueError("El manifiesto del predictor cambió antes de deserializar.")
    predictor = TrajectorySuccessPredictor.load(artifact, expected_sha256=_digest(manifest.get("artifact_sha256")))
    if predictor.models != MODELS or predictor.manifest() != manifest.get("predictor_manifest"):
        raise ValueError("El predictor cargado difiere de la política congelada.")
    return MaintenanceLearnedPolicy(predictor, manifest, expected_sha256)
