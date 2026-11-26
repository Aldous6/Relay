"""Paired interventions at public agent states, followed by a fixed cascade.

Only new, completed continuations create episode labels. Source outcomes never
select anchors. Prefix replay invokes in-memory tools, not generated code or a
final grader. Preparing a collection does not invoke its generation callback.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import zipfile

from relay_router.agent import AgentRequest, AgentStateV8, AlwaysStrong, parse_action
from relay_router.v8 import NReactiveCascade, model_order
from .agent_context import compact_progress
from .agent_evaluation import (
    _Journal, _execute, _fingerprint, _read, _recover, _seal, _single_writer,
    _source_bundle, _sources, _unseal, _validate, _write, object_hash,
)
from .agent_metrics import annotate_costs
from .agent_tasks import create_environment, validate_task_splits

TARGET = "episode_success_under_reference_continuation"
CONTINUATION = "n_reactive_cascade_after_probe_v1"
COST_BASIS = "provider_token_tariff_no_cache"


def request_from_dict(value):
    """Strict public request decoder; no hidden fields survive normalization."""
    if type(value) is not dict or set(value) != {"prompt", "history", "state"}:
        raise ValueError("Se requiere la solicitud publica completa.")
    state = deepcopy(value["state"])
    if state.pop("context_length_unit", None) != "utf8_bytes_prompt_and_history":
        raise ValueError("Unidad de contexto v8 desconocida.")
    for name in ("previous_errors", "available_tools", "infrastructure_errors", "failed_tiers"):
        if type(state.get(name)) is not list:
            raise ValueError("Listas de estado v8 ausentes.")
        state[name] = tuple(state[name])
    request = AgentRequest(value["prompt"], tuple(value["history"]), AgentStateV8(**state))
    if request.to_dict() != value:
        raise ValueError("La solicitud no conserva su forma canonica.")
    return request


def _initial_request(task, config, environment_factory):
    environment = environment_factory(deepcopy(task))
    tools, observation = environment.tools(), environment.observation()
    public = {k: v for k, v in observation.items() if k not in {"task_id", "family", "source_group", "id"}}
    history = ({"role": "tool", "content": {"name": "environment", "observation": public, "tools": tools}},)
    size = len(json.dumps({"prompt": task["prompt"], "history": history}, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8"))
    state = AgentStateV8(0, 0, (), tuple(t["name"] for t in tools), min(config["max_calls"], task["max_steps"]),
                         (), 0, size, compact_progress(observation, []), ())
    return AgentRequest(task["prompt"], history, state)


def initial_anchors(tasks, config, *, environment_factory=create_environment):
    """Declare all task-initial states without acquiring source trajectories."""
    _validate(config, [AlwaysStrong(config["models"]["strong"], config["models"]["weak"])])
    if config["schema_version"] != 2:
        raise ValueError("Las ramas requieren contrato v8.")
    validate_task_splits({config["evaluation_split"]: tasks})
    anchors = []
    for task in tasks:
        request = _initial_request(task, config, environment_factory).to_dict()
        anchors.append({"task_id": task["id"], "source_group": task["source_group"],
                        "request": request, "request_sha256": object_hash(request), "decision_index": 0,
                        "source": {"kind": "task_initial_state", "task_sha256": object_hash(task)}})
    return {"schema_version": 1, "anchors": anchors, "unavailable": [], "provenance": {
        "kind": "task_initial_state", "tasks_sha256": object_hash(tasks), "decision_indices": [0],
        "model_catalog": config["model_catalog"], "environment_source": _fingerprint(environment_factory)}}


def replay_prefix(task, request, environment_factory=create_environment):
    """Recreate public tool effects only, rejecting divergent observations.

    The environment final evaluator is never called. The restored environment
    is independent of the source trajectory and of every other intervention.
    """
    if not isinstance(request, AgentRequest) or not isinstance(request.state, AgentStateV8):
        raise ValueError("Un prefijo requiere AgentRequest v8.")
    if request.prompt != task["prompt"] or not request.history or request.state.infrastructure_attempt:
        raise ValueError("Prompt/prefijo incompatible o retry de transporte pendiente.")
    environment = environment_factory(deepcopy(task))
    tools, observation = environment.tools(), environment.observation()
    names = tuple(t["name"] for t in tools)
    public = {k: v for k, v in observation.items() if k not in {"task_id", "family", "source_group", "id"}}
    first = {"role": "tool", "content": {"name": "environment", "observation": public, "tools": tools}}
    if request.history[0] != first or request.state.available_tools != names:
        raise ValueError("La observacion inicial o las herramientas del prefijo cambiaron.")
    events, errors, position = [], [], 1
    while position < len(request.history):
        entry = request.history[position]
        if entry["role"] == "error":
            content = entry["content"]
            if type(content) is not dict or set(content) != {"error_type"} or type(content["error_type"]) is not str:
                raise ValueError("Error publico mal formado.")
            errors.append(content["error_type"])
            position += 1
            continue
        if entry["role"] != "assistant":
            raise ValueError("Resultado de herramienta sin accion previa.")
        position += 1
        try:
            action = parse_action(entry["content"], names)
        except (ValueError, TypeError):
            if position == len(request.history) or request.history[position]["role"] != "error":
                raise ValueError("Accion invalida sin error observado.") from None
            continue
        if action["type"] == "final":
            raise ValueError("No se puede ramificar despues de una respuesta final.")
        result = environment.execute(action["name"], deepcopy(action["arguments"]))
        expected = {"role": "tool", "content": {"name": action["name"], **deepcopy(result)}}
        if position == len(request.history) or request.history[position] != expected:
            raise ValueError("La reproduccion del prefijo cambio una observacion publica.")
        events.append({"action": action})
        position += 1
    size = len(json.dumps({"prompt": task["prompt"], "history": request.history}, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if (request.state.progress != compact_progress(environment.observation(), events)
            or request.state.context_utf8_bytes != size or request.state.previous_errors != tuple(errors)):
        raise ValueError("El progreso o los errores del prefijo no se reproducen.")
    return environment, events


def _audit_manifest(directory):
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    body = {k: v for k, v in manifest.items() if k != "freeze_sha256"}
    if object_hash(body) != manifest["freeze_sha256"] or object_hash(manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("Manifiesto o tareas fuente alterados.")
    archive_path = directory / "source_bundle.zip"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest["source_bundle_sha256"]:
        raise ValueError("Snapshot fuente alterado.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {*manifest["source_files_sha256"], "configuration.json"}:
            raise ValueError("Contenido del snapshot incompatible.")
        for name, digest in manifest["source_files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Fuentes del snapshot alteradas.")
        if json.loads(archive.read("configuration.json")) != manifest["configuration"]:
            raise ValueError("Configuracion del snapshot alterada.")
    return manifest


def _audit_episode(journal, key, task, config, *, require_complete=True):
    records = journal.for_key(key)
    if not records or records[-1]["kind"] != "episode_completed":
        raise ValueError("Episodio fuente incompleto.")
    episode = records[-1]["data"]
    starts = [r["data"] for r in records if r["kind"] == "call_started"]
    calls = [r["data"] for r in records if r["kind"] == "call_completed"]
    if require_complete and (len(starts) != len(calls) or episode["termination"] == "interrupted_not_retried"):
        raise ValueError("La fuente contiene un intento incierto o interrumpido.")
    for started, call in zip(starts, calls):
        if (any(call.get(k) != v for k, v in started.items())
                or object_hash(started["request"]) != started["request_sha256"]
                or started["request"]["prompt"] != task["prompt"]
                or started["model"] not in config["model_catalog"]):
            raise ValueError("La solicitud o decision fuente cambio despues de iniciarse.")
        request_from_dict(started["request"])
    if require_complete:
        if episode["events"] != annotate_costs(calls, config["pricing"], config["compute_rates"]):
            raise ValueError("El cierre fuente cambio sus eventos o costos.")
        final = calls[-1].get("final_success") if calls else None
        if ((final is not None and (type(final) is not bool or episode["success"] != final or episode["termination"] != "final_answer"))
                or (final is None and (episode["success"] is not False or episode["termination"] == "final_answer"))):
            raise ValueError("El cierre fuente cambio su desenlace registrado.")
    elif episode["termination"] == "interrupted_not_retried":
        class RecordedPolicy:
            name = key[1]
        if episode != _recover(task, RecordedPolicy(), records[:-1], config):
            raise ValueError("El cierre interrumpido cambio sus intentos o costos desconocidos.")
    if (episode["source_group"] != task["source_group"] or episode["family"] != task["family"]
            or episode["provenance"] != {k: config[k] for k in ("execution_mode", "evaluation_split", "independent_validation")}):
        raise ValueError("Procedencia fuente incompatible.")
    return episode, calls


def _indices(decision_indices):
    if (type(decision_indices) not in (tuple, list) or not decision_indices
            or any(type(i) is not int or i < 0 for i in decision_indices)
            or list(decision_indices) != sorted(set(decision_indices))):
        raise ValueError("Predeclarar indices de decision unicos, crecientes y no negativos.")
    return list(decision_indices)


def _pick(task, calls, indices, source):
    # Pinned transport retries are attempts, not new routing decisions.
    decisions = [c for c in calls if c.get("routing_source") != "pinned_infrastructure_retry"]
    anchors, unavailable = [], []
    for index in indices:
        if index >= len(decisions):
            unavailable.append({"task_id": task["id"], "source_group": task["source_group"],
                                "decision_index": index, "reason": "decision_not_reached", "source": source})
            continue
        call = decisions[index]
        anchors.append({"task_id": task["id"], "source_group": task["source_group"],
                        "request": deepcopy(call["request"]), "request_sha256": call["request_sha256"],
                        "decision_index": index, "source": {**source, "call_index": call["call_index"]}})
    return anchors, unavailable


def select_anchors(source_run, decision_indices=(0, 2), source_policy="always_weak"):
    """Audit a completed source run and select fixed decision positions only."""
    directory = Path(source_run).resolve()
    manifest = _audit_manifest(directory)
    config = manifest["configuration"]
    if config["schema_version"] != 2 or source_policy not in manifest["policies"]:
        raise ValueError("La fuente requiere contrato v8 y la politica solicitada.")
    indices = _indices(decision_indices)
    allowed = {(t["id"], p) for t in manifest["tasks"] for p in manifest["policies"]}
    journal = _Journal(directory, manifest["freeze_sha256"], allowed)
    summary = _unseal(directory / "summary.json")
    if (summary["freeze_sha256"] != manifest["freeze_sha256"] or summary["configuration"] != config
            or summary["journal_sha256"] != object_hash(journal.records)
            or len(summary["completed_pairs"]) != len(allowed)
            or set(map(tuple, summary["completed_pairs"])) != allowed):
        raise ValueError("La corrida fuente no esta completa o su resumen fue alterado.")
    anchors, unavailable = [], []
    for task in manifest["tasks"]:
        for policy in manifest["policies"]:
            _, calls = _audit_episode(journal, (task["id"], policy), task, config)
            if policy == source_policy:
                selected, missing = _pick(task, calls, indices, {"kind": "episode_run", "run": str(directory),
                    "freeze_sha256": manifest["freeze_sha256"], "episode_id": task["id"], "policy": policy})
                anchors.extend(selected)
                unavailable.extend(missing)
    return {"schema_version": 1, "anchors": anchors, "unavailable": unavailable, "provenance": {
        "kind": "episode_run", "run": str(directory), "freeze_sha256": manifest["freeze_sha256"],
        "journal_sha256": object_hash(journal.records), "tasks_sha256": manifest["tasks_sha256"],
        "source_policy": source_policy, "decision_indices": indices, "model_catalog": config["model_catalog"]}}


class _ProbeThenReference:
    name = "paired_rollout"

    def __init__(self, models, candidate):
        self.models, self.candidate, self.first = models, candidate, True
        self.reference = NReactiveCascade(models)

    def select(self, request):
        if self.first:
            self.first = False
            return self.candidate
        return self.reference.select(request)


def _tariff_rates(tariffs, models):
    rates = {}
    if type(tariffs) is not dict or type(tariffs.get("models")) is not dict:
        raise ValueError("Se requiere un documento de tarifas congeladas con models.")
    for model in models:
        rate = tariffs["models"].get(model, {})
        pair = tuple(rate.get(k) for k in ("input_usd_per_million", "output_usd_per_million"))
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pair):
            raise ValueError("Cada candidato requiere tarifas input/output finitas no negativas.")
        rates[model] = pair
    return rates


def remaining_provider_cost(episode, tariffs, models):
    """No-cache tariff scenario, all attempts once; missing usage stays unknown."""
    rates = _tariff_rates(tariffs, models)
    total = 0.
    for event in episode["events"]:
        if (event.get("backend_failure") or {}).get("category") == "model_identity":
            return None
        usage = event.get("usage") or {}
        counts = (usage.get("input_tokens"), usage.get("output_tokens"))
        if any(type(v) is not int or v < 0 for v in counts) or event["model"] not in rates:
            return None
        total += sum(count * rate / 1e6 for count, rate in zip(counts, rates[event["model"]]))
    return total


def _validate_anchor_bundle(tasks, anchors, config, environment_factory):
    kind = anchors.get("provenance", {}).get("kind")
    if kind == "task_initial_state":
        expected = initial_anchors(tasks, config, environment_factory=environment_factory)
    elif kind == "episode_run":
        p = anchors["provenance"]
        expected = select_anchors(p["run"], p["decision_indices"], p["source_policy"])
    elif kind == "paired_collection":
        p = anchors["provenance"]
        expected = select_followup_anchors(p["run"], p["decision_indices"], p["probe_model"])
    else:
        raise ValueError("Procedencia de anclas desconocida.")
    if (anchors != expected or object_hash(tasks) != anchors["provenance"]["tasks_sha256"]
            or config["model_catalog"] != anchors["provenance"]["model_catalog"]):
        raise ValueError("Anclas incompletas, modificadas o de otra poblacion/catalogo.")
    task_map = {t["id"]: t for t in tasks}
    for anchor in anchors["anchors"]:
        if object_hash(anchor["request"]) != anchor["request_sha256"]:
            raise ValueError("Solicitud del ancla alterada.")
        task = task_map[anchor["task_id"]]
        request = request_from_dict(anchor["request"])
        if not 1 <= request.state.remaining_calls <= min(config["max_calls"], task["max_steps"]):
            raise ValueError("El presupuesto del ancla no cabe en la configuracion de ramas.")
        replay_prefix(task, request, environment_factory)


def _audit_branch(journal, task, config, branch, anchor):
    episode, _ = _audit_episode(journal, (task["id"], _ProbeThenReference.name), task, config,
        require_complete=journal.records[-1]["data"].get("termination") != "interrupted_not_retried")
    starts = [r["data"] for r in journal.records if r["kind"] == "call_started"]
    if (not starts or len(starts) > anchor["request"]["state"]["remaining_calls"]
            or starts[0]["model"] != branch["candidate"] or starts[0]["request"] != anchor["request"]
            or starts[0]["request_sha256"] != anchor["request_sha256"]):
        raise ValueError("La rama no conserva la intervencion inicial congelada.")
    return episode


def collect_paired_rollouts(tasks, anchors, models, generate, config, output_dir, *,
                            max_generation_calls, tariffs, execute=False, environment_factory=create_environment):
    """Freeze or execute all N interventions per declared anchor.

    max_generation_calls is an explicit global hard bound, including retries.
    It must cover the worst-case remaining budgets before any call is sent.
    Interrupted branches are censored and never reissued on resume. Transport
    terminal failures remain in operational accounting, not capability labels.
    """
    models = model_order(tuple(models))
    tasks, anchors, config, tariffs = map(deepcopy, (tasks, anchors, config, tariffs))
    _validate(config, [AlwaysStrong(models[-1], models[0])])
    if config["schema_version"] != 2 or list(models) != config["model_catalog"] or not callable(generate):
        raise ValueError("Catalogo v8 y callback de generacion explicito requeridos.")
    _tariff_rates(tariffs, models)
    _validate_anchor_bundle(tasks, anchors, config, environment_factory)
    bound = sum(a["request"]["state"]["remaining_calls"] * len(models) for a in anchors["anchors"])
    if type(max_generation_calls) is not int or max_generation_calls < bound or max_generation_calls < 1:
        raise ValueError(f"El presupuesto global debe cubrir la cota predeclarada de {bound} llamadas.")
    branch_plan = []
    for index, anchor in enumerate(anchors["anchors"]):
        offset = index % len(models)
        for model in models[offset:] + models[:offset]:
            branch_plan.append({"branch_id": object_hash({"anchor": anchor, "model": model})[:24],
                                "anchor_index": index, "candidate": model})
    bundle = _source_bundle(config)
    body = {"schema_version": 1, "kind": "paired_collection", "configuration": config,
            "tasks": tasks, "tasks_sha256": object_hash(tasks), "anchors": anchors, "branches": branch_plan,
            "model_catalog": list(models), "target": TARGET, "continuation_policy": CONTINUATION,
            "cost_basis": COST_BASIS, "tariffs": tariffs, "max_generation_calls": max_generation_calls,
            "maximum_generation_calls": bound, "generation_source": _fingerprint(generate),
            "environment_source": _fingerprint(environment_factory), "source_files_sha256": _sources(),
            "source_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "cost_scope": "new branch attempts only; source/prefix acquisition excluded; not invoice or total operational cost"}
    frozen = {**body, "freeze_sha256": object_hash(body)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with _single_writer(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if _read(manifest_path) != frozen:
                raise ValueError("La coleccion congelada cambio: usar una salida nueva.")
        else:
            if any(p.name != ".writer.lock" for p in output.iterdir()):
                raise FileExistsError("La salida de coleccion debe ser nueva.")
            _write(manifest_path, frozen)
        archive = output / "source_bundle.zip"
        if archive.exists() and archive.read_bytes() != bundle:
            raise ValueError("El snapshot de la coleccion cambio.")
        if not archive.exists():
            archive.write_bytes(bundle)
        if not execute:
            return {"mode": "prepared", "freeze_sha256": frozen["freeze_sha256"], "generation_calls": 0,
                    "maximum_generation_calls": bound, "branches": len(branch_plan), "anchors": len(anchors["anchors"])}
        task_map, results = {t["id"]: t for t in tasks}, []
        starts_so_far = 0
        for branch in branch_plan:
            anchor = anchors["anchors"][branch["anchor_index"]]
            task = task_map[anchor["task_id"]]
            folder = output / "branches" / branch["branch_id"]
            folder.mkdir(parents=True, exist_ok=True)
            branch_body = {"collection_freeze_sha256": frozen["freeze_sha256"], **branch}
            branch_freeze = object_hash(branch_body)
            branch_manifest = {**branch_body, "freeze_sha256": branch_freeze}
            if (folder / "manifest.json").exists():
                if _read(folder / "manifest.json") != branch_manifest:
                    raise ValueError("La rama pertenece a otra coleccion.")
            else:
                _write(folder / "manifest.json", branch_manifest)
            policy = _ProbeThenReference(models, branch["candidate"])
            key = (task["id"], policy.name)
            journal = _Journal(folder, branch_freeze, {key})
            existing = journal.for_key(key)
            episode = (_recover(task, policy, existing, config) if existing else
                       _execute(task, policy, generate, config, journal, environment_factory,
                                initial_request=request_from_dict(anchor["request"])))
            if not existing or existing[-1]["kind"] != "episode_completed":
                journal.append("episode_completed", key, episode)
            _audit_branch(journal, task, config, branch, anchor)
            starts_so_far += sum(r["kind"] == "call_started" for r in journal.records)
            if starts_so_far > max_generation_calls:
                raise RuntimeError("Se excedio el limite global congelado.")
            reason = episode["termination"] if episode["termination"] in {"interrupted_not_retried", "infrastructure_failure"} else None
            if (episode["termination"] == "call_budget_exhausted"
                    and any(e.get("error_scope") == "infrastructure" for e in episode["events"])):
                # Even a recovered transport retry consumes a call. A later
                # budget failure cannot isolate the model's semantic ability.
                reason = "infrastructure_budget_exhausted"
            row = {"request": deepcopy(anchor["request"]), "model": branch["candidate"], "success": episode["success"],
                   "remaining_provider_cost_usd": remaining_provider_cost(episode, tariffs, models),
                   "source_group": anchor["source_group"], "target": TARGET, "continuation_policy": CONTINUATION,
                   "assignment": "synthetic_fixture" if config["execution_mode"] == "simulated" else "paired_state",
                   "cost_basis": COST_BASIS}
            results.append({**branch, "journal_sha256": object_hash(journal.records), "row": row,
                            "operational_success": episode["success"], "termination": episode["termination"],
                            "censored_reason": reason, "generation_calls": len(episode["events"])})
        if (_sources() != frozen["source_files_sha256"] or _read(manifest_path) != frozen
                or _fingerprint(generate) != frozen["generation_source"]
                or _fingerprint(environment_factory) != frozen["environment_source"]):
            raise ValueError("Las fuentes cambiaron durante la coleccion.")
        result = {"schema_version": 1, "collection_freeze_sha256": frozen["freeze_sha256"],
                  "target": TARGET, "continuation_policy": CONTINUATION, "cost_basis": COST_BASIS,
                  "model_catalog": list(models), "rows": [r["row"] for r in results if r["censored_reason"] is None],
                  "branches": results, "censored": [r["branch_id"] for r in results if r["censored_reason"]],
                  "training_ready": bool(results) and all(r["censored_reason"] is None for r in results),
                  "training_readiness_scope": "complete paired endpoint labels; class and known-cost support checked by trainer",
                  "all_costs_observed": bool(results) and all(r["row"]["remaining_provider_cost_usd"] is not None for r in results),
                  "unknown_cost_branches": [r["branch_id"] for r in results
                                             if r["row"]["remaining_provider_cost_usd"] is None],
                  "generation_calls": starts_so_far, "unavailable": anchors["unavailable"],
                  "execution_mode": config["execution_mode"], "independent_validation": False}
        _write(output / "dataset.json", _seal(result))
        return result


def select_followup_anchors(collection_dir, decision_indices=(2,), probe_model=None):
    """Use fixed positions on every declared probe trajectory, not successes."""
    directory = Path(collection_dir).resolve()
    manifest = _audit_manifest(directory)
    if manifest.get("kind") != "paired_collection":
        raise ValueError("Se requiere una coleccion pareada.")
    indices = _indices(decision_indices)
    probe_model = probe_model or manifest["model_catalog"][0]
    if probe_model not in manifest["model_catalog"]:
        raise ValueError("El probe debe pertenecer al catalogo congelado.")
    data = _unseal(directory / "dataset.json")
    if data["collection_freeze_sha256"] != manifest["freeze_sha256"]:
        raise ValueError("Dataset de otra coleccion.")
    by_branch = {b["branch_id"]: b for b in data["branches"]}
    if len(by_branch) != len(manifest["branches"]) or set(by_branch) != {b["branch_id"] for b in manifest["branches"]}:
        raise ValueError("Coleccion fuente incompleta.")
    tasks = {t["id"]: t for t in manifest["tasks"]}
    anchors, unavailable = [], []
    for branch in manifest["branches"]:
        original = manifest["anchors"]["anchors"][branch["anchor_index"]]
        task = tasks[original["task_id"]]
        folder = directory / "branches" / branch["branch_id"]
        branch_body = {"collection_freeze_sha256": manifest["freeze_sha256"], **branch}
        freeze = object_hash(branch_body)
        if _read(folder / "manifest.json") != {**branch_body, "freeze_sha256": freeze}:
            raise ValueError("Manifiesto de rama alterado.")
        journal = _Journal(folder, freeze, {(task["id"], _ProbeThenReference.name)})
        if object_hash(journal.records) != by_branch[branch["branch_id"]]["journal_sha256"]:
            raise ValueError("Journal de rama incompatible con el dataset.")
        _audit_branch(journal, task, manifest["configuration"], branch, original)
        _, calls = _audit_episode(journal, (task["id"], _ProbeThenReference.name), task, manifest["configuration"])
        if branch["candidate"] == probe_model:
            source = {"kind": "paired_collection", "run": str(directory), "freeze_sha256": manifest["freeze_sha256"],
                      "branch_id": branch["branch_id"], "episode_id": task["id"], "probe_model": probe_model}
            selected, missing = _pick(task, calls, indices, source)
            anchors.extend(selected)
            unavailable.extend(missing)
    return {"schema_version": 1, "anchors": anchors, "unavailable": unavailable, "provenance": {
        "kind": "paired_collection", "run": str(directory), "freeze_sha256": manifest["freeze_sha256"],
        "dataset_sha256": object_hash(data), "tasks_sha256": manifest["tasks_sha256"], "probe_model": probe_model,
        "decision_indices": indices, "model_catalog": manifest["model_catalog"]}}
