"""Read-only integrity audit of paired trajectory acquisition, complete or partial.

This report does not generate calls, replay tools, recover pending attempts,
train a model, or evaluate a routing policy. A strong first probe followed by
the common reference is not the always-strong baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from routerbench.agent_evaluation import _Journal, object_hash
from routerbench.agent_rollouts import (
    _audit_manifest, _audit_branch, _tariff_rates, remaining_provider_cost,
    request_from_dict, select_followup_anchors, TARGET, CONTINUATION, COST_BASIS,
)
from routerbench.agent_tasks import validate_task_splits
from routerbench.agent_metrics import annotate_costs

PARTITIONS = ("fit", "calibration", "development")
STAGES = ("initial", "followup")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Clave JSON duplicada: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Numero JSON no finito: {value}")


class Snapshot:
    """Remember read bytes and reject a changing acquisition snapshot."""
    def __init__(self, root):
        self.root, self.hashes = Path(root).resolve(), {}

    def track(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        name = path.relative_to(self.root).as_posix()
        if name in self.hashes and self.hashes[name] != digest:
            raise ValueError("La adquisicion cambio durante la auditoria; repetir la lectura.")
        self.hashes[name] = digest
        return raw

    def read(self, path, *, sealed=False):
        result = json.loads(self.track(path).decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
        if sealed:
            if (type(result) is not dict or set(result) != {"payload", "payload_sha256"}
                    or object_hash(result["payload"]) != result["payload_sha256"]):
                raise ValueError(f"Sobre alterado: {path}")
            return result["payload"]
        return result

    def verify(self):
        for name, expected in self.hashes.items():
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != expected:
                raise ValueError("La adquisicion cambio durante la auditoria; repetir la lectura.")


def _reason(episode):
    if episode["termination"] in {"interrupted_not_retried", "infrastructure_failure"}:
        return episode["termination"]
    if episode["termination"] == "call_budget_exhausted" and any(
            e.get("error_scope") == "infrastructure" for e in episode["events"]):
        return "infrastructure_budget_exhausted"
    return None


def _recorded_row(manifest, branch, anchor, episode):
    return {"request": anchor["request"], "model": branch["candidate"], "success": episode["success"],
            "remaining_provider_cost_usd": remaining_provider_cost(episode, manifest["tariffs"], manifest["model_catalog"]),
            "source_group": anchor["source_group"], "target": TARGET, "continuation_policy": CONTINUATION,
            "assignment": "synthetic_fixture" if manifest["configuration"]["execution_mode"] == "simulated" else "paired_state",
            "cost_basis": COST_BASIS}


def _account(events, tariffs, models):
    """Count each started attempt once, including pending and censored ones."""
    _tariff_rates(tariffs, models)
    aggregate = {"calls": len(events), "known_input_tokens": 0, "known_output_tokens": 0,
                 "input_token_unknown_calls": 0, "output_token_unknown_calls": 0,
                 "known_tariff_subtotal_usd": 0., "tariff_unknown_calls": 0,
                 "reported_provider_subtotal_usd": 0., "reported_provider_unknown_calls": 0}
    for event in events:
        for kind in ("input", "output"):
            count = (event.get("usage") or {}).get(kind + "_tokens")
            if type(count) is int and count >= 0:
                aggregate["known_" + kind + "_tokens"] += count
            else:
                aggregate[kind + "_token_unknown_calls"] += 1
        cost = remaining_provider_cost({"events": [event]}, tariffs, models)
        if cost is None:
            aggregate["tariff_unknown_calls"] += 1
        else:
            aggregate["known_tariff_subtotal_usd"] += cost
        reported = event.get("reported_llm_cost_usd")
        if reported is None:
            aggregate["reported_provider_unknown_calls"] += 1
        else:
            aggregate["reported_provider_subtotal_usd"] += reported
    return _complete_account(aggregate)


def _complete_account(account):
    for kind in ("input", "output"):
        account[kind + "_tokens"] = (None if account[kind + "_token_unknown_calls"]
                                      else account["known_" + kind + "_tokens"])
    account["tariff_scenario_usd"] = None if account["tariff_unknown_calls"] else account["known_tariff_subtotal_usd"]
    account["reported_provider_usd"] = (None if account["reported_provider_unknown_calls"]
                                       else account["reported_provider_subtotal_usd"])
    return account


def _sum_accounts(accounts):
    keys = ("calls", "known_input_tokens", "known_output_tokens", "input_token_unknown_calls",
            "output_token_unknown_calls", "known_tariff_subtotal_usd", "tariff_unknown_calls",
            "reported_provider_subtotal_usd", "reported_provider_unknown_calls")
    return _complete_account({key: sum(a[key] for a in accounts) for key in keys})


def _collection(directory, part, stage, tasks, models, snapshot):
    snapshot.read(directory / "manifest.json")
    snapshot.track(directory / "source_bundle.zip")
    manifest = _audit_manifest(directory)
    if (manifest.get("kind") != "paired_collection" or manifest["tasks"] != tasks
            or manifest["model_catalog"] != models or manifest["configuration"].get("data_partition") != part
            or manifest["target"] != TARGET or manifest["continuation_policy"] != CONTINUATION
            or manifest["cost_basis"] != COST_BASIS or manifest["configuration"]["independent_validation"] is not False):
        raise ValueError("Coleccion de otra particion, poblacion, catalogo u objetivo.")
    anchors = manifest["anchors"]
    if anchors["provenance"]["tasks_sha256"] != object_hash(tasks) or anchors["provenance"]["model_catalog"] != models:
        raise ValueError("Procedencia de anclas incompatible.")
    if stage == "followup":
        expected = select_followup_anchors(directory.parent / "initial", (2,), models[0])
        if anchors != expected:
            raise ValueError("Las anclas de seguimiento no corresponden a posiciones predeclaradas.")
    elif (anchors["provenance"]["kind"] != "task_initial_state" or anchors["provenance"]["decision_indices"] != [0]
          or len(anchors["anchors"]) != len(tasks) or anchors["unavailable"]):
        raise ValueError("Anclas iniciales incompletas o seleccionadas por resultados.")
    task_map = {t["id"]: t for t in tasks}
    expected_branches = []
    for index, anchor in enumerate(anchors["anchors"]):
        if (anchor["task_id"] not in task_map or anchor["source_group"] != task_map[anchor["task_id"]]["source_group"]
                or anchor["request_sha256"] != object_hash(anchor["request"])):
            raise ValueError("Identidad, grupo o solicitud de ancla incompatible.")
        request_from_dict(anchor["request"])
        if stage == "initial" and (anchor["source"] != {"kind": "task_initial_state", "task_sha256": object_hash(tasks[index])}
                                  or anchor["task_id"] != tasks[index]["id"]):
            raise ValueError("La poblacion inicial cambio de orden o identidad.")
        offset = index % len(models)
        for model in models[offset:] + models[:offset]:
            expected_branches.append({"branch_id": object_hash({"anchor": anchor, "model": model})[:24],
                                      "anchor_index": index, "candidate": model})
    if manifest["branches"] != expected_branches:
        raise ValueError("Falta o cambio una intervencion pareada predeclarada.")
    bound = sum(a["request"]["state"]["remaining_calls"] * len(models) for a in anchors["anchors"])
    if bound != manifest["maximum_generation_calls"] or bound > manifest["max_generation_calls"]:
        raise ValueError("Presupuesto global inconsistente.")
    branch_results, public_branches, attempts = [], [], []
    for branch in manifest["branches"]:
        anchor = anchors["anchors"][branch["anchor_index"]]
        task = task_map[anchor["task_id"]]
        folder = directory / "branches" / branch["branch_id"]
        branch_path = folder / "manifest.json"
        view = {**branch, "source_group": task["source_group"], "task_id": task["id"],
                "status": "not_started", "calls_started": 0, "calls_completed": 0, "pending_calls": 0}
        if not branch_path.exists():
            if (folder / "journal.json").exists():
                raise ValueError("Journal sin manifiesto de rama.")
            public_branches.append(view)
            continue
        branch_body = {"collection_freeze_sha256": manifest["freeze_sha256"], **branch}
        freeze = object_hash(branch_body)
        if snapshot.read(branch_path) != {**branch_body, "freeze_sha256": freeze}:
            raise ValueError("La rama no pertenece a la coleccion.")
        if not (folder / "journal.json").exists():
            public_branches.append(view)
            continue
        snapshot.read(folder / "journal.json", sealed=True)
        key = (task["id"], "paired_rollout")
        journal = _Journal(folder, freeze, {key})
        starts = [r["data"] for r in journal.records if r["kind"] == "call_started"]
        calls = [r["data"] for r in journal.records if r["kind"] == "call_completed"]
        if (len(starts) > anchor["request"]["state"]["remaining_calls"]
                or (starts and (starts[0]["model"] != branch["candidate"] or starts[0]["request"] != anchor["request"]))):
            raise ValueError("La rama cambio el probe, estado inicial o presupuesto.")
        for started in starts:
            if (started["request_sha256"] != object_hash(started["request"])
                    or started["model"] not in models or started["request"]["prompt"] != task["prompt"]):
                raise ValueError("Solicitud iniciada alterada.")
            request_from_dict(started["request"])
        for started, call in zip(starts, calls):
            if any(call.get(name) != value for name, value in started.items()):
                raise ValueError("La llamada completada cambio su decision previa.")
        annotate_costs(calls, manifest["configuration"]["pricing"], manifest["configuration"]["compute_rates"])
        raw_events = calls + [{**started, "usage": {}, "reported_llm_cost_usd": None} for started in starts[len(calls):]]
        attempts.extend(raw_events)
        view.update(calls_started=len(starts), calls_completed=len(calls), pending_calls=len(starts)-len(calls),
                    status="pending" if len(starts) > len(calls) else "in_progress")
        if journal.records and journal.records[-1]["kind"] == "episode_completed":
            episode = _audit_branch(journal, task, manifest["configuration"], branch, anchor)
            reason = _reason(episode)
            result = {**branch, "journal_sha256": object_hash(journal.records),
                      "row": _recorded_row(manifest, branch, anchor, episode),
                      "operational_success": episode["success"], "termination": episode["termination"],
                      "censored_reason": reason, "generation_calls": len(episode["events"])}
            branch_results.append(result)
            view.update(status="censored" if reason else "completed", operational_success=episode["success"],
                        termination=episode["termination"], censored_reason=reason)
        public_branches.append(view)
    if len(attempts) > manifest["max_generation_calls"]:
        raise ValueError("Se excedio el presupuesto de la coleccion.")
    complete = len(branch_results) == len(manifest["branches"])
    dataset_path = directory / "dataset.json"
    if dataset_path.exists():
        dataset = snapshot.read(dataset_path, sealed=True)
        expected_fields = {"collection_freeze_sha256": manifest["freeze_sha256"], "target": TARGET,
            "continuation_policy": CONTINUATION, "cost_basis": COST_BASIS, "model_catalog": models,
            "rows": [b["row"] for b in branch_results if b["censored_reason"] is None], "branches": branch_results,
            "censored": [b["branch_id"] for b in branch_results if b["censored_reason"]],
            "generation_calls": len(attempts), "unavailable": anchors["unavailable"]}
        if not complete or any(dataset.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("El dataset no reproduce los desenlaces/costos de todos los journals.")
    else:
        complete = False  # All branch calls can be done before the dataset commit.
    grouped, training_rows, excluded = {}, [], 0
    for branch in branch_results:
        grouped.setdefault(branch["anchor_index"], []).append(branch)
    for branches in grouped.values():
        if len(branches) == len(models) and not any(b["censored_reason"] for b in branches):
            training_rows.extend(b["row"] for b in branches)
        elif len(branches) == len(models):
            excluded += 1
    return {"partition": part, "stage": stage, "directory": str(directory.resolve()),
            "freeze_sha256": manifest["freeze_sha256"], "complete": complete, "branches": public_branches,
            "training_rows": training_rows, "generation_calls": len(attempts), "maximum_generation_calls": bound,
            "censored_branches": sum(bool(b["censored_reason"]) for b in branch_results),
            "excluded_censored_pairs": excluded, "unavailable": anchors["unavailable"],
            "accounting": _account(attempts, manifest["tariffs"], models),
            "actual_model_accounting": {m: _account([e for e in attempts if e["model"] == m], manifest["tariffs"], models) for m in models},
            "tariffs": manifest["tariffs"], "execution_mode": manifest["configuration"]["execution_mode"],
            "free_account_declaration": manifest["configuration"].get("spending", {}).get("account_is_free_tier") is True}


def audit_acquisition(directory, *, allow_partial=False):
    """Audit persisted evidence only. Partial means unknown, never failure."""
    directory = Path(directory).resolve()
    snapshot = Snapshot(directory)
    plan = snapshot.read(directory / "plan.json", sealed=True)
    splits = snapshot.read(directory / "task_splits.json")
    if (object_hash(splits) != plan["task_splits_sha256"] or set(splits) != {*PARTITIONS, "evaluation"}
            or plan["decision_indices"] != [0, 2] or plan["reserved_partition_executed"] is not False
            or plan["independent_validation"] is not False or plan["partitions"] != {k: len(v) for k, v in splits.items()}):
        raise ValueError("El plan, las particiones o sus posiciones predeclaradas cambiaron.")
    partition_audit = validate_task_splits(splits)
    if (directory / "evaluation").exists():
        raise ValueError("Esta auditoria de adquisicion no admite ejecuciones en evaluation.")
    models, collections, stages, observed = plan["models"], [], [], {part: [] for part in PARTITIONS}
    audit_records, all_complete = [], True
    for part in PARTITIONS:
        for stage in STAGES:
            path = directory / part / stage
            if (path / "manifest.json").exists():
                collection = _collection(path, part, stage, splits[part], models, snapshot)
                collections.append(collection)
                observed[part].extend(collection["training_rows"])
                all_complete = all_complete and collection["complete"]
                audit_records.append({key: collection[key] for key in ("partition", "stage", "directory", "freeze_sha256",
                    "generation_calls", "censored_branches", "excluded_censored_pairs", "unavailable")})
                stages.append({"partition": part, "stage": stage, "status": "complete" if collection["complete"] else "partial"})
            elif stage == "followup" and stages[-1]["status"] == "complete":
                if any(b.get("termination") == "interrupted_not_retried" for b in collections[-1]["branches"]):
                    all_complete = False
                    stages.append({"partition": part, "stage": stage, "status": "blocked_source_interruption"})
                    continue
                anchors = select_followup_anchors(directory / part / "initial", (2,), models[0])
                if not anchors["anchors"]:
                    stages.append({"partition": part, "stage": stage, "status": "unavailable"})
                    audit_records.append({"partition": part, "stage": stage, "unavailable": anchors["unavailable"]})
                    continue
                all_complete = False
                stages.append({"partition": part, "stage": stage, "status": "not_started"})
            else:
                all_complete = False
                stages.append({"partition": part, "stage": stage, "status": "not_started"})
    class_support = {part: {model: dict(Counter(str(row["success"]).lower() for row in observed[part]
        if row["model"] == model)) for model in models} for part in PARTITIONS}
    expected_observations = {"schema_version": 2, "model_catalog": models, **observed}
    excluded_pairs = sum(c["excluded_censored_pairs"] for c in collections)
    class_ready = all(set(class_support[part][model]) == {"true", "false"}
                      for part in ("fit", "calibration") for model in models)
    has_observations = (directory / "observations.json").exists()
    has_audit = (directory / "acquisition_audit.json").exists()
    root_complete = has_observations and has_audit
    if has_observations:
        observations = snapshot.read(directory / "observations.json")
        if not all_complete or observations != expected_observations:
            raise ValueError("Observations/acquisition_audit no se reproducen desde las ramas registradas.")
    if has_audit:
        audit = snapshot.read(directory / "acquisition_audit.json")
        expected_audit = {"plan_sha256": object_hash(plan), "collections": audit_records,
                         "excluded_censored_pairs": excluded_pairs, "reserved_partition_executed": False,
                         "independent_validation": False, "class_support": class_support,
                         "observations_sha256": object_hash(expected_observations), "training_class_support_ready": class_ready}
        if not all_complete or audit != expected_audit:
            raise ValueError("Observations/acquisition_audit no se reproducen desde las ramas registradas.")
    complete = all_complete and root_complete
    if not complete and not allow_partial:
        raise ValueError("La adquisicion sigue incompleta; usar --allow-partial para una instantanea sin etiquetas inventadas.")
    snapshot.verify()
    for collection in collections:
        del collection["training_rows"]
    branches = [b for c in collections for b in c["branches"]]
    result = {"schema_version": 1, "directory": str(directory), "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial", "plan_sha256": object_hash(plan), "model_catalog": models,
        "target": TARGET, "continuation_policy": CONTINUATION, "cost_basis": COST_BASIS,
        "stages": stages, "collections": collections, "partition_audit": partition_audit,
        "accounting": _sum_accounts([c["accounting"] for c in collections]),
        "actual_model_accounting": {m: _sum_accounts([c["actual_model_accounting"][m] for c in collections]) for m in models},
        "class_support": class_support, "training_class_support_ready": complete and class_ready,
        "training_class_support_scope": "both endpoint classes in fit/calibration per probe; not complete trainer eligibility",
        "training_rows": {p: len(observed[p]) for p in PARTITIONS}, "excluded_censored_pairs": excluded_pairs,
        "branches_started": sum(b["calls_started"] > 0 for b in branches),
        "branches_closed": sum(b["status"] in {"completed", "censored"} for b in branches),
        "pending_calls": sum(b["pending_calls"] for b in branches),
        "censored_branches": sum(b["status"] == "censored" for b in branches),
        "reserved_partition_executed": False, "independent_validation": False,
        "source_file_sha256": snapshot.hashes, "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "paired data acquisition under a fixed continuation; not policy evaluation, actual invoice, ROI or competitor superiority",
        "partial_scope": "persisted attempts only; pending/unstarted branches have no outcome; no calls are recovered or repeated",
        "probe_scope": "candidate identifies only the first decision; subsequent decisions use the same reactive reference",
        "token_scope": "measured counters for completed calls; synthetic fixture counters remain fictitious",
        "cost_scope": "no-cache frozen tariff scenario counts every new attempt once; tools, router, verification and invoice absent"}
    return result


def _number(value):
    return "desconocido" if value is None else f"{value:.8f}"


def render_report(report):
    account = report["accounting"]
    lines = ["Informe de adquisicion de trayectorias pareadas de Relay.", "",
        f"Estado auditado: {report['status']}. {report['branches_closed']} ramas cerradas, "
        f"{account['calls']} intentos registrados, {report['pending_calls']} intentos sin resultado persistido "
        f"y {report['censored_branches']} ramas censuradas. Los intentos pendientes no se convierten en fallos.", "",
        "El candidato identifica solo la primera llamada de cada rama. Despues sigue una cascada reactiva comun; "
        "el probe fuerte no es la politica Siempre Fuerte. Estas observaciones sirven para adquirir etiquetas, "
        "no para medir el router entrenado ni compararlo con otros productos.", "",
        f"Tokens de entrada: {account['input_tokens'] if account['input_tokens'] is not None else 'desconocido'}; "
        f"salida: {account['output_tokens'] if account['output_tokens'] is not None else 'desconocido'}. "
        f"Subtotales con medicion: {account['known_input_tokens']} entrada y {account['known_output_tokens']} salida. "
        f"Faltan mediciones de entrada en {account['input_token_unknown_calls']} llamadas y salida en "
        f"{account['output_token_unknown_calls']}. En simulaciones estos contadores son ficticios.", "",
        f"Escenario con tarifa congelada sin cache: US${_number(account['tariff_scenario_usd'])}; "
        f"subtotal conocido: US${_number(account['known_tariff_subtotal_usd'])}. "
        f"Importe informado por callbacks: US${_number(account['reported_provider_usd'])}. "
        "El escenario no es una factura; una declaracion Free no sustituye el importe observado. "
        "Se cuenta cada intento una sola vez y faltan costos de herramientas, router y verificacion. No se calcula ROI.", ""]
    for part, per_model in report["class_support"].items():
        for model, counts in per_model.items():
            lines.append(f"{part}, probe {model}: {counts.get('true', 0)} desenlaces positivos y "
                         f"{counts.get('false', 0)} negativos en pares completos no censurados.")
    lines.extend(["", f"Pares completos excluidos por censura: {report['excluded_censored_pairs']}. "
        f"Soporte minimo de ambas clases en fit/calibration: {report['training_class_support_ready']}. "
        "Esto no certifica calibracion, generalizacion ni elegibilidad completa para entrenar.", "",
        "La auditoria comprueba sobres, manifiestos, snapshots, cadenas de journals, decisiones previas a cada llamada, "
        "desenlaces y costos de ramas y, cuando existen, observations/acquisition_audit. No ejecuta herramientas ni "
        "reevalua respuestas. La ausencia de evaluation se comprueba dentro de este directorio; no certifica actividad externa. "
        "Los hashes comprueban consistencia local, no autentican la autoria."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, help="New report directory; omit for JSON on stdout only.")
    args = parser.parse_args(argv)
    report = audit_acquisition(args.run, allow_partial=args.allow_partial)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.json").write_text(json.dumps({"payload": report, "payload_sha256": object_hash(report)},
            ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "branches_closed": report["branches_closed"],
        "pending_calls": report["pending_calls"], "accounting": report["accounting"],
        "class_support": report["class_support"], "report_directory": str(args.output) if args.output else None}, indent=2))
    return report


if __name__ == "__main__":
    main()
