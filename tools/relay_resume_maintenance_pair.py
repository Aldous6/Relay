"""Explicit forensic continuation of a known terminal failure, never a retry.

The parent run remains byte-for-byte intact. A derived run preserves original
manifests and declares the collector-only adjudication amendment. Only the
missing first-edit branch and one unstarted task may make new provider calls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.maintenance import _save, run_maintenance
from relay_router.maintenance_acceptance import grade_pair, validate_acceptance
from relay_router.maintenance_collection import (
    _BranchRouter, _SharedInspectionProvider, _branch_summary, _hash,
    _known_output_failure, _semantic_outcome, collect_pair)
from relay_router.maintenance_competition import digest, load_run, read
from relay_router.maintenance_provider import GroqMaintenanceProvider
from relay_router.maintenance_workspace import Workspace, _linked


_OLD_ADJUDICATION = '            and all(call["state"] == "completed" for call in report["calls"])):'
_NEW_ADJUDICATION = ('            and report["calls"] and report["calls"][-1]["state"] == "completed"\n'
    '            and all(call["state"] == "completed" or _known_output_failure(call)\n'
    '                    for call in report["calls"])):')


def _tree_hashes(directory):
    paths = sorted(directory.rglob("*"))
    if any(_linked(path) for path in [directory, *paths]):
        raise ValueError("La evidencia no puede contener enlaces ni junctions.")
    return {str(path.relative_to(directory)): digest(path) for path in paths if path.is_file()}


def _sources():
    return {path.name: digest(path) for path in sorted((ROOT / "src/relay_router").glob("maintenance*.py"))}


def _baseline_identity(report):
    baseline = report["verifications"][0]
    return {key: baseline.get(key) for key in
            ("status", "passed", "tests_run", "returncode", "timed_out", "result")}


def preflight(source, destination, *, prior_calls, max_new_calls=6, total_cap=40):
    """Validate all available evidence and inputs before credentials or calls."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (not source.is_dir() or destination.exists() or destination == source
            or destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ValueError("Usa una carpeta de evidencia existente y una salida nueva, separada.")
    if (type(prior_calls) is not int or prior_calls < 0 or type(max_new_calls) is not int
            or not 1 <= max_new_calls <= 6 or type(total_cap) is not int or not 1 <= total_cap <= 40):
        raise ValueError("La continuación admite como máximo seis llamadas nuevas y cuarenta totales.")
    original_files = _tree_hashes(source)
    manifest, journal = read(source / "manifest.json"), read(source / "journal.json")
    if journal["manifest_sha256"] != digest(source / "manifest.json"):
        raise ValueError("El manifiesto original cambió.")
    plan = manifest.get("analysis_plan") or {}
    if (manifest.get("partition") != "development" or plan.get("inspection_mode") != "direct"
            or plan.get("output_mode") != "prompt_json"):
        raise ValueError("Esta recuperación conserva exclusivamente desarrollo con edición directa prompt_json.")
    frozen_sources, actual_sources = manifest["suite"]["sources_sha256"], _sources()
    if set(actual_sources) != set(frozen_sources):
        raise ValueError("Cambió el inventario del código congelado.")
    changes = {name: {"original_sha256": frozen_sources[name], "continuation_sha256": actual_sources[name]}
               for name in frozen_sources if actual_sources[name] != frozen_sources[name]}
    if set(changes) - {"maintenance_collection.py"}:
        raise ValueError("Cambió código del flujo, proveedor, prompts u otro contrato fuera del colector.")
    collector_bytes = (ROOT / "src/relay_router/maintenance_collection.py").read_bytes()
    original_collector_bytes = collector_bytes
    if changes:
        text = collector_bytes.decode("utf-8")
        if text.count(_NEW_ADJUDICATION) != 1:
            raise ValueError("La enmienda del colector no es la corrección mínima declarada.")
        original_collector_bytes = text.replace(_NEW_ADJUDICATION, _OLD_ADJUDICATION).encode("utf-8")
        if hashlib.sha256(original_collector_bytes).hexdigest() != frozen_sources["maintenance_collection.py"]:
            raise ValueError("El colector incluye cambios adicionales a la clasificación declarada.")
    # This also checks all completed collections, reports and private acceptance.
    load_run(source, need_reference=False)
    tasks = [task for task in manifest["suite"]["tasks"] if task["partition"] == manifest["partition"]]
    known_ids = {task["task_id"] for task in tasks}
    if set(journal["tasks"]) - known_ids:
        raise ValueError("El journal contiene tareas ajenas a la suite.")
    incomplete = [task for task in tasks if journal["tasks"].get(task["task_id"], {}).get("state") == "incomplete"]
    unstarted = [task for task in tasks if task["task_id"] not in journal["tasks"]]
    if (len(incomplete) != 1 or len(unstarted) != 1 or
            any(entry["state"] not in {"completed", "incomplete"} for entry in journal["tasks"].values())):
        raise ValueError("Se requiere exactamente un par incompleto conocido y una tarea sin comenzar.")
    observed_calls = 0
    for task in tasks:
        config, root = task["config"], Path(task["repository"])
        if config["limits"]["max_repairs"] != 1:
            raise ValueError("Cambió la continuación de una reparación.")
        current = Workspace(root, destination / "validation-unused", config["source_files"],
                            config["test_files"]).validate_inputs()
        if current != task["inventory"]:
            raise ValueError("Los archivos públicos difieren de la suite congelada.")
        if task.get("acceptance_files") and validate_acceptance(root, config["source_files"],
                config["test_files"], task["acceptance_files"]) != task["acceptance_inventory"]:
            raise ValueError("Las pruebas reservadas difieren de la suite congelada.")
        entry = journal["tasks"].get(task["task_id"])
        if not entry:
            continue
        directory = source / task["task_id"]
        if digest(directory / "collection.json") != entry["collection_sha256"]:
            raise ValueError("Cambió una colección original.")
        pair = read(directory / "collection.json")
        expected = {row["path"]: row["sha256"] for row in task["inventory"]["files"]}
        if pair["input_files_sha256"] != expected or pair.get("original_modified"):
            raise ValueError("El par no conserva los archivos originales congelados.")
        calls = pair["acquisition"]["calls"]
        if pair["acquisition"]["calls_started"] != len(calls):
            raise ValueError("La contabilidad física del par no coincide.")
        observed_calls += len(calls)
        for index, call in enumerate(calls):
            request_path = directory / "requests" / f"call-{index:03d}.json"
            if (call["index"] != index or call["request_path"].replace("\\", "/") != f"requests/call-{index:03d}.json"):
                raise ValueError("Cambió el orden o la ruta de las peticiones originales.")
            request = read(request_path)
            if (request.get("model") != call["model"] or
                    _hash({key: request[key] for key in ("messages", "response_schema")}) != call["request_sha256"]):
                raise ValueError("Una petición original cambió.")
        report_calls = {}
        for role, branch in pair["branches"].items():
            if branch["report_path"].replace("\\", "/") != f"{role}/report.json":
                raise ValueError("Ruta de rama no compatible.")
            report_path = directory / role / "report.json"
            if digest(report_path) != branch["report_sha256"]:
                raise ValueError("Cambió un informe de rama original.")
            report = read(report_path)
            Workspace.reopen(directory / role)
            for call in report["calls"]:
                if call["state"] != "completed" and not _known_output_failure(call):
                    raise ValueError("Hay llamadas anteriores inciertas; no se permite continuarlas.")
                if call.get("response_path"):
                    response = directory / role / call["response_path"]
                    if digest(response) != call["response_sha256"]:
                        raise ValueError("Cambió una respuesta original.")
                report_calls.setdefault(role, []).append(call)
        if sum(len(values) for values in report_calls.values()) != len(calls):
            raise ValueError("Los informes y la adquisición no contienen las mismas llamadas.")
        branch_offsets = {role: 0 for role in report_calls}
        for physical in calls:
            role = physical["branch"]
            logical = report_calls[role][branch_offsets[role]]
            branch_offsets[role] += 1
            if (physical["request_sha256"] != logical["request_sha256"]
                    or physical["state"] != logical["state"] or physical["usage"] != logical["usage"]):
                raise ValueError("Una llamada física no coincide con su evidencia de rama.")
    if observed_calls != sum(item["calls_started"] for item in journal["invocations"]):
        raise ValueError("El journal global no coincide con las llamadas físicas registradas.")
    if prior_calls + observed_calls + max_new_calls > total_cap:
        raise ValueError("La continuación excedería el límite global de llamadas autorizado.")
    failed_task, next_task = incomplete[0], unstarted[0]
    pair_dir = source / failed_task["task_id"]
    pair = read(pair_dir / "collection.json")
    if (pair.get("status") != "incomplete" or pair.get("branch_order") != ["weak", "strong"]
            or set(pair["branches"]) != {"weak"} or (pair_dir / "strong").exists()
            or pair.get("pairing_error") or pair.get("stop_reason") != "pairing_or_operational_failure"):
        raise ValueError("El par no es una primera rama débil terminada con fuerte aún ausente.")
    report = read(pair_dir / "weak/report.json")
    if (len(report["calls"]) != 2 or report["calls"][0].get("category") != "output_truncated"
            or not _known_output_failure(report["calls"][0]) or report["calls"][1]["state"] != "completed"
            or report.get("error", {}).get("type") != "ValueError"
            or report["error"].get("stage") != "repair" or _semantic_outcome(report) is not False
            or not report.get("usage_complete") or pair["branches"]["weak"].get("success") is not None):
        raise ValueError("La recuperación exige truncamiento conocido seguido de reparación inválida conocida.")
    request = read(pair_dir / "requests/call-000.json")
    snapshot = {key: request[key] for key in ("messages", "response_schema")}
    if (_hash(snapshot) != pair["editing_request_sha256"] or request["messages"][-1]["content"] != pair["routing_text"]
            or hashlib.sha256(pair["routing_text"].encode()).hexdigest() != pair["routing_text_sha256"]
            or pair["initial_diagnostics"] != report["public_baseline"]["diagnostics"]):
        raise ValueError("El estado público inicial no coincide con la petición efectivamente enviada.")
    if max_new_calls < 6:
        raise ValueError("Reserva seis llamadas: hasta dos para fuerte y cuatro para la última tarea.")
    keys = ("max_output_tokens", "request_timeout_seconds", "request_budget", "minimum_interval_seconds")
    if any(failed_task["config"]["limits"][key] != next_task["config"]["limits"][key] for key in keys):
        raise ValueError("Las tareas restantes requieren límites de proveedor distintos.")
    return {"source": source, "destination": destination, "manifest": manifest, "journal": journal,
            "original_files_sha256": original_files, "source_changes": changes, "actual_sources": actual_sources,
            "observed_calls": observed_calls, "prior_calls": prior_calls, "total_cap": total_cap,
            "max_new_calls": max_new_calls, "failed_task": failed_task, "next_task": next_task,
            "pair": pair, "weak_report": report, "editing_snapshot": snapshot,
            "original_collector_bytes": original_collector_bytes, "collector_bytes": collector_bytes}


class _CountedProvider:
    def __init__(self, delegate, maximum, journal, invocation, directory, source_hashes):
        self.delegate, self.maximum, self.journal, self.invocation = delegate, maximum, journal, invocation
        self.directory, self.source_hashes = directory, source_hashes
        self.output_mode = "prompt_json"
        self.used = 0

    def manifest(self):
        return self.delegate.manifest() if callable(getattr(self.delegate, "manifest", None)) else {"kind": "injected"}

    def complete(self, model, messages):
        return self._send(model, messages, None)

    def complete_structured(self, model, messages, response_schema):
        return self._send(model, messages, response_schema)

    def _send(self, model, messages, response_schema):
        if self.used >= self.maximum or _sources() != self.source_hashes:
            raise ValueError("Se agotó el presupuesto o cambió el código durante la continuación.")
        self.used += 1
        self.invocation["calls_started"] = self.used
        _save(self.directory / "journal.json", self.journal)
        if response_schema is not None:
            return self.delegate.complete_structured(model, messages, response_schema)
        return self.delegate.complete(model, messages)


def recover(source, destination, *, prior_calls, max_new_calls=6, total_cap=40, provider=None, progress=print):
    """Create a derived audit and acquire only missing work; inject a local key provider if needed."""
    check = preflight(source, destination, prior_calls=prior_calls, max_new_calls=max_new_calls, total_cap=total_cap)
    source, destination = check["source"], check["destination"]
    injected = provider is not None
    limits = check["failed_task"]["config"]["limits"]
    if provider is None:
        from relay_router.maintenance_cli import _key
        provider = GroqMaintenanceProvider(_key(), max_requests=max_new_calls, output_mode="prompt_json",
            max_output_tokens=limits["max_output_tokens"], timeout=limits["request_timeout_seconds"],
            request_budget=limits["request_budget"], minimum_interval_seconds=limits["minimum_interval_seconds"],
            on_wait=lambda seconds: progress(f"Esperando {seconds:.0f} s por la cuota del modelo…") if progress else None)
        expected_provider = check["weak_report"]["provider"]["provider"]
        actual_provider = provider.manifest()
        if any(actual_provider.get(key) != value for key, value in expected_provider.items() if key != "max_requests"):
            raise ValueError("La configuración del proveedor no coincide con la llamada original.")
    # Recheck after credential entry; no input or source mutation can hide in the wait.
    fresh = preflight(source, destination, prior_calls=prior_calls, max_new_calls=max_new_calls, total_cap=total_cap)
    if fresh["original_files_sha256"] != check["original_files_sha256"] or fresh["actual_sources"] != check["actual_sources"]:
        raise ValueError("La evidencia o el código cambió después de la validación.")
    shutil.copytree(source, destination)
    archive = destination / "recovery-originals"
    archive.mkdir()
    (archive / "maintenance_collection.original.py").write_bytes(check["original_collector_bytes"])
    (archive / "maintenance_collection.continuation.py").write_bytes(check["collector_bytes"])
    for name in ("manifest.json", "journal.json"):
        shutil.copy2(source / name, archive / name)
    shutil.copy2(source / check["failed_task"]["task_id"] / "collection.json", archive / "incomplete-collection.json")
    for name in ("reference.json", "comparison.json", "comparison.md"):
        if (destination / name).exists():
            (destination / name).replace(archive / name)
    audit = {"schema_version": 1, "kind": "explicit_known_failure_continuation", "status": "running",
        "source_directory": str(source), "original_files_sha256": check["original_files_sha256"],
        "original_manifest_sha256": digest(source / "manifest.json"),
        "original_journal_sha256": digest(source / "journal.json"),
        "recovery_tool_sha256": digest(Path(__file__)),
        "source_amendment": check["source_changes"], "continuation_sources_sha256": check["actual_sources"],
        "reason": "Known truncated generation followed by a completed repair rejected by the frozen exact-text edit contract was incorrectly classified as operational uncertainty. This does not by itself establish model incapacity.",
        "evidence_policy": "Original run, raw branch reports, responses and requests retained unchanged; no earlier model call repeated.",
        "scientific_scope": "Collector adjudication correction only. Workflow, prompts, models, provider options, repair budget, tasks and graders remain frozen. Development evidence, not independent superiority.",
        "weak_adjudication": {"from": None, "to": False, "report_sha256": digest(source / check["failed_task"]["task_id"] / "weak/report.json")},
        "calls_before_this_run": prior_calls + check["observed_calls"], "max_new_calls": max_new_calls,
        "total_call_cap": total_cap, "new_calls_started": 0, "automatic_resume": False}
    _save(destination / "recovery-audit.json", audit)
    manifest, journal = deepcopy(check["manifest"]), deepcopy(check["journal"])
    manifest["source_amendment"] = {"original_manifest_sha256": audit["original_manifest_sha256"],
        "source_directory": str(source), "collector_changes": check["source_changes"],
        "continuation_sources_sha256": check["actual_sources"],
        "scope": "Original suite source hashes describe original calls; continuation reports describe their actual amended collector."}
    _save(destination / "manifest.json", manifest)
    journal["manifest_sha256"] = digest(destination / "manifest.json")
    invocation = {"max_calls": max_new_calls, "at_utc": datetime.now(timezone.utc).isoformat(), "calls_started": 0,
                  "acquisition_origin": "injected_provider_unverified" if injected else "groq_https",
                  "continuation": "explicit_known_failure_no_replayed_calls",
                  "sources_sha256": check["actual_sources"]}
    journal["invocations"].append(invocation)
    _save(destination / "journal.json", journal)
    counted = _CountedProvider(provider, max_new_calls, journal, invocation, destination, check["actual_sources"])
    failed_task, next_task = check["failed_task"], check["next_task"]
    task_id, pair = failed_task["task_id"], deepcopy(check["pair"])
    pair_directory = destination / task_id
    pair["branches"]["weak"] = _branch_summary(check["weak_report"], pair_directory, "weak", False)
    pair.update(status="running", active_branch="strong", paired_state_verified=False)
    pair.pop("stop_reason", None)
    pair["recovery_audit"] = "../recovery-audit.json"
    pair["branches"]["strong"] = {"status": "started", "success": None, "outcome_known": False}
    bridge = _SharedInspectionProvider(counted, pair_directory, pair)
    bridge.editing_request = deepcopy(check["editing_snapshot"])
    bridge.anchor_baseline_identity = _baseline_identity(check["weak_report"])
    bridge.begin("strong")
    bridge.save()
    journal["tasks"][task_id] = {"state": "recovering_known_failure", "collection_sha256": digest(pair_directory / "collection.json")}
    _save(destination / "journal.json", journal)

    def register(task, collection):
        case = destination / task["task_id"]
        entry = {"state": collection["status"], "collection_sha256": digest(case / "collection.json")}
        journal["tasks"][task["task_id"]] = entry
        if collection["status"] == "completed" and task.get("acceptance_files"):
            if validate_acceptance(Path(task["repository"]), task["config"]["source_files"],
                    task["config"]["test_files"], task["acceptance_files"]) != task["acceptance_inventory"]:
                raise ValueError("Cambió la aceptación reservada antes de calificar; no se ejecutó.")
            grade = grade_pair(task["repository"], task["config"]["source_files"], task["acceptance_files"],
                               case, timeout_seconds=task["config"]["limits"]["test_timeout_seconds"])
            entry["acceptance_sha256"] = digest(case / "acceptance.json")
            if grade["status"] != "completed":
                entry["state"] = "acceptance_blocked"
        _save(destination / "journal.json", journal)
        return entry["state"] == "completed"

    try:
        config = failed_task["config"]
        if progress:
            progress(f"Continuando solamente la rama fuerte de {task_id}; no se repite la rama débil.")
        strong = run_maintenance(root=failed_task["repository"], source_files=config["source_files"],
            test_files=config["test_files"], task=failed_task["task"], run_dir=pair_directory / "strong",
            provider=bridge, router=_BranchRouter("strong", "direct"), max_repairs=1,
            timeout_seconds=config["limits"]["test_timeout_seconds"], progress=progress,
            initial_diagnostics=deepcopy(pair["initial_diagnostics"]), inspection_mode="direct")
        pair["branches"]["strong"] = _branch_summary(strong, pair_directory, "strong", False)
        complete = (_semantic_outcome(strong) is not None and not bridge.mismatch
                    and all(call["state"] == "completed" or _known_output_failure(call) for call in strong["calls"]))
        pair.update(status="completed" if complete else "incomplete", paired_state_verified=bool(complete),
                    ended_at_utc=datetime.now(timezone.utc).isoformat())
        if not complete:
            pair["stop_reason"] = "pairing_or_operational_failure"
        bridge.save()
        if not register(failed_task, pair):
            audit["status"] = "incomplete"
            return destination
        if max_new_calls - counted.used < 4:
            raise ValueError("No queda la reserva íntegra para la última tarea; no se inicia.")
        task_id, config = next_task["task_id"], next_task["config"]
        current = Workspace(Path(next_task["repository"]), destination / "validation-unused", config["source_files"],
                            config["test_files"]).validate_inputs()
        if current != next_task["inventory"]:
            raise ValueError("Cambió el input de la última tarea después de validar.")
        if next_task.get("acceptance_files") and validate_acceptance(Path(next_task["repository"]),
                config["source_files"], config["test_files"], next_task["acceptance_files"]) != next_task["acceptance_inventory"]:
            raise ValueError("Cambió la aceptación reservada de la última tarea; no se inicia.")
        journal["tasks"][task_id] = {"state": "started"}
        _save(destination / "journal.json", journal)
        order = ("weak", "strong") if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 2 == 0 else ("strong", "weak")
        last = collect_pair(root=next_task["repository"], source_files=config["source_files"], test_files=config["test_files"],
            task=next_task["task"], run_dir=destination / task_id, provider=counted, branch_order=order,
            inspection_mode="direct", timeout_seconds=config["limits"]["test_timeout_seconds"], progress=progress)
        audit["status"] = "completed" if register(next_task, last) else "incomplete"
        if audit["status"] == "completed":
            load_run(destination, need_reference=False)
    except BaseException as exc:
        audit.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                     error_type=type(exc).__name__)
        raise
    finally:
        audit.update(new_calls_started=counted.used, total_calls_after=prior_calls + check["observed_calls"] + counted.used,
                     original_run_unchanged=_tree_hashes(source) == check["original_files_sha256"],
                     ended_at_utc=datetime.now(timezone.utc).isoformat())
        _save(destination / "recovery-audit.json", audit)
        _save(destination / "journal.json", journal)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prior-calls", type=int, required=True,
                        help="Llamadas anteriores ajenas a la carpeta fuente, incluidos diagnósticos.")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = preflight(args.source, args.destination, prior_calls=args.prior_calls)
            print(json.dumps({"status": "ready", "recorded_source_calls": result["observed_calls"],
                "prior_calls": args.prior_calls, "max_new_calls": 6, "global_cap": 40,
                "source_amendment": result["source_changes"], "model_calls_made": 0}, ensure_ascii=False, indent=2))
        else:
            directory = recover(args.source, args.destination, prior_calls=args.prior_calls)
            print(f"Continuación derivada: {directory / 'recovery-audit.json'}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Continuación: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
