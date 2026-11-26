"""Bounded acquisition and audit for the maintenance routing comparison.

No benchmark is downloaded, no model is trained on a demonstration, and no
provider credentials are stored. A suite fixes task groups/splits before calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .maintenance import _save
from .maintenance_cli import load_config, _key
from .maintenance_workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "configs/maintenance_competition_demo.json"
TARIFFS = ROOT / "configs/groq_paid_tariffs_20261112.json"
PARTITIONS = ("fit", "calibration", "development", "holdout")
MODELS = {"weak": "groq/openai/gpt-oss-20b", "strong": "groq/openai/gpt-oss-120b"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def acquisition_contract(suite):
    contract = {key: suite[key] for key in ("suite_sha256", "tariffs_sha256", "sources_sha256", "scope")}
    return {**contract, "tasks_sha256": object_hash(suite["tasks"]), "models": MODELS, "continuation": {"inspection_model": MODELS["weak"],
            "repair_model": MODELS["strong"], "max_repairs": 1}}


def load_suite(path):
    path = Path(path).resolve()
    suite = read(path)
    if (type(suite) is not dict or set(suite) != {"schema_version", "scope", "tasks"}
            or type(suite["schema_version"]) is not int or suite["schema_version"] != 1
            or suite["scope"] not in {"diagnostic", "development_and_reserved_holdout"}
            or type(suite["tasks"]) is not list or not suite["tasks"]):
        raise ValueError("La suite requiere schema_version=1, scope y tasks explícitos.")
    tasks, ids, groups, duplicates = [], set(), {}, {}
    for row in suite["tasks"]:
        required = {"task_id", "group_id", "partition", "config", "task"}
        if type(row) is not dict or set(row) not in (required, required | {"acceptance_files"}):
            raise ValueError("Cada tarea requiere task_id, group_id, partition, config y task.")
        if (type(row["task_id"]) is not str or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", row["task_id"])
                or row["task_id"] in ids or type(row["group_id"]) is not str or not row["group_id"].strip()
                or row["partition"] not in PARTITIONS or type(row["task"]) is not str
                or not row["task"].strip() or len(row["task"]) > 16000
                or type(row["config"]) is not str):
            raise ValueError("Identidad, partición o petición de tarea inválida.")
        if suite["scope"] == "diagnostic" and row["partition"] != "development":
            raise ValueError("El ejemplo diagnóstico no puede convertirse en entrenamiento o holdout.")
        group = row["group_id"]
        if group in groups and groups[group] != row["partition"]:
            raise ValueError("Un grupo no puede aparecer en particiones diferentes.")
        config, root = load_config(path.parent / row["config"])
        if config["models"] != {"inspection": MODELS["weak"], "editing": MODELS["strong"]} or config["limits"]["max_repairs"] != 1:
            raise ValueError("La comparación fija GPT-OSS 20B/120B y exactamente una reparación permitida.")
        inventory = Workspace(root, root / "unused-validation", config["source_files"], config["test_files"]).validate_inputs()
        acceptance = {}
        if "acceptance_files" in row:
            from .maintenance_acceptance import validate_acceptance
            acceptance["acceptance_inventory"] = validate_acceptance(
                root, config["source_files"], config["test_files"], row["acceptance_files"])
        # Outcome-independent duplicate rule: normalized request and input bytes.
        duplicate = object_hash({"task": " ".join(row["task"].split()).casefold(),
                                 "files": sorted(item["sha256"] for item in inventory["files"])})
        if duplicate in duplicates:
            raise ValueError("La suite repite la misma petición y archivos; deduplica antes de recoger resultados.")
        duplicates[duplicate] = row["task_id"]
        ids.add(row["task_id"])
        groups[group] = row["partition"]
        tasks.append({**row, "config": config, "repository": str(root), "inventory": inventory, **acceptance})
    return {"schema_version": 1, "scope": suite["scope"], "tasks": tasks,
            "suite_sha256": digest(path), "tariffs": read(TARIFFS), "tariffs_sha256": digest(TARIFFS),
            "sources_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("maintenance*.py"))}}


def branch_cost(branch, tariffs):
    """Common noncached tariff scenario, not an invoice or assumed cache usage."""
    total = 0.
    for call in branch["calls"]:
        usage = call.get("usage")
        if (type(usage) is not dict or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ("input_tokens", "output_tokens"))):
            return None
        model = call.get("model") or call.get("decision", {}).get("selected_model")
        tariff = tariffs["models"][model]
        total += (usage["input_tokens"] * tariff["input_usd_per_million"]
                  + usage["output_tokens"] * tariff["output_usd_per_million"]) / 1e6
    return total if branch["calls"] else None


def load_run(directory, *, need_reference=True):
    directory = Path(directory).resolve()
    manifest, journal = read(directory / "manifest.json"), read(directory / "journal.json")
    if journal["manifest_sha256"] != digest(directory / "manifest.json"):
        raise ValueError("El manifiesto cambió después de comenzar la adquisición.")
    reference = read(directory / "reference.json") if need_reference else None
    scored = {row["task_id"]: row for row in reference["rows"]} if reference else {}
    if reference and (reference["reference"].get("checkpoint_verified") is not True or
                      reference.get("journal_sha256") != digest(directory / "journal.json")):
        raise ValueError("La referencia no está verificada o no corresponde a esta adquisición.")
    rows = []
    for task in manifest["suite"]["tasks"]:
        if task["partition"] != manifest["partition"]:
            continue
        entry = journal["tasks"].get(task["task_id"])
        if not entry or entry["state"] != "completed":
            continue
        location = directory / task["task_id"] / "collection.json"
        if digest(location) != entry["collection_sha256"]:
            raise ValueError("Una colección cambió después de registrarse.")
        pair = read(location)
        expected_files = {item["path"]: item["sha256"] for item in task["inventory"]["files"]}
        if (pair.get("input_files_sha256") != expected_files or pair.get("status") != "completed"
                or pair.get("paired_state_verified") is not True):
            raise ValueError("El par no corresponde a los archivos congelados antes de adquirirlo.")
        for branch in pair["branches"].values():
            report_path = directory / task["task_id"] / branch["report_path"]
            if digest(report_path) != branch["report_sha256"]:
                raise ValueError("Un informe de rama cambió después de recogerlo.")
        contract = acquisition_contract(manifest["suite"])
        if (manifest.get("analysis_plan") or {}).get("inspection_mode") == "direct":
            contract["continuation"].update(inspection_model=None, inspection_mode="direct")
        if pair.get("continuation") != contract["continuation"]:
            raise ValueError("La continuación del par no corresponde al contrato.")
        acceptance = None
        if "acceptance_files" in task:
            from .maintenance_acceptance import load_acceptance
            acceptance = load_acceptance(task, directory / task["task_id"], pair, entry.get("acceptance_sha256"))
        row = {"task_id": task["task_id"], "group_id": task["group_id"], "partition": task["partition"],
               "acquisition_contract": contract,
               "routing_text": pair["routing_text"], "branches": {
                   role: {"success": branch["success"], "total_tokens": branch["total_tokens"],
                          "cost_usd": branch_cost(branch, manifest["suite"]["tariffs"])}
                   for role, branch in pair["branches"].items()}}
        if acceptance is not None:
            row["independent_quality_validation"] = acceptance.get("private_verdicts_complete", True)
            for role, branch in row["branches"].items():
                branch.update(visible_success=branch["success"],
                              acceptance_success=acceptance["branches"][role]["acceptance_success"],
                              success=acceptance["branches"][role]["success"])
        if reference:
            score = scored.get(task["task_id"])
            if not score or score["routing_text_sha256"] != hashlib.sha256(row["routing_text"].encode()).hexdigest():
                raise ValueError("La referencia no puntuó el mismo mensaje de edición.")
            row["reference_probability"] = score["probability"]
        rows.append(row)
    return rows, manifest, journal, reference


def collect(suite_path, directory, partition, max_calls=5, *, provider=None, policy_sha256=None, analysis_plan=None):
    from .maintenance_collection import collect_pair
    from .maintenance_provider import GroqMaintenanceProvider
    injected = provider is not None
    inspection_mode = (analysis_plan or {}).get("inspection_mode", "model")
    output_mode = (analysis_plan or {}).get("output_mode", "strict")
    if inspection_mode not in {"model", "direct"} or output_mode not in {"strict", "prompt_json"}:
        raise ValueError("Modo de inspección o formato desconocido.")
    pair_reserve = 4 if inspection_mode == "direct" else 5
    if type(max_calls) is not int or not pair_reserve <= max_calls <= 30:
        raise ValueError(f"max_calls debe estar entre {pair_reserve} y 30; cada tarea reserva {pair_reserve} llamadas.")
    suite = load_suite(suite_path)
    if partition not in PARTITIONS:
        raise ValueError("Partición desconocida.")
    selected = [t for t in suite["tasks"] if t["partition"] == partition]
    if not selected:
        raise ValueError("La suite no contiene tareas de esa partición.")
    if partition == "holdout" and not policy_sha256:
        raise ValueError("Congela una política antes de adquirir holdout.")
    keys = ("max_output_tokens", "request_timeout_seconds", "request_budget", "minimum_interval_seconds")
    limits = selected[0]["config"]["limits"]
    if any(any(task["config"]["limits"][key] != limits[key] for key in keys) for task in selected):
        raise ValueError("Las tareas de una adquisición deben compartir los límites del proveedor.")
    directory = Path(directory).resolve()
    manifest = {"schema_version": 1, "suite": suite, "partition": partition, "policy_sha256": policy_sha256,
                "analysis_plan": analysis_plan}
    if directory.exists():
        if read(directory / "manifest.json") != manifest:
            raise ValueError("No se puede reanudar con otra suite, política, código o archivos.")
        journal = read(directory / "journal.json")
        if journal["manifest_sha256"] != digest(directory / "manifest.json"):
            raise ValueError("Manifiesto modificado.")
        if any(entry["state"] != "completed" for entry in journal["tasks"].values()):
            raise ValueError("Hay una tarea incompleta o incierta; no se reenvía automáticamente. Revisa sus eventos.")
        load_run(directory, need_reference=False)
    else:
        directory.mkdir(parents=True)
        _save(directory / "manifest.json", manifest)
        journal = {"manifest_sha256": digest(directory / "manifest.json"), "tasks": {}, "invocations": []}
        _save(directory / "journal.json", journal)
    todo = [task for task in selected if task["task_id"] not in journal["tasks"]]
    if not todo:
        return directory
    # One provider across cases preserves per-model quota waits. All cases must
    # share provider parameters; test timeouts/file scopes may differ.
    print(f"Salida: {directory}\nMáximo de llamadas reales en esta ejecución: {max_calls}.")
    print("Se enviarán solo la petición y los archivos Python declarados a tu cuenta Groq Free.")
    print("Las pruebas ejecutan código del proyecto con tus permisos. Usa repositorios de confianza.")
    if provider is None:
        provider = GroqMaintenanceProvider(_key(), max_requests=max_calls,
            output_mode=output_mode,
            max_output_tokens=limits["max_output_tokens"], timeout=limits["request_timeout_seconds"],
            request_budget=limits["request_budget"], minimum_interval_seconds=limits["minimum_interval_seconds"],
            on_wait=lambda seconds: print(f"Esperando {seconds:.0f} s por la cuota del modelo…", flush=True))
    invocation = {"max_calls": max_calls, "at_utc": datetime.now(timezone.utc).isoformat(), "calls_started": 0,
                  "acquisition_origin": "injected_provider_unverified" if injected else "groq_https"}
    journal["invocations"].append(invocation)
    _save(directory / "journal.json", journal)
    for task in todo:
        if max_calls - invocation["calls_started"] < pair_reserve:
            break
        task_id = task["task_id"]
        config = task["config"]
        current = Workspace(Path(task["repository"]), directory / "unused-validation",
                            config["source_files"], config["test_files"]).validate_inputs()
        if current != task["inventory"]:
            raise ValueError("Los archivos cambiaron después de congelar la suite; no se enviaron en esta tarea.")
        if task.get("acceptance_files"):
            from .maintenance_acceptance import validate_acceptance
            if validate_acceptance(Path(task["repository"]), config["source_files"], config["test_files"],
                                   task["acceptance_files"]) != task["acceptance_inventory"]:
                raise ValueError("Las pruebas reservadas cambiaron desde la congelación; no se envió esta tarea.")
        journal["tasks"][task_id] = {"state": "started"}
        _save(directory / "journal.json", journal)
        # Independent of observed outcomes; alternate first candidate by task ID.
        order = ("weak", "strong") if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 2 == 0 else ("strong", "weak")
        print(f"Caso {task_id}: mismo punto de partida, ambas primeras ediciones.", flush=True)
        pair = collect_pair(root=task["repository"], source_files=config["source_files"], test_files=config["test_files"],
            task=task["task"], run_dir=directory / task_id, provider=provider, branch_order=order,
            inspection_mode=inspection_mode,
            timeout_seconds=config["limits"]["test_timeout_seconds"], progress=lambda message: print(message, flush=True))
        invocation["calls_started"] += pair["acquisition"]["calls_started"]
        journal["tasks"][task_id] = {"state": pair["status"],
            "collection_sha256": digest(directory / task_id / "collection.json")}
        if pair["status"] == "completed" and task.get("acceptance_files"):
            from .maintenance_acceptance import grade_pair
            grading = grade_pair(root=task["repository"], source_files=config["source_files"],
                acceptance_files=task["acceptance_files"], pair_directory=directory / task_id,
                timeout_seconds=config["limits"]["test_timeout_seconds"])
            journal["tasks"][task_id]["acceptance_sha256"] = digest(directory / task_id / "acceptance.json")
            if grading["status"] != "completed":
                journal["tasks"][task_id]["state"] = "acceptance_blocked"
        _save(directory / "journal.json", journal)
        if journal["tasks"][task_id]["state"] != "completed":
            break
    return directory


def score(directory):
    from .maintenance_reference import score_records
    directory = Path(directory)
    rows, _, _, _ = load_run(directory, need_reference=False)
    if not rows:
        raise ValueError("No hay pares completos para puntuar.")
    result = score_records([{k: row[k] for k in ("task_id", "routing_text")} for row in rows])
    result["journal_sha256"] = digest(directory / "journal.json")
    _save(directory / "reference.json", result)
    return result


def evaluate(directory, policy=None):
    from .maintenance_comparison import compare_records, select_reference_threshold
    rows, manifest, journal, reference = load_run(directory)
    if not rows:
        raise ValueError("No hay pares completos para comparar.")
    if manifest["partition"] not in {"development", "holdout"}:
        raise ValueError("Fit/calibración se reservan para entrenamiento, no para un reporte de ventaja.")
    expected_count = sum(task["partition"] == manifest["partition"] for task in manifest["suite"]["tasks"])
    if any(item["state"] == "acceptance_blocked" for item in journal["tasks"].values()):
        raise ValueError("La aceptación reservada tiene un fallo operativo; no se descarta para comparar.")
    if manifest.get("analysis_plan") and len(rows) != expected_count:
        raise ValueError("La comparación predeclarada está incompleta; faltan tareas por adquirir o resolver.")
    if manifest["partition"] == "holdout" and len(rows) != expected_count:
        raise ValueError("Holdout incompleto: no se descartan tareas fallidas o pendientes para acreditar ventaja.")
    if policy is None:
        if manifest["partition"] != "development":
            raise ValueError("El holdout requiere una política aprendida y congelada.")
        plan = manifest.get("analysis_plan") or {}
        fixed_threshold = plan.get("reference_threshold")
        selection = ({"threshold": fixed_threshold, "selection_scope": "fixed_before_acquisition"}
                     if fixed_threshold is not None else select_reference_threshold(rows))
        verified = plan.get("operational_mode") == "verified"
        declaration = {"name": "relay_verified_cascade_v1" if verified else "current_activity_rule_inspection20_edit120",
                       "frozen": True, "learned": False,
                       "scope": "operational cheap-first cascade; no learned advantage" if verified else "configured activity rule"}
        for row in rows:
            row["relay_choice"] = "weak" if verified else "strong"
        from .maintenance_routing import ActivityRouter
        selector = ActivityRouter(MODELS["weak"], MODELS["strong"], mode="verified" if verified else "activity")
        router_times = [selector.route("editing")["router_seconds"] for _ in rows]
    else:
        declaration = policy.manifest()
        if manifest["partition"] == "holdout" and manifest["policy_sha256"] != declaration["policy_sha256"]:
            raise ValueError("La política no coincide con la congelada antes de adquirir holdout.")
        policy.audit_records(rows, evaluation_split=manifest["partition"])
        selection = {"threshold": declaration["reference_threshold"], "selection_scope": "frozen_in_training_development"}
        router_times = []
        for row in rows:
            decision = policy.choose(row["routing_text"])
            row["relay_choice"] = decision["choice"]
            router_times.append(decision["router_seconds"])
    report = compare_records(rows, selection["threshold"], declaration)
    if manifest["partition"] == "development" and selection.get("selection_scope") == "fixed_before_acquisition":
        try:
            calibrated = select_reference_threshold(rows)
            report["development_calibrated_reference"] = {
                "selection": calibrated,
                "comparison": compare_records(rows, calibrated["threshold"], declaration)["partitions"]["development"],
                "scope": "selected on these same development tasks; separate from fixed primary reference"}
        except ValueError:
            report["development_calibrated_reference"] = {"available": False, "reason": "No eligible threshold with complete costs."}
    acceptance = all(task.get("acceptance_files") for task in manifest["suite"]["tasks"] if task["partition"] == manifest["partition"])
    report.update(reference=reference["reference"], reference_selection=selection,
        reference_router_seconds=[row["router_seconds"] for row in reference["rows"]], relay_router_seconds=router_times,
        cohort_scope=manifest["suite"]["scope"], policy_manifest_frozen_before_collection=manifest["policy_sha256"],
        incomplete_or_unstarted_tasks=len([t for t in manifest["suite"]["tasks"] if t["partition"] == manifest["partition"]]) - len(rows),
        end_to_end_latency_advantage=None, full_operating_cost_advantage=None,
        proof_of_complete_task_correctness=False, cost_scenario="noncached_list_tariff_not_billing",
        acquisition_calls=sum(item["calls_started"] for item in journal["invocations"]),
        acquisition_origins=sorted({item.get("acquisition_origin", "unknown") for item in journal["invocations"]}),
        independent_acceptance_configured=acceptance,
        independent_acceptance_evaluated=acceptance and all(row.get("independent_quality_validation") for row in rows),
        analysis_plan=manifest.get("analysis_plan"),
        limitation="Paired replay from the same public initial state. Authored development tasks; no stochastic replications or independent holdout.")
    # Oracle is solely a retrospective feasibility bound, never a deployable router.
    known = all(branch["cost_usd"] is not None for row in rows for branch in row["branches"].values())
    report["development_opportunity"] = ({
        "all_weak_branches_passed": all(row["branches"]["weak"]["success"] for row in rows),
        "outcome_disagreements": sum(row["branches"]["weak"]["success"] != row["branches"]["strong"]["success"] for row in rows),
        "message": "No entrenar con clases constantes; si la cascada ya resuelve todo, hay que justificar cualquier router adicional."
    } if manifest["partition"] == "development" else None)
    report["all_selected_costs_known"] = known
    directory = Path(directory)
    _save(directory / "comparison.json", report)
    partition = report["partitions"][manifest["partition"]]
    lines = ["Comparación de mantenimiento", "", f"Casos completos: {len(rows)}. Alcance: {manifest['suite']['scope']}.",
             "Costo: escenario tarifario sin caché sobre uso observado; no ahorro facturado en Groq Free.",
             "Referencia: wrapper local del checkpoint RouteLLM-BERT transferido; no MF ni Not Diamond.",
             "Estado: todavía no se acredita ventaja competitiva independiente.", ""]
    for name, metric in partition["policies"].items():
        lines.append(f"{name}: {metric['success_count']}/{len(rows)} pasan; costo USD escenario={metric['total_cost_usd']}.")
    lines.extend(["", "Los controles weak_first y strong_first incluyen una reparación 120B si falla la primera edición.",
                  "Los intervalos pareados, costos desconocidos y decisiones están en comparison.json.",
                  "Una sola demo, pruebas visibles o datos usados para seleccionar no demuestran superioridad general."])
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparación acotada Relay / RouteLLM-BERT en mantenimiento Python.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run = commands.add_parser("run")
    run.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run.add_argument("--partition", choices=PARTITIONS, default="development")
    run.add_argument("--output", type=Path)
    run.add_argument("--max-calls", type=int, default=5)
    run.add_argument("--collect-only", action="store_true")
    run.add_argument("--policy", type=Path)
    run.add_argument("--policy-sha256")
    run.add_argument("--mode", choices=("activity", "verified"), default="activity")
    run.add_argument("--reference-threshold", type=float)
    run.add_argument("--inspection-mode", choices=("model", "direct"), default="model")
    run.add_argument("--output-mode", choices=("strict", "prompt_json"), default="strict")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("directory", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("directory", type=Path)
    compare.add_argument("--policy", type=Path)
    compare.add_argument("--policy-sha256")
    train = commands.add_parser("train")
    train.add_argument("directories", nargs="+", type=Path)
    train.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        policy = None
        if getattr(args, "policy", None):
            from .maintenance_learning import load_policy
            policy = load_policy(args.policy, expected_sha256=args.policy_sha256)
        elif getattr(args, "policy_sha256", None):
            raise ValueError("policy-sha256 requiere el archivo policy.")
        if args.command == "check":
            suite = load_suite(args.suite)
            print(f"Suite válida: {len(suite['tasks'])} tareas, scope={suite['scope']}. Sin APIs.")
            print("El ejemplo predeterminado comprueba el comparador; no sirve para entrenar ni demostrar ventaja.")
        elif args.command == "score":
            score(args.directory)
            print(f"Referencia puntuada: {args.directory / 'reference.json'}")
        elif args.command == "compare":
            evaluate(args.directory, policy)
            print(f"Informe: {args.directory / 'comparison.md'}")
        elif args.command == "train":
            from .maintenance_learning import train_policy
            rows = []
            training_acquisition_contract = None
            for directory in args.directories:
                items, manifest, journal, _ = load_run(directory)
                if manifest["suite"]["scope"] == "diagnostic" or manifest["partition"] == "holdout":
                    raise ValueError("No se entrena con la demo diagnóstica ni con holdout.")
                if any(item.get("acquisition_origin") != "groq_https" for item in journal["invocations"]):
                    raise ValueError("La CLI no entrena con proveedores simulados o de procedencia desconocida.")
                if len(items) != sum(task["partition"] == manifest["partition"] for task in manifest["suite"]["tasks"]):
                    raise ValueError("La partición está incompleta; no se descartan fallos de adquisición para entrenar.")
                contract = items[0]["acquisition_contract"]
                if training_acquisition_contract is not None and contract != training_acquisition_contract:
                    raise ValueError("Las adquisiciones no comparten suite, tarifas, código y continuación congelados.")
                training_acquisition_contract = contract
                rows.extend(items)
            result = train_policy(rows, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if args.reference_threshold is not None and not 0 <= args.reference_threshold <= 1.000001:
                raise ValueError("El umbral fijo debe estar entre 0 y 1.000001.")
            directory = args.output or ROOT / "outputs/maintenance-comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            scorer = ROOT / ".venv-routellm/Scripts/python.exe"
            if not args.collect_only and (not scorer.is_file() or not (ROOT / "artifacts/routellm-bert/manifest.json").is_file()):
                raise ValueError("Falta el entorno o checkpoint BERT local; no se enviaron peticiones ni se instala nada automáticamente.")
            if args.partition == "holdout" and (policy is None or not policy.manifest().get("acquisition_contract")):
                raise ValueError("Holdout requiere política con procedencia de adquisición real y digest externo.")
            if args.partition == "holdout" and acquisition_contract(load_suite(args.suite)) != policy.manifest()["acquisition_contract"]:
                raise ValueError("La suite de holdout difiere de la congelada al entrenar; no se hicieron llamadas.")
            collect(args.suite, directory, args.partition, args.max_calls,
                    policy_sha256=policy.manifest()["policy_sha256"] if policy else None,
                    analysis_plan={"operational_mode": args.mode, "reference_threshold": args.reference_threshold,
                                   "inspection_mode": args.inspection_mode, "output_mode": args.output_mode})
            journal_path = Path(directory) / "journal.json"
            if journal_path.is_file() and any(item["state"] != "completed" for item in read(journal_path)["tasks"].values()):
                print(f"La adquisición tiene una tarea bloqueada; no hay comparación completa. Revisa {journal_path}", file=sys.stderr)
                return 2
            if not args.collect_only:
                environment = {key: value for key, value in os.environ.items()
                               if not key.upper().endswith(("_API_KEY", "_TOKEN"))}
                result = subprocess.run([str(scorer), str(ROOT / "tools/relay_compare_maintenance.py"), "score", str(directory)],
                                        check=False, env=environment)
                if result.returncode:
                    return result.returncode
                if args.partition in {"development", "holdout"}:
                    evaluate(directory, policy)
                    print(f"Informe: {directory / 'comparison.md'}")
            print(f"Adquisición: {directory / 'journal.json'}")
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"Relay comparación: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
