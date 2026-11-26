"""Offline application diagnostic screening every branch of the final cohort.

No generation, provider access, extra repair, policy evaluation, or relabeling.
After the production EOL fix, apply each saved failed proposal once to a fresh
copy of its pre-application source and run public/private tests separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.maintenance import _object, _editable_path
from relay_router.maintenance_workspace import Workspace

class DiagnosticExclusion(ValueError):
    """Controlled reason why a saved mismatch is not solely EOL application."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def inside(root, name):
    require(type(name) is str and bool(name), "Ruta de evidencia vacía.")
    relative = PurePosixPath(name.replace("\\", "/"))
    require(not relative.is_absolute() and ":" not in str(relative) and ".." not in relative.parts,
            "La ruta de evidencia no es relativa y segura.")
    path = root.joinpath(*relative.parts).resolve()
    require(path.is_relative_to(root.resolve()), "La ruta de evidencia sale del directorio autorizado.")
    return path


def occurrences(text, fragment):
    if not fragment:
        return 0
    count, cursor = 0, 0
    while True:
        found = text.find(fragment, cursor)
        if found < 0:
            return count
        count, cursor = count + 1, found + 1


def eol_style(text):
    without_pairs = text.replace("\r\n", "")
    if "\r" in without_pairs:
        return "bare_cr"
    if "\r\n" in text and "\n" in without_pairs:
        return "mixed"
    return "crlf" if "\r\n" in text else "lf"


def audit_eol_only(payloads, edits):
    """Analyze sequential replacements without using this analysis to apply them."""
    state = {name: data.decode("utf-8") for name, data in payloads.items()}
    audit = []
    for index, edit in enumerate(edits):
        require(type(edit) is dict and set(edit) == {"path", "old_text", "new_text"}
                and all(type(value) is str for value in edit.values()), "Sustitución no compatible con el contrato original.")
        name = _editable_path(edit["path"], list(payloads))
        require(name is not None, "La respuesta intenta ampliar los archivos autorizados.")
        old, new, text = edit["old_text"], edit["new_text"], state[name]
        if not old:
            raise DiagnosticExclusion("empty_old_text")
        if any(eol_style(value) in {"mixed", "bare_cr"} for value in (text, old, new)):
            raise DiagnosticExclusion("unsafe_mixed_eol_or_bare_cr")
        normalized_text, normalized_old = text.replace("\r\n", "\n"), old.replace("\r\n", "\n")
        exact, equivalent = occurrences(text, old), occurrences(normalized_text, normalized_old)
        if equivalent == 0:
            raise DiagnosticExclusion("text_difference_beyond_line_endings")
        if exact > 1 or equivalent > 1:
            raise DiagnosticExclusion("ambiguous_or_overlapping_match")
        style = "\r\n" if eol_style(text) == "crlf" else "\n"
        native_old = normalized_old.replace("\n", style)
        native_new = new.replace("\r\n", "\n").replace("\n", style)
        state[name] = text.replace(native_old, native_new, 1)
        audit.append({"index": index, "path": name, "exact_occurrences": exact,
                      "eol_equivalent_occurrences": equivalent, "requires_eol_equivalence": exact == 0,
                      "file_eol_style": eol_style(text), "old_text_sha256": sha(old.encode()),
                      "new_text_sha256": sha(new.encode())})
    if not any(row["requires_eol_equivalence"] for row in audit):
        raise DiagnosticExclusion("no_line_ending_discrepancy")
    return audit, {name: sha(text.encode()) for name, text in state.items()}


def acquisition_hashes(directory, task_ids):
    # Concurrent offline scoring may add reference/comparison reports. Those
    # are not the immutable acquisition evidence checked by this diagnostic.
    paths = [directory / "manifest.json", directory / "journal.json"]
    for task_id in task_ids:
        paths.extend(path for path in (directory / task_id).rglob("*") if path.is_file())
    return {path.relative_to(directory).as_posix(): digest(path) for path in sorted(paths)}


def copy_payloads(directory, payloads):
    for name, data in payloads.items():
        path = inside(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def replay_case(run, destination, task, role, python_executable):
    task_id = task["task_id"]
    pair_directory, branch_directory = run / task_id, run / task_id / role
    pair, original_report = read(pair_directory / "collection.json"), read(branch_directory / "report.json")
    require(pair["branches"][role]["report_sha256"] == digest(branch_directory / "report.json"),
            "El informe original no coincide con la colección.")
    error = original_report.get("error", {})
    require(original_report["status"] == "blocked" and error.get("type") == "ValueError"
            and error.get("validation_message") == "old_text debe coincidir exactamente una sola vez.",
            "La rama no corresponde al fallo de aplicación seleccionado.")
    original_workspace = Workspace.reopen(branch_directory)
    sources, public_tests = task["config"]["source_files"], task["config"]["test_files"]
    require(original_workspace.source_files == sources and original_workspace.test_files == public_tests,
            "Cambió el alcance de la copia original.")
    public_payloads = {name: inside(original_workspace.originals, name).read_bytes() for name in sources + public_tests}
    source_payloads = {name: public_payloads[name] for name in sources}
    require(all(inside(original_workspace.worktree, name).read_bytes() == data for name, data in public_payloads.items()),
            "La rama ya aplicó cambios; no se puede reconstruir su estado anterior con este diagnóstico.")
    frozen_public = {row["path"]: row["sha256"] for row in task["inventory"]["files"]}
    require({name: sha(data) for name, data in public_payloads.items()} == frozen_public,
            "Las fuentes y pruebas públicas no corresponden al manifiesto original.")
    last = original_report["calls"][-1]
    require(last["state"] == "completed" and last["activity"] in {"editing", "repair"},
            "No hay una respuesta completada para reproducir.")
    response_path = inside(branch_directory, last["response_path"])
    require(digest(response_path) == last["response_sha256"], "Cambió la respuesta guardada.")
    proposal = _object(read(response_path)["content"])
    require(set(proposal) == {"summary", "edits"} and type(proposal["edits"]) is list and proposal["edits"],
            "La respuesta guardada no tiene el contrato de edición.")
    edits = [{**edit, "path": _editable_path(edit.get("path"), sources)} for edit in proposal["edits"]]
    evidence = {"task_id": task_id, "branch": role, "scope": "offline_saved_response_application_diagnostic",
                "original_report_sha256": digest(branch_directory / "report.json"),
                "original_response_sha256": digest(response_path), "original_response_path": str(response_path),
                "original_source_sha256": {name: sha(data) for name, data in source_payloads.items()},
                "replayed_call_index": last["index"], "replayed_activity": last["activity"],
                "original_generation_consumption": {"total_tokens": original_report.get("total_tokens"),
                    "usage_complete": original_report.get("usage_complete"),
                    "calls": [{"index": call["index"], "activity": call["activity"], "state": call["state"],
                               "model": call["decision"]["selected_model"], "usage": call.get("usage")}
                              for call in original_report["calls"]]},
                "additional_provider_calls": 0, "additional_repairs": 0,
                "original_outcome_relabelled": False, "counterfactual_cost_computed": False}
    try:
        audit, expected_hashes = audit_eol_only(source_payloads, edits)
    except DiagnosticExclusion as exc:
        result = {**evidence, "status": "excluded", "exclusion_reason": exc.reason}
        save(destination / "diagnostic.json", result)
        return result
    private_names = task["acceptance_files"]
    private_payloads = {name: inside(Path(task["repository"]), name).read_bytes() for name in private_names}
    require({name: sha(data) for name, data in private_payloads.items()}
            == {row["path"]: row["sha256"] for row in task["acceptance_inventory"]["files"]},
            "Las pruebas privadas no corresponden a la aceptación congelada.")
    inputs = destination / "public_inputs"
    copy_payloads(inputs, public_payloads)
    public = Workspace(inputs, destination / "public_execution", sources, public_tests)
    public.prepare()
    public.apply_edits(edits)  # Production implementation, never the audit's simulated text.
    patched = {name: inside(public.worktree, name).read_bytes() for name in sources}
    require({name: sha(data) for name, data in patched.items()} == expected_hashes,
            "La aplicación productiva no coincide con la sustitución limitada a EOL auditada.")
    public_verification = public.verify(python_executable, 30)
    public_integrity = all(inside(public.worktree, name).read_bytes() == data for name, data in patched.items())
    private_inputs = destination / "private_inputs"
    copy_payloads(private_inputs, {**patched, **private_payloads})
    private = Workspace(private_inputs, destination / "private_execution", sources, private_names)
    private.prepare()
    private_verification = private.verify(python_executable, 30)
    private_integrity = all(inside(private.worktree, name).read_bytes() == data for name, data in patched.items())
    result = {**evidence, "status": "replayed",
              "patched_source_sha256": expected_hashes, "acceptance_sha256": {name: sha(data) for name, data in private_payloads.items()},
              "edit_audit": audit,
              "public_verification": public_verification, "private_verification": private_verification,
              "source_integrity_preserved": public_integrity and private_integrity,
              "passes_both_after_application": bool(public_verification["passed"] and private_verification["passed"]
                                                    and public_integrity and private_integrity),
              "additional_provider_calls": 0, "additional_repairs": 0,
              "original_outcome_relabelled": False, "counterfactual_cost_computed": False}
    save(destination / "diagnostic.json", result)
    return result


def run_diagnostic(run, output, python_executable):
    run, output, python_executable = Path(run).resolve(), Path(output).resolve(), Path(python_executable).resolve()
    require(not output.exists(), "Usa una carpeta diagnóstica nueva; no se sobrescriben diagnósticos previos.")
    require(not output.is_relative_to(run) and not run.is_relative_to(output),
            "La salida diagnóstica debe quedar separada de la adquisición original.")
    manifest, journal = read(run / "manifest.json"), read(run / "journal.json")
    require(journal["manifest_sha256"] == digest(run / "manifest.json"), "El manifiesto original cambió.")
    tasks = {task["task_id"]: task for task in manifest["suite"]["tasks"] if task["partition"] == manifest["partition"]}
    require(len(tasks) == 8 and all(journal["tasks"].get(task_id, {}).get("state") == "completed" for task_id in tasks),
            "Se requiere la cohorte final de ocho casos completos antes del diagnóstico.")
    screening, candidates = [], []
    for task_id in tasks:
        pair = read(run / task_id / "collection.json")
        require(journal["tasks"][task_id]["collection_sha256"] == digest(run / task_id / "collection.json"),
                "La colección no corresponde al diario congelado.")
        for role in ("weak", "strong"):
            location = run / task_id / role / "report.json"
            require(pair["branches"][role]["report_sha256"] == digest(location), "Cambió un informe de rama.")
            branch = read(location)
            error = branch.get("error", {})
            eligible = (branch["status"] == "blocked" and error.get("type") == "ValueError"
                        and error.get("validation_message") == "old_text debe coincidir exactamente una sola vez.")
            screening.append({"task_id": task_id, "branch": role, "original_status": branch["status"],
                              "report_sha256": digest(location), "application_mismatch_candidate": eligible})
            if eligible:
                candidates.append((task_id, role))
    original_hashes = acquisition_hashes(run, tasks)
    output.mkdir(parents=True)
    result = {"schema_version": 1, "scope": "post_failure_selected_offline_application_diagnostic",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "original_run": str(run),
              "original_manifest_sha256": digest(run / "manifest.json"),
              "original_journal_sha256": digest(run / "journal.json"),
              "replay_tool_sha256": digest(Path(__file__)),
              "workspace_implementation_sha256": digest(ROOT / "src/relay_router/maintenance_workspace.py"),
              "original_workspace_implementation_sha256": manifest["suite"]["sources_sha256"].get("maintenance_workspace.py"),
              "provider_calls": 0, "remaining_repairs_executed": False, "new_comparison": False,
              "original_outcomes_relabelled": False, "counterfactual_cost_computed": False,
              "selection": "All 16 branches screened; all recorded old_text mismatch failures audited, with explicit exclusions.",
              "screening": screening, "screened_branch_count": len(screening),
              "application_mismatch_count": len(candidates),
              "original_unchanged_scope": "manifest, journal and every file in the eight acquisition task directories",
              "cases": [], "status": "running"}
    try:
        for task_id, role in candidates:
            destination = output / f"{task_id}-{role}"
            destination.mkdir()
            try:
                row = replay_case(run, destination, tasks[task_id], role, python_executable)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row = {"task_id": task_id, "branch": role, "status": "diagnostic_blocked", "error_type": type(exc).__name__}
                save(destination / "diagnostic.json", row)
            result["cases"].append(row)
        result["replayed_count"] = sum(row.get("status") == "replayed" for row in result["cases"])
        result["excluded_count"] = sum(row.get("status") == "excluded" for row in result["cases"])
        result["unresolved_count"] = sum(row.get("status") == "diagnostic_blocked" for row in result["cases"])
        result["status"] = "completed" if not result["unresolved_count"] else "blocked"
    finally:
        result["original_run_unchanged"] = acquisition_hashes(run, tasks) == original_hashes
        if not result["original_run_unchanged"]:
            result["status"] = "original_evidence_changed"
        result["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "report.json", result)
        lines = ["# Diagnóstico offline de aplicación LF/CRLF", "",
                 "Se revisan las 16 ramas de los ocho casos y se auditan todos los fallos registrados de coincidencia old_text. "
                 "Los desacuerdos que no sean exclusivamente EOL se excluyen con causa explícita. Se aplica cada propuesta elegible "
                 "guardada una sola vez sobre una copia del estado original y se ejecutan por separado pruebas públicas y privadas. "
                 "No hay llamadas API, reparaciones nuevas, comparación nueva ni cambio de las etiquetas originales.", "",
                 f"Ramas revisadas: {len(screening)}. Fallos de aplicación auditados: {len(candidates)}. "
                 f"Reproducidos: {result.get('replayed_count', 0)}. Excluidos: {result.get('excluded_count', 0)}. "
                 f"Pendientes por error: {result.get('unresolved_count', 0)}.", "",
                 "| Caso | Rama | Estado/causa | Públicas | Privadas | Integridad |", "|---|---|---|---|---|---|"]
        for row in result["cases"]:
            lines.append(f"| {row['task_id']} | {row['branch']} | {row.get('exclusion_reason', row['status'])} | "
                         f"{row.get('public_verification', {}).get('status', 'no ejecutadas')} | "
                         f"{row.get('private_verification', {}).get('status', 'bloqueado')} | {row.get('source_integrity_preserved', 'desconocida')} |")
        lines.extend(["", f"Adquisición original sin cambios: {result['original_run_unchanged']}.", "",
                      "El consumo archivado es el de las generaciones originales, incluido cualquier intento anterior fallido. "
                      "No se estima costo de una política alternativa ni se atribuye este resultado a una futura generación. "
                      "Pasar estas pruebas no demuestra corrección completa ni generalización. Hashes y evidencia: report.json y carpetas por caso.", ""])
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audita todas las ramas y reproduce fallos exclusivamente EOL; sin APIs.")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/maintenance-comparison/20261117-live-v3-complete")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/maintenance-line-ending-diagnostic-v1")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    try:
        result = run_diagnostic(args.run, args.output, args.python)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Diagnóstico detenido: {type(exc).__name__}.", file=sys.stderr)
        return 2
    print(f"Diagnóstico {result['status']}; originales intactos={result['original_run_unchanged']}; API=0. {args.output}")
    return 0 if result["status"] == "completed" and result["original_run_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
