"""Private, post-trajectory acceptance checks with replayable file bindings.

Acceptance files are never added to the model worktrees or model messages. The
fixed Workspace runner executes trusted local code; it is not a security sandbox.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .maintenance_workspace import Workspace, _digest, _inside, _json, _read, _write


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_hash(path):
    return _digest(Path(path).read_bytes())


def _hashes(root, names):
    return {name: _digest(_read(_inside(root, name))) for name in names}


def validate_acceptance(root, source_files, public_test_files, acceptance_files):
    """Validate disjoint explicit Python paths; return only private inventory."""
    if type(acceptance_files) is not list or not acceptance_files:
        raise ValueError("acceptance_files requiere una lista no vacía de archivos Python.")
    root = Path(root).resolve()
    workspace = Workspace(root, root / "unused-acceptance-validation", source_files,
                          public_test_files + acceptance_files)
    inventory = workspace.validate_inputs()
    files = [row for row in inventory["files"] if row["path"] in acceptance_files]
    return {"files": files, "total_bytes": sum(row["bytes"] for row in files)}


def grade_pair(root, source_files, acceptance_files, pair_directory,
               python_executable=None, timeout_seconds=30):
    """Grade both finished branches once, without repairs or original edits.

    Missing private verdicts remain None; they block the joint target only when
    public tests succeeded. A known public failure remains a joint failure.
    Collection metadata is read-only and bound by its hash.
    """
    root, directory = Path(root).resolve(), Path(pair_directory).resolve()
    pair = _load(directory / "collection.json")
    if (pair.get("status") != "completed" or pair.get("paired_state_verified") is not True
            or set(pair.get("branches", {})) != {"weak", "strong"}
            or pair.get("source_files") != source_files
            or Path(pair["repository"]).resolve() != root
            or any(type(branch.get("success")) is not bool for branch in pair["branches"].values())):
        raise ValueError("La aceptación requiere un par completo y resultados públicos conocidos.")
    if (directory / "acceptance.json").exists() or (directory / "acceptance").exists():
        raise ValueError("La aceptación ya existe; no se vuelve a ejecutar después de observarla.")
    inventory = validate_acceptance(root, source_files, pair["test_files"], acceptance_files)
    acceptance_hashes = {row["path"]: row["sha256"] for row in inventory["files"]}
    public_names = source_files + pair["test_files"]
    if _hashes(root, public_names) != pair["input_files_sha256"]:
        raise ValueError("Los archivos originales cambiaron después de adquirir el par.")
    # Read the private inputs once; both branches receive identical frozen bytes.
    private_payloads = {name: _read(_inside(root, name)) for name in acceptance_files}
    if {name: _digest(data) for name, data in private_payloads.items()} != acceptance_hashes:
        raise ValueError("Las pruebas privadas cambiaron durante la preparación.")
    result = {"schema_version": 1, "kind": "maintenance_private_acceptance", "status": "running",
              "collection_sha256": _file_hash(directory / "collection.json"),
              "input_files_sha256": pair["input_files_sha256"],
              "source_files": source_files, "acceptance_files": acceptance_files,
              "acceptance_files_sha256": acceptance_hashes,
              "scope": "post-trajectory private tests; no feedback or repair from acceptance",
              "branches": {}}
    (directory / "acceptance").mkdir()
    for role in ("weak", "strong"):
        branch = pair["branches"][role]
        row = {"success": None, "acceptance_success": None, "visible_success": branch["success"],
               "status": "infrastructure_error", "report_sha256": branch["report_sha256"]}
        result["branches"][role] = row
        try:
            if branch["report_path"].replace("\\", "/") != role + "/report.json":
                raise ValueError("Ruta de informe no compatible.")
            if _file_hash(directory / role / "report.json") != branch["report_sha256"]:
                raise ValueError("El informe público cambió antes de la aceptación.")
            original = Workspace.reopen(directory / role)
            if original.source_files != source_files or original.test_files != pair["test_files"]:
                raise ValueError("El alcance del workspace no corresponde al par.")
            source_payloads = {name: _read(_inside(original.worktree, name)) for name in source_files}
            source_hashes = {name: _digest(data) for name, data in source_payloads.items()}
            if branch["success"] and original._verified_hashes != _hashes(original.worktree, public_names):
                raise ValueError("El código público aprobado cambió antes de la aceptación.")
            row["source_files_sha256"] = source_hashes
            base = directory / "acceptance" / role
            inputs = base / "inputs"
            for name, data in {**source_payloads, **private_payloads}.items():
                _write(_inside(inputs, name, must_exist=False), data)
            workspace = Workspace(inputs, base / "execution", source_files, acceptance_files)
            workspace.prepare()
            row.update(snapshot_sha256=_file_hash(workspace.run_dir / "workspace_snapshot.json"),
                       verification_sha256=None, tests_run=None)
            try:
                verification = workspace.verify(Path(python_executable or sys.executable), timeout_seconds)
            except (OSError, ValueError) as exc:
                # The public failure already determines the joint target. Keep
                # the private verdict unknown rather than censoring that failure.
                row.update(success=False if branch["success"] is False else None,
                           error_type=type(exc).__name__)
                if (_hashes(original.worktree, source_files) != source_hashes
                        or _hashes(workspace.worktree, source_files + acceptance_files)
                        != {**source_hashes, **acceptance_hashes}):
                    row.update(status="workspace_changed", success=None)
                _json(directory / "acceptance.json", result)
                continue
            row.update(verification_sha256=_file_hash(workspace.run_dir / "verification.json"),
                       status=verification["status"], tests_run=verification["tests_run"],
                       success=False if branch["success"] is False else None)
            if verification["status"] in {"passed", "test_failed"}:
                row["acceptance_success"] = verification["passed"]
                row["success"] = branch["success"] and verification["passed"]
            # Bind even failed outcomes to the exact copied source and private tests.
            if (_hashes(original.worktree, source_files) != source_hashes
                    or _hashes(workspace.worktree, source_files + acceptance_files)
                    != {**source_hashes, **acceptance_hashes}):
                row.update(status="workspace_changed", success=None, acceptance_success=None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="infrastructure_error", success=None, acceptance_success=None,
                       error_type=type(exc).__name__)
        _json(directory / "acceptance.json", result)
    result["private_verdicts_complete"] = all(row["acceptance_success"] is not None for row in result["branches"].values())
    result["status"] = "completed" if all(row["success"] is not None for row in result["branches"].values()) else "blocked"
    _json(directory / "acceptance.json", result)
    return result


def load_acceptance(task, pair_directory, pair, expected_sha256):
    """Verify frozen evidence on replay, without re-running hidden tests."""
    directory = Path(pair_directory)
    location = directory / "acceptance.json"
    if not expected_sha256 or not location.is_file() or _file_hash(location) != expected_sha256:
        raise ValueError("La aceptación privada falta o cambió después de registrarse.")
    result = _load(location)
    expected = {item["path"]: item["sha256"] for item in task["acceptance_inventory"]["files"]}
    if (result.get("schema_version") != 1 or result.get("kind") != "maintenance_private_acceptance"
            or result.get("status") != "completed" or set(result.get("branches", {})) != {"weak", "strong"}
            or result.get("collection_sha256") != _file_hash(directory / "collection.json")
            or result.get("input_files_sha256") != pair["input_files_sha256"]
            or result.get("source_files") != task["config"]["source_files"]
            or result.get("acceptance_files") != task["acceptance_files"]
            or result.get("acceptance_files_sha256") != expected):
        raise ValueError("La aceptación privada no tiene un veredicto completo ligado a los inputs congelados.")
    sources, private = result["source_files"], result["acceptance_files"]
    if result.get("private_verdicts_complete") is not all(
            row.get("acceptance_success") is not None for row in result["branches"].values()):
        raise ValueError("Cambió la declaración de veredictos privados completos.")
    for role in ("weak", "strong"):
        row, public = result["branches"][role], pair["branches"][role]
        private_outcome = row.get("acceptance_success")
        joint = False if public["success"] is False else private_outcome
        if (type(row.get("success")) is not bool or (private_outcome is not None and type(private_outcome) is not bool)
                or row.get("visible_success") is not public["success"]
                or row.get("success") is not joint
                or row.get("report_sha256") != public["report_sha256"]):
            raise ValueError("Los resultados privados y públicos no corresponden al mismo par.")
        source_hashes = row["source_files_sha256"]
        combined = {**source_hashes, **expected}
        base = directory / "acceptance" / role
        execution = base / "execution"
        if (_hashes(directory / role / "worktree", sources) != source_hashes
                or _hashes(base / "inputs", sources + private) != combined
                or _hashes(execution / "worktree", sources + private) != combined
                or _hashes(execution / "originals", sources + private) != combined
                or _file_hash(execution / "workspace_snapshot.json") != row["snapshot_sha256"]):
            raise ValueError("Cambió el código o la evidencia congelada de aceptación privada.")
        workspace = Workspace.reopen(execution)
        if workspace.source_files != sources or workspace.test_files != private:
            raise ValueError("Cambió el alcance de la verificación privada.")
        if row.get("verification_sha256") is None:
            if (public["success"] is not False or private_outcome is not None
                    or row["status"] != "infrastructure_error" or not row.get("error_type")
                    or row["tests_run"] is not None or (execution / "verification.json").exists()):
                raise ValueError("Falta una verificación privada necesaria para determinar el resultado.")
            continue
        if _file_hash(execution / "verification.json") != row["verification_sha256"]:
            raise ValueError("Cambió el informe congelado de aceptación privada.")
        verification = _load(execution / "verification.json")
        observed = verification["passed"] if verification["status"] in {"passed", "test_failed"} else None
        if (verification["status"] != row["status"]
                or observed is not private_outcome
                or verification["tests_run"] != row["tests_run"]):
            raise ValueError("La verificación privada no corresponde al resultado registrado.")
    return result
