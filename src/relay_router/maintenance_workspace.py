"""Explicit Python-file work copies for the maintenance example.

This isolates edits, not execution. unittest runs trusted project code with the
user's OS permissions. Provider credentials are not inherited by test children.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import time
import uuid


MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 512 * 1024
MAX_FILES = 64
MAX_TEST_OUTPUT_BYTES = 1024 * 1024
_DENIED = {"data", "outputs", "artifacts", "build", "dist", "node_modules",
           "venv", "__pycache__", "secrets", "credentials"}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_hash(value: dict) -> str:
    return _digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))


def _linked(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Usa rutas relativas explícitas con separadores /.")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or any(part in {".", ".."} for part in path.parts)
            or any(part.startswith(".") or part.casefold() in _DENIED for part in path.parts)
            or path.suffix != ".py" or any(part.endswith((" ", ".")) for part in path.parts)):
        raise ValueError("Solo se permiten archivos Python de código y tests dentro del proyecto.")
    return value


def _inside(base: Path, name: str, *, must_exist: bool = True) -> Path:
    path = base / _relative_name(name)
    current = base
    if _linked(current):
        raise ValueError("No se permiten enlaces o junctions en el workspace.")
    for part in PurePosixPath(name).parts:
        current = current / part
        if _linked(current):
            raise ValueError("No se permiten enlaces o junctions en los archivos autorizados.")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("La ruta sale del workspace.")
    if must_exist:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
            raise ValueError("Se requiere un archivo regular sin enlaces adicionales.")
    return path


def _read(path: Path) -> bytes:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Un archivo supera el límite de 128 KiB.")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES or b"\0" in data:
        raise ValueError("Archivo demasiado grande o binario.")
    data.decode("utf-8")
    return data


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".relay-" + uuid.uuid4().hex)
    try:
        temporary.write_bytes(data)
        if path.is_file():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path: Path, value: dict) -> None:
    _write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


# A fixed runner loads only the authorized test files. It records an explicit
# count instead of equating a zero exit status or model prose with success.
_TEST_RUNNER = r'''
import importlib.util, json, pathlib, sys, unittest
root = pathlib.Path(sys.argv[1])
names = json.loads(sys.argv[2])
destination = pathlib.Path(sys.argv[3])
project_roots = set(json.loads(sys.argv[4]))
sys.path[:0] = [str(root), str(root / "src")] + sorted({str((root / p).parent) for p in names})
loader = unittest.TestLoader()
suite = unittest.TestSuite()
try:
    for index, name in enumerate(names):
        spec = importlib.util.spec_from_file_location("relay_verification_" + str(index), root / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "expected_failures": len(result.expectedFailures),
               "unexpected_successes": len(result.unexpectedSuccesses),
               "passed": result.wasSuccessful() and result.testsRun > 0
                         and result.testsRun > len(result.skipped) + len(result.expectedFailures)}
    invalid_imports = []
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in project_roots:
            continue
        origin = getattr(module, '__file__', None)
        search_paths = list(getattr(module, '__path__', ()))
        locations = ([origin] if origin else []) + search_paths
        if not locations or any(not pathlib.Path(p).resolve().is_relative_to(root) for p in locations):
            invalid_imports.append(name)
    if invalid_imports:
        payload.update(passed=False, import_provenance_errors=sorted(invalid_imports))
        print('Importaciones del proyecto fuera de la copia: ' + ', '.join(sorted(invalid_imports)), file=sys.stderr)
except BaseException as error:
    import traceback
    traceback.print_exc()
    payload = {"tests_run": 0, "passed": False, "runner_error": type(error).__name__}
destination.write_text(json.dumps(payload), encoding="utf-8")
sys.exit(0 if payload["passed"] else 1)
'''


def _test_environment() -> dict[str, str]:
    allowed = {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "PATH", "TEMP", "TMP", "LANG", "LC_ALL"}
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _stop_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        taskkill = Path(os.environ["SYSTEMROOT"]) / "System32" / "taskkill.exe"
        try:
            subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=False, shell=False)
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


@dataclass(frozen=True)
class _File:
    name: str
    editable: bool
    original_sha256: str
    size: int


class Workspace:
    """Stage authorized files, run fixed tests, and apply conflict-free edits."""

    def __init__(self, root: Path, run_dir: Path, source_files: list[str], test_files: list[str]):
        root, run_dir = Path(root), Path(run_dir)
        if any(_linked(p) for p in [root, *root.parents, run_dir, *run_dir.parents]):
            raise ValueError("El proyecto y la carpeta de ejecución deben ser directorios reales.")
        self.root, self.run_dir = root.resolve(), run_dir.resolve()
        if not self.root.is_dir():
            raise ValueError("No existe el directorio del proyecto.")
        if (type(source_files) is not list or not source_files or type(test_files) is not list
                or not test_files or len(source_files) + len(test_files) > MAX_FILES):
            raise ValueError("Declara fuentes y tests no vacíos, máximo 64 archivos en total.")
        names = [_relative_name(name) for name in source_files + test_files]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Las fuentes y los tests deben ser distintos y no repetirse.")
        self.source_files, self.test_files = list(source_files), list(test_files)
        self.worktree = self.run_dir / "worktree"
        self.originals = self.run_dir / "originals"
        self._files: list[_File] = []
        self._verified_hashes: dict[str, str] | None = None
        self._snapshot_sha256: str | None = None

    def _validated_payloads(self) -> dict[str, bytes]:
        selected = set(self.source_files + self.test_files)
        for name in selected:
            for directory in PurePosixPath(name).parents:
                if str(directory) == ".":
                    continue
                initializer = str(directory / "__init__.py")
                if (self.root / initializer).exists() and initializer not in selected:
                    raise ValueError("Incluye explícitamente el inicializador del paquete: " + initializer)
        payloads = {name: _read(_inside(self.root, name)) for name in self.source_files + self.test_files}
        if sum(map(len, payloads.values())) > MAX_TOTAL_BYTES:
            raise ValueError("Los archivos autorizados superan 512 KiB.")
        return payloads

    def _project_import_roots(self) -> list[str]:
        roots = set()
        for name in self.source_files:
            parts = list(PurePosixPath(name).parts)
            if len(parts) > 1 and parts[0] == "src":
                parts = parts[1:]
            root = parts[0] if len(parts) > 1 else PurePosixPath(parts[0]).stem
            if root != "__init__":
                roots.add(root)
        return sorted(roots)

    def validate_inputs(self) -> dict:
        """Read-only preflight of the complete explicit file list."""
        payloads = self._validated_payloads()
        return {"files": [{"path": name, "editable": name in self.source_files,
                           "bytes": len(data), "sha256": _digest(data)} for name, data in payloads.items()],
                "total_bytes": sum(map(len, payloads.values()))}

    def prepare(self) -> None:
        if self._files or self.worktree.exists() or self.originals.exists():
            raise ValueError("La ejecución ya tiene un workspace; usa una carpeta nueva.")
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise ValueError("La carpeta de ejecución debe ser nueva o estar vacía.")
        # Read every approved input before creating a partial snapshot.
        payloads = self._validated_payloads()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if any(_linked(p) for p in [self.run_dir, *self.run_dir.parents]):
            raise ValueError("La carpeta de ejecución contiene enlaces o junctions.")
        for name, data in payloads.items():
            _write(_inside(self.worktree, name, must_exist=False), data)
            _write(_inside(self.originals, name, must_exist=False), data)
            self._files.append(_File(name, name in self.source_files, _digest(data), len(data)))
        snapshot = {
            "schema_version": 1, "root": str(self.root), "source_files": self.source_files,
            "test_files": self.test_files, "files": self.inventory(),
            "scope": "isolated file copies; not an execution security sandbox"}
        self._snapshot_sha256 = _object_hash(snapshot)
        _json(self.run_dir / "workspace_snapshot.json", {
            "payload": snapshot, "payload_sha256": self._snapshot_sha256})

    @classmethod
    def reopen(cls, run_dir: Path) -> Workspace:
        """Load an existing local run for reviewed application, without running code.

        Hashes detect alteration relative to the saved metadata, not authorship.
        Only open runs created locally by this trusted application.
        """
        run_dir = Path(run_dir)
        if any(_linked(p) for p in [run_dir, *run_dir.parents]):
            raise ValueError("La ejecución contiene enlaces o junctions.")
        envelope = json.loads((run_dir / "workspace_snapshot.json").read_text(encoding="utf-8"))
        if (type(envelope) is not dict or set(envelope) != {"payload", "payload_sha256"}
                or type(envelope["payload"]) is not dict
                or _object_hash(envelope["payload"]) != envelope["payload_sha256"]):
            raise ValueError("El hash del snapshot no coincide.")
        snapshot = envelope["payload"]
        if snapshot.get("schema_version") != 1:
            raise ValueError("Versión de snapshot no compatible.")
        workspace = cls(Path(snapshot["root"]), run_dir, snapshot["source_files"], snapshot["test_files"])
        rows = snapshot["files"]
        if (type(rows) is not list or [r.get("path") for r in rows] != workspace.source_files + workspace.test_files
                or any(set(r) != {"path", "editable", "sha256", "bytes"} for r in rows)):
            raise ValueError("Inventario de snapshot no compatible.")
        for row in rows:
            if row["editable"] is not (row["path"] in workspace.source_files):
                raise ValueError("Cambió el alcance de edición del snapshot.")
            payload = _read(_inside(workspace.originals, row["path"]))
            if _digest(payload) != row["sha256"] or len(payload) != row["bytes"]:
                raise ValueError("Las copias originales no coinciden con el snapshot.")
            workspace._files.append(_File(row["path"], row["editable"], row["sha256"], row["bytes"]))
        workspace._snapshot_sha256 = envelope["payload_sha256"]
        workspace._check_tests()
        verification_path = run_dir / "verification.json"
        if verification_path.exists():
            if _linked(verification_path):
                raise ValueError("La verificación no puede ser un enlace.")
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            digest = verification.pop("verification_sha256", None)
            if _object_hash(verification) != digest:
                raise ValueError("El hash de la verificación no coincide.")
            if verification.get("snapshot_sha256") != workspace._snapshot_sha256:
                raise ValueError("La verificación pertenece a otro snapshot.")
            if (verification.get("passed") is True and verification.get("status") == "passed"
                    and verification.get("verified_hashes") == workspace._hashes(workspace.worktree)):
                workspace._verified_hashes = verification["verified_hashes"]
        return workspace

    def _ready(self) -> None:
        if not self._files:
            raise ValueError("Primero prepara el workspace.")

    def inventory(self) -> list[dict]:
        return [{"path": f.name, "editable": f.editable, "sha256": f.original_sha256,
                 "bytes": f.size} for f in self._files]

    def read_file(self, path: str) -> str:
        self._ready()
        if path not in self.source_files + self.test_files:
            raise ValueError("El archivo no está autorizado.")
        return _read(_inside(self.worktree, path)).decode("utf-8")

    def _hashes(self, base: Path) -> dict[str, str]:
        return {f.name: _digest(_read(_inside(base, f.name))) for f in self._files}

    def _check_tests(self) -> None:
        for file in self._files:
            if not file.editable and _digest(_read(_inside(self.worktree, file.name))) != file.original_sha256:
                raise ValueError("Las pruebas originales fueron modificadas en la copia.")

    @staticmethod
    def _replace_exact_eol(text: str, old: str, new: str) -> str:
        """Permit LF/CRLF equivalence only; preserve all other characters.

        Do not guess a newline convention for mixed files or mixed fragments.
        A newline-free exact replacement remains safe in a mixed-style file.
        Count overlapping matches as ambiguous too.
        """
        def style(value):
            remaining = value.replace("\r\n", "")
            if "\r" in remaining:
                return "bare_cr"
            if "\r\n" in value and "\n" in remaining:
                return "mixed"
            return "crlf" if "\r\n" in value else "lf"

        kind = style(text)
        if kind == "bare_cr" or any(style(value) in {"mixed", "bare_cr"} for value in (old, new)):
            raise ValueError("No se admiten fragmentos con saltos de línea mixtos o CR aislado.")
        if kind == "mixed":
            if any("\n" in value or "\r" in value for value in (old, new)):
                raise ValueError("Un archivo con saltos mixtos solo admite cambios exactos sin saltos de línea.")
        else:
            newline = "\r\n" if kind == "crlf" else "\n"
            old = old.replace("\r\n", "\n").replace("\n", newline)
            new = new.replace("\r\n", "\n").replace("\n", newline)
        found = text.find(old) if old else -1
        if found < 0 or text.find(old, found + 1) >= 0:
            raise ValueError("old_text debe coincidir exactamente una sola vez.")
        return text[:found] + new + text[found + len(old):]

    def apply_edits(self, edits: list[dict]) -> None:
        self._ready()
        if type(edits) is not list or not edits or len(edits) > 32:
            raise ValueError("Se requiere una lista de 1 a 32 sustituciones exactas.")
        self._check_tests()
        replacements: dict[str, bytes] = {}
        originals: dict[str, bytes] = {}
        for edit in edits:
            if (type(edit) is not dict or set(edit) != {"path", "old_text", "new_text"}
                    or any(type(edit.get(key)) is not str for key in edit)):
                raise ValueError("Cada cambio requiere path, old_text y new_text de texto.")
            name, old, new = edit["path"], edit["old_text"], edit["new_text"]
            if name not in self.source_files:
                raise ValueError("Solo se pueden modificar fuentes autorizadas; los tests son inmutables.")
            original = originals.setdefault(name, _read(_inside(self.worktree, name)))
            text = replacements.get(name, original).decode("utf-8")
            data = self._replace_exact_eol(text, old, new).encode("utf-8")
            if len(data) > MAX_FILE_BYTES or b"\0" in data:
                raise ValueError("El cambio excede el tamaño permitido o introduce contenido binario.")
            replacements[name] = data
        total = sum(len(replacements.get(f.name, _read(_inside(self.worktree, f.name)))) for f in self._files)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("El cambio supera el límite total del workspace.")
        committed = []
        try:
            for name, data in replacements.items():
                _write(_inside(self.worktree, name), data)
                committed.append(name)
        except BaseException:
            for name in reversed(committed):
                _write(_inside(self.worktree, name), originals[name])
            raise
        self._verified_hashes = None

    def diff(self) -> str:
        self._ready()
        parts = []
        for name in self.source_files:
            before = _read(_inside(self.originals, name)).decode("utf-8")
            after = self.read_file(name)
            parts.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                              fromfile="a/" + name, tofile="b/" + name))
        return "".join(parts)

    def verify(self, python_executable: Path, timeout_seconds: float = 60) -> dict:
        self._ready()
        self._check_tests()
        executable = Path(python_executable).resolve()
        if not executable.is_file() or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600:
            raise ValueError("Declara un intérprete Python y un timeout entre 0 y 600 segundos.")
        before = self._hashes(self.worktree)
        self._verified_hashes = None
        token = uuid.uuid4().hex
        result_path = self.run_dir / ("test-result-" + token + ".json")
        stdout_path, stderr_path = (self.run_dir / ("test-" + token + suffix) for suffix in (".stdout", ".stderr"))
        command = [str(executable), "-I", "-B", "-c", _TEST_RUNNER, str(self.worktree),
                   json.dumps(self.test_files), str(result_path), json.dumps(self._project_import_roots())]
        started = time.perf_counter()
        timed_out = output_limited = False
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            process = subprocess.Popen(command, cwd=self.worktree, env=_test_environment(),
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err, shell=False, **kwargs)
            while process.poll() is None:
                timed_out = time.perf_counter() - started > timeout_seconds
                output_limited = stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_TEST_OUTPUT_BYTES
                if timed_out or output_limited:
                    _stop_process(process)
                    break
                time.sleep(.05)
        elapsed = time.perf_counter() - started
        output_limited = output_limited or stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_TEST_OUTPUT_BYTES
        result = {}
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                result = {}
        if type(result) is not dict:
            result = {}
        try:
            unchanged = before == self._hashes(self.worktree)
        except (OSError, ValueError):
            unchanged = False
        count = result.get("tests_run", 0)
        passed = (process.returncode == 0 and not timed_out and not output_limited and unchanged
                  and result.get("passed") is True and type(count) is int and count > 0)
        status = ("passed" if passed else "timeout" if timed_out else "output_limit" if output_limited
                  else "workspace_modified_by_tests" if not unchanged
                  else "import_provenance_error" if result.get("import_provenance_errors")
                  else "runner_error" if result.get("runner_error") or not result
                  else "no_tests" if count == 0
                  else "no_executed_tests" if count <= result.get("skipped", 0) + result.get("expected_failures", 0)
                  else "test_failed")
        def output(path):
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 32768))
                return handle.read(32768).decode("utf-8", errors="replace")
        report = {"status": status, "passed": passed, "tests_run": count,
                  "returncode": process.returncode, "timed_out": timed_out,
                  "output_limited": output_limited, "seconds": elapsed,
                  "stdout": output(stdout_path), "stderr": output(stderr_path),
                  "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                  "result_path": str(result_path),
                  "result": result, "verified_hashes": before if passed else None,
                  "snapshot_sha256": self._snapshot_sha256,
                  "execution_scope": "trusted project tests with user OS permissions; no security sandbox"}
        report["verification_sha256"] = _object_hash(report)
        _json(self.run_dir / "verification.json", report)
        if passed:
            self._verified_hashes = before
        return report

    def apply_to_original(self) -> dict:
        self._ready()
        self._check_tests()
        if self._verified_hashes is None or self._hashes(self.worktree) != self._verified_hashes:
            raise ValueError("El contenido actual necesita una verificación aprobada antes de aplicarse.")
        expected = {f.name: f.original_sha256 for f in self._files}
        if self._hashes(self.root) != expected or self._hashes(self.originals) != expected:
            raise ValueError("El proyecto o su copia original cambiaron desde la preparación; no se sobrescriben.")
        changes = {name: _read(_inside(self.worktree, name)) for name in self.source_files
                   if self._verified_hashes[name] != expected[name]}
        if not changes:
            raise ValueError("No hay cambios para aplicar.")
        committed = []
        try:
            for name, data in changes.items():
                # Check again directly before each replacement.
                target = _inside(self.root, name)
                if _digest(_read(target)) != expected[name]:
                    raise ValueError("Conflicto al aplicar: el archivo original cambió.")
                _write(target, data)
                committed.append(name)
        except BaseException:
            for name in reversed(committed):
                target = _inside(self.root, name)
                if _digest(_read(target)) == _digest(changes[name]):
                    _write(target, _read(_inside(self.originals, name)))
            raise
        result = {"status": "applied", "files": list(changes), "backup_directory": str(self.originals),
                  "sha256": {name: _digest(data) for name, data in changes.items()}}
        _json(self.run_dir / "application.json", result)
        return result
