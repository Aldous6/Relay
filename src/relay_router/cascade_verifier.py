"""Host-declared operational checks for a two-model cascade.

This module has no knowledge of held-out acceptance labels. A ``passed`` result
means that the declared commands returned zero and the pinned check files kept
their contents. Declared failure exit codes represent failed tests; other exit
codes are inconclusive. It is not a general correctness certificate. The host
must pin all files that control its checks (tests, runners and relevant
configuration), and choose a runner that distinguishes test failures from setup
errors and zero executed tests.

Commands execute trusted project code with the current user's permissions.
Neither the environment filtering nor the integrity checks are an OS sandbox.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import signal
import subprocess
import time
from typing import Sequence


_STATUSES = frozenset({"passed", "failed", "inconclusive"})
_ENVIRONMENT_KEYS = frozenset({
    "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "PATH", "TEMP", "TMP", "LANG", "LC_ALL",
})


def _identifier(value: str, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) is None:
        raise ValueError(f"{name} must be a short, non-sensitive identifier")
    return value


def _relative(value: str, *, directory: bool = False) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise ValueError("Use nonempty workspace-relative paths")
    path = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if path.is_absolute() or windows.drive or windows.root or ".." in path.parts:
        raise ValueError("Check paths must remain inside the workspace")
    if not directory and str(path) == ".":
        raise ValueError("A protected path must identify a file")
    return str(path)


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(root: Path, relative: str) -> Path:
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        # On Windows a junction need not report is_symlink(). Reparse points
        # include both junctions and symlinks and are rejected, even internally.
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Check paths cannot traverse symlinks or reparse points")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Check path escaped the workspace")
    return path


@dataclass(frozen=True)
class CheckCommand:
    """A fixed host command; never construct argv from model-generated text."""

    check_id: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: float = 60.0
    failure_returncodes: tuple[int, ...] = (1,)

    def __post_init__(self):
        _identifier(self.check_id, "check_id")
        if not isinstance(self.argv, (list, tuple)) or not self.argv:
            raise ValueError("argv must be a nonempty list or tuple, not a shell string")
        if any(type(arg) is not str or "\0" in arg for arg in self.argv) or not self.argv[0]:
            raise ValueError("argv must contain strings and a nonempty executable")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 600):
            raise ValueError("timeout_seconds must be finite and between 0 and 600")
        if (not isinstance(self.failure_returncodes, (list, tuple))
                or any(type(code) is not int or not 1 <= code <= 255 for code in self.failure_returncodes)
                or len(set(self.failure_returncodes)) != len(self.failure_returncodes)):
            raise ValueError("failure_returncodes must contain unique integer exit codes between 1 and 255")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "cwd", _relative(self.cwd, directory=True))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "failure_returncodes", tuple(self.failure_returncodes))

    @property
    def command_sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    command_sha256: str
    status: str
    reason: str
    returncode: int | None
    duration_seconds: float


@dataclass(frozen=True)
class IntegrityResult:
    relative_path: str
    expected_sha256: str
    observed_sha256: str | None
    status: str
    phase: str


@dataclass(frozen=True)
class VerificationResult:
    status: str
    verifier_id: str
    config_sha256: str
    duration_seconds: float
    checks: tuple[CheckResult, ...] = ()
    integrity: tuple[IntegrityResult, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if self.status not in _STATUSES:
            raise ValueError("Unknown verification status")
        if (type(self.duration_seconds) not in (int, float)
                or not math.isfinite(self.duration_seconds) or self.duration_seconds < 0):
            raise ValueError("Verification duration must be finite and nonnegative")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict:
        """JSON-compatible metadata, without stdout, stderr or command contents."""
        return {"schema_version": "relay-command-verification-v1", "status": self.status,
                "passed": self.passed, "verifier_id": self.verifier_id,
                "config_sha256": self.config_sha256, "duration_seconds": self.duration_seconds,
                "reason": self.reason, "checks": [asdict(item) for item in self.checks],
                "integrity": [asdict(item) for item in self.integrity]}


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Best-effort process-tree cleanup on timeout; not process containment."""
    if os.name == "nt":
        system_root = os.environ.get("SYSTEMROOT")
        if system_root:
            try:
                subprocess.run([str(Path(system_root) / "System32" / "taskkill.exe"),
                                "/PID", str(process.pid), "/T", "/F"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10, check=False,
                               shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


@dataclass(frozen=True)
class CommandVerifier:
    """Integrity-pinned host checks reusable across fresh branch workspaces.

    Call ``capture`` on the trusted starting workspace, before running an agent.
    Expected failures in that starting source are allowed: capture hashes files
    and validates configuration but does not require tests to pass. The returned
    object is callable and also exposes ``verify(workspace)``.
    """

    verifier_id: str
    checks: tuple[CheckCommand, ...]
    protected_files: tuple[tuple[str, str], ...]

    def __post_init__(self):
        _identifier(self.verifier_id, "verifier_id")
        if (not isinstance(self.checks, (list, tuple)) or not self.checks
                or any(type(check) is not CheckCommand for check in self.checks)):
            raise ValueError("Declare at least one CheckCommand")
        if len({check.check_id for check in self.checks}) != len(self.checks):
            raise ValueError("check_id values must be unique")
        if not isinstance(self.protected_files, (list, tuple)) or not self.protected_files:
            raise ValueError("Pin at least one host check file before running an agent")
        paths = []
        for entry in self.protected_files:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("Protected files must be (relative_path, sha256) pairs")
            relative, digest = entry
            paths.append(_relative(relative))
            if type(digest) is not str or re.fullmatch("[a-f0-9]{64}", digest) is None:
                raise ValueError("Protected files require lowercase SHA-256 digests")
        if len(set(paths)) != len(paths):
            raise ValueError("Protected paths must be unique")
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "protected_files", tuple(sorted(zip(paths, (x[1] for x in self.protected_files)))))

    @classmethod
    def capture(cls, reference_workspace: str | Path, *, checks: Sequence[CheckCommand],
                protected_files: Sequence[str], verifier_id: str = "project-tests-v1") -> CommandVerifier:
        root = Path(reference_workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("reference_workspace must be a directory")
        if isinstance(protected_files, str) or isinstance(checks, (str, bytes)):
            raise ValueError("Declare sequences of checks and protected file paths")
        paths = [_relative(relative) for relative in protected_files]
        hashes = []
        for relative in paths:
            path = _checked_path(root, relative)
            if not path.is_file():
                raise ValueError("Protected paths must identify existing regular files")
            hashes.append((relative, _file_digest(path)))
        verifier = cls(verifier_id, tuple(checks), tuple(hashes))
        for check in verifier.checks:
            if not _checked_path(root, check.cwd).is_dir():
                raise ValueError("Every check cwd must be an existing workspace directory")
        return verifier

    @property
    def config_sha256(self) -> str:
        return _digest({"schema_version": "relay-command-verifier-v1", "verifier_id": self.verifier_id,
                        "checks": [asdict(check) for check in self.checks],
                        "protected_files": dict(self.protected_files),
                        "environment_keys": sorted(_ENVIRONMENT_KEYS), "shell": False})

    def _integrity(self, root: Path, phase: str) -> tuple[IntegrityResult, ...]:
        results = []
        for relative, expected in self.protected_files:
            actual, status = None, "unreadable"
            try:
                path = _checked_path(root, relative)
                if path.is_file():
                    actual = _file_digest(path)
                    status = "unchanged" if actual == expected else "changed"
                else:
                    status = "missing"
            except FileNotFoundError:
                status = "missing"
            except ValueError:
                status = "unsafe_path"
            except OSError:
                pass
            results.append(IntegrityResult(relative, expected, actual, status, phase))
        return tuple(results)

    def verify(self, workspace: str | Path) -> VerificationResult:
        started = time.perf_counter()
        checks: list[CheckResult] = []
        integrity: list[IntegrityResult] = []

        def finish(status: str, reason: str):
            return VerificationResult(status, self.verifier_id, self.config_sha256,
                                      time.perf_counter() - started, tuple(checks), tuple(integrity), reason)

        try:
            root = Path(workspace).resolve(strict=True)
            if not root.is_dir():
                return finish("inconclusive", "workspace_unavailable")
        except (OSError, ValueError):
            return finish("inconclusive", "workspace_unavailable")
        environment = {key: value for key, value in os.environ.items() if key.upper() in _ENVIRONMENT_KEYS}
        for check in self.checks:
            before = self._integrity(root, "before:" + check.check_id)
            integrity.extend(before)
            if any(item.status != "unchanged" for item in before):
                return finish("inconclusive", "protected_file_integrity_failed")
            check_start = time.perf_counter()
            status, reason, returncode = "inconclusive", "launch_error", None
            try:
                cwd = _checked_path(root, check.cwd)
                if not cwd.is_dir():
                    raise OSError("Check working directory is unavailable")
                kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                          if os.name == "nt" else {"start_new_session": True})
                with subprocess.Popen(check.argv, cwd=cwd, env=environment,
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, shell=False, **kwargs) as process:
                    try:
                        returncode = process.wait(timeout=check.timeout_seconds)
                        if returncode == 0:
                            status, reason = "passed", "exit_zero"
                        elif returncode in check.failure_returncodes:
                            status, reason = "failed", "check_failed"
                        else:
                            status, reason = "inconclusive", "unexpected_exit_code"
                    except subprocess.TimeoutExpired:
                        status, reason = "inconclusive", "timeout"
                        _stop_process_tree(process)
                        returncode = process.returncode
            except (OSError, ValueError, subprocess.SubprocessError):
                # Exception messages can contain argv, paths or application data.
                # Only the stable category is retained in exported metadata.
                pass
            checks.append(CheckResult(check.check_id, check.command_sha256, status, reason, returncode,
                                      time.perf_counter() - check_start))
            after = self._integrity(root, "after:" + check.check_id)
            integrity.extend(after)
            if any(item.status != "unchanged" for item in after):
                return finish("inconclusive", "protected_file_integrity_failed")
        if any(check.status == "inconclusive" for check in checks):
            return finish("inconclusive", "check_unavailable")
        if any(check.status == "failed" for check in checks):
            return finish("failed", "check_failed")
        return finish("passed", "all_declared_checks_passed")

    def __call__(self, workspace: str | Path) -> VerificationResult:
        return self.verify(workspace)
