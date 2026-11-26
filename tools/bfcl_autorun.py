"""Bounded continuation supervisor for an existing, closed BFCL Free study.

Prepare is the default. Execution and background launch are explicit. Only the
original ordered pending pairs are eligible. The 72-hour maximum is checked
between finite sessions and during waits; an active bounded session is allowed
to close instead of interrupting a provider call. Credentials remain in memory.
"""
from __future__ import annotations

import argparse
import _thread
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
HELPER_FILES = ("bfcl_autorun.py", "continue_bfcl.py", "report_bfcl.py", "report_bfcl_study.py", "run_bfcl.py")


def _helpers():
    return {name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() for name in HELPER_FILES}


# Capture before helper imports, then verify after imports and at every gate.
# Long-lived workers must never describe old imported code with new disk hashes.
LOADED_HELPER_HASHES = _helpers()
LOADED_SOURCE_HASHES = {path.relative_to(ROOT / "src").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for package in ("routerbench", "relay_router") for path in (ROOT / "src" / package).rglob("*.py")}
from routerbench.agent_evaluation import _seal, _single_writer, _sources, _unseal, _write, object_hash
from report_bfcl import Snapshot
from continue_bfcl import _load_previous, prepare_continuation, execute_continuation
import report_bfcl_study

KIND = "bfcl_bounded_autorun_v1"
DAY = 24 * 60 * 60
TARIFFS = ROOT / "configs/groq_paid_tariffs_20261112.json"
if LOADED_HELPER_HASHES != _helpers() or LOADED_SOURCE_HASHES != _sources():
    raise RuntimeError("Sources changed during supervisor imports; restart with consistent files.")


def _iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _loaded_sources_guard():
    if LOADED_HELPER_HASHES != _helpers() or LOADED_SOURCE_HASHES != _sources():
        raise ValueError("source_change")


def _normalize_previous(directory):
    directory = Path(directory).resolve()
    if (directory / "manifest.json").is_file() and (directory.parent / "continuation.json").is_file():
        directory = directory.parent
    return directory


def _state(directory):
    snapshot = Snapshot()
    state = _load_previous(directory, snapshot)
    snapshot.verify()
    return state, snapshot


def _free(config):
    spending = config.get("spending", {})
    models = config.get("model_catalog", [])
    limits = config.get("session_limits", {})
    if (config.get("execution_mode") != "external" or config.get("evaluation_split") != "development"
            or config.get("independent_validation") is not False
            or spending.get("account_is_free_tier") is not True or spending.get("max_provider_spend_usd") != 0
            or not models or any(not name.startswith("groq/") for name in models)
            or set(config.get("models_metadata", {})) != set(models)
            or any(row.get("api_key_env") != "GROQ_API_KEY" for row in config["models_metadata"].values())
            or config.get("session_stop_on_infrastructure") is not True
            or type(limits.get("max_tokens_per_model")) is not int or not 1 <= limits["max_tokens_per_model"] <= 200000
            or type(limits.get("max_provider_calls")) is not int or not 1 <= limits["max_provider_calls"] <= 250):
        raise ValueError("Autorun requiere el estudio Groq Free de desarrollo ya autorizado.")


def prepare_autorun(previous, output, *, max_hours=72, execution_authorized=False):
    if type(max_hours) not in (int, float) or not math.isfinite(max_hours) or not 0 < max_hours <= 72:
        raise ValueError("max-hours debe ser mayor que cero y no superar 72.")
    if type(execution_authorized) is not bool:
        raise ValueError("Execution authorization must be boolean.")
    _loaded_sources_guard()
    previous, output = _normalize_previous(previous), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Autorun necesita una carpeta NUEVA.")
    state, snapshot = _state(previous)
    _free(state["configuration"])
    if state["manifest"]["source_files_sha256"] != _sources():
        raise ValueError("Las fuentes cambiaron respecto al estudio cerrado.")
    tariff_raw = TARIFFS.read_bytes()
    tariffs = json.loads(tariff_raw)
    if tariffs.get("as_of") != "2026-11-12" or not set(state["configuration"]["model_catalog"]) <= set(tariffs.get("models", {})):
        raise ValueError("Las tarifas congeladas no cubren el catalogo del estudio.")
    plan = {"schema_version": 1, "kind": KIND, "previous": str(previous),
        "parent_file_sha256": snapshot.hashes, "study_parent": str(Path(state["run_directories"][0]).resolve()),
        "ordered_pending_pairs": state["pending"], "source_files_sha256": _sources(),
        "helper_sha256": _helpers(), "max_hours": max_hours,
        "tariffs": {"source_path": str(TARIFFS.resolve()), "snapshot_path": str(output / "tariffs.json"),
            "sha256": hashlib.sha256(tariff_raw).hexdigest(), "as_of": "2026-11-12",
            "scope": "fixed paid token scenario, not Free billing or full ROI"},
        "max_session_tokens": state["configuration"]["session_limits"]["max_tokens_per_model"],
        "execution_authorized": execution_authorized, "provider_calls": 0,
        "wait_policy": {"rate_limit_or_unknown_429_seconds": DAY, "local_session_cap_seconds": DAY,
            "sleep_chunk_max_seconds": 30, "respect_longer_retry_after": True,
            "other_infrastructure": "needs_attention; no automatic retry"},
        "deadline_scope": "no new finite session after deadline; an active bounded session may finish after it",
        "cancellation": "STOP checked during waits and active sessions; interrupt_main may wait for HTTP timeout; uncertain calls are never reissued",
        "independent_validation": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "tariffs.json").write_bytes(tariff_raw)
    _write(output / "autorun.json", _seal(plan))
    _write(output / "status.json", {"state": "prepared", "autorun_sha256": object_hash(plan),
        "previous": str(previous), "pending_pairs": len(state["pending"]), "sessions_closed": 0,
        "generation_calls": 0, "report_paths": [], "wake_time_utc": None})
    return plan


def _guard(plan):
    _loaded_sources_guard()
    if plan["source_files_sha256"] != _sources() or plan["helper_sha256"] != _helpers():
        raise ValueError("source_change")
    for path, digest in plan["parent_file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("parent_artifact_change")
    if hashlib.sha256(Path(plan["tariffs"]["snapshot_path"]).read_bytes()).hexdigest() != plan["tariffs"]["sha256"]:
        raise ValueError("tariff_snapshot_change")


class AutorunCancelled(Exception):
    pass


def _check_stop(output):
    if (Path(output) / "STOP").exists():
        raise AutorunCancelled()


@contextmanager
def watch_stop(output, *, interrupt=None, poll_seconds=1):
    """Only active execution needs an interrupt; ordinary waits check directly."""
    interrupt = _thread.interrupt_main if interrupt is None else interrupt
    finished = threading.Event()
    def watch():
        while not finished.is_set():
            if (Path(output) / "STOP").exists():
                interrupt()
                return
            finished.wait(min(30, poll_seconds))
    thread = threading.Thread(target=watch, name="bfcl-stop-watcher", daemon=True)
    try:
        thread.start()
        yield
    finally:
        finished.set()
        thread.join(timeout=2)


@contextmanager
def _study_mutex(study_parent, output):
    lock = _single_writer(study_parent)
    try:
        lock.__enter__()
    except RuntimeError:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        status.update(state="needs_attention", stop_reason={"reason": "study_locked"})
        _write(output / "status.json", status)
        raise
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _claim_study(study_parent, output, plan):
    """Durable exclusive ownership survives exit, cancellation, and crashes.

    Called only under both OS locks. This release intentionally requires manual
    audit before any later supervisor may reuse the study; it never resets a
    claim from an immutable old parent or guesses whether a call was sent.
    """
    path = study_parent / ".bfcl-autorun-claim.json"
    if path.exists():
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        status.update(state="needs_attention", stop_reason={"reason": "study_already_claimed", "claim_path": str(path)})
        _write(output / "status.json", status)
        raise RuntimeError("El estudio ya tiene un supervisor registrado; no se repiten pendientes desde un padre antiguo.")
    _write(path, _seal({"schema_version": 1, "study_parent": str(study_parent), "output": str(output),
        "autorun_sha256": object_hash(plan), "initial_cursor": plan["previous"],
        "status_path": str(output / "status.json"), "claimed_at_utc": _iso(time.time()),
        "scope": "exclusive study claim; never automatically cleared, resumed, or reassigned"}))


def stop_action(state):
    """No message parsing: only persisted structural reason/status fields."""
    reason = state["summary"].get("session_stop_reason")
    if not state["pending"]:
        return "complete", 0, None
    if not isinstance(reason, dict):
        return "needs_attention", 0, {"reason": "missing_closed_session_stop_reason"}
    category, status = reason.get("category"), reason.get("status_code")
    if category == "rate_limit" or status == 429:
        delay = DAY
        last_run = Path(state["run_directories"][-1])
        summary = _unseal(last_run / "summary.json")
        for episode in summary.get("episodes", [])[-1:]:
            for event in episode.get("events", [])[-1:]:
                retry = (event.get("backend_failure") or {}).get("retry_after_seconds")
                if type(retry) in (int, float) and math.isfinite(retry) and retry >= 0:
                    delay = max(delay, retry)
        return "wait", delay, {"reason": "provider_quota_wait", "minimum_seconds": delay}
    if reason.get("reason") in {"session_token_limit", "session_token_reservation_limit", "session_provider_call_limit"}:
        return "wait", DAY, {"reason": "local_session_cap_wait", "minimum_seconds": DAY}
    return "needs_attention", 0, {key: reason[key] for key in ("reason", "category", "status_code", "error_type") if key in reason}


def write_report(state, output, *, tariffs_path):
    directories = state["run_directories"]
    report_bfcl_study.main(["--parent", directories[0], "--runs", *directories[1:],
        "--tariffs", str(tariffs_path), "--output", str(output)])
    return str(output / "report.md")


def run_worker(output, *, clock=time.time, sleep=time.sleep, state_reader=_state,
               prepare=prepare_continuation, execute=execute_continuation, report=write_report):
    output = Path(output).resolve()
    plan = _unseal(output / "autorun.json")
    if plan.get("kind") != KIND or plan.get("execution_authorized") is not True:
        raise ValueError("Worker requires an explicitly authorized execution plan.")
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Worker requires the authorized key only in its environment.")
    if json.loads((output / "status.json").read_text(encoding="utf-8"))["state"] != "prepared":
        raise ValueError("No se reanuda un worker previo ni una llamada de estado incierto.")
    state, _ = state_reader(plan["previous"])
    study_parent = Path(state["run_directories"][0]).resolve()
    if str(study_parent) != plan["study_parent"]:
        raise ValueError("study_ancestor_changed")
    with _single_writer(output), _study_mutex(study_parent, output):
        existing = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if existing["state"] != "prepared":
            raise ValueError("No se reanuda un worker previo ni una llamada de estado incierto.")
        _guard(plan)
        _claim_study(study_parent, output, plan)
        started = clock()
        deadline = started + plan["max_hours"] * 3600
        cursor = plan["previous"]
        status = {**existing, "started_at_utc": _iso(started), "deadline_utc": _iso(deadline),
            "deadline_scope": plan["deadline_scope"], "worker_pid": os.getpid()}
        def update(name, **fields):
            status.update(state=name, updated_at_utc=_iso(clock()), **fields)
            _write(output / "status.json", status)
            print(json.dumps({"state": name, "sessions_closed": status["sessions_closed"],
                "pending_pairs": status["pending_pairs"], "wake_time_utc": status.get("wake_time_utc")}), flush=True)
        try:
            _guard(plan)
            _check_stop(output)
            paths = [report(state, output / "reports" / "session-000", tariffs_path=plan["tariffs"]["snapshot_path"])]
            update("audited", report_paths=paths)
            while True:
                _guard(plan)
                _check_stop(output)
                pending = state["pending"]
                original = plan["ordered_pending_pairs"]
                if pending and pending != original[len(original)-len(pending):]:
                    raise ValueError("pending_scope_changed")
                action, delay, reason = stop_action(state)
                if action == "complete":
                    update("complete", pending_pairs=0, wake_time_utc=None)
                    break
                if action == "needs_attention":
                    update("needs_attention", stop_reason=reason, wake_time_utc=None)
                    break
                wake = clock() + delay
                update("waiting", wake_time_utc=_iso(wake), stop_reason=reason)
                while clock() < wake and clock() < deadline:
                    _check_stop(output)
                    remaining = min(30, wake-clock(), deadline-clock())
                    if remaining > 0:
                        sleep(remaining)
                _check_stop(output)
                if clock() >= deadline:
                    update("deadline_reached", wake_time_utc=_iso(wake))
                    break
                _guard(plan)
                _check_stop(output)
                session_dir = output / f"session-{status['sessions_closed']+1:03d}"
                update("preparing_session", wake_time_utc=None)
                prepared = prepare(cursor, session_dir, max_session_tokens=plan["max_session_tokens"])
                if prepared["plan"]["ordered_pending_pairs"] != pending:
                    raise ValueError("prepared_scope_changed")
                if clock() >= deadline:
                    update("deadline_reached", active_session=None, wake_time_utc=None)
                    break
                update("running_session", active_session=str(session_dir))
                _check_stop(output)
                with watch_stop(output):
                    summary = execute(prepared)
                cursor = str(session_dir)
                state, _ = state_reader(cursor)
                paths.append(report(state, output / "reports" / f"session-{status['sessions_closed']+1:03d}",
                    tariffs_path=plan["tariffs"]["snapshot_path"]))
                update("session_closed", previous=cursor, active_session=None,
                    sessions_closed=status["sessions_closed"]+1,
                    generation_calls=status["generation_calls"]+summary["generation_calls"],
                    pending_pairs=len(state["pending"]), report_paths=list(paths))
        except (KeyboardInterrupt, AutorunCancelled):
            cancelled = (output / "STOP").exists()
            update("cancelled" if cancelled else "interrupted", stop_reason={
                "reason": "stop_requested" if cancelled else "interrupted_not_retried",
                "active_call_may_be_pending": bool(status.get("active_session")), "automatic_reissue": False}, wake_time_utc=None)
        except Exception as error:
            known = {"source_change", "parent_artifact_change", "pending_scope_changed", "prepared_scope_changed", "tariff_snapshot_change"}
            code = str(error) if str(error) in known else "local_execution_or_audit_failure"
            update("needs_attention", stop_reason={"reason": code, "error_type": type(error).__name__},
                wake_time_utc=None)
        return dict(status)


@contextmanager
def temporary_key():
    previous = os.environ.get("GROQ_API_KEY")
    try:
        key = previous
        if not key:
            if not sys.stdin.isatty():
                raise RuntimeError("Use terminal oculta o GROQ_API_KEY autorizado en memoria.")
            key = getpass.getpass("Groq API key (oculta; solo en memoria): ")
        if not key or any(character.isspace() for character in key):
            raise ValueError("La clave debe ser no vacia y sin espacios.")
        os.environ["GROQ_API_KEY"] = key
        del key
        yield
    finally:
        if previous is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous


def launch_background(output):
    output = Path(output).resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)]
    with (output / "worker.log").open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=dict(os.environ), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = {"pid": process.pid, "status_path": str(output / "status.json"), "log_path": str(output / "worker.log")}
    _write(output / "launch.json", result)
    return result


def request_stop(output):
    output = Path(output).resolve()
    plan = _unseal(output / "autorun.json")
    if plan.get("kind") != KIND:
        raise ValueError("La carpeta no corresponde a un supervisor BFCL.")
    _write(output / "STOP", {"schema_version": 1, "autorun_sha256": object_hash(plan),
        "requested_at_utc": _iso(time.time())})
    return {"state": "stop_requested", "stop_path": str(output / "STOP"),
        "scope": "cooperative cancellation; an active HTTP call may wait for its timeout; pending calls are never reissued"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=72)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launch-background", action="store_true")
    parser.add_argument("--stop", action="store_true", help="Request cancellation of the supervisor in --output; no PID kill.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stop:
        if args.execute or args.launch_background or args.worker or args.previous is not None:
            raise ValueError("Use --stop solo con --output de un supervisor existente.")
        result = request_stop(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    if args.worker:
        try:
            return run_worker(args.output)
        finally:
            os.environ.pop("GROQ_API_KEY", None)
    if args.previous is None or (args.launch_background and not args.execute):
        raise ValueError("Declare --previous; --launch-background requiere --execute explicito.")
    plan = prepare_autorun(args.previous, args.output, max_hours=args.max_hours, execution_authorized=args.execute)
    if not args.execute:
        result = {"mode": "prepare", "pending_pairs": len(plan["ordered_pending_pairs"]), "provider_calls": 0,
            "status_path": str(args.output.resolve() / "status.json")}
    else:
        with temporary_key():
            result = launch_background(args.output) if args.launch_background else run_worker(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
