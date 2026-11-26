"""Wait for one active BFCL run to close, then supervise its exact pending tail.

No provider request is made while awaiting the parent's validated summary.
The shared absolute deadline starts with this watcher. STOP also cancels an
active delegated supervisor; uncertain calls are never automatically reissued.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
LOADED_WATCH_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
import bfcl_autorun as supervisor
from routerbench.agent_evaluation import _seal, _single_writer, _unseal, _write, object_hash
from report_bfcl import Snapshot, _manifest

KIND = "bfcl_wait_then_supervise_v1"


def _helpers():
    return {**supervisor._helpers(), "watch_bfcl.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _read_parent(directory):
    snapshot = Snapshot()
    manifest = _manifest(directory, snapshot)
    snapshot.verify()
    return manifest, snapshot


def prepare_watch(parent, output, *, max_hours=72, execution_authorized=False):
    if type(max_hours) not in (int, float) or not math.isfinite(max_hours) or not 0 < max_hours <= 72:
        raise ValueError("max-hours debe ser positivo y no superar 72.")
    if type(execution_authorized) is not bool:
        raise ValueError("Execution authorization must be boolean.")
    supervisor._loaded_sources_guard()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != LOADED_WATCH_SHA256:
        raise ValueError("source_change")
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("El observador necesita una carpeta NUEVA.")
    manifest, snapshot = _read_parent(parent)
    supervisor._free(manifest["configuration"])
    if manifest["source_files_sha256"] != supervisor.LOADED_SOURCE_HASHES:
        raise ValueError("La corrida activa tiene otras fuentes; no se puede continuar automaticamente.")
    raw = supervisor.TARIFFS.read_bytes()
    tariffs = json.loads(raw)
    if tariffs.get("as_of") != "2026-11-12" or not set(manifest["configuration"]["model_catalog"]) <= set(tariffs["models"]):
        raise ValueError("Tarifas congeladas incompatibles con el catalogo.")
    plan = {"schema_version": 1, "kind": KIND, "parent_run": str(parent),
        "parent_freeze_sha256": manifest["freeze_sha256"], "parent_file_sha256": snapshot.hashes,
        "configuration_sha256": object_hash(manifest["configuration"]),
        "case_ids": [case["id"] for case in manifest["cases"]], "policies": manifest["policy_sequence"],
        "model_catalog": manifest["configuration"]["model_catalog"],
        "source_files_sha256": supervisor.LOADED_SOURCE_HASHES, "helper_sha256": _helpers(),
        "tariffs": {"source_path": str(supervisor.TARIFFS.resolve()), "snapshot_path": str(output / "tariffs.json"),
            "sha256": hashlib.sha256(raw).hexdigest(), "as_of": "2026-11-12"},
        "max_hours": max_hours, "execution_authorized": execution_authorized,
        "provider_calls": 0, "autorun_output": str(output / "autorun"),
        "scope": "Only pending pairs from this closed development cohort; no API before audited parent closure.",
        "deadline_scope": "absolute watcher deadline for waits and new sessions; a bounded active session may finish after it",
        "cancellation": "STOP forwarded to supervisor; active HTTP cancellation may wait for timeout; no uncertain call replay",
        "independent_validation": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "tariffs.json").write_bytes(raw)
    _write(output / "watch.json", _seal(plan))
    _write(output / "status.json", {"state": "prepared", "watch_sha256": object_hash(plan),
        "parent_run": str(parent), "autorun_status": str(output / "autorun/status.json"), "provider_calls": 0})
    return plan


def _guard(plan):
    supervisor._loaded_sources_guard()
    if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != LOADED_WATCH_SHA256
            or plan["helper_sha256"] != _helpers() or plan["source_files_sha256"] != supervisor._sources()):
        raise ValueError("source_change")
    for path, expected in plan["parent_file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("parent_contract_change")
    for name in ("source_path", "snapshot_path"):
        if hashlib.sha256(Path(plan["tariffs"][name]).read_bytes()).hexdigest() != plan["tariffs"]["sha256"]:
            raise ValueError("tariff_snapshot_change")


@contextmanager
def forward_stop(output, autorun_output, *, forward=supervisor.request_stop, poll_seconds=1):
    done = threading.Event()
    def watch():
        while not done.is_set():
            if (Path(output) / "STOP").exists():
                try:
                    forward(autorun_output)
                except Exception:
                    # Fail closed if the cancellation flag cannot be forwarded.
                    import _thread
                    _thread.interrupt_main()
                return
            done.wait(min(30, poll_seconds))
    thread = threading.Thread(target=watch, name="bfcl-watch-stop-forwarder", daemon=True)
    try:
        thread.start()
        yield
    finally:
        done.set()
        thread.join(timeout=2)


def _anchored_clock(anchor, clock):
    first = True
    def now():
        nonlocal first
        if first:
            first = False
            return anchor
        return clock()
    return now


def _parent_writer_released(directory):
    try:
        with _single_writer(Path(directory)):
            return True
    except RuntimeError:
        return False


def run_watch(output, *, clock=time.time, sleep=time.sleep, state_reader=supervisor._state,
              prepare=supervisor.prepare_autorun, worker=supervisor.run_worker):
    output = Path(output).resolve()
    plan = _unseal(output / "watch.json")
    if plan.get("kind") != KIND or plan.get("execution_authorized") is not True:
        raise ValueError("Watcher requires explicit execution authorization.")
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Watcher requires the authorized key only in its environment.")
    with _single_writer(output):
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if status["state"] != "prepared":
            raise ValueError("No se reanuda automaticamente un observador previo.")
        started = clock()
        deadline = started + plan["max_hours"] * 3600
        status.update(started_at_utc=supervisor._iso(started), deadline_utc=supervisor._iso(deadline),
            deadline_scope=plan["deadline_scope"], worker_pid=os.getpid())
        def update(state, **fields):
            status.update(state=state, updated_at_utc=supervisor._iso(clock()), **fields)
            _write(output / "status.json", status)
            print(json.dumps({"state": state, "autorun_status": status["autorun_status"]}), flush=True)
        try:
            update("awaiting_parent_close")
            while True:
                _guard(plan)
                supervisor._check_stop(output)
                if clock() >= deadline:
                    update("deadline_reached")
                    break
                if (Path(plan["parent_run"]) / "summary.json").is_file():
                    if not _parent_writer_released(plan["parent_run"]):
                        sleep(min(1, max(0, deadline-clock())))
                        continue
                    closed, _ = state_reader(plan["parent_run"])
                    if (closed["manifest"]["freeze_sha256"] != plan["parent_freeze_sha256"]
                            or object_hash(closed["configuration"]) != plan["configuration_sha256"]):
                        raise ValueError("closed_parent_contract_change")
                    _guard(plan)
                    supervisor._check_stop(output)
                    anchor = clock()
                    remaining_hours = (deadline-anchor)/3600
                    if remaining_hours <= 0:
                        update("deadline_reached")
                        break
                    autorun_output = Path(plan["autorun_output"])
                    update("preparing_supervisor", pending_pairs=len(closed["pending"]))
                    prepared = prepare(plan["parent_run"], autorun_output,
                        max_hours=remaining_hours, execution_authorized=True)
                    if (prepared["ordered_pending_pairs"] != closed["pending"]
                            or prepared["tariffs"]["sha256"] != plan["tariffs"]["sha256"]):
                        raise ValueError("delegated_contract_change")
                    supervisor._check_stop(output)
                    if clock() >= deadline:
                        update("deadline_reached")
                        break
                    update("delegated", provider_calls=None,
                        provider_call_scope="delegated supervisor accounting; watcher makes no provider calls")
                    with forward_stop(output, autorun_output):
                        result = worker(autorun_output, clock=_anchored_clock(anchor, clock), sleep=sleep)
                    state = "cancelled" if (output / "STOP").exists() else result["state"]
                    update(state, autorun_result_state=result["state"], report_paths=result.get("report_paths", []),
                        delegated_generation_calls=result.get("generation_calls"))
                    break
                remaining = min(30, deadline-clock())
                if remaining > 0:
                    sleep(remaining)
        except (KeyboardInterrupt, supervisor.AutorunCancelled):
            autorun_output = Path(plan["autorun_output"])
            if (autorun_output / "autorun.json").exists():
                supervisor.request_stop(autorun_output)
            update("cancelled" if (output / "STOP").exists() else "interrupted", automatic_reissue=False)
        except Exception as error:
            known = {"source_change", "parent_contract_change", "tariff_snapshot_change",
                     "closed_parent_contract_change", "delegated_contract_change"}
            update("needs_attention", stop_reason={"reason": str(error) if str(error) in known else "audit_or_supervisor_failure",
                "error_type": type(error).__name__})
        return dict(status)


def request_stop(output):
    output = Path(output).resolve()
    plan = _unseal(output / "watch.json")
    if plan.get("kind") != KIND:
        raise ValueError("La carpeta no corresponde a un observador BFCL.")
    _write(output / "STOP", {"watch_sha256": object_hash(plan), "requested_at_utc": supervisor._iso(time.time())})
    autorun = Path(plan["autorun_output"])
    if (autorun / "autorun.json").exists():
        supervisor.request_stop(autorun)
    return {"state": "stop_requested", "stop_path": str(output / "STOP")}


def launch_background(output):
    output = Path(output).resolve()
    with (output / "worker.log").open("a", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)],
            cwd=str(ROOT), env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = {"pid": process.pid, "status_path": str(output / "status.json"), "log_path": str(output / "worker.log")}
    _write(output / "launch.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=72)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launch-background", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stop:
        if args.execute or args.launch_background or args.worker or args.run is not None:
            raise ValueError("Use --stop solo con --output existente.")
        result = request_stop(args.output)
    elif args.worker:
        try:
            return run_watch(args.output)
        finally:
            os.environ.pop("GROQ_API_KEY", None)
    else:
        if args.run is None or (args.launch_background and not args.execute):
            raise ValueError("Declare --run; background requiere --execute explicito.")
        prepare_watch(args.run, args.output, max_hours=args.max_hours, execution_authorized=args.execute)
        if not args.execute:
            result = {"mode": "prepare", "provider_calls": 0, "status_path": str(args.output.resolve() / "status.json")}
        else:
            with supervisor.temporary_key():
                result = launch_background(args.output) if args.launch_background else run_watch(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
