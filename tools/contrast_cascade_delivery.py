"""Offline regression and fresh execution of saved cascade outputs.

Does not regenerate LLM responses, change frozen evidence, or certify savings.
Each invocation writes a new destination; public/private checks use fresh copies.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/harness-cascade/v1/run"
SUITE = ROOT / "data/imported/harness-cascade-v1/fresh_suite.json"
EVALUATORS = SUITE.with_name("private_evaluators.json")
PATTERNS = (
    "test_cascade.py", "test_cascade_evaluation.py", "test_cascade_verifier.py",
    "test_cascade_unittest.py", "test_cascade_validation_runner.py",
    "test_prepare_cascade_validation.py", "test_report_cascade_validation.py",
    "test_openhands_adapter.py", "test_openhands_cascade.py", "test_openhands_io.py",
    "test_harness_live.py", "test_harness_packaging.py", "test_runtime_imports.py",
    "test_workflow_cli.py", "test_workflow_policy.py", "test_workflow_runtime.py",
    "test_prompt_selection.py", "test_adaptive_policy.py", "test_adaptive_runtime.py",
    "test_adaptive_training.py", "test_adaptive_scoring.py", "test_adaptive_pilot_runner.py",
    "test_adaptive_continuation.py", "test_adaptive_initialization_recovery.py",
    "test_prepare_adaptive_pilot.py", "test_report_adaptive_pilot.py",
    "test_report_adaptive_recovery.py", "test_routellm_adapter.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def frozen_integrity():
    frozen = read(RUN / "frozen_policy.json")
    contract = frozen["contract"]
    expected = {ROOT / relative: digest for relative, digest in contract["source_sha256"].items()}
    expected.update({SUITE: contract["suite_sha256"], EVALUATORS: contract["private_evaluators_sha256"]})
    if any(sha(path) != digest for path, digest in expected.items()):
        raise ValueError("Frozen source or evaluation input changed")
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest for path, digest in expected.items()}


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_ids.append(test.id())


def regressions(output, patterns=PATTERNS):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                 "LITELLM_LOCAL_MODEL_COST_MAP", "OPENHANDS_SUPPRESS_BANNER"):
        os.environ[name] = "1" if name != "LITELLM_LOCAL_MODEL_COST_MAP" else "True"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    original_connect = socket.socket.connect

    def local_connect(sock, address):
        # Windows asyncio builds its internal socketpair over loopback TCP.
        # This is local process plumbing, not access to a model provider.
        try:
            local = isinstance(address, tuple) and ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            local = False
        if not local:
            raise AssertionError("Outbound network blocked by contrast suite")
        return original_connect(sock, address)

    with (output / "unittest.log").open("x", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream), patch.object(
            socket.socket, "connect", new=local_connect
        ), patch.object(socket, "create_connection", side_effect=AssertionError("Outbound network blocked")):
            suite = unittest.TestSuite()
            for pattern in patterns:
                suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern))
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    return {"kind": "offline_harness_regression_contrast", "passed": result.wasSuccessful(),
            "tests_run": result.testsRun, "passed_tests": len(result.passed_ids),
            "skipped": [{"id": test.id(), "reason": reason} for test, reason in result.skipped],
            "failures": [{"id": test.id(), "traceback": trace} for test, trace in result.failures],
            "errors": [{"id": test.id(), "traceback": trace} for test, trace in result.errors],
            "passed_test_ids": result.passed_ids, "test_file_sha256": {name: sha(ROOT / "tests" / name) for name in patterns},
            "log_sha256": sha(output / "unittest.log"), "provider_requests": 0,
            "network_scope": "External socket dispatch blocked; literal loopback allowed for Windows asyncio; subprocess fixtures are local checks"}


def execute_check(source, test_code, directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "solution.py").write_text(source, encoding="utf-8")
    driver = ("import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent))\n"
              "from solution import *\n" + test_code)
    (directory / "host_check.py").write_text(driver, encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT"}}
    started = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-I", "-B", "host_check.py"], cwd=directory,
                              env=env, capture_output=True, timeout=8)
        result = {"passed": proc.returncode == 0, "returncode": proc.returncode, "timeout": False}
        (directory / "output.txt").write_bytes(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as exc:
        result = {"passed": False, "returncode": None, "timeout": True}
        (directory / "output.txt").write_bytes((exc.stdout or b"") + (exc.stderr or b""))
    return {**result, "seconds": time.perf_counter() - started, "source_sha256": sha(directory / "solution.py"),
            "driver_sha256": sha(directory / "host_check.py"), "output_sha256": sha(directory / "output.txt")}


def reevaluate(output):
    suite, evaluators = read(SUITE), read(EVALUATORS)
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    observations = rows(RUN / "observations.jsonl")
    expected = {(task, model) for task in tasks for model in suite["models"]}
    if len(observations) != len(expected) or {(r["task_id"], r["model"]) for r in observations} != expected:
        raise ValueError("Require exactly the complete frozen cohort, without duplicate branches")
    sources = []
    for row in observations:
        model_short = row["model"].rsplit("/", 1)[-1]
        folder = "cascade" if row["stage"] == "cascade" else "control-" + model_short
        episode_root = RUN / "tasks" / row["task_id"] / folder
        result = read(episode_root / "result.json")
        attempts = [attempt for attempt in result["attempts"] if attempt["model"] == row["model"]]
        if len(attempts) != 1:
            raise ValueError("Expected a unique recorded model attempt")
        source_path = (episode_root / attempts[0]["workspace"] / "solution.py").resolve()
        if not source_path.is_relative_to(episode_root.resolve()) or sha(source_path) != row["source_sha256"]:
            raise ValueError("Saved source changed or escaped the recorded workspace")
        sources.append((row, source_path.read_text(encoding="utf-8")))
    checks = []
    for index, (row, source) in enumerate(sources, 1):
        task = tasks[row["task_id"]]
        branch = output / "checks" / row["task_id"] / row["model"].rsplit("/", 1)[-1]
        public = execute_check(source, task["public_tests"], branch / "public")
        private = execute_check(source, evaluators["tasks"][row["task_id"]]["private_tests"], branch / "private")
        accepted = public["passed"] and private["passed"] and row["episode_status"] == "completed"
        agrees = (public["passed"] == row["public_passed"] and private["passed"] == row["private_passed"]
                  and accepted == row["accepted"])
        checks.append({"task_id": row["task_id"], "model": row["model"], "public": public, "private": private,
                       "episode_status_preserved": row["episode_status"], "accepted": accepted, "matches_saved": agrees})
        if index % 20 == 0 or index == len(sources):
            print(f"Verified saved branches {index}/{len(sources)}", flush=True)
    return {"kind": "fresh_local_execution_of_saved_outputs", "passed": all(row["matches_saved"] for row in checks),
            "tasks": len(tasks), "branches": len(checks), "test_processes": 2 * len(checks),
            "mismatches": [{"task_id": row["task_id"], "model": row["model"]} for row in checks if not row["matches_saved"]],
            "provider_requests": 0, "new_model_samples": 0, "checks": checks,
            "limitation": "Rechecks saved code; does not add independent tasks, model outcomes or recover missing usage",
            "input_sha256": {"observations.jsonl": sha(RUN / "observations.jsonl"), "suite": sha(SUITE), "evaluators": sha(EVALUATORS)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("regression", "reevaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", action="append", help="Exact test filename, repeatable; defaults to the declared harness regression set")
    args = parser.parse_args(argv)
    patterns = tuple(args.pattern or PATTERNS)
    if any(Path(name).name != name or not name.startswith("test_") or not name.endswith(".py")
           or not (ROOT / "tests" / name).is_file() for name in patterns):
        raise ValueError("Patterns must be exact existing test filenames inside tests")
    fingerprints = frozen_integrity()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = regressions(args.output, patterns) if args.mode == "regression" else reevaluate(args.output)
    if frozen_integrity() != fingerprints:
        raise ValueError("Frozen files changed during verification")
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started,
                  frozen_files_verified=fingerprints, verifier_sha256=sha(Path(__file__)), python=sys.version)
    write(args.output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key in
                     {"kind", "passed", "tests_run", "passed_tests", "skipped", "tasks", "branches", "test_processes", "mismatches", "seconds"}}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
