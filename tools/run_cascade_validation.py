"""Frozen fresh-task validation: execute Relay dev4, then missing controls.

Only public task source enters the agent. Final evaluation occurs after the
runtime has sealed its decision. Every SDK transport request is audited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "data/imported/harness-cascade-v1/fresh_suite.json"
DEFAULT_EVALUATORS = ROOT / "data/imported/harness-cascade-v1/private_evaluators.json"
DEFAULT_RUN = ROOT / "outputs/harness-cascade/v1/run"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
from relay_router.cascade import SpendBudget, run_cascade
from relay_router.cascade_verifier import CheckCommand, CommandVerifier
from relay_router.harness_episode import SYSTEM_PROMPT, make_edit_tool, run_check, task_prompt
from run_adaptive_pilot import BoundedAuditedTransport, TARIFFS, _credential

SOURCES = (
    "tools/run_cascade_validation.py", "tools/prepare_cascade_validation.py",
    "tools/report_cascade_validation.py", "docs/openhands_cascade_validation_v1.md",
    "src/relay_router/cascade.py", "src/relay_router/cascade_verifier.py",
    "src/relay_router/cascade_evaluation.py", "src/relay_router/harness_episode.py",
    "src/relay_router/workflow.py", "src/relay_router/integrations/openhands.py",
    "src/relay_router/integrations/openhands_io.py", "src/relay_router/integrations/openhands_cascade.py",
    "tools/run_adaptive_pilot.py", "tools/relay_harness_live.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).exists() else []


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCES}


def validate_suite(suite):
    expected = set(TARIFFS)
    if set(suite["models"]) != expected or suite["baseline_model"] != "groq/openai/gpt-oss-120b":
        raise ValueError("This protocol uses the frozen Groq 20B/120B pair")
    if suite["max_calls_per_episode"] != 3 or suite["max_output_tokens"] != 1024:
        raise ValueError("Episode limits changed")
    ids, groups = set(), set()
    for task in suite["tasks"]:
        if not re.fullmatch(r"(?:HumanEval-[0-9]+|QuixBugs-[a-z0-9_]+)", task["task_id"]):
            raise ValueError("Task identity is outside this public source scope")
        if task["task_id"] in ids or task["group_id"] in groups:
            raise ValueError("Duplicate task/group")
        if {"private_tests", "gold_source", "accepted"}.intersection(task):
            raise ValueError("Private evaluator fields cannot enter agent-facing tasks")
        if any(hashlib.sha256(task[field].encode()).hexdigest() != task[field + "_sha256"]
               for field in ("initial_source", "public_tests")):
            raise ValueError("Public task source differs from its frozen digest")
        if task_prompt(task) != task["prompt_text"] or hashlib.sha256(task_prompt(task).encode()).hexdigest() != task["routing_text_sha256"]:
            raise ValueError("Actual agent prompt differs from the prepared prompt")
        ids.add(task["task_id"])
        groups.add(task["group_id"])
    if len(ids) < 20:
        raise ValueError("The declared pilot needs at least 20 fresh groups")
    return suite


def prepare_run(suite_path=DEFAULT_SUITE, evaluators_path=DEFAULT_EVALUATORS, output=DEFAULT_RUN):
    suite_path, evaluators_path, output = Path(suite_path), Path(evaluators_path), Path(output)
    suite = validate_suite(read(suite_path))
    preparation_path = suite_path.parent / "preparation_manifest.json"
    if preparation_path.exists():
        preparation = read(preparation_path)
        for entry in preparation["files"]:
            path = (suite_path.parent / entry["file"]).resolve()
            if path.parent != suite_path.parent.resolve() or file_hash(path) != entry["sha256"]:
                raise ValueError("Prepared data or its provenance changed")
        for relative, expected in preparation["source_sha256"].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError("Preparation source changed")
    evaluators = read(evaluators_path)
    if file_hash(evaluators_path) != suite["private_evaluator_sha256"]:
        raise ValueError("Evaluators differ from the frozen suite")
    if set(evaluators["tasks"]) != {task["task_id"] for task in suite["tasks"]}:
        raise ValueError("Final evaluators must cover the exact frozen cohort")
    for evaluator in evaluators["tasks"].values():
        if hashlib.sha256(evaluator["private_tests"].encode()).hexdigest() != evaluator["sha256"]:
            raise ValueError("Final evaluator digest mismatch")
    for task in suite["tasks"]:
        if task["private_tests_sha256"] != evaluators["tasks"][task["task_id"]]["sha256"]:
            raise ValueError("Task evaluator differs from its frozen digest")
    contract = {"suite_sha256": file_hash(suite_path), "suite_content_sha256": digest(suite),
                "private_evaluators_sha256": file_hash(evaluators_path),
                "source_sha256": source_hashes(), "tariffs": TARIFFS,
                "initial_public_checks": {task["task_id"]: task["initial_public_check"] for task in suite["tasks"]},
                "limits": {name: suite[name] for name in ("max_calls_per_episode", "max_output_tokens", "max_total_http_attempts", "max_observed_tokens_per_model")},
                "policy_config": {"require_initial_failure": True, "mode": "experimental", "cpu_hourly_usd": None,
                                  "max_task_cost_usd": .05, "max_task_actual_calls": 6, "max_task_admission_seconds": 600,
                                  "reserve_per_call_usd": {suite["models"][0]: .005, suite["models"][1]: .010}},
                "evaluation": "actual_runtime_then_missing_paired_controls_private_evaluation_after_decision",
                "packages": {name: importlib.metadata.version(name) for name in ("openhands-sdk", "litellm", "relay-harness")}}
    frozen_path = output / "frozen_policy.json"
    if frozen_path.exists():
        frozen = read(frozen_path)
        if frozen["contract"] != contract:
            raise ValueError("Frozen code, cohort, packages or policy changed")
        verify_frozen(suite, frozen)
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("A new run requires an empty output directory")
        frozen = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "contract": contract}
        frozen["freeze_id"] = digest(frozen)
        write(frozen_path, frozen)
    return suite, evaluators, frozen


def verify_frozen(suite, frozen):
    unsigned = {key: value for key, value in frozen.items() if key != "freeze_id"}
    if digest(unsigned) != frozen["freeze_id"] or digest(suite) != frozen["contract"]["suite_content_sha256"]:
        raise ValueError("Frozen manifest or task content changed")
    if frozen["contract"]["source_sha256"] != source_hashes():
        raise ValueError("Frozen source changed")
    return True


def read_observations(suite, output):
    observations = rows(Path(output) / "observations.jsonl")
    identities = set()
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    for row in observations:
        identity = row["task_id"], row["model"]
        if identity in identities or row["task_id"] not in tasks or row["model"] not in suite["models"]:
            raise ValueError("Duplicate or unexpected episode")
        if row["group_id"] != tasks[row["task_id"]]["group_id"]:
            raise ValueError("Group changed")
        if type(row["accepted"]) is not bool or type(row["public_passed"]) is not bool:
            raise ValueError("Acceptance and operational status must be explicit")
        if type(row.get("private_passed")) is not bool or row.get("episode_status") not in {"completed", "error"}:
            raise ValueError("Final evaluator and attempt state must be explicit")
        if row["accepted"] != (row["public_passed"] and row["private_passed"] and row["episode_status"] == "completed"):
            raise ValueError("Acceptance differs from the final evaluator and attempt state")
        if row["episode_id"] != _episode_id(row["task_id"], row["model"]):
            raise ValueError("Unexpected episode identity")
        record_path = Path(output) / "tasks" / row["task_id"] / (row["model"].rsplit("/", 1)[-1] + "-observation.json")
        if not record_path.is_file() or read(record_path) != row:
            raise ValueError("Episode record differs from its observation log")
        if row["cost_usd"] is not None and (type(row["cost_usd"]) not in (int, float) or not math.isfinite(row["cost_usd"]) or row["cost_usd"] < 0):
            raise ValueError("Invalid cost")
        identities.add(identity)
    return observations


def reconcile_http(suite, output, observations):
    output = Path(output)
    events = rows(output / "http_attempts.jsonl")
    starts = [read(path) for path in sorted((output / "attempt_starts").glob("*.json"))]
    ids = [row["attempt_id"] for row in events]
    if (any(type(value) is not int for value in ids) or ids != list(range(1, len(events) + 1))
            or len(starts) != len(events) or {row["attempt_id"] for row in starts} != set(ids)):
        raise ValueError("Duplicate or interrupted HTTP request; do not silently repeat")
    start_by_id = {row["attempt_id"]: row for row in starts}
    for event in events:
        if any(start_by_id[event["attempt_id"]].get(field) != event.get(field)
               for field in ("attempt_id", "episode_id", "model", "started_at_utc")):
            raise ValueError("Transport identity differs from its write-ahead record")
    if len(events) > suite["max_total_http_attempts"]:
        raise ValueError("Acquisition HTTP budget exceeded")
    known_episodes = {row["episode_id"] for row in observations}
    if any(event["episode_id"] not in known_episodes for event in events):
        raise ValueError("HTTP evidence lacks a finalized observation; do not repeat an interrupted task")
    for event in events:
        if event["model"] not in suite["models"] or event["status"] not in {"completed", "error"}:
            raise ValueError("Unexpected transport identity or state")
        if event["cost_usd"] is not None:
            if (type(event["cost_usd"]) not in (int, float) or not math.isfinite(event["cost_usd"]) or event["cost_usd"] < 0
                    or any(type(event[field]) is not int or event[field] < 0 for field in ("input_tokens", "output_tokens"))):
                raise ValueError("Invalid transport usage or cost")
            rates = TARIFFS[event["model"]]
            expected = (event["input_tokens"] * rates["input_per_million_usd"] + event["output_tokens"] * rates["output_per_million_usd"]) / 1e6
            if abs(expected - event["cost_usd"]) > 1e-12:
                raise ValueError("Transport price differs from frozen tariff")
    for model in suite["models"]:
        observed = sum((event["input_tokens"] or 0) + (event["output_tokens"] or 0) for event in events if event["model"] == model)
        if observed > suite["max_observed_tokens_per_model"]:
            raise ValueError("Observed token budget exceeded")
    for observation in observations:
        selected = [event for event in events if event["episode_id"] == observation["episode_id"]]
        if any(event["model"] != observation["model"] for event in selected):
            raise ValueError("Model episode contains another model's dispatch")
        complete = all(event["cost_usd"] is not None for event in selected)
        total = math.fsum(event["cost_usd"] for event in selected) if complete else None
        if len(selected) != observation["http_attempts"] or complete != observation["cost_complete"]:
            raise ValueError("Episode transport accounting differs")
        if (total is None) != (observation["cost_usd"] is None) or total is not None and abs(total - observation["cost_usd"]) > 1e-12:
            raise ValueError("Episode cost differs from actual requests")
    return events


def _driver(public_tests):
    return "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent))\nfrom solution import *\n" + public_tests


def _episode_id(task_id, model):
    return task_id + "-" + model.rsplit("/", 1)[-1]


def execute(suite, evaluators, frozen, output, *, key):
    import litellm
    from openhands.sdk import Agent, LLM
    import openhands.sdk.llm.llm as llm_module
    from relay_router.integrations.openhands_cascade import make_openhands_attempt
    output = Path(output)
    verify_frozen(suite, frozen)
    observations = read_observations(suite, output)
    reconcile_http(suite, output, observations)
    transport = BoundedAuditedTransport(llm_module.litellm_completion, suite, output)
    completed = {row["task_id"] for row in rows(output / "task_completions.jsonl")}
    started_ids = {path.stem for path in (output / "task_starts").glob("*.json")}
    if started_ids - completed:
        raise ValueError("A task is interrupted; its calls and outputs must be audited before any continuation")
    strong = suite["baseline_model"]
    cheap = next(model for model in suite["models"] if model != strong)
    litellm.telemetry = False
    logging.getLogger().setLevel(logging.ERROR)
    for position, task in enumerate(suite["tasks"], 1):
        if task["task_id"] in completed:
            continue
        verify_frozen(suite, frozen)
        task_root = output / "tasks" / task["task_id"]
        snapshot = task_root / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=False)
        (snapshot / "solution.py").write_text(task["initial_source"], encoding="utf-8")
        (snapshot / "host_check.py").write_text(_driver(task["public_tests"]), encoding="utf-8")
        verifier = CommandVerifier.capture(snapshot,
            checks=[CheckCommand("public-tests", (sys.executable, "-I", "-B", "host_check.py"), timeout_seconds=8)],
            protected_files=["host_check.py"], verifier_id="public-task-checks-v1")
        write(output / "task_starts" / (task["task_id"] + ".json"), {
            "task_id": task["task_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_id": frozen["freeze_id"]})
        phases = []

        def factory(context):
            rates = TARIFFS[context.model]
            llm = LLM(model=context.model, api_key=key, base_url="https://api.groq.com/openai/v1", num_retries=0,
                      api_mode="chat", timeout=90, temperature=0, max_output_tokens=1024, reasoning_effort="low",
                      log_completions=False, caching_prompt=False, usage_id="agent",
                      capability_overrides={"supports_prompt_cache_key": False, "supports_prompt_cache_retention": False},
                      input_cost_per_token=rates["input_per_million_usd"] / 1e6,
                      output_cost_per_token=rates["output_per_million_usd"] / 1e6)
            return Agent(llm=llm, condenser=None, critic=None, mcp_config={},
                         tools=[make_edit_tool(context.workspace, task["public_tests"], stop_on_public_pass=True)],
                         include_default_tools=["FinishTool"],
                         system_prompt=SYSTEM_PROMPT + " The host stops automatically when public checks pass; no closing response is needed.")

        sdk_attempt = make_openhands_attempt(factory, task_prompt(task),
            reservation_usd_by_model={cheap: .005, strong: .010}, tools_are_local=True, max_iterations=3)
        stage = "cascade"

        def attempt(context):
            if any(phase["model"] == context.model for phase in phases):
                raise ValueError("Do not repeat a model branch for this task")
            episode_id = _episode_id(task["task_id"], context.model)
            transport.episode, transport.active = episode_id, []
            start = time.perf_counter()
            outcome = sdk_attempt(context)
            phases.append({"model": context.model, "workspace": context.workspace, "episode_id": episode_id,
                           "stage": stage, "outcome": outcome, "http": list(transport.active),
                           "wall_seconds": time.perf_counter() - start})
            return outcome

        print(f"{position}/{len(suite['tasks'])} {task['task_id']} cascada real", flush=True)
        with patch.object(llm_module, "litellm_completion", transport):
            actual = run_cascade(snapshot_dir=snapshot, output_dir=task_root / "cascade", task_id=task["task_id"],
                candidate_model=cheap, baseline_model=strong, run_attempt=attempt, verifier=verifier,
                budget=SpendBudget(.05, 6, 600), mode="experimental", allow_experimental=True)
        result_path = task_root / "cascade/result.json"
        append(output / "cascade_results.jsonl", {"task_id": task["task_id"],
               "result_path": result_path.relative_to(output).as_posix(), "result_sha256": file_hash(result_path)})
        # The runtime has now committed its route. Only then acquire missing
        # fixed-model control branches. No private outcome is available to it.
        stage = "control"
        for model in (strong, cheap):
            if any(phase["model"] == model for phase in phases):
                continue
            with patch.object(llm_module, "litellm_completion", transport):
                run_cascade(snapshot_dir=snapshot, output_dir=task_root / ("control-" + model.rsplit("/", 1)[-1]),
                    task_id=task["task_id"], candidate_model=cheap if model == strong else strong,
                    baseline_model=model, run_attempt=attempt, verifier=verifier,
                    budget=SpendBudget(.05, 3, 600), mode="baseline", require_initial_failure=False)
        for phase in phases:
            public = verifier.verify(phase["workspace"])
            private = run_check((phase["workspace"] / "solution.py").read_text(encoding="utf-8"),
                evaluators["tasks"][task["task_id"]]["private_tests"],
                directory=task_root / "final-evaluation" / phase["model"].rsplit("/", 1)[-1], timeout=8)
            events = phase["http"]
            complete = all(event["cost_usd"] is not None for event in events)
            known = math.fsum(event["cost_usd"] for event in events if event["cost_usd"] is not None)
            row = {"task_id": task["task_id"], "group_id": task["group_id"], "workflow": task["workflow"],
                   "model": phase["model"], "episode_id": phase["episode_id"], "stage": phase["stage"],
                   "accepted": public.passed and private["passed"] and phase["outcome"].status == "completed",
                   "public_passed": public.passed, "private_passed": private["passed"],
                   "error_type": phase["outcome"].error_type, "episode_status": phase["outcome"].status,
                   "cost_usd": known if complete else None, "cost_complete": complete,
                   "cost_basis": "tariff_estimate" if complete else "unknown", "known_cost_subtotal_usd": known,
                   "http_attempts": len(events), "wall_seconds": phase["wall_seconds"],
                   "quota_wait_seconds": math.fsum(event["quota_wait_seconds"] for event in events),
                   "final_evaluation_seconds": private["seconds"] + public.duration_seconds,
                   "runtime_decision_sha256_before_final_evaluation": file_hash(result_path),
                   "final_evaluation_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_sha256": file_hash(phase["workspace"] / "solution.py")}
            write(task_root / (phase["model"].rsplit("/", 1)[-1] + "-observation.json"), row)
            append(output / "observations.jsonl", row)
            observations.append(row)
        append(output / "task_completions.jsonl", {"task_id": task["task_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        reconcile_http(suite, output, observations)
        print(json.dumps({"task_id": task["task_id"], "selected_model": actual["selected_model"],
                          "operational_passed": actual["operational_passed"], "cumulative_http": len(transport.events)}), flush=True)
        # Preserve provider/auth/quota failures and stop; no automatic task retry.
        failures = [event for phase in phases for event in phase["http"] if event["status"] == "error"]
        if failures:
            print("Fallo del proveedor conservado; no se repite ni se sustituye la tarea.", flush=True)
            return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--evaluators", type=Path, default=DEFAULT_EVALUATORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--credential-pipe")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args(argv)
    suite, evaluators, frozen = prepare_run(args.suite, args.evaluators, args.output)
    if args.freeze_only:
        print(json.dumps({"freeze_id": frozen["freeze_id"], "tasks": len(suite["tasks"]), "provider_generation_calls": 0}))
        return 0
    key = None
    try:
        key = _credential(args.credential_pipe)
        return execute(suite, evaluators, frozen, args.output, key=key)
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
