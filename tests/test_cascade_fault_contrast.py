"""Adversarial dev4 checks: real SDK with simulated transport, never a provider.

The frozen 43-task validation is read only. These cases are engineering evidence,
not new model-quality observations or a rerun of the failed task.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
SDK_AVAILABLE = (importlib.util.find_spec("openhands") is not None
                 and importlib.util.find_spec("litellm") is not None)

if SDK_AVAILABLE:
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline imports only")):
        from litellm import BadRequestError, ModelResponse
        from openhands.sdk import LLM
        import openhands.sdk.llm.llm as sdk_llm
        from relay_router.cascade import AccountingIncomplete, AttemptContext, AttemptOutcome, SpendBudget, run_cascade
        from relay_router.cascade_verifier import CheckCommand, CommandVerifier
        from relay_router.integrations.openhands_cascade import _audit_transport
        import run_cascade_validation as runner


CHEAP = "groq/openai/gpt-oss-20b"
STRONG = "groq/openai/gpt-oss-120b"
DIAGNOSTICS = []


@unittest.skipUnless(SDK_AVAILABLE, "Requires optional OpenHands SDK in .venv-harness")
class FaultContrast(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="relay-offline-contrast-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network"))
        network.start()
        self.addCleanup(network.stop)

    def fixture_error(self, *, with_usage=False):
        error = BadRequestError(message="tool_use_failed offline fixture", model=CHEAP, llm_provider="groq")
        error.body = {"error": {"code": "tool_use_failed", "message": "offline fixture"}}
        if with_usage:
            # An arbitrary exception payload is not an authenticated usage
            # reconciliation. Dev4 deliberately does not infer a bill from it.
            error.usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return error

    def direct_context(self):
        budget = SpendBudget(.05, 6)
        context = AttemptContext(1, CHEAP, self.root / "workspace", self.root / "state", budget)
        context.workspace.mkdir()
        llm = LLM(model=CHEAP, api_key="offline-fixture", num_retries=0, api_mode="chat",
                  timeout=10, max_output_tokens=128, caching_prompt=False,
                  input_cost_per_token=.075 / 1e6, output_cost_per_token=.3 / 1e6)
        kwargs = dict(model=CHEAP, messages=[{"role": "user", "content": "fixture"}],
                      max_tokens=128, timeout=10, max_retries=0)
        return context, llm, kwargs

    def test_exception_usage_is_not_silently_assumed_to_be_a_bill(self):
        context, llm, kwargs = self.direct_context()
        with patch.object(sdk_llm, "litellm_completion", side_effect=self.fixture_error(with_usage=True)) as transport:
            with _audit_transport(context, llm, reservation_usd=.005, max_request_bytes=1000):
                with self.assertRaises(BadRequestError):
                    sdk_llm.litellm_completion(**kwargs)
                with self.assertRaises(AccountingIncomplete):
                    sdk_llm.litellm_completion(**kwargs)
        self.assertEqual(transport.call_count, 1)
        self.assertIsNone(context.budget.snapshot()["cost_usd"])
        self.assertEqual(context.budget.snapshot()["calls"][0]["status"], "error")

    def test_another_thread_cannot_bypass_the_active_transport_budget(self):
        context, llm, kwargs = self.direct_context()
        failures = []

        def other_thread():
            try:
                sdk_llm.litellm_completion(**kwargs)
            except Exception as exc:
                failures.append(type(exc).__name__)

        with patch.object(sdk_llm, "litellm_completion") as transport:
            with _audit_transport(context, llm, reservation_usd=.005, max_request_bytes=1000):
                thread = threading.Thread(target=other_thread)
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
        transport.assert_not_called()
        self.assertEqual(failures, ["RuntimeError"])
        self.assertEqual(context.budget.snapshot()["http_calls"], 0)

    def run_sdk_fixture(self, behavior):
        task = {"task_id": "HumanEval-999", "group_id": "offline-fault-contrast",
                "workflow": "function-implementation", "instruction": "Implement add(a, b).",
                "initial_source": "def add(a, b):\n    return 0\n", "public_tests": "assert add(2, 3) == 5\n"}
        private_source = "assert add(9, 9) == 18  # PRIVATE_FAULT_CONTRAST_SENTINEL\n"
        suite = deepcopy(runner.read(runner.DEFAULT_SUITE))
        suite["tasks"] = [task]
        evaluators = {"tasks": {task["task_id"]: {"private_tests": private_source,
                    "sha256": hashlib.sha256(private_source.encode()).hexdigest()}}}
        source_hashes = {"offline-fixture": "e" * 64}
        frozen = {"schema_version": 1, "created_at_utc": "2026-11-18T19:00:00+00:00",
                  "contract": {"suite_content_sha256": runner.digest(suite), "source_sha256": source_hashes,
                               "private_evaluators_content_sha256": runner.digest(evaluators)}}
        frozen["freeze_id"] = runner.digest(frozen)
        output = self.root / "run"
        output.mkdir()
        seen = []
        cheap_calls = 0

        def completion(**kwargs):
            nonlocal cheap_calls
            self.assertNotIn("PRIVATE_FAULT_CONTRAST_SENTINEL", json.dumps(kwargs, default=str))
            cheap = str(kwargs["model"]).endswith("gpt-oss-20b")
            seen.append(CHEAP if cheap else STRONG)
            if cheap:
                cheap_calls += 1
            if cheap and behavior == "http400":
                raise self.fixture_error()
            if not cheap and behavior == "control_http400":
                raise self.fixture_error()
            source = "def add(a, b):\n    return a + b\n"
            if cheap and behavior == "public_only":
                source = "def add(a, b):\n    return 5\n"
            if cheap and behavior == "two_requests" and cheap_calls == 1:
                source = "def add(a, b):\n    return -1\n"
            tool = next(t for t in kwargs["tools"] if "source" in t["function"].get("parameters", {}).get("properties", {}))
            return ModelResponse(id="offline-fault", model=kwargs["model"],
                choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": f"offline-{len(seen)}", "type": "function",
                    "function": {"name": tool["function"]["name"], "arguments": json.dumps({"source": source})}}]}}],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})

        with patch.object(runner, "source_hashes", return_value=source_hashes), \
             patch.object(sdk_llm, "litellm_completion", side_effect=completion):
            status = runner.execute(suite, evaluators, frozen, output, key="offline-fixture")
        observed = runner.read_observations(suite, output)
        events = runner.reconcile_http(suite, output, observed)
        runtime = runner.read(output / "tasks/HumanEval-999/cascade/result.json")
        self.assertEqual(len(observed), 2)
        self.assertEqual(len(runner.rows(output / "task_completions.jsonl")), 1)
        self.assertIsNone(runtime["accepted"])
        self.assertTrue(runtime["original_snapshot_unchanged"])
        by_model = {r["model"]: r for r in observed}
        DIAGNOSTICS.append({"scenario": behavior, "exit_code": status, "simulated_dispatches": len(events),
                            "dispatch_order": seen, "runtime_selected_model": runtime["selected_model"],
                            "runtime_reason": runtime["reason"], "cascade_cost_complete": runtime["accounting"]["cost_complete"],
                            "accepted_by_model": {m:r["accepted"] for m,r in by_model.items()}})
        return status, runtime, by_model, events, seen

    def test_real_sdk_http400_reproduces_unknown_cost_stop_and_preserves_control(self):
        status, runtime, rows, events, seen = self.run_sdk_fixture("http400")
        self.assertEqual(status, 2)
        self.assertEqual(seen, [CHEAP, STRONG])
        self.assertEqual(runtime["reason"], "AccountingIncomplete")
        self.assertFalse(runtime["escalated"])
        self.assertFalse(runtime["operational_passed"])
        self.assertIsNone(runtime["accounting"]["cost_usd"])
        self.assertEqual(runtime["accounting"]["http_calls"], 1)
        self.assertFalse(rows[CHEAP]["accepted"])
        self.assertTrue(rows[STRONG]["accepted"])
        self.assertEqual(rows[STRONG]["stage"], "control")
        self.assertEqual(events[0]["provider_error_code"], "tool_use_failed")
        self.assertEqual(events[0]["http_status"], 400)

    def test_real_sdk_public_pass_private_fail_does_not_reopen_route_or_retry(self):
        status, runtime, rows, events, seen = self.run_sdk_fixture("public_only")
        self.assertEqual(status, 0)
        self.assertEqual(seen, [CHEAP, STRONG])
        self.assertEqual(runtime["selected_model"], CHEAP)
        self.assertTrue(runtime["operational_passed"])
        self.assertTrue(rows[CHEAP]["public_passed"])
        self.assertFalse(rows[CHEAP]["private_passed"])
        self.assertFalse(rows[CHEAP]["accepted"])
        self.assertTrue(rows[STRONG]["accepted"])
        self.assertEqual(len(events), 2)

    def test_real_sdk_second_edit_cost_is_included_before_missing_control(self):
        status, runtime, rows, events, seen = self.run_sdk_fixture("two_requests")
        self.assertEqual(status, 0)
        self.assertEqual(seen, [CHEAP, CHEAP, STRONG])
        self.assertEqual(runtime["accounting"]["http_calls"], 2)
        self.assertEqual(rows[CHEAP]["http_attempts"], 2)
        self.assertAlmostEqual(runtime["accounting"]["cost_usd"], 2 * (100 * .075 + 20 * .3) / 1e6)
        self.assertTrue(rows[CHEAP]["accepted"])
        self.assertEqual(len(events), 3)

    def test_control_failure_does_not_poison_known_operational_cost(self):
        status, runtime, rows, events, seen = self.run_sdk_fixture("control_http400")
        self.assertEqual(status, 2)
        self.assertEqual(seen, [CHEAP, STRONG])
        self.assertTrue(runtime["operational_passed"])
        self.assertTrue(runtime["accounting"]["cost_complete"])
        self.assertTrue(rows[CHEAP]["accepted"])
        self.assertFalse(rows[STRONG]["accepted"])
        self.assertIsNone(rows[STRONG]["cost_usd"])
        self.assertEqual(runtime["accounting"]["http_calls"], 1)
        self.assertEqual(len(events), 2)

    def known_error_case(self, budget):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "solution.py").write_text("value = 0\n", encoding="utf-8")
        (snapshot / "check.py").write_text("from pathlib import Path\nassert Path('solution.py').read_text() == 'value = 42\\n'\n", encoding="utf-8")
        verifier = CommandVerifier.capture(snapshot, checks=[CheckCommand("checks", (sys.executable, "-I", "-B", "check.py"))], protected_files=["check.py"])
        dispatches = []

        def attempt(context):
            self.assertEqual((context.workspace / "solution.py").read_text(), "value = 0\n")
            call = context.budget.begin_call(context.model, .001)
            dispatches.append(context.model)
            context.budget.finish_call(call, status="error" if context.model == CHEAP else "completed",
                                       cost_usd=.0005, input_tokens=100, output_tokens=20)
            if context.model == CHEAP:
                (context.workspace / "solution.py").write_text("value = -99\n")
                return AttemptOutcome("error", "FixtureKnownCostFailure")
            (context.workspace / "solution.py").write_text("value = 42\n")
            return AttemptOutcome()

        runtime = run_cascade(snapshot_dir=snapshot, output_dir=self.root / "cascade", task_id="offline-known-error",
            candidate_model=CHEAP, baseline_model=STRONG, run_attempt=attempt, verifier=verifier,
            budget=budget, mode="experimental", allow_experimental=True)
        return runtime, dispatches

    def test_accounted_error_can_escalate_from_original_and_includes_failure_cost(self):
        runtime, dispatches = self.known_error_case(SpendBudget(.01, 6))
        self.assertEqual(dispatches, [CHEAP, STRONG])
        self.assertTrue(runtime["operational_passed"])
        self.assertEqual(runtime["selected_model"], STRONG)
        self.assertTrue(runtime["escalated"])
        self.assertEqual(runtime["attempts"][0]["status"], "error")
        self.assertAlmostEqual(runtime["accounting"]["cost_usd"], .001)

    def test_accounted_error_cannot_evade_actual_http_call_limit(self):
        runtime, dispatches = self.known_error_case(SpendBudget(.01, 1))
        self.assertEqual(dispatches, [CHEAP])
        self.assertEqual(runtime["reason"], "BudgetExceeded")
        self.assertFalse(runtime["operational_passed"])
        self.assertAlmostEqual(runtime["accounting"]["cost_usd"], .0005)

    def test_accounted_error_cannot_evade_remaining_reservation_budget(self):
        runtime, dispatches = self.known_error_case(SpendBudget(.0011, 6))
        self.assertEqual(dispatches, [CHEAP])
        self.assertFalse(runtime["operational_passed"])
        self.assertEqual(runtime["attempts"][1]["error_type"], "BudgetExceeded")
        self.assertEqual(runtime["accounting"]["http_calls"], 1)
        self.assertAlmostEqual(runtime["accounting"]["cost_usd"], .0005)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError("Choose a new evidence destination")
    frozen = json.loads((ROOT / "outputs/harness-cascade/v1/run/frozen_policy.json").read_text(encoding="utf-8"))
    hashes = frozen["contract"]["source_sha256"]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if any(sha(ROOT / path) != expected for path, expected in hashes.items()):
        raise ValueError("Frozen source changed before contrast")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FaultContrast))
    unchanged = all(sha(ROOT / path) == expected for path, expected in hashes.items())
    report = {"kind": "offline_runtime_fault_contrast", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "passed": result.wasSuccessful(),
              "frozen_sources_unchanged": unchanged, "frozen_sources_verified": len(hashes),
              "provider_calls": 0, "real_sdk": SDK_AVAILABLE, "model_responses_simulated": True,
              "quality_or_savings_evidence": False, "python_executable": sys.executable,
              "packages": {name: importlib.metadata.version(name) for name in ("openhands-sdk", "litellm")} if SDK_AVAILABLE else {},
              "test_source_sha256": sha(Path(__file__)), "scenarios": DIAGNOSTICS,
              "failures_detail": [{"test": test.id(), "traceback": detail} for test, detail in result.failures + result.errors]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    raise SystemExit(0 if result.wasSuccessful() and unchanged else 1)
