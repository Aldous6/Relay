"""Real SDK integration and transport accounting, with all model replies mocked."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
AVAILABLE = importlib.util.find_spec("openhands") is not None

if AVAILABLE:
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network")):
        from litellm import ModelResponse
        from openhands.sdk import Agent, LLM
        from openhands.sdk.llm import Message, TextContent
        import openhands.sdk.llm.llm as sdk_llm
        from openhands.sdk.tool.builtins.finish import FinishTool
        from relay_router.cascade import AccountingIncomplete, AttemptContext, BudgetExceeded, SpendBudget, run_cascade
        from relay_router.cascade_verifier import CheckCommand, CommandVerifier
        from relay_router.integrations.openhands_cascade import _audit_transport, make_openhands_attempt
        import relay_router.integrations.openhands_cascade as integration


@unittest.skipUnless(AVAILABLE, "Requires the optional OpenHands SDK 1.62.0 environment")
class OpenHandsCascadeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.model = "openai/gpt-4.1-mini"
        self.baseline = "openai/gpt-4.1"
        self.budget = SpendBudget(.01, 4)
        self.context = AttemptContext(1, self.model, self.root / "workspace", self.root / "state", self.budget)
        self.context.workspace.mkdir()

    def llm(self, model=None):
        return LLM(model=model or self.model, api_key="secret-fixture-key", num_retries=0,
                   api_mode="chat", max_output_tokens=128, timeout=10, caching_prompt=False,
                   input_cost_per_token=1e-6, output_cost_per_token=2e-6)

    def raw(self, *, finish=False, usage=None):
        message = {"role": "assistant", "content": "private-response"}
        if finish:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "finish-fixture", "type": "function", "function": {
                    "name": FinishTool.create()[0].name,
                    "arguments": json.dumps({"message": "Fixture finished"}),
                },
            }]}
        return ModelResponse(id="fixture", model=self.model,
                             choices=[{"index": 0, "message": message,
                                       "finish_reason": "tool_calls" if finish else "stop"}],
                             usage=usage if usage is not None else {
                                 "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    def kwargs(self, **changes):
        return {"model": self.model, "messages": [{"role": "user", "content": "private-prompt"}],
                "tools": None, "max_tokens": 128, "max_retries": 0, "timeout": 10, **changes}

    def guard(self, llm=None, **changes):
        return _audit_transport(self.context, llm or self.llm(),
                                **{"reservation_usd": .001, "max_request_bytes": 1000, **changes})

    def test_each_actual_transport_dispatch_is_counted_and_report_is_private(self):
        with patch.object(sdk_llm, "litellm_completion", return_value=self.raw()) as transport:
            with self.guard():
                sdk_llm.litellm_completion(**self.kwargs())
                sdk_llm.litellm_completion(**self.kwargs())
        self.assertEqual(transport.call_count, 2)
        result = self.budget.snapshot()
        self.assertEqual(result["http_calls"], 2)
        self.assertTrue(result["cost_complete"])
        self.assertAlmostEqual(result["cost_usd"], .00004)
        for secret in ("secret-fixture-key", "private-prompt", "private-response"):
            self.assertNotIn(secret, json.dumps(result))

    def test_failed_call_remains_unknown_and_blocks_retry(self):
        with patch.object(sdk_llm, "litellm_completion", side_effect=RuntimeError("private-provider-message")) as transport:
            with self.guard():
                with self.assertRaises(RuntimeError):
                    sdk_llm.litellm_completion(**self.kwargs())
                with self.assertRaises(AccountingIncomplete):
                    sdk_llm.litellm_completion(**self.kwargs())
        self.assertEqual(transport.call_count, 1)
        report = self.budget.snapshot()
        self.assertEqual(report["calls"][0]["status"], "error")
        self.assertIsNone(report["cost_usd"])
        self.assertAlmostEqual(report["committed_usd"], .001)
        self.assertNotIn("private-provider-message", json.dumps(report))

    def test_missing_usage_is_not_free_and_blocks_next_dispatch(self):
        with patch.object(sdk_llm, "litellm_completion", return_value=self.raw(usage={"prompt_tokens": 0, "completion_tokens": 0})) as transport:
            with self.guard():
                sdk_llm.litellm_completion(**self.kwargs())
                with self.assertRaises(AccountingIncomplete):
                    sdk_llm.litellm_completion(**self.kwargs())
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(self.budget.snapshot()["cost_complete"])

    def test_measured_reservation_overrun_is_kept_and_stops_dispatch(self):
        with patch.object(sdk_llm, "litellm_completion", return_value=self.raw()) as transport:
            with self.guard(reservation_usd=.000001):
                sdk_llm.litellm_completion(**self.kwargs())
                with self.assertRaises(BudgetExceeded):
                    sdk_llm.litellm_completion(**self.kwargs())
        self.assertEqual(transport.call_count, 1)
        report = self.budget.snapshot()
        self.assertTrue(report["reservation_exceeded"])
        self.assertAlmostEqual(report["cost_usd"], .00002)

    def test_actual_call_cap_stops_before_provider(self):
        self.context = AttemptContext(1, self.model, self.context.workspace, self.context.state_dir, SpendBudget(.01, 1))
        with patch.object(sdk_llm, "litellm_completion", return_value=self.raw()) as transport:
            with self.guard():
                sdk_llm.litellm_completion(**self.kwargs())
                with self.assertRaises(BudgetExceeded):
                    sdk_llm.litellm_completion(**self.kwargs())
        self.assertEqual(transport.call_count, 1)

    def test_model_context_limit_and_retry_overrides_reject_before_dispatch(self):
        invalid = [{"model": self.baseline}, {"max_tokens": 129}, {"max_retries": 1}, {"num_retries": 1},
                   {"retries": 1}, {"context_window_fallback_dict": {self.model: self.baseline}}, {"n": 2},
                   {"timeout": None}, {"timeout": 11}, {"extra_body": {"max_tokens": 999}},
                   {"stream": True}, {"messages": [{"role": "user", "content": "x" * 2000}]}]
        with patch.object(sdk_llm, "litellm_completion", return_value=self.raw()) as transport:
            with self.guard():
                for change in invalid:
                    with self.subTest(change=list(change)):
                        with self.assertRaises(ValueError):
                            sdk_llm.litellm_completion(**self.kwargs(**change))
        transport.assert_not_called()
        self.assertEqual(self.budget.snapshot()["http_calls"], 0)

    def test_guard_rejects_overlap_and_restores_transport(self):
        original = sdk_llm.litellm_completion
        with self.guard():
            with self.assertRaises(RuntimeError):
                with self.guard():
                    pass
            with self.assertRaises(ValueError):
                sdk_llm.litellm_responses(model=self.model)
        self.assertIs(sdk_llm.litellm_completion, original)

    def test_sdk_internal_cache_repair_reenters_admission_and_is_blocked_after_unknown_cost(self):
        llm = self.llm()
        failure = RuntimeError("private-cache-error")
        messages = [Message(role="user", content=[TextContent(text="private-prompt")])]
        with patch.object(sdk_llm, "litellm_completion", side_effect=failure) as transport, \
             patch.object(sdk_llm, "is_prompt_cache_too_small", side_effect=lambda exc: exc is failure), \
             patch.object(LLM, "is_caching_prompt_active", return_value=True):
            with self.guard(llm):
                with self.assertRaises(Exception):
                    llm.completion(messages)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(self.budget.snapshot()["http_calls"], 1)
        self.assertIsNone(self.budget.snapshot()["cost_usd"])

    def test_real_sdk_cheap_pass_avoids_strong_and_keeps_final_acceptance_unknown(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "check.py").write_text("assert 2 + 3 == 5\n", encoding="utf-8")
        verifier = CommandVerifier.capture(snapshot, checks=[CheckCommand("checks", (sys.executable, "-B", "check.py"))],
                                            protected_files=["check.py"])
        models = []

        def factory(context):
            models.append(context.model)
            return Agent(llm=self.llm(context.model), tools=[], condenser=None, mcp_config={},
                         include_default_tools=["FinishTool"], system_prompt="Use finish for this fixture.")

        attempt = make_openhands_attempt(factory, "private-prompt",
                                         reservation_usd_by_model={self.model: .001, self.baseline: .002},
                                         tools_are_local=True, max_iterations=2)
        with patch.object(sdk_llm, "litellm_completion", return_value=self.raw(finish=True)) as transport:
            result = run_cascade(snapshot_dir=snapshot, output_dir=self.root / "run", task_id="fixture",
                                 candidate_model=self.model, baseline_model=self.baseline,
                                 run_attempt=attempt, verifier=verifier, budget=self.budget,
                                 mode="experimental", allow_experimental=True, require_initial_failure=False)
        self.assertEqual(models, [self.model])
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(result["operational_passed"])
        self.assertFalse(result["escalated"])
        self.assertIsNone(result["accepted"])
        self.assertIsNone(result["savings_usd"])

    def test_uninitialized_agent_requirement_and_local_tool_declaration(self):
        with self.assertRaises(ValueError):
            make_openhands_attempt(lambda _: None, "Task", reservation_usd_by_model={self.model: .001}, tools_are_local=False)
        agent = Agent(llm=self.llm(), tools=[], condenser=None)
        agent._initialized = True
        attempt = make_openhands_attempt(lambda _: agent, "Task", reservation_usd_by_model={self.model: .001}, tools_are_local=True)
        with patch.object(sdk_llm, "litellm_completion") as transport:
            outcome = attempt(self.context)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_type, "ValueError")
        transport.assert_not_called()
        fresh = Agent(llm=self.llm(), tools=[], condenser=None)
        under_reserved = make_openhands_attempt(lambda _: fresh, "Task",
            reservation_usd_by_model={self.model: 1e-9}, tools_are_local=True)
        with patch.object(sdk_llm, "litellm_completion") as transport:
            outcome = under_reserved(self.context)
        self.assertEqual(outcome.status, "error")
        transport.assert_not_called()

    def test_llm_extra_body_cannot_override_budget_parameters(self):
        for key, value in (("max_tokens", 999), ("max_completion_tokens", 999), ("timeout", 0),
                           ("stream", True), ("n", 2)):
            with self.subTest(key=key):
                llm = self.llm().model_copy(update={"litellm_extra_body": {key: value}})
                agent = Agent(llm=llm, tools=[], condenser=None)
                attempt = make_openhands_attempt(lambda _: agent, "Task",
                    reservation_usd_by_model={self.model: .001}, tools_are_local=True)
                with patch.object(sdk_llm, "litellm_completion") as transport:
                    outcome = attempt(self.context)
                self.assertEqual(outcome.status, "error")
                self.assertEqual(outcome.error_type, "ValueError")
                transport.assert_not_called()

    def test_unsupported_sdk_and_terminal_error_are_not_completed_attempts(self):
        def factory(context):
            return Agent(llm=self.llm(context.model), tools=[], condenser=None, mcp_config={})
        attempt = make_openhands_attempt(factory, "Task", reservation_usd_by_model={self.model: .001}, tools_are_local=True)
        with patch.object(integration, "distribution_version", return_value="unreviewed-version"), \
             patch.object(sdk_llm, "litellm_completion") as transport:
            outcome = attempt(self.context)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_type, "ValueError")
        transport.assert_not_called()
        fake = SimpleNamespace(state=SimpleNamespace(execution_status="error"),
                               send_message=lambda prompt: None, run=lambda: None, close=lambda: None)
        with patch("openhands.sdk.conversation.impl.local_conversation.LocalConversation", return_value=fake):
            outcome = attempt(self.context)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_type, "OpenHandsIncompleteAttempt")

    def test_full_demo_fallback_restores_workspace_and_accounts_both_attempts(self):
        spec = importlib.util.spec_from_file_location("cascade_demo_fixture", ROOT / "examples" / "relay_openhands_cascade.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_offline_demo(self.root / "demo")
        self.assertTrue(result["operational_passed"])
        self.assertTrue(result["escalated"])
        self.assertEqual(result["selected_attempt"], 2)
        self.assertEqual(result["accounting"]["http_calls"], 2)
        self.assertEqual([item["verification"]["status"] for item in result["attempts"]], ["failed", "passed"])
        self.assertTrue(result["original_snapshot_unchanged"])
        self.assertIsNone(result["accepted"])
        serialized = json.dumps(result)
        self.assertNotIn("offline-fixture-key", serialized)


if __name__ == "__main__":
    unittest.main()
