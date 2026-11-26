"""Optional integration tests against installed OpenHands, with no network.

Run in .venv-harness: python -m unittest discover -s tests -p test_openhands_adapter.py -v
The synthetic response is an integration fixture, never evidence of savings.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
AVAILABLE = importlib.util.find_spec("openhands") is not None
if AVAILABLE:
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network access")):
        from litellm import ModelResponse
        from openhands.sdk import Agent
        from openhands.sdk.llm import FallbackStrategy, LLM, LLMCallContext, LLMResponse, Message, RouterLLM, TextContent
        from openhands.sdk.llm.llm_registry import LLMRegistry
        from openhands.sdk.llm.utils.metrics import MetricsSnapshot
        from openhands.sdk.conversation.impl.local_conversation import LocalConversation
        from openhands.sdk.conversation.state import ConversationExecutionStatus
        from openhands.sdk.tool.builtins.finish import FinishAction, FinishTool
        from relay_router import Relay
        from relay_router.integrations.openhands import make_openhands_router, attach_openhands_agent


@unittest.skipUnless(AVAILABLE, "Install optional openhands-sdk==1.62.0 in a Python 3.12+ environment")
class OpenHandsAdapterTests(unittest.TestCase):
    def test_single_attachment_preserves_existing_agent_and_tools(self):
        agent = Agent(llm=self.llm, tools=[], condenser=None)
        relay = Relay(self.model)
        task, routed = attach_openhands_agent(agent, relay, workflow="python", models={self.model: self.llm}, max_calls=3)
        self.assertIs(agent.llm, self.llm)
        self.assertIs(routed.llm.relay_session, task)
        self.assertEqual(routed.tools, agent.tools)
        self.assertIsNone(task.finish(accepted=None)["cost_usd"])

    def setUp(self):
        # Windows builds its event-loop wakeup pipe with local socketpair().
        # Create that pipe before prohibiting all network in SDK execution.
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network access"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "observations.jsonl"
        self.model = "openai/gpt-4.1-mini"
        self.llm = LLM(model=self.model, api_key="offline-test-secret", num_retries=0,
                       api_mode="chat", input_cost_per_token=0.000001,
                       output_cost_per_token=0.000002, usage_id="agent")
        self.task = Relay(baseline_model=self.model, log_path=self.path).task("repair", max_calls=3)
        self.router = make_openhands_router(self.task, {self.model: self.llm})
        self.messages = [Message(role="user", content=[TextContent(text="private-test-prompt")])]

    def raw(self, **usage):
        return ModelResponse(id="offline-response", model=self.model,
                             choices=[{"message": {"role": "assistant", "content": "private-response"},
                                       "finish_reason": "stop", "index": 0}],
                             usage=usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    def response(self, raw=None):
        return LLMResponse(message=Message(role="assistant", content=[TextContent(text="private-response")]),
                           metrics=MetricsSnapshot(accumulated_cost=999), raw_response=raw or self.raw())

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_real_agent_and_registry_accept_router(self):
        self.assertIsInstance(self.router, RouterLLM)
        agent = Agent(llm=self.router, tools=[], condenser=None)
        self.assertIs(agent.llm, self.router)
        registry = LLMRegistry()
        registry.add(agent.llm)
        self.assertIs(registry.get("agent").metrics, self.llm.metrics)
        self.assertEqual(self.router.model, self.model)
        self.assertEqual(self.router.uses_responses_api(), self.llm.uses_responses_api())
        self.assertNotIn("relay_session", self.router.model_dump())

    def test_real_sdk_completion_cost_and_metadata_without_provider(self):
        # Only replace the transport. Message conversion, SDK telemetry and
        # tariff calculation all execute in the actual installed SDK.
        with patch("openhands.sdk.llm.llm.litellm_completion", return_value=self.raw()) as transport:
            first = self.router.generate(self.messages)
            second = self.router.generate(self.messages)
        self.assertIsInstance(first, LLMResponse)
        self.assertIsInstance(second, LLMResponse)
        self.assertEqual(transport.call_count, 2)
        result = self.task.finish(accepted=True, cost_complete=True)
        self.assertAlmostEqual(result["cost_usd"], 0.00004)
        self.assertEqual(result["cost_basis"], "tariff_estimate")
        ledger = self.path.read_text(encoding="utf-8")
        for private in ("private-test-prompt", "private-response", "offline-test-secret"):
            self.assertNotIn(private, ledger)

    def test_complete_local_conversation_preserves_live_session(self):
        """Exercise SDK persistence, agent loop, registry and FinishTool together."""
        agent = Agent(llm=self.router, tools=[], condenser=None,
                      include_default_tools=["FinishTool"], mcp_config={},
                      system_prompt="Complete the offline wiring fixture using finish.")
        root = Path(self.tmp.name)
        events = []
        conversation = LocalConversation(
            agent=agent, workspace=root / "workspace", persistence_dir=root / "state",
            profile_store_dir=root / "profiles", max_iteration_per_run=2,
            visualizer=None, callbacks=[events.append], plugins=[],
        )
        self.addCleanup(conversation.close)
        response = ModelResponse(
            id="offline-finish", model=self.model,
            choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "offline-finish-call", "type": "function",
                    "function": {"name": FinishTool.create()[0].name,
                                 "arguments": json.dumps({"message": "Fixture completed"})}}],
            }}], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        with patch("openhands.sdk.llm.llm.litellm_completion", return_value=response) as transport:
            conversation.send_message("private-test-prompt")
            conversation.run()
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(conversation.state.execution_status, ConversationExecutionStatus.FINISHED)
        self.assertIs(conversation.agent.llm.relay_session, self.task)
        self.assertTrue(any(isinstance(getattr(event, "action", None), FinishAction) for event in events))
        self.assertTrue(list((root / "state").rglob("base_state.json")))
        # Fixture verifier: expected SDK finish state plus observed tool event.
        # This acceptance is not a generated-text judgment or a product quality result.
        result = self.task.finish(accepted=True, cost_complete=True)
        self.assertEqual(result["logical_calls"], 1)
        self.assertAlmostEqual(result["cost_usd"], 0.00002)
        self.assertAlmostEqual(conversation.conversation_stats.get_combined_metrics().accumulated_cost,
                               result["cost_usd"])
        self.assertEqual(self.records()[-1]["kind"], "relay_task_observation")
        self.assertNotIn("private-test-prompt", self.path.read_text(encoding="utf-8"))

    def test_parameters_response_and_tool_identity_preserved(self):
        response = self.response()
        tools, options = [], {"custom": [1, 2]}
        context = LLMCallContext(session_id="host-session")
        callback = lambda token: None
        with patch.object(LLM, "completion", return_value=response) as completion:
            returned = self.router.completion(self.messages, tools=tools, on_token=callback,
                                              call_context=context, extra_body=options,
                                              add_security_risk_prediction=True)
        self.assertIs(returned, response)
        for key, expected in (("messages", self.messages), ("tools", tools), ("on_token", callback),
                              ("call_context", context), ("extra_body", options)):
            self.assertIs(completion.call_args.kwargs[key], expected)
        self.assertTrue(completion.call_args.kwargs["add_security_risk_prediction"])
        result = self.task.finish(accepted=None, cost_complete=True)
        self.assertIsNone(result["cost_usd"])

    def test_all_four_dispatch_routes_and_no_cumulative_cost(self):
        for method in ("completion", "responses", "acompletion", "aresponses"):
            with self.subTest(method=method):
                task = Relay(baseline_model=self.model).task("repair")
                router = make_openhands_router(task, {self.model: self.llm})
                response = self.response()
                asynchronous = method.startswith("a")
                mock_type = AsyncMock if asynchronous else None
                with patch.object(LLM, method, return_value=response, new_callable=mock_type) as dispatch:
                    kwargs = {"include": ["reasoning.encrypted_content"], "store": False} if "responses" in method else {}
                    result = getattr(router, method)(self.messages, **kwargs)
                    if asynchronous:
                        result = self.loop.run_until_complete(result)
                self.assertIs(result, response)
                self.assertEqual(dispatch.call_count, 1)
                for key, value in kwargs.items():
                    self.assertEqual(dispatch.call_args.kwargs[key], value)
                self.assertIsNone(task.finish(accepted=True, cost_complete=True)["cost_usd"])

    def test_exception_is_rethrown_and_recorded_once_without_content(self):
        failure = RuntimeError("private-error-content")
        with patch.object(LLM, "completion", side_effect=failure) as completion:
            with self.assertRaises(RuntimeError) as caught:
                self.router.completion(self.messages)
        self.assertIs(caught.exception, failure)
        self.assertEqual(completion.call_count, 1)
        result = self.task.finish(accepted=False, cost_complete=True)
        self.assertIsNone(result["cost_usd"])
        self.assertNotIn("private-error-content", self.path.read_text(encoding="utf-8"))

    def test_max_calls_rejects_before_sdk(self):
        task = Relay(baseline_model=self.model).task("repair", max_calls=1)
        router = make_openhands_router(task, {self.model: self.llm})
        with patch.object(LLM, "completion", return_value=self.response()) as completion:
            router.completion(self.messages)
            with self.assertRaises(ValueError):
                router.completion(self.messages)
        self.assertEqual(completion.call_count, 1)

    def test_zero_price_unknown_but_explicit_header_zero_known(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                task = Relay(baseline_model=self.model).task("repair")
                router = make_openhands_router(task, {self.model: self.llm})
                raw = self.raw()
                if explicit:
                    raw._hidden_params["additional_headers"] = {"llm_provider-x-litellm-response-cost": "0"}
                with patch.object(LLM, "completion", return_value=self.response(raw)):
                    router.completion(self.messages)
                result = task.finish(accepted=True, cost_complete=True)
                self.assertEqual(result["cost_usd"], 0 if explicit else None)

    def test_rejects_retry_fallback_alias_and_model_drift(self):
        with self.assertRaisesRegex(ValueError, "num_retries"):
            make_openhands_router(self.task, {self.model: LLM(model=self.model)})
        with self.assertRaisesRegex(ValueError, "exactly match"):
            make_openhands_router(self.task, {self.model: self.llm.model_copy(update={"model": "openai/gpt-4.1"})})
        with self.assertRaisesRegex(ValueError, "fallback_strategy"):
            make_openhands_router(self.task, {self.model: self.llm.model_copy(
                update={"fallback_strategy": FallbackStrategy(fallback_llms=["other"])}
            )})
        self.llm.num_retries = 5
        with patch.object(LLM, "completion") as completion:
            with self.assertRaisesRegex(ValueError, "num_retries"):
                self.router.completion(self.messages)
        completion.assert_not_called()

    def test_role_copy_keeps_task_with_independent_sdk_metrics(self):
        copied = self.router.model_copy(update={"usage_id": "condenser"})
        self.assertIs(copied.relay_session, self.task)
        self.assertIsNot(copied.metrics, self.router.metrics)
        self.assertEqual(copied.usage_id, "condenser")
        with patch.object(LLM, "completion", return_value=self.response()):
            copied.completion(self.messages)
        self.task.finish(accepted=None)

    def test_route_and_retry_overrides_are_rejected_before_dispatch(self):
        for method in ("completion", "responses", "acompletion", "aresponses"):
            for overrides in ({"model": "other"}, {"max_retries": 3},
                              {"api_base": "https://example.invalid"},
                              {"extra_body": {"model": "other"}}):
                with self.subTest(method=method, overrides=overrides):
                    with patch.object(LLM, method) as dispatch:
                        with self.assertRaises(ValueError):
                            returned = getattr(self.router, method)(self.messages, **overrides)
                            if method.startswith("a"):
                                self.loop.run_until_complete(returned)
                    dispatch.assert_not_called()

    def test_cancellation_finalizes_error_and_does_not_retry(self):
        with patch.object(LLM, "acompletion", new_callable=AsyncMock,
                          side_effect=asyncio.CancelledError) as completion:
            with self.assertRaises(asyncio.CancelledError):
                self.loop.run_until_complete(self.router.acompletion(self.messages))
        self.assertEqual(completion.call_count, 1)
        result = self.task.finish(accepted=None, cost_complete=True)
        self.assertEqual(result["calls"][0]["status"], "error")
        self.assertIsNone(result["cost_usd"])

    def test_missing_and_incompatible_usage_remains_unknown(self):
        raw = self.raw()
        del raw.usage
        with patch.object(LLM, "completion", return_value=self.response(raw)):
            self.router.completion(self.messages)
        result = self.task.finish(accepted=None, cost_complete=True)
        self.assertIsNone(result["usage"]["input_tokens"])
        self.assertIsNone(result["cost_usd"])

        task = Relay(baseline_model=self.model).task("repair")
        router = make_openhands_router(task, {self.model: self.llm})
        raw = self.raw(prompt_tokens=1, completion_tokens=2, total_tokens=3,
                       prompt_tokens_details={"cached_tokens": 20})
        with patch.object(LLM, "completion", return_value=self.response(raw)):
            router.completion(self.messages)
        result = task.finish(accepted=None)
        self.assertEqual(result["usage"]["input_tokens"], 1)
        self.assertIsNone(result["usage"]["cached_input_tokens"])


if __name__ == "__main__":
    unittest.main()
