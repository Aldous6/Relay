"""Exercise the real OpenHands/LiteLLM stack; replace only the HTTP transport."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
AVAILABLE = importlib.util.find_spec("openhands") is not None
if AVAILABLE:
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")):
        import httpx
        from openhands.sdk import Agent, LLM
        from relay_router.cascade import SpendBudget, run_cascade
        from relay_router.cascade_verifier import CheckCommand, CommandVerifier
        from relay_router.integrations.groq_recovery import BASE_URL, GroqDispatch
        from relay_router.integrations.openhands_cascade import make_openhands_attempt
        import openhands.sdk.llm.llm as sdk_llm
        from run_cascade_validation import make_edit_tool


CHEAP, STRONG = "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"


@unittest.skipUnless(AVAILABLE, "OpenHands integration environment required")
class GroqRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Provider network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        self.seen = []
        self.cheap_error_code = "tool_use_failed"
        self.error_usage = None

    def llm(self, model):
        rate = .075 if model == CHEAP else .15
        return LLM(model=model, api_key="offline-fixture", base_url=BASE_URL, num_retries=0,
                   api_mode="chat", max_output_tokens=1024, timeout=10, caching_prompt=False,
                   input_cost_per_token=rate / 1e6, output_cost_per_token=4 * rate / 1e6)

    def http(self, request):
        body = json.loads(request.content)
        model = "groq/" + body["model"]
        self.seen.append(model)
        self.assertEqual(str(request.url), BASE_URL + "/chat/completions")
        self.assertEqual(body["service_tier"], "on_demand")
        if model == CHEAP:
            payload = {"error": {"code": self.cheap_error_code, "message": "fixture",
                                 "failed_generation": "private-output-do-not-log"}}
            if self.error_usage:
                payload["usage"] = self.error_usage
            return httpx.Response(400, headers={"x-request-id": "req_recoveryfixture"}, json=payload)
        edit = next(tool["function"]["name"] for tool in body["tools"]
                    if "source" in tool["function"]["parameters"].get("properties", {}))
        return httpx.Response(200, json={"id": "chatcmpl-offline", "object": "chat.completion", "created": 1,
            "model": body["model"], "service_tier": "on_demand", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call_offline", "type": "function",
                    "function": {"name": edit, "arguments": json.dumps({"source": "def add(a, b):\n    return a + b\n"})}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

    def run_fixture(self, max_cost=.05, max_calls=6):
        source = self.root / "snapshot"
        source.mkdir()
        (source / "solution.py").write_text("def add(a, b):\n    return 0\n")
        (source / "checks.py").write_text("from solution import add\nassert add(2, 3) == 5\n")
        verifier = CommandVerifier.capture(source, checks=[CheckCommand("tests", (sys.executable, "-B", "checks.py"))],
                                           protected_files=["checks.py"])

        def factory(ctx):
            return Agent(llm=self.llm(ctx.model), tools=[make_edit_tool(ctx.workspace, "assert add(2, 3) == 5\n",
                         stop_on_public_pass=True)], condenser=None, include_default_tools=["FinishTool"],
                         system_prompt="Use the edit tool to repair add.")

        attempt = make_openhands_attempt(factory, "Repair add.", reservation_usd_by_model={CHEAP: .005, STRONG: .01},
                                         tools_are_local=True, recover_tool_use_failed=True)
        with patch.object(httpx.HTTPTransport, "handle_request", side_effect=self.http):
            result = run_cascade(snapshot_dir=source, output_dir=self.root / "run", task_id="wire-recovery",
                                 candidate_model=CHEAP, baseline_model=STRONG, run_attempt=attempt, verifier=verifier,
                                 budget=SpendBudget(max_cost, max_calls), mode="experimental", allow_experimental=True)
        self.assertNotIn("private-output-do-not-log", json.dumps(result))
        return result

    def test_real_sdk_error_reserves_its_full_bound_and_recovers(self):
        result = self.run_fixture()
        self.assertEqual(self.seen, [CHEAP, STRONG])
        self.assertEqual(result["selected_model"], STRONG)
        acc = result["accounting"]
        self.assertFalse(acc["cost_complete"])
        self.assertIsNone(acc["cost_usd"])
        cheap, strong = acc["calls"]
        self.assertEqual(cheap["provider_request_id"], "req_recoveryfixture")
        self.assertEqual((cheap["http_status"], cheap["provider_error_code"]), (400, "tool_use_failed"))
        self.assertIsNone(cheap["input_tokens"])
        self.assertIsNone(cheap["cost_usd"])
        ceiling = (131_072 * .075 + 1024 * .30) / 1_000_000
        self.assertAlmostEqual(cheap["cost_upper_bound_usd"], ceiling)
        self.assertAlmostEqual(acc["cost_upper_bound_usd"], ceiling + strong["cost_usd"])
        self.assertEqual(acc["known_cost_subtotal_usd"], strong["cost_usd"])
        self.assertEqual(acc["cost_bound_basis"], "tariff_estimate")
        journal = [json.loads(line) for line in (self.root / "run/transport.jsonl").read_text().splitlines()]
        self.assertEqual(len(journal), 4)
        self.assertEqual(journal[0]["tariff_bound"], cheap["tariff_bound"])
        self.assertAlmostEqual(journal[0]["reservation_usd"], ceiling)

    def test_full_upper_bound_blocks_unaffordable_fallback(self):
        result = self.run_fixture(max_cost=.03)
        self.assertEqual(self.seen, [CHEAP])
        self.assertFalse(result["operational_passed"])
        self.assertEqual(result["accounting"]["http_calls"], 1)

    def test_verified_error_usage_reconciles_cost(self):
        self.error_usage = {"prompt_tokens": 70, "completion_tokens": 15, "total_tokens": 85}
        result = self.run_fixture()
        self.assertEqual(self.seen, [CHEAP, STRONG])
        self.assertTrue(result["accounting"]["cost_complete"])
        self.assertAlmostEqual(result["accounting"]["cost_usd"], (70 * .075 + 15 * .3 + 100 * .15 + 20 * .6) / 1e6)

    def test_unrelated_error_without_usage_blocks_recovery(self):
        self.cheap_error_code = "other"
        result = self.run_fixture()
        self.assertEqual(self.seen, [CHEAP])
        self.assertEqual(result["reason"], "AccountingIncomplete")
        self.assertIsNone(result["accounting"]["cost_upper_bound_usd"])

    def test_invalid_error_usage_is_unknown_and_call_cap_still_applies(self):
        self.error_usage = {"prompt_tokens": 70, "completion_tokens": 15, "total_tokens": 1}
        result = self.run_fixture(max_calls=1)
        self.assertEqual(self.seen, [CHEAP])
        self.assertEqual(result["reason"], "BudgetExceeded")
        self.assertIsNone(result["accounting"]["cost_usd"])

    def test_http_contract_rejects_builtin_tool_higher_tier_or_hidden_resend(self):
        wire = GroqDispatch(self.llm(CHEAP), {}, 1024)
        url = BASE_URL + "/chat/completions"
        body = {"model": CHEAP.removeprefix("groq/"), "service_tier": "on_demand", "max_tokens": 1024}
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(400, json={}))) as client:
            with wire.observe(lambda: "simulated-call-id"):
                for changes in ({"tools": [{"type": "browser_search"}]}, {"service_tier": "performance"},
                                {"max_tokens": 1025}, {"n": 2}, {"web_search_options": {}}):
                    with self.assertRaises(ValueError):
                        client.post(url, json={**body, **changes})
                client.post(url, json=body)
                with self.assertRaises(ValueError):
                    client.post(url, json=body)

    def test_local_refusal_does_not_count_as_an_http_call(self):
        with patch.object(GroqDispatch, "_validate_request", side_effect=ValueError("local refusal")):
            result = self.run_fixture()
        self.assertEqual(self.seen, [])
        self.assertFalse(result["operational_passed"])
        self.assertTrue(result["accounting"]["http_calls_complete"])
        self.assertEqual(result["accounting"]["http_calls"], 0)
        self.assertEqual(result["accounting"]["cost_usd"], 0)

    def test_unobserved_sdk_path_blocks_without_asserting_zero_cost(self):
        with patch.object(sdk_llm, "litellm_completion", side_effect=RuntimeError("opaque")):
            result = self.run_fixture()
        accounting = result["accounting"]
        self.assertEqual(self.seen, [])
        self.assertEqual(accounting["http_calls"], 0)
        self.assertFalse(accounting["http_calls_complete"])
        self.assertEqual(len(accounting["unobserved_transport"]), 1)
        self.assertIsNone(accounting["cost_usd"])
        self.assertIsNone(accounting["cost_upper_bound_usd"])
        self.assertEqual(result["reason"], "AccountingIncomplete")

    def test_lost_network_response_keeps_the_dispatched_call_unknown(self):
        with patch.object(self, "http", side_effect=httpx.ReadTimeout("offline connection lost")):
            result = self.run_fixture()
        self.assertEqual(result["accounting"]["http_calls"], 1)
        self.assertTrue(result["accounting"]["http_calls_complete"])
        self.assertIsNone(result["accounting"]["cost_usd"])
        self.assertIsNone(result["accounting"]["cost_upper_bound_usd"])
        self.assertEqual(result["reason"], "AccountingIncomplete")


if __name__ == "__main__":
    unittest.main()
