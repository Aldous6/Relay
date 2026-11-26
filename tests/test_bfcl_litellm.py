from copy import deepcopy
import unittest
from unittest.mock import patch

from relay_router.errors import BackendActionError, BackendContextError, BackendIdentityError, BackendTransportError, BackendResponseError
from routerbench.bfcl_harness import BFCLRequest, BFCLReply, BFCLState
from routerbench.bfcl_litellm import BFCLLiteLLMGenerator, BFCLPacedGenerator

CATALOG = {"fixture/small": {"returned_model_ids": ["small"], "context_window": 131072, "api_key_env": "BFCL_TEST_KEY"}}
OPTIONS = {"max_output_tokens": 200, "temperature": 0, "timeout": 30}


def request():
    state = BFCLState(turn_index=1, step=2, attempt=0, previous_errors=(), infrastructure_errors=(),
        infrastructure_attempt=0, failed_tiers=(), remaining_calls=21, context_utf8_bytes=0)
    return BFCLRequest(({"role": "system", "content": "fixture"}, {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "done"}, {"role": "user", "content": "next turn"}),
        ({"type": "function", "function": {"name": "lookup", "description": "fixture", "parameters": {
            "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},), state)


def response(*, content=None, calls=None, finish="tool_calls", model="small"):
    return {"model": model, "choices": [{"finish_reason": finish,
        "message": {"role": "assistant", "content": content, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}}


class FakeClock:
    def __init__(self): self.time = 0.
    def __call__(self): return self.time
    def sleep(self, seconds): self.time += seconds


class BFCLTransportTests(unittest.TestCase):
    def test_estimated_reservation_uses_explicit_frozen_supported_counter(self):
        class Estimator:
            supported_models = tuple(CATALOG)
            def __init__(self): self.version = 1
            def manifest(self): return {"version": self.version}
            def estimate(self, request): return {"estimated_input_tokens": 100}
        counter, clock, seen = Estimator(), FakeClock(), []
        def completion(**kwargs):
            seen.append(kwargs)
            return response(content="done", finish="stop")
        base = BFCLLiteLLMGenerator(CATALOG, {**OPTIONS, "reasoning_effort": "medium"},
            completion=completion, input_estimator=counter)
        paced = BFCLPacedGenerator(base, min_interval_seconds=0, max_request_budget=812,
            max_output_tokens=200, tokens_per_minute=812, clock=clock, sleep=clock.sleep)
        with patch.dict("os.environ", {"BFCL_TEST_KEY": "fixture-only"}):
            reply = paced("fixture/small", request())
        self.assertEqual(seen[0]["reasoning_effort"], "medium")
        metadata = reply.usage["provider_metadata"]["quota_input_estimate"]
        self.assertEqual(metadata["input_budget_units"], 100)
        self.assertFalse(metadata["actual_provider_tokens"])
        self.assertEqual(base.manifest()["quota_input_estimator"], {"version": 1})
        counter.version = 2
        with self.assertRaisesRegex(ValueError, "frozen input estimator changed"):
            paced("fixture/small", request())
        self.assertEqual(len(seen), 1)
        counter.supported_models = ()
        with self.assertRaisesRegex(ValueError, "support every selected model"):
            BFCLLiteLLMGenerator(CATALOG, OPTIONS, input_estimator=counter)

    def test_prepare_is_offline_and_preserves_multiturn_payload(self):
        with patch.dict("os.environ", {}, clear=True), patch("socket.create_connection") as network:
            generator = BFCLLiteLLMGenerator(CATALOG, OPTIONS)
            manifest = generator.manifest()
            payload = generator.request_payload(request())
        network.assert_not_called()
        self.assertEqual(payload["messages"], list(request().messages))
        self.assertEqual(payload["tools"], list(request().tools))
        self.assertTrue(payload["parallel_tool_calls"])
        self.assertIsNone(manifest["fallbacks"])
        self.assertFalse(manifest["input_truncation_allowed"])

    def test_one_completion_preserves_batch_order_and_invalid_argument_text(self):
        calls = [{"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"name":"one"}'}},
                 {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": '{"name":1,"name":2}'}}]
        seen = []
        def completion(**kwargs):
            seen.append(kwargs)
            return response(calls=calls)
        generator = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=completion)
        with patch.dict("os.environ", {"BFCL_TEST_KEY": "fixture-only"}):
            reply = generator("fixture/small", request())
        self.assertEqual(list(reply.tool_calls), calls)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["num_retries"], 0)
        self.assertEqual(seen[0]["max_retries"], 0)
        self.assertIsNone(seen[0]["fallbacks"])
        self.assertFalse(seen[0]["caching"])
        self.assertNotIn("response_format", seen[0])
        self.assertEqual(reply.usage["total_tokens"], 100)
        self.assertIsNone(reply.cost_usd)

    def test_plain_text_ends_turn_but_exhausted_empty_response_is_semantic(self):
        with patch.dict("os.environ", {"BFCL_TEST_KEY": "fixture-only"}):
            good = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=lambda **kw: response(content="done", finish="stop"))
            self.assertEqual(good("fixture/small", request()).content, "done")
            exhausted = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=lambda **kw: response(finish="length"))
            with self.assertRaises(BackendActionError) as caught:
                exhausted("fixture/small", request())
            self.assertEqual(caught.exception.usage["total_tokens"], 100)
            self.assertEqual(caught.exception.usage["provider_metadata"]["finish_reason"], "length")

    def test_transport_and_identity_are_not_semantic_and_do_not_retry(self):
        seen = []
        def failed(**kwargs):
            seen.append(kwargs)
            raise ConnectionError("provider secret is not exposed")
        with patch.dict("os.environ", {"BFCL_TEST_KEY": "fixture-only"}):
            bad = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=failed)
            with self.assertRaises(BackendTransportError) as caught:
                bad("fixture/small", request())
            self.assertNotIn("provider secret", str(caught.exception))
            mismatch = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=lambda **kw: response(model="different"))
            with self.assertRaises(BackendIdentityError):
                mismatch("fixture/small", request())
        self.assertEqual(len(seen), 1)

    def test_missing_message_or_bad_envelope_is_provider_contract_with_usage(self):
        fixtures = []
        for message in (None, {}, "bad", {"role": "tool", "content": "wrong role"}):
            value = response()
            value["choices"][0]["message"] = message
            fixtures.append(value)
        valid = {"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        for field in ("id", "type", "arguments"):
            call = deepcopy(valid)
            if field == "arguments":
                call["function"][field] = {}
            else:
                call[field] = None
            fixtures.append(response(calls=[call]))
        fixtures.append(response(calls=[valid, valid]))
        for value in fixtures:
            with self.subTest(value=value), patch.dict("os.environ", {"BFCL_TEST_KEY": "fixture-only"}):
                generator = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=lambda **kw: value)
                with self.assertRaises(BackendResponseError) as caught:
                    generator("fixture/small", request())
                self.assertEqual(caught.exception.usage["total_tokens"], 100)

    def test_quota_over_bound_never_sends_and_known_usage_replaces_reservation(self):
        calls, clock = [], FakeClock()
        def completion(**kwargs):
            calls.append(kwargs)
            return response(content="done", finish="stop")
        base = BFCLLiteLLMGenerator(CATALOG, OPTIONS, completion=completion)
        too_small = BFCLPacedGenerator(base, min_interval_seconds=0, max_request_budget=100,
            max_output_tokens=200, tokens_per_minute=100, clock=clock, sleep=clock.sleep)
        with self.assertRaises(BackendContextError):
            too_small("fixture/small", request())
        self.assertEqual(calls, [])
        paced = BFCLPacedGenerator(base, min_interval_seconds=2.1, max_request_budget=8000,
            max_output_tokens=200, tokens_per_minute=8000, clock=clock, sleep=clock.sleep)
        with patch.dict("os.environ", {"BFCL_TEST_KEY": "fixture-only"}):
            first = paced("fixture/small", request())
            second = paced("fixture/small", request())
        self.assertIsInstance(first, BFCLReply)
        self.assertEqual(paced._reservations["fixture/small"][0][1], 100)
        self.assertAlmostEqual(second.usage["provider_metadata"]["quota_wait_seconds"], 2.1)


if __name__ == "__main__":
    unittest.main()
