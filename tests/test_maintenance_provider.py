"""No-network tests for bounded sends, complete inputs and known/unknown use."""
from copy import deepcopy
from io import BytesIO
import json
import hashlib
from urllib.error import HTTPError, URLError
import unittest
from unittest.mock import Mock, patch

from relay_router.maintenance_provider import (
    ENDPOINT, GroqMaintenanceProvider, MaintenanceProviderError, _NoRedirect,
)


MODEL = "groq/openai/gpt-oss-20b"
EDIT = "groq/openai/gpt-oss-120b"
KEY = "gsk_offline_test_credential"
MESSAGES = [{"role": "system", "content": "Devuelve únicamente JSON."},
            {"role": "user", "content": "Conserva este código:\r\n    return 'á'\r\n"}]
SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"},
    "files": {"type": "array", "items": {"type": "string", "enum": ["src/labels.py"]}}},
    "required": ["summary", "files"], "additionalProperties": False}


def response(**updates):
    return {"model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": '{"analysis":"ok","edits":[]}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}, **updates}


def provider(transport=None, **kwargs):
    return GroqMaintenanceProvider(KEY, tokenizer_mode="bytes", minimum_interval_seconds=0,
                                   transport=transport if transport is not None else Mock(return_value=response()), **kwargs)


class GroqMaintenanceProviderTests(unittest.TestCase):
    def test_complete_preserves_messages_and_sends_exact_model_once(self):
        transport = Mock(return_value=response())
        adapter = provider(transport)
        messages = deepcopy(MESSAGES)
        result = adapter.complete(MODEL, messages)
        request, timeout = transport.call_args.args
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(messages, MESSAGES)
        self.assertEqual(payload["model"], "openai/gpt-oss-20b")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["max_completion_tokens"], 2048)
        self.assertFalse(payload["stream"])
        self.assertEqual(request.get_header("Authorization"), "Bearer " + KEY)
        self.assertEqual(result["usage"], {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70})
        self.assertEqual(adapter.request_count, 1)
        self.assertEqual(transport.call_count, 1)
        self.assertNotIn(KEY, json.dumps(result))
        self.assertNotIn(KEY, json.dumps(adapter.manifest()))

    def test_strict_schema_limits_paths_and_preserves_inputs_and_usage(self):
        transport = Mock(return_value=response(choices=[{"message": {
            "content": '{"summary":"Corregir etiquetas","files":["src/labels.py"]}'}, "finish_reason": "stop"}]))
        adapter = provider(transport)
        schema, messages = deepcopy(SCHEMA), deepcopy(MESSAGES)
        result = adapter.complete_structured(MODEL, messages, schema)
        payload = json.loads(transport.call_args.args[0].data)
        expected_format = {"type": "json_schema", "json_schema": {
            "name": "relay_maintenance", "strict": True, "schema": SCHEMA}}
        self.assertEqual(payload["response_format"], expected_format)
        self.assertEqual(payload["messages"], MESSAGES)
        self.assertEqual(schema, SCHEMA)
        self.assertEqual(messages, MESSAGES)
        digest = hashlib.sha256(json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result["response_schema_sha256"], digest)
        self.assertEqual(result["response_format"], expected_format)
        self.assertEqual(result["output_mode"], "strict")
        self.assertEqual(result["usage"]["total_tokens"], 70)
        self.assertEqual(transport.call_count, 1)
        schema["properties"]["files"]["items"]["enum"].append("tests/test_labels.py")
        self.assertEqual(result["response_format"], expected_format)

    def test_prompt_json_omits_transport_format_preserves_schema_and_complete_messages(self):
        transport = Mock(return_value=response(choices=[{"message": {
            "content": '{"summary":"Corregir etiquetas","files":["src/labels.py"]}'}, "finish_reason": "stop"}]))
        adapter = provider(transport, output_mode="prompt_json")
        schema = deepcopy(SCHEMA)
        messages = deepcopy(MESSAGES)
        messages[-1]["content"] += json.dumps({"response_schema": schema})
        frozen_messages = deepcopy(messages)
        plain = adapter.preflight(MODEL, messages)
        estimate = adapter.preflight(MODEL, messages, response_schema=schema)
        self.assertEqual(estimate["additional_schema_reserve_tokens"], 0)
        self.assertEqual(estimate["response_format_utf8_bytes"], 0)
        self.assertEqual(estimate["reserved_total"], plain["reserved_total"])
        self.assertTrue(estimate["response_schema_sha256"])
        result = adapter.complete_structured(MODEL, messages, schema)
        payload = json.loads(transport.call_args.args[0].data)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["messages"], frozen_messages)
        self.assertEqual(messages, frozen_messages)
        self.assertEqual(schema, SCHEMA)
        self.assertEqual(result["response_schema_sha256"], estimate["response_schema_sha256"])
        self.assertIsNone(result["response_format"])
        self.assertEqual(result["output_mode"], "prompt_json")
        self.assertEqual(result["usage"]["total_tokens"], 70)
        self.assertEqual(adapter.manifest()["output_mode"], "prompt_json")
        self.assertIsNone(adapter.manifest()["response_format"])
        self.assertEqual(transport.call_count, 1)

    def test_prompt_json_rejects_malformed_and_fenced_json_with_usage_and_no_retry(self):
        for content in ('{"unfinished":', '```json\n{"files": []}\n```', 'PRIVATE GENERATED OUTPUT', '[]'):
            with self.subTest(content=content):
                transport = Mock(return_value=response(choices=[{
                    "message": {"content": content}, "finish_reason": "stop"}]))
                adapter = provider(transport, output_mode="prompt_json")
                with self.assertRaises(MaintenanceProviderError) as caught:
                    adapter.complete_structured(MODEL, MESSAGES, SCHEMA)
                self.assertEqual(caught.exception.category, "model_output_json")
                self.assertEqual(caught.exception.usage["total_tokens"], 70)
                self.assertEqual(caught.exception.input_estimate["output_mode"], "prompt_json")
                self.assertNotIn(content, str(caught.exception))
                self.assertEqual(transport.call_count, 1)

    def test_prompt_json_still_validates_supplied_schema_before_sending(self):
        transport = Mock()
        adapter = provider(transport, output_mode="prompt_json")
        with self.assertRaises(ValueError):
            adapter.complete_structured(MODEL, MESSAGES, {**SCHEMA, "additionalProperties": True})
        transport.assert_not_called()
        self.assertEqual(adapter.request_count, 0)
        for invalid in ("automatic", None, []):
            with self.subTest(output_mode=invalid), self.assertRaises(ValueError):
                provider(output_mode=invalid)

    def test_schema_cost_is_included_and_can_block_before_send(self):
        transport = Mock(return_value=response())
        adapter = provider(transport, request_budget=3000)
        base = adapter.preflight(MODEL, MESSAGES)
        schema = deepcopy(SCHEMA)
        schema["properties"]["summary"]["description"] = "x" * 700
        with self.assertRaises(MaintenanceProviderError) as caught:
            adapter.complete_structured(MODEL, MESSAGES, schema)
        estimate = caught.exception.input_estimate
        self.assertEqual(caught.exception.category, "request_budget")
        self.assertGreater(estimate["additional_schema_reserve_tokens"], 700)
        self.assertEqual(estimate["reserved_total"],
                         base["reserved_total"] + estimate["additional_schema_reserve_tokens"])
        self.assertFalse(estimate["input_truncated"])
        self.assertEqual(adapter.request_count, 0)
        transport.assert_not_called()

    def test_schema_is_frozen_before_transport_and_return_metadata_is_independent(self):
        schema = deepcopy(SCHEMA)
        def transport(request, timeout):
            schema["properties"]["files"]["items"]["enum"].append("private.py")
            self.assertEqual(json.loads(request.data)["response_format"]["json_schema"]["schema"], SCHEMA)
            return response()
        adapter = provider(transport)
        result = adapter.complete(MODEL, MESSAGES, response_schema=schema)
        self.assertEqual(result["response_format"]["json_schema"]["schema"], SCHEMA)

    def test_invalid_strict_schema_never_sends(self):
        invalid = [None, [], {}, {**SCHEMA, "type": "array"},
                   {**SCHEMA, "additionalProperties": True}, {**SCHEMA, "required": ["summary"]},
                   {**SCHEMA, "required": ["summary", "files", "files"]},
                   {**SCHEMA, "properties": {"summary": float("nan")}}]
        nonfinite = deepcopy(SCHEMA)
        nonfinite["properties"]["summary"]["untrusted_extension"] = float("nan")
        invalid.append(nonfinite)
        transport = Mock()
        adapter = provider(transport)
        for schema in invalid:
            with self.subTest(schema=repr(schema)), self.assertRaises(ValueError):
                adapter.complete_structured(MODEL, MESSAGES, schema)
        self.assertEqual(adapter.request_count, 0)
        transport.assert_not_called()

    def test_strict_mode_errors_preserve_usage_and_schema_budget(self):
        transport = Mock(return_value=response(choices=[{
            "message": {"content": '{"incomplete":'}, "finish_reason": "length"}]))
        adapter = provider(transport)
        with self.assertRaises(MaintenanceProviderError) as caught:
            adapter.complete_structured(MODEL, MESSAGES, SCHEMA)
        self.assertEqual(caught.exception.category, "output_truncated")
        self.assertEqual(caught.exception.usage["total_tokens"], 70)
        self.assertTrue(caught.exception.input_estimate["response_schema_sha256"])
        self.assertGreater(caught.exception.input_estimate["additional_schema_reserve_tokens"], 0)
        self.assertEqual(transport.call_count, 1)

    def test_identity_mismatch_and_truncated_output_preserve_reported_usage(self):
        variants = [response(model="different/model"),
                    response(choices=[{"message": {"content": '{"incomplete":'}, "finish_reason": "length"}])]
        for value, category in zip(variants, ("model_identity", "output_truncated")):
            transport = Mock(return_value=value)
            adapter = provider(transport)
            with self.assertRaises(MaintenanceProviderError) as caught:
                adapter.complete(MODEL, MESSAGES)
            self.assertEqual(caught.exception.category, category)
            self.assertEqual(caught.exception.usage["total_tokens"], 70)
            self.assertEqual(transport.call_count, 1)

    def test_missing_usage_is_unknown_not_zero(self):
        adapter = provider(Mock(return_value=response(usage=None)))
        self.assertEqual(adapter.complete(MODEL, MESSAGES)["usage"],
                         {"input_tokens": None, "output_tokens": None, "total_tokens": None})

    def test_malformed_json_keeps_usage_without_copying_generated_text_to_error(self):
        for content in ('PRIVATE generated text', '{"a":1,"a":2}', '{"n":NaN}', '[]'):
            adapter = provider(Mock(return_value=response(choices=[{
                "message": {"content": content}, "finish_reason": "stop"}])))
            with self.assertRaises(MaintenanceProviderError) as caught:
                adapter.complete(MODEL, MESSAGES)
            self.assertEqual(caught.exception.category, "model_output_json")
            self.assertEqual(caught.exception.usage["input_tokens"], 50)
            self.assertNotIn(content, str(caught.exception))
            self.assertNotIn("PRIVATE", json.dumps(caught.exception.details()))

    def test_request_budget_stops_before_transport_without_truncation(self):
        transport = Mock(return_value=response())
        adapter = provider(transport, request_budget=3000)
        messages = [{"role": "user", "content": "json " + "x" * 1000}]
        with self.assertRaises(MaintenanceProviderError) as caught:
            adapter.complete(MODEL, messages)
        self.assertEqual(caught.exception.category, "request_budget")
        self.assertFalse(caught.exception.input_estimate["input_truncated"])
        self.assertEqual(adapter.request_count, 0)
        transport.assert_not_called()
        self.assertEqual(len(messages[0]["content"]), 1005)

    def test_call_limit_counts_uncertain_sends_and_does_not_resend(self):
        transport = Mock(side_effect=TimeoutError("private details " + KEY))
        adapter = provider(transport, max_requests=1)
        with self.assertRaises(MaintenanceProviderError) as first:
            adapter.complete(MODEL, MESSAGES)
        self.assertEqual(first.exception.category, "timeout")
        self.assertEqual(first.exception.usage, {})
        with self.assertRaises(MaintenanceProviderError) as second:
            adapter.complete(MODEL, MESSAGES)
        self.assertEqual(second.exception.category, "call_budget")
        self.assertEqual(transport.call_count, 1)

    def test_rate_limit_preserves_retry_after_and_never_retries(self):
        error = HTTPError(ENDPOINT, 429, "SECRET " + KEY, {"retry-after": "73.5"}, BytesIO(b'private body'))
        transport = Mock(side_effect=error)
        adapter = provider(transport)
        with self.assertRaises(MaintenanceProviderError) as caught:
            adapter.complete(MODEL, MESSAGES)
        self.assertEqual(caught.exception.category, "rate_limit")
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after, 73.5)
        self.assertEqual(transport.call_count, 1)
        self.assertNotIn(KEY, str(caught.exception))
        self.assertNotIn(KEY, json.dumps(caught.exception.details()))
        self.assertIsNone(caught.exception.__context__)

    def test_http_json_validation_failure_keeps_safe_metadata_and_explicit_usage(self):
        body = {"error": {"code": "json_validate_failed", "type": "invalid_request_error",
                          "message": KEY + " " + MESSAGES[-1]["content"],
                          "failed_generation": "PRIVATE_GENERATED_SOURCE " + KEY},
                "usage": {"prompt_tokens": 51, "completion_tokens": 19, "total_tokens": 70}}
        error = HTTPError(ENDPOINT, 400, "PRIVATE_REASON " + KEY, {}, BytesIO(json.dumps(body).encode()))
        transport = Mock(side_effect=error)
        adapter = provider(transport)
        with self.assertRaises(MaintenanceProviderError) as caught:
            adapter.complete_structured(MODEL, MESSAGES, SCHEMA)
        failure = caught.exception
        self.assertEqual(failure.category, "model_output_json")
        self.assertEqual(failure.status_code, 400)
        self.assertEqual(failure.usage, {"input_tokens": 51, "output_tokens": 19, "total_tokens": 70})
        self.assertEqual(failure.details()["provider_error_code"], "json_validate_failed")
        self.assertEqual(failure.details()["provider_error_type"], "invalid_request_error")
        self.assertIn("JSON", failure.details()["provider_error_message"])
        self.assertTrue(failure.input_estimate["response_schema_sha256"])
        exported = str(failure) + json.dumps(failure.details(), ensure_ascii=False)
        for private in (KEY, MESSAGES[-1]["content"], "PRIVATE_GENERATED_SOURCE", "PRIVATE_REASON", "failed_generation"):
            self.assertNotIn(private, exported)
        self.assertIsNone(failure.__context__)
        self.assertIsNone(failure.__cause__)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(adapter.request_count, 1)

    def test_http_json_failure_without_usage_stays_unknown_and_invalid_request_stays_http(self):
        for code, category in (("json_validate_failed", "model_output_json"), ("invalid_parameter", "http")):
            with self.subTest(code=code):
                body = {"error": {"code": code, "type": "invalid_request_error", "message": "private request"}}
                transport = Mock(side_effect=HTTPError(ENDPOINT, 400, "bad", {}, BytesIO(json.dumps(body).encode())))
                adapter = provider(transport)
                with self.assertRaises(MaintenanceProviderError) as caught:
                    adapter.complete(MODEL, MESSAGES)
                self.assertEqual(caught.exception.category, category)
                self.assertEqual(caught.exception.usage, {})
                self.assertEqual(caught.exception.provider_error_code, code)
                self.assertEqual(transport.call_count, 1)

    def test_http_nested_usage_preserves_partial_unknown_counts(self):
        body = {"error": {"type": "invalid_request_error", "usage": {
            "prompt_tokens": 12, "completion_tokens": False, "total_tokens": -1},
            "message": KEY, "failed_generation": MESSAGES[-1]["content"]}}
        adapter = provider(Mock(side_effect=HTTPError(ENDPOINT, 400, "bad", {}, BytesIO(json.dumps(body).encode()))))
        with self.assertRaises(MaintenanceProviderError) as caught:
            adapter.complete(MODEL, MESSAGES)
        self.assertEqual(caught.exception.usage, {"input_tokens": 12, "output_tokens": None, "total_tokens": None})
        self.assertEqual(caught.exception.category, "http")
        self.assertNotIn(KEY, json.dumps(caught.exception.details()))

    def test_http_invalid_or_oversized_bodies_never_leak_or_parse_unknown_identifiers(self):
        class RecordingBody(BytesIO):
            def __init__(self, value):
                super().__init__(value)
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return super().read(size)

        unknown = json.dumps({"error": {"code": KEY, "type": "PRIVATE_INPUT_LABEL",
                                       "message": MESSAGES[-1]["content"], "failed_generation": KEY}}).encode()
        for value in (b"PRIVATE invalid body " + KEY.encode(), b"\xff", unknown,
                      b'{"error":{"code":"json_validate_failed","code":"invalid_parameter"}}',
                      b"x" * (64 * 1024 + 100)):
            with self.subTest(body_length=len(value)):
                stream = RecordingBody(value)
                transport = Mock(side_effect=HTTPError(ENDPOINT, 400, KEY, {}, stream))
                adapter = provider(transport)
                with self.assertRaises(MaintenanceProviderError) as caught:
                    adapter.complete(MODEL, MESSAGES)
                details = caught.exception.details()
                self.assertEqual(caught.exception.category, "http")
                self.assertEqual(caught.exception.usage, {})
                self.assertNotIn("provider_error_code", details)
                self.assertNotIn("provider_error_type", details)
                self.assertNotIn(KEY, json.dumps(details))
                self.assertNotIn("PRIVATE", json.dumps(details))
                self.assertIsNone(caught.exception.__context__)
                self.assertEqual(stream.read_sizes, [64 * 1024 + 1])
                self.assertTrue(stream.closed)
                self.assertEqual(transport.call_count, 1)

    def test_transport_errors_do_not_expose_credentials_or_exception_context(self):
        for error in (URLError(KEY), RuntimeError(KEY)):
            adapter = provider(Mock(side_effect=error))
            with self.assertRaises(MaintenanceProviderError) as caught:
                adapter.complete(MODEL, MESSAGES)
            self.assertEqual(caught.exception.category, "transport")
            self.assertNotIn(KEY, str(caught.exception))
            self.assertIsNone(caught.exception.__context__)

    def test_per_model_pacing_with_real_clock_progress_and_bounded_sleeps(self):
        now = [100.]
        sleeps = []
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        def transport(request, timeout):
            return response(model=json.loads(request.data)["model"])
        on_wait = Mock()
        adapter = GroqMaintenanceProvider(KEY, transport=transport, tokenizer_mode="bytes",
                   minimum_interval_seconds=61, clock=lambda: now[0], sleep=sleep, on_wait=on_wait)
        adapter.complete(MODEL, MESSAGES)
        adapter.complete(EDIT, MESSAGES)
        result = adapter.complete(MODEL, MESSAGES)
        self.assertEqual(sleeps, [60., 1.])
        self.assertEqual(result["quota_wait_seconds"], 61.)
        on_wait.assert_called_once_with(61.)

    def test_invalid_messages_and_models_stop_before_send(self):
        transport = Mock()
        adapter = provider(transport)
        invalid_messages = [None, [], [{"role": "tool", "content": "foo"}],
                            [{"role": "user", "content": {"value": 2}}],
                            [{"role": "user", "content": "foo", "tool_calls": []}]]
        for messages in invalid_messages:
            with self.assertRaises(ValueError):
                adapter.complete(MODEL, messages)
        with self.assertRaises(ValueError):
            adapter.complete("other/provider", MESSAGES)
        transport.assert_not_called()

    def test_missing_tokenizer_is_explicit_byte_fallback_without_network(self):
        adapter = GroqMaintenanceProvider(KEY, transport=Mock(return_value=response()), minimum_interval_seconds=0)
        with patch("routerbench.bfcl_tokens.GPTOSSTokenEstimator", side_effect=FileNotFoundError), patch("socket.create_connection") as network:
            estimate = adapter.preflight(MODEL, MESSAGES)
        self.assertEqual(estimate["method"], "utf8_bytes_conservative_estimate")
        network.assert_not_called()

    def test_corrupt_local_tokenizer_is_not_hidden_by_fallback(self):
        adapter = GroqMaintenanceProvider(KEY, transport=Mock(return_value=response()))
        with patch("routerbench.bfcl_tokens.GPTOSSTokenEstimator", side_effect=ValueError("hash mismatch")), patch("socket.create_connection") as network:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                adapter.preflight(MODEL, MESSAGES)
        network.assert_not_called()

    def test_redirects_are_refused(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "moved", {}, "https://other.example"))


if __name__ == "__main__":
    unittest.main()
