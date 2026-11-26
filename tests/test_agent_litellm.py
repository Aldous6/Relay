"""Offline contracts: one call, no hidden retries, safe errors and known usage."""
from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from relay_router.agent import AgentRequest, AgentState, ModelReply, parse_action
from relay_router.errors import (BackendActionError, BackendContextError, BackendHTTPError,
                                 BackendIdentityError, BackendRateLimitError,
                                 BackendResponseError, BackendTimeoutError,
                                 BackendTransportError)
from routerbench.agent_context import render_messages
from routerbench.agent_litellm import LiteLLMEpisodeGenerator, _default_completion, _provider_error


MODEL = "groq/vendor/declared-model"
CATALOG = {MODEL: {"returned_model_ids": ["vendor/declared-model"],
                   "context_window": 32768, "api_key_env": "TEST_LITELLM_KEY"}}
OPTIONS = {"max_output_tokens": 192, "temperature": 0, "timeout": 30}


def request(prompt="Tarea actual\n    conserva indentación"):
    return AgentRequest(prompt, ({"role": "tool", "content": {"name": "lookup", "ok": True,
                                                            "observation": {"value": 3}}},),
                        AgentState(1, 0, (), ("lookup",), 4))


def response(content='{"type":"final","answer":"done"}', **updates):
    return {"model": "vendor/declared-model", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                      "prompt_tokens_details": {"cached_tokens": 5}}, **updates}


class LiteLLMAdapterTests(unittest.TestCase):
    @patch.dict(os.environ, {'TEST_LITELLM_KEY': 'offline-test-secret'})
    def test_native_translation_keeps_malformed_actions_for_harness_and_usage(self):
        definition = {'name': 'lookup', 'description': 'Read an item', 'parameters': {'type': 'object',
                      'properties': {'key': {'type': 'string'}}, 'required': ['key']}}
        native_request = AgentRequest('task', ({'role': 'tool', 'content': {'name': 'environment',
            'tools': [definition], 'observation': {}}},), AgentState(0, 0, (), ('lookup',), 2))
        def generated(name='lookup', arguments='{"key":"ok"}', count=1):
            call = {'type': 'function', 'function': {'name': name, 'arguments': arguments}}
            return response(choices=[{'message': {'content': None, 'tool_calls': [call] * count},
                                      'finish_reason': 'tool_calls'}])
        completion = Mock(return_value=generated())
        adapter = LiteLLMEpisodeGenerator(CATALOG, {**OPTIONS, 'action_format': 'native_tools'}, completion=completion)
        reply = adapter(MODEL, native_request)
        self.assertEqual(parse_action(reply.content, ('lookup',)), {'type': 'tool', 'name': 'lookup', 'arguments': {'key': 'ok'}})
        args = completion.call_args.kwargs
        self.assertNotIn('response_format', args)
        self.assertEqual(args['tool_choice'], 'auto')
        self.assertFalse(args['parallel_tool_calls'])
        self.assertEqual(len(args['tools']), 1)
        self.assertGreater(adapter.request_input_bytes(native_request), sum(len(m['content'].encode()) for m in args['messages']))
        self.assertEqual(adapter.manifest()['format'], 'native_tools')
        invalid = [generated(count=2), generated(count=0), generated('unknown'),
                   generated(arguments='{"key":"one","key":"two"}'),
                   generated(arguments='{"key":NaN}'), generated(arguments='['),
                   generated(arguments='[]')]
        for item in invalid:
            completion.reset_mock()
            completion.return_value = item
            result = adapter(MODEL, native_request)
            completion.assert_called_once()
            self.assertEqual(result.usage['input_tokens'], 100)
            self.assertEqual(result.usage['output_tokens'], 20)
            with self.assertRaises(ValueError):
                parse_action(result.content, ('lookup',))
        final_text = '  exact final answer\n'
        completion.return_value = response(final_text)
        result = adapter(MODEL, native_request)
        self.assertEqual(parse_action(result.content, ('lookup',)), {'type': 'final', 'answer': final_text})

    @patch.dict(os.environ, {'TEST_LITELLM_KEY': 'gsk_private_credential_123456'})
    def test_diagnostic_reason_redacts_key_and_excludes_body_and_normal_journal(self):
        error = RuntimeError('SECRET')
        error.status_code = 400
        error.body = {'error': {'message': 'Unsupported value with gsk_private_credential_123456',
                                'type': 'invalid_request_error', 'failed_generation': 'PRIVATE OUTPUT'}}
        completion = Mock(side_effect=error)
        adapter = LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion, capture_error_reason=True)
        with self.assertRaises(BackendHTTPError) as caught:
            adapter(MODEL, request())
        self.assertEqual(caught.exception.operator_reason, 'Unsupported value with [REDACTED]')
        self.assertNotIn('PRIVATE OUTPUT', caught.exception.operator_reason)
        self.assertNotIn('operator_reason', caught.exception.details())
        adapter = LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)
        with self.assertRaises(BackendHTTPError) as caught:
            adapter(MODEL, request())
        self.assertFalse(hasattr(caught.exception, 'operator_reason'))

    def test_rejection_records_only_allowlisted_codes_never_messages_or_failed_generation(self):
        error = RuntimeError('SECRET reflected request')
        error.status_code = 400
        error.body = {'error': {'type': 'invalid_request_error', 'code': 'json_validate_failed',
                                'message': 'SECRET', 'failed_generation': 'SECRET'}}
        failure = _provider_error(error)
        details = failure.details()
        self.assertIsInstance(failure, BackendActionError)
        self.assertEqual(details['diagnostic']['provider_error_code'], 'json_validate_failed')
        self.assertEqual(details['diagnostic']['provider_error_type'], 'invalid_request_error')
        self.assertNotIn('SECRET', json.dumps(details))
        error.body['error']['code'] = 'SECRET'
        self.assertNotIn('provider_error_code', _provider_error(error).details()['diagnostic'])

    def test_only_known_generation_rejections_are_semantic(self):
        for status, code, semantic in ((400, 'tool_use_failed', True), (400, 'json_validate_failed', True),
                                       (400, 'output_parse_failed', True),
                                       (400, 'unsupported_parameter', False), (401, 'tool_use_failed', False),
                                       (500, 'tool_use_failed', False), (401, 'output_parse_failed', False),
                                       (500, 'output_parse_failed', False), (400, None, False)):
            error = RuntimeError('not persisted')
            error.status_code = status
            error.body = {'error': {'code': code}}
            failure = _provider_error(error)
            self.assertEqual(isinstance(failure, BackendActionError), semantic)

    def test_output_parse_failure_uses_exact_code_not_reflected_message(self):
        # Captured provider diagnostic: BFCL development v2, one isolated call.
        # The fixture contains no model output or real request/credential.
        error = RuntimeError('PRIVATE exception text')
        error.status_code = 400
        error.body = {'error': {'code': 'output_parse_failed', 'type': 'invalid_request_error',
            'message': 'Parsing failed. The model generated output that could not be parsed.',
            'failed_generation': 'PRIVATE generated output'}}
        failure = _provider_error(error)
        self.assertIsInstance(failure, BackendActionError)
        self.assertEqual(failure.category, 'model_action')
        self.assertFalse(failure.retryable)
        self.assertEqual(failure.details()['diagnostic']['provider_error_code'], 'output_parse_failed')
        self.assertNotIn('PRIVATE', json.dumps(failure.details()))
        self.assertNotIn('Parsing failed', json.dumps(failure.details()))
        del error.body['error']['code']
        self.assertIsInstance(_provider_error(error), BackendHTTPError)
        error.body['error']['code'] = 'OUTPUT_PARSE_FAILED'
        self.assertIsInstance(_provider_error(error), BackendHTTPError)

    @patch("routerbench.agent_litellm.importlib.import_module")
    def test_constructor_and_manifest_are_pure_independent_and_secret_free(self, load):
        metadata, options = deepcopy(CATALOG), deepcopy(OPTIONS)
        with patch.dict(os.environ, {"TEST_LITELLM_KEY": "secret-never-in-manifest"}):
            adapter = LiteLLMEpisodeGenerator(metadata, options)
            metadata[MODEL]["returned_model_ids"].append("tampered")
            options["timeout"] = 900
            manifest = adapter.manifest()
            self.assertEqual(manifest["models_metadata"], CATALOG)
            self.assertEqual(manifest["options"], OPTIONS)
            self.assertNotIn("secret-never-in-manifest", json.dumps(manifest))
            manifest["models_metadata"][MODEL]["context_window"] = 1
            self.assertEqual(adapter.manifest()["models_metadata"], CATALOG)
        load.assert_not_called()

    @patch.dict(os.environ, {"TEST_LITELLM_KEY": "offline-test-secret"})
    def test_exactly_one_call_and_resilience_is_disabled(self):
        completion = Mock(return_value=response())
        original = request()
        result = LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)(MODEL, original)
        completion.assert_called_once()
        payload = completion.call_args.kwargs
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["messages"], render_messages(original))
        self.assertEqual(payload["api_key"], "offline-test-secret")
        self.assertEqual(payload["num_retries"], 0)
        self.assertEqual(payload["max_retries"], 0)
        self.assertIsNone(payload["fallbacks"])
        self.assertEqual(payload["context_window_fallback_dict"], {})
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["caching"])
        self.assertFalse(payload["drop_params"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIsInstance(result, ModelReply)
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertEqual(result.usage["output_tokens"], 20)
        self.assertEqual(result.usage["cached_input_tokens"], 5)
        self.assertIsNone(result.cost_usd)
        self.assertNotIn("offline-test-secret", json.dumps(result.to_dict()))

    @patch.dict(os.environ, {"TEST_LITELLM_KEY": "offline-test-secret"})
    def test_unknown_cache_and_sdk_estimates_are_not_converted_to_invoice(self):
        generated = response(usage={"prompt_tokens": 7, "completion_tokens": 3, "cost": .123},
                             _hidden_params={"response_cost": .9})
        completion = Mock(return_value=generated)
        result = LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)(MODEL, request())
        self.assertEqual(result.usage["total_tokens"], 10)
        self.assertIsNone(result.usage["cached_input_tokens"])
        self.assertIsNone(result.cost_usd)

    @patch.dict(os.environ, {"TEST_LITELLM_KEY": "offline-test-secret"})
    def test_empty_or_malformed_action_and_truncation_preserve_usage(self):
        for content in ("", '{"type":"tool"', "not JSON"):
            generated = response(content)
            generated["choices"][0]["finish_reason"] = "length"
            completion = Mock(return_value=generated)
            with self.subTest(content=content):
                result = LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)(MODEL, request())
                self.assertEqual(result.content, content)
                self.assertEqual(result.usage["output_tokens"], 20)
                self.assertEqual(result.usage["provider_metadata"]["finish_reason"], "length")
                completion.assert_called_once()

    @patch.dict(os.environ, {"TEST_LITELLM_KEY": "offline-test-secret"})
    def test_identity_or_response_contract_error_retains_counter_coverage(self):
        cases = [(response(model="other-model"), BackendIdentityError),
                 (response(choices=[]), BackendResponseError),
                 (response(choices=[{"message": {"content": None}}]), BackendResponseError)]
        for value, expected in cases:
            with self.subTest(error=expected.__name__):
                completion = Mock(return_value=value)
                with self.assertRaises(expected) as caught:
                    LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)(MODEL, request())
                self.assertEqual(caught.exception.usage["input_tokens"], 100)
                completion.assert_called_once()

    @patch.dict(os.environ, {"TEST_LITELLM_KEY": "offline-test-secret"})
    def test_safe_exception_mapping_does_not_echo_provider_error_or_retry(self):
        cases = [(TimeoutError("private-reflected-request"), BackendTimeoutError),
                 (ConnectionError("private-reflected-request"), BackendTransportError),
                 (RuntimeError("private-reflected-request"), BackendResponseError)]
        for status, expected in ((401, BackendHTTPError), (403, BackendHTTPError),
                                 (429, BackendRateLimitError), (500, BackendHTTPError), (408, BackendTimeoutError)):
            error = RuntimeError("private-reflected-request")
            error.status_code = status
            error.response = SimpleNamespace(headers={"retry-after": "2.5"})
            cases.append((error, expected))
        for error, expected in cases:
            with self.subTest(error=expected.__name__, status=getattr(error, "status_code", None)):
                completion = Mock(side_effect=error)
                with self.assertRaises(expected) as caught:
                    LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)(MODEL, request())
                self.assertNotIn("private-reflected-request", str(caught.exception))
                self.assertNotIn("offline-test-secret", str(caught.exception))
                completion.assert_called_once()

    def test_invalid_request_missing_key_or_context_never_calls_completion(self):
        completion = Mock(return_value=response())
        adapter = LiteLLMEpisodeGenerator(CATALOG, OPTIONS, completion=completion)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(BackendHTTPError):
            adapter(MODEL, request())
        with self.assertRaises(BackendContextError):
            adapter(MODEL, request("x" * 40000))
        with self.assertRaises(ValueError):
            adapter("other-model", request())
        with self.assertRaises(ValueError):
            adapter(MODEL, "not a request")
        completion.assert_not_called()

    def test_invalid_options_catalog_or_implicit_routing_are_rejected(self):
        invalid = [{**OPTIONS, "temperature": .1}, {**OPTIONS, "temperature": True},
                   {**OPTIONS, "timeout": float("nan")}, {**OPTIONS, "max_output_tokens": True},
                   {**OPTIONS, "max_output_tokens": 40000}, {**OPTIONS, "fallbacks": ["other"]}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                LiteLLMEpisodeGenerator(CATALOG, options)
        for metadata in ({}, {"openrouter/auto": CATALOG[MODEL]},
                         {MODEL: {**CATALOG[MODEL], "returned_model_ids": []}},
                         {MODEL: {**CATALOG[MODEL], "api_key_env": "value-not-env-name"}},
                         {MODEL: {**CATALOG[MODEL], "api_base": "https://secret@example.test/v1"}}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                LiteLLMEpisodeGenerator(metadata, OPTIONS)

    def test_default_loader_disables_telemetry_and_refuses_global_hidden_routing(self):
        module = SimpleNamespace(completion=Mock())
        with patch.dict(os.environ, {}, clear=True), patch("routerbench.agent_litellm.importlib.import_module", return_value=module) as load:
            self.assertIs(_default_completion(), module.completion)
            self.assertEqual(os.environ["LITELLM_TELEMETRY"], "False")
            self.assertEqual(os.environ["LITELLM_LOCAL_MODEL_COST_MAP"], "True")
            self.assertFalse(module.telemetry)
            load.assert_called_once_with("litellm")
            module.model_alias_map = {MODEL: "hidden-replacement"}
            with self.assertRaises(RuntimeError):
                _default_completion()
        module.completion.assert_not_called()

    def test_per_call_zero_or_false_cannot_hide_sdk_global_defaults(self):
        for name, value in (("model_fallbacks", ["undeclared-replacement"]),
                            ("num_retries", 2), ("drop_params", True),
                            ("success_callback", ["external-tracking"]),
                            ("context_window_fallbacks", [{"a": "b"}]),
                            ("api_base", "https://undeclared-endpoint.invalid/v1")):
            module = SimpleNamespace(completion=Mock(), **{name: value})
            with self.subTest(name=name), patch.dict(os.environ, {}, clear=True), patch(
                    "routerbench.agent_litellm.importlib.import_module", return_value=module):
                with self.assertRaises(RuntimeError):
                    _default_completion()
                module.completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
