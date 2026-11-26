"""Real optional SDK, mocked HTTP: distinguish SDK-synthesized 500 from HTTP 500."""
import importlib.util
import json
import os
import unittest
from unittest.mock import patch

from relay_router.agent import AgentRequest, AgentStateV8, parse_action
from relay_router.errors import BackendTransportError, BackendHTTPError, BackendResponseError, BackendActionError
from routerbench.agent_litellm import LiteLLMEpisodeGenerator, _default_completion


@unittest.skipUnless(importlib.util.find_spec('litellm'), 'Optional LiteLLM SDK not installed')
class SDKIntegrationTests(unittest.TestCase):
    def test_actual_sdk_output_parse_failed_is_semantic_and_generic_400_stays_infrastructure(self):
        import httpx
        model = 'groq/openai/gpt-oss-20b'
        generator = LiteLLMEpisodeGenerator({model: {'returned_model_ids': ['openai/gpt-oss-20b'],
            'context_window': 131072, 'api_key_env': 'FIXTURE_ONLY_KEY'}},
            {'max_output_tokens': 2048, 'temperature': 0, 'timeout': 10}, completion=_default_completion())
        request = AgentRequest('Offline fixture.', (), AgentStateV8(0, 0, (), (), 1))
        http_request = httpx.Request('POST', 'https://api.groq.com/openai/v1/chat/completions')
        with patch.dict(os.environ, {'FIXTURE_ONLY_KEY': 'offline-not-a-real-key'}), \
             patch('litellm.llms.custom_httpx.http_handler.HTTPHandler.post') as post:
            for code, expected in [('output_parse_failed', BackendActionError), (None, BackendHTTPError)]:
                post.reset_mock()
                body = {'error': {'type': 'invalid_request_error', 'message': 'PRIVATE reflected text',
                                  'failed_generation': 'PRIVATE model output'}}
                if code is not None:
                    body['error']['code'] = code
                response = httpx.Response(400, json=body, request=http_request)
                post.side_effect = httpx.HTTPStatusError('PRIVATE', request=http_request, response=response)
                with self.assertRaises(expected) as caught:
                    generator(model, request)
                post.assert_called_once()
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(caught.exception.details()['diagnostic'].get('provider_error_code'), code)
                self.assertNotIn('PRIVATE', json.dumps(caught.exception.details()))

    def test_actual_sdk_native_tools_preserves_wire_contract_and_invalid_arguments(self):
        import httpx
        model = 'groq/openai/gpt-oss-120b'
        generator = LiteLLMEpisodeGenerator({model: {
            'returned_model_ids': ['openai/gpt-oss-120b'], 'context_window': 131072,
            'api_key_env': 'FIXTURE_ONLY_KEY'}}, {
            'max_output_tokens': 2048, 'temperature': 0, 'timeout': 10,
            'action_format': 'native_tools'}, completion=_default_completion())
        definition = {'name': 'lookup', 'description': 'Read a public item.', 'parameters': {
            'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key'],
            'additionalProperties': False}}
        request = AgentRequest('Read the item.', (
            {'role': 'tool', 'content': {'name': 'environment', 'observation': {}, 'tools': [definition]}},
            {'role': 'assistant', 'content': '{"type":"tool","name":"lookup","arguments":{"key":"x"}}'},
            {'role': 'tool', 'content': {'name': 'lookup', 'ok': True,
                                        'observation': {'value': 'public'}, 'error_type': None}},
        ), AgentStateV8(1, 0, (), ('lookup',), 3))
        http_request = httpx.Request('POST', 'https://api.groq.com/openai/v1/chat/completions')
        cases = [(None, 'public', True),
                 (None, '{"type":"tool","name":"lookup","arguments":{"key":"x"}}', True),
                 ('lookup', '{"key":"x"}', True),
                 ('lookup', '{"key":"x","key":"y"}', False)]
        with patch.dict(os.environ, {'FIXTURE_ONLY_KEY': 'offline-not-a-real-key'}), \
             patch('litellm.llms.custom_httpx.http_handler.HTTPHandler.post') as post:
            for name, arguments, valid in cases:
                with self.subTest(name=name, valid=valid):
                    post.reset_mock()
                    message = {'role': 'assistant', 'content': arguments if name is None else None}
                    if name is not None:
                        message['tool_calls'] = [{'id': 'provider-call', 'type': 'function',
                            'function': {'name': name, 'arguments': arguments}}]
                    payload = {'id': 'offline-fixture', 'object': 'chat.completion', 'created': 0,
                        'service_tier': 'on_demand', 'model': 'openai/gpt-oss-120b', 'choices': [
                            {'index': 0, 'message': message,
                             'finish_reason': 'stop' if name is None else 'tool_calls'}],
                        'usage': {'prompt_tokens': 15, 'completion_tokens': 7, 'total_tokens': 22}}
                    post.return_value = httpx.Response(200, json=payload, request=http_request)
                    reply = generator(model, request)
                    post.assert_called_once()
                    wire = json.loads(post.call_args.kwargs['data'])
                    self.assertEqual(wire['model'], 'openai/gpt-oss-120b')
                    self.assertEqual(wire['tool_choice'], 'auto')
                    self.assertIs(wire['parallel_tool_calls'], False)
                    self.assertNotIn('response_format', wire)
                    self.assertEqual([tool['function']['name'] for tool in wire['tools']],
                                     ['lookup'])
                    self.assertEqual([message['role'] for message in wire['messages']],
                                     ['system', 'user', 'assistant', 'tool'])
                    previous_call = wire['messages'][2]['tool_calls'][0]
                    self.assertEqual(previous_call['id'], wire['messages'][3]['tool_call_id'])
                    self.assertEqual(json.loads(previous_call['function']['arguments']), {'key': 'x'})
                    self.assertEqual(json.loads(wire['messages'][3]['content'])['observation'],
                                     {'value': 'public'})
                    self.assertEqual(reply.usage['input_tokens'], 15)
                    self.assertEqual(reply.usage['output_tokens'], 7)
                    self.assertIsNone(reply.cost_usd)
                    if name is None:
                        self.assertFalse(reply.usage['provider_metadata']['native_tool_calls'])
                    else:
                        self.assertEqual(reply.usage['provider_metadata']['native_tool_calls'][0]['arguments'],
                                         arguments)
                    if valid:
                        action = parse_action(reply.content, ('lookup',))
                        self.assertEqual(action['type'], 'final' if name is None else 'tool')
                        if name is None:
                            self.assertEqual(action['answer'], arguments)
                    else:
                        with self.assertRaises(ValueError):
                            parse_action(reply.content, ('lookup',))

    def test_actual_sdk_http_success_and_wrapped_transport_failure(self):
        import httpx
        completion = _default_completion()
        model = 'groq/openai/gpt-oss-120b'
        generator = LiteLLMEpisodeGenerator({model: {'returned_model_ids': ['openai/gpt-oss-120b'],
            'context_window': 131072, 'api_key_env': 'FIXTURE_ONLY_KEY'}},
            {'max_output_tokens': 2048, 'temperature': 0, 'timeout': 10}, completion=completion)
        request = AgentRequest('Responde con una acción final.', (), AgentStateV8(0, 0, (), (), 1))
        payload = {'id': 'offline-fixture', 'object': 'chat.completion', 'created': 0, 'service_tier': 'on_demand',
                   'model': 'openai/gpt-oss-120b', 'choices': [{'index': 0, 'message': {'role': 'assistant',
                   'content': '{"type":"final","answer":"ok"}'}, 'finish_reason': 'stop'}],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        http_request = httpx.Request('POST', 'https://api.groq.com/openai/v1/chat/completions')
        with patch.dict(os.environ, {'FIXTURE_ONLY_KEY': 'offline-not-a-real-key'}), \
             patch('litellm.llms.custom_httpx.http_handler.HTTPHandler.post') as post:
            post.return_value = httpx.Response(200, json=payload, request=http_request)
            try:
                reply = generator(model, request)
            except BackendTransportError as error:
                self.fail(json.dumps({'failure': error.details(), 'mock_http_calls': post.call_count}))
            self.assertEqual(reply.usage['input_tokens'], 10)
            post.assert_called_once()
            post.reset_mock()
            post.side_effect = httpx.ConnectError('SECRET reflected text must not appear')
            with self.assertRaises(BackendTransportError) as caught:
                generator(model, request)
            post.assert_called_once()
            self.assertIsNone(caught.exception.status_code)
            self.assertNotIn('SECRET', json.dumps(caught.exception.details()))
            self.assertIn('ConnectError', caught.exception.details()['diagnostic']['exception_types'])
            post.reset_mock()
            # HTTPHandler.post itself raises on non-2xx responses.
            failure = httpx.Response(500, json={'error': 'SECRET'}, request=http_request)
            post.side_effect = httpx.HTTPStatusError('SECRET', request=http_request, response=failure)
            with self.assertRaises(BackendHTTPError) as caught:
                generator(model, request)
            post.assert_called_once()
            self.assertEqual(caught.exception.status_code, 500)
            self.assertEqual(caught.exception.details()['diagnostic']['status_source'], 'http_response')
            post.reset_mock()
            post.side_effect = None
            missing_optional = dict(payload)
            missing_optional.pop('service_tier')
            post.return_value = httpx.Response(200, json=missing_optional, request=http_request)
            with self.assertRaises(BackendResponseError) as caught:
                generator(model, request)
            post.assert_called_once()
            self.assertFalse(caught.exception.retryable)
            self.assertEqual(caught.exception.details()['diagnostic']['missing_sdk_attribute'], 'service_tier')
