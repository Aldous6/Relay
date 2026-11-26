"""Provider contract tests; all generation is mocked and cannot spend money."""
import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from routerbench.live import LiveBackendError, discover_ollama, generate_ollama, generate_openrouter, validate_messages


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.data).encode("utf-8")


class LiveBackendTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "system", "content": "Answer from the supplied documents."},
                         {"role": "user", "content": "First question"},
                         {"role": "assistant", "content": "First answer"},
                         {"role": "user", "content": "  Compare them.\n" + "document body\n" * 1000}]

    @patch("routerbench.live.build_opener")
    def test_local_call_preserves_full_history_and_selected_model(self, build):
        build.return_value.open.return_value = FakeResponse({"model": "llama3:latest", "message": {"content": "Compared."},
                                                           "prompt_eval_count": 3010, "eval_count": 12, "total_duration": 2_000_000_000, "done_reason": "stop"})
        result = generate_ollama("llama3:latest", self.messages, max_output_tokens=256, context_window=8192)
        build.return_value.open.assert_called_once()
        request = build.return_value.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "llama3:latest")
        self.assertEqual(payload["messages"], self.messages)
        self.assertEqual(payload["options"], {"num_predict": 256, "num_ctx": 8192})
        self.assertFalse(payload["stream"])
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["usage"]["total_tokens"], 3022)
        self.assertEqual(result["provider_duration_seconds"], 2)
        self.assertIsNone(result["cost_usd"])
        self.assertFalse(result["input_truncated_by_client"])

    @patch("routerbench.live.build_opener")
    def test_local_discovery_never_generates_and_preserves_missing_data(self, build):
        build.return_value.open.return_value = FakeResponse({"models": [{"name": "b", "details": {}},
                                                                                     {"name": "a", "size": 100, "details": {"context_length": 8192}}]})
        models = discover_ollama()
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/tags")
        self.assertEqual([model["name"] for model in models], ["a", "b"])
        self.assertEqual(models[0]["context_length"], 8192)
        self.assertIsNone(models[1]["size_bytes"])

    @patch("routerbench.live.build_opener")
    def test_missing_usage_is_unknown_not_zero(self, build):
        build.return_value.open.return_value = FakeResponse({"message": {"content": "Hello"}})
        result = generate_ollama("local", self.messages)
        self.assertEqual(result["usage"], {"input_tokens": None, "output_tokens": None, "total_tokens": None, "cached_input_tokens": None})
        self.assertIsNone(result["provider_duration_seconds"])

    @patch("routerbench.live.build_opener")
    def test_openrouter_only_calls_concrete_model_and_reports_provider_cost(self, build):
        build.return_value.open.return_value = FakeResponse({"model": "vendor/chosen", "choices": [{"message": {"content": "Actual response"}, "finish_reason": "stop"}],
                                                           "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "cost": 0.0012}})
        result = generate_openrouter("secret-test-key", "vendor/chosen", self.messages)
        build.return_value.open.assert_called_once()
        request = build.return_value.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "vendor/chosen")
        self.assertEqual(payload["messages"], self.messages)
        self.assertEqual(payload["provider"], {"allow_fallbacks": False})
        self.assertNotIn("models", payload)
        self.assertEqual(result["cost_usd"], 0.0012)
        self.assertNotIn("secret-test-key", json.dumps(result))

    @patch("routerbench.live.build_opener")
    def test_failures_do_not_echo_credentials_body_or_retry(self, build):
        for error in (HTTPError("https://example.test/?secret-test-key", 401, "secret-test-key", {}, io.BytesIO(b"secret-test-key")),
                      URLError("secret-test-key")):
            with self.subTest(error=type(error).__name__):
                build.return_value.open.reset_mock()
                build.return_value.open.side_effect = error
                with self.assertRaises(LiveBackendError) as caught:
                    generate_openrouter("secret-test-key", "vendor/chosen", self.messages)
                self.assertNotIn("secret-test-key", str(caught.exception))
                build.return_value.open.assert_called_once()

    @patch("routerbench.live.build_opener")
    def test_reflected_provider_error_is_not_displayed(self, build):
        build.return_value.open.return_value = FakeResponse({"error": {"message": "Bearer secret-test-key"}})
        with self.assertRaises(LiveBackendError) as caught:
            generate_openrouter("secret-test-key", "vendor/chosen", self.messages)
        self.assertNotIn("secret-test-key", str(caught.exception))

    @patch("routerbench.live.build_opener")
    def test_invalid_inputs_and_autorouter_fail_before_network(self, build):
        invalid_messages = [[], "user text", [{"role": ["user"], "content": "data"}], [{"role": "tool", "content": "data"}],
                            [{"role": "user", "content": ["multimodal"]}],
                            [{"role": "user", "content": "ok", "extra": "hidden"}],
                            [{"role": "assistant", "content": "not a user request"}]]
        for messages in invalid_messages:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                generate_ollama("local", messages)
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generate_ollama("local", self.messages, max_output_tokens=value)
        with self.assertRaises(ValueError):
            generate_openrouter("secret-test-key", "openrouter/auto", self.messages)
        with self.assertRaises(ValueError):
            generate_ollama("local", self.messages, base_url="http://user:password@localhost:11434")
        build.assert_not_called()

    def test_validating_history_copies_without_mutating_or_reclassifying(self):
        validated = validate_messages(self.messages)
        self.assertEqual(validated, self.messages)
        self.assertIsNot(validated, self.messages)
        self.assertIsNot(validated[0], self.messages[0])


if __name__ == "__main__":
    unittest.main()
