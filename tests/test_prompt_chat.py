"""Standalone prompt-client contracts; all provider and model loads are mocked."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("relay_prompt_chat_tested", ROOT / "tools/relay_prompt_chat.py")
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)
from relay_router.prompt_selection import PairwisePromptSelector


class PromptChatTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in client tests"))
        guard.start()
        self.addCleanup(guard.stop)
        self.prompt = "  Primera línea.\n    return 'ñ'\n\n$literal `code`\n"
        self.key = "SYNTHETIC_KEY_DO_NOT_ECHO"

    def payload(self, selected_model, **changes):
        value = {"model": selected_model.removeprefix("groq/"), "choices": [{"message": {"content": "Respuesta"}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        value.update(changes)
        return value

    def generate(self, model, value):
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        with patch.object(chat, "build_opener", return_value=opener) as build:
            result = chat.groq_generate(model, prompt=self.prompt, api_key=self.key)
        return result, opener, build

    def test_one_dispatch_preserves_prompt_exact_model_and_both_tariffs(self):
        for model, cost in (("groq/openai/gpt-oss-20b", .0000225),
                            ("groq/openai/gpt-oss-120b", .000045)):
            with self.subTest(model=model):
                result, opener, build = self.generate(model, self.payload(model))
                opener.open.assert_called_once()
                request = opener.open.call_args.args[0]
                self.assertEqual(request.get_method(), "POST")
                self.assertEqual(request.full_url, "https://api.groq.com/openai/v1/chat/completions")
                self.assertEqual(request.get_header("Authorization"), "Bearer " + self.key)
                body = json.loads(request.data)
                self.assertEqual(body["model"], model.removeprefix("groq/"))
                self.assertEqual(body["messages"], [{"role": "user", "content": self.prompt}])
                self.assertFalse(body["stream"])
                self.assertEqual(body["max_completion_tokens"], 1024)
                self.assertEqual(body["reasoning_effort"], "low")
                self.assertEqual(result["measurement"]["input_tokens"], 100)
                self.assertEqual(result["measurement"]["output_tokens"], 50)
                self.assertAlmostEqual(result["measurement"]["cost_usd"], cost)
                self.assertEqual(result["measurement"]["cost_basis"], "tariff_estimate")
                self.assertNotIn(self.key, json.dumps(result))
                self.assertIsInstance(build.call_args.args[0], chat._NoRedirect)

    def test_unknown_usage_never_becomes_free_usage(self):
        model = "groq/openai/gpt-oss-20b"
        cases = [(None, None, None), ({}, None, None),
                 ({"prompt_tokens": 100}, 100, None),
                 ({"prompt_tokens": True, "completion_tokens": 50}, None, 50),
                 ({"prompt_tokens": -1, "completion_tokens": 50}, None, 50),
                 ({"prompt_tokens": 100, "completion_tokens": 1.5}, 100, None)]
        for usage, incoming, outgoing in cases:
            with self.subTest(usage=usage):
                result, _, _ = self.generate(model, self.payload(model, usage=usage))
                measured = result["measurement"]
                self.assertIsNone(measured["cost_usd"])
                self.assertEqual(measured["cost_basis"], "unknown")
                self.assertEqual(measured["input_tokens"], incoming)
                self.assertEqual(measured["output_tokens"], outgoing)

    def test_explicit_zero_usage_is_known_zero(self):
        model = "groq/openai/gpt-oss-120b"
        result, _, _ = self.generate(model, self.payload(model, usage={"prompt_tokens": 0, "completion_tokens": 0}))
        self.assertEqual(result["measurement"]["cost_usd"], 0)
        self.assertEqual(result["measurement"]["cost_basis"], "tariff_estimate")

    def test_unknown_model_rejected_without_network(self):
        with patch.object(chat, "build_opener") as build:
            with self.assertRaises(ValueError):
                chat.groq_generate("groq/other-model", prompt=self.prompt, api_key=self.key)
            build.assert_not_called()

    def test_provider_model_must_match_selected_model(self):
        model = "groq/openai/gpt-oss-20b"
        for wrong in (None, "openai/gpt-oss-120b", "another-provider/model"):
            with self.subTest(wrong=wrong), self.assertRaises(RuntimeError):
                self.generate(model, self.payload(model, model=wrong))

    def test_http_and_transport_failures_do_not_echo_secrets_or_retry(self):
        model = "groq/openai/gpt-oss-120b"
        body = Mock()
        body.read.side_effect = AssertionError("Do not read reflected error body")
        secret = self.key + self.prompt
        failures = [HTTPError("https://provider.invalid/?" + self.key, 429, secret, {}, body),
                    HTTPError("https://provider.invalid", 400, secret, {}, body),
                    URLError(secret), TimeoutError(secret)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                opener = Mock()
                opener.open.side_effect = failure
                with patch.object(chat, "build_opener", return_value=opener):
                    with self.assertRaises(RuntimeError) as error:
                        chat.groq_generate(model, prompt=self.prompt, api_key=self.key)
                opener.open.assert_called_once()
                self.assertNotIn(self.key, str(error.exception))
                self.assertNotIn(self.prompt, str(error.exception))
                self.assertIn("no se reintentó", str(error.exception))
        body.read.assert_not_called()
        self.assertIsNone(chat._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.invalid"))

    def test_invalid_provider_payloads_fail_without_reflecting_body(self):
        model = "groq/openai/gpt-oss-20b"
        for raw in (b"invalid JSON SYNTHETIC_KEY_DO_NOT_ECHO", b"[]", b"null", b"\xff", b"x" * (4 * 1024 * 1024 + 1)):
            with self.subTest(length=len(raw)):
                opener = Mock()
                opener.open.return_value = io.BytesIO(raw)
                with patch.object(chat, "build_opener", return_value=opener):
                    with self.assertRaises(RuntimeError) as error:
                        chat.groq_generate(model, prompt=self.prompt, api_key=self.key)
                self.assertNotIn(self.key, str(error.exception))
                opener.open.assert_called_once()
        for value in (self.payload(model, choices=[]), self.payload(model, choices=[{"message": {"content": " "}}]),
                      self.payload(model, choices=[{"message": {"content": {"secret": self.key}}}])):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.generate(model, value)

    def selector(self, scorer):
        return PairwisePromptSelector(weak_model="groq/openai/gpt-oss-20b", strong_model="groq/openai/gpt-oss-120b",
                                      scorer=scorer, threshold=.58, selector_id="synthetic-test-scorer")

    def test_route_only_prompt_file_uses_selector_without_key_or_http_and_private_log(self):
        scorer = Mock(return_value={"score": .2, "input_tokens": 12, "truncated_tokens": 0})
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "prompt.txt"
            prompt_path.write_text(self.prompt, encoding="utf-8", newline="\n")
            log_path = Path(tmp) / "ledger.jsonl"
            output = io.StringIO()
            with patch.object(chat, "load_groq_pilot_selector", return_value=(self.selector(scorer), {"load_seconds": .01})) as load, patch.object(
                    chat, "getpass", side_effect=AssertionError("No credential request in route-only")) as secret, patch.object(
                    chat, "build_opener", side_effect=AssertionError("No HTTP in route-only")) as http, patch(
                    "builtins.input", side_effect=AssertionError("Prompt supplied by file")), contextlib.redirect_stdout(output):
                status = chat.main(["--route-only", "--prompt-file", str(prompt_path), "--checkpoint", tmp, "--log", str(log_path)])
            self.assertEqual(status, 0)
            load.assert_called_once_with(Path(tmp))
            scorer.assert_called_once_with(self.prompt)
            secret.assert_not_called()
            http.assert_not_called()
            self.assertIn("groq/openai/gpt-oss-20b", output.getvalue())
            self.assertNotIn(self.prompt, output.getvalue())
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            observations = [row for row in events if row["kind"] == "relay_task_observation"]
            self.assertEqual(len(observations), 1)
            self.assertIsNone(observations[0]["accepted"])
            self.assertIsNone(observations[0]["cost_usd"])
            self.assertEqual(observations[0]["logical_calls"], 0)
            self.assertNotIn(self.prompt, json.dumps(events, ensure_ascii=False))

    def test_route_only_interactive_requests_reuse_one_resident_selector(self):
        scorer = Mock(side_effect=[{"score": .2}, {"score": .8}])
        output = io.StringIO()
        with patch.object(chat, "load_groq_pilot_selector", return_value=(self.selector(scorer), {"load_seconds": .01})) as load, patch.object(
                chat, "getpass", side_effect=AssertionError("No credential request")) as secret, patch.object(
                chat, "build_opener", side_effect=AssertionError("No HTTP")) as http, patch(
                "builtins.input", side_effect=["first", "second", ":salir"]), contextlib.redirect_stdout(output):
            status = chat.main(["--route-only"])
        self.assertEqual(status, 0)
        self.assertEqual(load.call_count, 1)
        self.assertEqual([call.args for call in scorer.call_args_list], [("first",), ("second",)])
        self.assertIn("groq/openai/gpt-oss-20b", output.getvalue())
        self.assertIn("groq/openai/gpt-oss-120b", output.getvalue())
        secret.assert_not_called()
        http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
