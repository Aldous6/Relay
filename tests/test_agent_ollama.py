"""All Ollama episode requests use mocked HTTP; no service or LLM is called."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from relay_router.agent import AgentRequest, AgentState, ModelReply, parse_action
from routerbench.agent_ollama import (AGENT_SYSTEM, OllamaEpisodeGenerator,
                                     render_agent_input, routing_text)
from routerbench.live import LiveBackendError


MODELS = {"qwen2.5:1.5b": {"digest": "a" * 64, "context_length": 32768},
          "qwen2.5:0.5b": {"digest": "b" * 64, "context_length": 32768}}
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 32768, "num_predict": 192}
SMALL = "qwen2.5:0.5b"


def request(prompt="Lee el archivo.\n    Conserva la indentacion y los acentos: acción."):
    history = ({"role": "tool", "content": {"name": "environment", "tools": [
        {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]}},
        {"role": "assistant", "content": '{"type":"tool","name":"read_file","arguments":{"path":"in.txt"}}'},
        {"role": "tool", "content": {"name": "read_file", "ok": True, "observation": "línea 1\n    línea 2"}},
        {"role": "error", "content": {"error_type": "prior public error"}})
    return AgentRequest(prompt, history, AgentState(2, 1, ("prior public error",), ("read_file",), 4))


def tags(context=True):
    return {"models": [{"name": name, "model": name, "digest": value["digest"],
                        "details": {"context_length": value["context_length"]} if context else {}}
                       for name, value in MODELS.items()]}


def reply(content='{"type":"final","answer":"done"}', **updates):
    return {"model": SMALL, "message": {"role": "assistant", "content": content},
            "done": True, "done_reason": "stop", "prompt_eval_count": 60, "eval_count": 12,
            "prompt_eval_cached_count": 20, "total_duration": 2_000_000_000,
            "load_duration": 100_000_000, "prompt_eval_duration": 200_000_000,
            "eval_duration": 1_700_000_000, **updates}


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.data).encode("utf-8")


class OllamaEpisodeTests(unittest.TestCase):
    def setup_http(self, build, result=None, catalog=None):
        result = reply() if result is None else result
        catalog = tags() if catalog is None else catalog
        def respond(http_request, **kwargs):
            if http_request.full_url.endswith("/api/tags"):
                return FakeResponse(catalog)
            if http_request.full_url.endswith("/api/chat"):
                return FakeResponse(result)
            if http_request.full_url.endswith("/api/show"):
                return FakeResponse({"model_info": {"qwen2.context_length": 32768}})
            raise AssertionError("Unexpected HTTP endpoint")
        build.return_value.open.side_effect = respond

    @patch("routerbench.live.build_opener")
    def test_constructor_manifest_and_render_are_pure_and_independent(self, build):
        metadata, options = deepcopy(MODELS), deepcopy(OPTIONS)
        adapter = OllamaEpisodeGenerator(metadata, options)
        metadata[SMALL]["digest"] = "c" * 64
        options["seed"] = 99
        manifest = adapter.manifest()
        self.assertEqual(manifest["models_metadata"], MODELS)
        self.assertEqual(manifest["options"], OPTIONS)
        self.assertEqual(manifest["system_sha256"], hashlib.sha256(AGENT_SYSTEM.encode()).hexdigest())
        self.assertEqual(manifest["format"], "json")
        self.assertEqual(manifest["keep_alive"], "10m")
        manifest["models_metadata"][SMALL]["digest"] = "tampered"
        self.assertEqual(adapter.manifest()["models_metadata"], MODELS)
        render_agent_input(request())
        routing_text(request())
        build.assert_not_called()

    def test_render_keeps_task_first_and_preserves_full_history_and_state(self):
        original = request("Inicio actual\n    " + "larga evidencia ñ\n" * 1000 + "fin")
        text = render_agent_input(original)
        expected_state = json.dumps(original.state.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected_history = json.dumps(original.to_dict()["history"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(text, "TASK:\n" + original.prompt + "\n\nCURRENT_STATE:\n" + expected_state
                         + "\n\nOBSERVED_HISTORY:\n" + expected_history)
        self.assertEqual(routing_text(original), "SYSTEM:\n" + AGENT_SYSTEM + "\n\nUSER:\n" + text)
        self.assertEqual(json.loads(text.split("\n\nOBSERVED_HISTORY:\n", 1)[1]), original.to_dict()["history"])
        with self.assertRaises(ValueError):
            render_agent_input({"prompt": "not an AgentRequest"})

    @patch("routerbench.live.build_opener")
    def test_http_payload_and_all_provider_measurements_are_preserved(self, build):
        self.setup_http(build)
        adapter = OllamaEpisodeGenerator(MODELS, OPTIONS)
        original = request()
        result = adapter(SMALL, original)
        calls = build.return_value.open.call_args_list
        self.assertEqual([call.args[0].get_method() for call in calls], ["GET", "POST"])
        self.assertEqual(calls[0].args[0].full_url, "http://127.0.0.1:11434/api/tags")
        payload = json.loads(calls[1].args[0].data)
        self.assertEqual(set(payload), {"model", "messages", "stream", "format", "keep_alive", "options"})
        self.assertEqual(payload["model"], SMALL)
        self.assertEqual(payload["messages"], [{"role": "system", "content": AGENT_SYSTEM},
                                                {"role": "user", "content": render_agent_input(original)}])
        self.assertEqual(payload["options"], OPTIONS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["keep_alive"], "10m")
        self.assertIsInstance(result, ModelReply)
        self.assertIsNone(result.cost_usd)
        self.assertEqual({key: result.usage[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens")},
                         {"input_tokens": 60, "output_tokens": 12, "cached_input_tokens": 20, "total_tokens": 72})
        meta = result.usage["provider_metadata"]
        self.assertEqual(meta["requested_model"], SMALL)
        self.assertEqual(meta["returned_model"], SMALL)
        self.assertEqual(meta["digest"], MODELS[SMALL]["digest"])
        self.assertEqual(meta["finish_reason"], "stop")
        self.assertEqual(meta["durations"], {"total_seconds": 2., "load_seconds": .1,
                                             "prompt_eval_seconds": .2, "eval_seconds": 1.7})
        self.assertEqual(meta["num_ctx"], 32768)
        self.assertEqual(meta["conservative_context_budget"], meta["input_utf8_bytes"] + 192 + 512)
        self.assertFalse(meta["input_truncated_by_client"])
        self.assertFalse(meta["provider_context_preservation_verified"])

    @patch("routerbench.live.build_opener")
    def test_verify_models_checks_context_and_digest_without_chat(self, build):
        self.setup_http(build, catalog=tags(context=False))
        adapter = OllamaEpisodeGenerator(MODELS, OPTIONS)
        self.assertEqual(adapter.verify_models(), MODELS)
        self.assertEqual([call.args[0].full_url.rsplit("/", 1)[-1] for call in build.return_value.open.call_args_list],
                         ["tags", "show", "show"])
        changed = tags()
        changed["models"][0]["details"]["context_length"] = 8192
        self.setup_http(build, catalog=changed)
        with self.assertRaisesRegex(LiveBackendError, "contexto"):
            adapter.verify_models()

    @patch("routerbench.live.build_opener")
    def test_unknown_model_oversized_context_and_invalid_input_never_make_http_calls(self, build):
        adapter = OllamaEpisodeGenerator(MODELS, OPTIONS)
        for model, value in (("another-model", request()), (SMALL, request("X" * 40000)), (SMALL, "not a request")):
            with self.subTest(model=model), self.assertRaises(ValueError):
                adapter(model, value)
        build.assert_not_called()

    @patch("routerbench.live.build_opener")
    def test_digest_change_and_missing_model_fail_before_chat_without_retry(self, build):
        adapter = OllamaEpisodeGenerator(MODELS, OPTIONS)
        changed = tags()
        changed["models"][1]["digest"] = "c" * 64
        for catalog in (changed, {"models": []}):
            build.return_value.open.reset_mock()
            self.setup_http(build, catalog=catalog)
            with self.assertRaises(LiveBackendError):
                adapter(SMALL, request())
            build.return_value.open.assert_called_once()
            self.assertEqual(build.return_value.open.call_args.args[0].get_method(), "GET")

    @patch("routerbench.live.build_opener")
    def test_changed_returned_model_or_nontext_response_is_rejected_without_retry(self, build):
        adapter = OllamaEpisodeGenerator(MODELS, OPTIONS)
        for response in (reply(model="other"), reply(model=None), reply(message={"content": {"type": "final"}})):
            build.return_value.open.reset_mock()
            self.setup_http(build, result=response)
            with self.assertRaises(LiveBackendError):
                adapter(SMALL, request())
            self.assertEqual(build.return_value.open.call_count, 2)

    @patch("routerbench.live.build_opener")
    def test_empty_and_truncated_content_preserve_usage_for_harness_failure_accounting(self, build):
        adapter = OllamaEpisodeGenerator(MODELS, OPTIONS)
        for content in ("", '{"type":"tool"'):
            self.setup_http(build, result=reply(content, done_reason="length"))
            result = adapter(SMALL, request())
            self.assertEqual(result.content, content)
            self.assertEqual(result.usage["input_tokens"], 60)
            self.assertEqual(result.usage["output_tokens"], 12)
            self.assertEqual(result.usage["provider_metadata"]["finish_reason"], "length")
            with self.assertRaises(ValueError):
                parse_action(result.content, request().state.available_tools)

    @patch("routerbench.live.build_opener")
    def test_missing_measurements_stay_unknown_and_transport_does_not_retry(self, build):
        self.setup_http(build, result={"model": SMALL, "message": {"content": "{}"}})
        result = OllamaEpisodeGenerator(MODELS, OPTIONS)(SMALL, request())
        for name in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens"):
            self.assertIsNone(result.usage[name])
        self.assertTrue(all(value is None for value in result.usage["provider_metadata"]["durations"].values()))
        self.assertIsNone(result.cost_usd)
        build.return_value.open.reset_mock()
        build.return_value.open.side_effect = URLError("private details")
        with self.assertRaises(LiveBackendError) as caught:
            OllamaEpisodeGenerator(MODELS, OPTIONS)(SMALL, request())
        self.assertNotIn("private details", str(caught.exception))
        build.return_value.open.assert_called_once()

    @patch("routerbench.live.build_opener")
    def test_catalog_options_and_nonlocal_urls_are_rejected_without_io(self, build):
        for url in ("https://example.com", "http://192.168.1.5:11434", "http://127.0.0.1.example.com",
                    "http://user:password@localhost:11434", "http://localhost:11434/path", "http://localhost:11434?x=1",
                    "http://local\nhost:11434", "http://localhost:0"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaEpisodeGenerator(MODELS, OPTIONS, base_url=url)
        for options in ({**OPTIONS, "temperature": True}, {**OPTIONS, "temperature": .5}, {**OPTIONS, "seed": -1},
                        {**OPTIONS, "num_ctx": 65536}, {**OPTIONS, "num_predict": 32768}, {**OPTIONS, "keep_alive": "10m"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                OllamaEpisodeGenerator(MODELS, options)
        for metadata in ({}, {SMALL: {"digest": "bad", "context_length": 32768}},
                         {SMALL: {"digest": "b" * 64, "context_length": True}}):
            with self.assertRaises(ValueError):
                OllamaEpisodeGenerator(metadata, OPTIONS)
        for url in ("http://localhost:11434", "http://[::1]:11434", "http://127.0.0.2:11434/"):
            OllamaEpisodeGenerator(MODELS, OPTIONS, base_url=url)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
