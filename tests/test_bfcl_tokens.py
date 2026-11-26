from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from routerbench.bfcl_harness import BFCLRequest, BFCLState
from routerbench.bfcl_tokens import (GPTOSSTokenEstimator, native_request_payload,
                                   DEFAULT_ASSET_ROOT, DEFAULT_PROVENANCE)


def request(messages=None, tools=None):
    state = BFCLState(0, 0, 0, (), (), 0, (), 21, 0)
    if messages is None:
        messages = ({"role": "system", "content": "Preserve lines.\n    Keep indentation."},
                    {"role": "user", "content": "Find the fixture record."})
    if tools is None:
        tools = ({"type": "function", "function": {"name": "lookup", "description": "Read a fixture record.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Record name"}},
                           "required": ["name"], "additionalProperties": False}}},)
    return BFCLRequest(tuple(messages), tuple(tools), state)


class TokenizerIntegrityTests(unittest.TestCase):
    def test_manifest_hash_is_checked_before_tokenizer_import_or_asset_use(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"assets":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance SHA256"), patch("socket.create_connection") as network:
                GPTOSSTokenEstimator(asset_root=directory, provenance_path=manifest)
            network.assert_not_called()

    def test_corrupt_asset_is_rejected_before_tokenizer_use(self):
        with tempfile.TemporaryDirectory() as directory:
            first = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))["assets"][0]
            target = Path(directory) / first["path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"not a template")
            with self.assertRaisesRegex(ValueError, "asset SHA256"), patch("socket.create_connection") as network:
                GPTOSSTokenEstimator(asset_root=directory)
            network.assert_not_called()

    def test_missing_cache_never_triggers_download(self):
        with tempfile.TemporaryDirectory() as directory, patch("socket.create_connection") as network:
            with self.assertRaises(FileNotFoundError):
                GPTOSSTokenEstimator(asset_root=directory)
            network.assert_not_called()


@unittest.skipUnless((DEFAULT_ASSET_ROOT / "o200k_base.tiktoken").exists(), "Pinned local tokenizer assets unavailable; no download in tests.")
class OfflineGPTOSSTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("socket.create_connection") as network, patch("tiktoken.get_encoding", side_effect=AssertionError("implicit loader")):
            cls.counter = GPTOSSTokenEstimator()
        network.assert_not_called()

    def test_encoding_matches_installed_official_implementation_without_network(self):
        import tiktoken
        import tiktoken_ext.openai_public as official
        with patch.object(official, "load_tiktoken_bpe", return_value=self.counter._encoding._mergeable_ranks):
            declared = official.o200k_harmony()
        self.assertEqual(self.counter._encoding._pat_str, declared["pat_str"])
        self.assertEqual(self.counter._encoding._special_tokens, declared["special_tokens"])
        oracle = tiktoken.Encoding(**declared)
        for text in ("hello world", "á 中文\n    indentation", "<|start|>assistant<|channel|>analysis<|message|>x"):
            self.assertEqual(self.counter._encoding.encode(text, allowed_special="all"), oracle.encode(text, allowed_special="all"))

    def test_deterministic_full_request_estimate_and_manifest_disclaim_provider_precision(self):
        current = request()
        before = deepcopy(current.to_dict())
        with patch("socket.create_connection") as network:
            estimate = self.counter.estimate(current)
        network.assert_not_called()
        self.assertEqual(estimate, self.counter.estimate(current))
        self.assertEqual(current.to_dict(), before)
        self.assertEqual(estimate["estimated_input_tokens"], self.counter.estimated_input_tokens(current))
        self.assertEqual(estimate["estimated_input_tokens"],
            max(estimate["serialized_payload_tokens"], estimate["published_harmony_tokens"]) + 256)
        self.assertEqual(estimate["published_template_status"], "rendered_published_template")
        self.assertFalse(estimate["is_proven_upper_bound"])
        self.assertFalse(estimate["provider_server_template_verified"])
        manifest = self.counter.manifest()
        self.assertEqual(manifest["server_template_verification"], "no server template verification")
        self.assertIn("tiktoken", manifest["packages"])
        self.assertEqual(manifest["supported_models"], ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"])
        self.assertEqual(len(manifest["provenance"]["assets"]), 10)
        manifest["provenance"]["assets"].clear()
        self.assertEqual(len(self.counter.manifest()["provenance"]["assets"]), 10)

    def test_native_payload_matches_transport_and_excludes_router_state(self):
        from routerbench.bfcl_litellm import BFCLLiteLLMGenerator
        metadata = {"fixture/model": {"returned_model_ids": ["model"], "context_window": 131072, "api_key_env": "FIXTURE_KEY"}}
        transport = BFCLLiteLLMGenerator(metadata, {"max_output_tokens": 200, "temperature": 0, "timeout": 30})
        current = request()
        self.assertEqual(native_request_payload(current), transport.request_payload(current))
        self.assertEqual(self.counter.estimate(current)["input_utf8_bytes"], transport.request_input_bytes(current))
        changed = BFCLRequest(current.messages, current.tools, replace(current.state, step=100,
            previous_errors=("BackendActionError",), context_utf8_bytes=999999))
        self.assertEqual(self.counter.estimate(current), self.counter.estimate(changed))

    def test_tools_unrepresented_schema_fields_and_all_parallel_calls_are_counted(self):
        current = request()
        calls = [{"id": str(i), "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"name": "record" + str(i)})}}
                 for i in range(3)]
        messages = current.messages + ({"role": "assistant", "content": None, "tool_calls": calls},)
        parallel = BFCLRequest(messages, current.tools, current.state)
        estimate = self.counter.estimate(parallel)
        self.assertIsNone(estimate["published_harmony_tokens"])
        self.assertEqual(estimate["published_template_status"], "unsupported_parallel_or_empty_tool_batch")
        native = json.dumps(native_request_payload(parallel), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.assertEqual(estimate["serialized_payload_tokens"], len(self.counter._encoding.encode_ordinary(native)))
        self.assertGreater(estimate["serialized_payload_tokens"], self.counter.estimate(current)["serialized_payload_tokens"])
        changed = deepcopy(current.tools)
        changed[0]["function"]["parameters"]["examples"] = [{"name": "public schema value " * 100}]
        with_extra_field = BFCLRequest(current.messages, changed, current.state)
        self.assertGreater(self.counter.estimate(with_extra_field)["serialized_payload_tokens"],
                           self.counter.estimate(current)["serialized_payload_tokens"])

    def test_history_unicode_newlines_and_literal_harmony_markers_are_preserved(self):
        content = "á 中文\n    " + "<|start|> " * 40
        current = request(messages=({"role": "user", "content": content},), tools=())
        payload = native_request_payload(current)
        self.assertEqual(payload, {"messages": [{"role": "user", "content": content}]})
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(self.counter.estimate(current)["serialized_payload_tokens"], len(self.counter._encoding.encode_ordinary(encoded)))
        expanded = request(messages=current.messages + ({"role": "assistant", "content": "a previous answer " * 100},
                                                       {"role": "user", "content": "Current turn only."}), tools=())
        self.assertGreater(self.counter.estimated_input_tokens(expanded), self.counter.estimated_input_tokens(current))

    def test_single_native_call_renders_without_changing_null_content_or_arguments(self):
        current = request()
        call = {"id": "one", "type": "function", "function": {"name": "lookup", "arguments": '{"name":"example"}'}}
        current = BFCLRequest(current.messages + ({"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "one", "content": '{"result":"ok"}'}), current.tools, current.state)
        before = current.to_dict()
        self.assertEqual(self.counter.estimate(current)["published_template_status"], "rendered_published_template")
        self.assertEqual(current.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
