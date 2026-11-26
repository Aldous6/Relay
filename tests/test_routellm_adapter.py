"""Local RouteLLM adapter contracts, without torch, model loads or network."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.integrations import routellm


class RouteLLMAdapterTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("No network in adapter unit tests"))
        guard.start()
        self.addCleanup(guard.stop)

    @contextlib.contextmanager
    def checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            digests = {}
            for name, content in (("model.safetensors", b"fake weights"),
                                  ("config.json", b'{"fake":"configuration"}'),
                                  ("tokenizer.json", b'{"fake":"tokenizer"}')):
                (directory / name).write_bytes(content)
                digests[name] = hashlib.sha256(content).hexdigest()
            with patch.dict(routellm.FILES, digests, clear=True):
                yield directory

    def test_import_does_not_load_optional_libraries_or_open_connections(self):
        # A fresh interpreter proves this even if other test modules loaded ML packages.
        code = """
import importlib.abc, json, socket, sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'numpy'}:
            raise AssertionError('Optional import on module import: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Network forbidden'))
sys.path.insert(0, sys.argv[1])
from relay_router.integrations import routellm
print(json.dumps({'imported': True, 'optional_loaded': sorted(set(sys.modules) & {'torch', 'transformers', 'numpy'})}))
"""
        result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src")],
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"imported": True, "optional_loaded": []})

    def test_pinned_file_contract_covers_weights_tokenizer_configuration_and_license(self):
        self.assertEqual(set(routellm.FILES), {"model.safetensors", "config.json", "tokenizer.json",
                                             "tokenizer_config.json", "special_tokens_map.json", "LICENSE"})
        self.assertTrue(all(len(value) == 64 and set(value) <= set("0123456789abcdef")
                            for value in routellm.FILES.values()))

    def test_verifies_every_input_file_and_rejects_missing_or_changed_file(self):
        with self.checkpoint() as directory:
            self.assertEqual(routellm.verify_checkpoint(directory), directory)
            (directory / "tokenizer.json").write_bytes(b"different tokenizer")
            with self.assertRaisesRegex(ValueError, "tokenizer.json"):
                routellm.verify_checkpoint(directory)
        with self.checkpoint() as directory:
            (directory / "config.json").unlink()
            with self.assertRaises(FileNotFoundError):
                routellm.verify_checkpoint(directory)

    def test_bad_checkpoint_fails_before_even_resolving_optional_packages(self):
        with self.checkpoint() as directory:
            (directory / "model.safetensors").write_bytes(b"corrupt")
            with patch.object(importlib.metadata, "version", side_effect=AssertionError("Package lookup too early")) as version:
                with self.assertRaisesRegex(ValueError, "model.safetensors"):
                    routellm.LocalBERTScorer(directory)
                version.assert_not_called()

    def test_auxiliary_overrides_cannot_bypass_the_verified_files(self):
        for filename in ("added_tokens.json", "adapter_config.json", "generation_config.json"):
            with self.subTest(filename=filename), self.checkpoint() as directory:
                (directory / filename).write_text("{}", encoding="utf-8")
                with patch.object(importlib.metadata, "version", side_effect=AssertionError("Package lookup too early")):
                    with self.assertRaisesRegex(ValueError, "auxiliary"):
                        routellm.LocalBERTScorer(directory)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(importlib.metadata, "version", side_effect=AssertionError("Package lookup too early")) as version:
                with self.assertRaises(FileNotFoundError):
                    routellm.LocalBERTScorer(directory)
                version.assert_not_called()

    def test_invalid_threads_rejected_before_file_or_package_access(self):
        with patch.object(routellm, "verify_checkpoint") as verify:
            for value in (True, False, 0, 65, 1.5, "4"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    routellm.LocalBERTScorer("not-used", threads=value)
            verify.assert_not_called()

    def test_missing_or_mismatched_dependency_fails_before_model_loader(self):
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = Mock()
        transformers.AutoModelForSequenceClassification = Mock()
        with self.checkpoint() as directory, patch.dict(sys.modules, {"transformers": transformers}):
            with patch.object(importlib.metadata, "version", return_value="0.0.0"):
                with self.assertRaisesRegex(ValueError, "torch==2.9.0"):
                    routellm.LocalBERTScorer(directory)
            with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError("torch")):
                with self.assertRaises(importlib.metadata.PackageNotFoundError):
                    routellm.LocalBERTScorer(directory)
            transformers.AutoTokenizer.from_pretrained.assert_not_called()
            transformers.AutoModelForSequenceClassification.from_pretrained.assert_not_called()

    def test_constructor_uses_only_verified_local_safetensors_and_loads_once(self):
        torch = ModuleType("torch")
        torch.set_num_threads = Mock()
        transformers = ModuleType("transformers")
        tokenizer = Mock(name="resident_tokenizer")
        model = Mock(name="resident_model")
        model.to.return_value = model
        model.eval.return_value = model
        transformers.AutoTokenizer = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers.AutoModelForSequenceClassification = Mock()
        transformers.AutoModelForSequenceClassification.from_pretrained.return_value = model
        versions = {"torch": "2.9.0+cpu", "transformers": "4.57.1"}
        with self.checkpoint() as directory, patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.object(
                importlib.metadata, "version", side_effect=versions.__getitem__):
            scorer = routellm.LocalBERTScorer(directory, threads=2)
            self.assertIs(scorer._tokenizer, tokenizer)
            self.assertIs(scorer._model, model)
            self.assertGreaterEqual(scorer.load_seconds, 0)
            torch.set_num_threads.assert_called_once_with(2)
            transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
                str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
            transformers.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(
                str(directory), local_files_only=True, trust_remote_code=False, use_safetensors=True, num_labels=3)
            model.to.assert_called_once_with("cpu")
            model.eval.assert_called_once_with()
            self.assertEqual(scorer.provenance["revision"], routellm.REVISION)
            self.assertEqual(scorer.provenance["weights_sha256"], routellm.FILES["model.safetensors"])
            self.assertEqual(scorer.provenance["max_input_tokens"], 512)
            self.assertFalse(scorer.provenance["success_probability"])
            self.assertIsNone(scorer.provenance["router_cost_usd"])

    def test_scorer_rejects_invalid_text_before_using_model(self):
        scorer = routellm.LocalBERTScorer.__new__(routellm.LocalBERTScorer)
        for text in (None, "", " \n ", 123, []):
            with self.subTest(text=text), self.assertRaises(ValueError):
                scorer(text)

    def test_pilot_factory_keeps_exact_pair_threshold_and_copies_provenance(self):
        scorer = Mock()
        scorer.provenance = {"checkpoint": routellm.CHECKPOINT, "revision": routellm.REVISION,
                             "router_cost_usd": None, "success_probability": False}
        scorer.side_effect = [{"score": .57, "input_tokens": 9, "truncated_tokens": 0},
                              {"score": .58, "input_tokens": 10, "truncated_tokens": 0}]
        with patch.object(routellm, "LocalBERTScorer", return_value=scorer) as load:
            selector, provenance = routellm.load_groq_pilot_selector("local/checkpoint")
        load.assert_called_once_with("local/checkpoint")
        self.assertEqual(selector.weak_model, "groq/openai/gpt-oss-20b")
        self.assertEqual(selector.strong_model, "groq/openai/gpt-oss-120b")
        self.assertEqual(selector.threshold, .58)
        self.assertEqual(selector.selector_id, "routellm-bert-groq-pilot-v2")
        self.assertEqual(selector.select("first request")["model"], selector.weak_model)
        self.assertEqual(selector.select("second request")["model"], selector.strong_model)
        self.assertEqual([call.args for call in scorer.call_args_list], [("first request",), ("second request",)])
        self.assertEqual(provenance, scorer.provenance)
        self.assertIsNot(provenance, scorer.provenance)


if __name__ == "__main__":
    unittest.main()
