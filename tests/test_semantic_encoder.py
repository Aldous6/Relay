import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from routerbench.data import sha256_file
from routerbench.semantic_encoder import ROOT, WindowedE5Encoder, aggregate_windows


def mock_encoder(max_tokens=512, overlap=64):
    encoder = WindowedE5Encoder.__new__(WindowedE5Encoder)
    encoder.spec = {"embedding_dimension": 384, "prefix": "query: ",
                    "max_tokens": max_tokens, "overlap": overlap, "batch_size": 1}
    encoder._prefix_ids = [41, 1294, 12]
    encoder._capacity = max_tokens - 5
    encoder._cls, encoder._sep, encoder._pad = 0, 2, 1
    encoder._ensure_runtime = Mock()
    encoder._tokenizer = Mock()
    return encoder


def valid_spec():
    return {
        "embedding_dimension": 384, "prefix": "query: ", "max_tokens": 512,
        "overlap": 64, "batch_size": 1, "threads": 4,
        "token_assembly_version": "e5-joint-prefix-tokenization-v3",
        "aggregation": "new_token_weighted_mean_of_l2_window_embeddings_then_l2",
        "model_dir": "artifacts/mock-encoder", "model_id": "mock/e5", "revision": "revision-1",
        "model_file": "model.onnx", "tokenizer_file": "tokenizer.json", "model_sha256": "not-used-by-runtime-mock",
        "packages": {"onnxruntime": "mock-ort", "tokenizers": "mock-tokenizers"},
    }


class AggregationTests(unittest.TestCase):
    def test_full_coverage_weights_preserve_each_windows_contribution(self):
        vectors = np.array([[3, 0], [0, 4]], dtype=np.float32)
        combined = aggregate_windows(vectors, np.array([3, 1]))
        expected = np.array([3, 1], dtype=np.float32)/np.sqrt(10)
        np.testing.assert_allclose(combined, expected, rtol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(combined)), 1.0, places=6)

    def test_invalid_or_cancelled_embeddings_rejected(self):
        for vectors, weights in (([[0, 0]], [1]), ([[1, 0]], [0]), ([[np.nan, 1]], [1]), ([[1, 0], [-1, 0]], [1, 1])):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                aggregate_windows(vectors, weights)


class EncodingContractTests(unittest.TestCase):
    def test_joint_prefix_preserves_boundary_before_literal_special_token(self):
        encoder = mock_encoder()
        encoder._tokenizer.encode_batch.return_value = [SimpleNamespace(ids=[41, 1294, 12, 6, 0, 812])]
        encoder._infer_windows = Mock(return_value=np.ones((1, 384), dtype=np.float32))
        vectors, stats, _, _ = encoder._encode_new(["<s> literal"])
        encoder._tokenizer.encode_batch.assert_called_once_with(["query: <s> literal"], add_special_tokens=False)
        self.assertEqual(encoder._infer_windows.call_args.args[0], [[0, 41, 1294, 12, 6, 0, 812, 2]])
        self.assertEqual(stats[0]["input_tokens"], 3)
        self.assertEqual(stats[0]["covered_tokens"], 3)
        self.assertEqual(stats[0]["truncated_tokens"], 0)
        self.assertEqual(np.asarray(vectors).shape, (1, 384))

    def test_length_bucketing_keeps_owners_and_novel_token_weights(self):
        encoder = mock_encoder(max_tokens=10, overlap=2)
        encoder._tokenizer.encode_batch.return_value = [
            SimpleNamespace(ids=[41, 1294, 12, *range(10, 22)]),
            SimpleNamespace(ids=[41, 1294, 12, 99]),
        ]
        seen_payloads = []

        def infer(sequences):
            self.assertEqual(len(sequences), 1)
            sequence = sequences[0]
            self.assertLessEqual(len(sequence), 10)
            self.assertEqual(sequence[:4], [0, 41, 1294, 12])
            self.assertEqual(sequence[-1], 2)
            payload = sequence[4:-1]
            seen_payloads.append(payload)
            vector = np.zeros(384, dtype=np.float32)
            vector[0], vector[1] = sum(payload), 1
            return vector[None, :]

        encoder._infer_windows = infer
        vectors, stats, _, _ = encoder._encode_new(["long prompt", "short prompt"])
        self.assertEqual(seen_payloads[0], [99])
        expected_windows = [[10, 11, 12, 13, 14], [13, 14, 15, 16, 17],
                            [16, 17, 18, 19, 20], [19, 20, 21]]
        self.assertEqual({tuple(payload) for payload in seen_payloads[1:]}, {tuple(payload) for payload in expected_windows})
        normalized = []
        for payload in expected_windows:
            vector = np.array([sum(payload), 1.0], dtype=np.float32)
            normalized.append(vector / np.linalg.norm(vector))
        expected = np.average(normalized, axis=0, weights=[5, 3, 3, 1])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0][:2], expected, rtol=1e-6)
        self.assertEqual(stats[0]["n_windows"], 4)
        self.assertEqual(stats[0]["input_tokens"], stats[0]["covered_tokens"])
        self.assertEqual(stats[1]["n_windows"], 1)
        short = np.array([99, 1], dtype=np.float32)
        np.testing.assert_allclose(vectors[1][:2], short / np.linalg.norm(short), rtol=1e-6)

    def test_unstable_prefix_and_empty_payload_reject_before_inference(self):
        encoder = mock_encoder()
        encoder._infer_windows = Mock()
        for ids in ([999, 41, 12, 10], [41, 1294, 12]):
            with self.subTest(ids=ids):
                encoder._tokenizer.encode_batch.return_value = [SimpleNamespace(ids=ids)]
                with self.assertRaises(ValueError):
                    encoder._encode_new(["text"])
        encoder._infer_windows.assert_not_called()

    def test_individual_string_is_rejected_before_assets_or_runtime(self):
        encoder = WindowedE5Encoder.__new__(WindowedE5Encoder)
        encoder._verify_assets = Mock()
        for value in ("hello", b"hello"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                encoder.encode(value)
        encoder._verify_assets.assert_not_called()


class OnnxInterfaceTests(unittest.TestCase):
    def test_pooling_ignores_padding_and_preserves_model_inputs(self):
        encoder = mock_encoder()
        encoder._input_names = {"input_ids", "attention_mask", "token_type_ids"}
        encoder._output_name = "last_hidden_state"
        output = np.empty((2, 5, 384), dtype=np.float32)
        output[0] = np.array([1, 2, 3, 999, 999])[:, None]
        output[1] = np.array([2, 4, 6, 8, 10])[:, None]
        encoder._session = Mock()
        encoder._session.run.return_value = [output]
        result = encoder._infer_windows([[0, 7, 2], [0, 8, 9, 10, 2]])
        np.testing.assert_allclose(result[0], np.full(384, 2))
        np.testing.assert_allclose(result[1], np.full(384, 6))
        outputs, feeds = encoder._session.run.call_args.args
        self.assertEqual(outputs, ["last_hidden_state"])
        np.testing.assert_array_equal(feeds["attention_mask"], [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(feeds["input_ids"][0], [0, 7, 2, 1, 1])
        np.testing.assert_array_equal(feeds["token_type_ids"], np.zeros((2, 5), dtype=np.int64))
        self.assertTrue(all(value.dtype == np.int64 for value in feeds.values()))

    def test_pooled_or_nonfinite_onnx_outputs_are_rejected(self):
        encoder = mock_encoder()
        encoder._input_names = {"input_ids", "attention_mask"}
        encoder._output_name = "last_hidden_state"
        encoder._session = Mock()
        for output in (np.ones((1, 384)), np.full((1, 3, 384), np.nan)):
            with self.subTest(shape=output.shape):
                encoder._session.run.return_value = [output]
                with self.assertRaises(ValueError):
                    encoder._infer_windows([[0, 7, 2]])

    def test_initialization_failure_can_be_retried_without_partial_runtime(self):
        spec = valid_spec()
        encoder = WindowedE5Encoder(spec)
        encoder._verify_assets = Mock()
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name=name, shape=["batch", "tokens"], type="tensor(int64)")
                                           for name in ("input_ids", "attention_mask", "token_type_ids")]
        session.get_outputs.return_value = [SimpleNamespace(name="last_hidden_state")]
        session.get_providers.return_value = ["CPUExecutionProvider"]
        tokenizer = Mock()
        tokenizer.encode.return_value = SimpleNamespace(ids=[41, 1294, 12])
        tokenizer.token_to_id.side_effect = {"<s>": 0, "</s>": 2, "<pad>": 1}.get
        tokenizer_factory = Mock()
        tokenizer_factory.from_file.side_effect = [ValueError("invalid tokenizer file"), tokenizer]
        fake_ort = SimpleNamespace(__version__="mock-ort", SessionOptions=SimpleNamespace,
                                   InferenceSession=Mock(return_value=session))
        fake_tokenizers = SimpleNamespace(__version__="mock-tokenizers", Tokenizer=tokenizer_factory)
        with patch.dict("sys.modules", {"onnxruntime": fake_ort, "tokenizers": fake_tokenizers}):
            with self.assertRaisesRegex(ValueError, "invalid tokenizer"):
                encoder._ensure_runtime()
            self.assertIsNone(encoder._session)
            self.assertIsNone(encoder._tokenizer)
            encoder._ensure_runtime()
        self.assertIs(encoder._session, session)
        self.assertIs(encoder._tokenizer, tokenizer)
        self.assertEqual(encoder.runtime_description_["payload_capacity"], 507)


class CacheIdentityTests(unittest.TestCase):
    def test_runtime_and_preprocessing_changes_invalidate_namespace(self):
        # Tiny fake assets exercise manifest identity without model weights or
        # inference packages. They remain inside ROOT as required by the loader.
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
            folder = Path(temporary)
            (folder / "model.onnx").write_bytes(b"model fixture")
            (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
            spec = valid_spec()
            spec["model_dir"] = folder.relative_to(ROOT).as_posix()
            spec["model_sha256"] = sha256_file(folder / "model.onnx")
            manifest = {"model_id": spec["model_id"], "revision": spec["revision"], "files": {
                name: {"sha256": sha256_file(folder / name)} for name in ("model.onnx", "tokenizer.json")}}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            original = WindowedE5Encoder(spec)
            original._verify_assets()
            for changes in ({"threads": 2}, {"overlap": 32}, {"max_tokens": 256},
                            {"packages": {"onnxruntime": "other", "tokenizers": "mock-tokenizers"}}):
                with self.subTest(changes=changes):
                    changed = WindowedE5Encoder({**spec, **changes})
                    changed._verify_assets()
                    self.assertNotEqual(original.cache_namespace_, changed.cache_namespace_)
            # batch_size and assembly version are part of identity in source,
            # while unsupported values are rejected before any asset access.
            for changes in ({"batch_size": 16}, {"token_assembly_version": "old-prefix-v1"}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    WindowedE5Encoder({**spec, **changes})


if __name__ == "__main__":
    unittest.main()
