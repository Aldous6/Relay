"""Semantic representation contracts without network access or model weights."""
import pickle
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from routerbench.features import FEATURE_NAMES, extract_features
from routerbench.semantic_representation import SemanticPromptRepresentation


class MockEncoder:
    def __init__(self, spec):
        self.spec = spec
        self.calls = []

    def encode(self, texts, use_cache=True):
        self.calls.append((list(texts), use_cache))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, :3] = [1, len(text), ord(text[-1]) if text else 0]
        if self.spec.get("behavior") == "wrong_shape":
            return vectors[:, :383]
        if self.spec.get("behavior") == "not_finite":
            vectors[0, 0] = np.nan
        return vectors

    def __getstate__(self):
        raise AssertionError("Runtime encoder instances must never be serialized.")


def mock_encoder_factory(spec):
    return MockEncoder(spec)


def frame(texts, precomputed=False):
    inputs = pd.DataFrame({"prompt_text": texts})
    if precomputed:
        inputs = pd.concat([inputs, extract_features(texts)], axis=1)
    return inputs


class SemanticRepresentationTests(unittest.TestCase):
    def make_transformer(self, mode="semantic", spec=None):
        return SemanticPromptRepresentation(mode=mode, encoder_spec={} if spec is None else spec,
                                            encoder_factory=mock_encoder_factory)

    def test_fit_is_lazy_and_fit_transform_returns_semantic_sparse_features(self):
        inputs = frame(["short", "a longer prompt"])
        transformer = self.make_transformer().fit(inputs)
        self.assertFalse(hasattr(transformer, "encoder_"))
        transformed = transformer.transform(inputs)
        self.assertTrue(sparse.isspmatrix_csr(transformed))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertEqual(transformed.shape, (2, 384))
        np.testing.assert_allclose(np.asarray(transformed.multiply(transformed).sum(axis=1)).ravel(), [1, 1], atol=1e-6)
        self.assertEqual(len(transformer.get_feature_names_out()), 384)
        self.assertTrue(all(name.startswith("semantic__") for name in transformer.get_feature_names_out()))

    def test_cache_setting_and_complete_text_reach_encoder_in_order(self):
        long_text = "word " * 2000 + "TAIL-MUST-BE-PRESENT"
        inputs = frame([long_text, "second"])
        inputs.index = [90, 10]
        transformer = self.make_transformer().fit(inputs)
        first = transformer.transform(inputs)
        self.assertEqual(transformer.encoder_.calls[-1], ([long_text, "second"], True))
        transformer.use_cache = False
        reversed_result = transformer.transform(inputs.iloc[::-1])
        self.assertEqual(transformer.encoder_.calls[-1], (["second", long_text], False))
        np.testing.assert_allclose(first.toarray()[::-1], reversed_result.toarray())

    def test_structural_scaler_fits_only_training_rows(self):
        train = frame(["a", "a b"], precomputed=True)
        transformer = self.make_transformer(mode="semantic_hybrid").fit(train)
        maxima = transformer.structural_scaler_.max_abs_.copy()
        validation = frame(["UPPERCASE " * 100], precomputed=True)
        result = transformer.transform(validation)
        np.testing.assert_array_equal(transformer.structural_scaler_.max_abs_, maxima)
        self.assertEqual(result.shape, (1, 398))
        self.assertEqual(transformer.get_feature_names_out()[-14:].tolist(), [f"structural__{name}" for name in FEATURE_NAMES])
        self.assertTrue(np.isfinite(result.data).all())

    def test_precomputed_and_raw_structural_features_have_same_transform(self):
        texts = ["ABC 123?", "line one\n    line two"]
        transformer = self.make_transformer(mode="semantic_hybrid").fit(frame(texts, precomputed=True))
        np.testing.assert_allclose(transformer.transform(frame(texts)).toarray(),
                                   transformer.transform(frame(texts, precomputed=True)).toarray(), atol=1e-7)

    def test_input_boundary_rejects_outcomes_and_partial_structural_columns(self):
        inputs = frame(["prompt"])
        transformer = self.make_transformer()
        for invalid in (inputs.assign(score=1), inputs.assign(n_chars=6), frame([None])):
            with self.subTest(columns=invalid.columns.tolist()), self.assertRaises((TypeError, ValueError)):
                transformer.fit(invalid)
        transformer.fit(inputs)
        with self.assertRaises(ValueError):
            transformer.transform(inputs.assign(eval_name="benchmark"))

    def test_encoder_shape_and_finite_values_validated(self):
        inputs = frame(["prompt"])
        for behavior in ("wrong_shape", "not_finite"):
            with self.subTest(behavior=behavior), self.assertRaises(ValueError):
                self.make_transformer(spec={"behavior": behavior}).fit_transform(inputs)

    def test_pickle_drops_runtime_encoder_and_recreates_it_lazily(self):
        inputs = frame(["first", "second"])
        transformer = self.make_transformer(mode="semantic_hybrid")
        expected = transformer.fit_transform(inputs)
        self.assertTrue(hasattr(transformer, "encoder_"))
        live_encoder = transformer.encoder_
        restored = pickle.loads(pickle.dumps(transformer))
        self.assertIs(transformer.encoder_, live_encoder)
        self.assertFalse(hasattr(restored, "encoder_"))
        np.testing.assert_allclose(restored.transform(inputs).toarray(), expected.toarray())
        self.assertTrue(hasattr(restored, "encoder_"))

    def test_refit_discards_runtime_encoder_without_encoding_new_rows(self):
        transformer = self.make_transformer()
        transformer.fit_transform(frame(["first"]))
        transformer.fit(frame(["second"]))
        self.assertFalse(hasattr(transformer, "encoder_"))


if __name__ == "__main__":
    unittest.main()
