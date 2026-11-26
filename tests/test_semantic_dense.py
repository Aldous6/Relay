"""Dense semantic composition contracts with a mock local encoder only."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from routerbench.features import FEATURE_NAMES, extract_features
from routerbench.semantic_dense import DenseSemanticRepresentation


class DenseMockEncoder:
    def __init__(self, spec):
        self.calls = []

    def encode(self, texts, use_cache=True):
        self.calls.append((list(texts), use_cache))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, :3] = [1, len(text), ord(text[-1])]
        return vectors

    def __getstate__(self):
        raise AssertionError("An encoder runtime must not be serialized.")


def dense_mock_factory(spec):
    return DenseMockEncoder(spec)


def frame(texts, precomputed=False):
    result = pd.DataFrame({"prompt_text": texts})
    return pd.concat([result, extract_features(texts)], axis=1) if precomputed else result


def transformer(mode="semantic_hybrid"):
    return DenseSemanticRepresentation(mode=mode, encoder_spec={},
                                       encoder_factory=dense_mock_factory)


class DenseSemanticTests(unittest.TestCase):
    def test_fit_never_constructs_or_calls_encoder_and_scaler_uses_fit_only(self):
        factory = Mock(side_effect=AssertionError("Fit must not load the encoder."))
        train = frame(["a", "a b"], precomputed=True)
        dense = DenseSemanticRepresentation(encoder_spec={}, encoder_factory=factory).fit(train)
        factory.assert_not_called()
        self.assertFalse(hasattr(dense.semantic_, "encoder_"))
        self.assertEqual(dense.n_features_out_, 398)
        np.testing.assert_allclose(dense.semantic_.structural_scaler_.max_abs_,
                                   train.loc[:, list(FEATURE_NAMES)].max().to_numpy())
        dense.encoder_factory = dense_mock_factory
        dense.semantic_.encoder_factory = dense_mock_factory
        before = dense.semantic_.structural_scaler_.max_abs_.copy()
        validation = frame(["UPPERCASE "*100], precomputed=True)
        result = dense.transform(validation)
        self.assertEqual(result.shape, (1, 398))
        np.testing.assert_array_equal(dense.semantic_.structural_scaler_.max_abs_, before)

    def test_full_text_cache_setting_and_input_order_reach_nested_encoder(self):
        long_text = "word "*3000+"TAIL-MUST-BE-PRESENT"
        inputs = frame([long_text, "second"])
        inputs.index = [90, 10]
        original = inputs.copy(deep=True)
        dense = transformer(mode="semantic").fit(inputs)
        result = dense.transform(inputs)
        self.assertEqual(result.shape, (2, 384))
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(dense.semantic_.encoder_.calls[-1], ([long_text, "second"], True))
        dense.use_cache = False
        reversed_result = dense.transform(inputs.iloc[::-1])
        self.assertEqual(dense.semantic_.encoder_.calls[-1], (["second", long_text], False))
        np.testing.assert_array_equal(result[::-1], reversed_result)
        pd.testing.assert_frame_equal(inputs, original)

    def test_clone_is_unfitted_and_names_are_preserved(self):
        dense = transformer().fit(frame(["first", "second"]))
        dense.transform(frame(["first", "second"]))
        copied = clone(dense)
        self.assertEqual(copied.get_params(), dense.get_params())
        self.assertFalse(hasattr(copied, "semantic_"))
        with self.assertRaises(NotFittedError):
            copied.transform(frame(["third"]))
        names = dense.get_feature_names_out()
        self.assertEqual(len(names), 398)
        self.assertEqual(names[-14:].tolist(), [f"structural__{name}" for name in FEATURE_NAMES])
        names[0] = "mutated"
        self.assertNotEqual(dense.get_feature_names_out()[0], "mutated")

    def test_joblib_drops_nested_runtime_and_recreates_it_lazily(self):
        inputs = frame(["first", "second"], precomputed=True)
        dense = transformer()
        expected = dense.fit_transform(inputs)
        self.assertTrue(hasattr(dense.semantic_, "encoder_"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.joblib"
            joblib.dump(dense, path)
            restored = joblib.load(path)
        self.assertFalse(hasattr(restored.semantic_, "encoder_"))
        np.testing.assert_array_equal(restored.transform(inputs), expected)
        self.assertTrue(hasattr(restored.semantic_, "encoder_"))

    def test_shape_nonfinite_and_width_are_rejected_before_densification(self):
        inputs = frame(["first", "second"])
        dense = transformer().fit(inputs)
        bad = sparse.csr_matrix(([np.nan], ([0], [0])), shape=(2, 398))
        for value in (sparse.csr_matrix((2, 513)), sparse.csr_matrix((1, 398)),
                      sparse.csr_matrix((2, 384)), bad, np.zeros((2, 398))):
            with self.subTest(shape=value.shape), patch.object(dense.semantic_, "transform", return_value=value), \
                    patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("Unexpected densification")):
                with self.assertRaises(ValueError):
                    dense.transform(inputs)

    def test_input_contract_and_refit_drop_old_runtime(self):
        dense = transformer()
        valid = frame(["first", "second"])
        for bad in (valid.assign(score=1), valid.assign(n_chars=3), valid.iloc[:0],
                    pd.concat([valid.iloc[[0]]]*2), frame([" "])):
            with self.assertRaises(ValueError):
                dense.fit(bad)
        dense.fit_transform(valid)
        previous = dense.semantic_
        dense.fit(frame(["different", "fitting texts"]))
        self.assertIsNot(dense.semantic_, previous)
        self.assertFalse(hasattr(dense.semantic_, "encoder_"))
        self.assertEqual(dense.transform(valid).shape, (2, 398))


if __name__ == "__main__":
    unittest.main()
