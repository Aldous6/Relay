import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from routerbench.context import window_token_ids
from routerbench.features import FEATURE_NAMES, extract_features
from routerbench.representations import PromptRepresentation


def prompt_frame(texts):
    return pd.DataFrame({"prompt_text": texts})


class RepresentationTests(unittest.TestCase):
    def test_validation_does_not_change_vocabulary_or_scaler(self):
        train = prompt_frame(["alpha common", "beta common"])
        validation = prompt_frame(["unseenword " * 100])
        transformer = PromptRepresentation(mode="hybrid", max_features=100, min_df=1).fit(train)
        word_vectorizer = dict(transformer.lexical_.transformer_list)["word"]
        vocabulary = word_vectorizer.vocabulary_.copy()
        scale = transformer.structural_scaler_.scale_.copy()
        transformed = transformer.transform(validation)
        self.assertEqual(word_vectorizer.vocabulary_, vocabulary)
        self.assertNotIn("unseenword", vocabulary)
        np.testing.assert_array_equal(transformer.structural_scaler_.scale_, scale)
        self.assertEqual(transformed.shape[1], transformer.n_features_out_)

    def test_lexical_fit_observes_tail_of_long_prompt(self):
        text = "common " * 2000 + "tailmarker"
        transformer = PromptRepresentation(mode="lexical", max_features=100, min_df=1).fit(prompt_frame([text]))
        words = dict(transformer.lexical_.transformer_list)["word"]
        self.assertIn("tailmarker", words.vocabulary_)

    def test_precomputed_structural_values_match_extraction_and_preserve_order(self):
        texts = ["AAA 12?\n  x", "b c d"]
        plain = prompt_frame(texts)
        precomputed = pd.concat([plain, extract_features(texts)], axis=1)
        transformer = PromptRepresentation(mode="structural").fit(plain)
        computed = transformer.transform(plain)
        supplied = transformer.transform(precomputed.loc[:, list(reversed(precomputed.columns))])
        np.testing.assert_allclose(computed.toarray(), supplied.toarray())
        self.assertEqual(computed.shape, (2, 14))
        self.assertTrue(sparse.isspmatrix_csr(computed))
        self.assertEqual(computed.dtype, np.float32)

    def test_hybrid_blocks_sparse_finite_and_unit_normalized(self):
        frame = prompt_frame(["first question", "second question"])
        transformer = PromptRepresentation(mode="hybrid", min_df=1)
        result = transformer.fit_transform(frame)
        self.assertTrue(sparse.isspmatrix_csr(result))
        self.assertTrue(np.isfinite(result.data).all())
        np.testing.assert_allclose(np.asarray(result.multiply(result).sum(axis=1)).ravel(), 1, rtol=1e-5)
        self.assertEqual(len(transformer.get_feature_names_out()), result.shape[1])
        self.assertEqual(clone(transformer).get_params()["mode"], "hybrid")

    def test_outcome_identifiers_partial_features_and_invalid_text_rejected(self):
        for extra in ("tier", "cost", "sample_id", "eval_name", "model|total_cost"):
            with self.subTest(extra=extra):
                frame = prompt_frame(["some prompt"])
                frame[extra] = 1
                with self.assertRaises(ValueError):
                    PromptRepresentation(mode="structural").fit(frame)
        partial = prompt_frame(["some prompt"])
        partial[FEATURE_NAMES[0]] = 11
        with self.assertRaises(ValueError):
            PromptRepresentation(mode="structural").fit(partial)
        with self.assertRaises(TypeError):
            PromptRepresentation(mode="structural").fit(prompt_frame([None]))

    def test_empty_lexical_vocabulary_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "min_df"):
            PromptRepresentation(mode="lexical", min_df=1).fit(prompt_frame([""]))


class WindowTests(unittest.TestCase):
    def test_all_positions_covered_and_offsets_reconstruct_tokens(self):
        tokens = list(range(13))
        windows = window_token_ids(tokens, max_tokens=5, overlap=2)
        self.assertEqual([(window.start, window.end) for window in windows], [(0, 5), (3, 8), (6, 11), (9, 13)])
        covered = set()
        for window in windows:
            covered.update(range(window.start, window.end))
            self.assertEqual(window.token_ids, tuple(tokens[window.start:window.end]))
            self.assertLessEqual(len(window.token_ids), 5)
        self.assertEqual(covered, set(range(len(tokens))))

    def test_empty_exact_boundary_and_no_overlap(self):
        self.assertEqual(window_token_ids([], 5), [])
        self.assertEqual(len(window_token_ids([1, 2, 3, 4, 5], 5, 2)), 1)
        windows = window_token_ids(list(range(12)), 5)
        self.assertEqual(tuple(token for window in windows for token in window.token_ids), tuple(range(12)))

    def test_invalid_window_configuration_rejected(self):
        for capacity, overlap in ((0, 0), (5, 5), (5, -1), (True, 0), (5, 1.5)):
            with self.subTest(capacity=capacity, overlap=overlap):
                with self.assertRaises(ValueError):
                    window_token_ids([1, 2, 3], capacity, overlap)
        with self.assertRaises(TypeError):
            window_token_ids("abc", 5)
        with self.assertRaises(ValueError):
            window_token_ids([1, -2], 5)


if __name__ == "__main__":
    unittest.main()
