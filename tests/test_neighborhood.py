"""Fit-only retrieval, calibrated pairwise contracts and bounded inference."""
from io import BytesIO
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from routerbench.neighborhood import NeighborhoodRegretRouter, _neighbor_features
from routerbench.pairwise import OUTCOME_CLASSES


def fixture():
    patterns = ('proof logic theorem', 'short lookup name', 'document shared fact', 'unsolved puzzle grid')
    index = pd.Index(range(48), name='source_row')
    x = pd.DataFrame({'prompt_text': [f'request {i} {patterns[i%4]}' for i in index]}, index=index)
    scores = pd.DataFrame({'strong': [1, 0, 1, 0]*12, 'weak': [0, 1, 1, 0]*12}, index=index, dtype=float)
    costs = pd.DataFrame({'strong': np.full(48, .01), 'weak': np.full(48, .001)}, index=index)
    return x, scores, costs


def model(**kwargs):
    return NeighborhoodRegretRouter('strong', 'weak', max_features=100, min_df=1,
                                    n_neighbors=4, query_batch_size=3, **kwargs)


def fit(router=None):
    x, y, c = fixture()
    router = model() if router is None else router
    return router.fit(x.iloc[:24], y.iloc[:24], c.iloc[:24], x.iloc[24:36], y.iloc[24:36])


class NeighborhoodTests(unittest.TestCase):
    def test_shared_failures_are_equal_and_all_fit_rows_enter_bank(self):
        router = fit()
        self.assertEqual(router.fit_rows_, 24)
        self.assertEqual(router.fit_vectors_.shape[0], 24)
        self.assertEqual(router.fit_codes_.shape, (24,))
        self.assertEqual(router.fit_codes_[:4].tolist(), [0, 1, 2, 2])
        self.assertEqual(router.target_counts_['fit'], {'strong_only': 6, 'weak_only': 6, 'equal': 12})
        self.assertEqual(router.calibration_rows_, 12)
        self.assertEqual(router.calibration_feature_count_, 9)
        self.assertTrue(sparse.isspmatrix_csr(router.fit_vectors_))
        self.assertEqual(router.fit_vectors_.dtype, np.float32)
        self.assertFalse(any(isinstance(value, pd.DataFrame) for value in vars(router).values()))

    def test_vocabulary_scaler_and_bank_use_fit_only_and_prediction_is_read_only(self):
        x, y, c = fixture()
        x.loc[24:, 'prompt_text'] += ' neverfittedword'
        router = model().fit(x.iloc[:24], y.iloc[:24], c.iloc[:24], x.iloc[24:36], y.iloc[24:36])
        self.assertFalse(any('neverfittedword' in name for name in router.transformer_.get_feature_names_out()))
        bank = router.fit_vectors_.copy()
        labels = router.fit_codes_.copy()
        scale = router.transformer_.structural_scaler_.scale_.copy()
        names = router.transformer_.get_feature_names_out().copy()
        p, costs = router.predict_tables(x.iloc[36:])
        self.assertEqual(list(p), [*OUTCOME_CLASSES, 'expected_quality_gain'])
        self.assertTrue(p.index.equals(x.iloc[36:].index))
        self.assertEqual(costs.columns.tolist(), ['strong', 'weak'])
        np.testing.assert_allclose(p.loc[:, OUTCOME_CLASSES].sum(axis=1), 1)
        np.testing.assert_allclose(p.expected_quality_gain, p.strong_only-p.weak_only)
        self.assertTrue(np.isfinite(costs.values).all())
        self.assertEqual((router.fit_vectors_ != bank).nnz, 0)
        np.testing.assert_array_equal(labels, router.fit_codes_)
        np.testing.assert_array_equal(scale, router.transformer_.structural_scaler_.scale_)
        np.testing.assert_array_equal(names, router.transformer_.get_feature_names_out())

    def test_global_control_omits_neighbor_bank_and_calls(self):
        with patch('routerbench.neighborhood._neighbor_features', side_effect=AssertionError('retrieval called')):
            router = fit(model(use_neighbors=False))
            p, _ = router.predict_tables(fixture()[0].iloc[36:])
        self.assertEqual(router.calibration_feature_count_, 3)
        self.assertIsNone(router.fit_vectors_)
        self.assertIsNone(router.fit_codes_)
        np.testing.assert_allclose(p.loc[:, OUTCOME_CLASSES].sum(axis=1), 1)

    def test_tie_boundary_is_deterministic_and_smoothing_is_explicit(self):
        bank = sparse.csr_matrix(np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32))
        query = sparse.csr_matrix(np.array([[1, 0]], dtype=np.float32))
        result = _neighbor_features(query, bank, np.array([0, 1, 2, 2]), np.full(3, 1/3), 2, .05, 4., 1)
        np.testing.assert_allclose(result[0, :3], np.array([1+4/3, 1+4/3, 4/3])/6)
        np.testing.assert_allclose(result[0, 3:], [1, 1, 1])
        zero = _neighbor_features(query*0, bank, np.array([0, 1, 2, 2]), np.full(3, 1/3), 2, .05, 4., 1)
        np.testing.assert_allclose(zero[0, :3], result[0, :3])
        np.testing.assert_allclose(zero[0, 3:], [0, 0, 1])

    def test_dense_similarity_allocation_is_batched_and_batch_invariant(self):
        bank = sparse.csr_matrix(np.eye(4, dtype=np.float32))
        query = sparse.csr_matrix(np.tile([1, 0, 0, 0], (5, 1)).astype(np.float32))
        labels, prior = np.array([0, 1, 2, 2]), np.full(3, 1/3)
        shapes = []
        original = sparse.csr_matrix.toarray
        def tracked(matrix, *args, **kwargs):
            shapes.append(matrix.shape)
            return original(matrix, *args, **kwargs)
        with patch.object(sparse.csr_matrix, 'toarray', tracked):
            small = _neighbor_features(query, bank, labels, prior, 3, .05, 4., 2)
        self.assertEqual(shapes, [(2, 4), (2, 4), (1, 4)])
        wide = _neighbor_features(query, bank, labels, prior, 3, .05, 4., 5)
        np.testing.assert_allclose(small, wide)

    def test_missing_classes_and_single_class_have_explicit_simplex_fallbacks(self):
        x, y, c = fixture()
        y.loc[:, :] = 0
        router = model().fit(x.iloc[:24], y.iloc[:24], c.iloc[:24], x.iloc[24:36], y.iloc[24:36])
        p, _ = router.predict_tables(x.iloc[36:])
        self.assertIsNone(router.global_head_)
        self.assertIsNone(router.calibrator_)
        self.assertEqual(set(router.constants_), set(OUTCOME_CLASSES))
        np.testing.assert_allclose(p.iloc[0, :3], [1/15, 1/15, 13/15])
        np.testing.assert_allclose(p.expected_quality_gain, 0)
        x, y, c = fixture()
        # Calibration lacks weak-only outcomes, but fit retains them.
        y.loc[24:35, 'strong'] = 1
        router.fit(x.iloc[:24], y.iloc[:24], c.iloc[:24], x.iloc[24:36], y.iloc[24:36])
        p, _ = router.predict_tables(x.iloc[36:])
        self.assertEqual(set(router.constants_), {'weak_only'})
        np.testing.assert_allclose(p.weak_only, 1/15)
        np.testing.assert_allclose(p.loc[:, OUTCOME_CLASSES].sum(axis=1), 1)

    def test_normalized_duplicate_and_overlap_rejected_before_vectorization(self):
        for kind in ('overlap', 'duplicate', 'index'):
            x, y, c = fixture()
            if kind == 'overlap':
                x.loc[24, 'prompt_text'] = '  '+x.loc[0, 'prompt_text'].upper()+'  '
            elif kind == 'duplicate':
                x.loc[1, 'prompt_text'] = x.loc[0, 'prompt_text']
            with patch('routerbench.neighborhood.PromptRepresentation.fit_transform') as transform:
                with self.assertRaises(ValueError):
                    model().fit(x.iloc[:24], y.iloc[:24], c.iloc[:24],
                                x.iloc[:12] if kind == 'index' else x.iloc[24:36],
                                y.iloc[:12] if kind == 'index' else y.iloc[24:36])
                transform.assert_not_called()

    def test_future_columns_invalid_outcomes_alignment_and_parameters_rejected(self):
        x, y, c = fixture()
        for column in ('score', 'cost', 'response', 'model', 'eval_name'):
            with self.assertRaises(ValueError):
                model().fit(x.iloc[:24].assign(**{column: 'forbidden'}), y.iloc[:24], c.iloc[:24], x.iloc[24:36], y.iloc[24:36])
        for bad_scores, bad_costs in [(y.iloc[:24].assign(strong=2), c.iloc[:24]),
                                      (y.iloc[:24], c.iloc[:24].assign(weak=-1)),
                                      (y.iloc[:24], c.iloc[:24].iloc[::-1])]:
            with self.assertRaises(ValueError):
                model().fit(x.iloc[:24], bad_scores, bad_costs, x.iloc[24:36], y.iloc[24:36])
        for name, value in [('temperature', 0), ('prior_weight', np.nan), ('alpha', -1),
                            ('calibration_C', True), ('use_neighbors', 'yes'),
                            ('n_neighbors', 1.5), ('query_batch_size', 257)]:
            router = model().set_params(**{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                fit(router)
        router = fit()
        with self.assertRaises(ValueError):
            router.predict_tables(x.iloc[36:].assign(observed_cost=1))

    def test_serialization_replay_clone_and_parameter_change_guard(self):
        router = fit()
        x = fixture()[0].iloc[36:]
        expected = router.predict_tables(x)
        buffer = BytesIO()
        joblib.dump(router, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        actual = restored.predict_tables(x)
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)
        self.assertFalse(hasattr(clone(router), 'fit_vectors_'))
        router.set_params(temperature=.2)
        with self.assertRaisesRegex(ValueError, 'parámetros'):
            router.predict_tables(x)


if __name__ == '__main__':
    unittest.main()
