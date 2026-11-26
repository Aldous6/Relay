import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from routerbench.guarded import GuardedPairRouter, choose_guarded, guarded_targets


def fixture(start, n=16):
    ids = pd.Index(range(start, start+n))
    inputs = pd.DataFrame({'prompt_text': [f'Unique {start+i} item question '+('code example' if i%2 else 'document text')
                                          for i in range(n)]}, index=ids)
    scores = pd.DataFrame({'strong': ([1, 1, 0, 0]*(n//4)),
                           'weak': ([1, 0, 1, 0]*(n//4))}, index=ids)
    costs = pd.DataFrame({'strong': np.full(n, 2.), 'weak': np.full(n, .1)}, index=ids)
    return inputs, scores, costs


class GuardedTests(unittest.TestCase):
    def test_targets_distinguish_shared_failure_from_weak_success(self):
        _, scores, _ = fixture(0)
        result = guarded_targets(scores, 'strong', 'weak')
        self.assertEqual(result.iloc[:4].values.tolist(), [[0, 1], [1, 0], [0, 1], [0, 0]])

    def test_gate_checks_harm_success_and_predicted_saving(self):
        predictions = pd.DataFrame({'harm_probability': [.05, .051, .01, .01, .01],
                                    'weak_success_probability': [.8, .99, .79, .9, .9]})
        costs = pd.DataFrame({'strong': [2]*5, 'weak': [1, 1, 1, 2, 3]})
        routes = choose_guarded(predictions, costs, 'strong', 'weak', .05, .8)
        self.assertEqual(routes.selected_model.tolist(), ['weak', 'strong', 'strong', 'strong', 'strong'])
        self.assertEqual(routes.decision.tolist(), ['guarded_weak', 'harm_guard', 'weak_quality_guard',
                                                   'no_predicted_saving', 'no_predicted_saving'])
        reversed_columns = choose_guarded(predictions.iloc[:, ::-1], costs.iloc[:, ::-1], 'strong', 'weak', .05, .8)
        pd.testing.assert_frame_equal(routes, reversed_columns)

    def test_invalid_or_misaligned_predictions_cannot_silently_route(self):
        p = pd.DataFrame({'harm_probability': [.01, .02], 'weak_success_probability': [.8, .9]})
        c = pd.DataFrame({'strong': [2., 2.], 'weak': [1., 1.]})
        for bad in [p.assign(harm_probability=np.nan), p.assign(harm_probability=-1),
                    p.assign(weak_success_probability=1.1), p.assign(score=1), p.set_axis([0, 0])]:
            with self.assertRaises(ValueError):
                choose_guarded(bad, c, 'strong', 'weak', .05, .8)
        for bad in [c.iloc[::-1], c.assign(weak=-1), c.assign(strong=np.inf), c.assign(future_cost=1)]:
            with self.assertRaises(ValueError):
                choose_guarded(p, bad, 'strong', 'weak', .05, .8)
        for limit in [True, -1, np.nan, np.inf, 2]:
            with self.assertRaises(ValueError):
                choose_guarded(p, c, 'strong', 'weak', limit, .8)

    def model(self):
        return GuardedPairRouter('strong', 'weak', min_df=1, max_features=100)

    def test_fit_preserves_all_rows_and_fit_only_vocabulary(self):
        x, y, c = fixture(0)
        cx, cy, _ = fixture(100)
        cx['prompt_text'] += ' calibrationonlytoken'
        model = self.model().fit(x, y, c, cx, cy)
        self.assertEqual(model.fit_rows_, 16)
        self.assertEqual(model.calibration_rows_, 16)
        words = model.transformer_.lexical_.transformer_list[0][1].vocabulary_
        self.assertNotIn('calibrationonlytoken', words)
        names = model.transformer_.get_feature_names_out().copy()
        scales = model.transformer_.structural_scaler_.scale_.copy()
        p, estimates = model.predict_tables(cx)
        self.assertTrue(p.index.equals(cx.index))
        self.assertTrue(((p >= 0) & (p <= 1)).all().all())
        self.assertTrue(np.isfinite(estimates.values).all())
        np.testing.assert_array_equal(names, model.transformer_.get_feature_names_out())
        np.testing.assert_array_equal(scales, model.transformer_.structural_scaler_.scale_)
        self.assertFalse(hasattr(clone(model), 'transformer_'))

    def test_normalized_prompt_overlap_is_rejected_even_with_distinct_ids(self):
        x, y, c = fixture(0)
        cx, cy, _ = fixture(100)
        cx.iloc[0, 0] = '  '+x.iloc[0, 0].upper()+'  '
        with self.assertRaisesRegex(ValueError, 'grupos'):
            self.model().fit(x, y, c, cx, cy)

    def test_scores_and_costs_are_never_predictors(self):
        x, y, c = fixture(0)
        cx, cy, _ = fixture(100)
        for name in ['score', 'model', 'response', 'eval_name', 'observed_cost']:
            with self.assertRaises(ValueError):
                self.model().fit(x.assign(**{name: 1}), y, c, cx, cy)
        model = self.model().fit(x, y, c, cx, cy)
        with self.assertRaises(ValueError):
            model.predict_tables(cx.assign(response='gold'))

    def test_missing_classes_use_explicit_smoothed_calibration_prevalence(self):
        x, y, c = fixture(0)
        cx, cy, _ = fixture(100)
        y.loc[:, :] = 0
        cy.loc[:, :] = 0
        model = self.model().fit(x, y, c, cx, cy)
        p, _ = model.predict_tables(cx)
        np.testing.assert_allclose(p.values, 1/18)
        self.assertEqual(len(model.constants_), 2)

    def test_outcome_alignment_and_duplicate_inputs_rejected(self):
        x, y, c = fixture(0)
        cx, cy, _ = fixture(100)
        with self.assertRaises(ValueError):
            self.model().fit(x, y.iloc[::-1], c, cx, cy)
        with self.assertRaises(ValueError):
            self.model().fit(x, y, c, x, y)
        x.iloc[1, 0] = x.iloc[0, 0]
        with self.assertRaises(ValueError):
            self.model().fit(x, y, c, cx, cy)


if __name__ == '__main__':
    unittest.main()
