"""Dense nonlinear representations preserve train/calibration boundaries."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from routerbench.features import FEATURE_NAMES
from routerbench.nonlinear import DensePromptRepresentation, NonlinearRegretRouter
from routerbench.pairwise import OUTCOME_CLASSES


def fixture():
    texts = [f"question {i} " + ("prove theorem reasoning" if i % 3 == 0 else
                               "simple lookup fact" if i % 3 == 1 else "compare examples together")
             for i in range(72)]
    x = pd.DataFrame({"prompt_text": texts})
    scores = pd.DataFrame({"strong": [float(i % 3 != 1) for i in range(72)],
                           "weak": [float(i % 3 != 0) for i in range(72)]}, index=x.index)
    costs = pd.DataFrame({"strong": [.01]*72, "weak": [.001]*72}, index=x.index)
    return x, scores, costs


def router(**kwargs):
    parameters = dict(strong="strong", weak="weak", n_components=5, max_features=100,
                      min_df=1, max_iter=8, max_leaf_nodes=5, min_samples_leaf=2)
    parameters.update(kwargs)
    return NonlinearRegretRouter(**parameters)


class DenseRepresentationTests(unittest.TestCase):
    def test_rank_vocab_and_full_prompt_fit_only(self):
        x, _, _ = fixture()
        x.loc[:29, "prompt_text"] += " " + "first "*200 + "tailfitmarker"
        x.loc[30:, "prompt_text"] += " calibrationsecretword"
        transform = DensePromptRepresentation(n_components=64, max_features=1000, min_df=1)
        transformed = transform.fit_transform(x.iloc[:30])
        self.assertEqual(transformed.shape, (30, 29+14))
        self.assertEqual(transform.transform(x.iloc[30:]).shape, (42, 43))
        vocabulary = transform.lexical_.get_feature_names_out()
        self.assertTrue(any("tailfitmarker" in item for item in vocabulary))
        self.assertFalse(any("calibrationsecretword" in item for item in vocabulary))
        self.assertEqual(clone(transform).get_params(), transform.get_params())
        self.assertTrue(np.isfinite(transformed).all())

    def test_structural_does_not_erase_length_and_does_not_refit_on_transform(self):
        fit = pd.DataFrame({"prompt_text": ["word", "word word"]})
        transform = DensePromptRepresentation(mode="structural").fit(fit)
        before = transform.structural_scaler_.max_abs_.copy()
        output = transform.transform(pd.DataFrame({"prompt_text": ["word", "word word word word"]}))
        self.assertEqual(output.shape, (2, len(FEATURE_NAMES)))
        self.assertGreater(output[1, 0], output[0, 0])
        self.assertGreater(output[1, 0], 1)
        self.assertFalse(np.allclose(np.linalg.norm(output, axis=1), 1))
        np.testing.assert_array_equal(before, transform.structural_scaler_.max_abs_)
        self.assertTrue(np.isfinite(output).all())

    def test_rejects_metadata_incomplete_structural_columns_and_empty_text(self):
        x, _, _ = fixture()
        for frame in (x.assign(eval_name="benchmark"), x.assign(n_chars=3),
                      pd.DataFrame({"prompt_text": [" "]}), x.iloc[:0], pd.concat([x.iloc[[0]]]*2)):
            with self.subTest(columns=frame.columns.tolist()), self.assertRaises(ValueError):
                DensePromptRepresentation(mode="structural").fit(frame)


class NonlinearRouterTests(unittest.TestCase):
    def test_calibrated_multiclass_prediction_and_clone(self):
        x, scores, costs = fixture()
        x.loc[36:, "prompt_text"] += " calibrationsecretword"
        model = router()
        with threadpool_limits(1):
            model.fit(x.iloc[:36], scores.iloc[:36], costs.iloc[:36], x.iloc[36:54], scores.iloc[36:54])
            p, predicted_cost = model.predict_tables(x.iloc[54:])
        self.assertEqual(p.shape, (18, 4))
        self.assertEqual(p.index.tolist(), list(range(54, 72)))
        self.assertEqual(predicted_cost.columns.tolist(), ["strong", "weak"])
        self.assertTrue(np.allclose(p.loc[:, OUTCOME_CLASSES].sum(axis=1), 1))
        self.assertTrue(np.isfinite(predicted_cost.to_numpy()).all())
        self.assertTrue(predicted_cost["strong"].gt(predicted_cost["weak"]).all())
        self.assertFalse(model.predictor_.estimator.estimator.early_stopping)
        self.assertEqual(model.target_counts_["fit"], dict.fromkeys(OUTCOME_CLASSES, 12))
        self.assertFalse(any("calibrationsecretword" in name for name in model.transformer_.lexical_.get_feature_names_out()))
        self.assertEqual(clone(model).get_params(), model.get_params())

    def test_absent_class_and_all_constant_targets_have_explicit_fallbacks(self):
        x, scores, costs = fixture()
        # Calibration has strong-only and equal, with no weak-only.
        scores.loc[36:53, "strong"] = 1
        scores.loc[36:53, "weak"] = [float(i % 2) for i in range(18)]
        model = router(representation="structural")
        with threadpool_limits(1):
            model.fit(x.iloc[:36], scores.iloc[:36], costs.iloc[:36], x.iloc[36:54], scores.iloc[36:54])
            p, _ = model.predict_tables(x.iloc[54:])
        self.assertIn("weak_only", model.constants_)
        self.assertEqual(model.classifier_fit_rows_, 24)
        self.assertTrue(np.allclose(p["weak_only"], 1/21))
        self.assertTrue(np.allclose(p.loc[:, OUTCOME_CLASSES].sum(axis=1), 1))
        scores.loc[:, :] = 0
        with threadpool_limits(1):
            model.fit(x.iloc[:36], scores.iloc[:36], costs.iloc[:36], x.iloc[36:54], scores.iloc[36:54])
            p, _ = model.predict_tables(x.iloc[54:])
        self.assertIsNone(model.predictor_)
        self.assertTrue(p["equal"].gt(p["strong_only"]).all())
        self.assertTrue(p["expected_quality_gain"].eq(0).all())

    def test_invalid_overlap_outcomes_and_input_axes_are_rejected(self):
        x, scores, costs = fixture()
        with self.assertRaisesRegex(ValueError, "disjuntos"):
            router().fit(x.iloc[:36], scores.iloc[:36], costs.iloc[:36], x.iloc[20:54], scores.iloc[20:54])
        for bad_scores, bad_costs in ((scores.assign(strong=np.nan), costs),
                                      (scores, costs.assign(weak=-1)),
                                      (scores, costs.iloc[::-1])):
            with self.assertRaises(ValueError):
                router().fit(x.iloc[:36], bad_scores.iloc[:36], bad_costs.iloc[:36], x.iloc[36:54], scores.iloc[36:54])
        model = router(representation="structural")
        with threadpool_limits(1):
            model.fit(x.iloc[:36], scores.iloc[:36], costs.iloc[:36], x.iloc[36:54], scores.iloc[36:54])
        for bad in (x.iloc[:0], pd.concat([x.iloc[[0]]]*2), x.iloc[[0]].assign(score=1)):
            with self.assertRaises(ValueError):
                model.predict_tables(bad)


if __name__ == "__main__":
    unittest.main()
