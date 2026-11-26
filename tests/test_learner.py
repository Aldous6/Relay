import unittest

import numpy as np
import pandas as pd

from routerbench.learner import SuccessCostRouter


def fixtures():
    inputs = pd.DataFrame({"prompt_text": ["simple question" if index % 2 else "advanced proof reasoning" for index in range(40)]}, index=range(40))
    scores = pd.DataFrame({"cheap": [index % 2 for index in range(40)], "strong": [1] * 40}, index=inputs.index)
    costs = pd.DataFrame({"cheap": [0.001] * 40, "strong": [0.01] * 40}, index=inputs.index)
    return inputs, scores, costs


class LearnerTests(unittest.TestCase):
    def test_disjoint_calibration_and_prediction_contract(self):
        inputs, scores, costs = fixtures()
        router = SuccessCostRouter(representation="hybrid", min_df=1, max_features=100)
        router.fit(inputs.iloc[:20], scores.iloc[:20], costs.iloc[:20], inputs.iloc[20:30], scores.iloc[20:30])
        probabilities, predicted_costs = router.predict_tables(inputs.iloc[30:])
        self.assertEqual(probabilities.shape, (10, 2))
        self.assertEqual(probabilities.index.tolist(), list(range(30, 40)))
        self.assertTrue(probabilities.ge(0).all().all() and probabilities.le(1).all().all())
        self.assertTrue(np.isfinite(predicted_costs.to_numpy()).all())
        self.assertTrue(predicted_costs.ge(0).all().all())
        self.assertTrue(predicted_costs["strong"].gt(predicted_costs["cheap"]).all())
        self.assertIn("strong", router.constants_)

    def test_calibration_overlap_and_misaligned_costs_rejected(self):
        inputs, scores, costs = fixtures()
        router = SuccessCostRouter(representation="structural")
        with self.assertRaises(ValueError):
            router.fit(inputs.iloc[:20], scores.iloc[:20], costs.iloc[:20], inputs.iloc[10:30], scores.iloc[10:30])
        with self.assertRaises(ValueError):
            router.fit(inputs.iloc[:20], scores.iloc[:20], costs.iloc[:20].iloc[::-1], inputs.iloc[20:30], scores.iloc[20:30])

    def test_validation_words_do_not_enter_vocabulary(self):
        inputs, scores, costs = fixtures()
        inputs.loc[20:, "prompt_text"] = inputs.loc[20:, "prompt_text"] + " calibrationsecretword"
        router = SuccessCostRouter(representation="lexical", min_df=1, max_features=100)
        router.fit(inputs.iloc[:20], scores.iloc[:20], costs.iloc[:20], inputs.iloc[20:30], scores.iloc[20:30])
        names = router.transformer_.get_feature_names_out()
        self.assertFalse(any("calibrationsecretword" in name for name in names))

    def test_one_candidate_and_extreme_costs_stay_aligned_and_finite(self):
        inputs, scores, costs = fixtures()
        costs.loc[0, "cheap"] = 1e308
        router = SuccessCostRouter(representation="structural")
        router.fit(inputs.iloc[:20], scores.loc[:19, ["cheap"]], costs.loc[:19, ["cheap"]],
                   inputs.iloc[20:30], scores.loc[20:29, ["cheap"]])
        probabilities, predicted_costs = router.predict_tables(inputs.iloc[30:35])
        self.assertEqual(probabilities.shape, (5, 1))
        self.assertEqual(predicted_costs.shape, (5, 1))
        self.assertTrue(np.isfinite(predicted_costs.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
