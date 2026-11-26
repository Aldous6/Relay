"""Regret targets, no-leakage fitting and deployable pairwise policy checks."""
import unittest

import numpy as np
import pandas as pd

from routerbench.pairwise import OUTCOME_CLASSES, PairwiseRegretRouter, choose_pairwise, pairwise_targets


def fixture():
    inputs = pd.DataFrame({"prompt_text": [f"request {i} " + ("proof theorem reasoning" if i % 3 == 0 else "simple fact answer") for i in range(60)]}, index=range(60))
    scores = pd.DataFrame({"strong": [1 if i % 3 != 1 else 0 for i in range(60)],
                           "weak": [1 if i % 3 != 0 else 0 for i in range(60)]}, index=inputs.index, dtype=float)
    costs = pd.DataFrame({"strong": np.full(60, .01), "weak": np.full(60, .001)}, index=inputs.index)
    return inputs, scores, costs


def model(**kwargs):
    return PairwiseRegretRouter(strong="strong", weak="weak", max_features=100, min_df=1, **kwargs)


class PairwiseTargetsTests(unittest.TestCase):
    def test_targets_are_thresholded_success_difference_including_both_failures(self):
        scores = pd.DataFrame({"strong": [1, 0, 1, 0, .75], "weak": [0, 1, 1, 0, .74]}, index=[8, 7, 6, 5, 4])
        targets = pairwise_targets(scores, "strong", "weak")
        self.assertEqual(targets.tolist(), ["strong_only", "weak_only", "equal", "equal", "strong_only"])
        self.assertEqual(targets.index.tolist(), scores.index.tolist())
        for threshold in (-.1, 1.1, np.nan, True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                pairwise_targets(scores, "strong", "weak", threshold)


class PairwiseLearnerTests(unittest.TestCase):
    def test_fit_prediction_and_vocabulary_use_only_fit_rows(self):
        x, scores, costs = fixture()
        x.loc[30:, "prompt_text"] += " calibrationsecretword"
        router = model()
        router.fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30], x.iloc[30:45], scores.iloc[30:45])
        advantage, estimates = router.predict_tables(x.iloc[45:])
        self.assertEqual(advantage.shape, (15, 4))
        self.assertEqual(list(advantage.columns), [*OUTCOME_CLASSES, "expected_quality_gain"])
        self.assertEqual(estimates.columns.tolist(), ["strong", "weak"])
        self.assertTrue(advantage.index.equals(estimates.index))
        self.assertTrue(np.allclose(advantage.loc[:, OUTCOME_CLASSES].sum(axis=1), 1))
        self.assertTrue(np.isfinite(estimates.to_numpy()).all())
        self.assertTrue(estimates["strong"].gt(estimates["weak"]).all())
        self.assertFalse(any("calibrationsecretword" in name for name in router.transformer_.get_feature_names_out()))
        self.assertEqual(router.target_counts_["fit"], {"strong_only": 10, "weak_only": 10, "equal": 10})

    def test_logistic_backend_and_constant_missing_classes_remain_valid(self):
        x, scores, costs = fixture()
        router = model(model_type="logistic")
        router.fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30], x.iloc[30:45], scores.iloc[30:45])
        advantage, _ = router.predict_tables(x.iloc[45:])
        self.assertTrue(np.allclose(advantage.loc[:, OUTCOME_CLASSES].sum(axis=1), 1))
        # Equal includes both models failing; no artificial strong preference.
        scores.loc[:, :] = 0
        router = model()
        router.fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30], x.iloc[30:45], scores.iloc[30:45])
        advantage, _ = router.predict_tables(x.iloc[45:])
        self.assertEqual(set(router.constants_), set(OUTCOME_CLASSES))
        self.assertTrue(advantage["equal"].gt(advantage["strong_only"]).all())
        self.assertTrue(advantage["expected_quality_gain"].eq(0).all())
        # Class absent only from calibration is also explicit and non-crashing.
        x, scores, costs = fixture()
        scores.loc[30:44, "weak"] = scores.loc[30:44, "strong"]
        router.fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30], x.iloc[30:45], scores.iloc[30:45])
        self.assertIn("strong_only", router.constants_)
        self.assertIn("weak_only", router.constants_)
        self.assertTrue(np.isfinite(router.predict_tables(x.iloc[45:])[0].to_numpy()).all())

    def test_invalid_partitions_inputs_and_outcomes_are_rejected(self):
        x, scores, costs = fixture()
        with self.assertRaisesRegex(ValueError, "disjuntos"):
            model().fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30], x.iloc[20:40], scores.iloc[20:40])
        with self.assertRaises(ValueError):
            model().fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30].iloc[::-1], x.iloc[30:45], scores.iloc[30:45])
        for field, value in (("score", np.nan), ("score", 1.1), ("cost", -1), ("cost", np.inf), ("prompt", " ")):
            bad_x, bad_scores, bad_costs = x.copy(), scores.copy(), costs.copy()
            if field == "score":
                bad_scores.loc[0, "strong"] = value
            elif field == "cost":
                bad_costs.loc[0, "strong"] = value
            else:
                bad_x.loc[0, "prompt_text"] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                model().fit(bad_x.iloc[:30], bad_scores.iloc[:30], bad_costs.iloc[:30], x.iloc[30:45], scores.iloc[30:45])
        router = model().fit(x.iloc[:30], scores.iloc[:30], costs.iloc[:30], x.iloc[30:45], scores.iloc[30:45])
        with self.assertRaises(ValueError):
            router.predict_tables(x.iloc[:0])
        with self.assertRaises(ValueError):
            router.predict_tables(pd.concat([x.iloc[[0]], x.iloc[[0]]]))
        with self.assertRaises(ValueError):
            router.predict_tables(x.iloc[[0]].assign(score=1))


class PairwisePolicyTests(unittest.TestCase):
    def test_policy_trades_signed_extra_cost_against_expected_success_gain(self):
        advantage = pd.DataFrame({"strong_only": [.5, .5, .25, 0], "weak_only": [.25, .25, .25, .25],
                                  "equal": [.25, .25, .5, .75]}, index=[10, 11, 12, 13])
        costs = pd.DataFrame({"strong": [1.5, 1.5, .5, .5], "weak": [1., 1., 1., 1.]}, index=advantage.index)
        # Tradeoff=.5: first two utility ties choose weak; cheaper strong wins
        # equal quality on row12; row13 is an exact negative/negative utility tie.
        selected = choose_pairwise(advantage, costs, "strong", "weak", .5)
        self.assertEqual(selected["selected_model"].tolist(), ["weak", "weak", "strong", "weak"])
        self.assertEqual(selected.index.tolist(), advantage.index.tolist())
        self.assertEqual(choose_pairwise(advantage, costs, "strong", "weak", 0)["selected_model"].tolist(), ["strong", "strong", "weak", "weak"])

    def test_policy_checks_simplex_alignment_and_gain_consistency(self):
        p = pd.DataFrame({"strong_only": [.4], "weak_only": [.1], "equal": [.5]})
        c = pd.DataFrame({"strong": [.01], "weak": [.001]})
        for bad_p, bad_c, tradeoff in [(p.assign(equal=.9), c, 1),
                                       (p.assign(expected_quality_gain=.9), c, 1),
                                       (p, c.set_axis([3]), 1), (p, c.assign(weak=-1), 1),
                                       (p, c, -1), (p, c, np.nan), (p, c, True),
                                       (p.iloc[:0], c.iloc[:0], 1)]:
            with self.subTest(tradeoff=tradeoff), self.assertRaises(ValueError):
                choose_pairwise(bad_p, bad_c, "strong", "weak", tradeoff)


if __name__ == "__main__":
    unittest.main()
