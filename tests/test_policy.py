"""Manual policy scenarios covering selection and retrospective accounting."""

import json
import unittest

import numpy as np
import pandas as pd

from routerbench.policy import choose_models, evaluate_policy


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = pd.DataFrame(
            {"cheap": [0.8, 0.4, 0.95], "premium": [0.95, 0.7, 0.9]},
            index=[10, 20, 30],
        )
        self.costs = pd.DataFrame(
            {"cheap": [1.0, 1.0, 1.0], "premium": [5.0, 5.0, 5.0]},
            index=self.probabilities.index,
        )

    def test_threshold_selects_cheapest_and_fallback_best_probability(self):
        result = choose_models(self.probabilities, self.costs, 0.75)
        self.assertEqual(result["selected_model"].tolist(), ["cheap", "premium", "cheap"])
        self.assertEqual(result["decision"].tolist(), ["threshold_met", "fallback", "threshold_met"])
        self.assertEqual(result["predicted_cost"].tolist(), [1, 5, 1])

    def test_capabilities_override_price_and_abstain_if_none_allowed(self):
        allowed = pd.DataFrame({"cheap": [False, False, False], "premium": [True, True, False]}, index=self.probabilities.index)
        result = choose_models(self.probabilities, self.costs, allowed_models=allowed)
        self.assertEqual(result.loc[10, "selected_model"], "premium")
        self.assertEqual(result.loc[20, "selected_model"], "premium")
        self.assertEqual(result.loc[30, "decision"], "abstain")
        self.assertTrue(result.loc[30, ["selected_model", "predicted_success", "predicted_cost"]].isna().all())

    def test_ties_and_column_order_do_not_change_selection(self):
        probabilities = pd.DataFrame({"zulu": [0.9, 0.4, 0.4], "beta": [0.9, 0.4, 0.4], "alpha": [0.9, 0.4, 0.3]})
        costs = pd.DataFrame({"alpha": [1, 1, 0], "beta": [1, 2, 1], "zulu": [1, 1, 1]})
        result = choose_models(probabilities, costs, 0.75)
        self.assertEqual(result["selected_model"].tolist(), ["alpha", "alpha", "beta"])
        changed = choose_models(probabilities.iloc[:, ::-1], costs.iloc[:, ::-1], 0.75)
        pd.testing.assert_frame_equal(result, changed)

    def test_invalid_predictions_thresholds_and_alignment_rejected(self):
        for value in (-0.1, 1.1, float("nan"), True):
            with self.subTest(threshold=value):
                with self.assertRaises(ValueError):
                    choose_models(self.probabilities, self.costs, value)
        for table, bad_value in (("probabilities", np.nan), ("probabilities", 1.1), ("costs", -1), ("costs", np.inf)):
            probabilities, costs = self.probabilities.copy(), self.costs.copy()
            (probabilities if table == "probabilities" else costs).iloc[0, 0] = bad_value
            with self.subTest(table=table, value=bad_value):
                with self.assertRaises(ValueError):
                    choose_models(probabilities, costs)
        with self.assertRaises(ValueError):
            choose_models(self.probabilities, self.costs.iloc[::-1])
        with self.assertRaises(ValueError):
            choose_models(self.probabilities, self.costs.rename(columns={"cheap": "other"}))

    def test_masks_must_be_boolean_complete_and_aligned(self):
        for mask in (self.probabilities * 0, pd.DataFrame({"cheap": [True, True, True], "premium": [True, None, True]}, index=self.probabilities.index)):
            with self.assertRaises(ValueError):
                choose_models(self.probabilities, self.costs, allowed_models=mask)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame({"cheap": [1.0, 0.0, 0.0], "premium": [1.0, 1.0, 0.0]}, index=[10, 20, 30])
        self.costs = pd.DataFrame({"cheap": [1.0, 1.0, 1.0], "premium": [4.0, 5.0, 6.0]}, index=self.scores.index)

    def test_accounts_for_only_selected_call_and_keeps_unsolved_prompt(self):
        selected = pd.Series(["cheap", "cheap", "premium"], index=self.scores.index)
        result = evaluate_policy(selected, self.scores, self.costs)
        self.assertEqual(result["rows"]["score"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result["rows"]["cost"].tolist(), [1.0, 1.0, 6.0])
        self.assertEqual(result["metrics"]["n_prompts"], 3)
        self.assertEqual(result["metrics"]["success_rate"], 1 / 3)
        self.assertEqual(result["metrics"]["cost_per_1000"], 8000 / 3)
        # The premium model could solve row 20; evaluation must not replace the
        # actually selected cheap model with that retrospective better choice.
        self.assertFalse(result["rows"].loc[20, "success"])

    def test_abstentions_lower_coverage_and_overall_success(self):
        selected = pd.Series(["cheap", pd.NA, pd.NA], dtype="string", index=self.scores.index)
        result = evaluate_policy(selected, self.scores, self.costs)
        self.assertEqual(result["metrics"]["coverage"], 1 / 3)
        self.assertEqual(result["metrics"]["success_rate"], 1 / 3)
        self.assertEqual(result["metrics"]["success_rate_on_routed"], 1)
        self.assertEqual(result["rows"]["cost"].tolist(), [1, 0, 0])
        self.assertTrue(pd.isna(result["rows"].loc[20, "score"]))

    def test_all_abstain_metrics_are_json_safe(self):
        selected = pd.Series([None, None, None], index=self.scores.index)
        metrics = evaluate_policy(selected, self.scores, self.costs)["metrics"]
        self.assertEqual(metrics["success_rate"], 0)
        self.assertEqual(metrics["cost_per_1000"], 0)
        self.assertIsNone(metrics["success_rate_on_routed"])
        json.dumps(metrics, allow_nan=False)

    def test_outcome_order_and_unknown_model_cannot_silently_misalign(self):
        selected = pd.Series(["cheap", "cheap", "unknown"], index=self.scores.index)
        with self.assertRaises(ValueError):
            evaluate_policy(selected, self.scores, self.costs)
        selected.iloc[-1] = "premium"
        with self.assertRaises(ValueError):
            evaluate_policy(selected.iloc[::-1], self.scores, self.costs)
        with self.assertRaises(ValueError):
            evaluate_policy(selected, self.scores, self.costs.iloc[::-1])
        with self.assertRaises(ValueError):
            evaluate_policy(selected.iloc[:0], self.scores.iloc[:0], self.costs.iloc[:0])


if __name__ == "__main__":
    unittest.main()
