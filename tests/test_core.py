"""Small tests for data contracts and leakage-sensitive transformations."""

import unittest

import numpy as np
import pandas as pd

from routerbench.features import FEATURE_NAMES, extract_features, normalize_for_dedup, parse_prompt
from routerbench.labels import discover_models, fit_tiers, make_labels, outcome_tables
from routerbench.models import build_models


def example_frame():
    return pd.DataFrame({
        "alpha": ["0.8", "0.2", "1.0"],
        "beta": ["0.9", "0.4", "1.0"],
        "gamma": ["0.1", "0.1", "1.0"],
        "alpha|total_cost": [2.0, 1.0, 3.0],
        "beta|total_cost": [1.0, 4.0, 3.0],
        "gamma|total_cost": [5.0, 5.0, 5.0],
    }, index=[10, 20, 30])


class FeatureTests(unittest.TestCase):
    def test_serialized_prompt_preserves_code_indentation(self):
        raw = repr(["System", "def f():\n    return 2\n"])
        self.assertEqual(parse_prompt(raw), "System\ndef f():\n    return 2\n")
        self.assertEqual(parse_prompt("    keep this\n"), "    keep this\n")

    def test_expressions_are_not_executed(self):
        expression = "[__import__('os').getcwd()]"
        self.assertEqual(parse_prompt(expression), expression)
        self.assertEqual(parse_prompt("[1, 2]"), "[1, 2]")

    def test_allowlist_contains_only_fourteen_prompt_features(self):
        features = extract_features(["ABC 2?\n汉字", ""])
        self.assertEqual(tuple(features.columns), FEATURE_NAMES)
        self.assertEqual(features.shape, (2, 14))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertTrue((features.iloc[1] == 0).all())
        self.assertEqual(features.iloc[0]["n_chars"], len("ABC 2?\n汉字"))
        self.assertEqual(features.iloc[0]["n_questions"], 1)
        self.assertEqual(features.iloc[0]["n_lines"], 2)
        self.assertEqual(features.iloc[0]["cjk_ratio"], 2 / len("ABC 2?\n汉字"))
        self.assertFalse(set(features.columns) & {"tier", "eval_name", "sample_id", "score", "cost"})

    def test_duplicate_key_is_separate_from_features(self):
        left, right = "Ａ\n   B", "a b"
        self.assertEqual(normalize_for_dedup(left), normalize_for_dedup(right))
        self.assertNotEqual(extract_features([left]).iloc[0]["n_chars"], extract_features([right]).iloc[0]["n_chars"])


class LabelTests(unittest.TestCase):
    def test_actual_row_cost_and_threshold_define_target(self):
        frame = example_frame()
        labels = make_labels(frame, {"alpha": 0, "beta": 1, "gamma": 2})
        self.assertEqual(labels.index.tolist(), [10, 20, 30])
        self.assertEqual(labels.loc[10, "cheapest_model"], "beta")
        self.assertEqual(labels.loc[10, "tier"], 1)
        self.assertFalse(labels.loc[20, "eligible"])
        self.assertTrue(pd.isna(labels.loc[20, "tier"]))
        self.assertTrue(pd.isna(labels.loc[20, "cheapest_model"]))
        self.assertEqual(str(labels["tier"].dtype), "Int64")

    def test_ties_are_independent_of_column_order(self):
        frame = example_frame()
        labels = make_labels(frame.loc[:, list(reversed(frame.columns))], {"alpha": 0, "beta": 1, "gamma": 2})
        self.assertEqual(labels.loc[30, "cheapest_model"], "alpha")

    def test_tiers_fit_training_costs_only(self):
        train = example_frame()
        tier_map = fit_tiers(train, (1, 1, 1))
        self.assertEqual(tier_map, {"alpha": 0, "beta": 1, "gamma": 2})
        test = train.copy()
        test["alpha|total_cost"] = 1000.0
        make_labels(test, tier_map)
        self.assertEqual(tier_map, {"alpha": 0, "beta": 1, "gamma": 2})
        self.assertEqual(fit_tiers(test, (1, 1, 1))["alpha"], 2)

    def test_invalid_outcomes_are_rejected(self):
        for column, value in [
            ("alpha", "bad"), ("alpha", np.nan), ("alpha", 1.1),
            ("alpha|total_cost", -1), ("alpha|total_cost", np.inf),
        ]:
            with self.subTest(column=column, value=value):
                frame = example_frame()
                frame.loc[10, column] = value
                with self.assertRaises(ValueError):
                    outcome_tables(frame)

    def test_incomplete_schema_and_map_are_rejected(self):
        frame = example_frame()
        with self.assertRaises(ValueError):
            discover_models(frame.drop(columns="alpha"))
        with self.assertRaises(ValueError):
            make_labels(frame, {"alpha": 0})
        with self.assertRaises(ValueError):
            make_labels(frame, {"alpha": 0, "beta": 1, "gamma": 2}, 1.5)
        with self.assertRaises(ValueError):
            fit_tiers(frame)


class ModelTests(unittest.TestCase):
    def test_length_reference_ignores_other_columns(self):
        features = extract_features(["a", "a b", "a b c", "z", "z q", "z q r"])
        target = [0, 1, 2, 0, 1, 2]
        model = build_models()["length"].fit(features, target)
        changed = features.copy()
        changed.loc[:, [name for name in FEATURE_NAMES if name != "n_chars"]] = 999.0
        np.testing.assert_array_equal(model.predict(features), model.predict(changed))


if __name__ == "__main__":
    unittest.main()
