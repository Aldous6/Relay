"""Protocol-level checks for the unchanged original tier call policy."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from routerbench.data import OFFICIAL_SHA256, sha256_file, write_json
from routerbench.features import FEATURE_NAMES
from routerbench.labels import outcome_tables
from routerbench.tier_policy import (
    evaluate_frozen_routes, label_balance, retrospective_labels,
    route_predicted_tiers, run_tier_policy, select_representatives,
)


class TierPolicyRulesTests(unittest.TestCase):
    def test_representatives_use_success_then_mean_cost_then_name(self):
        scores = pd.DataFrame({"z0": [1, 1, 0], "a0": [.75, .8, 0],
                               "a1": [1, 1, 1], "z1": [1, 0, 0], "z2": [1, 1, 1], "a2": [1, 1, 1]})
        costs = pd.DataFrame({"z0": [1, 1, 1], "a0": [1, 1, 1],
                              "a1": [9, 9, 9], "z1": [.1, .1, .1], "z2": [3, 3, 3], "a2": [4, 4, 4]})
        mapping = {"a0": 0, "z0": 0, "a1": 1, "z1": 1, "a2": 2, "z2": 2}
        result = select_representatives(scores, costs, mapping)
        self.assertEqual(result["representatives"], {0: "a0", 1: "a1", 2: "z2"})
        self.assertEqual(result["n_train"], 3)
        self.assertEqual(result["threshold"], .75)
        with self.assertRaises(ValueError):
            select_representatives(scores, costs.iloc[::-1], mapping)
        with self.assertRaises(ValueError):
            select_representatives(scores, costs, {model: 0 for model in scores})

    def test_routing_only_accepts_integer_tiers_and_keeps_index(self):
        prediction = pd.Series([2, 0, 1], index=[50, 30, 80])
        result = route_predicted_tiers(prediction, {0: "cheap", 1: "middle", 2: "premium"})
        self.assertEqual(result.tolist(), ["premium", "cheap", "middle"])
        self.assertEqual(result.index.tolist(), prediction.index.tolist())
        for values in ([0, 3], [0, None], [0.0, 1.0], [True, False]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                route_predicted_tiers(pd.Series(values), {0: "a", 1: "b", 2: "c"})

    def test_oracle_retains_unsolved_rows_and_breaks_cost_ties_by_name(self):
        scores = pd.DataFrame({"c": [0, 1, 0], "b": [0, 1, 1], "a": [0, 1, 1]}, index=[9, 8, 7])
        costs = pd.DataFrame({"c": [3, 1, 3], "b": [2, 1, 2], "a": [1, 1, 1]}, index=scores.index)
        labels = retrospective_labels(scores, costs, {"a": 0, "b": 1, "c": 2}, .75)
        self.assertTrue(pd.isna(labels.loc[9, "tier"]))
        self.assertEqual(labels.loc[8, "cheapest_model"], "a")
        self.assertEqual(label_balance(labels)["tier_counts"], {"0": 2, "1": 0, "2": 0})
        self.assertEqual(label_balance(labels)["n_ineligible"], 1)

    def test_sensitivity_never_changes_calls_or_representatives(self):
        scores = pd.DataFrame({"a": [.6, .8, 0], "b": [1, 1, 0], "c": [1, 1, 0]})
        costs = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5], "c": [9, 9, 9]})
        routes = pd.Series(["a", "a", "c"], dtype="string")
        before = routes.copy(deep=True)
        results = evaluate_frozen_routes(routes, scores, costs, {0: "a", 1: "b", 2: "c"}, {"a": 0, "b": 1, "c": 2})
        pd.testing.assert_series_equal(routes, before)
        self.assertEqual([entry["policy"]["success_rate"] for entry in results.values()], [2/3, 1/3, 0])
        self.assertEqual({entry["policy"]["total_cost"] for entry in results.values()}, {12})
        self.assertEqual(results["0.75"]["retrospective_oracle"]["coverage"], 2/3)
        self.assertEqual(results["0.75"]["retrospective_oracle"]["total_cost"], 7)


def prepare_fixture(root: Path):
    prepared, tiers = root / "data/processed", root / "reports/final-tiers"
    prepared.mkdir(parents=True)
    tiers.mkdir(parents=True)
    features = pd.DataFrame(np.arange(7*len(FEATURE_NAMES)).reshape(7, -1), columns=FEATURE_NAMES)
    rows = pd.DataFrame({"source_row": range(7), "split": ["train"]*4+["test"]*3,
                         "sample_id": [str(i) for i in range(7)], "eval_name": ["fixture"]*7})
    raw = pd.DataFrame({"sample_id": range(7), "prompt": ["fixture"]*7, "eval_name": ["fixture"]*7,
                        "m0": [1, 1, 0, 0, 0, 1, 0],
                        "m1": [1, 0, 1, 0, 0, .5, .75],
                        "m2": [1, 1, 1, 1, 0, 1, 1],
                        "m0|total_cost": [1.0]*7, "m1|total_cost": [2.0]*7, "m2|total_cost": [3.0]*7})
    features.to_parquet(prepared / "features.parquet")
    rows.to_parquet(prepared / "rows.parquet")
    pd.DataFrame({"eligible": [True]*4+[False, True, True]}).to_parquet(prepared / "labels.parquet")
    hashes = {name: sha256_file(prepared / name) for name in ("features.parquet", "rows.parquet", "labels.parquet")}
    mapping = {"m0": 0, "m1": 1, "m2": 2}
    manifest = {"dataset_sha256": OFFICIAL_SHA256, "configuration": {"dataset": {"path": "data/raw/fixture.pkl"},
                 "quality_threshold": .75}, "tier_map": mapping, "feature_names": list(FEATURE_NAMES), "files_sha256": hashes}
    write_json(prepared / "manifest.json", manifest)
    model = DummyClassifier(strategy="constant", constant=1).fit(features.iloc[:4], [0, 1, 2, 0])
    joblib.dump(model, tiers / "model.joblib")
    summary = {"selection_used_test": False, "final_test_evaluated": True, "dataset_sha256": OFFICIAL_SHA256,
               "tier_map": mapping, "feature_names": list(FEATURE_NAMES), "prepared_files_sha256": hashes,
               "artifacts": {"random_forest": {"file": "model.joblib", "sha256": sha256_file(tiers / "model.joblib")}},
               "selected_model": "random_forest", "train_eligible_rows": 4, "test_eligible_rows": 2}
    write_json(tiers / "summary.json", summary)
    return raw


class TierPolicyArtifactTests(unittest.TestCase):
    def test_end_to_end_persists_routes_before_test_outcomes_and_protects_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = prepare_fixture(root)
            output = root / "reports/new-policy"
            source_path = root / "reports/final-tiers/summary.json"
            old_hash = sha256_file(source_path)
            checked_test = []

            def checked_outcomes(frame):
                if frame.index.min() >= 4:
                    self.assertTrue((output / "routes.parquet").is_file())
                    frozen = json.loads((output / "routing_frozen.json").read_text(encoding="utf-8"))
                    self.assertEqual(frozen["routes_sha256"], sha256_file(output / "routes.parquet"))
                    self.assertFalse(frozen["test_outcomes_consulted_for_routing"])
                    checked_test.append(True)
                return outcome_tables(frame)

            with patch("routerbench.tier_policy.load_verified_pickle", return_value=raw), patch("routerbench.tier_policy.outcome_tables", side_effect=checked_outcomes):
                result = run_tier_policy(root, output, bootstrap_replicates=100)
            self.assertTrue(checked_test)
            self.assertEqual(result["n_test"], 3)
            self.assertEqual(result["policy"]["coverage"], 1)
            self.assertEqual(result["policy"]["success_rate"], 1/3)
            self.assertEqual(result["policy"]["cost_per_1000"], 2000)
            self.assertEqual(result["policy_on_eligible_subset"]["success_rate"], 1/2)
            self.assertFalse(result["retrospective_oracle"]["deployable"])
            self.assertIn("held-out", result["comparison_vs_premium"]["scope"])
            self.assertEqual(result["label_balance"]["test"]["0.75"]["n_ineligible"], 1)
            self.assertEqual(sha256_file(source_path), old_hash)
            self.assertFalse((root / "reports/final").exists())
            self.assertTrue((output / "summary.json").is_file())
            with self.assertRaises(FileExistsError):
                run_tier_policy(root, output, bootstrap_replicates=100)
            with self.assertRaises(ValueError):
                run_tier_policy(root, root / "reports/final", bootstrap_replicates=100)

    def test_model_and_prepared_hashes_are_checked_before_deserialization(self):
        for target in ("reports/final-tiers/model.joblib", "data/processed/features.parquet"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                prepare_fixture(root)
                with (root / target).open("ab") as stream:
                    stream.write(b"changed")
                with patch("routerbench.tier_policy.joblib.load") as load:
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        run_tier_policy(root, bootstrap_replicates=100)
                    load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
