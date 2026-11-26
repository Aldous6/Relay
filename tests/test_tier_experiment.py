"""Small provenance and leakage-sensitive contracts for the original study."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from routerbench.data import sha256_file, write_json
from routerbench.features import FEATURE_NAMES
from routerbench.tier_experiment import classification_metrics, load_tier_inputs, run_tier_experiment


def fixture(root: Path, reverse_test_labels=False):
    prepared = root / "data/processed"
    prepared.mkdir(parents=True)
    raw = root / "raw-provenance.bin"
    raw.write_bytes(b"verified fixture; never deserialized")
    count, train_count = 180, 150
    tier = np.arange(count) % 3
    original_tier = tier.copy()
    if reverse_test_labels:
        tier[train_count:] = 2 - tier[train_count:]
    features = pd.DataFrame(np.ones((count, 14)), columns=FEATURE_NAMES)
    features["n_chars"] = original_tier + 1
    rows = pd.DataFrame({"source_row": range(count), "prompt_group": [f"group-{index}" for index in range(count)],
                         "split": ["train"] * train_count + ["test"] * (count - train_count)})
    labels = pd.DataFrame({"tier": pd.array(tier, dtype="Int64"), "eligible": True})
    features.to_parquet(prepared / "features.parquet", index=False)
    rows.to_parquet(prepared / "rows.parquet", index=False)
    labels.to_parquet(prepared / "labels.parquet", index=False)
    config = {"dataset": {"path": "raw-provenance.bin", "sha256": sha256_file(raw)}, "seed": 42,
              "quality_threshold": 0.75, "cv_folds": 5}
    write_json(prepared / "manifest.json", {
        "dataset_sha256": config["dataset"]["sha256"],
        "files_sha256": {name: sha256_file(prepared / name) for name in ("features.parquet", "rows.parquet", "labels.parquet")},
        "tier_map": {"a": 0, "b": 1, "c": 2},
        "counts": {"train": {"all": 150, "eligible": 150}, "test": {"all": 30, "eligible": 30}},
    })
    return config


def small_models(seed=42):
    return {"dummy": DummyClassifier(strategy="most_frequent"),
            "length": DecisionTreeClassifier(max_depth=1, random_state=seed),
            "logistic": DecisionTreeClassifier(max_depth=2, random_state=seed),
            "random_forest": DecisionTreeClassifier(max_depth=2, random_state=seed)}


class TierMetricTests(unittest.TestCase):
    def test_macro_f1_includes_unpredicted_tier_and_supports(self):
        result = classification_metrics([0, 1, 2], [0, 0, 0])
        self.assertAlmostEqual(result["f1_macro"], 1/6)
        self.assertEqual(result["tier2_recall"], 0)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(result["per_class"]["2"]["support"], 1)
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0, 3])


class TierProvenanceTests(unittest.TestCase):
    def test_raw_hash_is_verified_before_loading_prepared_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            (root / config["dataset"]["path"]).write_bytes(b"modified original")
            with patch("routerbench.tier_experiment.pd.read_parquet") as reader:
                with self.assertRaises(ValueError):
                    load_tier_inputs(root, config)
                reader.assert_not_called()

    def test_prepared_hash_is_verified_before_reading_any_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            path = root / "data/processed/features.parquet"
            path.write_bytes(path.read_bytes() + b"changed")
            with patch("routerbench.tier_experiment.pd.read_parquet") as reader:
                with self.assertRaises(ValueError):
                    load_tier_inputs(root, config)
                reader.assert_not_called()


class TierExperimentTests(unittest.TestCase):
    def test_complete_small_run_uses_cv_winner_and_training_only_permutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            output = root / "results"
            with patch("routerbench.tier_experiment.build_models", side_effect=small_models):
                report = run_tier_experiment(root, config, output)
            self.assertEqual(report["selected_model"], "logistic")
            self.assertEqual(report["train_eligible_rows"], 150)
            self.assertEqual(report["test_eligible_rows"], 30)
            self.assertFalse(report["selection_used_test"])
            self.assertTrue(report["final_test_evaluated"])
            cv_folds = pd.read_parquet(output / "cv_folds.parquet")
            permutation_folds = pd.read_parquet(output / "permutation_folds.parquet")
            self.assertEqual(len(cv_folds), 150)
            self.assertTrue(cv_folds["source_row"].lt(150).all())
            self.assertTrue(permutation_folds["source_row"].lt(150).all())
            self.assertEqual(len(permutation_folds), 150)
            prediction = pd.read_parquet(output / "test_predictions.parquet")
            self.assertEqual(prediction["source_row"].tolist(), list(range(150, 180)))
            self.assertEqual(report["primary_test_result"]["f1_macro"], 1.0)
            self.assertEqual(report["permutation_importance"]["top_three"][0], "n_chars")
            importance = pd.read_parquet(output / "permutation_importance.parquet")
            self.assertAlmostEqual(importance["top3_fold_fraction"].sum(), 3.0)
            self.assertTrue((output / "model.joblib").exists())
            self.assertEqual(sha256_file(output / "selection.json"), report["selection_sha256"])
            with self.assertRaises(FileExistsError):
                run_tier_experiment(root, config, output)

    def test_changing_only_test_labels_does_not_change_cv_selection(self):
        selections = []
        scores = []
        for reverse in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = fixture(root, reverse_test_labels=reverse)
                with patch("routerbench.tier_experiment.build_models", side_effect=small_models):
                    report = run_tier_experiment(root, config, root / "results")
                selections.append((report["selected_model"], report["cv"]))
                scores.append(report["primary_test_result"]["f1_macro"])
        self.assertEqual(selections[0], selections[1])
        self.assertNotEqual(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
