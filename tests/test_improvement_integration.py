"""Small real-model integration of v3's development-only artifact contract."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from routerbench.data import sha256_file, write_json
from routerbench.improvement import choose_development_point, run_improvement


def fixture(root: Path):
    ids = pd.Index(range(100, 142), name="source_row")
    texts = ["advanced theorem proof" if i % 2 == 0 else "simple known fact" for i in range(len(ids))]
    inputs = pd.DataFrame({"prompt_text": texts}, index=ids)
    scores = pd.DataFrame({"strong": np.ones(len(ids)),
                           "weak": [int(i % 2 != 0) for i in range(len(ids))]}, index=ids, dtype=float)
    costs = pd.DataFrame({"strong": np.full(len(ids), .01), "weak": np.full(len(ids), .001)}, index=ids)
    parts = ["fit"]*12 + ["unused_fit_pilot"]*6 + ["calibration"]*12 + ["validation"]*12
    rows = pd.DataFrame({"source_row": ids, "prompt_group": [f"group-{i}" for i in ids],
                         "split": "train", "development_split": parts}, index=ids)
    source = root / "reports/source"
    source.mkdir(parents=True)
    write_json(source / "summary.json", {"kind": "fixture development source", "outer_test_accessed": False})
    write_json(source / "selection.json", {"test_outcomes_used_for_selection": False})
    validation_ids = rows.index[rows["development_split"].eq("validation")]
    bert = pd.DataFrame({"probability": [.9 if value == 0 else .1
                                         for value in scores.loc[validation_ids, "weak"]]}, index=validation_ids)
    bert.to_parquet(source / "bert_development.parquet")
    prepared = root / "data/processed"
    prepared.mkdir(parents=True)
    inputs.to_parquet(prepared / "features.parquet")
    rows.to_parquet(prepared / "rows.parquet")
    scores.to_parquet(prepared / "labels.parquet")
    manifest = {"files_sha256": {name: sha256_file(prepared / name)
                                 for name in ("features.parquet", "rows.parquet", "labels.parquet")}}
    config = {
        "experiment_id": "v3-integration-fixture", "seed": 42,
        "outer_test_evaluation_allowed": False, "automatic_live_promotion_allowed": False,
        "reference_split_run": "reports/source",
        "reference_split_summary_sha256": sha256_file(source / "summary.json"),
        "reference": {"selection_file": "reports/source/selection.json",
                      "selection_sha256": sha256_file(source / "selection.json"),
                      "development_probabilities_file": "reports/source/bert_development.parquet",
                      "development_probabilities_sha256": sha256_file(source / "bert_development.parquet"),
                      "threshold_grid": [value/10 for value in range(1, 10)]},
        "expected_population": {"fit": 18, "calibration": 12, "validation": 12},
        "common_pair": {"strong": "strong", "weak": "weak"}, "quality_threshold": .75,
        "selection": {"quality_loss_margin": .005},
        "lexical": {"max_features_per_block": 40, "min_df": 1},
        "candidates": [
            {"id": "absolute", "objective": "absolute", "representation": "hybrid", "alpha": .001},
            {"id": "pairwise", "objective": "pairwise", "representation": "hybrid", "alpha": .001}],
        "absolute_probability_grid": [value/10 for value in range(1, 10)],
        "pairwise_tradeoff_grid": [0, 1, 2, 5, 10, 20, 40, 80, 160],
        "bootstrap_replicates": 100, "latency_sample_size": 2,
    }
    return config, (inputs, scores, costs, rows, manifest)


class ImprovementIntegrationTests(unittest.TestCase):
    def test_real_models_preserve_development_only_selection_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = fixture(root)
            original_rows = loaded[3].copy(deep=True)
            output = root / "reports/v3"
            with patch("routerbench.improvement.load_development_inputs", return_value=loaded) as loader:
                result = run_improvement(root, config, output)
            loader.assert_called_once_with(root, config)
            self.assertFalse(result["outer_test_accessed_for_v3"])
            self.assertFalse(result["confirmation_of_superiority"])
            self.assertFalse(result["automatic_live_promotion"])
            self.assertEqual(result["partition_counts"], {"fit": 18, "calibration": 12, "validation": 12})
            pd.testing.assert_frame_equal(loaded[3], original_rows)
            persisted = pd.read_parquet(output / "development_rows.parquet")
            self.assertTrue(persisted["split"].eq("train").all())
            self.assertEqual(persisted["original_development_split"].eq("unused_fit_pilot").sum(), 6)
            self.assertEqual(persisted["development_split"].eq("fit").sum(), 18)
            floor = result["fixed_strong"]["success_rate"]-.005
            self.assertAlmostEqual(result["quality_floor"], floor)
            reference = result["routellm_bert"]
            self.assertEqual([point["parameter"] for point in reference["curve"]], config["reference"]["threshold_grid"])
            self.assertEqual(reference["selected_point"], choose_development_point(reference["curve"], floor))
            self.assertGreaterEqual(reference["selected_point"]["metrics"]["success_rate"], floor)
            self.assertEqual(set(result["candidates"]), {"absolute", "pairwise"})
            all_points = []
            for name, entry in result["candidates"].items():
                self.assertEqual(len(entry["curve"]), 9)
                self.assertEqual(entry["selected_point"], choose_development_point(entry["curve"], floor))
                all_points.extend(entry["curve"])
                self.assertEqual(entry["artifact_sha256"], sha256_file(output / entry["artifact"]))
                fitted = joblib.load(output / entry["artifact"])
                self.assertEqual(fitted.fit_rows_, 18)
                self.assertEqual(fitted.calibration_rows_, 12)
                self.assertEqual(entry["single_request_latency"]["n"], 2)
                self.assertEqual(entry["paired_development_comparison_vs_bert"]["replicates"], 100)
                self.assertGreaterEqual(entry["selected_point"]["metrics"]["success_rate"], floor)
            chosen = choose_development_point(all_points, floor)
            self.assertEqual(result["selected_for_future_evaluation"]["point"], chosen)
            self.assertTrue(result["selected_for_future_evaluation"]["requires_independent_evaluation"])
            for name, digest in result["files_sha256"].items():
                self.assertEqual(Path(name).name, name)
                self.assertEqual(sha256_file(output / name), digest)
            disk = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(disk["configuration"], config)
            self.assertEqual(disk["outer_test_accessed_for_v3"], False)
            self.assertFalse(any("test" in path.name for path in output.iterdir()))
            with patch("routerbench.improvement.load_development_inputs") as loader:
                with self.assertRaises(FileExistsError):
                    run_improvement(root, config, output)
                loader.assert_not_called()

    def test_changed_sources_are_rejected_before_loading_or_training(self):
        for filename in ("summary.json", "selection.json", "bert_development.parquet"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = fixture(root)
                with (root / "reports/source" / filename).open("ab") as stream:
                    stream.write(b"changed")
                with patch("routerbench.improvement.load_development_inputs") as loader:
                    with self.assertRaisesRegex(ValueError, "Fuente modificada"):
                        run_improvement(root, config, root / "reports/v3")
                    loader.assert_not_called()
                self.assertFalse((root / "reports/v3").exists())

    def test_outer_test_rows_and_invalid_reference_contracts_fail_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = fixture(root)
            loaded[3].loc[100, "split"] = "test"
            with patch("routerbench.improvement.load_development_inputs", return_value=loaded), \
                    patch("routerbench.improvement.SuccessCostRouter.fit") as absolute_fit, \
                    patch("routerbench.pairwise.PairwiseRegretRouter.fit") as pairwise_fit:
                with self.assertRaisesRegex(ValueError, "train exterior"):
                    run_improvement(root, config, root / "reports/v3")
                absolute_fit.assert_not_called()
                pairwise_fit.assert_not_called()
            self.assertFalse((root / "reports/v3").exists())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root)
            selection = root / config["reference"]["selection_file"]
            write_json(selection, {"test_outcomes_used_for_selection": True})
            config["reference"]["selection_sha256"] = sha256_file(selection)
            with patch("routerbench.improvement.load_development_inputs") as loader:
                with self.assertRaisesRegex(ValueError, "exclusivamente con desarrollo"):
                    run_improvement(root, config, root / "reports/v3")
                loader.assert_not_called()

    def test_misaligned_bert_and_forbidden_evaluation_flags_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = fixture(root)
            probabilities = root / config["reference"]["development_probabilities_file"]
            bert = pd.read_parquet(probabilities)
            bert.iloc[::-1].to_parquet(probabilities)
            config["reference"]["development_probabilities_sha256"] = sha256_file(probabilities)
            with patch("routerbench.improvement.load_development_inputs", return_value=loaded), \
                    patch("routerbench.improvement.SuccessCostRouter.fit") as fit:
                with self.assertRaisesRegex(ValueError, "BERT desalineadas"):
                    run_improvement(root, config, root / "reports/v3")
                fit.assert_not_called()
            for flag in ("outer_test_evaluation_allowed", "automatic_live_promotion_allowed"):
                forbidden = copy.deepcopy(config)
                forbidden[flag] = True
                with self.subTest(flag=flag), patch("routerbench.improvement.load_development_inputs") as loader:
                    with self.assertRaisesRegex(ValueError, "no permite"):
                        run_improvement(root, forbidden, root / "reports/v3")
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
