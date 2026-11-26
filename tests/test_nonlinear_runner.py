"""Small real nonlinear model through the development-only experiment runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from routerbench.data import sha256_file
from routerbench.improvement import run_improvement
from routerbench.nonlinear import NonlinearRegretRouter
from routerbench.pairwise import OUTCOME_CLASSES
from test_improvement_integration import fixture


def nonlinear_fixture(root):
    config, loaded = fixture(root)
    config["experiment_id"] = "nonlinear-runner-fixture"
    config["candidates"] = [{
        "id": "nonlinear", "objective": "pairwise", "backend": "hist_gradient_boosting",
        "representation": "lexical_svd", "n_components": 3,
        "max_leaf_nodes": 3, "max_iter": 5, "min_samples_leaf": 2,
        "l2_regularization": 1.,
    }]
    inputs, _, _, rows, _ = loaded
    fit = rows["development_split"].isin(["fit", "unused_fit_pilot"])
    inputs.loc[fit, "prompt_text"] += " trainingmarker"
    inputs.loc[~fit, "prompt_text"] += " calibrationsecretword"
    return config, loaded


class NonlinearRunnerTests(unittest.TestCase):
    def test_real_nonlinear_backend_keeps_boundaries_and_persists_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = nonlinear_fixture(root)
            original_rows = loaded[3].copy(deep=True)
            output = root / "reports/nonlinear"
            with patch("routerbench.improvement.load_development_inputs", return_value=loaded) as loader, \
                    threadpool_limits(1):
                result = run_improvement(root, config, output)
            loader.assert_called_once_with(root, config)
            self.assertFalse(result["outer_test_accessed_for_v3"])
            self.assertFalse(result["confirmation_of_superiority"])
            self.assertFalse(result["automatic_live_promotion"])
            self.assertEqual(result["partition_counts"], {"fit": 18, "calibration": 12, "validation": 12})
            pd.testing.assert_frame_equal(loaded[3], original_rows)
            candidate = result["candidates"]["nonlinear"]
            self.assertEqual(candidate["specification"]["objective"], "pairwise")
            self.assertEqual(candidate["specification"]["backend"], "hist_gradient_boosting")
            self.assertEqual(len(candidate["curve"]), len(config["pairwise_tradeoff_grid"]))
            self.assertIsNotNone(candidate["selected_point"])
            self.assertGreaterEqual(candidate["selected_point"]["metrics"]["success_rate"], result["quality_floor"])
            self.assertEqual(candidate["paired_development_comparison_vs_bert"]["replicates"], 100)

            model_path = output / candidate["artifact"]
            self.assertEqual(sha256_file(model_path), candidate["artifact_sha256"])
            model = joblib.load(model_path)
            self.assertIsInstance(model, NonlinearRegretRouter)
            self.assertEqual((model.fit_rows_, model.calibration_rows_), (18, 12))
            self.assertEqual(model.classifier_fit_rows_, 18)
            self.assertEqual(model.target_counts_["fit"]["weak_only"], 0)
            self.assertIn("weak_only", model.constants_)
            vocabulary = model.transformer_.lexical_.get_feature_names_out()
            self.assertTrue(any("trainingmarker" in name for name in vocabulary))
            self.assertFalse(any("calibrationsecretword" in name for name in vocabulary))
            self.assertFalse(model.predictor_.estimator.estimator.early_stopping)

            validation = loaded[3].index[loaded[3]["development_split"].eq("validation")]
            stored = pd.read_parquet(output / "nonlinear_predictions.parquet")
            self.assertTrue(stored.index.equals(validation))
            self.assertEqual(stored.columns.tolist(), [*OUTCOME_CLASSES, "expected_quality_gain"])
            self.assertTrue(np.allclose(stored.loc[:, OUTCOME_CLASSES].sum(axis=1), 1))
            self.assertTrue(np.isfinite(stored.to_numpy()).all())
            with threadpool_limits(1):
                replay, costs = model.predict_tables(loaded[0].loc[validation])
            pd.testing.assert_frame_equal(replay, stored)
            self.assertEqual(costs.columns.tolist(), ["strong", "weak"])

            fresh = clone(model)
            self.assertEqual(fresh.get_params(), model.get_params())
            self.assertFalse(hasattr(fresh, "transformer_"))
            self.assertFalse(hasattr(fresh, "predictor_"))
            with self.assertRaises(NotFittedError):
                fresh.predict_tables(loaded[0].loc[validation])
            for name, digest in result["files_sha256"].items():
                self.assertEqual(sha256_file(output / name), digest)
            disk = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(disk["configuration"], config)
            self.assertFalse(any("test" in path.name for path in output.iterdir()))

    def test_outer_test_rows_are_rejected_before_nonlinear_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = nonlinear_fixture(root)
            loaded[3].loc[100, "split"] = "test"
            with patch("routerbench.improvement.load_development_inputs", return_value=loaded), \
                    patch("routerbench.nonlinear.NonlinearRegretRouter.fit") as fit:
                with self.assertRaisesRegex(ValueError, "train exterior"):
                    run_improvement(root, config, root / "reports/nonlinear")
                fit.assert_not_called()
            self.assertFalse((root / "reports/nonlinear").exists())


if __name__ == "__main__":
    unittest.main()
