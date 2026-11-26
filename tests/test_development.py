import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from routerbench.data import sha256_file, write_json
from routerbench.development import calibration_diagnostics, development_partition
from routerbench.recommend import recommend


class DevelopmentTests(unittest.TestCase):
    def test_group_split_is_stable_disjoint_and_reorder_invariant(self):
        groups = pd.Series([f"group-{index}" for index in range(500)] + ["group-1"])
        split = development_partition(groups)
        self.assertEqual(set(split), {"fit", "calibration", "validation"})
        self.assertEqual(split.iloc[1], split.iloc[-1])
        pd.testing.assert_series_equal(split, development_partition(groups.iloc[::-1]).sort_index())
        with self.assertRaises(ValueError):
            development_partition(groups, fit_fraction=0.9, calibration_fraction=0.2)

    def test_reliability_bins_include_probability_one_and_brier_manual(self):
        probabilities = pd.DataFrame({"model": [0.0, 0.5, 1.0]})
        scores = pd.DataFrame({"model": [0.0, 1.0, 1.0]})
        diagnostics = calibration_diagnostics(probabilities, scores, 0.75)
        self.assertAlmostEqual(diagnostics["mean_brier"], 0.25/3)
        self.assertEqual(sum(item["n"] for item in diagnostics["reliability_bins_pooled_diagnostic_only"]), 3)
        with self.assertRaises(ValueError):
            calibration_diagnostics(probabilities, scores.iloc[::-1], 0.75)

    def test_modified_artifact_is_rejected_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.joblib").write_bytes(b"untrusted artifact")
            write_json(root / "summary.json", {"variants": {"hybrid": {"artifact": "model.joblib", "artifact_sha256": "0" * 64}}})
            with self.assertRaises(ValueError):
                recommend(root, "hybrid", "question")

    def test_serialized_empty_prompt_is_rejected_before_artifact_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.joblib"
            artifact.write_bytes(b"not executable model data")
            write_json(root / "summary.json", {"variants": {"hybrid": {"artifact": "model.joblib", "artifact_sha256": sha256_file(artifact)}}})
            for prompt in ("[]", '["", " "]'):
                with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "vacío"):
                    recommend(root, "hybrid", prompt)


if __name__ == "__main__":
    unittest.main()
