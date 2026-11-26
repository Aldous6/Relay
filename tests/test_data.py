import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from routerbench.cli import run_prepare
from routerbench.data import audit_frame, hash_partition, load_verified_pickle, prompt_metadata


def fixture():
    size = 60
    return pd.DataFrame({
        "sample_id": [str(i) for i in range(size)],
        "eval_name": ["example"] * size,
        "prompt": [repr(["System", f"Question {i}"]) for i in range(size)],
        "a": [0.0 if i % 4 == 0 else 1.0 for i in range(size)],
        "b": [0.0] * size,
        "c": [0.0] * size,
        "a|total_cost": [1.0] * size,
        "b|total_cost": [2.0] * size,
        "c|total_cost": [3.0] * size,
    })


class DataTests(unittest.TestCase):
    def test_hash_mismatch_never_deserializes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bad.pkl"
            source.write_bytes(b"not a trusted pickle")
            with patch("routerbench.data.pd.read_pickle") as reader:
                with self.assertRaises(ValueError):
                    load_verified_pickle(source)
                reader.assert_not_called()

    def test_split_stable_under_reordering_and_duplicates(self):
        groups = pd.Series(["same", "other", "same", "third"])
        original = hash_partition(groups)
        reordered = hash_partition(groups.iloc[::-1]).sort_index()
        pd.testing.assert_series_equal(original, reordered)
        self.assertEqual(original.iloc[0], original.iloc[2])
        with self.assertRaises(ValueError):
            hash_partition(groups, test_fraction=1.0)

    def test_metadata_uses_positions_with_nondefault_index(self):
        frame = fixture().iloc[:2].copy()
        frame.index = [100, 200]
        metadata = prompt_metadata(frame)
        self.assertEqual(metadata["source_row"].tolist(), [0, 1])
        self.assertEqual(metadata.loc[0, "prompt_text"], "System\nQuestion 0")

    def test_duplicate_conflicts_and_exclusions_are_visible(self):
        frame = fixture().iloc[:2].copy()
        frame.loc[1, "prompt"] = frame.loc[0, "prompt"]
        report = audit_frame(frame, [0.75])
        self.assertEqual(report["duplicate_extra_rows"], 1)
        self.assertEqual(report["duplicate_groups_with_conflicting_results"], 1)
        self.assertEqual(report["thresholds"]["0.75"]["ineligible_raw"], 1)
        self.assertEqual(report["thresholds"]["0.75"]["eligible_after_dedup"], 0)

    def test_invalid_audit_thresholds_rejected(self):
        for thresholds in ([], [-1.0], [2.0], [float("nan")]):
            with self.subTest(thresholds=thresholds):
                with self.assertRaises(ValueError):
                    audit_frame(fixture(), thresholds)

    def test_prepare_keeps_ineligible_rows_and_features_separate(self):
        frame = fixture()
        config = {"dataset": {"expected_models": 3}, "seed": 42,
                  "split": {"strategy": "prompt_hash", "test_fraction": 0.2},
                  "tier_sizes": [1, 1, 1], "quality_threshold": 0.75}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.txt"
            source.write_text("fixture provenance", encoding="utf-8")
            run_prepare(frame, config, Path(tmp) / "output", source)
            features = pd.read_parquet(Path(tmp) / "output/features.parquet")
            rows = pd.read_parquet(Path(tmp) / "output/rows.parquet")
            labels = pd.read_parquet(Path(tmp) / "output/labels.parquet")
            self.assertEqual(features.shape, (60, 14))
            self.assertEqual(len(rows), 60)
            self.assertEqual(labels["tier"].isna().sum(), 15)
            self.assertFalse(set(features.columns) & {"tier", "eval_name", "sample_id", "eligible"})
            self.assertEqual(set(rows["split"]), {"train", "test"})
            with self.assertRaises(FileExistsError):
                run_prepare(frame, config, Path(tmp) / "output", source)


if __name__ == "__main__":
    unittest.main()
