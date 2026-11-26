import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import relay_trajectory_local as runner


class LocalTrajectoryTests(unittest.TestCase):
    def test_preparation_has_disjoint_groups_and_reserved_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan"
            with patch.dict("os.environ", {}, clear=True):
                plan = runner.prepare(output)
            self.assertEqual(plan["partitions"], {"fit": 6, "calibration": 3, "development": 3, "evaluation": 6})
            self.assertFalse(plan["reserved_partition_executed"])
            splits = runner.make_splits(plan["seed"])
            self.assertEqual(runner.object_hash(splits), plan["task_splits_sha256"])
            self.assertEqual(plan["decision_indices"], [0, 2])
            with self.assertRaises(FileExistsError):
                runner.prepare(output)

    def test_censor_removes_whole_pair_only_from_training(self):
        models = ("fixture/a", "fixture/b")
        branches = [{"anchor_index": 0, "candidate": model, "row": {"model": model}, "censored_reason": None}
                    for model in models]
        self.assertEqual(runner.paired_training_rows({"branches": branches}, models)[1], 0)
        branches[0]["censored_reason"] = "infrastructure_failure"
        self.assertEqual(runner.paired_training_rows({"branches": branches}, models), ([], 1))
        self.assertEqual(len(branches), 2)

    def test_no_execution_without_account_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan"
            runner.prepare(output)
            with self.assertRaisesRegex(ValueError, "Free"):
                runner.execute(output)


if __name__ == "__main__":
    unittest.main()
