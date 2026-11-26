from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from select_agent_trajectory import select_threshold, THRESHOLDS
from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor
from test_agent_trajectory_learning import MODELS, rows


class SelectionTests(unittest.TestCase):
    def predictor(self):
        return TrajectorySuccessPredictor(MODELS, max_text_features=60).fit(rows("fit"), rows("cal"))

    def test_selection_is_grouped_reference_evidence_and_frozen_grid(self):
        report = select_threshold(self.predictor(), rows("dev"))
        self.assertEqual(report["groups"], 16)
        self.assertEqual(report["states"], 16)
        self.assertEqual(report["threshold_grid"], list(THRESHOLDS))
        self.assertFalse(report["independent_validation"])
        self.assertEqual(len(report["grid"]), len(THRESHOLDS))
        for point in report["grid"]:
            self.assertEqual(sum(point["model_selections"].values()), 16)

    def test_unknown_cost_and_incomplete_pair_block_selection(self):
        values = rows("dev")
        values[0] = replace(values[0], remaining_provider_cost_usd=None)
        with self.assertRaisesRegex(ValueError, "Unknown development costs"):
            select_threshold(self.predictor(), values)
        with self.assertRaisesRegex(ValueError, "every model"):
            select_threshold(self.predictor(), rows("dev")[1:])

    def test_training_data_cannot_be_used_for_threshold_selection(self):
        with self.assertRaises(ValueError):
            select_threshold(self.predictor(), rows("fit"))


if __name__ == "__main__":
    unittest.main()
