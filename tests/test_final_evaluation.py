import unittest

import pandas as pd

from routerbench.final_evaluation import assess_primary, cluster_bootstrap


class FinalEvaluationTests(unittest.TestCase):
    def test_noninferiority_requires_prespecified_margin_and_lower_cost(self):
        self.assertTrue(assess_primary({"success_rate_delta_95ci": [-0.009, 0.005], "cost_per_1000_delta_95ci": [-2, -1]}, 0.01)["both_primary_criteria_met"])
        crossing = assess_primary({"success_rate_delta_95ci": [-0.02, 0.005], "cost_per_1000_delta_95ci": [-2, -1]}, 0.01)
        self.assertFalse(crossing["both_primary_criteria_met"])
        self.assertEqual(crossing["quality_status"], "inconclusive")
        self.assertFalse(assess_primary({"success_rate_delta_95ci": [-0.01, 0.005], "cost_per_1000_delta_95ci": [-2, -1]}, 0.01)["quality_noninferiority_supported"])

    def test_group_resampling_preserves_shared_reference_pairs(self):
        left = pd.DataFrame({"success": [1, 0, 1, 0], "cost": [1.0]*4})
        right = pd.DataFrame({"success": [1, 0, 1, 0], "cost": [2.0]*4})
        result = cluster_bootstrap(left, right, pd.Series(["a", "a", "b", "b"]), replicates=100)
        self.assertEqual(result["success_rate_delta_95ci"], [0, 0])
        self.assertEqual(result["cost_per_1000_delta_95ci"], [-1000, -1000])


if __name__ == "__main__":
    unittest.main()
