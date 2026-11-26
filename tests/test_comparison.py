import unittest

import numpy as np
import pandas as pd

from routerbench.comparison import paired_bootstrap


class ComparisonTests(unittest.TestCase):
    def test_same_requests_and_known_difference(self):
        candidate = pd.DataFrame({"success": [1, 0, 1, 0], "cost": [1.0] * 4})
        reference = pd.DataFrame({"success": [1, 0, 1, 0], "cost": [2.0] * 4})
        result = paired_bootstrap(candidate, reference, replicates=100)
        self.assertEqual(result["success_rate_delta"], 0)
        self.assertEqual(result["success_rate_delta_95ci"], [0, 0])
        self.assertEqual(result["relative_call_cost_saving"], 0.5)
        self.assertEqual(result["cost_per_1000_delta_95ci"], [-1000, -1000])
        with self.assertRaises(ValueError):
            paired_bootstrap(candidate, reference.iloc[::-1], replicates=100)

    def test_no_reference_cost_does_not_invent_relative_savings(self):
        results = pd.DataFrame({"success": [0, 0], "cost": [0.0, 0.0]})
        comparison = paired_bootstrap(results, results, replicates=100)
        self.assertIsNone(comparison["relative_call_cost_saving"])
        self.assertIsNone(comparison["relative_call_cost_saving_95ci"])


if __name__ == "__main__":
    unittest.main()
