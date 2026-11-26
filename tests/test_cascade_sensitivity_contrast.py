"""Independent algebraic and analytical-distribution checks; no provider calls."""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from contrast_cascade_sensitivity import threshold, binomial_percentile_quality_interval


class SensitivityContrastTests(unittest.TestCase):
    def test_threshold_uses_ratio_of_totals_and_includes_failed_task_spend(self):
        # 2 of 4 successes, $6 known including $2 on failed tasks; baseline 3
        # successes for $15. At $4 unknown cost the costs per accepted tie at $5.
        self.assertEqual(threshold(6, 2, 15, 3, 0), 4)

    def test_exact_threshold_brackets_savings_inequality(self):
        for target in [F(0), F(1,10), F(1,2)]:
            bound = threshold(F(3,100), 7, F(12,100), 9, target)
            def saving(x):
                return 1 - ((F(3,100) + x) / 7) / (F(12,100) / 9)
            self.assertEqual(saving(bound), target)
            self.assertLess(saving(bound + F(1,100000)), target)
            self.assertGreater(saving(bound - F(1,100000)), target)

    def test_negative_threshold_preserved_as_no_feasible_nonnegative_cost(self):
        self.assertEqual(threshold(20, 1, 10, 1, 0), -10)

    def test_invalid_denominators_and_targets_rejected(self):
        for args in [(0,0,1,1,0), (0,1,1,0,0), (0,True,1,1,0), (0,1,0,1,0), (0,1,1,1,1), (-1,1,1,1,0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                threshold(*args)

    def test_exact_empirical_bootstrap_one_loss_in_43(self):
        self.assertEqual(binomial_percentile_quality_interval(43,1), [-F(3,43), F(0)])

    def test_exact_bootstrap_degenerate_does_not_invent_variation(self):
        self.assertEqual(binomial_percentile_quality_interval(43,0), [F(0),F(0)])
        self.assertEqual(binomial_percentile_quality_interval(43,43), [F(-1),F(-1)])

    def test_small_bootstrap_distribution_can_be_enumerated_by_hand(self):
        # n=2, one observed loss: {-1:1/4, -1/2:1/2, 0:1/4}.
        self.assertEqual(binomial_percentile_quality_interval(2,1), [F(-1),F(0)])

    def test_invalid_distribution_arguments_rejected(self):
        for args in [(0,0), (2,3), (3,-1), (True,1), (3,False)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                binomial_percentile_quality_interval(*args)


if __name__ == "__main__":
    unittest.main()
