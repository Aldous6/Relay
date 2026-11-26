import unittest

import pandas as pd

from routerbench.improvement import activate_full_fit, choose_development_point


class ImprovementTests(unittest.TestCase):
    def test_activation_preserves_calibration_validation_and_refuses_outer_test(self):
        rows = pd.DataFrame({"split": ["train"]*4, "prompt_group": list("abcd"),
                             "development_split": ["fit", "unused_fit_pilot", "calibration", "validation"]})
        result = activate_full_fit(rows)
        self.assertEqual(result["development_split"].tolist(), ["fit", "fit", "calibration", "validation"])
        self.assertEqual(rows.loc[1, "development_split"], "unused_fit_pilot")
        rows.loc[1, "split"] = "test"
        with self.assertRaises(ValueError):
            activate_full_fit(rows)

    def test_selection_is_minimum_cost_subject_to_declared_quality(self):
        def point(name, quality, cost, parameter):
            return {"candidate_id": name, "parameter": parameter,
                    "metrics": {"success_rate": quality, "cost_per_1000": cost}}
        points = [point("cheap_failure", .6, 1, .5), point("a", .79, 3, .5), point("b", .8, 3, .5), point("c", .785, 2, .5)]
        self.assertEqual(choose_development_point(points, .78)["candidate_id"], "c")
        self.assertEqual(choose_development_point(points, .79)["candidate_id"], "b")
        self.assertIsNone(choose_development_point(points, .9))
        with self.assertRaises(ValueError):
            choose_development_point([point("bad", .8, float("nan"), .5)], .78)


if __name__ == "__main__":
    unittest.main()
