"""Offline contracts for the product's explicit initial activity policy."""
import unittest

from relay_router.maintenance_routing import ActivityRouter


class ActivityRoutingTests(unittest.TestCase):
    def test_models_are_selected_per_harness_activity_without_historical_scores(self):
        router = ActivityRouter("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")
        decisions = [router.route(activity) for activity in ("inspection", "editing", "repair")]
        self.assertEqual([row["selected_model"] for row in decisions],
                         [router.inspection_model, router.editing_model, router.editing_model])
        for decision in decisions:
            self.assertEqual(decision["policy_mode"], "activity_policy_v1")
            self.assertEqual(decision["evaluation_scope"], "configured_activity_policy_not_learned")
            self.assertNotIn("probability", decision)
            self.assertNotIn("predicted_cost", decision)
            self.assertTrue(decision["reason"])
        self.assertFalse(router.manifest()["learned"])

    def test_fixed_control_always_uses_same_model_and_labels_override(self):
        router = ActivityRouter("model/inspection", "model/editing", mode="always_strong")
        for activity in ("inspection", "editing", "repair"):
            decision = router.route(activity, attempt=2)
            self.assertEqual(decision["selected_model"], "model/editing")
            self.assertEqual(decision["policy_mode"], "always_strong")
            self.assertEqual(decision["attempt"], 2)

    def test_verified_policy_escalates_only_for_the_repair_activity(self):
        router = ActivityRouter("cheap", "strong", mode="verified")
        self.assertEqual([router.route(activity)["selected_model"]
                          for activity in ("inspection", "editing", "repair")], ["cheap", "cheap", "strong"])
        self.assertFalse(router.manifest()["learned"])
        self.assertEqual(router.manifest()["policy"], "verified_cascade_v1")

    def test_unknown_activities_and_invalid_catalog_values_are_rejected(self):
        router = ActivityRouter("inspection", "editing")
        for activity in ("run rm -rf", "choose/other", "", None):
            with self.assertRaises(ValueError):
                router.route(activity)
        for attempt in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                router.route("repair", attempt=attempt)
        for model in ("", "model one", " model", None):
            with self.assertRaises(ValueError):
                ActivityRouter(model, "editing")
        with self.assertRaises(ValueError):
            ActivityRouter("a", "b", mode="auto")


if __name__ == "__main__":
    unittest.main()
