"""Contracts for paired episode accounting and development-only selection."""
from copy import deepcopy
import unittest

from relay_router.maintenance_comparison import compare_records, select_reference_threshold


POLICY = {"name": "frozen_test_policy", "frozen": True, "learned": False}


def record(task, *, group=None, partition="development", probability=.5,
           choice="weak", weak_success=True, strong_success=True,
           weak_cost=1., strong_cost=4.):
    return {
        "task_id": task, "group_id": group or task, "partition": partition,
        "routing_text": "Repair the authorized Python source.",
        "reference_probability": probability, "relay_choice": choice,
        "branches": {
            "weak": {"success": weak_success, "cost_usd": weak_cost, "total_tokens": 100},
            "strong": {"success": strong_success, "cost_usd": strong_cost, "total_tokens": 400},
        },
    }


class MaintenanceComparisonTests(unittest.TestCase):
    def test_paired_totals_count_complete_episodes_and_actual_selections(self):
        rows = [record("easy", probability=.2),
                record("hard", probability=.8, choice="strong", weak_success=False)]
        original = deepcopy(rows)
        report = compare_records(rows, .5, POLICY)
        result = report["partitions"]["development"]
        self.assertEqual(result["policies"]["relay"]["success_count"], 2)
        self.assertEqual(result["policies"]["relay"]["total_cost_usd"], 5.)
        self.assertEqual(result["policies"]["relay"]["cost_per_success_usd"], 2.5)
        self.assertEqual(result["policies"]["relay"]["total_tokens"], 500)
        self.assertEqual(result["policies"]["weak_first"]["success_count"], 1)
        self.assertEqual(result["paired_comparisons"]["relay_vs_reference"]["success_delta"], 0.)
        self.assertEqual(result["paired_comparisons"]["relay_vs_strong_first"]["relative_cost_savings"], .375)
        self.assertFalse(result["interpretation"]["superiority_claim"])
        self.assertEqual(result["interpretation"]["scope"], "exploratory_development")
        self.assertEqual(rows, original)

    def test_unknown_cost_is_not_zero_and_unknown_usage_remains_unknown(self):
        rows = [record("one", weak_cost=None), record("two", choice="strong")]
        rows[0]["branches"]["weak"]["total_tokens"] = None
        result = compare_records(rows, 0., POLICY)["partitions"]["development"]
        relay = result["policies"]["relay"]
        self.assertIsNone(relay["total_cost_usd"])
        self.assertEqual(relay["known_cost_subtotal_usd"], 4.)
        self.assertEqual(relay["unknown_cost_count"], 1)
        self.assertIsNone(relay["cost_per_success_usd"])
        self.assertIsNone(relay["total_tokens"])
        comparison = result["paired_comparisons"]["relay_vs_reference"]
        self.assertIsNone(comparison["relative_cost_savings"])
        self.assertIsNone(comparison["bootstrap_ci95"]["relative_cost_savings"])
        self.assertEqual(comparison["success_delta"], 0.)
        self.assertEqual(result["policies"]["strong_first"]["total_cost_usd"], 8.)

    def test_zero_reference_cost_and_no_success_do_not_create_savings_or_cost_per_success(self):
        rows = [record("one", weak_success=False, strong_success=False, strong_cost=0.)]
        result = compare_records(rows, 0., POLICY)["partitions"]["development"]
        pair = result["paired_comparisons"]["relay_vs_reference"]
        self.assertIsNone(pair["relative_cost_savings"])
        self.assertIsNone(pair["delta_cost_per_success_usd"])
        self.assertIsNone(pair["bootstrap_ci95"]["delta_cost_per_success_usd"])

    def test_bootstrap_resamples_whole_groups_and_is_reproducible(self):
        rows = [record("one", group="same", weak_success=False),
                record("two", group="same", weak_success=True)]
        first = compare_records(rows, 0., POLICY)
        second = compare_records(rows, 0., POLICY)
        self.assertEqual(first, second)
        result = first["partitions"]["development"]
        self.assertEqual(result["group_count"], 1)
        self.assertEqual(result["group_results"]["same"]["relay"]["success_rate"], .5)
        # Row-wise resampling would produce different success rates here.
        interval = result["paired_comparisons"]["relay_vs_reference"]["bootstrap_ci95"]["success_delta"]
        self.assertEqual(interval, [-.5, -.5])
        self.assertTrue(result["interpretation"]["warnings"])

    def test_outcomes_never_change_frozen_relay_or_reference_choices(self):
        rows = [record("one", probability=.5, choice="weak"),
                record("two", probability=.49, choice="strong")]
        before = compare_records(rows, .5, POLICY)
        for row in rows:
            row["branches"]["weak"].update(success=False, cost_usd=100.)
            row["branches"]["strong"].update(success=True, cost_usd=.001)
        after = compare_records(rows, .5, POLICY)
        self.assertEqual(before["decisions"], after["decisions"])
        self.assertEqual([row["reference_choice"] for row in before["decisions"]], ["strong", "weak"])
        self.assertEqual([row["relay_choice"] for row in before["decisions"]], ["weak", "strong"])

    def test_partitions_remain_separate_and_holdout_choice_is_not_recalibrated(self):
        rows = [record("dev", probability=.1),
                record("test", partition="holdout", probability=.9, choice="weak", weak_success=False)]
        report = compare_records(rows, .5, POLICY)
        self.assertEqual(set(report["partitions"]), {"development", "holdout"})
        holdout = report["partitions"]["holdout"]
        self.assertEqual(holdout["policies"]["relay"]["success_count"], 0)
        self.assertEqual(holdout["policies"]["routellm_bert_transfer"]["success_count"], 1)
        self.assertFalse(holdout["interpretation"]["superiority_claim"])

    def test_duplicate_tasks_and_group_leakage_are_rejected(self):
        for rows in ([record("same"), record("same")],
                     [record("one", group="shared"), record("two", group="shared", partition="holdout")]):
            with self.assertRaises(ValueError):
                compare_records(rows, .5, POLICY)

    def test_invalid_branch_and_selector_values_are_rejected(self):
        mutations = [
            ("reference_probability", float("nan")),
            ("reference_probability", float("inf")),
            ("reference_probability", True),
            ("reference_probability", 1.1),
            ("relay_choice", "best_observed"),
            ("partition", "test"),
            ("routing_text", ""),
        ]
        for key, value in mutations:
            row = record("one")
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                compare_records([row], .5, POLICY)
        for key, values in {"success": [1, 0., None, "true"],
                            "cost_usd": [-1., float("nan"), float("inf"), True, 10 ** 1000],
                            "total_tokens": [-1, 1.5, True]}.items():
            for value in values:
                row = record("one")
                row["branches"]["weak"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    compare_records([row], .5, POLICY)
        for threshold in (-.1, float("nan"), True, 1.1):
            with self.assertRaises(ValueError):
                compare_records([record("one")], threshold, POLICY)
        for policy in ({"name": "unfrozen", "frozen": False, "learned": False},
                       {"name": "unknown", "frozen": True}, {}, None):
            with self.assertRaises(ValueError):
                compare_records([record("one")], .5, policy)

    def test_cost_fields_must_be_explicit_and_aggregates_cannot_overflow(self):
        row = record("one")
        del row["branches"]["weak"]["cost_usd"]
        with self.assertRaisesRegex(ValueError, "cost_usd"):
            compare_records([row], .5, POLICY)
        with self.assertRaisesRegex(ValueError, "agregado"):
            compare_records([record("one", weak_cost=1e308),
                             record("two", weak_cost=1e308)], .5, POLICY)

    def test_reference_selection_uses_only_development_cost_and_quality(self):
        rows = [record("easy", probability=.1),
                record("hard", probability=.9, weak_success=False)]
        result = select_reference_threshold(rows)
        self.assertEqual(result["threshold"], .2)
        chosen = [row for row in result["candidates"] if row["threshold"] == .2][0]
        self.assertEqual(chosen["total_cost_usd"], 5.)
        self.assertEqual(chosen["success_rate"], 1.)
        self.assertFalse(result["candidates"][-1]["eligible"])
        for row in rows:
            row["relay_choice"] = "not_read_by_reference_selection"
        self.assertEqual(select_reference_threshold(rows), result)
        rows[0]["partition"] = "holdout"
        with self.assertRaisesRegex(ValueError, "development"):
            select_reference_threshold(rows)

    def test_reference_endpoint_and_missing_cost_cannot_be_misranked(self):
        row = record("one", probability=1.)
        self.assertEqual(select_reference_threshold([row])["threshold"], 1.000001)
        row["branches"]["weak"]["cost_usd"] = None
        selected = select_reference_threshold([row])
        self.assertEqual(selected["threshold"], 0.)
        self.assertFalse(selected["candidates"][-1]["eligible"])
        row["branches"]["strong"]["cost_usd"] = None
        with self.assertRaisesRegex(ValueError, "costo completo"):
            select_reference_threshold([row])

    def test_undefined_resamples_do_not_create_cost_per_success_interval(self):
        rows = [record("solved"), record("failed", weak_success=False, strong_success=False)]
        pair = compare_records(rows, 0., POLICY)["partitions"]["development"]["paired_comparisons"]["relay_vs_reference"]
        self.assertIsNotNone(pair["delta_cost_per_success_usd"])
        self.assertIsNone(pair["bootstrap_ci95"]["delta_cost_per_success_usd"])


if __name__ == "__main__":
    unittest.main()
