"""Episode cost completeness, paired diagnostics and non-promotional scope."""
from copy import deepcopy
import json
import unittest

from routerbench.agent_metrics import annotate_costs, summarize


PRICING = {"model-v1": {"input_per_million": 2., "output_per_million": 4.,
                         "cached_input_per_million": .5, "source": "fixture tariff", "as_of": "2026-11-12"}}
RATES = {"router_usd_per_second": .1, "verification_usd_per_second": .2, "tool_usd_per_second": .3}


def event(**changes):
    result = {"step": 0, "attempt": 0, "is_retry": False, "model": "model-v1",
              "router_seconds": .1, "generation_seconds": 1., "verification_seconds": .2, "tool_seconds": .3,
              "usage": {"input_tokens": 1000, "output_tokens": 500, "cached_input_tokens": 200},
              "reported_llm_cost_usd": None}
    result.update(changes)
    return result


def episode(identifier, policy, *, success=True, cost=1., mode="simulated", group=None, **changes):
    events = annotate_costs([event(reported_llm_cost_usd=cost)], {}, {name: 0. for name in RATES})
    result = {"episode_id": identifier, "source_group": group or identifier, "family": "fixture family",
              "policy": policy, "success": success, "termination": "success" if success else "max_steps",
              "events": events, "end_to_end_seconds": 1.6,
              "provenance": {"execution_mode": mode, "evaluation_split": "evaluation", "independent_validation": True}}
    result.update(changes)
    return result


def paired(mode="simulated", n=4):
    return [episode(str(i), policy, cost=cost, mode=mode) for policy, cost in [("relay", 1.), ("always_strong", 2.)] for i in range(n)]


class AgentCostTests(unittest.TestCase):
    def test_empty_preflight_validates_tariffs_without_events(self):
        self.assertEqual(annotate_costs([], PRICING, RATES), [])
        self.assertEqual(annotate_costs([], {}, {}), [])
        with self.assertRaises(ValueError):
            annotate_costs([], {"model-v1": {**PRICING["model-v1"], "cached_input_per_million": True}}, RATES)
        with self.assertRaises(ValueError):
            annotate_costs([], PRICING, {"router_usd_per_second": -1})

    def test_cached_subset_and_retry_charged_once_including_failed_task(self):
        original = [event(), event(attempt=1, is_retry=True, reported_llm_cost_usd=.8)]
        before = deepcopy(original)
        annotated = annotate_costs(original, PRICING, RATES)
        self.assertEqual(original, before)
        self.assertAlmostEqual(annotated[0]["costs"]["llm_usd"], .0037)
        self.assertEqual(annotated[0]["costs"]["llm_cost_basis"], "estimated")
        self.assertEqual(annotated[1]["costs"]["llm_cost_basis"], "reported")
        self.assertAlmostEqual(annotated[0]["costs"]["total_usd"], .1437)
        failed = episode("failure", "relay", success=False, events=annotated)
        result = summarize([failed, episode("success", "relay", cost=1.)], {}, bootstrap_replicates=100)
        metrics = result["policies"]["relay"]
        self.assertEqual(metrics["success_rate"], .5)
        self.assertEqual(metrics["retry_calls"], 1)
        self.assertAlmostEqual(metrics["costs"]["retry_llm_usd"], .8)
        self.assertAlmostEqual(metrics["costs"]["total_usd"], 2.0837)
        self.assertAlmostEqual(metrics["costs"]["cost_per_task_usd"], 2.0837/2)
        self.assertEqual(metrics["tokens"]["input_tokens"]["total"], 3000)

    def test_missing_cache_tokens_or_tariff_not_assumed_zero(self):
        for usage, tariff in [({"input_tokens": 1000, "output_tokens": 0, "cached_input_tokens": None}, PRICING),
                              ({"input_tokens": 1000, "output_tokens": 0, "cached_input_tokens": 10},
                               {"model-v1": {**PRICING["model-v1"], "cached_input_per_million": None}})]:
            with self.subTest(usage=usage):
                value = annotate_costs([event(usage=usage)], tariff, RATES)[0]["costs"]
                self.assertIsNone(value["llm_usd"])
                self.assertIsNone(value["total_usd"])
                self.assertEqual(value["known_components"], 3)
                self.assertAlmostEqual(value["known_subtotal_usd"], .14)
        no_cache = event(usage={"input_tokens": 1000, "output_tokens": 0, "cached_input_tokens": 0})
        tariffs = {"model-v1": {**PRICING["model-v1"], "cached_input_per_million": None}}
        self.assertEqual(annotate_costs([no_cache], tariffs, RATES)[0]["costs"]["llm_usd"], .002)

    def test_missing_rate_duration_and_provider_outcome_remain_unknown(self):
        item = event(router_seconds=None, usage={"input_tokens": None, "output_tokens": None, "cached_input_tokens": None})
        result = annotate_costs([item], PRICING, {**RATES, "tool_usd_per_second": None})[0]["costs"]
        self.assertEqual(result["known_components"], 1)
        self.assertIsNone(result["total_usd"])
        self.assertAlmostEqual(result["known_subtotal_usd"], .04)
        reported = annotate_costs([event(reported_llm_cost_usd=0., usage={})], {}, RATES)[0]
        self.assertEqual(reported["costs"]["llm_usd"], 0.)
        self.assertEqual(reported["costs"]["llm_cost_basis"], "reported")

    def test_prices_use_exact_version_and_metadata_is_preserved(self):
        self.assertIsNone(annotate_costs([event(model="alias")], PRICING, RATES)[0]["costs"]["llm_usd"])
        self.assertEqual(annotate_costs([event()], PRICING, RATES)[0]["costs"]["pricing"]["as_of"], "2026-11-12")

    def test_invalid_values_and_cached_subset_rejected_even_with_reported_cost(self):
        invalid = [event(reported_llm_cost_usd=True), event(router_seconds=float("nan")), event(tool_seconds=-1),
                   event(attempt=True), event(usage={"input_tokens": 1, "output_tokens": 0, "cached_input_tokens": 2}, reported_llm_cost_usd=1),
                   event(usage={"input_tokens": 1.5, "output_tokens": 0, "cached_input_tokens": 0}), event(is_retry=1)]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                annotate_costs([item], PRICING, RATES)
        for value in [True, -1, float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                annotate_costs([event()], PRICING, {**RATES, "router_usd_per_second": value})
        with self.assertRaises(ValueError):
            annotate_costs([event()], {"model-v1": {**PRICING["model-v1"], "output_per_million": False}}, RATES)


class AgentSummaryTests(unittest.TestCase):
    def test_paired_ids_are_aligned_by_identity_and_bootstrap_clusters_sources(self):
        rows = paired(mode="external")
        rows[0]["success"] = False
        rows[0]["source_group"] = rows[1]["source_group"] = "shared"
        rows[4]["source_group"] = rows[5]["source_group"] = "shared"
        result = summarize(rows[::-1], {}, bootstrap_replicates=300)
        again = summarize(rows, {}, bootstrap_replicates=300)
        comparison = result["comparisons"]["relay"]["paired"]
        self.assertEqual(comparison, again["comparisons"]["relay"]["paired"])
        self.assertEqual(comparison["n_source_groups"], 3)
        self.assertEqual(comparison["bootstrap_unit"], "source_group")
        self.assertEqual(comparison["success_rate_delta"], -.25)
        self.assertEqual(comparison["relative_total_cost_saving"], .5)
        self.assertEqual(comparison["cost_per_task_usd_delta"], -1.)
        self.assertEqual(comparison["relative_total_cost_saving_95ci"], [.5, .5])

    def test_incomplete_or_changed_pairs_rejected_without_dropping_rows(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows[-1].update(family="changed"),
                       lambda rows: rows[-1].update(source_group="changed"), lambda rows: rows.append(deepcopy(rows[0]))):
            rows = paired()
            mutate(rows)
            with self.assertRaises(ValueError):
                summarize(rows, {}, bootstrap_replicates=100)

    def test_simulation_never_qualifies_even_with_integrity_and_independence_claims(self):
        rows = paired()
        for row in rows:
            row["provenance"].update(hashes_verified=True, protocol_frozen=True)
        result = summarize(rows, {}, bootstrap_replicates=100)
        roi = result["comparisons"]["relay"]["roi"]
        self.assertTrue(roi["point_criteria_met"])
        self.assertTrue(roi["confidence_criteria_met"])
        self.assertEqual(roi["adoption_status"], "not_assessed_simulation")
        self.assertFalse(roi["automatic_adoption"])
        json.dumps(result, allow_nan=False)

    def test_development_and_calibration_cannot_confirm_roi(self):
        for split in ("development", "calibration", "fit"):
            rows = paired(mode="external")
            for row in rows:
                row["provenance"].update(evaluation_split=split, hashes_verified=True)
            roi = summarize(rows, {}, bootstrap_replicates=100)["comparisons"]["relay"]["roi"]
            self.assertEqual(roi["adoption_status"], "not_assessed_development")
            self.assertFalse(roi["independent_validation_declared"])

    def test_missing_baseline_cost_or_latency_is_indeterminate(self):
        rows = paired(mode="external")
        without_baseline = [row for row in rows if row["policy"] == "relay"]
        roi = summarize(without_baseline, {}, bootstrap_replicates=100)["comparisons"]["relay"]["roi"]
        self.assertEqual(roi["adoption_status"], "indeterminate")

        self.assertIn("baseline_policy_missing", roi["missing_requirements"])
        rows[0]["events"] = annotate_costs([event(reported_llm_cost_usd=None, usage={})], {}, RATES)
        result = summarize(rows, {}, bootstrap_replicates=100)
        self.assertIsNone(result["policies"]["relay"]["costs"]["cost_per_task_usd"])
        self.assertEqual(result["policies"]["relay"]["coverage"]["unknown_cost_episodes"], 1)
        self.assertEqual(result["comparisons"]["relay"]["roi"]["adoption_status"], "indeterminate")
        rows = paired(mode="external")
        rows[0]["end_to_end_seconds"] = None
        roi = summarize(rows, {}, bootstrap_replicates=100)["comparisons"]["relay"]["roi"]
        self.assertEqual(roi["adoption_status"], "indeterminate")
        rows = paired(mode="external")
        rows[-1]["end_to_end_seconds"] = None
        roi = summarize(rows, {}, bootstrap_replicates=100)["comparisons"]["relay"]["roi"]
        self.assertEqual(roi["adoption_status"], "indeterminate")

    def test_interrupted_call_preserves_known_router_and_unknown_phases(self):
        rows = paired(mode="external")
        interrupted = event(generation_seconds=None, verification_seconds=None, tool_seconds=None, usage={})
        rows[0].update(success=False, termination="interrupted_not_retried", end_to_end_seconds=None,
                       events=annotate_costs([interrupted], {}, RATES))
        result = summarize(rows, {}, bootstrap_replicates=100)
        policy = result["policies"]["relay"]
        self.assertEqual(policy["coverage"]["unknown_cost_calls"], 1)
        self.assertEqual(policy["end_to_end_latency"]["unknown"], 1)
        self.assertTrue(policy["routing_latency"]["complete"])
        self.assertEqual(result["comparisons"]["relay"]["roi"]["adoption_status"], "indeterminate")

    def test_integrity_metadata_alone_does_not_establish_independence(self):
        rows = paired(mode="external")
        for row in rows:
            row["provenance"].pop("independent_validation")
            row["provenance"]["hashes_verified"] = True
        roi = summarize(rows, {}, bootstrap_replicates=100)["comparisons"]["relay"]["roi"]
        self.assertEqual(roi["adoption_status"], "not_assessed_independence")
        self.assertTrue(roi["point_criteria_met"])

    def test_unknown_retry_cost_is_not_dropped_and_zero_baseline_has_no_saving(self):
        row = episode("failed", "relay", success=False)
        row["events"].extend(annotate_costs([event(attempt=1, is_retry=True, usage={})], {}, RATES))
        report = summarize([row], {}, bootstrap_replicates=100)["policies"]["relay"]
        self.assertIsNone(report["costs"]["total_usd"])
        self.assertIsNone(report["costs"]["retry_llm_usd"])
        self.assertEqual(report["costs"]["retry_llm_unknown_calls"], 1)
        rows = [episode(str(i), name, cost=0., mode="external") for name in ["relay", "always_strong"] for i in range(2)]
        result = summarize(rows, {}, bootstrap_replicates=100)["comparisons"]["relay"]
        self.assertIsNone(result["paired"]["relative_total_cost_saving"])
        self.assertEqual(result["roi"]["adoption_status"], "indeterminate")

    def test_duplicate_call_and_inconsistent_saved_total_rejected(self):
        row = episode("one", "relay")
        row["events"].append(deepcopy(row["events"][0]))
        with self.assertRaisesRegex(ValueError, "double-count"):
            summarize([row], {}, bootstrap_replicates=100)
        row = episode("one", "relay")
        row["events"][0]["costs"]["total_usd"] = 999.
        with self.assertRaisesRegex(ValueError, "totals"):
            summarize([row], {}, bootstrap_replicates=100)

    def test_one_source_has_no_confidence_interval_and_empty_journal_is_unknown(self):
        rows = paired(mode="external", n=1)
        result = summarize(rows, {}, bootstrap_replicates=100)
        self.assertIsNone(result["comparisons"]["relay"]["paired"]["success_rate_delta_95ci"])
        self.assertEqual(result["comparisons"]["relay"]["roi"]["adoption_status"], "inconclusive")
        row = episode("empty", "relay", events=[])
        result = summarize([row], {}, bootstrap_replicates=100)
        self.assertIsNone(result["policies"]["relay"]["costs"]["total_usd"])
        self.assertFalse(result["policies"]["relay"]["routing_latency"]["complete"])

    def test_latency_cutoff_strict_and_full_cost_includes_all_phases(self):
        rows = paired(mode="external")
        for row in rows:
            if row["policy"] == "relay":
                row["events"] = annotate_costs([event(router_seconds=.2, reported_llm_cost_usd=1.)], {}, {name: 0 for name in RATES})
        result = summarize(rows, {}, bootstrap_replicates=100)
        roi = result["comparisons"]["relay"]["roi"]
        self.assertFalse(roi["point_criteria"]["router_p95"])
        self.assertEqual(roi["adoption_status"], "does_not_meet_gates")


if __name__ == "__main__":
    unittest.main()
