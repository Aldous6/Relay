"""Policy-leakage, accounting and statistical contracts; no provider calls."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.cascade_evaluation import (
    exact_harm_upper, zero_loss_required_groups, operational_cascade_route,
    replay_public_cascade, paired_comparison,
)


def record(task, model, accepted, public=True, cost=1.):
    return {"task_id": task, "group_id": task, "model": model, "accepted": accepted,
            "public_passed": public, "cost_usd": cost, "cost_complete": cost is not None,
            "cost_basis": "tariff_estimate" if cost is not None else "unknown",
            "http_attempts": 1, "logical_calls": 1}


class CascadeEvaluationTests(unittest.TestCase):
    def setUp(self):
        # A false pass causes a paired loss; another task's gain masks it in aggregate.
        self.rows = [record("loss", "cheap", False), record("loss", "strong", True, cost=3.),
                     record("gain", "cheap", True), record("gain", "strong", False, cost=3.),
                     record("retry", "cheap", False, public=False), record("retry", "strong", True, cost=3.)]

    def replay(self, rows=None, **kwargs):
        return replay_public_cascade(self.rows if rows is None else rows, candidate_model="cheap", baseline_model="strong",
                                     resamples=500, **kwargs)

    def test_operational_failure_escalates_and_all_failed_attempt_costs_count(self):
        report = self.replay()
        cascade = report["policies"]["public_cascade"]
        self.assertEqual(cascade["accepted"], 2)
        self.assertEqual(cascade["total_cost_usd"], 6.)
        self.assertEqual(cascade["cost_per_accepted_usd"], 3.)
        self.assertEqual(cascade["escalated_tasks"], 1)
        retries = [r for r in report["task_rows"]["public_cascade"] if r["task_id"] == "retry"][0]
        self.assertEqual(retries["attempted_models"], ["cheap", "strong"])
        self.assertEqual(retries["http_attempts"], 2)

    def test_equal_aggregate_quality_cannot_hide_paired_loss_or_verifier_false_pass(self):
        report = self.replay()
        comparison = report["comparisons_to_baseline"]["public_cascade"]
        self.assertEqual(comparison["acceptance_difference"], 0.)
        self.assertEqual(comparison["lost_task_ids"], ["loss"])
        self.assertEqual(comparison["gained_task_ids"], ["gain"])
        self.assertEqual(report["operational_verifier_diagnostics"]["cheap"]["false_pass_rate_given_pass"], .5)
        self.assertFalse(report["activation_eligible"])

    def test_hidden_labels_and_prices_cannot_change_the_route(self):
        changed = deepcopy(self.rows)
        for row in changed:
            row["accepted"] = not row["accepted"]
            row["private_passed"] = not row.get("private_passed", False)
            row["cost_usd"] *= 100 if row["model"] == "cheap" else .1
        before, after = self.replay(), self.replay(changed)
        for left, right in zip(before["task_rows"]["public_cascade"], after["task_rows"]["public_cascade"]):
            self.assertEqual(left["attempted_models"], right["attempted_models"])
        with self.assertRaises(TypeError):
            operational_cascade_route(candidate_passed=True, accepted=False)

    def test_unknown_failed_attempt_poison_cost_instead_of_zero_imputation(self):
        changed = deepcopy(self.rows)
        changed[4].update(cost_usd=None, cost_complete=False, cost_basis="unknown")
        report = self.replay(changed)
        cascade = report["policies"]["public_cascade"]
        self.assertFalse(cascade["cost_complete"])
        self.assertIsNone(cascade["total_cost_usd"])
        self.assertIsNone(report["comparisons_to_baseline"]["public_cascade"]["savings_per_accepted_task"])
        self.assertEqual(cascade["known_cost_subtotal_usd"], 5.)

    def test_bounded_failed_request_keeps_known_retries_and_sums_fallback(self):
        failed = record("a", "cheap", False, public=False, cost=None)
        failed.update(known_cost_subtotal_usd=.25, cost_upper_bound_usd=.75,
                      cost_bound_basis="tariff_estimate", http_attempts=2)
        report = self.replay([failed, record("a", "strong", True, cost=2.)])
        policy = report["policies"]["public_cascade"]
        self.assertEqual(policy["total_cost_bounds_usd"], [2.25, 2.75])
        self.assertEqual(policy["known_cost_subtotal_usd"], 2.25)
        self.assertEqual(report["task_rows"]["public_cascade"][0]["http_attempts"], 3)
        comparison = report["comparisons_to_baseline"]["public_cascade"]
        self.assertEqual(comparison["savings_per_accepted_bounds"], [-.375, -.125])
        self.assertEqual(comparison["bootstrap"]["savings_bounds_envelope_95"], [-.375, -.125])
        self.assertEqual(comparison["bootstrap"]["undefined_savings_resamples"], 500)
        self.assertIsNone(comparison["bootstrap"]["savings_interval_95"])
        self.assertIsNone(comparison["savings_per_accepted_task"])
        del failed["cost_upper_bound_usd"]
        failed["reservation_usd"] = 1.
        report = self.replay([failed, record("a", "strong", True, cost=2.)])
        self.assertEqual(report["policies"]["public_cascade"]["total_cost_bounds_usd"], [2.25, None])
        self.assertIsNone(report["comparisons_to_baseline"]["public_cascade"]["savings_per_accepted_bounds"])

    def test_incompatible_or_invalid_bounds_cannot_produce_saving(self):
        row = record("a", "cheap", True, cost=None)
        base = [record("a", "strong", True, cost=2.)]
        for upper in (float("nan"), -.1, .1, True):
            row.update(known_cost_subtotal_usd=.25, cost_upper_bound_usd=upper, cost_bound_basis="tariff_estimate")
            with self.subTest(upper=upper), self.assertRaises(ValueError):
                paired_comparison([row], base, resamples=100)
        row.update(cost_upper_bound_usd=.3, cost_bound_basis="provider_invoice")
        self.assertIsNone(paired_comparison([row], base, resamples=100)["savings_per_accepted_bounds"])

    def test_undefined_bound_draw_invalidates_envelope(self):
        rows = [record("a", "cheap", False, cost=None), record("b", "cheap", True)]
        rows[0].update(cost_upper_bound_usd=.5, cost_bound_basis="tariff_estimate")
        base = [record("a", "strong", True, cost=2.), record("b", "strong", True, cost=2.)]
        comparison = paired_comparison(rows, base, resamples=100)
        self.assertEqual(comparison["savings_per_accepted_bounds"], [.25, .5])
        self.assertIsNone(comparison["bootstrap"]["savings_bounds_envelope_95"])
        self.assertGreater(comparison["bootstrap"]["undefined_savings_bound_resamples"], 0)

    def test_incomplete_or_duplicated_cohort_rejected(self):
        variants = [self.rows[:-1], self.rows + [self.rows[0]], []]
        for rows in variants:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.replay(rows)
        changed = deepcopy(self.rows)
        changed[2]["group_id"] = changed[3]["group_id"] = "loss"
        with self.assertRaises(ValueError):
            self.replay(changed)

    def test_malformed_or_missing_cost_is_never_known(self):
        for value in (True, float("nan"), float("inf"), -1):
            changed = deepcopy(self.rows)
            changed[0]["cost_usd"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.replay(changed)
        changed = deepcopy(self.rows)
        changed[0]["cost_usd"] = None
        with self.assertRaises(ValueError):
            self.replay(changed)

    def test_exact_binomial_bound_and_sample_planning(self):
        self.assertAlmostEqual(exact_harm_upper(0, 24), 1 - .05 ** (1 / 24))
        self.assertAlmostEqual(exact_harm_upper(1, 1), 1.)
        self.assertAlmostEqual(exact_harm_upper(1, 2), math.sqrt(.95))
        self.assertEqual(zero_loss_required_groups(), 149)
        self.assertLessEqual(exact_harm_upper(0, 149), .02)
        self.assertGreater(exact_harm_upper(0, 148), .02)
        for k, n in ((-1, 4), (1, 0), (True, 4), (5, 4)):
            with self.assertRaises(ValueError):
                exact_harm_upper(k, n)

    def test_bootstrap_is_paired_reproducible_and_cannot_drop_undefined_ratios(self):
        report = self.replay()
        self.assertEqual(report, self.replay(list(reversed(self.rows))))
        own = [record("bad", "cheap", False)]
        base = [record("bad", "strong", True, cost=3.)]
        comparison = paired_comparison(own, base, resamples=100)
        self.assertIsNone(comparison["savings_per_accepted_task"])
        self.assertIsNone(comparison["bootstrap"]["savings_interval_95"])
        self.assertEqual(comparison["bootstrap"]["undefined_savings_resamples"], 100)

    def test_reference_must_cover_same_candidates_and_tasks(self):
        routes = {row["task_id"]: "strong" for row in self.rows}
        result = self.replay(reference_routes={"frozen_bert": routes})
        self.assertEqual(result["policies"]["frozen_bert"], result["policies"]["fixed_baseline"])
        self.assertTrue(result["comparisons_to_baseline"]["frozen_bert"]["bootstrap"]["acceptance_has_no_discordant_pairs"])
        with self.assertRaises(ValueError):
            self.replay(reference_routes={"frozen_bert": {"loss": "strong"}})
        with self.assertRaises(ValueError):
            self.replay(reference_routes={"frozen_bert": {task: "third" for task in routes}})

    def test_inputs_remain_unchanged_and_mixed_basis_is_not_saving(self):
        before = deepcopy(self.rows)
        self.replay()
        self.assertEqual(self.rows, before)
        for row in before:
            if row["model"] == "strong":
                row["cost_basis"] = "provider_invoice"
        report = self.replay(before)
        self.assertIsNone(report["comparisons_to_baseline"]["fixed_candidate"]["savings_per_accepted_task"])
        self.assertIsNone(report["policies"]["public_cascade"]["total_cost_usd"])

    def test_initial_guard_uses_only_original_public_check_and_counts_verification_time(self):
        checks = {"loss": {"status": "passed", "seconds": .2},
                  "gain": {"status": "failed", "seconds": .3},
                  "retry": {"status": "failed", "seconds": .4}}
        report = self.replay(initial_public_checks=checks)
        guarded = report["policies"]["initial_failure_guarded_cascade"]
        self.assertEqual(guarded["accepted"], 3)
        self.assertEqual(guarded["total_cost_usd"], 8.)
        loss = next(row for row in report["task_rows"]["initial_failure_guarded_cascade"] if row["task_id"] == "loss")
        self.assertEqual(loss["attempted_models"], ["strong"])
        self.assertTrue(loss["candidate_skipped_initial_tests_passed"])
        self.assertAlmostEqual(report["initial_public_check_audit"]["initial_check_total_seconds"], .9)
        self.assertFalse(report["scope"]["net_monetary_savings_verified"])
        self.assertIn("unchanged original source", report["scope"]["decision_input"])
        changed = deepcopy(self.rows)
        for row in changed:
            row["accepted"] = not row["accepted"]
        before_routes = [row["attempted_models"] for row in report["task_rows"]["initial_failure_guarded_cascade"]]
        after = self.replay(changed, initial_public_checks=checks)
        self.assertEqual(before_routes, [row["attempted_models"] for row in after["task_rows"]["initial_failure_guarded_cascade"]])

    def test_inconclusive_initial_check_cannot_remove_bad_cases_from_saving_estimate(self):
        checks = {task: {"status": "failed", "seconds": .2} for task in ("loss", "gain", "retry")}
        checks["loss"]["status"] = "inconclusive"
        report = self.replay(initial_public_checks=checks)
        self.assertFalse(report["initial_public_check_audit"]["complete_replay_available"])
        self.assertEqual(report["initial_public_check_audit"]["inconclusive_task_ids"], ["loss"])
        self.assertNotIn("initial_failure_guarded_cascade", report["policies"])
        self.assertEqual(report["policies"]["public_cascade"]["tasks"], 3)
        del checks["loss"]
        with self.assertRaises(ValueError):
            self.replay(initial_public_checks=checks)


if __name__ == "__main__":
    unittest.main()
