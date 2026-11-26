"""Primary runtime outcomes cannot be replaced by a more favorable replay."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from report_cascade_validation import audit_http, summarize_actual


def observation(task, model, *, accepted=True, public=True, cost=.1, count=1):
    return {"task_id": task, "group_id": task, "model": model, "accepted": accepted,
            "public_passed": public, "cost_usd": cost, "cost_complete": cost is not None,
            "cost_basis": "tariff_estimate" if cost is not None else "unknown",
            "http_attempts": count, "logical_calls": count, "episode_id": task + "-" + model}


def runtime(task, model_rows, *, selected=True):
    total = sum(row["cost_usd"] or 0 for row in model_rows)
    complete = all(row["cost_usd"] is not None for row in model_rows)
    return {"task_id": task, "mode": "experimental", "accepted": None, "require_initial_failure": True,
            "attempts": [{"model": row["model"], "status": "completed", "verification": {
                "status": "passed" if row["public_passed"] else "failed"}} for row in model_rows],
            "selected_attempt": len(model_rows) if selected else None,
            "selected_model": model_rows[-1]["model"] if selected else None,
            "operational_passed": selected,
            "accounting": {"http_calls": sum(row["http_attempts"] for row in model_rows),
                           "calls": ["fixture"] * sum(row["http_attempts"] for row in model_rows),
                           "cost_complete": complete, "cost_usd": total if complete else None,
                           "known_cost_subtotal_usd": total},
            "initial_verification": {"status": "failed", "duration_seconds": .1},
            "wall_seconds": .5, "reason": "fixture", "route_reason": "fixture"}


class CascadeValidationReportTests(unittest.TestCase):
    def setUp(self):
        self.suite = {"models": ["cheap", "strong"], "baseline_model": "strong",
                      "tasks": [{"task_id": name, "group_id": name} for name in ("a", "b")]}
        self.rows = [observation("a", "cheap"), observation("a", "strong", cost=.3),
                     observation("b", "cheap", accepted=False, public=False), observation("b", "strong", cost=.3)]
        self.actual = {"a": runtime("a", [self.rows[0]]), "b": runtime("b", self.rows[2:])}

    def report(self, rows=None, actual=None):
        return summarize_actual(self.suite, self.rows if rows is None else rows,
                                self.actual if actual is None else actual, resamples=100)

    def test_actual_failed_branch_and_control_only_cost_are_separated(self):
        result = self.report()
        primary = result["policies"]["relay_cascade_actual"]
        self.assertEqual(primary["accepted"], 2)
        self.assertAlmostEqual(primary["total_cost_usd"], .5)
        self.assertAlmostEqual(result["policies"]["fixed_baseline"]["total_cost_usd"], .6)
        self.assertEqual(primary["escalated_tasks"], 1)
        self.assertFalse(result["activation_eligible"])
        self.assertIn("fewer_than_20_fresh_groups", result["pilot_criteria"]["reasons"])

    def test_actual_no_selection_remains_failure_even_if_replay_would_pass(self):
        changed = deepcopy(self.actual)
        changed["a"].update(selected_attempt=None, selected_model=None, operational_passed=False)
        changed["a"]["attempts"][0]["status"] = "error"
        result = self.report(actual=changed)
        self.assertEqual(result["policies"]["relay_cascade_actual"]["accepted"], 1)
        self.assertEqual(result["policies"]["naive_public_cascade_replay"]["accepted"], 2)
        self.assertEqual(result["comparisons_to_baseline"]["relay_cascade_actual"]["paired_losses"], 1)

    def test_missing_or_duplicate_branch_cannot_be_dropped(self):
        for variant in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                self.report(rows=variant)

    def test_runtime_cost_must_include_every_actual_attempt(self):
        changed = deepcopy(self.actual)
        changed["b"]["accounting"]["cost_usd"] = .3
        with self.assertRaises(ValueError):
            self.report(actual=changed)

    def test_private_label_cannot_be_inserted_in_runtime(self):
        changed = deepcopy(self.actual)
        changed["a"]["accepted"] = True
        with self.assertRaises(ValueError):
            self.report(actual=changed)

    def test_initial_inconclusive_no_dispatch_has_explicit_zero_and_failure(self):
        changed = deepcopy(self.actual)
        empty = runtime("a", [], selected=False)
        empty["initial_verification"]["status"] = "inconclusive"
        changed["a"] = empty
        result = self.report(actual=changed)
        row = next(row for row in result["task_rows"]["relay_cascade_actual"] if row["task_id"] == "a")
        self.assertFalse(row["accepted"])
        self.assertEqual(row["cost_usd"], 0.)
        self.assertEqual(row["http_attempts"], 0)

    def test_unknown_cost_stays_unknown(self):
        changed = deepcopy(self.rows)
        changed[0].update(cost_usd=None, cost_complete=False, cost_basis="unknown")
        actual = deepcopy(self.actual)
        actual["a"] = runtime("a", [changed[0]])
        report = self.report(rows=changed, actual=actual)
        self.assertIsNone(report["policies"]["relay_cascade_actual"]["total_cost_usd"])
        self.assertIsNone(report["comparisons_to_baseline"]["relay_cascade_actual"]["savings_per_accepted_task"])

    def test_recovery_is_counted_with_a_tariff_ceiling_and_every_observed_retry(self):
        changed = deepcopy(self.rows)
        failed = changed[2]
        failed.update(cost_usd=None, cost_complete=False, cost_basis="unknown",
                      http_attempts=2, known_cost_subtotal_usd=.1)
        actual = deepcopy(self.actual)
        result = runtime("b", changed[2:])
        actual["b"] = result
        result["attempts"][0]["status"] = "error"
        budget = result["accounting"]
        bound = {"source": "https://example.org/frozen-tariff", "max_input_tokens": 100,
                 "max_output_tokens": 20, "input_usd_per_token": .001, "output_usd_per_token": .005}
        budget.update(journal_complete=True, cost_upper_bound_usd=.6, known_cost_subtotal_usd=.4,
                      calls=[{"call_id": "1", "model": "cheap", "cost_usd": .1, "cost_upper_bound_usd": .1,
                              "cost_basis": "tariff_estimate"},
                             {"call_id": "2", "model": "cheap", "cost_usd": None, "cost_upper_bound_usd": .2,
                              "cost_bound_basis": "tariff_estimate", "tariff_bound": bound, "status": "error",
                              "http_status": 400, "provider_error_code": "tool_use_failed"},
                             {"call_id": "3", "model": "strong", "cost_usd": .3,
                              "cost_upper_bound_usd": .3, "cost_basis": "tariff_estimate"}])
        report = self.report(changed, actual)
        primary = report["policies"]["relay_cascade_actual"]
        self.assertEqual(primary["accepted"], 2)
        self.assertIsNone(primary["total_cost_usd"])
        self.assertAlmostEqual(primary["total_cost_bounds_usd"][0], .5)
        self.assertAlmostEqual(primary["total_cost_bounds_usd"][1], .7)
        self.assertAlmostEqual(report["policies"]["fixed_candidate"]["known_cost_subtotal_usd"], .2)
        self.assertIn("savings_per_accepted_not_estimable", report["pilot_criteria"]["reasons"])
        self.assertNotIn("less_than_10_percent_savings_per_accepted", report["pilot_criteria"]["reasons"])
        budget["calls"][1]["cost_upper_bound_usd"] = .01
        with self.assertRaisesRegex(ValueError, "token limits"):
            self.report(changed, actual)
        budget["calls"][1]["cost_upper_bound_usd"] = .2
        budget["calls"][1]["provider_error_code"] = "other"
        with self.assertRaisesRegex(ValueError, "documented tariff"):
            self.report(changed, actual)

    def test_selected_branch_must_have_operational_pass(self):
        actual = deepcopy(self.actual)
        actual["a"]["attempts"][0]["verification"]["status"] = "failed"
        with self.assertRaises(ValueError):
            self.report(actual=actual)

    def test_unobserved_wire_never_turns_empty_trace_into_free_attempt(self):
        rows = deepcopy(self.rows)
        rows[0] = observation("a", "cheap", accepted=False, public=False, cost=0., count=0)
        actual = deepcopy(self.actual)
        result = runtime("a", [rows[0]], selected=False)
        actual["a"] = result
        result["accounting"].update(cost_complete=False, cost_usd=None, cost_upper_bound_usd=None,
                                    http_calls_complete=False, journal_complete=True,
                                    unobserved_transport=[{"model": "cheap"}])
        report = self.report(rows, actual)
        for arm in ("relay_cascade_actual", "fixed_candidate"):
            cost = report["policies"][arm]
            self.assertIsNone(cost["total_cost_usd"])
            self.assertIsNone(cost["total_cost_bounds_usd"][1])
            self.assertFalse(report["task_rows"][arm][0]["http_attempts_complete"])


class HttpAuditTests(unittest.TestCase):
    def setUp(self):
        self.suite = {"models": ["cheap", "strong"]}
        self.tariffs = {"cheap": {"input_per_million_usd": 1., "output_per_million_usd": 2.},
                        "strong": {"input_per_million_usd": 3., "output_per_million_usd": 6.}}
        self.event = {"attempt_id": 1, "episode_id": "a-cheap", "model": "cheap", "status": "completed",
                      "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cost_usd": .00002}

    def test_reprices_usage_and_derives_known_subtotal(self):
        row = observation("a", "cheap", cost=.00002)
        result = audit_http(self.suite, [row], [self.event], self.tariffs)
        self.assertEqual(row["known_cost_subtotal_usd"], .00002)
        self.assertEqual(result["observed_tokens_by_model"]["cheap"], 15)

    def test_unknown_error_is_not_free(self):
        event = {**self.event, "status": "error", "input_tokens": None, "output_tokens": None,
                 "total_tokens": None, "cost_usd": None}
        row = observation("a", "cheap", cost=None)
        result = audit_http(self.suite, [row], [event], self.tariffs)
        self.assertEqual(result["unknown_cost_http_attempts"], 1)
        row.update(cost_usd=0., cost_complete=True, cost_basis="tariff_estimate")
        with self.assertRaises(ValueError):
            audit_http(self.suite, [row], [event], self.tariffs)

    def test_unrecorded_dispatch_or_incorrect_price_rejected(self):
        row = observation("a", "cheap", cost=.00002)
        for event in ({**self.event, "cost_usd": 0.}, {**self.event, "attempt_id": 2},
                      {**self.event, "episode_id": "missing"}):
            with self.assertRaises(ValueError):
                audit_http(self.suite, [row], [event], self.tariffs)

    def test_zero_dispatch_requires_empty_event_list(self):
        row = observation("a", "cheap", cost=0., count=0)
        result = audit_http(self.suite, [row], [], self.tariffs)
        self.assertEqual(result["http_attempts"], 0)
        with self.assertRaises(ValueError):
            audit_http(self.suite, [row], [self.event], self.tariffs)


if __name__ == "__main__":
    unittest.main()
