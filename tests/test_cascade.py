"""Offline checks of dispatch accounting and clean verifier-driven escalation."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.cascade import (
    AccountingIncomplete, AttemptOutcome, BudgetExceeded, SpendBudget, run_cascade,
)
from relay_router.cascade_verifier import CheckCommand, CommandVerifier


class SpendBudgetTests(unittest.TestCase):
    def test_counts_failed_requests_and_blocks_unpriced_retries(self):
        budget = SpendBudget(.02, 4)
        first = budget.begin_call("cheap", .01)
        budget.finish_call(first, status="error")
        with self.assertRaises(AccountingIncomplete):
            budget.begin_call("strong", .01)
        result = budget.snapshot()
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(result["committed_usd"], .01)
        self.assertEqual(result["http_calls"], 1)

    def test_reserves_before_transport_and_releases_known_unused_balance(self):
        budget = SpendBudget(.02, 2)
        first = budget.begin_call("cheap", .015)
        with self.assertRaises(AccountingIncomplete):
            budget.begin_call("cheap", .001)
        budget.finish_call(first, cost_usd=.005)
        second = budget.begin_call("strong", .015)
        budget.finish_call(second, cost_usd=.01)
        self.assertAlmostEqual(budget.snapshot()["cost_usd"], .015)
        with self.assertRaises(BudgetExceeded):
            budget.begin_call("cheap", .001)

    def test_reservation_overrun_is_not_hidden_and_stops_next_call(self):
        budget = SpendBudget(.02, 4)
        first = budget.begin_call("cheap", .001)
        budget.finish_call(first, cost_usd=.025)
        self.assertEqual(budget.snapshot()["cost_usd"], .025)
        self.assertTrue(budget.snapshot()["reservation_exceeded"])
        with self.assertRaises(BudgetExceeded):
            budget.begin_call("strong", .01)

    def test_invalid_cost_cannot_clear_pending_request(self):
        budget = SpendBudget(.02, 4)
        first = budget.begin_call("cheap", .001)
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                budget.finish_call(first, cost_usd=value)
        self.assertIsNone(budget.snapshot()["cost_usd"])
        with self.assertRaises(AccountingIncomplete):
            budget.begin_call("strong", .01)

    def test_rejected_reservation_does_not_consume_call_or_cost(self):
        budget = SpendBudget(.01, 2)
        with self.assertRaises(BudgetExceeded):
            budget.begin_call("strong", .02)
        self.assertEqual(budget.snapshot()["http_calls"], 0)
        self.assertEqual(budget.snapshot()["cost_usd"], 0)

    def test_mixed_price_bases_stop_future_dispatch(self):
        budget = SpendBudget(.10, 4)
        for basis in ("tariff_estimate", "provider_reported"):
            call = budget.begin_call("cheap", .01)
            budget.finish_call(call, cost_usd=.001, cost_basis=basis)
        self.assertFalse(budget.snapshot()["cost_complete"])
        with self.assertRaises(AccountingIncomplete):
            budget.begin_call("cheap", .01)

    def test_failed_journal_blocks_fallback_and_keeps_known_subtotal(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = SpendBudget(.10, 4)
            budget._bind_journal(Path(directory) / "calls.jsonl")
            call = budget.begin_call("cheap", .01)
            with patch.object(Path, "open", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    budget.finish_call(call, cost_usd=.001)
            self.assertEqual(budget.snapshot()["known_cost_subtotal_usd"], .001)
            self.assertFalse(budget.snapshot()["journal_complete"])
            self.assertIsNone(budget.snapshot()["cost_usd"])
            with self.assertRaises(AccountingIncomplete):
                budget.begin_call("strong", .01)

    def test_deadline_prevents_dispatch_without_claiming_inflight_cancellation(self):
        budget = SpendBudget(.10, 4, max_seconds=1)
        with patch("relay_router.cascade.time.monotonic", return_value=budget._started + 2):
            with self.assertRaises(BudgetExceeded):
                budget.begin_call("cheap", .01)
        self.assertEqual(budget.snapshot()["http_calls"], 0)


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        (self.source / "solution.py").write_text("value = 0\n", encoding="utf-8")
        (self.source / "check.py").write_text(
            "from pathlib import Path\nassert 'value = 42' in Path('solution.py').read_text()\n",
            encoding="utf-8")
        self.verifier = CommandVerifier.capture(
            self.source, checks=[CheckCommand("tests", (sys.executable, "-I", "-B", "check.py"))],
            protected_files=["check.py"])
        self.seen = []

    def run_task(self, callback, **kwargs):
        settings = dict(snapshot_dir=self.source, output_dir=self.root / "run", task_id="case",
                        candidate_model="cheap", baseline_model="strong", run_attempt=callback,
                        verifier=self.verifier, budget=SpendBudget(.03, 6),
                        mode="experimental", allow_experimental=True)
        settings.update(kwargs)
        return run_cascade(**settings)

    def generation(self, context, good=False):
        self.seen.append(context.model)
        self.assertEqual((context.workspace / "solution.py").read_text(), "value = 0\n")
        call = context.budget.begin_call(context.model, .01)
        (context.workspace / "solution.py").write_text("value = 42\n" if good else "value = -1\n")
        context.budget.finish_call(call, cost_usd=.004, input_tokens=10, output_tokens=5)
        return AttemptOutcome()

    def test_failure_escalates_from_pristine_copy_and_sums_both_costs(self):
        result = self.run_task(lambda ctx: self.generation(ctx, ctx.model == "strong"))
        self.assertEqual(self.seen, ["cheap", "strong"])
        self.assertTrue(result["escalated"])
        self.assertEqual(result["selected_model"], "strong")
        self.assertEqual(result["accounting"]["cost_usd"], .008)
        self.assertIsNone(result["accepted"])
        self.assertIsNone(result["savings_usd"])
        self.assertTrue(result["original_snapshot_unchanged"])
        self.assertEqual((self.source / "solution.py").read_text(), "value = 0\n")
        journal = (self.root / "run/transport.jsonl").read_text()
        self.assertEqual(len(journal.splitlines()), 4)

    def test_public_success_stops_but_never_claims_final_acceptance(self):
        result = self.run_task(lambda ctx: self.generation(ctx, True))
        self.assertEqual(self.seen, ["cheap"])
        self.assertTrue(result["operational_passed"])
        self.assertIsNone(result["accepted"])
        self.assertEqual(result["validation_status"], "not_independently_validated")

    def test_tests_must_reproduce_defect_before_cheap_is_admitted(self):
        (self.source / "check.py").write_text("assert True\n")
        self.verifier = CommandVerifier.capture(
            self.source, checks=[CheckCommand("weak", (sys.executable, "-I", "-B", "check.py"))],
            protected_files=["check.py"])
        result = self.run_task(lambda ctx: self.generation(ctx, True))
        self.assertEqual(self.seen, ["strong"])
        self.assertEqual(result["route_reason"], "baseline_checks_do_not_reproduce_defect")
        self.assertFalse(result["escalated"])

    def test_inconclusive_initial_checks_stop_all_calls(self):
        (self.source / "check.py").write_text("modified after capture\n")
        result = self.run_task(lambda ctx: self.generation(ctx, True))
        self.assertEqual(self.seen, [])
        self.assertEqual(result["accounting"]["http_calls"], 0)
        self.assertEqual(result["reason"], "initial_verification_inconclusive")

    def test_mutated_input_is_recorded_without_losing_result(self):
        def execute(ctx):
            result = self.generation(ctx, False)
            (self.source / "solution.py").unlink()
            return result
        result = self.run_task(execute)
        self.assertEqual(result["reason"], "snapshot_changed")
        self.assertFalse(result["original_snapshot_unchanged"])
        self.assertTrue((self.root / "run/result.json").is_file())

    def test_default_baseline_and_explicit_experimental_opt_in(self):
        result = self.run_task(lambda ctx: self.generation(ctx, True), mode="baseline", allow_experimental=False)
        self.assertEqual(self.seen, ["strong"])
        self.assertFalse(result["escalated"])
        with self.assertRaises(ValueError):
            self.run_task(lambda ctx: self.generation(ctx, True), allow_experimental=False)

    def test_test_tampering_does_not_pass_and_fallback_has_original_tests(self):
        original = (self.source / "check.py").read_text()
        def execute(ctx):
            outcome = self.generation(ctx, ctx.model == "strong")
            if ctx.model == "cheap":
                (ctx.workspace / "check.py").write_text("pass\n")
            else:
                self.assertEqual((ctx.workspace / "check.py").read_text(), original)
            return outcome
        result = self.run_task(execute)
        self.assertEqual(result["attempts"][0]["verification"]["status"], "inconclusive")
        self.assertEqual(result["selected_model"], "strong")

    def test_missing_cost_stops_fallback_and_preserves_failure(self):
        def execute(ctx):
            call = ctx.budget.begin_call(ctx.model, .01)
            ctx.budget.finish_call(call, status="error")
            raise RuntimeError("private provider message")
        result = self.run_task(execute)
        self.assertFalse(result["escalated"])
        self.assertEqual(result["reason"], "AccountingIncomplete")
        self.assertIsNone(result["accounting"]["cost_usd"])
        self.assertNotIn("private provider message", json.dumps(result))

    def test_incomplete_pending_request_blocks_fallback(self):
        def execute(ctx):
            ctx.budget.begin_call(ctx.model, .01)
            raise RuntimeError("interrupted")
        result = self.run_task(execute)
        self.assertEqual(result["reason"], "AccountingIncomplete")
        self.assertEqual(result["accounting"]["calls"][0]["status"], "pending")

    def test_total_budget_rejects_fallback_before_second_transport(self):
        result = self.run_task(lambda ctx: self.generation(ctx, False), budget=SpendBudget(.011, 6))
        self.assertEqual(result["accounting"]["http_calls"], 1)
        self.assertEqual(result["attempts"][1]["error_type"], "BudgetExceeded")
        self.assertFalse(result["operational_passed"])

    def test_compute_is_explicit_wall_time_scenario_and_no_free_default(self):
        result = self.run_task(lambda ctx: self.generation(ctx, True), cpu_hourly_usd=.50)
        self.assertGreater(result["compute_cost_scenario_usd"], 0)
        self.assertAlmostEqual(result["total_cost_scenario_usd"],
                               .004 + result["compute_cost_scenario_usd"])

    def test_output_must_not_overlap_snapshot_or_overwrite_prior_run(self):
        with self.assertRaises(ValueError):
            self.run_task(lambda ctx: AttemptOutcome(), output_dir=self.source / "output")
        self.run_task(lambda ctx: self.generation(ctx, True))
        with self.assertRaises(FileExistsError):
            self.run_task(lambda ctx: AttemptOutcome())


if __name__ == "__main__":
    unittest.main()
