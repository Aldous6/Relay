"""Synthetic software fixtures only; none of these numbers measure real savings."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from relay_router.workflow import Relay
from relay_router.workflow_cli import main, read_ledger, report_ledger
from relay_router.workflow_policy import validate_artifact


ROOT = Path(__file__).resolve().parents[1]


def fixture(split):
    return [{"schema_version": 1, "kind": "relay_task_observation", "workflow": "fixture-review",
             "task_id": f"{split}-{i}", "group_id": f"{split}-family-{i}", "split": split,
             "model": model, "accepted": True, "cost_usd": cost, "cost_complete": True,
             "cost_basis": "tariff_estimate", "evaluation_id": "synthetic-software-only-v1"}
            for i in range(3) for model, cost in (("fixture-baseline", 1.), ("fixture-small", .4))]


def invoke(*args):
    output, error = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        status = main(list(map(str, args)))
    return status, output.getvalue(), error.getvalue()


class WorkflowCLITests(unittest.TestCase):
    def test_repository_command_help_and_synthetic_policy_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            dev, val, proposal, policy = [folder / name for name in ("dev.json", "val.jsonl", "proposal.json", "policy.json")]
            dev.write_text(json.dumps(fixture("development")), encoding="utf-8")
            val.write_text("\n".join(json.dumps(row) for row in fixture("validation")), encoding="utf-8")
            base = [sys.executable, str(ROOT / "tools" / "relay_harness.py")]
            help_result = subprocess.run(base + ["--help"], capture_output=True, text=True, check=False)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("propose", help_result.stdout)
            for arguments in (["propose", str(dev), "--baseline", "fixture-baseline", "--min-groups", "3", "--output", str(proposal)],
                              ["validate", str(proposal), str(val), "--output", str(policy)]):
                result = subprocess.run(base + arguments, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
            artifact = validate_artifact(json.loads(policy.read_text(encoding="utf-8")))
            self.assertEqual(artifact["routes"]["fixture-review"], "fixture-small")
            self.assertTrue(artifact["ready"])

    def test_report_separates_cost_bases_unknowns_and_unfinished_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            relay = Relay("fixture-baseline", log_path=path)
            for cost, basis in ((1., "provider_reported"), (2., "tariff_estimate"), (None, "unknown")):
                task = relay.task("fixture-review")
                call = task.begin_call()
                task.record_call(call, cost_usd=cost, cost_basis=basis)
                task.finish(accepted=True if cost is not None else None, cost_complete=True)
            relay.task("fixture-review").begin_call()
            report = report_ledger(*read_ledger(path))
            self.assertEqual(report["observation_count"], 3)
            self.assertEqual(report["accepted_count"], 2)
            self.assertEqual(report["unknown_outcome_count"], 1)
            self.assertEqual(report["unknown_cost_count"], 1)
            self.assertEqual((report["unfinished_tasks"], report["unfinished_calls"]), (1, 1))
            self.assertEqual(report["costs_by_basis"]["provider_reported"]["total_cost_usd"], 1.)
            self.assertEqual(report["costs_by_basis"]["tariff_estimate"]["total_cost_usd"], 2.)
            self.assertIsNone(report["costs_by_basis"]["unknown"]["total_cost_usd"])
            self.assertIsNone(report["savings_usd"])
            self.assertNotIn("total_cost_usd", report)
            status, output, error = invoke("report", path)
            self.assertEqual(status, 0, error)
            self.assertEqual(json.loads(output)["unfinished_tasks"], 1)

    def test_pending_attempts_cannot_disappear_from_proposal(self):
        with tempfile.TemporaryDirectory() as temp:
            path, output = Path(temp) / "ledger.jsonl", Path(temp) / "proposal.json"
            rows = fixture("development") + [{"kind": "relay_task_started", "session_id": "unfinished"}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            status, _, error = invoke("propose", path, "--baseline", "fixture-baseline", "--output", output)
            self.assertEqual(status, 2)
            self.assertIn("sin finalizar", error)
            self.assertFalse(output.exists())

    def test_json_lists_can_omit_kind_but_events_are_not_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.json"
            rows = fixture("development")
            for row in rows:
                row.pop("kind")
            path.write_text(json.dumps(rows + [{"kind": "unrelated_event"}]), encoding="utf-8")
            events, observations = read_ledger(path)
            self.assertEqual(len(observations), 6)
            self.assertEqual(len(events), 7)
            self.assertEqual(report_ledger(events, observations)["observation_count"], 6)

    def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(self):
        for content in ('[{"cost_usd": NaN}]', '[{"cost_usd": Infinity}]', '[{"cost_usd": 1e999}]',
                        '[{"cost_usd": 0, "cost_usd": 1}]'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "invalid.json"
                path.write_text(content, encoding="utf-8")
                status, _, error = invoke("report", path)
                self.assertEqual(status, 2)
                self.assertIn("No se pudo completar", error)

    def test_report_never_imputes_unknown_cost_to_zero(self):
        rows = fixture("development")[:2]
        rows[0]["cost_usd"] = None
        rows[0]["cost_complete"] = False
        result = report_ledger(rows, rows)
        costs = result["costs_by_basis"]["tariff_estimate"]
        self.assertEqual(costs["known_cost_subtotal_usd"], .4)
        self.assertIsNone(costs["total_cost_usd"])
        self.assertEqual(costs["unknown_cost_count"], 1)
        rows[1]["cost_usd"] = None
        self.assertIsNone(report_ledger(rows, rows)["costs_by_basis"]["tariff_estimate"]["known_cost_subtotal_usd"])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path, output = Path(temp) / "observations.json", Path(temp) / "report.json"
            path.write_text(json.dumps(fixture("production")), encoding="utf-8")
            output.write_text("preserve existing artifact", encoding="utf-8")
            status, _, error = invoke("report", path, "--output", output)
            self.assertEqual(status, 2)
            self.assertIn("No se sobrescribió", error)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve existing artifact")

    def test_breakdown_keeps_workflows_models_and_cost_bases_separate(self):
        rows = []
        for workflow, model, accepted, cost, basis in (
                ("review", "fixture-small", True, 1., "tariff_estimate"),
                ("review", "fixture-small", False, 3., "tariff_estimate"),
                ("review", "fixture-small", True, 10., "provider_reported"),
                ("review", "fixture-baseline", True, 7., "tariff_estimate"),
                ("repair", "fixture-small", True, 2., "tariff_estimate"),
                ("repair", "fixture-small", True, 4., "tariff_estimate")):
            rows.append({**fixture("production")[0], "workflow": workflow, "model": model,
                         "task_id": f"fixture-{len(rows)}", "accepted": accepted,
                         "cost_usd": cost, "cost_basis": basis})
        report = report_ledger(rows, rows)
        breakdown = {(row["workflow"], row["model"]): row for row in report["by_workflow_model"]}
        self.assertEqual(len(breakdown), 3)
        small_review = breakdown[("review", "fixture-small")]
        self.assertEqual((small_review["accepted_count"], small_review["rejected_count"]), (2, 1))
        self.assertEqual(small_review["costs_by_basis"]["tariff_estimate"]["cost_per_accepted_task_usd"], 4.)
        self.assertEqual(small_review["costs_by_basis"]["provider_reported"]["cost_per_accepted_task_usd"], 10.)
        self.assertEqual(breakdown[("review", "fixture-baseline")]["costs_by_basis"]["tariff_estimate"]["cost_per_accepted_task_usd"], 7.)
        self.assertEqual(breakdown[("repair", "fixture-small")]["costs_by_basis"]["tariff_estimate"]["cost_per_accepted_task_usd"], 3.)
        self.assertEqual(report["metrics_scope"], "descriptive_observed_cohorts_not_causal")

    def test_cost_per_accepted_task_requires_complete_costs_outcomes_and_acceptance(self):
        for changed in ({"cost_usd": None, "cost_complete": False}, {"accepted": None},
                        {"accepted": False, "all_failed": True}):
            with self.subTest(changed=changed):
                rows = fixture("production")[::2]
                rows[0].update({key: value for key, value in changed.items() if key != "all_failed"})
                if changed.get("all_failed"):
                    for row in rows:
                        row["accepted"] = False
                report = report_ledger(rows, rows)
                for summary in (report, report["by_workflow_model"][0]):
                    self.assertIsNone(summary["costs_by_basis"]["tariff_estimate"]["cost_per_accepted_task_usd"])

    def test_invalid_validation_never_writes_a_policy_and_unknowns_keep_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            dev, val, proposal, policy = [folder / name for name in ("dev.json", "val.json", "proposal.json", "policy.json")]
            dev.write_text(json.dumps(fixture("development")), encoding="utf-8")
            self.assertEqual(invoke("propose", dev, "--baseline", "fixture-baseline", "--min-groups", 3, "--output", proposal)[0], 0)
            rows = fixture("validation")
            for row in rows:
                row["group_id"] = "development-family-0"
            val.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(invoke("validate", proposal, val, "--output", policy)[0], 2)
            self.assertFalse(policy.exists())
            rows = fixture("validation")
            rows[0]["accepted"] = None
            val.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(invoke("validate", proposal, val, "--output", policy)[0], 0)
            self.assertFalse(json.loads(policy.read_text(encoding="utf-8"))["ready"])


if __name__ == "__main__":
    unittest.main()
