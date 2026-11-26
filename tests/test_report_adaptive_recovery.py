"""Secondary reporting preserves initial failures; all execution is synthetic."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_adaptive_recovery_test", ROOT / "tools/report_adaptive_recovery.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class RecoveryReportTests(unittest.TestCase):
    def data(self):
        initial = dict(task_id="a", group_id="a", model="small", evaluation_id="original", accepted=False,
                       error_type="PermissionError", logical_calls=0, http_attempts=0, cost_usd=None,
                       cost_complete=False, cost_basis="unknown", wall_seconds=2., elapsed_seconds=2.,
                       verification_seconds=.2, transport_seconds=0., quota_wait_seconds=0.)
        after = dict(initial, evaluation_id="original-recovery", accepted=True, public_passed=True, private_passed=True,
                     error_type=None, logical_calls=1, http_attempts=1, cost_usd=.3, cost_complete=True,
                     cost_basis="tariff_estimate", wall_seconds=4., elapsed_seconds=4., transport_seconds=3.)
        evidence = dict(episode_id="a-small", task_id="a", group_id="a", model="small", original_logical_calls=0,
                        original_http_attempts=0, original_http_start_records=0, zero_dispatch_tariff_usd=0.)
        return initial, after, evidence

    def test_combines_one_retry_cost_and_both_attempt_times_without_mutation(self):
        initial, after, evidence = self.data()
        before = deepcopy(initial)
        combined, provenance = reporter.combine_episodes([initial], [after], [evidence])
        self.assertEqual(initial, before)
        self.assertEqual(combined[0]["cost_usd"], .3)
        self.assertEqual(combined[0]["wall_seconds"], 6.)
        self.assertTrue(combined[0]["accepted"])
        self.assertFalse(provenance[0]["initial_accepted"])
        self.assertIsNone(provenance[0]["initial_raw_cost_usd"])
        self.assertEqual(provenance[0]["initial_llm_tariff_verified_usd"], 0.)

    def test_unknown_actual_dispatch_cost_stays_unknown(self):
        initial, after, evidence = self.data()
        after.update(cost_usd=None, cost_complete=False, cost_basis="unknown", error_type="BadRequestError",
                     accepted=False, public_passed=False, private_passed=False)
        combined, _ = reporter.combine_episodes([initial], [after], [evidence])
        self.assertIsNone(combined[0]["cost_usd"])
        self.assertFalse(combined[0]["cost_complete"])
        with self.assertRaisesRegex(ValueError, "cannot be priced as zero"):
            reporter.combine_episodes([initial], [after], [evidence], [evidence])

    def test_every_eligible_case_required_and_initial_request_blocks_zero(self):
        initial, after, evidence = self.data()
        with self.assertRaisesRegex(ValueError, "every independently eligible"):
            reporter.combine_episodes([initial], [], [evidence])
        initial["http_attempts"] = 1
        with self.assertRaisesRegex(ValueError, "zero-dispatch evidence"):
            reporter.combine_episodes([initial], [after], [evidence])

    def test_full_synthetic_primary_recovery_and_secondary_validation(self):
        helper_spec = importlib.util.spec_from_file_location("primary_report_fixture", ROOT / "tests/test_report_adaptive_pilot.py")
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        fixture = helper.ReportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.fixture(initial_failure=True)
        output = fixture.run / "pre_dispatch_recovery"
        original_snapshot = {name: (fixture.run / name).read_bytes() for name in reporter.recovery.ORIGINAL_INPUTS}
        reference_contract = dict(path="reference.json", sha256=hashlib.sha256(fixture.reference_path.read_bytes()).hexdigest())
        with patch.object(reporter.primary.runner, "reference_contract", return_value=reference_contract):
            state = reporter.recovery.load_original(fixture.run, fixture.suite_path, fixture.policy_path)
            manifest = reporter.recovery.prepare_recovery(state, output)
            module = SimpleNamespace(litellm_completion=lambda **kw: SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)))
            def episode(task, llm, session, directory, **kwargs):
                directory.mkdir(parents=True)
                call_id = session.begin_call()
                module.litellm_completion(model=llm.model, messages=[])
                session.record_call(call_id)
                row = session.finish(accepted=True, cost_complete=False)
                row.update(public_passed=True, private_passed=True, error_type=None,
                           wall_seconds=.5, verification_seconds=.1)
                return row
            with redirect_stdout(io.StringIO()):
                reporter.recovery.execute_recovery(state, manifest, output, key="synthetic-not-a-credential",
                    llm_class=lambda **kw: SimpleNamespace(**kw), llm_module=module,
                    versions={"openhands-sdk": "synthetic", "litellm": "synthetic"}, episode_runner=episode)
            report, validation, records = reporter.make_report(fixture.run, fixture.suite_path, fixture.policy_path,
                                                              output, fixture.reference_path)
        self.assertEqual(report["infrastructure_recovery"]["eligible_count"], 1)
        self.assertFalse(report["infrastructure_recovery"]["primary_result_replaced"])
        self.assertEqual(report["policies"]["fixed_cheap"]["accepted_count"], 2)
        self.assertEqual(report["primary_result"]["policies"]["fixed_cheap"]["accepted_count"], 1)
        self.assertFalse(validation["activation_eligible"])
        self.assertEqual(report["infrastructure_recovery"]["episode_provenance"][0]["times"]["wall_seconds"]["combined"], 1.)
        self.assertEqual(len(records), 4)
        self.assertEqual(original_snapshot, {name: (fixture.run / name).read_bytes() for name in original_snapshot})
        paths = reporter.write_report(report, validation, records, output, fixture.directory / "secondary")
        self.assertIn("Análisis secundario", Path(paths["markdown"]).read_text(encoding="utf-8"))
        self.assertTrue(Path(paths["validation"]).is_file())
        with self.assertRaises(FileExistsError):
            reporter.write_report(report, validation, records, output, fixture.directory / "secondary")


if __name__ == "__main__":
    unittest.main()
