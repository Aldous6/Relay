"""Reporting checks use synthetic local transports, never fresh model outputs."""
from datetime import datetime, timezone
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_adaptive_pilot_test", ROOT / "tools/report_adaptive_pilot.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def controls(self):
        tasks = [dict(task_id=str(index), group_id=str(index)) for index in range(2)]
        suite = dict(tasks=tasks, models=["cheap", "strong"], baseline_model="strong")
        rows = [dict(task_id=task["task_id"], group_id=task["group_id"], model=model,
                     accepted=model == "strong" or task["task_id"] == "0", cost_usd=cost,
                     cost_basis="tariff_estimate", cost_complete=True)
                for task in tasks for model, cost in (("cheap", .2), ("strong", 1.))]
        routes = [dict(task_id=task["task_id"], model="cheap" if task["task_id"] == "0" else "strong") for task in tasks]
        references = [dict(task_id="0", score=.55, truncated_tokens=0),
                      dict(task_id="1", score=.2, truncated_tokens=1)]
        return suite, rows, routes, references

    def test_paired_controls_use_complete_episode_costs_and_truncation(self):
        suite, rows, routes, references = self.controls()
        stats, comparisons = reporter.summarize_controls(suite, rows, routes, references)
        self.assertEqual(stats["fixed_cheap"]["accepted_count"], 1)
        self.assertEqual(stats["fixed_cheap"]["total_cost_usd"], .4)
        self.assertEqual(stats["fixed_cheap"]["cost_per_accepted_task_usd"], .4)
        self.assertEqual(stats["relay_adaptive"]["accepted_count"], 2)
        self.assertEqual(stats["relay_adaptive"]["total_cost_usd"], 1.2)
        self.assertEqual(stats["bert_0_50"]["selected_baseline_count"], 2)
        self.assertEqual(stats["bert_0_58"]["selected_baseline_count"], 1)
        self.assertTrue(comparisons["bert_0_58"]["same_choices"])
        self.assertEqual(stats["relay_adaptive"]["paired_losses_vs_strong"], 0)
        self.assertGreater(stats["relay_adaptive"]["paired_loss_upper_95"], 0)

    def test_unknown_cost_never_becomes_zero_or_a_break_even_price(self):
        suite, rows, routes, references = self.controls()
        rows[0].update(cost_usd=None, cost_complete=False, cost_basis="unknown")
        stats, comparisons = reporter.summarize_controls(suite, rows, routes, references)
        self.assertIsNone(stats["relay_adaptive"]["total_cost_usd"])
        self.assertEqual(stats["relay_adaptive"]["known_cost_subtotal_usd"], 1.)
        self.assertIsNone(comparisons["fixed_strong"]["relay_relative_savings_per_accepted"])
        self.assertIsNone(reporter.break_even_cpu_hourly(stats["fixed_strong"], stats["relay_adaptive"], 5))

    def test_break_even_is_a_conditional_price_and_incomplete_pairs_fail(self):
        suite, rows, routes, references = self.controls()
        stats, _ = reporter.summarize_controls(suite, rows, routes, references)
        self.assertAlmostEqual(reporter.break_even_cpu_hourly(stats["fixed_strong"], stats["relay_adaptive"], 5), 576.)
        self.assertIsNone(reporter.break_even_cpu_hourly(stats["fixed_strong"], stats["relay_adaptive"], 0))
        with self.assertRaises(ValueError):
            reporter.summarize_controls(suite, rows[:-1], routes, references)

    def fixture(self, initial_failure=False):
        runner = reporter.runner
        models = list(runner.MODELS)
        records = [dict(task_id="old" + str(index), group_id="old" + str(index), prompt_text="historical " + str(index),
                        model=model, accepted=True, cost_usd=.1, cost_basis="tariff_estimate")
                   for index in range(3) for model in models]
        policy = reporter.policy_api.train_policy(records, baseline_model=models[1], candidate_model=models[0])
        tasks = [dict(task_id="new" + str(index), group_id="new" + str(index), workflow="function-repair", split="validation",
                      instruction="Repair function " + str(index), initial_source="def add(a,b): return a-b\n",
                      public_tests="assert add(1,2)==3", private_tests="assert add(2,3)==5") for index in range(2)]
        suite = dict(runner.FIXED_LIMITS, models=models, baseline_model=models[1], tasks=tasks,
                     evaluation_id="synthetic-report", schema_version=1)
        self.suite_path, self.policy_path = self.directory / "suite.json", self.directory / "policy.json"
        self.reference_path, self.run = self.directory / "reference.json", self.directory / "run"
        runner.write(self.suite_path, suite)
        runner.write(self.policy_path, policy)
        from adaptive_scoring import CHECKPOINT, REVISION, FILES
        reference = dict(kind="frozen_historical_BERT_controls", at_utc=datetime.now(timezone.utc).isoformat(),
                         suite_sha256=hashlib.sha256(self.suite_path.read_bytes()).hexdigest(),
                         reference=dict(implementation="local_published_routellm_bert_scoring", checkpoint=CHECKPOINT,
                                        revision=REVISION, weights_sha256=FILES["model.safetensors"], max_input_tokens=512,
                                        source_supervision="GPT-4/Mixtral preferences", success_probability=False, load_seconds=1.),
                         thresholds=[.5, .58], truncation_policy="keep_baseline",
                         threshold_provenance="historical_public_v2_development_frozen_no_new_outcomes",
                         source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in reporter.REFERENCE_SOURCES},
                         rows=[dict(task_id=task["task_id"], prompt_sha256=hashlib.sha256(runner.task_prompt(task).encode()).hexdigest(),
                                    score=.55, input_tokens=300, truncated_tokens=0, routing_seconds=.2) for task in tasks])
        runner.write(self.reference_path, reference)
        reference_contract = dict(path="reference.json", sha256=hashlib.sha256(self.reference_path.read_bytes()).hexdigest())
        with patch.object(runner, "reference_contract", return_value=reference_contract):
            suite, policy, frozen = runner.prepare_run(self.suite_path, self.policy_path, self.run)
            module = SimpleNamespace(litellm_completion=lambda **kw: SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)))
            def episode(task, llm, session, directory, **kwargs):
                directory.mkdir(parents=True)
                if initial_failure and task["task_id"] == "new0" and llm.model == models[0]:
                    row = session.finish(accepted=False, cost_complete=False)
                    row.update(public_passed=False, private_passed=False, error_type="PermissionError",
                               wall_seconds=.5, verification_seconds=.1)
                    return row
                call_id = session.begin_call()
                module.litellm_completion(model=llm.model, messages=[])
                session.record_call(call_id)
                row = session.finish(accepted=True, cost_complete=False)
                row.update(public_passed=True, private_passed=True, error_type=None,
                           wall_seconds=.5, verification_seconds=.1)
                return row
            with redirect_stdout(io.StringIO()):
                runner.execute_cohort(suite, policy, frozen, self.run, key="test-only-not-a-credential",
                                      llm_class=lambda **kw: SimpleNamespace(**kw), llm_module=module,
                                      versions={"openhands-sdk": "synthetic", "litellm": "synthetic"}, episode_runner=episode)
                if initial_failure:
                    import continue_adaptive_pilot as continuation
                    continuation.prepare_amendment(self.suite_path, self.policy_path, self.run)
                    runner.execute_cohort(suite, policy, frozen, self.run, key="test-only-not-a-credential",
                                          llm_class=lambda **kw: SimpleNamespace(**kw), llm_module=module,
                                          versions={"openhands-sdk": "synthetic", "litellm": "synthetic"}, episode_runner=episode)
        return reporter.make_report(self.run, self.suite_path, self.policy_path, self.reference_path)

    def test_completed_audited_fixture_reproduces_metrics_and_writes_exclusively(self):
        result = self.fixture()
        self.assertEqual(result["fresh_tasks"], 2)
        self.assertEqual(result["acquisition"]["episodes"], 4)
        self.assertFalse(result["activation"]["eligible"])
        self.assertTrue(result["audit"]["validation_reproduced_without_tuning"])
        self.assertFalse(result["claim_limits"]["general_routellm_superiority_established"])
        self.assertIsNone(result["routing_compute"]["break_even_cpu_price_usd_per_hour_vs_strong"])
        paths = reporter.write_report(result, self.directory / "report")
        self.assertIn("no cumple", Path(paths["markdown"]).read_text(encoding="utf-8"))
        before = Path(paths["json"]).read_bytes()
        with self.assertRaises(FileExistsError):
            reporter.write_report(result, self.directory / "report")
        self.assertEqual(Path(paths["json"]).read_bytes(), before)

    def test_changed_reference_source_or_sealed_validation_fails(self):
        self.fixture()
        with patch.object(reporter.runner, "source_hashes", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "no longer match"):
                reporter.make_report(self.run, self.suite_path, self.policy_path, self.reference_path)
        original = self.reference_path.read_bytes()
        self.reference_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "BERT cache"):
            reporter.make_report(self.run, self.suite_path, self.policy_path, self.reference_path)
        self.reference_path.write_bytes(original)
        records_path = self.run / "validation_records.json"
        records_original = records_path.read_bytes()
        records_path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Saved validation records"):
            reporter.make_report(self.run, self.suite_path, self.policy_path, self.reference_path)
        records_path.write_bytes(records_original)
        path = self.run / "validation.json"
        value = json.loads(path.read_text())
        value["activation_eligible"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            reporter.make_report(self.run, self.suite_path, self.policy_path, self.reference_path)

    def test_execution_amendment_preserves_initial_failure_and_audits_io(self):
        import continue_adaptive_pilot as continuation
        run = self.directory / "amended"
        run.mkdir()
        frozen = dict(freeze_id="frozen", frozen_at_utc="2026-11-18T00:00:00+00:00")
        reporter.runner.write(run / "frozen_routes.json", frozen)
        observations = [dict(task_id="initial", model="small", error_type="PermissionError", http_attempts=0),
                        dict(task_id="later", model="large", error_type=None, http_attempts=1)]
        events = [dict(started_at_utc="2026-11-18T00:00:04+00:00")]
        for index, row in enumerate(observations):
            reporter.runner.write(run / "episode_starts" / (reporter.runner._episode_id(row["task_id"], row["model"]) + ".json"),
                                  dict(started_at_utc=f"2026-11-18T00:00:0{1 if index == 0 else 3}+00:00"))
        amendment = dict(schema_version=1, kind="bounded_filesystem_execution_amendment", freeze_id="frozen", artifact_id="policy",
                         frozen_routes_sha256=hashlib.sha256((run / "frozen_routes.json").read_bytes()).hexdigest(),
                         source_sha256=continuation.source_hashes(), reason=continuation.REASON,
                         write_retry=continuation._write_contract(), created_at_utc="2026-11-18T00:00:02+00:00",
                         original_snapshot=dict(completed_episodes=1, http_attempts=0,
                             completed_episode_ids=["initial-small"], error_episode_ids=["initial-small"],
                             pre_http_PermissionError_episode_ids=["initial-small"],
                             observations_prefix_sha256=reporter.runner._digest(observations[:1]), http_prefix_sha256=reporter.runner._digest([])))
        amendment["amendment_id"] = reporter.runner._digest(amendment)
        reporter.runner.write(run / "execution_amendment.json", amendment)
        io_event = dict(event_id=1, amendment_id=amendment["amendment_id"], at_utc="2026-11-18T00:00:03+00:00",
                        kind="local_atomic_write_retry", path="episodes/later-large/state/test.json", attempt=1,
                        next_attempt=2, max_attempts=5, errno=13, winerror=5, added_HTTP_requests=0)
        (run / "io_retry_events.jsonl").write_text(json.dumps(io_event) + "\n", encoding="utf-8")
        result = reporter.audit_execution_amendment(run, {"artifact_id": "policy"}, frozen, observations, events)
        self.assertEqual(result["original_completed_episodes_before_guard"], 1)
        self.assertEqual(result["original_pre_http_permission_errors_preserved"], ["initial-small"])
        self.assertEqual(result["io_retry_events"], 1)
        io_event["added_HTTP_requests"] = 1
        (run / "io_retry_events.jsonl").write_text(json.dumps(io_event) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IO retry metadata"):
            reporter.audit_execution_amendment(run, {"artifact_id": "policy"}, frozen, observations, events)


if __name__ == "__main__":
    unittest.main()
