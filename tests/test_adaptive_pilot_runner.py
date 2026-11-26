"""Offline acquisition-contract tests; synthetic responses prove no savings."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("relay_adaptive_runner_test", ROOT / "tools/run_adaptive_pilot.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.output = self.directory / "acquisition"
        self.task = dict(task_id="fresh-one", group_id="fresh-one", workflow="function-repair", split="validation",
                         instruction="Repair add", initial_source="def add(a,b): return a-b\n",
                         public_tests="assert add(2,3)==5\n", private_tests="assert add(1,1)==2\n")
        self.suite = {**runner.FIXED_LIMITS, "models": list(runner.MODELS), "baseline_model": runner.MODELS[1],
                      "tasks": [self.task], "evaluation_id": "synthetic-adaptive"}
        self.policy = dict(artifact_id="synthetic-policy", baseline_model=runner.MODELS[1], candidate_model=runner.MODELS[0])
        self.suite_path, self.policy_path = self.directory / "suite.json", self.directory / "policy.json"
        runner.write(self.suite_path, self.suite)
        runner.write(self.policy_path, self.policy)
        self.api = SimpleNamespace(
            validate_artifact=lambda policy: None,
            validate_fresh_prompts=lambda policy, rows: None,
            select_model=lambda policy, text: dict(model=runner.MODELS[0], score=.2, routing_seconds=.001),
            evaluate_policy=lambda policy, rows: {"artifact_id": policy["artifact_id"], "activation_eligible": False,
                                                   "observations": len(rows), "reason": "synthetic_test_only"},
        )
        self.addCleanup(patch.stopall)
        patch.object(runner, "_policy_api", return_value=self.api).start()
        patch.object(runner, "source_hashes", return_value={"synthetic-source": "fixed"}).start()
        patch.object(runner, "reference_contract", return_value=None).start()

    def freeze(self):
        return runner.prepare_run(self.suite_path, self.policy_path, self.output)

    def execute(self, *, fail_native=False, max_new_episodes=48):
        suite, policy, frozen = self.freeze()
        class NativeToolFailure(Exception):
            status_code = 400
        def actual_transport(**kwargs):
            self.assertTrue((self.output / "frozen_routes.json").exists())
            self.assertTrue((self.output / "episode_starts").is_dir())
            if fail_native and kwargs["model"] == runner.MODELS[0]:
                raise NativeToolFailure("tool_use_failed")
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
        module = SimpleNamespace(litellm_completion=actual_transport)
        def llm_class(**kwargs):
            self.assertTrue((self.output / "frozen_routes.json").exists())
            self.assertEqual(kwargs["num_retries"], 0)
            self.assertFalse(kwargs["capability_overrides"]["supports_prompt_cache_key"])
            return SimpleNamespace(**kwargs)
        def episode(task, llm, session, directory, **kwargs):
            directory.mkdir(parents=True)
            error = None
            try:
                module.litellm_completion(model=llm.model, messages=[], api_key=llm.api_key)
            except NativeToolFailure:
                error = "NativeToolFailure"
            row = session.finish(accepted=error is None, cost_complete=False)
            row["error_type"] = error
            return row
        result = runner.execute_cohort(suite, policy, frozen, self.output,
                                       key="synthetic-do-not-record-secret", llm_class=llm_class, llm_module=module,
                                       versions={"openhands-sdk": "synthetic", "litellm": "synthetic"},
                                       max_new_episodes=max_new_episodes, episode_runner=episode)
        return result

    def test_freezes_before_dispatch_and_accounts_both_branches_without_secret(self):
        self.assertEqual(self.execute(), 0)
        report = runner._read(self.output / "comparison.json")
        self.assertEqual(report["completed_episodes"], 2)
        self.assertEqual(report["http_attempts"], 2)
        self.assertEqual(report["unknown_http_costs"], 0)
        self.assertIsNone(report["net_monetary_savings"])
        self.assertAlmostEqual(report["acquisition_total_tariff_usd"], .0000675)
        frozen = runner._read(self.output / "frozen_routes.json")
        self.assertEqual(len(frozen["routes"]), 1)
        self.assertNotIn("prompt_text", frozen["routes"][0])
        records = runner._read(self.output / "validation_records.json")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["prompt_text"], runner.task_prompt(self.task))
        for path in self.output.rglob("*.json*"):
            self.assertNotIn("synthetic-do-not-record-secret", path.read_text(encoding="utf-8"))

    def test_native_tool_failure_preserves_unknown_cost_and_complete_cohort(self):
        self.assertEqual(self.execute(fail_native=True), 0)
        report = runner._read(self.output / "comparison.json")
        self.assertEqual(report["completed_episodes"], 2)
        self.assertEqual(report["unknown_http_costs"], 1)
        self.assertIsNone(report["acquisition_total_tariff_usd"])
        rows = runner.read_observations(self.suite, self.output)
        failed = next(row for row in rows if row["model"] == runner.MODELS[0])
        self.assertFalse(failed["accepted"])
        self.assertIsNone(failed["cost_usd"])

    def test_partial_resume_does_not_repeat_finished_branch(self):
        self.assertEqual(self.execute(max_new_episodes=1), 0)
        self.assertEqual(len(runner._rows(self.output / "http_attempts.jsonl")), 1)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(runner._rows(self.output / "http_attempts.jsonl")), 2)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(runner._rows(self.output / "http_attempts.jsonl")), 2)

    def test_resume_rejects_changed_policy_source_and_suite(self):
        self.freeze()
        changed = {**self.policy, "artifact_id": "changed-policy"}
        runner.write(self.policy_path, changed, exclusive=False)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.freeze()
        runner.write(self.policy_path, self.policy, exclusive=False)
        with patch.object(runner, "source_hashes", return_value={"synthetic-source": "changed"}):
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                self.freeze()
        with patch.object(runner, "reference_contract", return_value={"sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                self.freeze()
        self.suite["tasks"][0]["private_tests"] += "assert True\n"
        runner.write(self.suite_path, self.suite, exclusive=False)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.freeze()

    def test_interrupted_episode_or_dispatch_is_not_discarded(self):
        self.freeze()
        runner.write(self.output / "episode_starts" / "interrupted.json", {"episode_id": "interrupted"})
        with self.assertRaisesRegex(ValueError, "unresolved"):
            runner.read_observations(self.suite, self.output)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runner.write(output / "attempt_starts" / "1.json", {"attempt_id": 1})
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                runner.reconcile_http(self.suite, output, [])

    def test_unknown_usage_cannot_be_zeroed_and_missing_tasks_block_report(self):
        self.execute(max_new_episodes=1)
        suite, policy, frozen = self.freeze()
        with self.assertRaisesRegex(ValueError, "complete paired cohort"):
            runner.build_report(suite, policy, frozen, self.output)
        rows = runner._rows(self.output / "http_attempts.jsonl")
        rows[0].update(input_tokens=None, output_tokens=None, total_tokens=None, cost_usd=0)
        (self.output / "http_attempts.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot be assigned zero"):
            runner.reconcile_http(suite, self.output, runner.read_observations(suite, self.output))

    def test_budget_exhaustion_never_dispatches(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            transport = runner.AuditedTransport(lambda **kw: calls.append(kw),
                dict(models=list(runner.MODELS), max_output_tokens=1024,
                     max_total_http_attempts=0, max_observed_tokens_per_model=180000), Path(tmp))
            with self.assertRaisesRegex(ValueError, "cap reached"):
                transport(model=runner.MODELS[0], messages=[])
            self.assertEqual(calls, [])
            self.assertEqual(transport.events, [])

    def test_freshness_check_precedes_output_and_dispatch(self):
        def contaminated(policy, records):
            raise ValueError("development prompt reused")
        self.api.validate_fresh_prompts = contaminated
        with self.assertRaisesRegex(ValueError, "development prompt"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_token_totals_and_per_model_observed_budget_are_reconciled(self):
        self.execute()
        observations = runner.read_observations(self.suite, self.output)
        events = runner._rows(self.output / "http_attempts.jsonl")
        events[0]["total_tokens"] += 1
        (self.output / "http_attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "total_tokens does not equal"):
            runner.reconcile_http(self.suite, self.output, observations)
        events[0]["total_tokens"] -= 1
        (self.output / "http_attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
        smaller_budget = {**self.suite, "max_observed_tokens_per_model": 149}
        with self.assertRaisesRegex(ValueError, "observed token acquisition cap"):
            runner.reconcile_http(smaller_budget, self.output, observations)

    def test_full_request_admission_unknown_reservation_and_observed_overrun(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            suite = dict(models=list(runner.MODELS), max_output_tokens=1024,
                         max_total_http_attempts=4, max_observed_tokens_per_model=10000)
            transport = runner.BoundedAuditedTransport(lambda **kw: calls.append(kw), suite, Path(tmp))
            # A body above 7900 tokens must not be clipped for the daily guard.
            with self.assertRaisesRegex(ValueError, "reserve exhausted before dispatch"):
                transport(model=runner.MODELS[0], messages=[{"content": "x" * 8500}])
            self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as tmp:
            def unknown(**kwargs):
                raise ValueError("synthetic_tool_failure")
            suite["max_observed_tokens_per_model"] = 5000
            transport = runner.BoundedAuditedTransport(unknown, suite, Path(tmp))
            with self.assertRaisesRegex(ValueError, "synthetic_tool_failure"):
                transport(model=runner.MODELS[0], messages=[])
            event = runner._rows(Path(tmp) / "http_attempts.jsonl")[0]
            self.assertIsNone(event["total_tokens"])
            self.assertIsNone(event["cost_usd"])
            self.assertGreater(event["admission_reserved_tokens"], 3000)
            resumed = runner.BoundedAuditedTransport(unknown, suite, Path(tmp))
            with self.assertRaisesRegex(ValueError, "reserve exhausted before dispatch"):
                resumed(model=runner.MODELS[0], messages=[])
            self.assertEqual(len(resumed.events), 1)
        with tempfile.TemporaryDirectory() as tmp:
            suite["max_observed_tokens_per_model"] = 10000
            transport = runner.BoundedAuditedTransport(
                lambda **kw: SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10001, completion_tokens=0)),
                suite, Path(tmp))
            with self.assertRaisesRegex(ValueError, "attempt persisted"):
                transport(model=runner.MODELS[0], messages=[])
            event = runner._rows(Path(tmp) / "http_attempts.jsonl")[0]
            self.assertEqual(event["total_tokens"], 10001)
            self.assertIsNotNone(event["cost_usd"])

    def test_cached_bert_has_exact_suite_binding_and_does_not_hide_encoder_time(self):
        self.policy["representation"] = "routellm-bert"
        runner.write(self.policy_path, self.policy, exclusive=False)
        reference_path = self.directory / "bert-reference.json"
        reference = {"suite_sha256": runner.hashlib.sha256(self.suite_path.read_bytes()).hexdigest(),
                     "rows": [{"task_id": self.task["task_id"], "routing_seconds": 1.25}]}
        runner.write(reference_path, reference)
        def scorer(text):
            self.assertEqual(text, runner.task_prompt(self.task))
            return dict(score=.4, input_tokens=100, truncated_tokens=0)
        scorer.provenance = {"original_load_seconds": 5.}
        def factory(path, tasks):
            self.assertEqual(path, reference_path)
            self.assertEqual(tasks, [self.task])
            return scorer
        seen = []
        def select(policy, text, *, scorer):
            value = scorer(text)
            seen.append("select")
            return dict(model=runner.MODELS[0], score=value["score"], routing_seconds=.002,
                        input_tokens=value["input_tokens"], truncated_tokens=0)
        def evaluate(policy, records, *, scorer):
            self.assertEqual(scorer(records[0]["prompt_text"])["score"], .4)
            seen.append("evaluate")
            return {"activation_eligible": False, "reason": "synthetic_test_only"}
        self.api.select_model, self.api.evaluate_policy = select, evaluate
        with patch.object(runner, "BERT_REFERENCE_PATH", reference_path), patch.dict(
            runner.sys.modules, {"adaptive_scoring": SimpleNamespace(cached_bert_scorer=factory)}
        ):
            self.assertEqual(self.execute(), 0)
            report = runner._read(self.output / "comparison.json")
            self.assertAlmostEqual(report["routing_seconds"], 1.252)
            self.assertEqual(report["routing_encoder_seconds"], 1.25)
            self.assertEqual(report["routing_encoder_load_seconds"], 5.)
            self.assertIn("not end-to-end", report["routing_measurement"])
            self.assertIn("evaluate", seen)
            reference["suite_sha256"] = "different-suite"
            runner.write(reference_path, reference, exclusive=False)
            with self.assertRaisesRegex(ValueError, "different frozen suite"):
                self.freeze()


if __name__ == "__main__":
    unittest.main()
