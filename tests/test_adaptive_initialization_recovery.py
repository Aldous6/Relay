"""Synthetic recovery accounting tests; no provider, SDK or model evaluation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("relay_recovery_test", ROOT / "tools/recover_adaptive_initialization.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.output = self.root / "primary", self.root / "separate_recovery"
        self.run.mkdir()
        self.task = dict(task_id="fresh-task", group_id="fresh-task", workflow="function-repair", split="validation",
                         instruction="Repair add", initial_source="def add(a,b): return a-b\n",
                         public_tests="assert add(2,3)==5\n", private_tests="assert add(1,1)==2\n")
        self.suite = {**recovery.original.FIXED_LIMITS, "models": list(recovery.original.MODELS),
                      "baseline_model": recovery.original.MODELS[1], "tasks": [self.task], "evaluation_id": "synthetic-primary"}
        self.original_row = {"task_id": self.task["task_id"], "group_id": self.task["group_id"],
                             "model": recovery.original.MODELS[0], "accepted": False,
                             "error_type": "PermissionError", "logical_calls": 0, "http_attempts": 0}
        self.original_events = [{"attempt_id": 1, "episode_id": "other-primary-strong", "model": recovery.original.MODELS[1],
                                 "started_at_utc": "2020-01-01T00:00:00+00:00", "total_tokens": 100,
                                 "input_tokens": 50, "output_tokens": 50, "cost_usd": .0000375}]
        self.state = {
            "run": self.run, "suite": self.suite, "policy": {"artifact_id": "policy-unchanged"},
            "frozen": {"freeze_id": "original-freeze"}, "observations": [self.original_row],
            "events": self.original_events, "amendment": {"amendment_id": "original-io-fix"},
            "eligible": recovery.eligible_primary_episodes([self.original_row], self.original_events, self.run),
            "original_input_sha256": {"observations.jsonl": "synthetic-primary-hash"},
            "suite_sha256": "synthetic-suite-hash", "policy_sha256": "synthetic-policy-hash",
        }
        self.addCleanup(patch.stopall)
        patch.object(recovery, "source_hashes", return_value={"recovery-script": "fixed"}).start()

    def prepare(self):
        return recovery.prepare_recovery(self.state, self.output)

    def test_eligibility_includes_all_models_and_never_attempted_llm_episodes(self):
        second = {**self.original_row, "model": recovery.original.MODELS[1], "accepted": True}
        native = {**self.original_row, "task_id": "called", "logical_calls": 1, "http_attempts": 1}
        logical_only = {**self.original_row, "task_id": "logical", "logical_calls": 1}
        wrong_error = {**self.original_row, "task_id": "timeout", "error_type": "TimeoutError"}
        rows = recovery.eligible_primary_episodes([self.original_row, second, native, logical_only, wrong_error], [], self.run)
        self.assertEqual({row["model"] for row in rows}, set(recovery.original.MODELS))
        self.assertEqual(len(rows), 2)
        original_id = recovery.original._episode_id(self.original_row["task_id"], self.original_row["model"])
        with self.assertRaisesRegex(ValueError, "recorded HTTP attempt"):
            recovery.eligible_primary_episodes([self.original_row], [{"episode_id": original_id}], self.run)
        recovery.original.write(self.run / "attempt_starts" / "1.json", {"episode_id": original_id})
        with self.assertRaisesRegex(ValueError, "recorded HTTP attempt"):
            recovery.eligible_primary_episodes([self.original_row], [], self.run)

    def test_manifest_is_writeahead_and_rejects_source_or_primary_drift(self):
        manifest = self.prepare()
        self.assertEqual(recovery.verify_recovery_manifest(manifest, self.state), manifest)
        self.assertEqual(manifest["eligible_episode_ids"], [self.state["eligible"][0]["episode_id"]])
        self.assertTrue(manifest["execution"]["no_original_observations_replaced"])
        with patch.object(recovery, "source_hashes", return_value={"recovery-script": "changed"}):
            with self.assertRaisesRegex(ValueError, "contract changed"):
                self.prepare()
        self.state["original_input_sha256"]["observations.jsonl"] = "changed-primary"
        with self.assertRaisesRegex(ValueError, "contract changed"):
            self.prepare()

    def test_one_new_episode_preserves_original_and_resume_never_repeats_it(self):
        manifest = self.prepare()
        original_snapshot = json.dumps(self.original_row, sort_keys=True)
        called = []
        def provider(**kwargs):
            self.assertTrue((self.output / "recovery_manifest.json").is_file())
            self.assertEqual(kwargs["api_base"], recovery.GROQ_BASE_URL)
            called.append(kwargs["model"])
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
        module = SimpleNamespace(litellm_completion=provider)
        def episode(task, llm, session, directory, **kwargs):
            directory.mkdir(parents=True)
            module.litellm_completion(model=llm.model, messages=[], api_key=llm.api_key)
            row = session.finish(accepted=True, cost_complete=False)
            row.update(error_type=None, wall_seconds=.5, verification_seconds=.1, public_passed=True, private_passed=True)
            return row
        for _ in range(2):
            result = recovery.execute_recovery(self.state, manifest, self.output, key="synthetic-secret-never-persisted",
                llm_class=lambda **kwargs: SimpleNamespace(**kwargs), llm_module=module,
                versions={"openhands-sdk": "synthetic", "litellm": "synthetic"}, episode_runner=episode)
            self.assertEqual(result, 0)
        self.assertEqual(len(called), 1)
        self.assertEqual(json.dumps(self.original_row, sort_keys=True), original_snapshot)
        rows, events, budget = recovery.read_recovery(self.state, manifest, self.output)
        self.assertEqual(rows[0]["original_http_attempts_verified"], 0)
        self.assertEqual(rows[0]["evaluation_id"], "synthetic-primary-recovery")
        self.assertEqual(len(events), 1)
        self.assertEqual(budget[recovery.original.MODELS[1]]["observed_tokens"], 100)
        for path in self.output.rglob("*.json*"):
            self.assertNotIn("synthetic-secret-never-persisted", path.read_text(encoding="utf-8"))

    def test_combined_daily_reserve_and_http_budget_block_dispatch(self):
        self.output.mkdir()
        calls = []
        suite = recovery.recovery_suite(self.state)
        suite["max_observed_tokens_per_model"] = 10000
        old = [dict(self.original_events[0], model=recovery.original.MODELS[0], total_tokens=8000)]
        transport = recovery.RecoveryTransport(lambda **kw: calls.append(kw), suite, self.output, old)
        with self.assertRaisesRegex(ValueError, "reserve exhausted before dispatch"):
            transport(model=recovery.original.MODELS[0], messages=[])
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(ValueError, "fixed Groq API endpoint"):
            transport(model=recovery.original.MODELS[0], messages=[], api_base="https://unexpected.example/")
        old = [dict(old[0], total_tokens=None, admission_reserved_tokens=8000)]
        transport = recovery.RecoveryTransport(lambda **kw: calls.append(kw), suite, self.output, old)
        with self.assertRaisesRegex(ValueError, "reserve exhausted before dispatch"):
            transport(model=recovery.original.MODELS[0], messages=[])
        suite["max_observed_tokens_per_model"] = 180000
        suite["max_total_http_attempts"] = 0
        transport = recovery.RecoveryTransport(lambda **kw: calls.append(kw), suite, self.output, [])
        with self.assertRaisesRegex(ValueError, "cap reached"):
            transport(model=recovery.original.MODELS[0], messages=[])
        self.assertEqual(calls, [])

    def test_recent_primary_attempts_are_retained_in_rate_window(self):
        self.output.mkdir()
        event = dict(self.original_events[0], started_at_utc=recovery.datetime.now(recovery.timezone.utc).isoformat())
        transport = recovery.RecoveryTransport(lambda **kw: None, recovery.recovery_suite(self.state), self.output, [event])
        self.assertEqual(len(transport.windows[event["model"]]), 1)
        self.assertEqual(transport.windows[event["model"]][0][1], event["total_tokens"])

    def test_no_eligible_episode_never_requests_credential(self):
        self.state["eligible"] = []
        with patch.object(recovery, "load_original", return_value=self.state), patch.object(
            recovery.original, "_credential", side_effect=AssertionError("No credential should be requested")
        ):
            result = recovery.main(["--run", str(self.run), "--output", str(self.output)])
        self.assertEqual(result, 0)
        status = recovery._read(self.output / "recovery_status.json")
        self.assertTrue(status["no_op"])
        self.assertTrue(status["complete"])
        self.assertEqual(status["recovery_http_attempts"], 0)

    def test_unresolved_recovery_is_blocked_instead_of_retried(self):
        manifest = self.prepare()
        episode_id = self.state["eligible"][0]["episode_id"]
        recovery.original.write(self.output / "episode_starts" / (episode_id + ".json"), {"episode_id": episode_id})
        with self.assertRaisesRegex(ValueError, "unresolved"):
            recovery.read_recovery(self.state, manifest, self.output)


if __name__ == "__main__":
    unittest.main()
