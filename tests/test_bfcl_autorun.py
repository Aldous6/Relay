"""Supervisor tests use fake time, fake sessions, and mocked child processes."""
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from routerbench.agent_evaluation import _seal, _sources, _write

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bfcl_autorun_tool", ROOT / "tools/bfcl_autorun.py")
AUTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTO)


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 0., []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        assert 0 < seconds <= 30
        self.sleeps.append(seconds)
        self.now += seconds


class AutorunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.previous = self.root / "previous"
        self.previous.mkdir()
        self.output = self.root / "autorun"
        self.pairs = [["fixture-1", "always_strong"], ["fixture-2", "always_strong"]]
        self.config = {"execution_mode": "external", "evaluation_split": "development", "independent_validation": False,
            "spending": {"account_is_free_tier": True, "max_provider_spend_usd": 0},
            "model_catalog": ["groq/openai/gpt-oss-20b"],
            "models_metadata": {"groq/openai/gpt-oss-20b": {"api_key_env": "GROQ_API_KEY"}},
            "session_stop_on_infrastructure": True,
            "session_limits": {"max_tokens_per_model": 160000, "max_provider_calls": 250}}
        self.state = {"configuration": self.config, "manifest": {"source_files_sha256": _sources()},
            "run_directories": [str(self.previous)], "pending": self.pairs,
            "summary": {"session_stop_reason": {"reason": "session_token_limit"}}}
        self.snapshot = SimpleNamespace(hashes={})

    def prepare(self, *, hours=72, execute=True):
        with patch.object(AUTO, "_state", return_value=(deepcopy(self.state), self.snapshot)):
            return AUTO.prepare_autorun(self.previous, self.output, max_hours=hours, execution_authorized=execute)

    def run_fake(self, states, *, hours=72, prepare=None, execute=None):
        self.prepare(hours=hours)
        clock, report = FakeClock(), Mock(side_effect=lambda state, output, **kwargs: str(output / "report.md"))
        reader = Mock(side_effect=[(deepcopy(state), self.snapshot) for state in states])
        prepare = prepare or Mock(side_effect=lambda previous, output, **kwargs: {
            "plan": {"ordered_pending_pairs": states[reader.call_count-1]["pending"]}})
        execute = execute or Mock(return_value={"generation_calls": 1})
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-not-real"}), redirect_stdout(io.StringIO()):
            result = AUTO.run_worker(self.output, clock=clock, sleep=clock.sleep, state_reader=reader,
                prepare=prepare, execute=execute, report=report)
        return result, clock, prepare, execute, report

    def test_two_daily_sessions_keep_exact_cohort_and_report_each_closure(self):
        first = deepcopy(self.state)
        first["pending"] = self.pairs[1:]
        final = deepcopy(self.state)
        final["pending"] = []
        result, clock, prepare, execute, report = self.run_fake([self.state, first, final])
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["sessions_closed"], 2)
        self.assertEqual(result["generation_calls"], 2)
        self.assertEqual(clock.now, 48 * 3600)
        self.assertEqual(report.call_count, 3)
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(len(result["report_paths"]), 3)
        self.assertEqual(json.loads((self.output / "status.json").read_text())["pending_pairs"], 0)

    def test_quota_wait_respects_longer_retry_after_and_never_sleeps_over_30_seconds(self):
        self.state["summary"]["session_stop_reason"] = {"reason": "shared_infrastructure_failure",
            "category": "rate_limit", "status_code": 429}
        _write(self.previous / "summary.json", _seal({"episodes": [{"events": [{
            "backend_failure": {"retry_after_seconds": 90000}}]}]}))
        final = deepcopy(self.state)
        final["pending"] = []
        result, clock, _, execute, _ = self.run_fake([self.state, final])
        self.assertEqual(result["state"], "complete")
        self.assertEqual(clock.now, 90000)
        self.assertLessEqual(max(clock.sleeps), 30)
        execute.assert_called_once()

    def test_deadline_ends_wait_without_consuming_another_episode(self):
        result, clock, prepare, execute, _ = self.run_fake([self.state], hours=1)
        self.assertEqual(result["state"], "deadline_reached")
        self.assertEqual(clock.now, 3600)
        self.assertEqual(result["pending_pairs"], 2)
        prepare.assert_not_called()
        execute.assert_not_called()

    def test_other_infrastructure_stops_for_attention_without_retry(self):
        for status in (400, 401, 500):
            with self.subTest(status=status):
                state = deepcopy(self.state)
                state["summary"]["session_stop_reason"] = {"reason": "shared_infrastructure_failure",
                    "category": "http", "status_code": status}
                self.assertEqual(AUTO.stop_action(state)[0], "needs_attention")
        self.state["summary"]["session_stop_reason"] = {"reason": "shared_infrastructure_failure",
            "category": "http", "status_code": 401}
        result, clock, prepare, execute, _ = self.run_fake([self.state])
        self.assertEqual(result["state"], "needs_attention")
        self.assertEqual(result["stop_reason"]["status_code"], 401)
        self.assertEqual(clock.sleeps, [])
        prepare.assert_not_called()
        execute.assert_not_called()

    def test_scope_or_sources_changes_stop_before_generation(self):
        changed = Mock(return_value={"plan": {"ordered_pending_pairs": [["NEW_CASE", "always_strong"]]}})
        result, _, _, execute, _ = self.run_fake([self.state], prepare=changed)
        self.assertEqual(result["state"], "needs_attention")
        execute.assert_not_called()
        plan = json.loads((self.output / "autorun.json").read_text())["payload"]
        with patch.object(AUTO, "_sources", return_value={}):
            with self.assertRaisesRegex(ValueError, "source_change"):
                AUTO._guard(plan)

    def test_prepare_is_secret_free_and_background_requires_execute(self):
        with patch.object(AUTO.getpass, "getpass", side_effect=AssertionError("No credentials in prepare")):
            plan = self.prepare(execute=False)
        self.assertFalse(plan["execution_authorized"])
        self.assertEqual(plan["provider_calls"], 0)
        with patch.object(AUTO, "prepare_autorun") as prepare:
            with self.assertRaises(ValueError):
                AUTO.main(["--previous", "fixture", "--output", "fixture", "--launch-background"])
            prepare.assert_not_called()
        with self.assertRaises(ValueError):
            AUTO.prepare_autorun(self.previous, self.root / "bad", max_hours=73)

    def test_background_launch_has_no_secret_arguments_and_uses_hidden_child(self):
        self.prepare()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-only-secret"}), \
             patch.object(AUTO.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)) as popen:
            result = AUTO.launch_background(self.output)
        self.assertEqual(result["pid"], 1234)
        command = popen.call_args.args[0]
        options = popen.call_args.kwargs
        self.assertNotIn("fixture-only-secret", " ".join(command))
        self.assertIn("--worker", command)
        self.assertEqual(options["env"]["GROQ_API_KEY"], "fixture-only-secret")
        self.assertEqual(options["stdin"], AUTO.subprocess.DEVNULL)
        self.assertEqual(options["creationflags"], getattr(AUTO.subprocess, "CREATE_NO_WINDOW", 0))
        self.assertNotIn("fixture-only-secret", (self.output / "launch.json").read_text())
        self.assertNotIn("fixture-only-secret", (self.output / "worker.log").read_text())

    def test_key_cleanup_on_failure_and_worker_refuses_reentry(self):
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(AUTO.getpass, "getpass", return_value="fixture-hidden"), \
             patch.object(AUTO.sys.stdin, "isatty", return_value=True):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(RuntimeError):
                with AUTO.temporary_key():
                    self.assertEqual(os.environ["GROQ_API_KEY"], "fixture-hidden")
                    raise RuntimeError("fixture")
            self.assertNotIn("GROQ_API_KEY", os.environ)

    def test_shared_study_mutex_blocks_another_output_and_releases(self):
        self.prepare()
        second = self.root / "second-output"
        second.mkdir()
        _write(second / "status.json", {"state": "prepared"})
        with AUTO._study_mutex(self.previous.resolve(), self.output):
            with self.assertRaises(RuntimeError):
                with AUTO._study_mutex(self.previous.resolve(), second):
                    self.fail("Same study cannot have two workers")
        self.assertEqual(json.loads((second / "status.json").read_text())["stop_reason"]["reason"], "study_locked")
        with AUTO._study_mutex(self.previous.resolve(), second):
            pass

    def test_persistent_claim_blocks_stale_second_plan_after_first_worker_completes(self):
        self.prepare()
        second = self.root / "second-worker"
        with patch.object(AUTO, "_state", return_value=(deepcopy(self.state), self.snapshot)):
            AUTO.prepare_autorun(self.previous, second, execution_authorized=True)
        final = deepcopy(self.state)
        final["pending"] = []
        first_clock = FakeClock()
        execute_first, execute_second = Mock(return_value={"generation_calls": 1}), Mock()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-hidden"}), redirect_stdout(io.StringIO()):
            result = AUTO.run_worker(self.output, clock=first_clock, sleep=first_clock.sleep,
                state_reader=Mock(side_effect=[(deepcopy(self.state), self.snapshot), (final, self.snapshot)]),
                prepare=lambda *args, **kwargs: {"plan": {"ordered_pending_pairs": self.pairs}},
                execute=execute_first, report=lambda *args, **kwargs: "fixture-report")
            self.assertEqual(result["state"], "complete")
            with self.assertRaisesRegex(RuntimeError, "supervisor registrado"):
                AUTO.run_worker(second, state_reader=lambda cursor: (deepcopy(self.state), self.snapshot),
                    execute=execute_second)
        execute_first.assert_called_once()
        execute_second.assert_not_called()
        self.assertEqual(json.loads((second / "status.json").read_text())["stop_reason"]["reason"], "study_already_claimed")
        self.assertTrue((self.previous / ".bfcl-autorun-claim.json").exists())

    def test_stop_during_wait_cancels_without_generation(self):
        self.prepare()
        clock = FakeClock()
        def stop_sleep(seconds):
            clock.sleep(seconds)
            AUTO.request_stop(self.output)
        execute = Mock()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-hidden"}), redirect_stdout(io.StringIO()):
            result = AUTO.run_worker(self.output, clock=clock, sleep=stop_sleep,
                state_reader=lambda cursor: (deepcopy(self.state), self.snapshot), execute=execute,
                report=lambda state, output, **kwargs: str(output / "report.md"))
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(clock.now, 30)
        self.assertFalse(result["stop_reason"]["automatic_reissue"])
        execute.assert_not_called()

    def test_active_watcher_requests_interrupt_with_mock_and_cancelled_run_keeps_pending(self):
        self.prepare()
        event = threading.Event()
        interrupt = Mock(side_effect=event.set)
        with AUTO.watch_stop(self.output, interrupt=interrupt, poll_seconds=.01):
            AUTO.request_stop(self.output)
            self.assertTrue(event.wait(1))
        interrupt.assert_called_once()
        (self.output / "STOP").unlink()
        clock = FakeClock()
        def interrupted(prepared):
            AUTO.request_stop(self.output)
            raise KeyboardInterrupt()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-hidden"}), \
             patch.object(AUTO, "watch_stop") as watch, redirect_stdout(io.StringIO()):
            watch.return_value.__enter__.return_value = None
            result = AUTO.run_worker(self.output, clock=clock, sleep=clock.sleep,
                state_reader=lambda cursor: (deepcopy(self.state), self.snapshot), execute=interrupted,
                prepare=lambda *args, **kwargs: {"plan": {"ordered_pending_pairs": self.pairs}},
                report=lambda *args, **kwargs: "fixture-report")
        self.assertEqual(result["state"], "cancelled")
        self.assertTrue(result["stop_reason"]["active_call_may_be_pending"])
        self.assertEqual(result["sessions_closed"], 0)
        self.assertFalse(result["stop_reason"]["automatic_reissue"])

    def test_loaded_hashes_and_tariffs_are_frozen_and_report_uses_snapshot(self):
        plan = self.prepare()
        self.assertIn("run_bfcl.py", plan["helper_sha256"])
        self.assertEqual(Path(plan["tariffs"]["snapshot_path"]).read_bytes(), AUTO.TARIFFS.read_bytes())
        with patch.object(AUTO, "LOADED_HELPER_HASHES", {}):
            with self.assertRaisesRegex(ValueError, "source_change"):
                AUTO._guard(plan)
        with patch.object(AUTO.report_bfcl_study, "main") as report:
            AUTO.write_report(self.state, self.root / "report", tariffs_path=plan["tariffs"]["snapshot_path"])
        argv = report.call_args.args[0]
        self.assertEqual(argv[argv.index("--tariffs")+1], plan["tariffs"]["snapshot_path"])
        Path(plan["tariffs"]["snapshot_path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tariff_snapshot_change"):
            AUTO._guard(plan)

    def test_stop_cli_never_requests_key_or_launches_process(self):
        self.prepare()
        with patch.object(AUTO.getpass, "getpass") as prompt, \
             patch.object(AUTO.subprocess, "Popen") as process, redirect_stdout(io.StringIO()):
            result = AUTO.main(["--stop", "--output", str(self.output)])
        self.assertEqual(result["state"], "stop_requested")
        prompt.assert_not_called()
        process.assert_not_called()
        _write(self.output / "status.json", {"state": "running_session"})
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-hidden"}):
            with self.assertRaisesRegex(ValueError, "No se reanuda"):
                AUTO.run_worker(self.output)
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-hidden"}), \
             patch.object(AUTO, "run_worker", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                AUTO.main(["--worker", "--output", str(self.output)])
            self.assertNotIn("GROQ_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
