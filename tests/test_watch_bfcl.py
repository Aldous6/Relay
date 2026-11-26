"""Fake parent closure and fake supervisor; no real processes or APIs."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
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

from routerbench.agent_evaluation import _seal, _write

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("watch_bfcl_tool", ROOT / "tools/watch_bfcl.py")
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)


class Clock:
    def __init__(self):
        self.now, self.sleeps = 1000., []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        assert 0 <= seconds <= 30
        self.now += seconds
        self.sleeps.append(seconds)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent, self.output = self.root / "active", self.root / "watch"
        self.parent.mkdir()
        (self.parent / "manifest.json").write_text('{"fixture":true}')
        (self.parent / "source_bundle.zip").write_bytes(b"offline-fixture")
        self.config = {"execution_mode": "external", "evaluation_split": "development", "independent_validation": False,
            "model_catalog": ["groq/openai/gpt-oss-20b"],
            "models_metadata": {"groq/openai/gpt-oss-20b": {"api_key_env": "GROQ_API_KEY"}},
            "spending": {"account_is_free_tier": True, "max_provider_spend_usd": 0},
            "session_stop_on_infrastructure": True,
            "session_limits": {"max_tokens_per_model": 160000, "max_provider_calls": 250}}
        self.manifest = {"configuration": self.config, "freeze_sha256": "fixture-freeze",
            "source_files_sha256": WATCH.supervisor._sources(), "cases": [{"id": "fixture"}],
            "policy_sequence": ["always_strong"]}
        self.snapshot = SimpleNamespace(hashes={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.parent.iterdir()})
        self.closed = {"manifest": self.manifest, "configuration": self.config, "pending": [["fixture", "always_strong"]]}

    def prepare(self, *, hours=72, execute=True):
        with patch.object(WATCH, "_read_parent", return_value=(deepcopy(self.manifest), self.snapshot)):
            return WATCH.prepare_watch(self.parent, self.output, max_hours=hours, execution_authorized=execute)

    def test_prepare_without_key_never_requests_generation_or_changes_parent(self):
        before = {p.name: p.read_bytes() for p in self.parent.iterdir()}
        with patch.object(WATCH.supervisor, "temporary_key", side_effect=AssertionError("No key in prepare")):
            plan = self.prepare(execute=False)
        self.assertEqual(plan["provider_calls"], 0)
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse((self.output / "autorun").exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.parent.iterdir()})

    def test_waits_for_parent_then_delegates_only_pending_with_original_absolute_deadline(self):
        plan = self.prepare()
        clock = Clock()
        absolute_deadline = clock.now + 72*3600
        def close_parent(seconds):
            clock.sleep(seconds)
            (self.parent / "summary.json").write_text("fixture-closed")
        def prepare(parent, output, **kwargs):
            self.assertTrue((self.parent / "summary.json").exists())
            self.assertEqual(str(parent), str(self.parent))
            clock.now += 7  # Preparation must not extend the global deadline.
            return {"ordered_pending_pairs": self.closed["pending"],
                "max_hours": kwargs["max_hours"], "tariffs": plan["tariffs"]}
        prepared = Mock(side_effect=prepare)
        def worker(output, *, clock, sleep):
            anchor = clock()
            remaining_hours = prepared.call_args.kwargs["max_hours"]
            self.assertAlmostEqual(anchor + remaining_hours*3600, absolute_deadline)
            self.assertGreater(clock(), anchor)
            return {"state": "complete", "report_paths": ["fixture-report"], "generation_calls": 2}
        delegated = Mock(side_effect=worker)
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
            result = WATCH.run_watch(self.output, clock=clock, sleep=close_parent,
                state_reader=lambda path: (self.closed, self.snapshot), prepare=prepared, worker=delegated)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(clock.sleeps, [30])
        self.assertEqual(result["delegated_generation_calls"], 2)
        prepared.assert_called_once()
        delegated.assert_called_once()

    def test_deadline_before_parent_closes_never_delegates(self):
        self.prepare(hours=.001)
        clock, prepare, worker = Clock(), Mock(), Mock()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
            result = WATCH.run_watch(self.output, clock=clock, sleep=clock.sleep, prepare=prepare, worker=worker)
        self.assertEqual(result["state"], "deadline_reached")
        self.assertAlmostEqual(clock.now, 1003.6)
        prepare.assert_not_called()
        worker.assert_not_called()

    def test_stop_while_waiting_cancels_without_api_or_supervisor(self):
        self.prepare()
        clock, worker = Clock(), Mock()
        def stop(seconds):
            clock.sleep(seconds)
            WATCH.request_stop(self.output)
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
            result = WATCH.run_watch(self.output, clock=clock, sleep=stop, worker=worker)
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(clock.sleeps, [30])
        worker.assert_not_called()

    def test_stop_is_forwarded_to_active_supervisor_without_pid_kill(self):
        self.prepare()
        forwarded = threading.Event()
        forward = Mock(side_effect=lambda path: forwarded.set())
        with WATCH.forward_stop(self.output, self.output / "autorun", forward=forward, poll_seconds=.01):
            WATCH.request_stop(self.output)
            self.assertTrue(forwarded.wait(1))
        forward.assert_called_once_with(self.output / "autorun")
        autorun = self.output / "autorun"
        autorun.mkdir()
        _write(autorun / "autorun.json", _seal({"kind": WATCH.supervisor.KIND}))
        WATCH.request_stop(self.output)
        self.assertTrue((autorun / "STOP").exists())

    def test_source_or_tariff_change_stops_before_delegation(self):
        plan = self.prepare()
        (self.parent / "source_bundle.zip").write_bytes(b"changed")
        worker = Mock()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
            result = WATCH.run_watch(self.output, worker=worker)
        self.assertEqual(result["state"], "needs_attention")
        self.assertEqual(result["stop_reason"]["reason"], "parent_contract_change")
        worker.assert_not_called()
        (self.parent / "source_bundle.zip").write_bytes(b"offline-fixture")
        Path(plan["tariffs"]["snapshot_path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "tariff_snapshot_change"):
            WATCH._guard(plan)

    def test_parent_lock_must_be_released_before_delegation(self):
        self.prepare(hours=.001)
        (self.parent / "summary.json").write_text("fixture-closed")
        clock, prepare = Clock(), Mock()
        with WATCH._single_writer(self.parent), patch.dict(os.environ, {"GROQ_API_KEY": "fixture-only"}), \
             redirect_stdout(io.StringIO()):
            result = WATCH.run_watch(self.output, clock=clock, sleep=clock.sleep, prepare=prepare)
        self.assertEqual(result["state"], "deadline_reached")
        prepare.assert_not_called()
        self.assertTrue(WATCH._parent_writer_released(self.parent))

    def test_hidden_process_uses_environment_only_and_worker_cleanup(self):
        self.prepare()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-secret"}), \
             patch.object(WATCH.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as process:
            WATCH.launch_background(self.output)
        self.assertNotIn("fixture-secret", " ".join(process.call_args.args[0]))
        self.assertEqual(process.call_args.kwargs["env"]["GROQ_API_KEY"], "fixture-secret")
        self.assertEqual(process.call_args.kwargs["stdin"], WATCH.subprocess.DEVNULL)
        self.assertEqual(process.call_args.kwargs["creationflags"], getattr(WATCH.subprocess, "CREATE_NO_WINDOW", 0))
        with patch.dict(os.environ, {"GROQ_API_KEY": "fixture-secret"}), \
             patch.object(WATCH, "run_watch", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                WATCH.main(["--worker", "--output", str(self.output)])
            self.assertNotIn("GROQ_API_KEY", os.environ)
        with patch.object(WATCH, "prepare_watch") as prepare:
            with self.assertRaises(ValueError):
                WATCH.main(["--run", "fixture", "--output", "fixture", "--launch-background"])
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
