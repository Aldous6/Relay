"""Continuation scaffold tests with a stub runner: no SDK or network calls."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("relay_continue_test", ROOT / "tools/continue_adaptive_pilot.py")
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.suite = {"tasks": [dict(task_id="a"), dict(task_id="b")], "models": ["cheap", "strong"]}
        self.policy = {"artifact_id": "original-policy"}
        self.frozen = {"freeze_id": "original-freeze"}
        (self.output / "frozen_routes.json").write_text(json.dumps(self.frozen), encoding="utf-8")
        (self.output / "run_manifest.json").write_text("{}", encoding="utf-8")
        self.rows = [dict(task_id="a", model="cheap", accepted=False, error_type="PermissionError", http_attempts=0)]
        self.events = []
        self.addCleanup(patch.stopall)
        patch.object(continuation, "source_hashes", return_value={"wrapper": "fixed", "io_helper": "fixed"}).start()
        patch.object(continuation.original, "prepare_run", return_value=(self.suite, self.policy, self.frozen)).start()
        patch.object(continuation.original, "read_observations", side_effect=lambda *a: list(self.rows)).start()
        patch.object(continuation.original, "reconcile_http", side_effect=lambda *a: list(self.events)).start()

    def prepare(self):
        return continuation.prepare_amendment("unused-suite", "unused-policy", self.output)

    def test_writeahead_precedes_context_and_original_main_and_preserves_failure(self):
        order = []
        @contextmanager
        def context(output, callback):
            self.assertTrue((output / "execution_amendment.json").is_file())
            order.append("context")
            callback(dict(path="episodes/a-cheap/state/state.json", attempt=1, next_attempt=2,
                          max_attempts=5, errno=13, winerror=5, contents="secret-not-allowed"))
            yield
            order.append("restored")
        def original_main(args):
            self.assertTrue((self.output / "execution_amendment.json").is_file())
            order.append("main")
            self.assertEqual(self.rows[0]["error_type"], "PermissionError")
            self.assertIn("--credential-pipe", args)
            return 0
        arguments = ["--suite", "unused-suite", "--policy", "unused-policy", "--output", str(self.output),
                     "--credential-pipe", "memory-only-pipe"]
        with patch.object(continuation, "_retry_context", side_effect=context), patch.object(
            continuation.original, "main", side_effect=original_main
        ):
            self.assertEqual(continuation.main(arguments), 0)
        self.assertEqual(order, ["context", "main", "restored"])
        event = json.loads((self.output / "io_retry_events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(event["added_HTTP_requests"], 0)
        self.assertNotIn("contents", event)
        self.assertNotIn("secret-not-allowed", json.dumps(event))
        amendment = continuation._read(self.output / "execution_amendment.json")
        self.assertEqual(amendment["original_snapshot"]["pre_http_PermissionError_episode_ids"], ["a-cheap"])

    def test_repeated_continuation_keeps_first_snapshot_and_rejects_drift(self):
        first = self.prepare()
        self.rows.append(dict(task_id="a", model="strong", accepted=True, error_type=None, http_attempts=1))
        self.events.append(dict(attempt_id=1, model="strong"))
        self.assertEqual(self.prepare(), first)
        with patch.object(continuation, "source_hashes", return_value={"wrapper": "changed"}):
            with self.assertRaisesRegex(ValueError, "changed amendment"):
                self.prepare()
        self.rows[0]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "observations or dispatches changed"):
            self.prepare()

    def test_requires_recorded_io_failure_and_existing_frozen_acquisition(self):
        self.rows[0]["error_type"] = "RateLimitError"
        with self.assertRaisesRegex(ValueError, "pre-HTTP PermissionError"):
            self.prepare()
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, "existing frozen"):
                continuation.prepare_amendment("unused", "unused", empty)

    def test_logger_rejects_escaping_paths_and_invalid_retry_budget(self):
        amendment = self.prepare()
        callback = continuation.retry_logger(self.output, amendment)
        event = dict(path="../outside.json", attempt=1, next_attempt=2, max_attempts=5, errno=13, winerror=5)
        with self.assertRaisesRegex(ValueError, "outside"):
            callback(event)
        event.update(path="state.json", next_attempt=6)
        with self.assertRaisesRegex(ValueError, "counters differ"):
            callback(event)
        self.assertFalse((self.output / "io_retry_events.jsonl").exists())

    def test_dry_mode_records_amendment_without_loading_sdk(self):
        args = ["--suite", "unused-suite", "--policy", "unused-policy", "--output", str(self.output), "--freeze-only"]
        with patch.object(continuation.original, "main", return_value=0) as original_main, patch.object(
            continuation, "_retry_context", side_effect=AssertionError("SDK must not be loaded")
        ):
            self.assertEqual(continuation.main(args), 0)
        original_main.assert_called_once_with(args)
        self.assertTrue((self.output / "execution_amendment.json").is_file())


if __name__ == "__main__":
    unittest.main()
