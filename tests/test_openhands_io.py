"""Local I/O injection only: no provider calls or generation outcomes."""
import importlib.util
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

from relay_router.integrations.openhands_io import retry_openhands_writes

AVAILABLE = importlib.util.find_spec("openhands") is not None
if AVAILABLE:
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network disabled in I/O tests")):
        import openhands.sdk.io.local as sdk_io


@unittest.skipUnless(AVAILABLE, "Optional OpenHands SDK is required")
class OpenHandsIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "base_state.json"
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network disabled in I/O tests"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_permission_retries_same_write_and_callback_has_only_metadata(self):
        attempts, events = [], []
        original = sdk_io.atomic_write_text
        def transient(path, value, mode=0o600):
            attempts.append((path, value, mode))
            if len(attempts) < 3:
                raise PermissionError(13, "synthetic error containing PRIVATE_PAYLOAD")
            return original(path, value, mode)
        with patch.object(sdk_io, "atomic_write_text", side_effect=transient) as baseline:
            with retry_openhands_writes(self.root, max_attempts=3, base_delay=0, on_retry=events.append):
                sdk_io.atomic_write_text(self.path, "PRIVATE_PAYLOAD", 0o600)
            self.assertIs(sdk_io.atomic_write_text, baseline)
        self.assertEqual(self.path.read_text(), "PRIVATE_PAYLOAD")
        self.assertEqual(attempts, [(self.path, "PRIVATE_PAYLOAD", 0o600)] * 3)
        self.assertEqual([event["attempt"] for event in events], [1, 2])
        self.assertEqual(events[0]["path"], "base_state.json")
        self.assertEqual(set(events[0]), {"path", "attempt", "next_attempt", "max_attempts", "errno", "winerror"})
        self.assertNotIn("PRIVATE_PAYLOAD", str(events))

    def test_outside_resolved_root_and_other_errors_are_not_retried(self):
        scope = self.root / "scoped"
        scope.mkdir()
        for destination, exception in ((self.path, PermissionError(13, "outside")),
                                       (scope / "file.json", OSError(28, "full"))):
            with self.subTest(path=destination, error=type(exception).__name__):
                with patch.object(sdk_io, "atomic_write_text", side_effect=exception) as write:
                    with retry_openhands_writes(scope, max_attempts=5, base_delay=0), self.assertRaises(type(exception)):
                        sdk_io.atomic_write_text(destination, "data")
                    self.assertEqual(write.call_count, 1)
                    self.assertIs(sdk_io.atomic_write_text, write)

    def test_exhaustion_preserves_exception_and_always_restores_original(self):
        error = PermissionError(13, "persistent")
        with patch.object(sdk_io, "atomic_write_text", side_effect=error) as baseline:
            with self.assertRaises(PermissionError) as caught:
                with retry_openhands_writes(self.root, max_attempts=3, base_delay=0):
                    sdk_io.atomic_write_text(self.path, "data")
            self.assertIs(caught.exception, error)
            self.assertEqual(baseline.call_count, 3)
            self.assertIs(sdk_io.atomic_write_text, baseline)
        original = sdk_io.atomic_write_text
        with self.assertRaises(RuntimeError):
            with retry_openhands_writes(self.root):
                raise RuntimeError("caller aborted")
        self.assertIs(sdk_io.atomic_write_text, original)

    def test_nested_and_concurrent_contexts_are_rejected_without_restoration_race(self):
        original = sdk_io.atomic_write_text
        failures = []
        with retry_openhands_writes(self.root):
            active = sdk_io.atomic_write_text
            with self.assertRaises(RuntimeError):
                with retry_openhands_writes(self.root):
                    self.fail("nested context entered")
            def second_context():
                try:
                    with retry_openhands_writes(self.root):
                        failures.append("entered")
                except RuntimeError:
                    failures.append("rejected")
            thread = threading.Thread(target=second_context)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, ["rejected"])
            self.assertIs(sdk_io.atomic_write_text, active)
        self.assertIs(sdk_io.atomic_write_text, original)

    @unittest.skipUnless(os.name == "nt", "Reproduces Windows file-sharing denial")
    def test_real_windows_lock_retries_atomic_write_without_reusing_deleted_temp(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.path.write_text("original", encoding="utf-8")
        # Allow reading/writing but deny FILE_SHARE_DELETE, reproducing the
        # rename race without depending on an antivirus or another process.
        handle = kernel.CreateFileW(str(self.path), 0, 3, None, 3, 128, None)
        self.assertNotEqual(handle, wintypes.HANDLE(-1).value)
        events = []
        def unlock(event):
            nonlocal handle
            events.append(event)
            self.assertEqual(self.path.read_text(), "original")
            self.assertFalse(list(self.root.glob("*.tmp")))  # original SDK cleans failed temp
            kernel.CloseHandle(handle)
            handle = None
        try:
            with retry_openhands_writes(self.root, max_attempts=2, base_delay=0, on_retry=unlock):
                sdk_io.atomic_write_text(self.path, "replacement")
        finally:
            if handle is not None:
                kernel.CloseHandle(handle)
        self.assertEqual(len(events), 1)
        self.assertIn(events[0]["winerror"], (5, 32))
        self.assertEqual(self.path.read_text(), "replacement")
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
