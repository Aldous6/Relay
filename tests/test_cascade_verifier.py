"""Operational verifier contracts; subprocesses are local Python fixtures only."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.cascade_verifier import CheckCommand, CommandVerifier


class CascadeVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "original"
        self.workspace.mkdir()
        self.test_path = self.workspace / "host_check.py"
        self.test_path.write_text("assert 2 + 2 == 4\n", encoding="utf-8")

    def verifier(self, *, checks=None, protected_files=("host_check.py",)):
        return CommandVerifier.capture(self.workspace, checks=checks or [
            CheckCommand("unit", [sys.executable, "-I", "-B", "host_check.py"], timeout_seconds=10)],
            protected_files=protected_files)

    def test_pinned_checks_work_in_fresh_workspace_without_reserved_labels(self):
        verifier = self.verifier()
        branch = self.root / "branch"
        shutil.copytree(self.workspace, branch)
        result = verifier(branch)
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.config_sha256, verifier.config_sha256)
        self.assertEqual(result.checks[0].returncode, 0)
        self.assertGreater(result.duration_seconds, 0)
        self.assertEqual([item.phase for item in result.integrity], ["before:unit", "after:unit"])
        self.assertNotIn("accepted", result.to_dict())
        self.assertEqual(json.loads(json.dumps(result.to_dict()))["status"], "passed")

    def test_initial_bug_does_not_prevent_freezing_but_failure_is_observable(self):
        self.test_path.write_text("assert False\n", encoding="utf-8")
        verifier = self.verifier()
        result = verifier.verify(self.workspace)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "check_failed")
        self.assertNotEqual(result.checks[0].returncode, 0)

    def test_tampering_before_verification_prevents_any_check_execution(self):
        self.test_path.write_text("from pathlib import Path\nPath('executed.txt').write_text('x')\n", encoding="utf-8")
        verifier = self.verifier()
        self.test_path.write_text("print('model rewrote tests to pass')\n", encoding="utf-8")
        result = verifier.verify(self.workspace)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.reason, "protected_file_integrity_failed")
        self.assertEqual(result.checks, ())
        self.assertEqual(result.integrity[0].status, "changed")
        self.assertFalse((self.workspace / "executed.txt").exists())

    def test_tampering_during_successful_command_overrides_zero_exit(self):
        self.test_path.write_text("from pathlib import Path\nPath(__file__).write_text('pass')\n", encoding="utf-8")
        result = self.verifier().verify(self.workspace)
        self.assertEqual(result.checks[0].returncode, 0)
        self.assertEqual(result.checks[0].status, "passed")
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.integrity[-1].status, "changed")

    def test_deleted_protected_file_is_inconclusive_before_launch(self):
        verifier = self.verifier()
        self.test_path.unlink()
        result = verifier.verify(self.workspace)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.integrity[0].status, "missing")
        self.assertFalse(result.checks)

    def test_timeout_is_inconclusive_and_process_is_reaped(self):
        self.test_path.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        verifier = self.verifier(checks=[CheckCommand("bounded", [sys.executable, "-I", "-B", "host_check.py"],
                                                    timeout_seconds=.2)])
        result = verifier.verify(self.workspace)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.checks[0].reason, "timeout")
        self.assertIsNotNone(result.checks[0].returncode)
        self.assertLess(result.duration_seconds, 15)

    def test_unavailable_executable_does_not_count_as_solution_failure(self):
        verifier = self.verifier(checks=[CheckCommand("unavailable", [str(self.root / "missing-interpreter.exe")])])
        result = verifier.verify(self.workspace)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.checks[0].reason, "launch_error")
        self.assertIsNone(result.checks[0].returncode)

    def test_setup_errors_and_no_tests_exit_codes_are_inconclusive(self):
        for code in (2, 5):
            with self.subTest(returncode=code):
                verifier = self.verifier(checks=[CheckCommand(
                    "runner", [sys.executable, "-I", "-c", f"raise SystemExit({code})"])])
                result = verifier.verify(self.workspace)
                self.assertEqual(result.status, "inconclusive")
                self.assertEqual(result.checks[0].reason, "unexpected_exit_code")
                self.assertEqual(result.checks[0].returncode, code)

    def test_only_explicit_failure_returncodes_mean_failed_tests(self):
        for failure_codes, expected in (((1,), "inconclusive"), ((3,), "failed"), ((), "inconclusive")):
            with self.subTest(failure_returncodes=failure_codes):
                verifier = self.verifier(checks=[CheckCommand(
                    "custom-runner", [sys.executable, "-I", "-c", "raise SystemExit(3)"],
                    failure_returncodes=failure_codes)])
                result = verifier.verify(self.workspace)
                self.assertEqual(result.status, expected)
        for invalid in ((0,), (-1,), (256,), (True,), (1.0,), (1, 1), "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CheckCommand("runner", [sys.executable], failure_returncodes=invalid)

    def test_empty_or_duplicate_checks_and_unpinned_tests_rejected(self):
        check = CheckCommand("unit", [sys.executable, "host_check.py"])
        for checks, files in (([], ["host_check.py"]), ([check], []),
                              ([check, check], ["host_check.py"]),
                              ([check], ["host_check.py", ".\\host_check.py"])):
            with self.subTest(checks=checks, files=files), self.assertRaises(ValueError):
                CommandVerifier.capture(self.workspace, checks=checks, protected_files=files)

    def test_invalid_paths_and_shell_string_rejected(self):
        for invalid in ("../outside.py", "C:/outside.py", "/absolute.py", "\\\\server\\file.py", "."):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                self.verifier(protected_files=[invalid])
        with self.assertRaises(ValueError):
            CheckCommand("unit", "python host_check.py")
        with self.assertRaises(ValueError):
            CheckCommand("unit", [sys.executable], cwd="../outside")
        for invalid in (0, -1, True, float("nan"), float("inf"), 601):
            with self.subTest(timeout=invalid), self.assertRaises(ValueError):
                CheckCommand("unit", [sys.executable], timeout_seconds=invalid)

    def test_no_output_arguments_or_secrets_are_exported_and_environment_is_filtered(self):
        secret = "PRIVATE_VERIFIER_FIXTURE"
        self.test_path.write_text(
            "import os, sys\nassert 'GROQ_API_KEY' not in os.environ\n"
            "assert 'OPENAI_API_KEY' not in os.environ\n"
            "print(sys.argv[1])\nprint(sys.argv[1], file=sys.stderr)\n", encoding="utf-8")
        verifier = self.verifier(checks=[CheckCommand("privacy", [sys.executable, "-I", "-B", "host_check.py", secret])])
        with patch.dict(os.environ, {"GROQ_API_KEY": secret, "OPENAI_API_KEY": secret}):
            result = verifier.verify(self.workspace)
        self.assertTrue(result.passed)
        serialized = json.dumps(result.to_dict())
        self.assertNotIn(secret, serialized)
        self.assertNotIn(sys.executable, serialized)
        self.assertNotIn("stdout", serialized)
        self.assertNotIn("stderr", serialized)
        self.assertNotIn("argv", serialized)

    def test_all_declared_checks_must_pass_and_unknown_dominates_failure(self):
        fail = CheckCommand("failure", [sys.executable, "-I", "-c", "raise SystemExit(1)"])
        passed = CheckCommand("success", [sys.executable, "-I", "-c", "pass"])
        missing = CheckCommand("missing", [str(self.root / "missing.exe")])
        result = self.verifier(checks=[fail, passed]).verify(self.workspace)
        self.assertEqual(result.status, "failed")
        self.assertEqual([check.status for check in result.checks], ["failed", "passed"])
        result = self.verifier(checks=[fail, missing]).verify(self.workspace)
        self.assertEqual(result.status, "inconclusive")

    def test_config_digest_tracks_commands_timeouts_and_pinned_check_bytes(self):
        original = self.verifier()
        self.assertEqual(original.config_sha256, self.verifier().config_sha256)
        different = self.verifier(checks=[CheckCommand("unit", [sys.executable, "-I", "-B", "host_check.py"],
                                                      timeout_seconds=11)])
        self.assertNotEqual(original.config_sha256, different.config_sha256)
        different_codes = self.verifier(checks=[CheckCommand(
            "unit", [sys.executable, "-I", "-B", "host_check.py"], timeout_seconds=10,
            failure_returncodes=(1, 3))])
        self.assertNotEqual(original.config_sha256, different_codes.config_sha256)
        self.test_path.write_text("assert True\n", encoding="utf-8")
        self.assertNotEqual(original.config_sha256, self.verifier().config_sha256)

    def test_deleted_workspace_and_working_directory_fail_closed(self):
        nested = self.workspace / "nested"
        nested.mkdir()
        verifier = self.verifier(checks=[CheckCommand("nested", [sys.executable, "-c", "pass"], cwd="nested")])
        nested.rmdir()
        self.assertEqual(verifier.verify(self.workspace).status, "inconclusive")
        self.assertEqual(verifier.verify(self.root / "missing").reason, "workspace_unavailable")

    def test_symlink_substitution_is_rejected_even_for_identical_contents(self):
        verifier = self.verifier()
        target = self.root / "external.py"
        target.write_bytes(self.test_path.read_bytes())
        self.test_path.unlink()
        try:
            self.test_path.symlink_to(target)
        except OSError:
            self.skipTest("This Windows account cannot create symbolic links")
        result = verifier.verify(self.workspace)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.integrity[0].status, "unsafe_path")


if __name__ == "__main__":
    unittest.main()
