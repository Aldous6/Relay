import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import repository_public_checks as grader
import repository_snapshot_host as host


class OverlayGradingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.repo = self.folder / "work"
        self.repo.mkdir()
        (self.repo / "calc.py").write_text("def add(x): return x + 1\n", encoding="utf-8")
        (self.repo / "test_calc.py").write_text("from calc import add\ndef test_old(): assert add(0) == 1\n", encoding="utf-8")
        self.dossier = self.folder / "private.json"
        self.row = {"log_parser": "pytest", "test_patch":
                    "--- a/test_calc.py\n+++ b/test_calc.py\n@@ -1,2 +1,3 @@\n from calc import add\n def test_old(): assert add(0) == 1\n+def test_new(): assert add(1) == 2\n",
                    "FAIL_TO_PASS": ["test_calc.py::test_new"], "PASS_TO_PASS": ["test_calc.py::test_old"]}
        self.dossier.write_text(json.dumps(self.row), encoding="utf-8")
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.repo)
        env = patch.dict(os.environ, RELAY_GRADER_SOURCE_FILES='["calc.py"]', RELAY_GRADER_CHANGED_FILES='["calc.py"]')
        env.start()
        self.addCleanup(env.stop)

    @unittest.skipUnless(shutil.which("git"), "Git is required to apply a synthetic patch")
    def test_private_patch_is_applied_and_no_worker_evidence_cannot_pass(self):
        status = grader.check(self.dossier)
        self.assertIn("def test_new", (self.repo / "test_calc.py").read_text(encoding="utf-8"))
        expected = 0 if importlib.util.find_spec("pytest") else 2
        self.assertEqual(status, expected)

    def test_no_unverified_required_case_is_accepted(self):
        ids = self.row["FAIL_TO_PASS"] + self.row["PASS_TO_PASS"]
        good = {"pytest_status": 0, "collected": ids, "overlay_modules": ["calc.py"],
                "phases": {i: {"setup": "passed", "call": "passed", "teardown": "passed"} for i in ids}}
        def classify(evidence):
            return grader._verdict(evidence, ids, ["calc.py"], ["calc.py"])
        self.assertEqual(classify(good), 0)
        self.assertEqual(classify({**good, "collected": ids[:1]}), 2)
        self.assertEqual(classify({**good, "overlay_modules": ["test_calc.py"]}), 2)
        self.assertEqual(classify({**good, "pytest_status": 3}), 2)
        for name in ids:
            for outcome in ("skipped", "rerun", "failed"):
                phases = {**good["phases"], name: {**good["phases"][name], "call": outcome}}
                result = classify({**good, "phases": phases, "pytest_status": int(outcome == "failed")})
                self.assertEqual(result, 10 if outcome == "failed" else 2)

    def test_only_per_instance_data_for_grading_enters_private_mount(self):
        repo = "example/project"
        task = {"task_id": "example", "revision": "a" * 40, "group_id": repo}
        config = {"official_instance_id": "example", "image": "example@sha256:" + "b" * 64,
                  "revision": task["revision"], "editable_paths": ["calc.py"]}
        row = {**self.row, "instance_id": "example", "base_commit": task["revision"], "test_cmds": ["untrusted"],
               "rebuild_cmds": ["untrusted"]}
        raw = json.dumps(row).encode()
        (self.folder / "official_cases.jsonl.gz").write_bytes(gzip.compress(raw + b"\n"))
        def isolated(workspace, checker, *args, private_record):
            self.assertEqual(workspace, self.repo)
            self.assertEqual(checker, host.PRIVATE_GRADER)
            self.assertEqual(json.loads(private_record.read_text(encoding="utf-8")), self.row)
            self.assertFalse(private_record.resolve().is_relative_to(self.repo))
            return 2
        with patch.object(host, "_case", return_value=(self.folder, config)), patch.object(host, "OFFICIAL_PRIVATE", self.folder), \
                patch.object(host, "_official_entry", return_value={"row_sha256": hashlib.sha256(raw).hexdigest()}), \
                patch.object(host, "_isolated_check", side_effect=isolated):
            self.assertEqual(host.private_verdict(task, self.repo), "inconclusive")

    def test_missing_batch_credential_fails_without_prompting(self):
        with patch.object(host, "_GROQ_CREDENTIAL", None), patch.dict(os.environ, GROQ_API_KEY=""), \
                patch.object(sys.stdin, "isatty", return_value=False), patch.object(host.getpass, "getpass", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                host.prepare_acquisition()

    def test_interactive_credential_only_requested_once(self):
        with patch.object(host, "_GROQ_CREDENTIAL", None), patch.dict(os.environ, GROQ_API_KEY=""), \
                patch.object(sys.stdin, "isatty", return_value=True), patch.object(host.getpass, "getpass", return_value="mock-credential") as prompt:
            host.prepare_acquisition()
            host.prepare_acquisition()
            prompt.assert_called_once()
            self.assertEqual(os.environ["GROQ_API_KEY"], "")


if __name__ == "__main__":
    unittest.main()
