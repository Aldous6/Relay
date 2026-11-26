"""Ensure empty/broken test discovery cannot authorize the cheap branch."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "src/relay_router/cascade_unittest.py"


class CascadeUnittestRunnerTests(unittest.TestCase):
    def invoke(self, source=None, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            if source is not None:
                (root / "tests/test_fixture.py").write_text(source, encoding="utf-8")
            result = subprocess.run([sys.executable, "-I", "-B", str(RUNNER), *arguments],
                                    cwd=root, capture_output=True, timeout=15)
            return result.returncode

    def test_missing_empty_and_import_error_are_inconclusive(self):
        self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.invoke("import nonexistent_relay_fixture_12345\n"), 2)
        self.assertEqual(self.invoke(None, "--start-directory", "missing"), 2)

    def test_actual_pass_and_actual_failure_are_distinct(self):
        prefix = "import unittest\nclass Case(unittest.TestCase):\n    def test_one(self):\n"
        self.assertEqual(self.invoke(prefix + "        self.assertEqual(2, 2)\n"), 0)
        self.assertEqual(self.invoke(prefix + "        self.assertEqual(2, 3)\n"), 1)
        self.assertEqual(self.invoke(prefix + "        raise RuntimeError('bug')\n"), 1)

    def test_skipped_and_expected_failures_are_not_success_evidence(self):
        prefix = "import unittest\nclass Case(unittest.TestCase):\n"
        self.assertEqual(self.invoke(prefix + "    @unittest.skip('disabled')\n    def test_one(self): pass\n"), 2)
        self.assertEqual(self.invoke(prefix + "    @unittest.expectedFailure\n    def test_one(self): self.fail()\n"), 2)

    def test_declared_minimum_is_enforced(self):
        source = "import unittest\nclass Case(unittest.TestCase):\n    def test_one(self): pass\n"
        self.assertEqual(self.invoke(source, "--min-tests", "2"), 2)


if __name__ == "__main__":
    unittest.main()
