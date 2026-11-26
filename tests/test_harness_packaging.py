"""Validate the distributed runtime, isolated from source and site packages."""
import base64
import configparser
import csv
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCES = {
    "workflow.py", "workflow_policy.py", "workflow_cli.py", "prompt_selection.py",
    "adaptive_policy.py", "adaptive_runtime.py", "adaptive_training.py",
    "cascade.py", "cascade_verifier.py", "cascade_evaluation.py", "cascade_unittest.py",
    "integrations/__init__.py", "integrations/openhands.py", "integrations/openhands_io.py",
    "integrations/openhands_cascade.py", "integrations/groq_recovery.py", "integrations/routellm.py",
}


class HarnessPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(ROOT / "tools/build_harness_runtime.py"),
             "--output", str(cls.directory / "dist")],
            cwd=cls.directory, capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        wheels = list((cls.directory / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError("Builder must emit exactly one runtime wheel")
        cls.wheel = wheels[0]
        with zipfile.ZipFile(cls.wheel) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise AssertionError("Duplicate archive entries")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise AssertionError("Archive member escapes installation root")
            cls.contents = {name: archive.read(name) for name in names}
            cls.extracted = cls.directory / "installed"
            archive.extractall(cls.extracted)
        info = {name.split("/")[0] for name in names if ".dist-info/" in name}
        if len(info) != 1:
            raise AssertionError("Wheel must contain one distribution metadata directory")
        cls.info = info.pop()

    def test_only_runtime_allowlist_and_distribution_metadata_are_packaged(self):
        expected = {"relay_harness/" + name for name in RUNTIME_SOURCES}
        expected.update({"relay_harness/__init__.py", "relay_harness/__main__.py",
                         "relay_harness/source_manifest.json"})
        expected.update(self.info + "/" + name for name in
                        ("METADATA", "WHEEL", "entry_points.txt", "RECORD"))
        self.assertEqual(set(self.contents), expected)
        self.assertFalse(any(name.startswith("relay_router/") for name in self.contents))

    def test_record_covers_every_byte_and_external_checksum_matches(self):
        record_path = self.info + "/RECORD"
        rows = list(csv.reader(io.StringIO(self.contents[record_path].decode())))
        self.assertEqual(len(rows), len(self.contents))
        self.assertEqual({row[0] for row in rows}, set(self.contents))
        for name, encoded_hash, size in rows:
            if name == record_path:
                self.assertEqual((encoded_hash, size), ("", ""))
                continue
            content = self.contents[name]
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            self.assertEqual(encoded_hash, "sha256=" + digest, name)
            self.assertEqual(int(size), len(content), name)
        checksum = self.wheel.with_suffix(".whl.sha256").read_text(encoding="utf-8")
        self.assertEqual(checksum, hashlib.sha256(self.wheel.read_bytes()).hexdigest()
                         + "  " + self.wheel.name + "\n")

    def test_manifest_traces_exact_reviewed_sources_without_scientific_data(self):
        manifest = json.loads(self.contents["relay_harness/source_manifest.json"])
        self.assertIs(manifest["scientific_data_included"], False)
        self.assertEqual(manifest["scope"], "local_runtime_preview_no_demonstrated_savings")
        source_hashes = manifest["source_sha256"]
        self.assertEqual(set(source_hashes), {"src/relay_router/" + name for name in RUNTIME_SOURCES})
        for relative in RUNTIME_SOURCES:
            content = self.contents["relay_harness/" + relative]
            self.assertEqual(content, (ROOT / "src/relay_router" / relative).read_bytes())
            self.assertEqual(source_hashes["src/relay_router/" + relative],
                             hashlib.sha256(content).hexdigest())
        metadata = Parser().parsestr(self.contents[self.info + "/METADATA"].decode())
        self.assertEqual(metadata["Version"], manifest["version"])

    def test_metadata_has_no_required_dependencies_and_separate_entrypoint(self):
        metadata = Parser().parsestr(self.contents[self.info + "/METADATA"].decode())
        self.assertEqual(metadata["Name"], "relay-harness")
        self.assertEqual(metadata["Requires-Python"], ">=3.11")
        self.assertEqual(metadata.get_all("Provides-Extra"), ["openhands", "routellm-bert", "adapt"])
        self.assertEqual(metadata.get_all("Requires-Dist"),
                         ['openhands-sdk==1.62.0; extra == "openhands"',
                          'httpx==0.28.1; extra == "openhands"',
                          'litellm==1.105.7; extra == "openhands"',
                          'torch==2.9.0; extra == "routellm-bert"',
                          'transformers==4.57.1; extra == "routellm-bert"',
                          'scikit-learn==1.8.0; extra == "adapt"'])
        wheel = Parser().parsestr(self.contents[self.info + "/WHEEL"].decode())
        self.assertEqual(wheel["Root-Is-Purelib"], "true")
        self.assertEqual(wheel["Tag"], "py3-none-any")
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(self.contents[self.info + "/entry_points.txt"].decode())
        self.assertEqual(dict(entrypoints["console_scripts"]),
                         {"relay-harness": "relay_harness.workflow_cli:main"})

    def test_extracted_wheel_imports_and_cli_help_without_site_or_repository(self):
        code = "import sys\nsys.path.insert(0, " + repr(str(self.extracted)) + ")\n"
        code += textwrap.dedent('''
            from relay_harness import Relay, TaskSession, load_adapted_relay, SpendBudget, run_cascade, CheckCommand, CommandVerifier
            from relay_harness.cascade_evaluation import replay_public_cascade
            from relay_harness.workflow_policy import propose_policy, validate_policy
            from relay_harness.prompt_selection import PairwisePromptSelector
            from relay_harness.integrations.routellm import LocalBERTScorer
            import relay_harness
            import runpy
            task = Relay("fixture-model").task("fixture-workflow")
            assert isinstance(task, TaskSession)
            assert task.finish(accepted=None)["cost_usd"] is None
            sys.argv = ["relay-harness", "--help"]
            try:
                runpy.run_module("relay_harness", run_name="__main__")
            except SystemExit as error:
                assert error.code in (None, 0), error.code
            for prefix in ("relay_router", "pandas", "numpy", "joblib", "sklearn", "pyarrow", "openhands", "litellm", "torch", "transformers"):
                assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
        ''')
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=self.directory,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertIn("--help", result.stdout)


if __name__ == "__main__":
    unittest.main()
