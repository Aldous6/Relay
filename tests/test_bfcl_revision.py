from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_bfcl_revision as runner


def fixture():
    config = runner.run_bfcl._config(ROOT / "configs/bfcl_groq_free.example.json")
    cases = [{"id": f"multi_turn_base_{i}", "source_group": f"fixture-{i}",
              "entry": {"id": f"multi_turn_base_{i}", "question": [[{"role": "user", "content": "fixture"}]]},
              "reference": {"fixture": True}} for i in range(24)]
    manifest = {"cases": cases, "configuration": config, "policy_sequence": runner.run_bfcl.POLICIES,
                "freeze_sha256": "f" * 64, "source_files_sha256": {runner.CLASSIFIER_KEY: "a" * 64}}
    report = {"status": "stopped", "summary_current": True, "summary_present": True,
              "episodes": [{"episode_id": case["id"], "calls_started": int(i < 5),
                            "status": "completed" if i < 5 else "not_started", "success": i % 2 == 0}
                           for i, case in enumerate(cases)], "source_file_sha256": {}}
    return manifest, report


class BFCLRevisionTests(unittest.TestCase):
    def test_selection_excludes_any_attempt_and_never_consults_success(self):
        manifest, report = fixture()
        selected, excluded = runner.select_unattempted(manifest, report)
        self.assertEqual(len(selected), 19)
        self.assertEqual(excluded, [f"multi_turn_base_{i}" for i in range(5)])
        for row in report["episodes"]:
            row["success"] = None
        self.assertEqual((selected, excluded), runner.select_unattempted(manifest, report))
        self.assertEqual([case["id"] for case in selected], [f"multi_turn_base_{i}" for i in range(5, 24)])

    def test_running_parent_or_pending_attempt_is_rejected(self):
        manifest, report = fixture()
        report["status"] = "in_progress"
        with self.assertRaisesRegex(ValueError, "closed"):
            runner.select_unattempted(manifest, report)
        report["status"] = "stopped"
        report["episodes"][0]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "closed"):
            runner.select_unattempted(manifest, report)

    def test_prepare_is_offline_preserves_generation_and_declares_new_caps_and_hashes(self):
        manifest, report = fixture()
        original = deepcopy(manifest["configuration"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root / "parent"; parent.mkdir()
            path = parent / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
            report["source_file_sha256"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            evidence = root / "diagnostic.json"
            evidence.write_text(json.dumps({"benchmark_freeze_sha256": manifest["freeze_sha256"], "outcome": "error",
                "provider_error_facts": [{"code": "output_parse_failed"}], "provider_calls": 1}), encoding="utf-8")
            fake = Mock(); fake.manifest.return_value = {"kind": "fixture generator"}
            with (patch.object(runner, "audit_bfcl", return_value=report),
                  patch.object(runner.run_bfcl, "build_generator", return_value=fake),
                  patch.object(runner, "BFCLPacedGenerator", return_value=fake),
                  patch.object(runner, "BFCLSessionGenerator", return_value=fake),
                  patch.object(runner.run_bfcl, "preflight", return_value={"eligible": [None]*19, "excluded": [], "denominator": 19}),
                  patch.object(runner, "prepare_bfcl_episodes", return_value={"freeze_sha256": "fixture-freeze"}) as prepare,
                  patch.object(runner.getpass, "getpass") as prompt, patch("socket.create_connection") as network,
                  patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO())):
                context = runner.prepare_revision(parent, evidence, root / "new")
            prompt.assert_not_called(); network.assert_not_called(); prepare.assert_called_once()
            config = context["config"]
            for field in ("generation_options", "harness", "max_calls_per_turn", "model_catalog", "pacing", "quota_input_accounting", "policies", "spending"):
                self.assertEqual(config[field], original[field])
            self.assertEqual(config["max_total_generation_calls"], 19*21*3)
            self.assertEqual(config["session_limits"], {"max_tokens_per_model": 200000, "max_provider_calls": 250})
            change = config["revision_provenance"]["classification_change"]
            self.assertNotEqual(change["before_sha256"], change["after_sha256"])
            self.assertEqual(change["diagnosed_provider_code"], "output_parse_failed")
            self.assertFalse(config["revision_provenance"]["pool_with_parent"])
            self.assertTrue((root / "new/plan.json").is_file())

    def test_execution_restores_environment_on_error_and_never_passes_key_as_argument(self):
        context = {"cases": [], "policies": [], "generate": object(), "directory": Path("fixture"), "config": {}}
        secret = "fixture-hidden-key"
        def failure(*args):
            self.assertEqual(os.environ["GROQ_API_KEY"], secret)
            self.assertNotIn(secret, repr(args))
            raise RuntimeError("fixture stopped")
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys.stdin, "isatty", return_value=True),
              patch.object(runner.getpass, "getpass", return_value=secret), patch.object(runner, "run_bfcl_episodes", side_effect=failure)):
            with self.assertRaises(RuntimeError):
                runner.execute_revision(context)
            self.assertNotIn("GROQ_API_KEY", os.environ)
        with (patch.dict(os.environ, {"GROQ_API_KEY": secret}, clear=True), patch.object(runner.getpass, "getpass") as prompt,
              patch.object(runner, "run_bfcl_episodes", side_effect=failure)):
            with self.assertRaises(RuntimeError):
                runner.execute_revision(context)
            self.assertEqual(os.environ["GROQ_API_KEY"], secret)
            prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
