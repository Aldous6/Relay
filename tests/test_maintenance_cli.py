"""CLI integration with a scripted provider, actual files and actual unittest.

No provider credentials or network are required. These exercise product wiring,
not model quality: the scripted patch is explicitly a test fixture.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from relay_router import maintenance_cli


SECRET = "gsk_CLI_OFFLINE_SECRET_MUST_NOT_APPEAR"
SOURCE = "def add(left, right):\n    return left - right\n"
TESTS = '''import unittest
from calculator import add

class AdditionTests(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)
    def test_negative(self):
        self.assertEqual(add(-2, -3), -5)
    def test_identity(self):
        self.assertEqual(add(7, 0), 7)
'''


class FixtureProvider:
    kind = "offline_scripted_cli_test_only"

    def __init__(self):
        self.models = []

    def manifest(self):
        return {"kind": self.kind, "network": False}

    def complete(self, model, messages):
        self.models.append(model)
        payload = json.loads(messages[-1]["content"])
        if "baseline" in payload:
            result = {"summary": "Corregir la suma.", "files": ["src/calculator.py"]}
        else:
            result = {"summary": "Usar suma en lugar de resta.", "edits": [{
                "path": "src/calculator.py", "old_text": "return left - right",
                "new_text": "return left + right"}]}
        return {"content": json.dumps(result), "returned_model": model,
                "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                "request_seconds": .01, "finish_reason": "stop"}


class MaintenanceCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repository"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "tests").mkdir()
        (self.repo / "src/calculator.py").write_text(SOURCE, encoding="utf-8")
        (self.repo / "tests/test_calculator.py").write_text(TESTS, encoding="utf-8")
        self.source_bytes = (self.repo / "src/calculator.py").read_bytes()
        self.config_path = self.base / "maintenance.json"
        self.run_dir = self.base / "run"
        self.config = {
            "schema_version": 1, "repository": "repository",
            "source_files": ["src/calculator.py"],
            "test_files": ["tests/test_calculator.py"],
            "models": {"inspection": "groq/openai/gpt-oss-20b",
                       "editing": "groq/openai/gpt-oss-120b"},
            "limits": {"max_repairs": 0, "max_output_tokens": 2048,
                       "max_requests": 2, "request_budget": 8000,
                       "minimum_interval_seconds": 60, "request_timeout_seconds": 90,
                       "test_timeout_seconds": 20},
            "spending": {"account_is_free_tier": True, "max_provider_spend_usd": 0},
        }
        self.write_config()

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self, value=None):
        self.config_path.write_text(json.dumps(self.config if value is None else value), encoding="utf-8")

    def invoke(self, argv):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = maintenance_cli.main(argv)
        return result, stdout.getvalue(), stderr.getvalue()

    def run_fixture(self, mode="activity"):
        fixture = FixtureProvider()
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
                patch.object(maintenance_cli, "_key", return_value=SECRET) as key, \
                patch.object(maintenance_cli, "GroqMaintenanceProvider", return_value=fixture) as factory, \
                patch("socket.create_connection", side_effect=AssertionError("Network forbidden in CLI tests")):
            result = self.invoke(["run", "--config", str(self.config_path),
                                  "--task", "Corrige la suma.", "--output", str(self.run_dir),
                                  "--mode", mode])
        key.assert_called_once_with()
        self.assertEqual(factory.call_args.args, (SECRET,))
        self.assertEqual(factory.call_args.kwargs["max_requests"], 2)
        self.assertEqual(factory.call_args.kwargs["request_budget"], 8000)
        return result, fixture

    def test_load_config_resolves_repository_and_does_not_create_workspace(self):
        value, root = maintenance_cli.load_config(self.config_path)
        self.assertEqual(root, self.repo.resolve())
        self.assertEqual(value, self.config)
        self.assertFalse((self.repo / "relay-validation-unused").exists())
        self.assertEqual((self.repo / "src/calculator.py").read_bytes(), self.source_bytes)

    def test_default_demo_runs_but_custom_config_requires_a_task(self):
        fixture = FixtureProvider()
        with patch.object(maintenance_cli, "DEFAULT_CONFIG", self.config_path), \
                patch.object(maintenance_cli, "_key", return_value=SECRET), \
                patch.object(maintenance_cli, "GroqMaintenanceProvider", return_value=fixture):
            status, out, err = self.invoke(["run", "--output", str(self.run_dir)])
        self.assertEqual(status, 0, out + err)
        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["task"], maintenance_cli.DEMO_TASK)
        with patch.object(maintenance_cli, "_key") as key:
            status, _, err = self.invoke(["run", "--config", str(self.config_path)])
        self.assertEqual(status, 2)
        self.assertIn("--task", err)
        key.assert_not_called()

    def test_check_is_offline_and_never_prints_or_requests_secret(self):
        for secret in ("", SECRET):
            with self.subTest(key_available=bool(secret)), patch.dict(os.environ, {"GROQ_API_KEY": secret}), \
                    patch.object(maintenance_cli, "_key") as key, \
                    patch.object(maintenance_cli, "GroqMaintenanceProvider") as factory, \
                    patch("socket.create_connection") as network:
                status, out, err = self.invoke(["check", "--config", str(self.config_path)])
                self.assertEqual(status, 0, err)
                self.assertIn("Configuración válida", out)
                self.assertNotIn(SECRET, out + err)
                self.assertIn("Clave disponible" if secret else "no está en este proceso", out)
                key.assert_not_called()
                factory.assert_not_called()
                network.assert_not_called()
        self.assertFalse(self.run_dir.exists())

    def test_missing_declared_file_blocks_check_before_credentials_or_provider(self):
        (self.repo / "tests/test_calculator.py").unlink()
        with patch.object(maintenance_cli, "_key") as key, \
                patch.object(maintenance_cli, "GroqMaintenanceProvider") as factory:
            status, out, err = self.invoke(["check", "--config", str(self.config_path)])
        self.assertEqual(status, 2)
        self.assertNotIn("Configuración válida", out)
        self.assertTrue(err)
        key.assert_not_called()
        factory.assert_not_called()

    def test_unknown_config_fields_and_paid_account_block_before_provider(self):
        variants = []
        unknown = deepcopy(self.config)
        unknown["api_key"] = SECRET
        variants.append(unknown)
        paid = deepcopy(self.config)
        paid["spending"]["account_is_free_tier"] = False
        variants.append(paid)
        spending = deepcopy(self.config)
        spending["spending"]["max_provider_spend_usd"] = 1
        variants.append(spending)
        for config in variants:
            with self.subTest(config_keys=list(config)):
                self.write_config(config)
                with patch.object(maintenance_cli, "_key") as key, \
                        patch.object(maintenance_cli, "GroqMaintenanceProvider") as factory:
                    status, out, err = self.invoke(["run", "--config", str(self.config_path),
                        "--task", "Corrige la suma.", "--output", str(self.run_dir)])
                self.assertEqual(status, 2)
                self.assertNotIn(SECRET, out + err)
                key.assert_not_called()
                factory.assert_not_called()
                self.assertFalse(self.run_dir.exists())

    def test_run_routes_and_verifies_patch_without_modifying_original(self):
        (status, out, err), provider = self.run_fixture()
        self.assertEqual(status, 0, out + err)
        self.assertEqual(provider.models, ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"])
        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "ready_for_review")
        self.assertEqual([v["passed"] for v in report["verifications"]], [False, True])
        self.assertEqual(report["verifications"][-1]["tests_run"], 3)
        self.assertEqual(report["total_tokens"], 40)
        self.assertEqual((self.repo / "src/calculator.py").read_bytes(), self.source_bytes)
        self.assertIn("Para aplicar después de revisar", out)
        self.assertNotIn(SECRET, out + err)
        for name in ("report.json", "report.md", "configuration.json", "events.jsonl"):
            self.assertNotIn(SECRET, (self.run_dir / name).read_text(encoding="utf-8"))

    def test_blocked_run_does_not_advertise_nonexistent_diff(self):
        fixture = FixtureProvider()
        fixture.complete = Mock(return_value={"content": '{"summary":"missing files"}',
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}})
        with patch.object(maintenance_cli, "_key", return_value=SECRET), \
                patch.object(maintenance_cli, "GroqMaintenanceProvider", return_value=fixture):
            status, out, err = self.invoke(["run", "--config", str(self.config_path),
                "--task", "Corrige la suma.", "--output", str(self.run_dir)])
        self.assertEqual(status, 2, out + err)
        self.assertNotIn("Diff:", out)
        self.assertFalse((self.run_dir / "changes.diff").exists())

    def test_always_strong_mode_uses_editing_model_for_both_activities(self):
        (status, out, err), provider = self.run_fixture(mode="always_strong")
        self.assertEqual(status, 0, out + err)
        self.assertEqual(provider.models, ["groq/openai/gpt-oss-120b"] * 2)
        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["policy"]["mode"], "always_strong")

    def test_existing_output_is_rejected_before_key_or_provider(self):
        self.run_dir.mkdir()
        with patch.object(maintenance_cli, "_key") as key, \
                patch.object(maintenance_cli, "GroqMaintenanceProvider") as factory:
            status, _, err = self.invoke(["run", "--config", str(self.config_path),
                "--task", "Corrige la suma.", "--output", str(self.run_dir)])
        self.assertEqual(status, 2)
        self.assertIn("carpeta de salida ya existe", err)
        key.assert_not_called()
        factory.assert_not_called()

    def test_apply_writes_verified_content_and_hashes_once_without_credentials(self):
        (status, out, err), _ = self.run_fixture()
        self.assertEqual(status, 0, out + err)
        with patch.object(maintenance_cli, "_key") as key, \
                patch.object(maintenance_cli, "GroqMaintenanceProvider") as factory:
            status, out, err = self.invoke(["apply", str(self.run_dir)])
        self.assertEqual(status, 0, out + err)
        content = (self.repo / "src/calculator.py").read_bytes()
        self.assertIn(b"return left + right", content)
        result = json.loads((self.run_dir / "application.json").read_text(encoding="utf-8"))
        self.assertEqual(result["sha256"]["src/calculator.py"], hashlib.sha256(content).hexdigest())
        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "applied")
        self.assertTrue(report["original_modified"])
        key.assert_not_called()
        factory.assert_not_called()
        second, _, _ = self.invoke(["apply", str(self.run_dir)])
        self.assertEqual(second, 2)
        self.assertEqual((self.repo / "src/calculator.py").read_bytes(), content)

    def test_apply_refuses_to_overwrite_original_changed_since_run(self):
        (status, out, err), _ = self.run_fixture()
        self.assertEqual(status, 0, out + err)
        changed = self.source_bytes + b"\n# User changed this after the run.\n"
        (self.repo / "src/calculator.py").write_bytes(changed)
        status, _, err = self.invoke(["apply", str(self.run_dir)])
        self.assertEqual(status, 2)
        self.assertTrue(err)
        self.assertEqual((self.repo / "src/calculator.py").read_bytes(), changed)
        self.assertFalse((self.run_dir / "application.json").exists())

    def test_missing_key_in_noninteractive_run_blocks_before_provider(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
                patch.object(maintenance_cli.sys.stdin, "isatty", return_value=False), \
                patch.object(maintenance_cli.getpass, "getpass") as prompt, \
                patch.object(maintenance_cli, "GroqMaintenanceProvider") as factory:
            status, out, err = self.invoke(["run", "--config", str(self.config_path),
                "--task", "Corrige la suma.", "--output", str(self.run_dir)])
        self.assertEqual(status, 2)
        self.assertIn("Falta GROQ_API_KEY", err)
        prompt.assert_not_called()
        factory.assert_not_called()
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
