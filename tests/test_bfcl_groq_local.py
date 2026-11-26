from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bfcl_groq_local as helper


class BFCLGroqLocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "comparison"

    def test_prepare_needs_no_key_or_network_and_preserves_catalog_caps(self):
        with (patch.dict(os.environ, {}, clear=True), patch("socket.create_connection") as network,
              patch.object(helper.getpass, "getpass") as prompt,
              patch.object(helper.run_bfcl, "main", return_value={"provider_calls": 0}) as runner):
            result = helper.main(["--output", str(self.output)])
        prompt.assert_not_called()
        network.assert_not_called()
        self.assertEqual(result, {"provider_calls": 0})
        runner.assert_called_once_with(["--config", str(self.output / "config.json"),
                                       "--output", str(self.output / "runplan")])
        config = json.loads((self.output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["model_catalog"], list(helper.MODELS))
        self.assertEqual(config["policies"], ["always_strong", "always_weak", "cheap_then_escalate"])
        self.assertEqual(config["max_total_generation_calls"], 4851)
        self.assertEqual(config["session_limits"], {"max_tokens_per_model": 160000, "max_provider_calls": 250})
        self.assertTrue(config["session_stop_on_infrastructure"])
        self.assertEqual(config["run_id"], "bfcl-development-comparison")
        self.assertEqual(config["spending"]["max_provider_spend_usd"], 0)
        self.assertEqual(config["local_helper_source"]["sha256"], hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest())

    def test_hidden_key_is_removed_even_when_execution_fails_and_never_written(self):
        secret = "fixture-test-secret"
        def failure(argv):
            self.assertEqual(os.environ.get("GROQ_API_KEY"), secret)
            self.assertIn("--execute", argv)
            self.assertNotIn(secret, repr(argv))
            raise RuntimeError("fixture execution failure")
        output = StringIO()
        with (patch.dict(os.environ, {}, clear=True), patch.object(helper.sys.stdin, "isatty", return_value=True),
              patch.object(helper.getpass, "getpass", return_value=secret) as prompt,
              patch.object(helper.run_bfcl, "main", side_effect=failure), redirect_stdout(output)):
            with self.assertRaisesRegex(RuntimeError, "fixture execution failure"):
                helper.main(["--output", str(self.output), "--execute"])
            self.assertNotIn("GROQ_API_KEY", os.environ)
        prompt.assert_called_once()
        self.assertNotIn(secret, output.getvalue())
        self.assertTrue(all(secret not in path.read_text(encoding="utf-8") for path in self.output.rglob("*") if path.is_file()))

    def test_authorized_environment_key_needs_no_prompt_and_is_restored(self):
        previous = "fixture-existing-key"
        def failure(argv):
            self.assertEqual(os.environ["GROQ_API_KEY"], previous)
            os.environ["GROQ_API_KEY"] = "changed-by-fixture"
            raise RuntimeError("fixture execution failure")
        with (patch.dict(os.environ, {"GROQ_API_KEY": previous}, clear=True),
              patch.object(helper.getpass, "getpass") as prompt,
              patch.object(helper.run_bfcl, "main", side_effect=failure)):
            with self.assertRaises(RuntimeError):
                helper.main(["--output", str(self.output), "--execute"])
            self.assertEqual(os.environ["GROQ_API_KEY"], previous)
        prompt.assert_not_called()

    def test_nonterminal_execution_without_key_fails_without_echo_fallback(self):
        with (patch.dict(os.environ, {}, clear=True), patch.object(helper.sys.stdin, "isatty", return_value=False),
              patch.object(helper.getpass, "getpass") as prompt,
              patch.object(helper.run_bfcl, "main") as runner):
            with self.assertRaisesRegex(RuntimeError, "terminal"):
                helper.main(["--output", str(self.output), "--execute"])
            self.assertNotIn("GROQ_API_KEY", os.environ)
        prompt.assert_not_called()
        runner.assert_not_called()

    def test_output_is_new_and_config_does_not_accept_another_provider(self):
        helper.prepare_config(self.output)
        with self.assertRaises(FileExistsError):
            helper.prepare_config(self.output)
        config = helper.run_bfcl._config(helper.DEFAULT_CONFIG)
        config["model_catalog"] = ["fixture/cheap", "fixture/strong"]
        other = self.output.parent / "other"
        with patch.object(helper.run_bfcl, "_config", return_value=config), self.assertRaisesRegex(ValueError, "Groq"):
            helper.prepare_config(other)
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
