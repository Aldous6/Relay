"""Offline boundaries of API planning; never import a provider or read a key."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_agent_api_v8", ROOT / "tools/run_agent_api_v8.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def selected_config():
    config = runner._read(ROOT / "configs/agent_api_v8.example.json")
    config["catalog_status"] = "selected"
    config["models"] = {"strong": "mock/strong", "weak": "mock/weak"}
    config["model_catalog"] = ["mock/weak", "mock/strong"]
    config["models_metadata"] = {model: {"returned_model_ids": [model], "context_window": 8192,
                                          "api_key_env": "MOCK_API_KEY"} for model in config["model_catalog"]}
    config["spending"]["account_is_free_tier"] = True
    config["spending"]["account_tier_source"] = "Account inspected for this frozen development run"
    return config


class ApiRunnerTests(unittest.TestCase):
    def test_concrete_groq_example_prepares_but_cannot_execute_without_account_tier(self):
        config_path = ROOT / "configs/agent_api_v8.groq_free.example.json"
        config = runner._read(config_path)
        self.assertTrue(runner._catalog_ready(config))
        self.assertFalse(config["spending"]["account_is_free_tier"])
        self.assertEqual(config["pricing"], {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frozen"
            with patch.dict(sys.modules, {"litellm": None}), patch("os.getenv", side_effect=AssertionError("no credential reads")), redirect_stdout(StringIO()):
                plan = runner.main(["--config", str(config_path), "--only-strong", "--output", str(output)])
            self.assertTrue(plan["frozen"])
            self.assertEqual(plan["policies"], ["always_strong"])
            self.assertIn("confirm_free_account_tier_and_source", plan["execution_blockers"])
            with patch.object(runner, "run_episodes") as run, self.assertRaisesRegex(ValueError, "confirm_free_account"):
                runner.main(["--config", str(config_path), "--only-strong", "--output", str(output), "--execute"])
            run.assert_not_called()

    def test_selected_catalog_freezes_without_provider_import_or_credentials(self):
        config = selected_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            output = Path(directory) / "frozen"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(sys.modules, {"litellm": None}), patch("os.getenv", side_effect=AssertionError("no credential reads")), redirect_stdout(StringIO()):
                result = runner.main(["--config", str(path), "--output", str(output)])
            self.assertTrue(result["frozen"])
            self.assertEqual(result["generation_calls"], 0)
            self.assertEqual(result["policies"], ["always_strong", "always_weak", "cheap_then_escalate"])
            manifest = runner._read(output / "manifest.json")
            self.assertEqual(manifest["configuration"]["harness"]["contract_version"], 2)
            self.assertNotIn("relay_v8_predictive", manifest["policies"])
            self.assertFalse((output / "journal.json").exists())

    def test_template_plan_has_no_frozen_output_provider_import_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "untouched"
            with patch.dict(sys.modules, {"litellm": None}), patch("os.getenv", side_effect=AssertionError("no credential reads")), redirect_stdout(StringIO()):
                result = runner.main(["--output", str(output)])
            self.assertEqual(result["generation_calls"], 0)
            self.assertEqual(result["mode"], "template_plan_only")
            self.assertFalse(result["frozen"])
            self.assertFalse(output.exists())
            self.assertIn("select_exact_catalog_and_verified_model_metadata", result["execution_blockers"])
            self.assertFalse(result["predictive_control_available"])

    def test_setting_selected_alone_does_not_make_template_executable(self):
        config = runner._read(ROOT / "configs/agent_api_v8.example.json")
        config["catalog_status"] = "selected"
        self.assertFalse(runner._catalog_ready(config))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(runner, "prepare_episodes") as prepare, patch.object(runner, "run_episodes") as run:
                with self.assertRaisesRegex(ValueError, "incomplete catalog"):
                    runner.main(["--config", str(path), "--execute"])
            prepare.assert_not_called()
            run.assert_not_called()

    def test_zero_budget_needs_account_tier_but_does_not_imply_zero_total_cost(self):
        config = selected_config()
        plan = runner._spending_plan(config, 30)
        self.assertEqual(plan["execution_blockers"], [])
        self.assertIsNone(plan["conservative_provider_cost_bound_usd"])
        config["spending"]["account_is_free_tier"] = False
        self.assertIn("confirm_free_account_tier_and_source", runner._spending_plan(config, 30)["execution_blockers"])

    def test_paid_budget_is_checked_before_preparation_or_provider(self):
        config = selected_config()
        config["spending"]["max_provider_spend_usd"] = .01
        self.assertIn("complete_exact_model_tariffs_for_paid_execution", runner._spending_plan(config, 30)["execution_blockers"])
        config["pricing"] = {model: {"input_per_million": 20., "output_per_million": 40.,
                                     "cached_input_per_million": 5., "source": "official price source", "as_of": "2026-11-12"}
                             for model in config["model_catalog"]}
        plan = runner._spending_plan(config, 30)
        self.assertAlmostEqual(plan["conservative_provider_cost_bound_usd"], 30 * (8192*20 + 1024*40)/1e6)
        self.assertIn("conservative_provider_cost_bound_exceeds_budget", plan["execution_blockers"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(runner, "prepare_episodes") as prepare:
                with self.assertRaisesRegex(ValueError, "bound_exceeds_budget"):
                    runner.main(["--config", str(path), "--execute"])
            prepare.assert_not_called()

    def test_call_cap_covers_every_policy_and_infrastructure_attempt(self):
        config = selected_config()
        config["spending"]["max_total_generation_calls"] = 10
        self.assertIn("generation_call_cap_below_frozen_maximum", runner._spending_plan(config, 11)["execution_blockers"])
        for value in (True, -1, None):
            config["spending"]["max_total_generation_calls"] = value
            self.assertIn("generation_call_cap_below_frozen_maximum", runner._spending_plan(config, 1)["execution_blockers"])

    def test_inline_secret_rejected_before_output(self):
        config = selected_config()
        config["models_metadata"]["mock/weak"]["api_key"] = "do-not-print"
        with self.assertRaisesRegex(ValueError, "credentials"):
            runner._check_no_inline_secrets(config)

    def test_synthetic_defaults_cannot_be_presented_as_reserved_evaluation(self):
        config = selected_config()
        config["evaluation_split"] = "evaluation"
        with self.assertRaisesRegex(ValueError, "development only"):
            runner._tasks(config)

    def test_source_group_overlap_between_supplied_partitions_is_rejected(self):
        config = selected_config()
        fit = runner.build_tasks(1, 1)
        evaluation = runner.build_tasks(2, 1)
        evaluation[0]["source_group"] = fit[0]["source_group"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps({"fit": fit, "evaluation": evaluation}), encoding="utf-8")
            config["task_splits_path"] = str(path)
            config["evaluation_split"] = "evaluation"
            with self.assertRaisesRegex(ValueError, "source_group"):
                runner._tasks(config)


if __name__ == "__main__":
    unittest.main()
