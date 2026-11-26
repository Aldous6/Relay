"""Runner fixtures check preparation and gates without issuing any API request."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_bfcl as runner


class FixtureRuntime:
    def __init__(self, snapshot, entry):
        self.entry = entry

    def native_tools(self):
        return [{"type": "function", "function": {"name": "fixture_tool", "description": "Fixture only",
                 "parameters": {"type": "object", "properties": {}}}}]


def cases(length=8):
    return [{"id": f"multi_turn_base_{index}", "source_group": f"fixture-{index}",
             "entry": {"id": f"multi_turn_base_{index}", "question": [[{"role": "user", "content": "x" * (length + index)}]],
                       "initial_config": {}, "involved_classes": ["FixtureRuntime"]}, "reference": None}
            for index in range(24)]


class BFCLRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/bfcl_groq_free.example.json").read_text(encoding="utf-8"))
        self.config["snapshot_path"] = str(self.root / "snapshot")
        self.config["exposure_record_path"] = str(self.root / "incident.json")
        self.config["console_progress"] = False
        self.config["quota_input_accounting"] = {"method": "utf8_bytes_bound"}
        self.path = self.root / "config.json"

    def invoke(self, values, *extra, prepare=None):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        ledger = {"exposed_ids": [f"fixture-exposed-{i}" for i in range(18)],
                  "blocked_evaluation_ids": [f"fixture-blocked-{i}" for i in range(3)]}
        snapshot = {"selection": {"development": [case["id"] for case in values]}, "manifest_sha256": "a" * 64,
                    "manifest": {"selection_sha256": "b" * 64}, "exposure_ledger": ledger,
                    "exposure_ledger_sha256": "c" * 64}
        def load(directory, ids=None, *, include_reference=False):
            selected = [deepcopy(case) for case in values if ids is None or case["id"] in ids]
            if include_reference:
                for case in selected:
                    case["reference"] = {"id": case["id"], "ground_truth": [["fixture_tool()"]]}
            return selected
        with (patch.object(runner, "read_snapshot", return_value=snapshot),
              patch.object(runner, "register_exposure_record") as exposure,
              patch.object(runner, "load_selected_cases", side_effect=load) as load_cases,
              patch.object(runner, "BFCLRuntime", FixtureRuntime),
              patch.object(runner, "prepare_bfcl_episodes", return_value={"freeze_sha256": "fixture-freeze"}) as freeze,
              patch.object(runner, "run_bfcl_episodes") as run,
              patch("socket.create_connection") as network,
              patch.dict("os.environ", {}, clear=True),
              redirect_stdout(StringIO())):
            result = runner.main(["--config", str(self.path), "--output", str(self.root / "output"), *extra])
            network.assert_not_called()
            run.assert_not_called()
            exposure.assert_called_once()
            return result, load_cases.call_args_list, freeze.call_args_list

    def test_zero_eligible_freezes_full_denominator_without_claiming_actual_token_excess(self):
        result, loads, freezes = self.invoke(cases(12000))
        self.assertEqual(result["preflight_denominator"], 24)
        self.assertEqual(result["eligible_groups"], 0)
        self.assertEqual(result["excluded_groups"], 24)
        self.assertEqual(result["generation_calls"], 0)
        self.assertFalse(freezes)
        self.assertEqual(len(loads), 1)
        report = json.loads((self.root / "output/preflight.json").read_text())
        self.assertEqual({row["status"] for row in report["excluded"]}, {"unverified_under_bound"})
        self.assertFalse(report["actual_token_counts_measured"])
        plan = json.loads((self.root / "output/plan.json").read_text())["payload"]
        self.assertEqual(plan["exposure_ledger_sha256"], "c" * 64)
        self.assertEqual(len(plan["exposure_ledger"]["blocked_evaluation_ids"]), 3)

    def test_smoke_subset_preserves_all_controls_and_uses_native_payload_bound(self):
        result, loads, freezes = self.invoke(cases(), "--limit", "1")
        self.assertEqual(result["preflight_denominator"], 24)
        self.assertEqual(result["eligible_groups"], 24)
        self.assertEqual(result["selected_groups"], 1)
        self.assertEqual(result["selected_ids"], ["multi_turn_base_0"])
        self.assertEqual(result["maximum_generation_calls"], 63)
        self.assertEqual(result["freeze_sha256"], "fixture-freeze")
        self.assertEqual(len(loads), 2)
        self.assertTrue(loads[-1].kwargs["include_reference"])
        selected, policies, _, config = freezes[0].args
        self.assertEqual(len(selected), 1)
        self.assertEqual([policy.name for policy in policies], runner.POLICIES)
        self.assertTrue(all(tuple(policy.models) == tuple(config["model_catalog"]) for policy in policies))
        self.assertEqual(config["exposure_ledger_sha256"], "c" * 64)
        report = json.loads((self.root / "output/preflight.json").read_text())
        first = report["eligible"][0]
        request = runner.build_initial_request(cases()[0], self.config, runtime_factory=FixtureRuntime)
        base = runner.BFCLLiteLLMGenerator(self.config["models_metadata"], self.config["generation_options"])
        self.assertEqual(first["initial_input_utf8_bytes"], base.request_input_bytes(request))
        self.assertEqual(first["conservative_request_bound"], base.request_input_bytes(request) + 2048 + 512)

    def test_insufficient_call_cap_is_reported_without_expanding_or_reading_references(self):
        self.config["max_total_generation_calls"] = 1
        result, loads, freezes = self.invoke(cases(), "--limit", "1")
        self.assertIn("declared_call_cap_below_selected_episode_bound", result["execution_blockers"])
        self.assertFalse(freezes)
        self.assertEqual(len(loads), 1)
        plan = json.loads((self.root / "output/plan.json").read_text())["payload"]
        self.assertEqual(plan["configuration"]["max_total_generation_calls"], 1)

    def test_missing_free_declaration_is_a_blocker_and_never_imputed(self):
        self.config["spending"]["account_is_free_tier"] = False
        result, _, _ = self.invoke(cases(), "--limit", "1")
        self.assertIn("free_account_declaration_and_source_required", result["execution_blockers"])
        self.assertEqual(result["generation_calls"], 0)

    def test_execute_with_zero_eligible_is_blocked_before_any_provider(self):
        with self.assertRaisesRegex(ValueError, "Execution blocked"):
            self.invoke(cases(12000), "--execute")
        self.assertTrue((self.root / "output/plan.json").exists())

    def test_smoke_selection_is_order_independent_and_uses_no_outcomes(self):
        eligible = [{"id": str(index), "initial_input_utf8_bytes": index % 3} for index in range(10)]
        self.assertEqual(runner.select_smoke_ids(eligible, 4), runner.select_smoke_ids(eligible[::-1], 4))
        with self.assertRaises(ValueError):
            runner.select_smoke_ids(eligible, 0)

    def test_config_refuses_holdout_policy_changes_and_inline_secrets(self):
        for updates in ({"evaluation_split": "evaluation"}, {"policies": ["always_strong"]}, {"api_key": "fixture-only"}):
            value = {**self.config, **updates}
            self.path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                runner._config(self.path)

    def test_token_estimate_is_explicit_and_never_labelled_a_conservative_bound(self):
        class Estimator:
            supported_models = tuple(self.config["model_catalog"])
            def manifest(self): return {"kind": "fixture-token-estimate"}
            def estimate(self, request): return {"estimated_input_tokens": 100}
        self.config["quota_input_accounting"] = {"method": "gptoss_local_token_estimate",
            "margin_tokens": 256, "template_date": "2026-11-12", "reasoning_effort": "medium"}
        with patch("routerbench.bfcl_tokens.GPTOSSTokenEstimator", return_value=Estimator()):
            result, _, _ = self.invoke(cases(12000), "--limit", "1")
        self.assertEqual(result["eligible_groups"], 24)
        report = json.loads((self.root / "output/preflight.json").read_text())
        first = report["eligible"][0]
        self.assertIsNone(first["conservative_request_bound"])
        self.assertEqual(first["request_budget_estimate"], 100 + 2048 + 512)
        self.assertEqual(first["status"], "eligible_under_local_estimate")
        self.assertFalse(first["input_accounting"]["actual_provider_tokens"])

    def test_mismatched_template_and_generation_reasoning_effort_is_rejected(self):
        self.config["quota_input_accounting"] = {"method": "gptoss_local_token_estimate", "reasoning_effort": "low"}
        with self.assertRaisesRegex(ValueError, "same reasoning effort"):
            runner.build_generator(self.config)


if __name__ == "__main__":
    unittest.main()
