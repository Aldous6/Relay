"""Offline acquisition CLI contracts: bounded calls, provenance, and holdout gates."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from relay_router import maintenance_competition as cli
from relay_router.maintenance_provider import MaintenanceProviderError


class Provider:
    def __init__(self, *, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def complete(self, model, messages):
        self.calls.append((model, deepcopy(messages)))
        if self.fail_on == len(self.calls):
            raise MaintenanceProviderError("private provider text", category="timeout")
        payload = json.loads(messages[-1]["content"])
        result = ({"summary": "Corrige incremento.", "files": ["increment.py"]}
                  if "baseline" in payload else
                  {"summary": "Incrementa en uno.", "edits": [{"path": "increment.py",
                   "old_text": "return value", "new_text": "return value + 1"}]})
        return {"content": json.dumps(result), "returned_model": model, "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                "request_seconds": .01, "quota_wait_seconds": 0.}


class MaintenanceCompetitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        (self.repo / "increment.py").write_text("def increment(value):\n    return value\n", encoding="utf-8")
        (self.repo / "test_increment.py").write_text(
            "import unittest\nfrom increment import increment\n"
            "class Tests(unittest.TestCase):\n"
            "    def test_increment(self):\n        self.assertEqual(increment(1), 2)\n", encoding="utf-8")
        self.config = {
            "schema_version": 1, "repository": "repo", "source_files": ["increment.py"],
            "test_files": ["test_increment.py"],
            "models": {"inspection": cli.MODELS["weak"], "editing": cli.MODELS["strong"]},
            "limits": {"max_repairs": 1, "max_output_tokens": 2048, "max_requests": 3,
                       "request_budget": 8000, "minimum_interval_seconds": 60,
                       "request_timeout_seconds": 90, "test_timeout_seconds": 15},
            "spending": {"account_is_free_tier": True, "max_provider_spend_usd": 0}}
        self.config_path = self.base / "config.json"
        self.write(self.config_path, self.config)
        self.suite_path, self.directory = self.base / "suite.json", self.base / "run"
        self.suite = {"schema_version": 1, "scope": "diagnostic", "tasks": []}
        self.configure()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def configure(self, partitions=("development",)):
        self.suite = {"schema_version": 1,
            "scope": "diagnostic" if set(partitions) == {"development"} else "development_and_reserved_holdout",
            "tasks": [{"task_id": f"case_{index}", "group_id": f"family_{index}",
                       "partition": partition, "config": "config.json",
                       "task": f"Incrementa el argumento en uno. Petición independiente {index}."}
                      for index, partition in enumerate(partitions)]}
        self.write(self.suite_path, self.suite)

    def collect(self, provider, partition="development", max_calls=5, **kwargs):
        with redirect_stdout(io.StringIO()):
            return cli.collect(self.suite_path, self.directory, partition, max_calls,
                               provider=provider, **kwargs)

    def score_without_model(self):
        def fake_score(records):
            return {"reference": {"checkpoint_verified": True, "implementation": "offline_test_fixture"},
                    "rows": [{"task_id": row["task_id"], "probability": .25,
                              "routing_text_sha256": hashlib.sha256(row["routing_text"].encode()).hexdigest(),
                              "router_seconds": .01, "input_tokens": 30, "retained_tokens": 30,
                              "truncated_tokens": 0} for row in records]}
        with patch("relay_router.maintenance_reference.score_records", side_effect=fake_score):
            return cli.score(self.directory)

    def enable_acceptance(self):
        self.acceptance_path = self.repo / "accept_increment.py"
        self.acceptance_path.write_text(
            "import unittest\nfrom increment import increment\n"
            "PRIVATE_SENTINEL = 'not for model context'\n"
            "class Acceptance(unittest.TestCase):\n"
            "    def test_negative(self):\n        self.assertEqual(increment(-3), -2)\n",
            encoding="utf-8")
        for task in self.suite["tasks"]:
            task["acceptance_files"] = [self.acceptance_path.name]
        self.write(self.suite_path, self.suite)

    def test_private_acceptance_collect_load_and_fixed_reference_remain_separate_from_calibration(self):
        class OverfitWeakProvider(Provider):
            def complete(self, model, messages):
                reply = super().complete(model, messages)
                if model == cli.MODELS["weak"] and "baseline" not in json.loads(messages[-1]["content"]):
                    payload = json.loads(reply["content"])
                    payload["edits"][0]["new_text"] = "return 2"
                    reply["content"] = json.dumps(payload)
                return reply

        self.enable_acceptance()
        provider = OverfitWeakProvider()
        plan = {"operational_mode": "verified", "reference_threshold": .5}
        self.collect(provider, analysis_plan=plan)
        rows, manifest, journal, _ = cli.load_run(self.directory, need_reference=False)
        self.assertEqual(manifest["analysis_plan"], plan)
        entry = journal["tasks"]["case_0"]
        self.assertEqual(entry["state"], "completed")
        self.assertEqual(entry["acceptance_sha256"], cli.digest(self.directory / "case_0" / "acceptance.json"))
        self.assertTrue(rows[0]["branches"]["weak"]["visible_success"])
        self.assertFalse(rows[0]["branches"]["weak"]["acceptance_success"])
        self.assertFalse(rows[0]["branches"]["weak"]["success"])
        self.assertTrue(rows[0]["branches"]["strong"]["success"])
        self.assertEqual(len(provider.calls), 3)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(provider.calls))
        self.assertNotIn("accept_increment.py", json.dumps(provider.calls))
        for branch in ("weak", "strong"):
            self.assertFalse((self.directory / "case_0" / branch / "worktree" / "accept_increment.py").exists())
        self.score_without_model()
        report = cli.evaluate(self.directory)
        self.assertTrue(report["independent_acceptance_evaluated"])
        self.assertFalse(report["proof_of_complete_task_correctness"])
        self.assertEqual(report["relay_policy"]["name"], "relay_verified_cascade_v1")
        self.assertEqual(report["reference_selection"], {"threshold": .5, "selection_scope": "fixed_before_acquisition"})
        self.assertEqual(report["reference_threshold"], .5)
        self.assertEqual(report["decisions"][0]["relay_choice"], "weak")
        self.assertEqual(report["decisions"][0]["reference_choice"], "weak")
        primary = report["partitions"]["development"]["policies"]
        self.assertEqual(primary["relay"]["success_count"], 0)
        self.assertEqual(primary["routellm_bert_transfer"]["success_count"], 0)
        calibrated = report["development_calibrated_reference"]
        self.assertLessEqual(calibrated["selection"]["threshold"], .25)
        self.assertEqual(calibrated["selection"]["selection_scope"], "development_only_empirical_not_noninferiority")
        self.assertEqual(calibrated["comparison"]["policies"]["routellm_bert_transfer"]["success_count"], 1)
        resumed = Provider()
        with patch("relay_router.maintenance_acceptance.grade_pair", side_effect=AssertionError("No regrading")):
            self.collect(resumed, analysis_plan=plan)
        self.assertEqual(resumed.calls, [])

    def test_private_acceptance_mutation_after_suite_freeze_blocks_before_provider(self):
        self.enable_acceptance()
        frozen = cli.load_suite(self.suite_path)
        self.acceptance_path.write_text(self.acceptance_path.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")
        provider = Provider()
        with patch.object(cli, "load_suite", return_value=frozen), self.assertRaises(ValueError):
            self.collect(provider)
        self.assertEqual(provider.calls, [])

    def test_private_acceptance_changes_cannot_resume_acquisition(self):
        self.enable_acceptance()
        self.collect(Provider())
        self.acceptance_path.write_text(self.acceptance_path.read_text(encoding="utf-8") + "\n# changed after acquisition\n", encoding="utf-8")
        resumed = Provider()
        with self.assertRaisesRegex(ValueError, "No se puede reanudar"):
            self.collect(resumed)
        self.assertEqual(resumed.calls, [])

    def test_collect_load_score_compare_preserves_usage_tariffs_and_provenance(self):
        provider = Provider()
        self.collect(provider)
        rows, manifest, journal, reference = cli.load_run(self.directory, need_reference=False)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(reference)
        self.assertEqual(journal["tasks"]["case_0"]["state"], "completed")
        self.assertEqual(journal["invocations"][0]["calls_started"], 3)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(rows[0]["branches"]["weak"]["total_tokens"], 60)
        self.assertEqual(rows[0]["branches"]["strong"]["total_tokens"], 60)
        self.assertAlmostEqual(rows[0]["branches"]["weak"]["cost_usd"], 0.0000135)
        self.assertAlmostEqual(rows[0]["branches"]["strong"]["cost_usd"], 0.00002025)
        self.assertTrue(all(branch["success"] for branch in rows[0]["branches"].values()))
        self.assertEqual(manifest["suite"]["tariffs_sha256"], cli.digest(cli.TARIFFS))
        self.assertTrue(manifest["suite"]["sources_sha256"])
        self.score_without_model()
        scored, _, _, reference = cli.load_run(self.directory)
        self.assertEqual(scored[0]["reference_probability"], .25)
        self.assertEqual(reference["journal_sha256"], cli.digest(self.directory / "journal.json"))
        result = cli.evaluate(self.directory)
        self.assertFalse(result["relay_policy"]["learned"])
        self.assertFalse(result["partitions"]["development"]["interpretation"]["superiority_claim"])
        self.assertFalse(result["proof_of_complete_task_correctness"])
        self.assertEqual(result["cost_scenario"], "noncached_list_tariff_not_billing")
        self.assertTrue((self.directory / "comparison.md").is_file())

    def test_direct_acquisition_has_no_inspection_cost_and_preserves_pair_contract(self):
        provider = Provider()
        self.collect(provider, max_calls=4,
                     analysis_plan={"inspection_mode": "direct", "output_mode": "prompt_json",
                                    "operational_mode": "verified", "reference_threshold": .5})
        rows, manifest, journal, _ = cli.load_run(self.directory, need_reference=False)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(journal["invocations"][0]["calls_started"], 2)
        self.assertEqual(rows[0]["branches"]["weak"]["total_tokens"], 30)
        self.assertIsNone(rows[0]["acquisition_contract"]["continuation"]["inspection_model"])
        self.score_without_model()
        report = cli.evaluate(self.directory)
        self.assertEqual(report["reference_threshold"], .5)
        self.assertEqual(report["relay_policy"]["name"], "relay_verified_cascade_v1")

    def test_resuming_complete_acquisition_does_not_call_provider_or_credentials(self):
        self.collect(Provider())
        resumed = Provider()
        with patch("relay_router.maintenance_competition._key", side_effect=AssertionError("No key read")):
            self.collect(resumed)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(len(cli.read(self.directory / "journal.json")["invocations"]), 1)

    def test_five_call_budget_reserves_whole_pairs_and_resume_collects_only_remaining_task(self):
        self.configure(("development", "development"))
        first = Provider()
        self.collect(first, max_calls=5)
        journal = cli.read(self.directory / "journal.json")
        self.assertEqual(set(journal["tasks"]), {"case_0"})
        self.assertLessEqual(len(first.calls), 5)
        second = Provider()
        self.collect(second, max_calls=5)
        journal = cli.read(self.directory / "journal.json")
        self.assertEqual(set(journal["tasks"]), {"case_0", "case_1"})
        self.assertEqual(len(second.calls), 3)
        self.assertEqual([row["calls_started"] for row in journal["invocations"]], [3, 3])
        rows, _, _, _ = cli.load_run(self.directory, need_reference=False)
        self.assertEqual(len(rows), 2)

    def test_uncertain_task_is_not_retried_or_loaded_as_quality_failure(self):
        self.collect(Provider(fail_on=2))
        rows, _, journal, _ = cli.load_run(self.directory, need_reference=False)
        self.assertEqual(rows, [])
        self.assertEqual(journal["tasks"]["case_0"]["state"], "incomplete")
        resumed = Provider()
        with self.assertRaises(ValueError):
            self.collect(resumed)
        self.assertEqual(resumed.calls, [])

    def test_config_change_cannot_resume_old_collection(self):
        self.collect(Provider())
        self.config["limits"]["max_output_tokens"] = 1024
        self.write(self.config_path, self.config)
        resumed = Provider()
        with self.assertRaises(ValueError):
            self.collect(resumed)
        self.assertEqual(resumed.calls, [])

    def test_modified_branch_report_is_rejected_by_hash(self):
        self.collect(Provider())
        pair = cli.read(self.directory / "case_0" / "collection.json")
        path = self.directory / "case_0" / pair["branches"]["weak"]["report_path"]
        report = cli.read(path)
        report["status"] = "fabricated"
        self.write(path, report)
        with self.assertRaises(ValueError):
            cli.load_run(self.directory, need_reference=False)

    def test_holdout_requires_frozen_policy_before_any_calls(self):
        self.configure(("holdout",))
        provider = Provider()
        with self.assertRaises(ValueError):
            self.collect(provider, partition="holdout")
        self.assertEqual(provider.calls, [])
        self.assertFalse(self.directory.exists())

    def test_holdout_evaluation_does_not_fall_back_to_activity_rule(self):
        self.configure(("holdout",))
        self.collect(Provider(), partition="holdout", policy_sha256="a" * 64)
        self.score_without_model()
        with self.assertRaises(ValueError):
            cli.evaluate(self.directory)

    def test_invalid_budget_and_cross_partition_groups_fail_before_calls(self):
        provider = Provider()
        with self.assertRaises(ValueError):
            self.collect(provider, max_calls=4)
        self.configure(("development", "holdout"))
        self.suite["tasks"][1]["group_id"] = self.suite["tasks"][0]["group_id"]
        self.write(self.suite_path, self.suite)
        with self.assertRaises(ValueError):
            cli.load_suite(self.suite_path)
        self.assertEqual(provider.calls, [])
        self.assertFalse(self.directory.exists())

    def test_missing_usage_is_unknown_cost_instead_of_zero(self):
        tariffs = cli.read(cli.TARIFFS)
        self.assertIsNone(cli.branch_cost({"calls": [{"model": cli.MODELS["weak"], "usage": None}]}, tariffs))
        self.assertIsNone(cli.branch_cost({"calls": []}, tariffs))
        self.assertIsNone(cli.branch_cost({"calls": [{"model": cli.MODELS["weak"],
            "usage": {"input_tokens": True, "output_tokens": 20}}]}, tariffs))

    def test_pair_input_hash_must_match_inventory_frozen_before_acquisition(self):
        self.collect(Provider())
        location = self.directory / "case_0" / "collection.json"
        pair = cli.read(location)
        pair["input_files_sha256"]["increment.py"] = "a" * 64
        self.write(location, pair)
        journal = cli.read(self.directory / "journal.json")
        journal["tasks"]["case_0"]["collection_sha256"] = cli.digest(location)
        self.write(self.directory / "journal.json", journal)
        with self.assertRaisesRegex(ValueError, "congelados"):
            cli.load_run(self.directory, need_reference=False)

    def test_holdout_cannot_report_only_completed_subset(self):
        self.configure(("holdout", "holdout"))
        self.collect(Provider(), partition="holdout", policy_sha256="a" * 64)
        self.score_without_model()
        with self.assertRaisesRegex(ValueError, "Holdout incompleto"):
            cli.evaluate(self.directory)

    def test_missing_credentials_leave_resumable_empty_journal(self):
        with redirect_stdout(io.StringIO()), patch.object(cli, "_key", side_effect=ValueError("Missing key")):
            with self.assertRaises(ValueError):
                cli.collect(self.suite_path, self.directory, "development")
        self.assertEqual(cli.read(self.directory / "journal.json")["tasks"], {})
        self.collect(Provider())
        self.assertEqual(cli.load_run(self.directory, need_reference=False)[2]["tasks"]["case_0"]["state"], "completed")

    def test_training_cli_rejects_simulated_acquisition(self):
        self.configure(("fit",))
        self.collect(Provider(), partition="fit")
        self.score_without_model()
        with redirect_stdout(io.StringIO()):
            code = cli.main(["train", str(self.directory), "--output", str(self.base / "policy")])
        self.assertEqual(code, 2)
        self.assertFalse((self.base / "policy").exists())

    def test_holdout_cli_binds_expanded_inputs_and_calls_collector_only_after_matching_policy(self):
        self.configure(("holdout",))
        contract = cli.acquisition_contract(cli.load_suite(self.suite_path))
        policy = Mock()
        policy.manifest.return_value = {"acquisition_contract": contract, "policy_sha256": "a" * 64}
        arguments = ["run", "--suite", str(self.suite_path), "--partition", "holdout", "--collect-only",
                     "--policy", str(self.base / "policy.json"), "--policy-sha256", "a" * 64,
                     "--output", str(self.directory)]
        with patch("relay_router.maintenance_learning.load_policy", return_value=policy), patch.object(cli, "collect") as collector:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(arguments), 0)
            collector.assert_called_once()
        self.config["limits"]["max_output_tokens"] = 1024
        self.write(self.config_path, self.config)
        with patch("relay_router.maintenance_learning.load_policy", return_value=policy), patch.object(cli, "collect") as collector:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(arguments), 2)
            collector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
