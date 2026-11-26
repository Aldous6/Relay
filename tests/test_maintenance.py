"""Offline provider, real files and real Python tests through the complete flow."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from relay_router.maintenance import run_maintenance, _inspection_selection, _response_schema
from relay_router.maintenance_provider import MaintenanceProviderError
from relay_router.maintenance_routing import ActivityRouter
from relay_router.maintenance_workspace import Workspace


DEMO = Path(__file__).resolve().parents[1] / "examples" / "maintenance_demo"
FIX = ("    result = []\n"
       "    for label in labels:\n"
       "        value = label.strip().casefold()\n"
       "        if value and value not in result:\n"
       "            result.append(value)\n"
       "    return result")


class ScriptedProvider:
    kind = "offline_scripted_test_only"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, model, messages):
        self.calls.append((model, messages))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return {"content": json.dumps(result), "returned_model": model,
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}, "request_seconds": 0.01,
                "finish_reason": "stop"}


def inspection():
    return {"summary": "Corregir normalización y orden de etiquetas.", "files": ["src/labels.py"]}


def patch(old="    return list(set(labels))", new=FIX):
    return {"summary": "Normalizar antes de eliminar duplicados, conservando orden.",
            "edits": [{"path": "src/labels.py", "old_text": old, "new_text": new}]}


class MaintenanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        shutil.copytree(DEMO, self.repo, ignore=shutil.ignore_patterns("__pycache__"))
        self.original = (self.repo / "src/labels.py").read_bytes()
        self.run_dir = self.base / "run"
        self.router = ActivityRouter("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")

    def tearDown(self):
        self.tmp.cleanup()

    def run_agent(self, provider, max_repairs=1, source_files=None, **kwargs):
        return run_maintenance(root=self.repo, source_files=source_files or ["src/labels.py"],
            test_files=["tests/test_labels.py"], task="Corrige normalización de etiquetas.",
            run_dir=self.run_dir, provider=provider, router=self.router,
            python_executable=sys.executable, max_repairs=max_repairs, timeout_seconds=20, **kwargs)

    def test_direct_mode_uses_all_configured_sources_and_one_editing_call(self):
        (self.repo / "src/extra.py").write_text("CONSTANT = 1\n", encoding="utf-8")
        provider = ScriptedProvider([patch()])
        sources = ["src/labels.py", "src/extra.py"]
        result = self.run_agent(provider, inspection_mode="direct", source_files=sources)
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertEqual([call["activity"] for call in result["calls"]], ["editing"])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result["inspection_mode"], "direct")
        self.assertEqual(result["plan_source"], "direct_configured_scope")
        self.assertEqual(result["inspection_selection"]["source"], "direct_configured_scope")
        self.assertEqual(result["plan"]["files"], sources)
        payload = json.loads(provider.calls[0][1][-1]["content"])
        self.assertEqual(set(payload), {"task", "plan", "editable_paths", "editable_files", "read_only_tests",
                                        "verification", "response_schema", "instruction"})
        self.assertEqual(payload["editable_paths"], sources)
        self.assertEqual([item["path"] for item in payload["editable_files"]], sources)
        self.assertEqual([result["stage"] for result in result["verifications"]], ["baseline", "editing"])
        self.assertEqual(result["total_tokens"], 30)
        self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_direct_reuses_paired_baseline_and_preserves_identical_first_edit(self):
        first = ScriptedProvider([patch()])
        first_result = self.run_agent(first, inspection_mode="direct")
        self.run_dir = self.base / "paired-run"
        second = ScriptedProvider([patch()])
        second_result = self.run_agent(second, inspection_mode="direct",
            initial_diagnostics=first_result["public_baseline"]["diagnostics"])
        self.assertEqual(second_result["status"], "ready_for_review", second_result)
        self.assertEqual(first.calls[0][1], second.calls[0][1])
        self.assertEqual(second_result["public_baseline"]["source"], "paired_anchor_snapshot")
        self.assertEqual(second_result["verifications"][0]["stage"], "baseline")

    def test_direct_mode_still_rejects_outside_scope_and_invalid_mode(self):
        bad = patch()
        bad["edits"][0]["path"] = "tests/test_labels.py"
        result = self.run_agent(ScriptedProvider([bad]), inspection_mode="direct")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)
        self.run_dir = self.base / "invalid-mode"
        provider = ScriptedProvider([])
        with self.assertRaises(ValueError):
            self.run_agent(provider, inspection_mode="other")
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(provider.calls, [])

    def test_known_output_failures_are_repaired_with_original_baseline_and_usage(self):
        self.router = ActivityRouter("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b", mode="verified")
        for category in ("model_output_json", "model_output", "output_truncated"):
            with self.subTest(category=category):
                self.run_dir = self.base / category
                failure = MaintenanceProviderError("gsk_PRIVATE_ERROR_MESSAGE", category=category,
                    usage={"input_tokens": 11, "output_tokens": 37, "total_tokens": 48})
                provider = ScriptedProvider([failure, patch()])
                result = self.run_agent(provider, inspection_mode="direct")
                self.assertEqual(result["status"], "ready_for_review", result)
                self.assertEqual([call["activity"] for call in result["calls"]], ["editing", "repair"])
                self.assertEqual([model for model, _ in provider.calls],
                                 ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"])
                self.assertEqual([row["stage"] for row in result["verifications"]], ["baseline", "repair"])
                self.assertEqual(result["total_tokens"], 78)
                self.assertEqual(result["calls"][0]["usage"]["total_tokens"], 48)
                first_payload, repair_payload = [json.loads(messages[-1]["content"]) for _, messages in provider.calls]
                self.assertEqual(first_payload["verification"], repair_payload["verification"])
                self.assertEqual(first_payload["editable_files"], repair_payload["editable_files"])
                self.assertIn("Previous output violated the JSON contract; source unchanged", repair_payload["instruction"])
                self.assertEqual(result["generation_failures"][0]["latest_verification_stage"], "baseline")
                self.assertTrue(result["generation_failures"][0]["repair_scheduled"])
                self.assertFalse(result["generation_failures"][0]["source_changed_by_failed_generation"])
                self.assertNotIn("gsk_PRIVATE_ERROR_MESSAGE", (self.run_dir / "report.json").read_text())
                self.assertNotIn("gsk_PRIVATE_ERROR_MESSAGE", (self.run_dir / "events.jsonl").read_text())
                self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_json_rejection_code_can_repair_with_unknown_usage_preserved(self):
        failure = MaintenanceProviderError("provider text", category="model_output_json", status_code=400,
            provider_error_code="json_validate_failed", usage=None)
        result = self.run_agent(ScriptedProvider([failure, patch()]), inspection_mode="direct")
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertEqual(len(result["calls"]), 2)
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["total_tokens"])
        self.assertFalse(result["generation_failures"][0]["usage"])
        self.assertEqual(result["calls"][0]["provider_error_code"], "json_validate_failed")

    def test_known_failure_on_final_attempt_remains_blocked_without_fake_tests(self):
        def failure():
            return MaintenanceProviderError("bad JSON", category="model_output_json",
                                            usage={"input_tokens": 8, "output_tokens": 7})
        provider = ScriptedProvider([failure(), failure()])
        result = self.run_agent(provider, inspection_mode="direct")
        self.assertEqual(result["status"], "blocked", result)
        self.assertEqual(result["error"]["stage"], "repair")
        self.assertEqual(result["error"]["category"], "model_output_json")
        self.assertEqual(result["total_tokens"], 30)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(result["verifications"]), 1)
        self.assertEqual(len(result["generation_failures"]), 2)
        self.assertFalse(result["generation_failures"][-1]["repair_scheduled"])
        self.assertEqual((self.run_dir / "worktree/src/labels.py").read_bytes(), self.original)

    def test_transport_auth_budget_and_unobserved_output_failure_never_retry(self):
        failures = [MaintenanceProviderError("private", category=category,
                    usage={"input_tokens": 2, "output_tokens": 3})
                    for category in ("timeout", "transport", "http", "rate_limit", "request_budget", "call_budget")]
        failures.extend([MaintenanceProviderError("unknown output", category="model_output_json"),
                         MaintenanceProviderError("not output category", category="http",
                             provider_error_code="json_validate_failed")])
        for index, failure in enumerate(failures):
            with self.subTest(category=failure.category, index=index):
                self.run_dir = self.base / f"no-retry-{index}"
                provider = ScriptedProvider([failure, patch()])
                result = self.run_agent(provider, inspection_mode="direct")
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(result["generation_failures"], [])
                self.assertEqual(len(result["verifications"]), 1)

    def test_generation_failure_after_changed_source_preserves_last_real_verdict(self):
        failure = MaintenanceProviderError("bad JSON", category="model_output_json",
                                          usage={"input_tokens": 10, "output_tokens": 20})
        provider = ScriptedProvider([patch(new="    return []"), failure, patch(old="    return []")])
        result = self.run_agent(provider, max_repairs=2, inspection_mode="direct")
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertEqual([row["stage"] for row in result["verifications"]], ["baseline", "editing", "repair"])
        second, third = [json.loads(messages[-1]["content"]) for _, messages in provider.calls[1:]]
        self.assertEqual(second["verification"], third["verification"])
        self.assertEqual(second["editable_files"], third["editable_files"])
        self.assertIn("return []", third["editable_files"][0]["content"])
        self.assertEqual(result["generation_failures"][0]["latest_verification_stage"], "editing")

    def test_complete_flow_keeps_original_then_applies_reviewed_patch(self):
        provider = ScriptedProvider([inspection(), patch()])
        result = self.run_agent(provider)
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertFalse(result["verifications"][0]["passed"])
        self.assertTrue(result["verifications"][1]["passed"])
        self.assertEqual(result["verifications"][1]["tests_run"], 5)
        self.assertEqual([c["activity"] for c in result["calls"]], ["inspection", "editing"])
        self.assertEqual(provider.calls[0][0], "groq/openai/gpt-oss-20b")
        self.assertEqual(provider.calls[1][0], "groq/openai/gpt-oss-120b")
        self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)
        self.assertIn("casefold", (self.run_dir / "changes.diff").read_text())
        self.assertEqual(result["total_tokens"], 60)
        self.assertEqual(len((self.run_dir / "events.jsonl").read_text().splitlines()), 4)
        Workspace.reopen(self.run_dir).apply_to_original()
        self.assertNotEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_failing_edit_is_repaired_using_actual_test_output(self):
        provider = ScriptedProvider([inspection(), patch(new="    return []"), patch(old="    return []")])
        result = self.run_agent(provider)
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertEqual([v["passed"] for v in result["verifications"]], [False, False, True])
        self.assertEqual(result["calls"][-1]["activity"], "repair")
        self.assertIn("AssertionError", provider.calls[-1][1][-1]["content"])

    def test_limit_stops_without_modifying_original(self):
        result = self.run_agent(ScriptedProvider([inspection(), patch(new="    return []")]), max_repairs=0)
        self.assertEqual(result["status"], "tests_failed")
        self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_invalid_patch_explains_why_without_modifying_original(self):
        result = self.run_agent(ScriptedProvider([inspection(), patch(old="does not exist")]))
        self.assertEqual(result["status"], "blocked")
        self.assertIn("old_text", result["error"]["validation_message"])
        self.assertEqual(result["error"]["stage"], "editing")
        self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_inspection_paths_are_advisory_and_never_expand_authorized_scope(self):
        variants = [["editable/path.py"], ["tests/test_labels.py"], [], "src/labels.py",
                    ["../outside.py"], ["/outside.py"], ["C:\\outside.py"],
                    ["./src\\labels.py", "src/labels.py"], [{"path": "src/labels.py"}]]
        for index, received in enumerate(variants):
            with self.subTest(received=received):
                self.run_dir = self.base / f"selection-{index}"
                suggested = {**inspection(), "files": received}
                provider = ScriptedProvider([suggested, patch()])
                result = self.run_agent(provider)
                self.assertEqual(result["status"], "ready_for_review", result)
                self.assertEqual(len(provider.calls), 2)
                self.assertEqual(result["inspection_selection"]["received"], received)
                self.assertEqual(result["plan"]["files"], ["src/labels.py"])
                sent = json.loads(provider.calls[1][1][-1]["content"])
                self.assertEqual(sent["editable_paths"], ["src/labels.py"])
                self.assertEqual([f["path"] for f in sent["editable_files"]], ["src/labels.py"])
                self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_inspection_keeps_valid_subset_and_removes_read_only_or_unknown_paths(self):
        selection = _inspection_selection(["tests/test.py", "./src\\a.py", "src/a.py", "other.py"],
                                           ["src/a.py", "src/b.py"])
        self.assertEqual(selection["selected"], ["src/a.py"])
        self.assertEqual(selection["rejected"], ["tests/test.py", "other.py"])
        self.assertEqual(selection["source"], "inspection")
        # Traversal is never rewritten into an authorized path, even with a matching basename.
        self.assertEqual(_inspection_selection(["src/../src/a.py"], ["src/a.py"])["rejected"],
                         ["src/../src/a.py"])

    def test_schema_and_preserved_response_use_actual_paths_not_placeholders(self):
        for is_inspection in (True, False):
            schema = _response_schema(["src/labels.py"], inspection=is_inspection)
            self.assertNotIn("editable/path.py", json.dumps(schema))
        invalid = {"summary": "No files field"}
        result = self.run_agent(ScriptedProvider([invalid]))
        self.assertEqual(result["status"], "blocked")
        record = result["calls"][0]
        saved = json.loads((self.run_dir / record["response_path"]).read_text(encoding="utf-8"))
        self.assertEqual(json.loads(saved["content"]), invalid)
        self.assertTrue(record["response_sha256"])
        self.assertEqual(result["total_tokens"], 30)
        self.assertFalse((self.run_dir / "changes.diff").exists())
        self.assertNotIn("Revisa changes.diff", (self.run_dir / "report.md").read_text(encoding="utf-8"))

    def test_structured_provider_receives_schema_at_each_activity(self):
        class StructuredProvider(ScriptedProvider):
            def __init__(self, replies):
                super().__init__(replies)
                self.schemas = []
            def complete_structured(self, model, messages, response_schema):
                self.schemas.append(response_schema)
                return super().complete(model, messages)
        provider = StructuredProvider([inspection(), patch()])
        result = self.run_agent(provider)
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertEqual(provider.schemas[0]["properties"]["files"]["items"]["enum"], ["src/labels.py"])
        self.assertEqual(provider.schemas[1]["properties"]["edits"]["items"]["properties"]["path"]["enum"],
                         ["src/labels.py"])
        self.assertTrue(all(c["structured_output"] for c in result["calls"]))

    def test_prompt_json_provider_does_not_claim_server_schema_enforcement(self):
        class PromptJSONProvider(ScriptedProvider):
            output_mode = "prompt_json"
            def complete_structured(self, model, messages, response_schema):
                return {**super().complete(model, messages), "output_mode": self.output_mode}
        result = self.run_agent(PromptJSONProvider([patch()]), inspection_mode="direct")
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertFalse(result["calls"][0]["structured_output"])
        self.assertEqual(result["calls"][0]["output_mode"], "prompt_json")

    def test_public_diagnostics_exclude_execution_path_and_timing_but_keep_raw_evidence(self):
        provider = ScriptedProvider([inspection(), patch()])
        result = self.run_agent(provider)
        self.assertEqual(result["status"], "ready_for_review")
        diagnostics = json.loads(provider.calls[0][1][-1]["content"])["baseline"]
        self.assertNotIn(str(self.run_dir), diagnostics["stderr"]["text"])
        self.assertIn("<workspace>", diagnostics["stderr"]["text"])
        self.assertIn("<elapsed>", diagnostics["stderr"]["text"])
        self.assertIn(str(self.run_dir), result["verifications"][0]["stderr"])
        self.assertNotIn("<elapsed>", result["verifications"][0]["stderr"])

    def test_syntax_error_from_edit_can_be_repaired(self):
        provider = ScriptedProvider([inspection(), patch(new="    return ["), patch(old="    return [")])
        result = self.run_agent(provider)
        self.assertEqual(result["status"], "ready_for_review", result)
        self.assertEqual(result["verifications"][1]["status"], "runner_error")
        self.assertTrue(result["verifications"][2]["passed"])

    def test_model_cannot_edit_tests_or_escape_to_other_paths(self):
        for path in ("tests/test_labels.py", "../outside.py"):
            with self.subTest(path=path):
                self.run_dir = self.base / ("test-edit" if path.startswith("tests") else "escape")
                bad = patch()
                bad["edits"][0]["path"] = path
                result = self.run_agent(ScriptedProvider([inspection(), bad]))
                self.assertEqual(result["status"], "blocked")
                self.assertEqual((self.repo / "src/labels.py").read_bytes(), self.original)

    def test_provider_error_does_not_retry_or_expose_exception_text(self):
        provider = ScriptedProvider([RuntimeError("gsk_DO_NOT_LOG provider raw secret")])
        result = self.run_agent(provider)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(result["total_tokens"])
        self.assertNotIn("gsk_DO_NOT_LOG", (self.run_dir / "report.json").read_text())
        self.assertNotIn("gsk_DO_NOT_LOG", (self.run_dir / "events.jsonl").read_text())

    def test_interrupted_call_remains_pending_without_inventing_result(self):
        class Interrupted:
            def complete(self, *args):
                raise KeyboardInterrupt()
        result = self.run_agent(Interrupted())
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["calls"][0]["state"], "pending")
        self.assertIsNone(result["total_tokens"])


if __name__ == "__main__":
    unittest.main()
