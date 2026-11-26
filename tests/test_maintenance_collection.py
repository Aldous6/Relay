"""Offline paired trajectories with actual isolated Python verification."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from relay_router.maintenance_collection import collect_pair, _semantic_outcome, WEAK, STRONG
from relay_router.maintenance_provider import MaintenanceProviderError


class Provider:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, model, messages):
        self.calls.append((model, deepcopy(messages)))
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return {"content": json.dumps(value), "returned_model": model,
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "request_seconds": .01, "quota_wait_seconds": 0., "finish_reason": "stop"}


def inspection():
    return {"summary": "Corrige el incremento.", "files": ["increment.py"]}


def edit(old="return value", new="return value + 1"):
    return {"summary": "Corrige el incremento.",
            "edits": [{"path": "increment.py", "old_text": old, "new_text": new}]}


class MaintenanceCollectionTests(unittest.TestCase):
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
        self.original = (self.repo / "increment.py").read_bytes()
        self.directory = self.base / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def collect(self, provider, **kwargs):
        return collect_pair(root=self.repo, source_files=["increment.py"], test_files=["test_increment.py"],
                            task="Incrementa el argumento en uno.", run_dir=self.directory,
                            provider=provider, python_executable=sys.executable, timeout_seconds=15, **kwargs)

    def test_pair_has_identical_first_edit_and_separate_logical_and_acquisition_usage(self):
        provider = Provider([inspection(), edit(), edit()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(result["paired_state_verified"])
        self.assertEqual([model for model, _ in provider.calls], [WEAK, WEAK, STRONG])
        self.assertEqual(provider.calls[1][1], provider.calls[2][1])
        self.assertEqual(result["routing_text"], provider.calls[1][1][-1]["content"])
        self.assertEqual(result["acquisition"]["calls_started"], 3)
        self.assertEqual(result["acquisition"]["inspection_reuses"], 1)
        self.assertEqual(result["acquisition"]["total_tokens"], 90)
        self.assertEqual(result["branches"]["weak"]["total_tokens"], 60)
        self.assertEqual(result["branches"]["strong"]["total_tokens"], 60)
        self.assertTrue(all(branch["success"] for branch in result["branches"].values()))
        self.assertEqual((self.repo / "increment.py").read_bytes(), self.original)
        saved = json.loads((self.directory / "collection.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["acquisition"], result["acquisition"])

    def test_known_output_budget_exhaustion_uses_one_strong_repair_and_counts_failed_usage(self):
        provider = Provider([inspection(), MaintenanceProviderError("bounded output",
            category="output_truncated", usage={"input_tokens": 10, "output_tokens": 2048}), edit(), edit()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "completed", result)
        self.assertIs(result["branches"]["weak"]["success"], True)
        self.assertIs(result["branches"]["strong"]["success"], True)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(result["branches"]["strong"]["total_tokens"], 60)
        self.assertEqual((self.repo / "increment.py").read_bytes(), self.original)
        saved = json.loads((self.directory / "collection.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["acquisition"], result["acquisition"])

    def test_direct_mode_shares_first_edit_context_without_inspection_call(self):
        provider = Provider([edit(), edit()])
        result = self.collect(provider, inspection_mode="direct")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([model for model, _ in provider.calls], [WEAK, STRONG])
        self.assertEqual(provider.calls[0][1], provider.calls[1][1])
        self.assertEqual(result["acquisition"]["inspection_reuses"], 0)
        self.assertIsNone(result["continuation"]["inspection_model"])
        self.assertEqual(result["branches"]["weak"]["total_tokens"], 30)

    def test_weak_failed_attempt_uses_same_strong_repair_continuation(self):
        provider = Provider([inspection(), edit(new="return value - 1"),
                             edit(old="return value - 1"), edit()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([model for model, _ in provider.calls], [WEAK, WEAK, STRONG, STRONG])
        self.assertEqual([call["activity"] for call in result["branches"]["weak"]["calls"]],
                         ["inspection", "editing", "repair"])
        self.assertEqual(result["branches"]["weak"]["total_tokens"], 90)
        self.assertEqual(result["branches"]["strong"]["total_tokens"], 60)
        self.assertIn("AssertionError", provider.calls[2][1][-1]["content"])

    def test_both_failed_initial_edits_remain_bounded_at_five_provider_calls(self):
        provider = Provider([inspection(), edit(new="return value - 1"),
                             edit(old="return value - 1", new="return value - 2"),
                             edit(new="return value - 1"), edit(old="return value - 1")])
        result = self.collect(provider)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(provider.calls), 5)
        self.assertEqual(result["branches"]["weak"]["status"], "tests_failed")
        self.assertFalse(result["branches"]["weak"]["success"])
        self.assertTrue(result["branches"]["strong"]["success"])
        self.assertEqual(result["acquisition"]["total_tokens"], 150)
        self.assertEqual(result["branches"]["weak"]["total_tokens"], 90)
        self.assertEqual(result["branches"]["strong"]["total_tokens"], 90)

    def test_reversed_order_preserves_candidate_labels_and_shared_state(self):
        provider = Provider([inspection(), edit(), edit()])
        result = self.collect(provider, branch_order=("strong", "weak"))
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([model for model, _ in provider.calls], [WEAK, STRONG, WEAK])
        self.assertTrue(result["branches"]["weak"]["inspection_reused"])
        self.assertFalse(result["branches"]["strong"]["inspection_reused"])

    def test_unknown_provider_failure_stops_without_second_branch_or_zero_cost(self):
        provider = Provider([inspection(), MaintenanceProviderError("private provider detail", category="timeout")])
        result = self.collect(provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(set(result["branches"]), {"weak"})
        self.assertIsNone(result["acquisition"]["total_tokens"])
        self.assertIsNone(result["branches"]["weak"]["total_tokens"])
        self.assertIsNone(result["branches"]["weak"]["success"])
        self.assertEqual(len(provider.calls), 2)
        self.assertNotIn("private provider detail", (self.directory / "collection.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            self.collect(Provider([]))

    def test_invalid_inspection_is_not_reissued(self):
        provider = Provider([{"summary": "Missing files"}])
        result = self.collect(provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(result["routing_text"])

    def test_interrupt_is_journaled_without_second_branch_or_automatic_resume(self):
        provider = Provider([inspection(), KeyboardInterrupt()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["branches"]["weak"]["status"], "interrupted")
        self.assertEqual(result["acquisition"]["calls"][-1]["state"], "interrupted")
        self.assertEqual(len(provider.calls), 2)
        self.assertFalse(result["automatic_resume"])
        self.assertIsNone(result["acquisition"]["total_tokens"])
        self.assertIsNone(result["branches"]["weak"]["success"])

    def test_operational_verification_failure_is_unknown_and_stops_the_pair(self):
        from relay_router.maintenance import run_maintenance, _save
        def blocked(**kwargs):
            report = run_maintenance(**kwargs)
            report["status"] = "verification_blocked"
            _save(Path(kwargs["run_dir"]) / "report.json", report)
            return report
        provider = Provider([inspection(), edit()])
        with patch("relay_router.maintenance_collection.run_maintenance", side_effect=blocked):
            result = self.collect(provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["branches"]["weak"]["success"])
        self.assertFalse(result["branches"]["weak"]["outcome_known"])
        self.assertEqual(len(provider.calls), 2)

    def test_filesystem_error_is_not_a_semantic_failure(self):
        self.assertIsNone(_semantic_outcome({"status": "blocked", "calls": [],
            "error": {"stage": "editing", "type": "OSError"}}))

    def test_known_truncation_then_invalid_repair_is_observed_failure(self):
        report = {"status": "blocked", "error": {"stage": "repair", "type": "ValueError"},
            "calls": [{"state": "error", "category": "output_truncated",
                       "usage": {"input_tokens": 10, "output_tokens": 2048}},
                      {"state": "completed"}]}
        self.assertIs(_semantic_outcome(report), False)
        report["calls"][0]["category"] = "timeout"
        self.assertIsNone(_semantic_outcome(report))

    def test_truncation_then_invalid_repair_still_collects_other_branch(self):
        provider = Provider([inspection(), MaintenanceProviderError("truncated", category="output_truncated",
            usage={"input_tokens": 10, "output_tokens": 2048}), edit(old="not in source"), edit()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "completed", result)
        self.assertFalse(result["branches"]["weak"]["success"])
        self.assertTrue(result["branches"]["strong"]["success"])
        self.assertEqual(len(provider.calls), 4)

    def test_original_change_between_branches_invalidates_pair(self):
        owner = self
        class MutatingProvider(Provider):
            def complete(self, model, messages):
                response = super().complete(model, messages)
                if len(self.calls) == 2:
                    (owner.repo / "increment.py").write_text(
                        "def increment(value):\n    return value + 7\n", encoding="utf-8")
                return response
        provider = MutatingProvider([inspection(), edit()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["original_modified"])
        self.assertEqual(result["pairing_error"], "original_input_changed")
        self.assertEqual(len(provider.calls), 2)

    def test_known_invalid_patch_is_retained_as_failure_without_discarding_pair(self):
        provider = Provider([inspection(), edit(old="not in source"), edit()])
        result = self.collect(provider)
        self.assertEqual(result["status"], "completed", result)
        self.assertFalse(result["branches"]["weak"]["success"])
        self.assertTrue(result["branches"]["strong"]["success"])
        self.assertEqual(len(provider.calls), 3)

    def test_changed_second_task_is_stopped_before_api_dispatch(self):
        from relay_router.maintenance import run_maintenance
        def changed(**kwargs):
            if Path(kwargs["run_dir"]).name == "strong":
                kwargs["task"] += " Changed request."
            return run_maintenance(**kwargs)
        provider = Provider([inspection(), edit()])
        with patch("relay_router.maintenance_collection.run_maintenance", side_effect=changed):
            result = self.collect(provider)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["pairing_error"], "inspection_request_changed_or_incomplete")
        self.assertEqual(len(provider.calls), 2)

    def test_invalid_branch_order_fails_before_files_or_calls(self):
        provider = Provider([])
        with self.assertRaises(ValueError):
            self.collect(provider, branch_order=("weak", "weak"))
        self.assertFalse(self.directory.exists())
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
