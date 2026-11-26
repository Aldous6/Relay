"""Offline preparation integrity, not evidence of routing quality or savings."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_adaptive_pilot_test", ROOT / "tools/prepare_adaptive_pilot.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def historical(self):
        task = dict(task_id="old", group_id="old-group", workflow="repair", split="validation", prompt="  Keep\nIndentation\n")
        task["routing_text_sha256"] = prepare.digest(task["prompt"].encode())
        suite = dict(tasks=[task], models=["small", "large"], evaluation_id="old-evaluation")
        rows = [dict(task_id="old", group_id="old-group", workflow="repair", split="validation", model=model,
                     evaluation_id="old-evaluation", accepted=success, cost_usd=cost,
                     cost_basis="unknown" if cost is None else "tariff_estimate", cost_complete=cost is not None,
                     cost_scope="all_attempts", known_recorded_cost_subtotal_usd=.3)
                for model, success, cost in (("small", False, None), ("large", True, .5))]
        return suite, rows

    def test_consumed_validation_preserves_failures_unknown_cost_and_prompt(self):
        suite, rows = self.historical()
        records = prepare.development_records(suite, rows, lambda task: task["prompt"])
        self.assertEqual(len(records), 2)
        self.assertFalse(records[0]["accepted"])
        self.assertIsNone(records[0]["cost_usd"])
        self.assertEqual(records[0]["known_recorded_cost_subtotal_usd"], .3)
        self.assertEqual(records[0]["cost_basis"], "unknown")
        self.assertEqual(records[0]["split"], "development")
        self.assertEqual(records[0]["historical_split"], "validation")
        self.assertEqual(records[0]["prompt_text"], "  Keep\nIndentation\n")

    def test_missing_duplicate_or_untracked_outcomes_fail(self):
        suite, rows = self.historical()
        for malformed in (rows[:1], rows + rows[:1], [dict(rows[0], model="untracked"), rows[1]]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                prepare.development_records(suite, malformed, lambda task: task["prompt"])

    def test_rejects_changed_prompt_and_false_cost_completeness(self):
        suite, rows = self.historical()
        with self.assertRaises(ValueError):
            prepare.development_records(suite, rows, lambda task: task["prompt"].strip())
        rows[0]["cost_usd"] = .3
        with self.assertRaises(ValueError):
            prepare.development_records(suite, rows, lambda task: task["prompt"])

    def test_normalized_duplicates_fail_even_with_different_ids(self):
        old = [dict(task_id="old", group_id="old", prompt="  Ｋeep\nIndentation")]
        fresh = [dict(task_id="new", group_id="new", prompt="keep   indentation")]
        with self.assertRaisesRegex(ValueError, "normalized prompt"):
            prepare.assert_disjoint(old, fresh, lambda task: task["prompt"])
        fresh[0]["prompt"] = "different"
        prepare.assert_disjoint(old, fresh, lambda task: task["prompt"])
        fresh[0]["group_id"] = "old"
        with self.assertRaisesRegex(ValueError, "group_id"):
            prepare.assert_disjoint(old, fresh, lambda task: task["prompt"])

    def test_preflight_never_replaces_a_fixed_task(self):
        row = dict(task_id="replacement", group_id="replacement", workflow="repair", initial_source="initial",
                   gold_source="gold", public_tests="public", private_tests="private", prompt="new")
        helper = SimpleNamespace(rank=lambda value: value, task_prompt=lambda task: task["prompt"],
                                 run_check=lambda source, tests, **kw: dict(passed=source == "gold", category="test"))
        with patch.object(prepare, "SELECTED", {"repair": ("fixed",)}), self.assertRaisesRegex(ValueError, "selection"):
            prepare.freeze_fresh_tasks([row], [], helper)
        row["task_id"] = "fixed"
        with patch.object(prepare, "SELECTED", {"repair": ("fixed",)}):
            tasks, exclusions = prepare.freeze_fresh_tasks([row], [], helper)
        self.assertEqual(tasks[0]["split"], "validation")
        self.assertNotIn("gold_source", tasks[0])
        self.assertEqual(exclusions, [])

    def test_write_is_exclusive_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "fresh"
            manifest = prepare.write_bundle(destination, {"tasks": []}, {"records": []}, [])
            before = (destination / "fresh_suite.json").read_bytes()
            self.assertEqual(manifest["files"][0]["sha256"], prepare.digest(before))
            with self.assertRaises(FileExistsError):
                prepare.write_bundle(destination, {"changed": True}, {}, [])
            self.assertEqual((destination / "fresh_suite.json").read_bytes(), before)

    def test_pinned_input_tamper_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input.json").write_text("changed", encoding="utf-8")
            with patch.object(prepare, "ROOT", root), patch.object(prepare, "PINNED_INPUTS", {"input.json": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    prepare.read_pinned_inputs()


if __name__ == "__main__":
    unittest.main()
