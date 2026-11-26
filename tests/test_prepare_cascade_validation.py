"""Freshness, evaluator isolation and outcome-independent cohort contracts."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_cascade_validation as prep


class Helper:
    @staticmethod
    def rank(value):
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def task_prompt(row):
        return row["instruction"] + "\n" + row["initial_source"] + "\n" + row["public_tests"]

    @staticmethod
    def run_check(source, tests, *, directory, timeout):
        if source == "GOLD":
            return {"passed": True, "category": "passed", "seconds": .01}
        if tests == "PRIVATE":
            return {"passed": False, "category": "test_failure", "seconds": .01}
        return {"passed": False, "category": "test_timeout", "seconds": .01}


def task(task_id="HumanEval-70", instruction="implement", initial="pass", public="assert False"):
    return {"task_id": task_id, "workflow": "function-implementation", "instruction": instruction,
            "initial_source": initial, "public_tests": public,
            "private_tests": "PRIVATE", "gold_source": "GOLD"}


def empty_history():
    return {"task_ids": [], "group_ids": [], "normalized_prompt_sha256": []}


class CascadePreparationTests(unittest.TestCase):
    def test_history_excludes_failures_preflight_mentions_and_slash_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "history.json"
            old = task("HumanEval/70")
            old.update(accepted=False, group_id="old-group")
            path.write_text(json.dumps({"tasks": [old], "excluded_preflight": [{"task_id": "QuixBugs-knapsack"}]}))
            history = prep.history_from_paths([path], Helper, root=root)
            self.assertEqual(set(history["task_ids"]), {"HumanEval-70", "QuixBugs-knapsack"})
            self.assertIn("old-group", history["group_ids"])
            self.assertEqual(history["files"][0]["sha256"], prep.digest(path.read_bytes()))
            self.assertIn(prep.prior.normalized_prompt_digest(Helper.task_prompt(old)), history["normalized_prompt_sha256"])

    def test_normalized_prompt_overlap_is_removed_even_with_a_new_id(self):
        history = empty_history()
        original = task()
        history["normalized_prompt_sha256"] = [prep.prior.normalized_prompt_digest(Helper.task_prompt(original))]
        changed = task("HumanEval-71", instruction=" IMPLEMENT ")
        fresh, exclusions = prep.fresh_inventory([changed], history, Helper)
        self.assertEqual(fresh, [])
        self.assertEqual(exclusions[0]["reason"], "historical_normalized_prompt")

    def test_duplicates_keep_hash_first_without_consulting_outcome(self):
        left, right = task("HumanEval-70"), task("HumanEval-71")
        left["accepted"], right["accepted"] = False, True
        expected = min([left, right], key=lambda row: Helper.rank(row["task_id"]))["task_id"]
        fresh, exclusions = prep.fresh_inventory([left, right], empty_history(), Helper)
        self.assertEqual([row["task_id"] for row in fresh], [expected])
        left["accepted"], right["accepted"] = True, False
        changed, _ = prep.fresh_inventory([right, left], empty_history(), Helper)
        self.assertEqual([row["task_id"] for row in changed], [expected])
        self.assertEqual(exclusions[0]["reason"], "duplicate_normalized_prompt_in_candidate_pool")

    def test_private_evaluator_and_gold_never_enter_agent_facing_tasks(self):
        fresh, evaluators, checks, excluded = prep.preflight_all([task()], Helper)
        self.assertEqual(excluded, [])
        self.assertEqual(len(fresh), 1)
        self.assertNotIn("private_tests", fresh[0])
        self.assertNotIn("gold_source", fresh[0])
        self.assertNotIn("PRIVATE", fresh[0]["prompt_text"])
        self.assertEqual(evaluators[fresh[0]["task_id"]]["private_tests"], "PRIVATE")
        self.assertEqual(evaluators[fresh[0]["task_id"]]["sha256"], fresh[0]["private_tests_sha256"])

    def test_inconclusive_initial_operational_check_remains_in_cohort(self):
        fresh, _, _, excluded = prep.preflight_all([task()], Helper)
        self.assertEqual(excluded, [])
        self.assertEqual(fresh[0]["initial_public_check"]["status"], "inconclusive")

    def test_original_reference_eligibility_does_not_select_on_model_outcomes(self):
        passed, failed = {"passed": True}, {"passed": False}
        self.assertTrue(prep.reference_eligible(passed, passed, failed))
        self.assertFalse(prep.reference_eligible(failed, passed, failed))
        self.assertFalse(prep.reference_eligible(passed, failed, failed))
        self.assertFalse(prep.reference_eligible(passed, passed, passed))

    def test_duplicate_or_historical_task_cannot_be_frozen(self):
        rows, _, _, _ = prep.preflight_all([task()], Helper)
        prep.validate_disjoint(rows, empty_history())
        with self.assertRaises(ValueError):
            prep.validate_disjoint(rows + rows, empty_history())
        historical = empty_history()
        historical["task_ids"] = [rows[0]["task_id"]]
        with self.assertRaises(ValueError):
            prep.validate_disjoint(rows, historical)
        rows[0]["private_tests"] = "leak"
        with self.assertRaises(ValueError):
            prep.validate_disjoint(rows, empty_history())

    def test_exclusive_write_does_not_replace_previous_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            prep.exclusive(path, {"model_calls": 0})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                prep.exclusive(path, {"model_calls": 1})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
