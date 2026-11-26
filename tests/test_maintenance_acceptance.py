"""Private acceptance stays outside model context and binds replay to evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from relay_router import maintenance_competition as cli
from relay_router.maintenance_acceptance import grade_pair, load_acceptance, validate_acceptance
from relay_router.maintenance_collection import collect_pair
from relay_router.maintenance_workspace import Workspace


class Provider:
    def __init__(self, *, invalid_weak=False):
        self.calls = []
        self.invalid_weak = invalid_weak

    def complete(self, model, messages):
        self.calls.append((model, deepcopy(messages)))
        payload = json.loads(messages[-1]["content"])
        if "baseline" in payload:
            answer = {"summary": "Revisa incremento.", "files": ["increment.py"]}
        else:
            weak = model == cli.MODELS["weak"]
            answer = {"summary": "Corrige incremento.", "edits": [{"path": "increment.py",
                "old_text": "missing text" if weak and self.invalid_weak else "return value",
                "new_text": "return 2" if weak else "return value + 1"}]}
        return {"content": json.dumps(answer), "usage": {"input_tokens": 10, "output_tokens": 20},
                "returned_model": model, "finish_reason": "stop"}


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.sources = ["increment.py"]
        self.public = ["test_increment.py"]
        self.private = ["accept_increment.py"]
        self.write("increment.py", "def increment(value):\n    return value\n")
        self.write("test_increment.py", "import unittest\nfrom increment import increment\n"
                   "class Tests(unittest.TestCase):\n"
                   "    def test_public(self):\n        self.assertEqual(increment(1), 2)\n")
        self.write("accept_increment.py", "import unittest\nfrom increment import increment\n"
                   "PRIVATE_ACCEPTANCE_SENTINEL = 'held_back'\n"
                   "class Acceptance(unittest.TestCase):\n"
                   "    def test_negative(self):\n        self.assertEqual(increment(-3), -2)\n"
                   "    def test_general(self):\n        self.assertEqual(increment(9), 10)\n")
        self.directory = self.base / "run"
        self.directory.mkdir()
        self.pair_directory = self.directory / "case_0"
        config = {"schema_version": 1, "repository": "repo", "source_files": self.sources,
                  "test_files": self.public, "models": {"inspection": cli.MODELS["weak"], "editing": cli.MODELS["strong"]},
                  "limits": {"max_repairs": 1, "max_output_tokens": 2048, "max_requests": 3,
                             "request_budget": 8000, "minimum_interval_seconds": 60,
                             "request_timeout_seconds": 90, "test_timeout_seconds": 15},
                  "spending": {"account_is_free_tier": True, "max_provider_spend_usd": 0}}
        self.dump(self.base / "config.json", config)
        self.suite = {"schema_version": 1, "scope": "diagnostic", "tasks": [{
            "task_id": "case_0", "group_id": "increment", "partition": "development",
            "config": "config.json", "task": "Incrementa cualquier número en uno.",
            "acceptance_files": self.private}]}
        self.suite_path = self.base / "suite.json"
        self.dump(self.suite_path, self.suite)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, content):
        (self.root / name).write_text(content, encoding="utf-8")

    @staticmethod
    def dump(path, data):
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def collect(self, provider=None):
        self.provider = provider or Provider()
        self.expanded = cli.load_suite(self.suite_path)
        self.task = self.expanded["tasks"][0]
        self.pair = collect_pair(root=self.root, source_files=self.sources, test_files=self.public,
            task=self.task["task"], run_dir=self.pair_directory, provider=self.provider, timeout_seconds=10)
        self.assertEqual(self.pair["status"], "completed")
        return self.pair

    def grade(self, **kwargs):
        return grade_pair(self.root, self.sources, self.private, self.pair_directory, **kwargs)

    def replay(self):
        return load_acceptance(self.task, self.pair_directory, self.pair,
                               cli.digest(self.pair_directory / "acceptance.json"))

    def bind_run(self):
        self.dump(self.directory / "manifest.json", {"schema_version": 1, "suite": self.expanded,
                  "partition": "development", "policy_sha256": None})
        self.dump(self.directory / "journal.json", {
            "manifest_sha256": cli.digest(self.directory / "manifest.json"), "invocations": [],
            "tasks": {"case_0": {"state": "completed", "collection_sha256": cli.digest(self.pair_directory / "collection.json"),
                                    "acceptance_sha256": cli.digest(self.pair_directory / "acceptance.json")}}})

    def test_hidden_checks_reject_public_overfit_and_accept_fix_without_feedback(self):
        baseline = Workspace(self.root, self.base / "baseline", self.sources, self.private)
        baseline.prepare()
        self.assertEqual(baseline.verify(Path(sys.executable), 10)["status"], "test_failed")
        original = (self.root / "increment.py").read_bytes()
        self.collect()
        self.assertTrue(all(branch["success"] for branch in self.pair["branches"].values()))
        with patch.dict("os.environ", {"GROQ_API_KEY": "not-a-real-key"}):
            result = self.grade()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["branches"]["weak"]["success"])
        self.assertTrue(result["branches"]["strong"]["success"])
        self.assertEqual(len(self.provider.calls), 3)
        self.assertNotIn("PRIVATE_ACCEPTANCE_SENTINEL", json.dumps(self.provider.calls))
        for role in ("weak", "strong"):
            self.assertFalse((self.pair_directory / role / "worktree" / self.private[0]).exists())
        self.assertEqual((self.root / "increment.py").read_bytes(), original)
        self.assertEqual(self.replay(), result)
        self.bind_run()
        rows, _, _, _ = cli.load_run(self.directory, need_reference=False)
        self.assertTrue(rows[0]["branches"]["weak"]["visible_success"])
        self.assertFalse(rows[0]["branches"]["weak"]["success"])
        self.assertTrue(rows[0]["branches"]["strong"]["success"])
        self.assertTrue(rows[0]["independent_quality_validation"])

    def test_invalid_public_action_remains_failed_even_if_hidden_test_passes(self):
        self.write(self.private[0], "import unittest\nfrom increment import increment\n"
                   "class Tests(unittest.TestCase):\n"
                   "    def test_acceptance(self):\n        self.assertEqual(increment(1), 1)\n")
        self.collect(Provider(invalid_weak=True))
        result = self.grade()
        self.assertFalse(result["branches"]["weak"]["visible_success"])
        self.assertTrue(result["branches"]["weak"]["acceptance_success"])
        self.assertFalse(result["branches"]["weak"]["success"])
        self.replay()

    def test_zero_tests_are_unknown_and_block_comparison(self):
        self.write(self.private[0], "# No executable acceptance checks\n")
        self.collect()
        result = self.grade()
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["branches"]["weak"]["success"])
        self.bind_run()
        with self.assertRaisesRegex(ValueError, "veredicto completo"):
            cli.load_run(self.directory, need_reference=False)
        with self.assertRaisesRegex(ValueError, "ya existe"):
            self.grade()

    def test_missing_interpreter_is_unknown_infrastructure(self):
        self.collect()
        result = self.grade(python_executable=self.base / "missing-python.exe")
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(all(row["success"] is None for row in result["branches"].values()))

    def test_known_public_failures_remain_failures_when_private_runner_cannot_import_broken_code(self):
        class SyntaxFailureProvider(Provider):
            def complete(self, model, messages):
                reply = super().complete(model, messages)
                if "baseline" not in json.loads(messages[-1]["content"]):
                    answer = json.loads(reply["content"])
                    answer["edits"][0]["new_text"] = "return ("
                    reply["content"] = json.dumps(answer)
                return reply

        self.collect(SyntaxFailureProvider())
        self.assertTrue(all(branch["success"] is False for branch in self.pair["branches"].values()))
        result = self.grade()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["private_verdicts_complete"])
        for row in result["branches"].values():
            self.assertFalse(row["success"])
            self.assertIsNone(row["acceptance_success"])
            self.assertEqual(row["status"], "runner_error")
        self.assertEqual(self.replay(), result)
        self.bind_run()
        rows, _, _, _ = cli.load_run(self.directory, need_reference=False)
        self.assertTrue(all(branch["success"] is False for branch in rows[0]["branches"].values()))
        self.assertTrue(all(branch["acceptance_success"] is None for branch in rows[0]["branches"].values()))

    def test_missing_private_verdict_is_unknown_only_for_public_success(self):
        self.collect(Provider(invalid_weak=True))
        result = self.grade(python_executable=self.base / "missing-python.exe")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["branches"]["weak"]["success"])
        self.assertIsNone(result["branches"]["weak"]["acceptance_success"])
        self.assertIsNone(result["branches"]["strong"]["success"])

    def test_acceptance_children_do_not_inherit_provider_credentials(self):
        self.write(self.private[0], "import os, unittest\n"
                   "class Tests(unittest.TestCase):\n"
                   "    def test_environment(self):\n"
                   "        self.assertNotIn('GROQ_API_KEY', os.environ)\n"
                   "        self.assertNotIn('OPENAI_API_KEY', os.environ)\n")
        self.collect()
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-groq", "OPENAI_API_KEY": "test-openai"}):
            result = self.grade()
        self.assertTrue(all(row["success"] for row in result["branches"].values()))

    def test_replay_rejects_changed_final_source_private_copy_snapshot_or_verdict(self):
        self.collect()
        self.grade()
        self.bind_run()
        for relative in ("weak/worktree/increment.py", "acceptance/weak/inputs/accept_increment.py",
                         "acceptance/weak/execution/originals/accept_increment.py",
                         "acceptance/weak/execution/worktree/increment.py",
                         "acceptance/weak/execution/workspace_snapshot.json",
                         "acceptance/weak/execution/verification.json", "acceptance.json"):
            with self.subTest(relative=relative):
                path = self.pair_directory / relative
                before = path.read_bytes()
                path.write_bytes(before + b"\n ")
                try:
                    with self.assertRaises(ValueError):
                        cli.load_run(self.directory, need_reference=False)
                finally:
                    path.write_bytes(before)

    def test_missing_private_evidence_is_not_replaced_by_public_success(self):
        self.collect()
        self.grade()
        self.bind_run()
        journal = cli.read(self.directory / "journal.json")
        journal["tasks"]["case_0"].pop("acceptance_sha256")
        self.dump(self.directory / "journal.json", journal)
        with self.assertRaisesRegex(ValueError, "falta o cambió"):
            cli.load_run(self.directory, need_reference=False)

    def test_suite_freezes_private_inventory_and_rejects_scope_overlap(self):
        inventory = validate_acceptance(self.root, self.sources, self.public, self.private)
        self.assertEqual([row["path"] for row in inventory["files"]], self.private)
        self.assertFalse(inventory["files"][0]["editable"])
        first = cli.acquisition_contract(cli.load_suite(self.suite_path))
        self.write(self.private[0], (self.root / self.private[0]).read_text() + "\n# changed\n")
        self.assertNotEqual(first, cli.acquisition_contract(cli.load_suite(self.suite_path)))
        for invalid in ([], ["increment.py"], ["test_increment.py"], ["../escape.py"], "accept_increment.py"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_acceptance(self.root, self.sources, self.public, invalid)


if __name__ == "__main__":
    unittest.main()
