"""Offline diagnostic scope and exact EOL application checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eol_replay_tested", ROOT / "tools/replay_maintenance_line_endings.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class EOLReplayTests(unittest.TestCase):
    def test_audit_accepts_only_newline_equivalence_and_records_hashes(self):
        source = b"def f():\r\n    return 1\r\n"
        old, new = source.decode().replace("\r\n", "\n"), "def f():\n    return 2\n"
        audit, expected = replay.audit_eol_only({"module.py": source},
            [{"path": "module.py", "old_text": old, "new_text": new}])
        self.assertEqual(audit[0]["exact_occurrences"], 0)
        self.assertEqual(audit[0]["eol_equivalent_occurrences"], 1)
        self.assertEqual(expected["module.py"], replay.sha(source.replace(b"1", b"2")))
        self.assertNotIn("return 1", json.dumps(audit))

    def test_text_mismatch_ambiguity_mixed_and_exact_only_are_explicit_exclusions(self):
        examples = [(b"x\r\ny\r\n", "x\nz\n", "text_difference_beyond_line_endings"),
                    (b"x\r\nx\r\nx", "x\nx", "ambiguous_or_overlapping_match"),
                    (b"x\r\ny\n", "x\ny\n", "unsafe_mixed_eol_or_bare_cr"),
                    (b"x\r\ny\r\n", "x", "no_line_ending_discrepancy")]
        for source, old, reason in examples:
            with self.subTest(reason=reason), self.assertRaises(replay.DiagnosticExclusion) as caught:
                replay.audit_eol_only({"module.py": source}, [{"path": "module.py", "old_text": old, "new_text": "z"}])
            self.assertEqual(caught.exception.reason, reason)

    def test_public_and_private_replay_use_saved_response_without_changing_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, run, output = root / "repo", root / "run", root / "diagnostic"
            repo.mkdir(); output.mkdir()
            source = b"def add(a, b):\r\n    return a - b\r\n"
            public = b"import unittest\nfrom module import add\nclass Public(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(2, 3), 5)\n"
            private = public.replace(b"class Public", b"class Private").replace(b"add(2, 3), 5", b"add(-2, -3), -5")
            for name, data in (("module.py", source), ("test_public.py", public), ("acceptance_test.py", private)):
                (repo / name).write_bytes(data)
            branch = run / "case" / "weak"
            workspace = replay.Workspace(repo, branch, ["module.py"], ["test_public.py"])
            workspace.prepare()
            response = branch / "responses/call-000.json"
            response.parent.mkdir()
            replay.save(response, {"content": json.dumps({"summary": "Fix add", "edits": [{"path": "module.py",
                "old_text": source.decode().replace("\r\n", "\n"),
                "new_text": source.decode().replace("\r\n", "\n").replace("a - b", "a + b")}]})})
            original_report = {"status": "blocked", "error": {"type": "ValueError",
                "validation_message": "old_text debe coincidir exactamente una sola vez."},
                "total_tokens": 30, "usage_complete": True, "calls": [{"index": 0, "activity": "editing",
                    "state": "completed", "decision": {"selected_model": "groq/openai/gpt-oss-20b"},
                    "usage": {"input_tokens": 10, "output_tokens": 20}, "response_path": "responses/call-000.json",
                    "response_sha256": replay.digest(response)}]}
            replay.save(branch / "report.json", original_report)
            replay.save(run / "case/collection.json", {"branches": {"weak": {"report_sha256": replay.digest(branch / "report.json")}}})
            task = {"task_id": "case", "repository": str(repo), "config": {"source_files": ["module.py"], "test_files": ["test_public.py"]},
                    "inventory": {"files": [{"path": name, "sha256": replay.sha(data)} for name, data in (("module.py", source), ("test_public.py", public))]},
                    "acceptance_files": ["acceptance_test.py"], "acceptance_inventory": {"files": [{"path": "acceptance_test.py", "sha256": replay.sha(private)}]}}
            originals = {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()}
            result = replay.replay_case(run, output, task, "weak", Path(sys.executable))
            self.assertTrue(result["public_verification"]["passed"])
            self.assertTrue(result["private_verification"]["passed"])
            self.assertEqual(result["additional_provider_calls"], 0)
            self.assertFalse(result["original_outcome_relabelled"])
            self.assertFalse(result["counterfactual_cost_computed"])
            self.assertEqual(result["original_generation_consumption"]["total_tokens"], 30)
            self.assertEqual(originals, {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()})
            self.assertEqual((repo / "module.py").read_bytes(), source)


if __name__ == "__main__":
    unittest.main()
