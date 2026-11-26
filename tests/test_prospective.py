"""Durability, bounded generation and hidden-reference checks with callbacks."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from routerbench.features import normalize_for_dedup
from routerbench.prospective import prepare_prospective, run_prospective


MANIFEST = {"candidate": {"artifact_sha256": "a"*64, "parameter": 20},
            "reference": {"artifact_sha256": "b"*64, "parameter": .5},
            "models": {"strong": "local-large", "weak": "local-small"}}


def dataset(root):
    rows = [{"id": "a", "prompt": "Name the project", "reference": "Bruma", "grade": "exact"},
            {"id": "b", "prompt": "State the approved budget", "reference": "17", "grade": "contains"}]
    path = root / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
    return path


def answer(model, prompt):
    return {"content": " BRUMA " if prompt == "Name the project" else "Approved: 17 million",
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "duration_seconds": .1}


class ProspectiveTests(unittest.TestCase):
    def test_freeze_precedes_all_callbacks_and_all_routes_precede_shared_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = dataset(root), root / "run"
            calls = []

            def route(prompt):
                self.assertTrue((output / "freeze.json").is_file())
                self.assertNotIn("Bruma", prompt)
                self.assertNotIn("17", prompt)
                return "weak"

            def generate(model, prompt):
                routes = json.loads((output / "routes.json").read_text())["payload"]
                self.assertEqual(len(routes), 2)
                self.assertTrue(all(set(row["routes"]) == {"candidate", "reference"} for row in routes))
                calls.append((model, prompt))
                self.assertNotIn("Bruma", prompt)
                self.assertNotIn("17", prompt)
                return answer(model, prompt)

            summary = run_prospective(path, output, MANIFEST, route, route, generate)
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["unique_generation_calls"], 2)
            self.assertEqual(summary["policies"]["candidate"]["by_grade"]["exact"]["success_rate"], 1)
            self.assertEqual(summary["policies"]["candidate"]["by_grade"]["contains"]["success_rate"], 1)
            self.assertNotIn("success_rate", summary["policies"]["candidate"])
            self.assertIsNone(summary["cost_usd"])
            self.assertFalse(summary["production_validation"])
            original = (output / "summary.json").read_bytes()
            never = Mock(side_effect=AssertionError("Completed callbacks must not run again"))
            self.assertEqual(run_prospective(path, output, MANIFEST, never, never, never), summary)
            never.assert_not_called()
            self.assertEqual((output / "summary.json").read_bytes(), original)

    def test_collect_both_models_shares_calls_with_fixed_baselines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate = Mock(side_effect=answer)
            summary = run_prospective(dataset(root), root / "run", MANIFEST,
                                      lambda _: "strong", lambda _: "strong", generate,
                                      collect_both_models=True)
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(set(summary["policies"]), {"candidate", "reference", "baseline_strong", "baseline_weak"})
            for policy in summary["policies"].values():
                self.assertEqual(policy["by_grade"]["exact"]["successes"], 1)
                self.assertIsNone(policy["cost_usd"])
            for prompt in {call.args[1] for call in generate.call_args_list}:
                self.assertEqual(sum(call.args[1] == prompt for call in generate.call_args_list), 2)

    def test_resume_does_not_reroute_or_retry_an_interrupted_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = dataset(root), root / "run"
            route = Mock(return_value="weak")
            first = Mock(side_effect=[answer("weak", "Name the project"), KeyboardInterrupt()])
            with self.assertRaises(KeyboardInterrupt):
                run_prospective(path, output, MANIFEST, route, route, first)
            self.assertEqual(route.call_count, 4)
            never = Mock(side_effect=AssertionError("No hidden retry"))
            summary = run_prospective(path, output, MANIFEST, never, never, never)
            never.assert_not_called()
            self.assertEqual(summary["policies"]["candidate"]["generation_failures"], 1)
            responses = json.loads((output / "responses.json").read_text())["payload"]
            self.assertIn("InterruptedGeneration", [record["error_type"] for record in responses.values()])

    def test_provider_errors_are_cached_failures_and_unknown_tokens_stay_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def failed(model, prompt):
                if model == "strong":
                    raise RuntimeError("Do not expose this provider detail")
                return {"content": "wrong"}
            generate = Mock(side_effect=failed)
            summary = run_prospective(dataset(root), root / "run", MANIFEST,
                                      lambda _: "strong", lambda _: "weak", generate)
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(summary["policies"]["candidate"]["generation_failures"], 2)
            self.assertEqual(summary["policies"]["reference"]["generation_failures"], 0)
            self.assertIsNone(summary["policies"]["reference"]["tokens"]["input_tokens"]["known_total"])
            self.assertNotIn("Do not expose", (root / "run/responses.json").read_text())

    def test_input_manifest_and_checkpoint_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = dataset(root), root / "run"
            prepare_prospective(path, output, MANIFEST)
            with self.assertRaises(ValueError):
                prepare_prospective(path, output, {"changed": True})
            with self.assertRaises(ValueError):
                prepare_prospective(path, output, MANIFEST, collect_both_models=True)
            path.write_text(path.read_text()+"\n")
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                prepare_prospective(path, output, MANIFEST)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = dataset(root), root / "run"
            run_prospective(path, output, MANIFEST, lambda _: "weak", lambda _: "weak", answer)
            saved = json.loads((output / "responses.json").read_text())
            next(iter(saved["payload"].values()))["content"] = "tampered"
            (output / "responses.json").write_text(json.dumps(saved))
            never = Mock()
            with self.assertRaises(ValueError):
                run_prospective(path, output, MANIFEST, never, never, never)
            never.assert_not_called()

    def test_schema_duplicates_overlap_and_prepare_only_import_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = dataset(root)
            group = hashlib.sha256(normalize_for_dedup("Name the project").encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "solapamiento"):
                prepare_prospective(path, root / "run", MANIFEST, {group})
            original = json.loads(path.read_text().splitlines()[0])
            for bad in (dict(original, extra="forbidden"), dict(original, grade="score"),
                        dict(original, reference="")):
                path.write_text(json.dumps(bad)+"\n")
                with self.assertRaises(ValueError):
                    prepare_prospective(path, root / "run", MANIFEST)
            path.write_text(json.dumps(original)+"\n"+json.dumps(dict(original, id="another", prompt=" NAME   THE PROJECT "))+"\n")
            with self.assertRaises(ValueError):
                prepare_prospective(path, root / "run", MANIFEST)
        import routerbench.prospective as module
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertFalse(any(name and any(token in name for token in ("requests", "urllib", "ollama", "live", "torch")) for name in imported))

    def test_completed_runs_with_missing_checkpoints_never_call_routers(self):
        for missing in ("routes.json", "responses.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, output = dataset(root), root / "run"
                run_prospective(path, output, MANIFEST, lambda _: "weak", lambda _: "weak", answer)
                (output / missing).unlink()
                never = Mock(side_effect=AssertionError("A completed run cannot reroute"))
                with self.assertRaisesRegex(ValueError, "checkpoints intactos"):
                    run_prospective(path, output, MANIFEST, never, never, never)
                never.assert_not_called()

    def test_partial_routing_resume_keeps_completed_route_and_input_change_blocks_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = dataset(root), root / "run"
            candidate = Mock(return_value="weak")
            with self.assertRaises(KeyboardInterrupt):
                run_prospective(path, output, MANIFEST, candidate, Mock(side_effect=KeyboardInterrupt()), answer)
            self.assertEqual(candidate.call_count, 1)
            candidate.reset_mock()
            run_prospective(path, output, MANIFEST, candidate, lambda _: "weak", answer)
            self.assertEqual(candidate.call_count, 1)  # Only the second case remained unrouted.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = dataset(root), root / "run"
            def changed(model, prompt):
                path.write_text(path.read_text()+"\n")
                return answer(model, prompt)
            with self.assertRaisesRegex(ValueError, "durante la evaluación"):
                run_prospective(path, output, MANIFEST, lambda _: "weak", lambda _: "weak", changed)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
