"""Offline continuation fixtures; no BFCL data, provider or private benchmark access."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relay_router.errors import BackendHTTPError
from routerbench.agent_evaluation import _seal, _unseal, _write, object_hash
from routerbench.bfcl_harness import (BFCLReply, BFCLAlwaysStrong, BFCLAlwaysWeak,
    BFCLReactiveCascade, UPSTREAM_COMMIT, prepare_bfcl_episodes, run_bfcl_episodes)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continue_bfcl_tool", ROOT / "tools/continue_bfcl.py")
CONTINUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTINUE)


class ContinuationRuntime:
    evaluations = 0
    def __init__(self, snapshot, entry):
        pass
    def native_tools(self):
        return []
    def evaluate(self, decoded, reference):
        type(self).evaluations += 1
        return {"valid": True}


class ContinuationGenerator:
    def __init__(self, config):
        self.limit = config["session_limits"]["max_tokens_per_model"]
        self.max_calls = config["session_limits"]["max_provider_calls"]
        self.calls = []
        self.stop_after = None
        self.failure = None
    def manifest(self):
        return {"kind": "fixture_session", "max_tokens_per_model": self.limit, "max_provider_calls": self.max_calls}
    def session_stop_reason(self):
        if len(self.calls) >= self.limit or (self.stop_after is not None and len(self.calls) >= self.stop_after):
            return {"reason": "fixture_session_budget"}
        return None
    def __call__(self, model, request):
        self.calls.append((model, request.messages[0]["content"]))
        if self.failure:
            raise self.failure
        return BFCLReply("done", usage={"input_tokens": 2, "output_tokens": 1})


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = self.root / "previous"
        self.models = ["fixture/cheap", "fixture/strong"]
        self.policies = [kind(self.models) for kind in (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade)]
        self.cases = [{"id": f"fixture-{index}", "source_group": f"group-{index}", "entry": {
            "id": f"fixture-{index}", "question": [[{"role": "user", "content": f"case-{index}"}]]},
            "reference": {"private": "fixture only"}} for index in range(3)]
        self.config = {"schema_version": 1, "run_id": "continuation-fixture", "execution_mode": "simulated",
            "snapshot_path": "fixture://only", "upstream_commit": UPSTREAM_COMMIT,
            "model_catalog": self.models, "max_calls_per_turn": 1, "max_total_generation_calls": 9,
            "harness": {"max_transport_retries": 0, "retry_delay_seconds": 0},
            "session_limits": {"max_tokens_per_model": 160000, "max_provider_calls": 250},
            "session_stop_on_infrastructure": True, "pricing": {}, "compute_rates": {}}
        self.config["generation_manifest"] = ContinuationGenerator(self.config).manifest()

    def parent(self, *, stop_after=1, failure=None):
        generate = ContinuationGenerator(self.config)
        generate.stop_after, generate.failure = stop_after, failure
        return run_bfcl_episodes(self.cases, self.policies, generate, self.previous,
            self.config, runtime_factory=ContinuationRuntime)

    def prepare(self, previous=None, name="next", limit=160000):
        return CONTINUE.prepare_continuation(previous or self.previous, self.root / name,
            max_session_tokens=limit, generator_factory=ContinuationGenerator, runtime_factory=ContinuationRuntime)

    def test_prepare_keeps_exact_suffix_split_and_never_generates_or_grades(self):
        old = self.parent()
        before = ContinuationRuntime.evaluations
        prepared = self.prepare()
        plan = prepared["plan"]
        self.assertEqual(plan["ordered_pending_pairs"], old["unattempted_pairs"])
        self.assertEqual([g["policies"] for g in plan["runs"]],
            [["always_weak", "cheap_then_escalate"], [p.name for p in self.policies]])
        self.assertEqual([g["case_ids"] for g in plan["runs"]], [["fixture-0"], ["fixture-1", "fixture-2"]])
        self.assertEqual(prepared["generate"].calls, [])
        self.assertEqual(ContinuationRuntime.evaluations, before)
        self.assertFalse((prepared["output"] / "summary.json").exists())
        self.assertEqual(plan["configuration"], self.config)
        self.assertIn(str(self.previous / "summary.json"), plan["parent"]["file_sha256"])

    def test_shared_generator_budget_spans_rectangles_and_next_day_keeps_remaining(self):
        old = self.parent()
        prepared = self.prepare(limit=3)
        result = CONTINUE.execute_continuation(prepared)
        self.assertEqual(len(prepared["generate"].calls), 3)
        self.assertEqual(result["completed_pairs"], old["unattempted_pairs"][:3])
        self.assertEqual(result["unattempted_pairs"], old["unattempted_pairs"][3:])
        self.assertEqual(len(result["run_directories"]), 2)
        self.assertTrue(result["session_stopped"])
        self.assertEqual(CONTINUE.execute_continuation(prepared), result)
        self.assertEqual(len(prepared["generate"].calls), 3)
        tomorrow = self.prepare(prepared["output"], name="tomorrow")
        self.assertEqual(tomorrow["plan"]["ordered_pending_pairs"], result["unattempted_pairs"])
        done = CONTINUE.execute_continuation(tomorrow)
        self.assertTrue(done["execution_complete"])
        all_done = old["completed_pairs"] + result["completed_pairs"] + done["completed_pairs"]
        self.assertEqual(len(all_done), 9)
        self.assertEqual(len({tuple(pair) for pair in all_done}), 9)

    def test_censored_parent_and_cross_rectangle_outage_are_never_retried(self):
        old = self.parent(failure=BackendHTTPError(status_code=401))
        self.assertIsNone(old["episodes"][0]["success"])
        prepared = self.prepare()
        prepared["generate"].failure = BackendHTTPError(status_code=401)
        result = CONTINUE.execute_continuation(prepared)
        self.assertEqual(result["generation_calls"], 1)
        self.assertNotIn(old["completed_pairs"][0], result["planned_pairs"])
        self.assertEqual(result["completed_pairs"], [["fixture-0", "always_weak"]])
        self.assertEqual(result["session_stop_reason"]["category"], "http")
        # Leave one policy in the first rectangle; its failure must stop the
        # next rectangle even though that subrun itself is complete.
        tomorrow = self.prepare(prepared["output"], name="tomorrow")
        tomorrow["generate"].failure = BackendHTTPError(status_code=401)
        final = CONTINUE.execute_continuation(tomorrow)
        self.assertEqual(final["generation_calls"], 1)
        self.assertEqual(final["completed_pairs"], [["fixture-0", "cheap_then_escalate"]])
        self.assertEqual(len(final["unattempted_pairs"]), 6)
        self.assertEqual(len(final["run_directories"]), 1)

    def test_no_pending_writes_zero_call_summary(self):
        self.parent(stop_after=None)
        prepared = self.prepare()
        self.assertEqual(prepared["plan"]["runs"], [])
        saved = _unseal(prepared["output"] / "summary.json")
        self.assertEqual(saved["generation_calls"], 0)
        self.assertTrue(saved["execution_complete"])
        self.assertEqual(prepared["generate"].calls, [])

    def test_unclosed_and_pending_parent_are_rejected_without_callback(self):
        generate = ContinuationGenerator(self.config)
        prepare_bfcl_episodes(self.cases, self.policies, self.previous, self.config,
            generate=generate, runtime_factory=ContinuationRuntime)
        with self.assertRaisesRegex(ValueError, "cerrado"):
            self.prepare()
        generate.failure = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_bfcl_episodes(self.cases, self.policies, generate, self.previous, self.config,
                runtime_factory=ContinuationRuntime)
        with self.assertRaisesRegex(ValueError, "cerrado"):
            self.prepare()
        self.assertEqual(len(generate.calls), 1)

    def test_tampered_summary_sources_or_generator_are_rejected(self):
        self.parent()
        with patch.object(CONTINUE, "_sources", return_value={}):
            with self.assertRaisesRegex(ValueError, "fuentes"):
                self.prepare()
        class ChangedGenerator(ContinuationGenerator):
            def manifest(self):
                return {**super().manifest(), "different": True}
        with self.assertRaisesRegex(ValueError, "generador"):
            CONTINUE.prepare_continuation(self.previous, self.root / "bad", generator_factory=ChangedGenerator,
                runtime_factory=ContinuationRuntime)
        summary = _unseal(self.previous / "summary.json")
        summary["unattempted_pairs"].insert(0, summary["completed_pairs"][0])
        _write(self.previous / "summary.json", _seal(summary))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_limits_new_output_and_non_suffix_are_rejected(self):
        self.parent()
        for limit in (0, 200001, True):
            with self.assertRaises(ValueError):
                self.prepare(limit=limit)
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "cola ordenada"):
            CONTINUE.split_pending(self.cases, [p.name for p in self.policies], [["fixture-0", "always_weak"]])

    def test_parent_mutation_after_prepare_blocks_before_any_generation(self):
        self.parent()
        prepared = self.prepare()
        path = self.previous / "summary.json"
        path.write_text(path.read_text()+" ", encoding="utf-8")
        with self.assertRaises(ValueError):
            CONTINUE.execute_continuation(prepared)
        self.assertEqual(prepared["generate"].calls, [])

    def test_cli_prepare_does_not_read_key_and_execute_restores_environment(self):
        self.parent()
        prepared = self.prepare()
        config = prepared["plan"]["configuration"]
        config.update(execution_mode="external", evaluation_split="development", independent_validation=False,
            spending={"account_is_free_tier": True, "max_provider_spend_usd": 0},
            model_catalog=["groq/fixture"], models_metadata={"groq/fixture": {"api_key_env": "GROQ_API_KEY"}})
        with patch.object(CONTINUE, "prepare_continuation", return_value=prepared), \
             patch.object(CONTINUE.getpass, "getpass", side_effect=AssertionError("No key in prepare")), \
             patch.object(CONTINUE, "execute_continuation") as execute:
            CONTINUE.main(["--previous", "ignored", "--output", "ignored"])
            execute.assert_not_called()
        def fail(_):
            self.assertEqual(os.environ.get("GROQ_API_KEY"), "fixture-only")
            raise RuntimeError("Fixture interruption")
        with patch.dict(os.environ, {}, clear=False), patch.object(CONTINUE, "prepare_continuation", return_value=prepared), \
             patch.object(CONTINUE.getpass, "getpass", return_value="fixture-only"), \
             patch.object(CONTINUE.sys.stdin, "isatty", return_value=True), \
             patch.object(CONTINUE, "execute_continuation", side_effect=fail):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "Fixture interruption"):
                CONTINUE.main(["--previous", "ignored", "--output", "ignored", "--execute"])
            self.assertNotIn("GROQ_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
