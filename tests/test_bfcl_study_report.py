"""Closed-session aggregation fixtures; no providers or benchmark data."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relay_router.errors import BackendHTTPError
from routerbench.agent_evaluation import _seal, _unseal, _write, object_hash
from routerbench.bfcl_harness import (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade,
    UPSTREAM_COMMIT, prepare_bfcl_episodes, run_bfcl_episodes)

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURE = load("bfcl_continuation_fixture", ROOT / "tests/test_continue_bfcl.py")
STUDY = load("bfcl_study_report", ROOT / "tools/report_bfcl_study.py")
CONTINUE = FIXTURE.CONTINUE


class StudyGenerator(FIXTURE.ContinuationGenerator):
    def manifest(self):
        return {**super().manifest(), "kind": "bfcl_session_budget"}


class BFCLStudyReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root / "parent"
        self.models = ["fixture/cheap", "fixture/strong"]
        self.policies = [kind(self.models) for kind in (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade)]
        self.cases = [{"id": f"fixture-{i}", "source_group": f"group-{i}", "entry": {
            "id": f"fixture-{i}", "question": [[{"role": "user", "content": f"case-{i}"}]]},
            "reference": {"private": "fixture only"}} for i in range(3)]
        self.config = {"schema_version": 1, "run_id": "study-fixture", "execution_mode": "simulated",
            "snapshot_path": "fixture://only", "upstream_commit": UPSTREAM_COMMIT, "model_catalog": self.models,
            "max_calls_per_turn": 1, "max_total_generation_calls": 9,
            "harness": {"max_transport_retries": 0, "retry_delay_seconds": 0},
            "session_limits": {"max_tokens_per_model": 160000, "max_provider_calls": 250},
            "session_stop_on_infrastructure": True, "pricing": {}, "compute_rates": {}}
        self.config["generation_manifest"] = StudyGenerator(self.config).manifest()
        self.tariffs = self.root / "tariffs.json"
        _write(self.tariffs, {"currency": "USD", "models": {model: {
            "input_usd_per_million": i+1., "output_usd_per_million": 2*(i+1.)}
            for i, model in enumerate(self.models)}})

    def start(self, failure=None):
        generator = StudyGenerator(self.config)
        generator.stop_after, generator.failure = 1, failure
        return run_bfcl_episodes(self.cases, self.policies, generator, self.parent, self.config,
            runtime_factory=FIXTURE.ContinuationRuntime)

    def continue_run(self, previous=None, *, name="next", limit=3):
        prepared = CONTINUE.prepare_continuation(previous or self.parent, self.root / name,
            max_session_tokens=limit, generator_factory=StudyGenerator, runtime_factory=FIXTURE.ContinuationRuntime)
        CONTINUE.execute_continuation(prepared)
        return prepared

    def test_parent_only_keeps_pending_population_unknown(self):
        self.start()
        report = STUDY.audit_study(self.parent, tariffs_path=self.tariffs)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["closed_pairs"], 1)
        self.assertEqual(len(report["unattempted_pairs"]), 8)
        self.assertEqual(report["policies"]["always_weak"]["known_failures"], 0)
        self.assertIsNone(report["policies"]["always_weak"]["operational_success_rate"])
        self.assertIsNone(report["paired_comparisons"]["always_strong__always_weak"]["tariff_no_cache_cost_ratio_second_to_first"])

    def test_accumulates_two_chains_once_and_pairs_across_session_boundaries(self):
        self.start()
        first = self.continue_run()
        second = self.continue_run(first["output"], name="tomorrow", limit=160000)
        before = FIXTURE.ContinuationRuntime.evaluations
        with patch.object(StudyGenerator, "__call__", side_effect=AssertionError("No generation")), \
             patch.object(FIXTURE.ContinuationRuntime, "evaluate", side_effect=AssertionError("No grading")):
            report = STUDY.audit_study(self.parent, [first["output"], second["output"]], tariffs_path=self.tariffs)
        self.assertEqual(FIXTURE.ContinuationRuntime.evaluations, before)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["closed_pairs"], 9)
        self.assertEqual(report["accounting"]["calls"], 9)
        self.assertEqual(report["accounting"]["input_tokens"], 18)
        self.assertAlmostEqual(report["accounting"]["tariff_no_cache_usd"], .000048)
        self.assertIsNone(report["accounting"]["routing_p95_seconds"])
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["closed_pairs"], 3)
        self.assertEqual(pair["uncensored_pairs"], 3)
        self.assertAlmostEqual(pair["tariff_no_cache_cost_ratio_second_to_first"], .5)
        self.assertEqual(pair["accounting_first"]["calls"], 3)
        self.assertIsNone(pair["accounting_first"]["routing_p95_seconds"])
        text = STUDY.render_study(report)
        self.assertIn("No se promedian p95", text)
        self.assertIn("reports/run-000.md", text)
        self.assertIn("no prueban el margen de dos puntos", text)

    def test_censored_attempt_and_missing_cost_retained_once(self):
        self.start(BackendHTTPError(status_code=401))
        next_run = self.continue_run(limit=160000)
        report = STUDY.audit_study(self.parent, [next_run["output"]], tariffs_path=self.tariffs)
        self.assertEqual(report["closed_pairs"], 9)
        self.assertEqual(report["policies"]["always_strong"]["censored_conversations"], 1)
        self.assertEqual(report["policies"]["always_strong"]["known_failures"], 0)
        self.assertIsNone(report["accounting"]["tariff_no_cache_usd"])
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["closed_pairs"], 3)
        self.assertEqual(pair["uncensored_pairs"], 2)
        self.assertIsNone(pair["tariff_no_cache_cost_ratio_second_to_first"])

    def test_duplicates_and_omitted_ancestors_are_rejected(self):
        self.start()
        first = self.continue_run()
        with self.assertRaisesRegex(ValueError, "mas de una"):
            STUDY.audit_study(self.parent, [first["output"], first["output"]])
        second = self.continue_run(first["output"], name="tomorrow", limit=160000)
        with self.assertRaisesRegex(ValueError, "antecesoras"):
            STUDY.audit_study(self.parent, [second["output"]])

    def test_unclosed_parent_or_continuation_is_rejected(self):
        generator = StudyGenerator(self.config)
        prepare_bfcl_episodes(self.cases, self.policies, self.parent, self.config,
            generate=generator, runtime_factory=FIXTURE.ContinuationRuntime)
        with self.assertRaisesRegex(ValueError, "cerrado"):
            STUDY.audit_study(self.parent)
        self.start()
        prepared = CONTINUE.prepare_continuation(self.parent, self.root / "next", max_session_tokens=3,
            generator_factory=StudyGenerator, runtime_factory=FIXTURE.ContinuationRuntime)
        with self.assertRaises((ValueError, FileNotFoundError)):
            STUDY.audit_study(self.parent, [prepared["output"]])

    def test_changed_scientific_configuration_or_reference_is_rejected(self):
        self.start()
        parent = STUDY.base._manifest(self.parent, STUDY.base.Snapshot())
        for mutate in (
            lambda m: m["configuration"]["harness"].update(max_transport_retries=1),
            lambda m: m["configuration"]["session_limits"].update(max_provider_calls=300),
            lambda m: m["configuration"]["generation_manifest"].update(max_provider_calls=300),
            lambda m: m["cases"][0]["reference"].update(private="changed"),
            lambda m: m["configuration"].update(run_id="undeclared change"),
        ):
            changed = deepcopy(parent)
            mutate(changed)
            with self.assertRaises(ValueError):
                STUDY._compare_manifests(parent, changed)

    def test_lineage_ancestor_list_cannot_be_resealed_away(self):
        self.start()
        prepared = self.continue_run()
        path = prepared["output"] / "continuation.json"
        plan = _unseal(path)
        plan["ancestor_run_directories"] = []
        _write(path, _seal(plan))
        summary_path = prepared["output"] / "summary.json"
        summary = _unseal(summary_path)
        summary["continuation_sha256"] = object_hash(plan)
        _write(summary_path, _seal(summary))
        with self.assertRaisesRegex(ValueError, "antecesores|pendientes del padre"):
            STUDY.audit_study(self.parent, [prepared["output"]])

    def test_cli_writes_new_sealed_reports_and_preserves_sources(self):
        self.start()
        before = {path.name: path.read_bytes() for path in self.parent.iterdir() if path.is_file()}
        output = self.root / "report"
        result = STUDY.main(["--parent", str(self.parent), "--tariffs", str(self.tariffs), "--output", str(output)])
        self.assertEqual(_unseal(output / "report.json")["closed_pairs"], result["closed_pairs"])
        self.assertTrue((output / "reports/run-000.md").is_file())
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.parent.iterdir() if path.is_file()})
        with self.assertRaises(FileExistsError):
            STUDY.main(["--parent", str(self.parent), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
