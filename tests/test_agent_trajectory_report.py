"""Read-only report tests use synthetic continuations, never providers."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from relay_router.agent import ModelReply
from relay_router.errors import BackendTransportError
from routerbench.agent_evaluation import _seal, _unseal, _write, object_hash
from routerbench.agent_rollouts import initial_anchors, collect_paired_rollouts, select_followup_anchors
from routerbench.agent_simulation import ScriptedToolFixture
from routerbench.agent_tasks import build_tasks

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trajectory_report_tool", ROOT / "tools/report_agent_trajectory.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class AcquisitionReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT / "configs/agent_benchmark_v1.json").read_text())
        self.models = ["simulated/weak", "simulated/strong"]
        self.config.update(schema_version=2, model_catalog=self.models,
            harness={"contract_version": 2, "max_transport_retries": 0, "retry_delay_seconds": 0}, bootstrap_replicates=100)
        self.splits = {part: [build_tasks(seed=900+i, per_family=1)[1]]
                       for i, part in enumerate((*REPORT.PARTITIONS, "evaluation"))}
        self.plan = {"schema_version": 1, "models": self.models, "decision_indices": [0, 2],
            "task_splits_sha256": object_hash(self.splits), "partitions": {k: len(v) for k, v in self.splits.items()},
            "reserved_partition_executed": False, "independent_validation": False}
        _write(self.root / "plan.json", _seal(self.plan))
        _write(self.root / "task_splits.json", self.splits)
        self.tariffs = {"purpose": "fictitious fixture", "models": {model: {
            "input_usd_per_million": i+1., "output_usd_per_million": (i+1.)*2} for i, model in enumerate(self.models)}}

    def collect(self, part="fit", stage="initial", generate=None):
        config = {**self.config, "data_partition": part}
        tasks = self.splits[part]
        anchors = (initial_anchors(tasks, config) if stage == "initial" else
                   select_followup_anchors(self.root / part / "initial", (2,), self.models[0]))
        if not anchors["anchors"]:
            return None, {"partition": part, "stage": stage, "unavailable": anchors["unavailable"]}
        path = self.root / part / stage
        result = collect_paired_rollouts(tasks, anchors, self.models, generate or ScriptedToolFixture(), config,
            path, max_generation_calls=64, tariffs=self.tariffs, execute=True)
        record = {"partition": part, "stage": stage, "directory": str(path.resolve()),
            "freeze_sha256": result["collection_freeze_sha256"], "generation_calls": result["generation_calls"],
            "censored_branches": len(result["censored"]),
            "excluded_censored_pairs": 1 if result["censored"] else 0, "unavailable": result["unavailable"]}
        return result, record

    def finish(self, generate=None):
        collections, observed, results = [], {}, []
        for part in REPORT.PARTITIONS:
            observed[part] = []
            for stage in REPORT.STAGES:
                result, record = self.collect(part, stage, generate)
                collections.append(record)
                if result:
                    results.append(result)
                    if not result["censored"]:
                        observed[part].extend(result["rows"])
        support = {part: {model: dict(REPORT.Counter(str(row["success"]).lower() for row in observed[part]
                    if row["model"] == model)) for model in self.models} for part in REPORT.PARTITIONS}
        data = {"schema_version": 2, "model_catalog": self.models, **observed}
        audit = {"plan_sha256": object_hash(self.plan), "collections": collections,
                 "excluded_censored_pairs": sum(c.get("excluded_censored_pairs", 0) for c in collections),
                 "reserved_partition_executed": False, "independent_validation": False, "class_support": support,
                 "observations_sha256": object_hash(data), "training_class_support_ready": False}
        _write(self.root / "observations.json", data)
        _write(self.root / "acquisition_audit.json", audit)
        return results

    def test_completed_report_reconstructs_data_and_counts_new_calls_once(self):
        results = self.finish()
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = REPORT.audit_acquisition(self.root)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["accounting"]["calls"], sum(r["generation_calls"] for r in results))
        self.assertEqual(report["accounting"]["input_tokens"], 100 * report["accounting"]["calls"])
        self.assertEqual(report["accounting"]["output_tokens"], 20 * report["accounting"]["calls"])
        self.assertIsNone(report["accounting"]["reported_provider_usd"])
        self.assertFalse(report["training_class_support_ready"])
        self.assertEqual(report["training_rows"], {p: 4 for p in REPORT.PARTITIONS})
        self.assertEqual(report["class_support"]["fit"][self.models[0]], {"true": 2})
        text = REPORT.render_report(report)
        self.assertIn("el probe fuerte no es la politica Siempre Fuerte", text)
        self.assertIn("No se calcula ROI", text)
        self.assertNotIn("ahorro", text)
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_partial_has_no_fabricated_unstarted_outcomes(self):
        with self.assertRaisesRegex(ValueError, "incompleta"):
            REPORT.audit_acquisition(self.root)
        report = REPORT.audit_acquisition(self.root, allow_partial=True)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["accounting"]["calls"], 0)
        self.assertEqual(report["branches_closed"], 0)
        self.assertEqual(report["training_rows"], {p: 0 for p in REPORT.PARTITIONS})
        self.assertTrue(all(not counts for row in report["class_support"].values() for counts in row.values()))

    def test_pending_attempt_counts_unknown_usage_and_no_outcome(self):
        def interrupted(model, request):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.collect(generate=interrupted)
        report = REPORT.audit_acquisition(self.root, allow_partial=True)
        self.assertEqual(report["accounting"]["calls"], 1)
        self.assertEqual(report["pending_calls"], 1)
        self.assertEqual(report["branches_closed"], 0)
        self.assertIsNone(report["accounting"]["input_tokens"])
        self.assertIsNone(report["accounting"]["tariff_scenario_usd"])
        self.assertEqual(report["training_rows"]["fit"], 0)
        self.assertFalse(any("operational_success" in b for b in report["collections"][0]["branches"]))

    def test_completed_transport_censorship_keeps_costs_and_excludes_entire_pair(self):
        def transport(model, request):
            raise BackendTransportError()
        self.collect(generate=transport)
        report = REPORT.audit_acquisition(self.root, allow_partial=True)
        self.assertEqual(report["censored_branches"], 2)
        self.assertEqual(report["excluded_censored_pairs"], 1)
        self.assertEqual(report["accounting"]["calls"], 2)
        self.assertEqual(report["training_rows"]["fit"], 0)
        self.assertIsNone(report["accounting"]["tariff_scenario_usd"])

    def test_recovered_interruption_is_reported_without_reissuing_or_followup(self):
        calls = []
        fixture = ScriptedToolFixture()
        def generate(model, request):
            calls.append(model)
            if len(calls) == 1:
                raise KeyboardInterrupt()
            return fixture(model, request)
        with self.assertRaises(KeyboardInterrupt):
            self.collect(generate=generate)
        self.collect(generate=generate)
        before = len(calls)
        report = REPORT.audit_acquisition(self.root, allow_partial=True)
        self.assertEqual(len(calls), before)
        self.assertEqual(report["censored_branches"], 1)
        self.assertEqual(report["training_rows"]["fit"], 0)
        self.assertEqual(report["stages"][1]["status"], "blocked_source_interruption")

    def test_real_negative_labels_and_unavailable_positions_are_preserved(self):
        def final_wrong(model, request):
            return ModelReply('{"type":"final","answer":"wrong"}', usage={"input_tokens": 20, "output_tokens": 3})
        self.finish(final_wrong)
        report = REPORT.audit_acquisition(self.root)
        self.assertEqual(report["accounting"]["calls"], 6)
        self.assertEqual(sum(s["status"] == "unavailable" for s in report["stages"]), 3)
        self.assertTrue(all(counts == {"false": 1} for row in report["class_support"].values() for counts in row.values()))

    def test_changed_top_level_labels_are_not_trusted_even_with_matching_hash(self):
        self.finish()
        data_path = self.root / "observations.json"
        data = json.loads(data_path.read_text())
        data["fit"][0]["success"] = False
        _write(data_path, data)
        audit_path = self.root / "acquisition_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["observations_sha256"] = object_hash(data)
        _write(audit_path, audit)
        with self.assertRaisesRegex(ValueError, "no se reproducen"):
            REPORT.audit_acquisition(self.root)

    def test_modified_branch_dataset_is_rejected(self):
        self.collect()
        path = self.root / "fit/initial/dataset.json"
        data = _unseal(path)
        data["rows"][0]["success"] = False
        _write(path, _seal(data))
        with self.assertRaisesRegex(ValueError, "dataset no reproduce"):
            REPORT.audit_acquisition(self.root, allow_partial=True)

    def test_missing_snapshot_and_reserved_execution_are_rejected(self):
        self.collect()
        (self.root / "fit/initial/source_bundle.zip").unlink()
        with self.assertRaises(FileNotFoundError):
            REPORT.audit_acquisition(self.root, allow_partial=True)
        (self.root / "evaluation").mkdir()
        with self.assertRaisesRegex(ValueError, "evaluation"):
            REPORT.audit_acquisition(self.root, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
