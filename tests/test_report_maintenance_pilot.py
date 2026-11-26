"""Synthetic, offline coverage for maintenance delivery accounting and rendering."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("maintenance_pilot_report_tested", ROOT / "tools/report_maintenance_pilot.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

WEAK, STRONG = "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"
TARIFFS = {"models": {WEAK: {"input_usd_per_million": 1, "output_usd_per_million": 2},
                      STRONG: {"input_usd_per_million": 3, "output_usd_per_million": 4}}}


def call(model=WEAK, activity="editing", unknown=False):
    return {"model": model, "activity": activity, "state": "completed",
            "usage": {} if unknown else {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}}


class MaintenancePilotReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.current, self.prior, self.additional = (self.base / name for name in ("v3", "v1", "v2"))
        self.tasks = [{"task_id": f"case_{i}", "group_id": f"group_{i}", "partition": "development"} for i in range(8)]
        self.rows = []
        for directory in (self.current, self.prior, self.additional):
            directory.mkdir()
            manifest = {"fixture_run": directory.name, "partition": "development",
                        "suite": {"tasks": self.tasks, "tariffs": TARIFFS, "tariffs_sha256": "synthetic_tariffs"},
                        "analysis_plan": {"operational_mode": "verified", "reference_threshold": .5,
                                          "inspection_mode": "direct" if directory == self.current else "model"}}
            self.write(directory / "manifest.json", manifest)
            journal = {"manifest_sha256": report.digest(directory / "manifest.json"), "tasks": {},
                       "invocations": [{"calls_started": 0}]}
            for i, task in enumerate(self.tasks):
                if directory != self.current and i:
                    continue
                location = directory / task["task_id"]
                location.mkdir()
                if directory == self.current:
                    weak_calls = [call()] + ([call(STRONG, "repair")] if i < 3 else [])
                    strong_calls = [call(STRONG)]
                    calls = weak_calls + strong_calls
                    branches = {"weak": {"calls": weak_calls}, "strong": {"calls": strong_calls}}
                    row = {**task, "routing_text": "Synthetic fixture only", "reference_probability": .25 if i % 2 else .75,
                           "relay_choice": "weak", "independent_quality_validation": True, "branches": {}}
                    for role, branch in branches.items():
                        summary = report.summarize_calls(branch["calls"], TARIFFS)
                        row["branches"][role] = {"success": True, "visible_success": True, "acceptance_success": True,
                            "total_tokens": summary["total_tokens"], "cost_usd": summary["total_scenario_cost_usd"]}
                    self.rows.append(row)
                else:
                    calls = [call() for _ in range(3 if directory == self.prior else 10)]
                    calls[-1] = call(unknown=True)
                    branches = {}
                pair = {"fixture_run": directory.name, "task_id": task["task_id"], "branches": branches,
                        "acquisition": {"calls": calls, "calls_started": len(calls)}}
                self.write(location / "collection.json", pair)
                journal["tasks"][task["task_id"]] = {"state": "completed" if directory == self.current else "incomplete",
                    "collection_sha256": report.digest(location / "collection.json")}
                journal["invocations"][0]["calls_started"] += len(calls)
            self.write(directory / "journal.json", journal)
        self.diagnostic = self.base / "original-diagnostic.json"
        self.additional_diagnostic = self.base / "request-20b.json"
        self.write_diagnostic(self.diagnostic, WEAK)
        # The intentionally misleading filename must never decide the model.
        self.write_diagnostic(self.additional_diagnostic, STRONG.removeprefix("groq/"))

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_diagnostic(self, path, model):
        self.write(path, {"kind": "explicit_single_request_diagnostic", "model": model,
                         "not_part_of_comparison": True, "automatic_retry": False, "status": "completed",
                         "response": {"usage": call()["usage"], "content": "PRIVATE_RESPONSE_NOT_FOR_REPORT"}})

    def build(self, **kwargs):
        return report.build_report(self.current, self.prior, self.diagnostic,
                                   kwargs.get("additional_runs", [self.additional]),
                                   kwargs.get("additional_diagnostics", [self.additional_diagnostic]))

    def incomplete(self):
        journal = report.read(self.current / "journal.json")
        journal["tasks"]["case_7"]["state"] = "incomplete"
        self.write(self.current / "journal.json", journal)

    def test_all_prior_runs_and_diagnostics_count_once_with_unknown_usage(self):
        self.incomplete()
        result = self.build()
        ledger = result["acquisition"]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual([ledger[name]["registered_calls"] for name in ("current_run", "prior_run", "separate_diagnostic")],
                         [19, 3, 1])
        self.assertEqual(ledger["additional_prior_runs"][0]["registered_calls"], 10)
        self.assertEqual(ledger["additional_diagnostics"][0]["registered_calls"], 1)
        self.assertEqual(ledger["all_research"]["registered_calls"], 34)
        self.assertEqual(ledger["all_research"]["known_token_subtotal"], 960)
        self.assertEqual(ledger["all_research"]["unknown_token_call_count"], 2)
        self.assertEqual(ledger["all_research"]["unknown_cost_call_count"], 2)
        self.assertIsNone(ledger["all_research"]["total_tokens"])
        self.assertIsNone(ledger["all_research"]["total_scenario_cost_usd"])
        markdown = report.render(result)
        for label in ("Ejecución previa 1 (v1)", "Ejecución previa 2 (v2)",
                      f"Diagnóstico separado 1 ({WEAK})", f"Diagnóstico separado 2 ({STRONG})"):
            self.assertEqual(markdown.count(label), 1)
        self.assertNotIn("%", markdown)
        self.assertNotIn("PRIVATE_RESPONSE_NOT_FOR_REPORT", json.dumps(result))

    def test_diagnostic_model_metadata_fallbacks_ignore_filename(self):
        record = report.read(self.additional_diagnostic)
        for metadata in ({"model": STRONG}, {"requested_model": STRONG},
                         {"model": STRONG, "returned_model": None}):
            with self.subTest(metadata=metadata):
                changed = {k: v for k, v in record.items() if k != "model"}
                changed.update(metadata)
                self.write(self.additional_diagnostic, changed)
                summary, _ = report.diagnostic_acquisition(self.additional_diagnostic, TARIFFS)
                self.assertEqual(summary["calls"][0]["model"], STRONG)
                self.assertAlmostEqual(summary["total_scenario_cost_usd"], .00011)
        record["response"]["returned_model"] = STRONG
        record["model"] = None
        self.write(self.additional_diagnostic, record)
        summary, _ = report.diagnostic_acquisition(self.additional_diagnostic, TARIFFS)
        self.assertEqual(summary["calls"][0]["model"], STRONG)

    def test_repeated_run_aliases_or_copies_are_rejected(self):
        for directories in ([self.prior], [self.current], [self.additional, self.additional / ".." / "v2"]):
            with self.subTest(directories=directories), self.assertRaises(ValueError):
                self.build(additional_runs=directories)
        copied = self.base / "copied-v2"
        shutil.copytree(self.additional, copied)
        with self.assertRaisesRegex(ValueError, "misma adquisición"):
            self.build(additional_runs=[self.additional, copied])

    def test_repeated_diagnostic_aliases_or_copies_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "diagnóstico está repetido"):
            self.build(additional_diagnostics=[self.diagnostic.parent / "." / self.diagnostic.name])
        copied = self.base / "copied-diagnostic.json"
        copied.write_bytes(self.additional_diagnostic.read_bytes())
        with self.assertRaisesRegex(ValueError, "mismo diagnóstico"):
            self.build(additional_diagnostics=[self.additional_diagnostic, copied])

    def test_complete_report_preserves_primary_secondary_and_direct_contract(self):
        from relay_router.maintenance_comparison import compare_records
        manifest, journal = (report.read(self.current / name) for name in ("manifest.json", "journal.json"))
        reference = {"reference": {"checkpoint_verified": True, "implementation": "synthetic_fixture"},
                     "rows": [{"task_id": row["task_id"], "probability": row["reference_probability"],
                               "input_tokens": 600, "retained_tokens": 512, "truncated_tokens": 88,
                               "router_seconds": .01} for row in self.rows]}
        policy = {"name": "relay_verified_cascade_v1", "frozen": True, "learned": False}
        comparison = compare_records(self.rows, .5, policy)
        comparison.update(reference=reference["reference"], analysis_plan=manifest["analysis_plan"],
                          reference_selection={"selection_scope": "fixed_before_acquisition"},
                          incomplete_or_unstarted_tasks=0, acquisition_calls=19)
        comparison["development_calibrated_reference"] = {
            "selection": {"threshold": .8, "selection_scope": "development_only_empirical_not_noninferiority"},
            "comparison": compare_records(self.rows, .8, policy)["partitions"]["development"]}
        self.write(self.current / "comparison.json", comparison)
        self.write(self.current / "reference.json", reference)
        frozen = report.digest(self.current / "comparison.json")
        with patch.object(report, "load_run", return_value=(self.rows, manifest, journal, reference)):
            result = self.build()
            with patch("builtins.print"):
                code = report.main(["--run", str(self.current), "--prior-run", str(self.prior),
                    "--additional-prior-run", str(self.additional), "--diagnostic", str(self.diagnostic),
                    "--additional-diagnostic", str(self.additional_diagnostic)])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["primary"]["paired_comparisons"], comparison["partitions"]["development"]["paired_comparisons"])
        self.assertEqual(result["development_calibrated_reference"]["selection"]["threshold"], .8)
        markdown = report.render(result)
        self.assertEqual(markdown.count("## Comparación primaria:"), 1)
        self.assertEqual(markdown.count("## Comparación secundaria:"), 1)
        self.assertEqual(markdown.count("| Política | Públicas | Privadas |"), 2)
        self.assertIn("no hay llamada de inspección", markdown)
        self.assertNotIn("Todas las trayectorias incluyen inspección", markdown)
        self.assertNotIn("Solo la segunda ejecución", markdown)
        self.assertEqual(frozen, report.digest(self.current / "comparison.json"))
        self.assertTrue((self.current / "delivery_report.md").exists())


if __name__ == "__main__":
    unittest.main()
