"""Offline pricing coverage and immutability checks for the Groq pilot report."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("groq_pilot_report_tested", ROOT / "tools/report_groq_pilot.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
TARIFFS = report.read(report.DEFAULT_TARIFFS)
MODEL = "groq/openai/gpt-oss-120b"


def event(incoming=1000, outgoing=100, cached=200, **changes):
    return {"model": MODEL, "call_index": 0, "is_retry": False,
            "usage": {"input_tokens": incoming, "output_tokens": outgoing, "cached_input_tokens": cached}, **changes}


class GroqPilotReportTests(unittest.TestCase):
    def test_known_cache_is_counted_once_and_zero_usage_is_known(self):
        result = report.event_scenario(event(), TARIFFS)
        expected = (800 * .15 + 200 * .075 + 100 * .60) / 1_000_000
        self.assertAlmostEqual(result["paid_token_cost_usd_range"]["lower"], expected)
        self.assertEqual(result["paid_token_cost_usd_range"]["lower"], result["paid_token_cost_usd_range"]["upper"])
        self.assertEqual(report.event_scenario(event(0, 0, 0), TARIFFS)["paid_token_cost_usd_range"], {"lower": 0., "upper": 0.})

    def test_unknown_cache_produces_bounds_without_imputing_cache(self):
        original = event(cached=None)
        result = report.event_scenario(original, TARIFFS)
        self.assertEqual(result["paid_token_cost_usd_range"], {"lower": .000135, "upper": .00021})
        self.assertIsNone(result["cached_input_tokens"])
        self.assertIsNone(original["usage"]["cached_input_tokens"])

    def test_unknown_or_invalid_usage_and_identity_do_not_receive_prices(self):
        cases = [event(incoming=None), event(outgoing=None), event(incoming=True), event(cached=1001),
                 event(cached=-1), event(cached=True), event(backend_failure={"category": "model_identity"}),
                 event(model="groq/unknown-model")]
        for value in cases:
            with self.subTest(value=value):
                self.assertIsNone(report.event_scenario(value, TARIFFS)["paid_token_cost_usd_range"])

    def test_failed_retries_count_once_and_incomplete_policy_keeps_only_subtotal(self):
        episodes = [{"episode_id": "failed", "policy": "always_strong", "success": False,
                     "events": [event(), event(is_retry=True, call_index=1, error_type="InvalidAction")]},
                    {"episode_id": "unknown", "policy": "always_strong", "success": False,
                     "events": [event(incoming=None, outgoing=None)]}]
        original = deepcopy(episodes)
        row = report.build_scenario(episodes, TARIFFS)["policies"]["always_strong"]
        self.assertEqual((row["episodes"], row["successes"], row["attempts"], row["retry_attempts"]), (2, 0, 3, 1))
        self.assertEqual((row["priced_attempts"], row["unknown_attempts"]), (2, 1))
        self.assertIsNone(row["paid_token_cost_usd_range"])
        self.assertEqual(row["known_subtotal_paid_token_cost_usd_range"], {"lower": .00039, "upper": .00039})
        self.assertEqual(episodes, original)

    def test_complete_bounded_policy_remains_counterfactual_not_observed_cost(self):
        episodes = [{"episode_id": "one", "policy": "always_strong", "success": True,
                     "events": [event(cached=None), event(cached=0, call_index=1)]}]
        scenario = report.build_scenario(episodes, TARIFFS)
        row = scenario["policies"]["always_strong"]
        self.assertEqual(row["paid_token_cost_usd_range"], {"lower": .000345, "upper": .00042})
        self.assertEqual(row["bounded_unknown_cache_attempts"], 1)
        self.assertIsNone(row["observed_invoice_usd"])
        self.assertIsNone(row["full_operating_cost_usd"])
        self.assertFalse(scenario["roi_assessed"])

    def test_report_writes_only_new_artifacts_with_verified_provenance(self):
        token_metric = {"known_total": 1000, "known_calls": 1, "unknown_calls": 0}
        metric = {"successes": 1, "n_episodes": 1, "calls": 1, "tokens": {
            "input_tokens": token_metric, "output_tokens": {**token_metric, "known_total": 100}},
            "failure_diagnostics": {"semantic": {}, "infrastructure": {}},
            "routing_latency": {"p95_seconds": .001}}
        manifest = {"freeze_sha256": "fixture-freeze", "configuration": {"execution_mode": "external",
                    "predictive_policy": None, "independent_validation": False,
                    "generation_options": {"action_format": "native_tools"}}}
        summary = {"n_tasks": 1, "n_episodes": 3, "metrics": {"policies": {name: deepcopy(metric) for name in report.LABELS}}}
        episodes = [{"episode_id": "one", "policy": name, "success": True, "events": [event()]} for name in report.LABELS]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in report.ARTIFACTS:
                (directory / name).write_bytes(b"unchanged test evidence")
            note = directory / "development_note.txt"
            note.write_text("Historial técnico conservado; no nueva cohorte.", encoding="utf-8")
            before = {name: (directory / name).read_bytes() for name in report.ARTIFACTS}
            with patch.object(report, "audit_run", return_value=(manifest, {"records": []}, summary, episodes)) as audit:
                report_path = report.render_report(directory, context_note=note)
            audit.assert_called_once_with(directory.resolve())
            self.assertEqual(before, {name: (directory / name).read_bytes() for name in report.ARTIFACTS})
            provenance = json.loads((directory / "pilot_report_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["pilot_report_sha256"], report.digest(report_path))
            self.assertEqual(provenance["context_note_sha256"], report.digest(note))
            self.assertEqual(provenance["paid_token_scenario_sha256"], report.digest(directory / "paid_token_scenario.json"))
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("CONTRAFACTUAL", text)
            self.assertIn("Historial técnico conservado", text)
            self.assertIn("No se entrenó ni evaluó un predictor Relay v8", text)

    def test_failed_audit_never_writes_a_report(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(report, "audit_run", side_effect=ValueError("unfinished")):
            with self.assertRaises(ValueError):
                report.render_report(temp)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
