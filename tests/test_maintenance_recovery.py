"""Forensic continuation uses only missing actions and preserves prior evidence."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from relay_router import maintenance_competition as cli
from relay_router.maintenance_provider import MaintenanceProviderError
from test_maintenance_collection import Provider as ScriptedProvider, edit
from test_maintenance_competition import MaintenanceCompetitionTests, Provider
from tools.relay_resume_maintenance_pair import recover, preflight, _tree_hashes


class StructuredProvider(Provider):
    output_mode = "prompt_json"

    def __init__(self):
        super().__init__()
        self.schemas = []

    def complete_structured(self, model, messages, response_schema):
        self.schemas.append(response_schema)
        return self.complete(model, messages)


class MaintenanceRecoveryTests(unittest.TestCase):
    setUp = MaintenanceCompetitionTests.setUp
    tearDown = MaintenanceCompetitionTests.tearDown
    write = staticmethod(MaintenanceCompetitionTests.write)
    configure = MaintenanceCompetitionTests.configure

    def test_only_missing_strong_and_final_pair_run_without_mutating_original(self):
        self.configure(("development", "development"))
        self.suite["tasks"][0]["task_id"] = "decimal_allocation"
        self.suite["tasks"][1]["task_id"] = "scoped_posix_paths"
        self.write(self.suite_path, self.suite)
        old = ScriptedProvider([MaintenanceProviderError("truncated", category="output_truncated",
            usage={"input_tokens": 10, "output_tokens": 2048}), edit(old="missing original text")])
        old.output_mode = "prompt_json"
        with redirect_stdout(io.StringIO()), patch("relay_router.maintenance_collection._semantic_outcome", return_value=None):
            cli.collect(self.suite_path, self.directory, "development", 6, provider=old,
                analysis_plan={"inspection_mode": "direct", "output_mode": "prompt_json",
                               "operational_mode": "verified", "reference_threshold": .5})
        before = _tree_hashes(self.directory)
        self.assertEqual(len(old.calls), 2)
        self.assertEqual(cli.read(self.directory / "journal.json")["tasks"]["decimal_allocation"]["state"], "incomplete")
        destination = self.base / "derived"
        check = preflight(self.directory, destination, prior_calls=0)
        self.assertEqual(check["observed_calls"], 2)
        new = StructuredProvider()
        with redirect_stdout(io.StringIO()):
            recover(self.directory, destination, prior_calls=0, provider=new, progress=None)
        self.assertEqual(len(new.calls), 3)
        self.assertEqual(len(new.schemas), 3)
        self.assertEqual(new.calls[0][0], cli.MODELS["strong"])
        self.assertEqual(new.calls[0][1], old.calls[0][1])
        self.assertEqual(_tree_hashes(self.directory), before)
        pair = cli.read(destination / "decimal_allocation/collection.json")
        self.assertIs(pair["branches"]["weak"]["success"], False)
        self.assertIs(pair["branches"]["strong"]["success"], True)
        self.assertEqual(pair["acquisition"]["calls_started"], 3)
        self.assertEqual(cli.digest(destination / "decimal_allocation/weak/report.json"),
                         cli.digest(self.directory / "decimal_allocation/weak/report.json"))
        audit = cli.read(destination / "recovery-audit.json")
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(audit["new_calls_started"], 3)
        self.assertEqual(audit["total_calls_after"], 5)
        self.assertTrue(audit["original_run_unchanged"])
        rows, _, journal, _ = cli.load_run(destination, need_reference=False)
        self.assertEqual(len(rows), 2)
        self.assertEqual([entry["calls_started"] for entry in journal["invocations"]], [2, 3])
        with self.assertRaises(ValueError):
            preflight(self.directory, destination, prior_calls=0)
        with self.assertRaises(ValueError):
            preflight(self.directory, self.base / "over-budget", prior_calls=33)


if __name__ == "__main__":
    unittest.main()
