"""Offline training command using labelled synthetic fixtures only."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from test_agent_learning import MODELS, rows
from routerbench.agent_learning import StateSuccessPredictor
from train_agent_v8 import main, parse_dataset, parse_request, read_json


def dataset():
    data = {"schema_version": 1, "model_catalog": list(MODELS)}
    for part in ("fit", "calibration", "development"):
        data[part] = [{"request": row.request.to_dict(), "model": row.model, "success": row.success,
                       "source_group": row.source_group, "target": row.target,
                       "continuation_policy": row.continuation_policy, "assignment": row.assignment}
                      for row in rows(part)]
    return data


class TrainingCommandTests(unittest.TestCase):
    def test_command_writes_verified_artifact_and_development_record_without_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            artifact = Path(directory) / "predictor.joblib"
            path.write_text(json.dumps(dataset()), encoding="utf-8")
            with patch("socket.create_connection") as connection, redirect_stdout(StringIO()) as output:
                result = main(["--data", str(path), "--output", str(artifact), "--min-samples-leaf", "2"])
            connection.assert_not_called()
            printed = json.loads(output.getvalue())
            self.assertEqual(printed, result)
            self.assertEqual(result["provider_calls"], 0)
            predictor = StateSuccessPredictor.load(artifact, expected_sha256=result["artifact_sha256"])
            record = read_json(result["training_record"])
            self.assertEqual(record["data_file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["predictor"], predictor.manifest())
            self.assertEqual(record["partition_rows"], dict.fromkeys(("fit", "calibration", "development"), 48))
            self.assertFalse(record["threshold_selection_executed"])
            with redirect_stdout(StringIO()), self.assertRaises(FileExistsError):
                main(["--data", str(path), "--output", str(artifact)])

    def test_holdout_leakage_fails_before_writing_artifact(self):
        data = dataset()
        data["development"][0]["source_group"] = data["fit"][0]["source_group"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            artifact = Path(directory) / "predictor.joblib"
            path.write_text(json.dumps(data), encoding="utf-8")
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "overlap"):
                main(["--data", str(path), "--output", str(artifact)])
            self.assertFalse(artifact.exists())
            self.assertFalse(artifact.with_name(artifact.name + ".manifest.json").exists())

    def test_catalog_order_must_match_data_and_frozen_config(self):
        with self.assertRaisesRegex(ValueError, "catalog"):
            parse_dataset(dataset(), MODELS[::-1])
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "labels.json"
            config = Path(directory) / "config.json"
            artifact = Path(directory) / "predictor.joblib"
            data_path.write_text(json.dumps(dataset()), encoding="utf-8")
            config.write_text(json.dumps({"model_catalog": list(MODELS[::-1])}), encoding="utf-8")
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "catalog"):
                main(["--data", str(data_path), "--config", str(config), "--output", str(artifact)])
            self.assertFalse(artifact.exists())

    def test_state_round_trip_rejects_private_fields_and_wrong_context_unit(self):
        payload = dataset()["fit"][0]["request"]
        self.assertEqual(parse_request(payload).to_dict(), payload)
        for field, value in (("private_reference", "secret"), ("context_length_unit", "tokens")):
            changed = deepcopy(payload)
            changed["state"][field] = value
            with self.assertRaises(ValueError):
                parse_request(changed)
        changed = deepcopy(payload)
        changed["gold"] = "secret"
        with self.assertRaises(ValueError):
            parse_request(changed)
        legacy = deepcopy(payload)
        legacy["state"] = {key: value for key, value in legacy["state"].items()
                           if key in {"step", "attempt", "previous_errors", "available_tools", "remaining_calls"}}
        self.assertEqual(parse_request(legacy).to_dict(), legacy)

    def test_missing_labels_partitions_and_json_duplicates_are_rejected(self):
        data = dataset()
        data["fit"][0].pop("success")
        with self.assertRaisesRegex(ValueError, "no labels are inferred"):
            parse_dataset(data)
        data = dataset()
        data["development"] = []
        with self.assertRaisesRegex(ValueError, "development"):
            parse_dataset(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"model_catalog":[],"model_catalog":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_json(path)


if __name__ == "__main__":
    unittest.main()
