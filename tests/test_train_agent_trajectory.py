"""Offline CLI fixtures are synthetic, never evidence of routing performance."""
from contextlib import redirect_stdout
from dataclasses import fields
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

from test_agent_trajectory_learning import MODELS, rows
from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor
from train_agent_trajectory import main, parse_dataset, read_json


def dataset():
    data = {"schema_version": 2, "model_catalog": list(MODELS)}
    for part in ("fit", "calibration", "development"):
        data[part] = [{**{field.name: getattr(row, field.name) for field in fields(row) if field.name != "request"},
                       "request": row.request.to_dict()} for row in rows(part)]
    return data


class TrajectoryTrainingCommandTests(unittest.TestCase):
    def test_offline_training_writes_verified_artifact_and_complete_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path, artifact = Path(directory) / "labels.json", Path(directory) / "model.joblib"
            path.write_text(json.dumps(dataset()), encoding="utf-8")
            with patch("socket.create_connection") as connection, redirect_stdout(StringIO()) as output:
                result = main(["--data", str(path), "--output", str(artifact), "--max-text-features", "100"])
            connection.assert_not_called()
            self.assertEqual(json.loads(output.getvalue()), result)
            self.assertEqual(result["provider_calls"], 0)
            model = TrajectorySuccessPredictor.load(artifact, expected_sha256=result["artifact_sha256"])
            record = read_json(result["training_record"])
            self.assertEqual(record["data_file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["predictor"], model.manifest())
            self.assertEqual(record["partition_rows"], dict.fromkeys(("fit", "calibration", "development"), 48))
            self.assertFalse(record["threshold_selection_executed"])
            self.assertFalse(record["policy_thresholds_frozen"])
            self.assertFalse(record["independent_validation"])
            self.assertLessEqual(model.manifest()["features"]["text_vocabulary_size"], 100)
            self.assertEqual(model.manifest()["max_text_features"], 100)
            with self.assertRaisesRegex(ValueError, "development"):
                model.audit_task_groups([{"source_group": "development-0", "prompt": "unseen prompt"}])
            with redirect_stdout(StringIO()), self.assertRaises(FileExistsError):
                main(["--data", str(path), "--output", str(artifact)])

    def test_overlap_and_missing_cost_fail_before_artifact_is_written(self):
        for variant in ("overlap", "unknown_cost"):
            data = dataset()
            if variant == "overlap":
                data["development"][0]["source_group"] = data["fit"][0]["source_group"]
            else:
                for row in data["fit"]:
                    row["remaining_provider_cost_usd"] = None
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                path, artifact = Path(directory) / "labels.json", Path(directory) / "model.joblib"
                path.write_text(json.dumps(data), encoding="utf-8")
                with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                    main(["--data", str(path), "--output", str(artifact)])
                self.assertFalse(artifact.exists())
                self.assertFalse(artifact.with_name(artifact.name + ".manifest.json").exists())

    def test_schema_catalog_and_row_contract_are_strict(self):
        with self.assertRaisesRegex(ValueError, "catalog"):
            parse_dataset(dataset(), MODELS[::-1])
        for field in ("remaining_provider_cost_usd", "cost_basis", "success"):
            data = dataset()
            data["fit"][0].pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "no labels or costs are inferred"):
                parse_dataset(data)
        data = dataset()
        data["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema_version=2"):
            parse_dataset(data)
        data = dataset()
        data["test"] = []
        with self.assertRaises(ValueError):
            parse_dataset(data)

    def test_config_order_and_duplicate_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, config, artifact = (Path(directory) / name for name in ("labels.json", "config.json", "model.joblib"))
            path.write_text(json.dumps(dataset()), encoding="utf-8")
            config.write_text(json.dumps({"model_catalog": list(MODELS[::-1])}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "catalog"):
                main(["--data", str(path), "--config", str(config), "--output", str(artifact)])
            self.assertFalse(artifact.exists())
            path.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                main(["--data", str(path), "--output", str(artifact)])


if __name__ == "__main__":
    unittest.main()
