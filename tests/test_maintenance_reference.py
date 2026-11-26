"""Reference scoring contracts without model weights, network or provider calls."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relay_router.maintenance_reference import PINNED, _load_predictor, score_records


def prediction(**overrides):
    return {"probability": .25, "input_tokens": 600, "retained_tokens": 512,
            "truncated_tokens": 88, **overrides}


class Predictor:
    def __init__(self, result=None):
        self.result = [prediction()] if result is None else result
        self.calls = []

    def predict_batch(self, texts):
        self.calls.append(texts)
        return self.result


class MaintenanceReferenceTests(unittest.TestCase):
    def test_scores_original_text_once_per_record_and_audits_injected_origin(self):
        predictor = Predictor()
        text = "  def f():\n    return 1\n"
        inputs = [{"task_id": "b", "routing_text": text},
                  {"task_id": "a", "routing_text": "Second task"}]
        with patch("relay_router.maintenance_reference.time.perf_counter",
                   side_effect=[10., 10.5, 20., 20.75]):
            result = score_records(inputs, predictor)
        self.assertEqual(predictor.calls, [[text], ["Second task"]])
        self.assertEqual([row["task_id"] for row in result["rows"]], ["b", "a"])
        self.assertEqual([row["router_seconds"] for row in result["rows"]], [.5, .75])
        row = result["rows"][0]
        self.assertEqual(row["routing_text_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(row["input_characters"], len(text))
        self.assertEqual(row["truncated_tokens"], 88)
        self.assertFalse(result["reference"]["checkpoint_verified"])
        self.assertEqual(result["reference"]["provenance"], "injected_predictor_unverified")
        self.assertIsNone(result["reference"]["runtime_packages"])
        self.assertEqual(result["reference"]["load_seconds"], 0.)
        self.assertFalse(result["reference"]["threshold_selected"])
        self.assertNotIn("selected_model", row)
        self.assertEqual(inputs[0]["routing_text"], text)

    def test_default_predictor_loads_once_and_excludes_load_time_from_inference(self):
        predictor = Predictor()
        with patch("relay_router.maintenance_reference._load_predictor", return_value=predictor) as load:
            with patch("relay_router.maintenance_reference.time.perf_counter",
                       side_effect=[1., 4., 5., 5.25, 6., 6.5]):
                result = score_records([{"task_id": "a", "routing_text": "one"},
                                        {"task_id": "b", "routing_text": "two"}])
        load.assert_called_once_with()
        self.assertEqual(result["reference"]["load_seconds"], 3.)
        self.assertEqual([row["router_seconds"] for row in result["rows"]], [.25, .5])
        self.assertTrue(result["reference"]["checkpoint_verified"])
        self.assertEqual(len(predictor.calls), 2)

    def test_empty_input_does_not_load_or_claim_verified_weights(self):
        with patch("relay_router.maintenance_reference._load_predictor") as load:
            result = score_records([])
        load.assert_not_called()
        self.assertEqual(result["rows"], [])
        self.assertFalse(result["reference"]["checkpoint_verified"])
        self.assertEqual(result["reference"]["load_seconds"], 0.)

    def test_rejects_all_bad_inputs_before_inference_or_loading(self):
        valid = {"task_id": "a", "routing_text": "task"}
        bad_records = [None, {}, [None], [{}],
                       [{"task_id": "", "routing_text": "task"}],
                       [{"task_id": 1, "routing_text": "task"}],
                       [{"task_id": "b", "routing_text": " \n"}],
                       [{"task_id": "b", "routing_text": ["task"]}],
                       [valid, dict(valid)], [valid, {**valid, "outcome": True}],
                       [valid, {"task_id": "b", "routing_text": None}]]
        for records in bad_records:
            with self.subTest(records=records):
                predictor = Predictor()
                with self.assertRaises(ValueError):
                    score_records(records, predictor)
                self.assertEqual(predictor.calls, [])
                with patch("relay_router.maintenance_reference._load_predictor") as load:
                    with self.assertRaises(ValueError):
                        score_records(records)
                load.assert_not_called()

    def test_rejects_malformed_predictions_and_inconsistent_coverage(self):
        invalid = [[], [{}, {}], [{}], [None],
                   *[[prediction(probability=p)] for p in (math.nan, math.inf, -math.inf, -.01, 1.01, True, "0.5")],
                   [prediction(input_tokens=0)], [prediction(retained_tokens=0)],
                   [prediction(retained_tokens=513, truncated_tokens=87)],
                   [prediction(input_tokens=10, retained_tokens=11, truncated_tokens=0)],
                   [prediction(truncated_tokens=87)], [prediction(input_tokens=600.)],
                   [prediction(truncated_tokens=False)]]
        for result in invalid:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    score_records([{"task_id": "a", "routing_text": "task"}], Predictor(result))

    def test_probability_endpoints_are_valid_without_selecting_a_model(self):
        for value in (0., 1.):
            result = score_records([{"task_id": "a", "routing_text": "task"}],
                                   Predictor([prediction(probability=value)]))
            self.assertEqual(result["rows"][0]["probability"], value)
            self.assertNotIn("threshold", result["rows"][0])

    def test_invalid_predictor_fails_without_loading(self):
        with self.assertRaises(TypeError):
            score_records([], object())

    def test_changed_configuration_or_manifest_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            directory = root / PINNED["model_dir"]
            directory.mkdir(parents=True)
            config = root / "configs/routellm_reference.json"
            config.write_text(json.dumps({**PINNED, "model_dir": "../other"}), encoding="utf-8")
            with patch("relay_router.maintenance_reference.ROOT", root):
                with self.assertRaisesRegex(ValueError, "configuración"):
                    _load_predictor()
                config.write_text(json.dumps(PINNED), encoding="utf-8")
                (directory / "manifest.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifiesto"):
                    _load_predictor()


if __name__ == "__main__":
    unittest.main()
