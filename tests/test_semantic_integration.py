"""Small semantic-router integration tests without model weights or network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs
from routerbench.features import extract_features
from routerbench.learner import SuccessCostRouter
from routerbench.semantic_representation import SemanticPromptRepresentation


class SemanticMockEncoder:
    def __init__(self, spec):
        self.calls = []

    def encode(self, texts, use_cache=True):
        self.calls.append((list(texts), use_cache))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, :3] = [1.0, len(text), text.count("proof") + 1]
        return vectors


def semantic_mock_factory(spec):
    return SemanticMockEncoder(spec)


def learner_fixture():
    inputs = pd.DataFrame({"prompt_text": ["simple question" if index % 2 else "advanced proof reasoning" for index in range(40)]})
    scores = pd.DataFrame({"cheap": [index % 2 for index in range(40)], "strong": [1] * 40})
    costs = pd.DataFrame({"cheap": [0.001] * 40, "strong": [0.01] * 40})
    return inputs, scores, costs


def write_frozen_split_fixture(root):
    """Create an eight-row preparation with six train and two protected test rows."""
    prepared = root / "prepared"
    prepared.mkdir()
    texts = [f"Question {index}" for index in range(8)]
    rows = pd.DataFrame({
        "source_row": range(8),
        "sample_id": [str(index) for index in range(8)],
        "eval_name": ["fixture"] * 8,
        "prompt_group": [f"group-{index}" for index in range(8)],
        "split": ["train"] * 6 + ["test"] * 2,
    })
    rows.to_parquet(prepared / "rows.parquet", index=False)
    extract_features(texts).to_parquet(prepared / "features.parquet", index=False)
    manifest = {"dataset_sha256": "fixture-dataset-sha",
                "files_sha256": {name: sha256_file(prepared / name) for name in ("rows.parquet", "features.parquet")}}
    write_json(prepared / "manifest.json", manifest)
    reference = root / "reference"
    reference.mkdir()
    reference_rows = rows.iloc[:6].copy()
    reference_rows["development_split"] = ["fit", "fit", "calibration", "calibration", "validation", "unused_fit_pilot"]
    reference_rows.to_parquet(reference / "development_rows.parquet", index=False)
    reference_summary = {"dataset_sha256": manifest["dataset_sha256"],
                         "source_prepared_files_sha256": manifest["files_sha256"],
                         "files_sha256": {"development_rows.parquet": sha256_file(reference / "development_rows.parquet")}}
    write_json(reference / "summary.json", reference_summary)
    config = {
        "prepared_dir": "prepared", "reference_split_run": "reference", "seed": 42,
        "dataset": {"path": "not-read.pkl", "sha256": manifest["dataset_sha256"], "expected_models": 2},
        "internal_split": {"strategy": "prompt_hash", "fit_fraction": 0.70,
                           "calibration_fraction": 0.15, "validation_fraction": 0.15},
    }
    # Non-numeric outer-test outcomes catch accidental validation before the
    # loader slices the original table to its protected training population.
    raw = pd.DataFrame({
        "sample_id": rows["sample_id"], "eval_name": rows["eval_name"], "prompt": texts,
        "cheap": [0, 1, 0, 1, 0, 1, "DO_NOT_INSPECT", "DO_NOT_INSPECT"],
        "strong": [1, 1, 1, 1, 1, 1, "DO_NOT_INSPECT", "DO_NOT_INSPECT"],
        "cheap|total_cost": [0.001] * 6 + ["DO_NOT_INSPECT"] * 2,
        "strong|total_cost": [0.01] * 6 + ["DO_NOT_INSPECT"] * 2,
    })
    return config, raw, rows, reference_rows


class SemanticLearnerIntegrationTests(unittest.TestCase):
    def test_transformer_is_cloned_and_calibration_and_raw_prediction_work(self):
        inputs, scores, costs = learner_fixture()
        template = SemanticPromptRepresentation(mode="semantic_hybrid", encoder_spec={"fixture": True},
                                                encoder_factory=semantic_mock_factory)
        cloned = clone(template)
        self.assertIsNot(cloned, template)
        self.assertEqual(cloned.get_params()["encoder_spec"], {"fixture": True})
        router = SuccessCostRouter(representation="semantic_hybrid", feature_transformer=template)
        router.fit(inputs.iloc[:20], scores.iloc[:20], costs.iloc[:20], inputs.iloc[20:30], scores.iloc[20:30])
        self.assertIsNot(router.transformer_, template)
        self.assertFalse(hasattr(template, "mode_"))
        self.assertEqual(router.transformer_.n_features_out_, 398)
        self.assertEqual(router.transformer_.encoder_.calls[0], (inputs.iloc[:20]["prompt_text"].tolist(), True))
        self.assertEqual(router.transformer_.encoder_.calls[1], (inputs.iloc[20:30]["prompt_text"].tolist(), True))
        router.transformer_.use_cache = False
        probabilities, predicted_costs = router.predict_tables(inputs.iloc[30:])
        self.assertEqual(router.transformer_.encoder_.calls[-1], (inputs.iloc[30:]["prompt_text"].tolist(), False))
        self.assertEqual(probabilities.index.tolist(), list(range(30, 40)))
        self.assertEqual(probabilities.columns.tolist(), ["cheap", "strong"])
        self.assertTrue(probabilities.ge(0).all().all() and probabilities.le(1).all().all())
        self.assertTrue(np.isfinite(predicted_costs.to_numpy()).all())
        self.assertTrue(predicted_costs["strong"].gt(predicted_costs["cheap"]).all())
        self.assertEqual(router.fit_rows_, 20)
        self.assertEqual(router.calibration_rows_, 10)

    def test_outcomes_cannot_enter_semantic_training_or_prediction_inputs(self):
        inputs, scores, costs = learner_fixture()
        template = SemanticPromptRepresentation(mode="semantic", encoder_spec={}, encoder_factory=semantic_mock_factory)
        router = SuccessCostRouter(feature_transformer=template)
        with self.assertRaisesRegex(ValueError, "no permitidas"):
            router.fit(inputs.iloc[:20].assign(score=1), scores.iloc[:20], costs.iloc[:20],
                       inputs.iloc[20:30], scores.iloc[20:30])
        router.fit(inputs.iloc[:20], scores.iloc[:20], costs.iloc[:20], inputs.iloc[20:30], scores.iloc[20:30])
        for name in ("score", "cheap|total_cost", "eval_name"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "no permitidas"):
                router.predict_tables(inputs.iloc[30:].assign(**{name: 1}))


class FrozenReferenceIntegrationTests(unittest.TestCase):
    def test_loader_reuses_reference_partitions_without_inspecting_outer_test_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, raw, _, reference_rows = write_frozen_split_fixture(root)
            with patch("routerbench.development.load_verified_pickle", return_value=raw) as reader:
                inputs, scores, costs, metadata, _ = load_development_inputs(root, config)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(inputs.index.tolist(), list(range(6)))
            self.assertEqual(scores.index.tolist(), list(range(6)))
            self.assertEqual(costs.index.tolist(), list(range(6)))
            self.assertEqual(metadata["development_split"].tolist(), reference_rows["development_split"].tolist())
            self.assertFalse(set(metadata.index) & {6, 7})

    def test_altered_reference_file_is_rejected_by_its_saved_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, raw, _, _ = write_frozen_split_fixture(root)
            target = root / "reference/development_rows.parquet"
            target.write_bytes(target.read_bytes() + b"modified")
            with patch("routerbench.development.load_verified_pickle", return_value=raw), self.assertRaisesRegex(ValueError, "modificadas"):
                load_development_inputs(root, config)

    def test_outer_test_row_in_reference_is_rejected_even_with_consistent_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, raw, rows, reference_rows = write_frozen_split_fixture(root)
            target = root / "reference/development_rows.parquet"
            reference_rows.loc[5, ["source_row", "prompt_group"]] = rows.loc[6, ["source_row", "prompt_group"]].to_numpy()
            reference_rows.to_parquet(target, index=False)
            summary_path = root / "reference/summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["files_sha256"]["development_rows.parquet"] = sha256_file(target)
            write_json(summary_path, summary)
            with patch("routerbench.development.load_verified_pickle", return_value=raw), self.assertRaisesRegex(ValueError, "train reservado"):
                load_development_inputs(root, config)

    def test_reference_from_another_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, raw, _, _ = write_frozen_split_fixture(root)
            summary_path = root / "reference/summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source_prepared_files_sha256"]["features.parquet"] = "0" * 64
            write_json(summary_path, summary)
            with patch("routerbench.development.load_verified_pickle", return_value=raw), self.assertRaisesRegex(ValueError, "otra preparación"):
                load_development_inputs(root, config)


if __name__ == "__main__":
    unittest.main()
