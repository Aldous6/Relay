"""Local adapter checks use generated fixtures only, never native LLM calls."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

from routerbench.data import sha256_file
from routerbench.nonlinear import NonlinearRegretRouter
from routerbench.prospective import run_prospective

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("local_adapter_tool", ROOT / "tools/train_local_adapter.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class MockBERTFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, reference_config=None):
        self.reference_config = reference_config

    def fit(self, X, y=None):
        self.n_features_out_ = 768
        self.fit_ids_ = X.index.tolist()
        return self

    def transform(self, X):
        result = np.zeros((len(X), 768))
        result[:, 0] = X["prompt_text"].map(len)
        result[:, 1] = X["prompt_text"].map(lambda value: value.count("calculation"))
        return result


def fixture(root):
    config = json.loads((ROOT / "configs/local_adapter_v1.json").read_text())
    config["expected_cases"] = 48
    config["source_dir"] = "reports/audit"
    config["dataset_path"] = "cohort.jsonl"
    config["learner"]["max_iter"] = 5
    config["bootstrap_replicates"] = 100
    rows = [{"id": str(i), "prompt": f"Question {i}: "+("short lookup" if i % 2 else "long calculation "*5),
             "reference": "yes", "grade": "exact"} for i in range(48)]
    path = root / config["dataset_path"]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    manifest = {"models": config["models"], "fixture": True}
    reference = root / "reports/reference.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"test_outcomes_used_for_selection": False,
        "configuration": {"checkpoint_revision": "fixture", "weights_sha256": "a"*64}}))
    config["bert_reference_selection_file"] = "reports/reference.json"
    config["bert_reference_selection_sha256"] = sha256_file(reference)
    def generate(model, prompt):
        i = int(prompt.split(":")[0].split()[1])
        correct = i % 7 != 0 if model == "strong" else i % 2 == 1
        return {"content": "yes" if correct else "no", "duration_seconds": 2 if model == "strong" else .5,
                "usage": {"input_tokens": 10, "output_tokens": 1}}
    run_prospective(path, root / config["source_dir"], manifest, lambda _: "strong", lambda _: "weak",
                    generate, collect_both_models=True)
    return config


class LocalAdapterTests(unittest.TestCase):
    def test_real_adapter_has_disjoint_hash_split_seconds_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            loaded = adapter.load_local_development(root, config)
            self.assertTrue(loaded[0]["prompt_text"].str.startswith("SYSTEM:\n"+adapter.SYSTEM+"\n\nUSER:\n").all())
            original_split = loaded[3]["split"].copy()
            changed_scores = loaded[1]*0
            self.assertTrue(adapter.local_partition(loaded[3]["prompt_group"]).equals(original_split))
            output = root / "reports/adapter"
            with threadpool_limits(1), patch.object(adapter, "make_bert_transformer", side_effect=MockBERTFeatures):
                report = adapter.train_adapter(root, config, output)
            self.assertEqual(report["configuration"]["cost_unit"], "seconds")
            self.assertFalse(report["independent_validation"])
            self.assertIsNone(report["monetary_cost_usd"])
            self.assertEqual(sum(report["partition_counts"].values()), 48)
            groups = report["partition_prompt_groups"]
            self.assertFalse(set(groups["fit"]) & set(groups["calibration"]))
            self.assertGreaterEqual(report["selected_calibration_metrics"]["success_rate"], report["fixed_strong_calibration_metrics"]["success_rate"])
            self.assertEqual(len(report["curve"]), 9)
            model = joblib.load(output / "adapter.joblib")
            self.assertIsInstance(model, NonlinearRegretRouter)
            self.assertEqual(model.strong, "qwen2.5:1.5b")
            self.assertEqual(model.weak, "qwen2.5:0.5b")
            self.assertEqual(model.fit_rows_, report["partition_counts"]["fit"])
            self.assertEqual(model.calibration_rows_, report["partition_counts"]["calibration"])
            control_report = report["controls"]["bert_adapted"]
            control = joblib.load(output / control_report["artifact"])
            self.assertIsInstance(control.transformer_, MockBERTFeatures)
            self.assertEqual(control.transformer_.n_features_out_, 768)
            self.assertEqual(control.transformer_.fit_ids_, loaded[3].index[loaded[3]["split"].eq("fit")].tolist())
            self.assertEqual((control.fit_rows_, control.calibration_rows_), (model.fit_rows_, model.calibration_rows_))
            self.assertEqual(control.target_counts_, model.target_counts_)
            for parameter in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "quality_threshold", "seed"):
                self.assertEqual(getattr(control, parameter), getattr(model, parameter))
            self.assertEqual(len(control_report["curve"]), 9)
            self.assertGreaterEqual(control_report["selected_calibration_metrics"]["success_rate"], control_report["fixed_strong_calibration_metrics"]["success_rate"])
            self.assertEqual(control_report["selected_specification"]["cost_unit"], "seconds")
            self.assertIn(report["selected_specification"]["mode"], {"pairwise", "fixed_strong_fallback"})
            self.assertTrue(report["primary_candidate_not_selected_by_architecture"])
            for name, digest in report["files_sha256"].items():
                self.assertEqual(sha256_file(output / name), digest)
            self.assertEqual(report["artifact_sha256"], sha256_file(output / "adapter.joblib"))
            with self.assertRaises(FileExistsError):
                adapter.train_adapter(root, config, output)

    def test_changed_cache_or_dataset_are_rejected_before_fitting(self):
        for target in ("responses.json", "dataset"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = fixture(root)
                path = root / (config["dataset_path"] if target == "dataset" else config["source_dir"]+"/responses.json")
                path.write_text(path.read_text()+" ")
                with patch.object(adapter.NonlinearRegretRouter, "fit") as fit:
                    with self.assertRaises(ValueError):
                        adapter.train_adapter(root, config, root / "reports/adapter")
                    fit.assert_not_called()

    def test_split_is_outcome_independent_and_fixed_generation_grading_not_monetary(self):
        groups = pd.Series([str(i) for i in range(100)])
        first = adapter.local_partition(groups)
        self.assertTrue(first.equals(adapter.local_partition(groups)))
        self.assertEqual(set(first), {"fit", "calibration"})
        with self.assertRaises(ValueError):
            adapter.local_partition(groups, fit_fraction=1)
        selected = pd.Series(["strong", "weak"], index=[1, 2])
        scores = pd.DataFrame({"strong": [1, 0], "weak": [0, 1]}, index=[1, 2])
        durations = pd.DataFrame({"strong": [3., 3.], "weak": [.5, .5]}, index=[1, 2])
        metrics, observed = adapter.evaluate_seconds(selected, scores, durations)
        self.assertEqual(metrics["total_selected_generation_seconds"], 3.5)
        self.assertEqual(metrics["success_rate"], 1)
        self.assertIsNone(metrics["monetary_cost_usd"])


if __name__ == "__main__":
    unittest.main()
