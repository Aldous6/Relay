import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from routerbench.routellm_reference import (
    LocalBERTReference, _pair_predictions, cached_probabilities, coverage_summary,
    object_hash, reference_probabilities, reference_routes,
    require_frozen_selection, run_reference, select_operating_point, verify_local_model,
)


class FakePredictor:
    def __init__(self):
        self.batches = []

    def predict_batch(self, prompts):
        self.batches.append(prompts)
        return [{"probability": 0.4, "input_tokens": len(text)+2,
                 "retained_tokens": min(len(text)+2, 512),
                 "truncated_tokens": max(0, len(text)+2-512)} for text in prompts]


class ReferenceTests(unittest.TestCase):
    def test_official_three_class_probability_and_threshold_direction(self):
        logits = np.array([[0, 0, 0], [1000, 999, 998], [-1000, 0, 0]])
        actual = reference_probabilities(logits)
        np.testing.assert_allclose(actual[:2], [1/3, 1/(1+np.exp(-1)+np.exp(-2))])
        self.assertEqual(actual[2], 0)
        routes = reference_routes(pd.Series([0.3, 0.5, 0.9], index=[7, 8, 9]), 0.5,
                                  {"strong": "gpt4", "weak": "mixtral"})
        self.assertEqual(routes.tolist(), ["mixtral", "gpt4", "gpt4"])
        self.assertEqual(routes.index.tolist(), [7, 8, 9])
        for invalid in ([[1, 2]], [[1, np.nan, 2]], [1, 2, 3]):
            with self.assertRaises(ValueError):
                reference_probabilities(invalid)

    def test_selects_cheapest_feasible_point_and_deterministic_tie(self):
        curve = [{"threshold": .1, "success_rate": .78, "cost_per_1000": 1},
                 {"threshold": .2, "success_rate": .79, "cost_per_1000": 3},
                 {"threshold": .3, "success_rate": .81, "cost_per_1000": 3},
                 {"threshold": .4, "success_rate": .84, "cost_per_1000": 8}]
        selected = select_operating_point(curve, .80, .01)
        self.assertEqual(selected["threshold"], .2)
        self.assertTrue(selected["quality_constraint_met"])
        self.assertFalse(selected["fallback_used"])
        self.assertEqual(curve[0]["threshold"], .1)

    def test_no_feasible_point_uses_quality_then_cost_then_threshold(self):
        curve = [{"threshold": .1, "success_rate": .6, "cost_per_1000": 1},
                 {"threshold": .2, "success_rate": .7, "cost_per_1000": 3},
                 {"threshold": .3, "success_rate": .7, "cost_per_1000": 2}]
        selected = select_operating_point(curve, .9, .01)
        self.assertEqual(selected["threshold"], .3)
        self.assertTrue(selected["fallback_used"])

    def test_cache_resume_uses_exact_full_text_ids_and_protocol_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            prompts = pd.Series(["a\n  b", "x"*600], index=pd.Index([11, 91], name="source_row"))
            predictor = FakePredictor()
            first = cached_probabilities(prompts, predictor, cache, {"partition": "validation", "revision": "pinned"}, 1)
            self.assertEqual(predictor.batches, [["a\n  b"], ["x"*600]])
            second = FakePredictor()
            pd.testing.assert_frame_equal(first, cached_probabilities(prompts, second, cache, {"partition": "validation", "revision": "pinned"}, 1))
            self.assertEqual(second.batches, [])
            self.assertEqual(coverage_summary(first)["prompts_truncated"], 1)
            self.assertEqual(first.loc[91, "truncated_tokens"], 90)
            for changed in (prompts.iloc[::-1], prompts + " "):
                with self.assertRaises(ValueError):
                    cached_probabilities(changed, second, cache, {"partition": "validation", "revision": "pinned"}, 1)
            with self.assertRaises(ValueError):
                cached_probabilities(prompts, second, cache, {"partition": "validation", "revision": "changed"}, 1)
            payload = json.loads(cache.read_text())
            payload["rows"][0]["probability"] = .99
            cache.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                cached_probabilities(prompts, second, cache, {"partition": "validation", "revision": "pinned"}, 1)

    def test_partial_cache_resumes_without_repeating_completed_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            prompts = pd.Series([f"prompt {i}" for i in range(264)], index=range(264))
            predictor = FakePredictor()
            cached_probabilities(prompts, predictor, cache, {"partition": "validation"})
            payload = json.loads(cache.read_text())
            payload["rows"] = payload["rows"][:256]
            payload["rows_sha256"] = object_hash(payload["rows"])
            cache.write_text(json.dumps(payload))
            resumed = FakePredictor()
            result = cached_probabilities(prompts, resumed, cache, {"partition": "validation"})
            self.assertEqual(len(resumed.batches), 1)
            self.assertEqual(resumed.batches[0][0], "prompt 256")
            self.assertEqual(len(result), 264)

    def test_frozen_selection_gate_rejects_missing_or_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            with self.assertRaisesRegex(ValueError, "antes"):
                require_frozen_selection(path, "config")
            path.write_text(json.dumps({"status": "frozen_before_secondary_test", "configuration_sha256": "config"}))
            self.assertEqual(require_frozen_selection(path, "config")["status"], "frozen_before_secondary_test")
            with self.assertRaises(ValueError):
                require_frozen_selection(path, "changed")

    def test_pair_restriction_has_no_other_candidate_and_checks_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = pd.Index([4, 8], name="source_row")
            frame = pd.DataFrame({"strong": [.8, .9], "weak": [.5, .6], "outside": [1, 1]}, index=ids)
            frame.to_parquet(root / "p.parquet")
            frame.to_parquet(root / "c.parquet")
            probs, costs, hashes = _pair_predictions(root, "p.parquet", "c.parquet", ids, ["strong", "weak"])
            self.assertEqual(probs.columns.tolist(), ["strong", "weak"])
            self.assertEqual(costs.columns.tolist(), ["strong", "weak"])
            self.assertEqual(len(hashes), 2)
            with self.assertRaises(ValueError):
                _pair_predictions(root, "p.parquet", "c.parquet", ids[::-1], ["strong", "weak"])
            with self.assertRaises(ValueError):
                _pair_predictions(root, "p.parquet", "c.parquet", ids, ["strong", "weak"], {"p.parquet": "wrong"})

    def test_adapter_forwards_text_once_with_official_truncation(self):
        calls = []
        class Tensor:
            def __init__(self, value): self.value = value
            def sum(self, dim): return Tensor(np.asarray(self.value).sum(axis=dim))
            def tolist(self): return self.value.tolist()
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray(self.value)
        def tokenizer(prompts, **kwargs):
            calls.append((prompts, kwargs))
            if not kwargs["truncation"]:
                return {"input_ids": [[1]*602]}
            return {"input_ids": Tensor([[1]*512]), "attention_mask": Tensor([[1]*512])}
        class Model:
            def __call__(self, **kwargs):
                self.last = kwargs
                return type("Output", (), {"logits": Tensor([[0, 0, 0]])})()
        encoder = LocalBERTReference.__new__(LocalBERTReference)
        encoder.tokenizer, encoder.model, encoder.max_length = tokenizer, Model(), 512
        encoder.torch = type("Torch", (), {"inference_mode": staticmethod(contextlib.nullcontext)})()
        text = "begin\n    " + "long document "*100
        result = encoder.predict_batch([text])
        self.assertEqual([c[0] for c in calls], [[text], [text]])
        self.assertEqual(calls[1][1], {"padding": True, "truncation": True, "max_length": 512, "return_tensors": "pt"})
        self.assertEqual(result[0]["input_tokens"], 602)
        self.assertEqual(result[0]["retained_tokens"], 512)
        self.assertEqual(result[0]["truncated_tokens"], 90)

    def test_model_verification_rejects_modified_assets_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"checkpoint": "official", "checkpoint_revision": "revision", "weights_file": "model.safetensors", "weights_sha256": "weights"}
            manifest = {"checkpoint": "official", "revision": "revision", "files": {}}
            for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "LICENSE"]:
                manifest["files"][name] = {"sha256": "weights" if name == "model.safetensors" else "other"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch("routerbench.routellm_reference.sha256_file", return_value="tampered"):
                with self.assertRaisesRegex(ValueError, "modificado"):
                    verify_local_model(root, config)

    def test_integrated_secondary_run_freezes_before_test_and_outcomes_follow_actions(self):
        from routerbench.data import sha256_file
        from routerbench.routellm_reference import _pair_outcomes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"protocol_id": "secondary", "scope": "secondary transfer", "prepared_dir": "prepared",
                      "dataset": {"path": "unused.pkl", "sha256": "dataset"}, "model_dir": "model",
                      "relay_development_run": "development", "relay_final_dir": "final",
                      "output_dir": "output", "relay_variant": "lexical_hybrid", "checkpoint": "official",
                      "checkpoint_revision": "revision", "source_revision": "source", "weights_sha256": "weights",
                      "relay_final_probability_file": "p.parquet", "relay_final_predicted_cost_file": "c.parquet",
                      "common_pair": {"strong": "strong", "weak": "weak"}, "quality_score_threshold": .75,
                      "expected_population": {"development_validation": 2, "outer_test": 2},
                      "threshold_grid": [point/10 for point in range(1, 10)], "development_quality_loss_margin": .01,
                      "reference_tokenization": {"max_length": 512, "truncation": True, "padding": True, "prefix": None},
                      "batch_size": 8, "bootstrap_replicates": 100, "seed": 42, "limitations": ["secondary"]}
            rows = pd.DataFrame({"source_row": range(4), "prompt_group": list("abcd"),
                                 "split": ["train", "train", "test", "test"]}).set_index("source_row", drop=False)
            prepared = {"dataset_sha256": "dataset", "files_sha256": {"rows.parquet": "prepared"}}
            development = root / "development"
            development.mkdir()
            metadata = rows.iloc[:2].copy()
            metadata["development_split"] = "validation"
            metadata.to_parquet(development / "development_rows.parquet", index=False)
            for folder, ids, names in ((development, rows.index[:2], ["lexical_hybrid_probabilities.parquet", "lexical_hybrid_predicted_costs.parquet"]),
                                       (root / "final", rows.index[2:], ["p.parquet", "c.parquet"])):
                folder.mkdir(exist_ok=True)
                pd.DataFrame({"strong": [.8, .9], "weak": [.4, .5], "outside": [1, 1]}, index=ids).to_parquet(folder / names[0])
                pd.DataFrame({"strong": [2., 2.], "weak": [1., 1.], "outside": [0, 0]}, index=ids).to_parquet(folder / names[1])
            summary = {"dataset_sha256": "dataset", "source_prepared_files_sha256": prepared["files_sha256"],
                       "files_sha256": {p.name: sha256_file(p) for p in development.iterdir()}}
            (development / "summary.json").write_text(json.dumps(summary))
            (root / "model").mkdir()
            (root / "model/manifest.json").write_text("{}")
            raw = pd.DataFrame({"prompt": ["a", "b", "c", "d"], "strong": [1, 1, 1, 0],
                                "weak": [1, 0, 0, 1], "strong|total_cost": [2.]*4, "weak|total_cost": [1.]*4})
            accessed = []
            def observed(frame, ids, pair):
                accessed.append(ids.tolist())
                if 2 in ids:
                    frozen = require_frozen_selection(root / "output/selection.json", object_hash(config))
                    self.assertFalse(frozen["test_outcomes_used_for_selection"])
                    for method in ("routellm_bert", "relay_common_pair", "fixed_strong", "fixed_weak"):
                        self.assertTrue((root / f"output/{method}_routes.parquet").is_file())
                return _pair_outcomes(frame, ids, pair)
            with patch("routerbench.routellm_reference._verified_preparation", return_value=(rows, prepared)), \
                 patch("routerbench.routellm_reference.load_verified_pickle", return_value=raw), \
                 patch("routerbench.routellm_reference.LocalBERTReference", return_value=FakePredictor()), \
                 patch("routerbench.routellm_reference._pair_outcomes", side_effect=observed):
                report = run_reference(root, config)
            self.assertEqual(accessed, [[0, 1], [2, 3]])
            self.assertEqual(report["n_test"], 2)
            self.assertFalse(report["test_outcomes_used_for_selection"])
            self.assertFalse(report["primary_protocol_changed"])
            self.assertEqual(set(pd.read_parquet(root / "output/relay_common_pair_routes.parquet")["selected_model"]), {"strong"})


if __name__ == "__main__":
    unittest.main()
