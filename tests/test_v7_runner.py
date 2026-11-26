"""Frozen primary/control roles, prior-policy replay and development-only v7."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from routerbench.data import sha256_file, write_json
from test_v6_runner import v6_fixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_v7", ROOT / "tools/train_v7.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeRouter:
    def __init__(self, strong, weak, refuse_weak=False):
        self.strong, self.weak, self.refuse_weak = strong, weak, refuse_weak

    def fit(self, inputs, scores, costs, calibration, calibration_scores):
        self.fit_ids, self.calibration_ids = inputs.index.tolist(), calibration.index.tolist()
        return self

    def predict_tables(self, inputs):
        simple = inputs.prompt_text.str.contains("simple")
        if self.refuse_weak:
            simple[:] = False
        p = pd.DataFrame({"strong_only": np.where(simple, .01, .8), "weak_only": np.where(simple, .2, .01),
                          "equal": np.where(simple, .79, .19)}, index=inputs.index)
        p["expected_quality_gain"] = p.strong_only-p.weak_only
        c = pd.DataFrame({self.strong: .01, self.weak: .001}, index=inputs.index)
        return p, c


def fixture(root, strong_miss=False):
    c, loaded = v6_fixture(root)
    c.update(quantiles=101, primary_candidate_id="neighborhood_primary", controls={},
             candidates=[{"id": "neighborhood_primary", "use_neighbors": True}, {"id": "global_control", "use_neighbors": False}])
    c.pop("max_harm_grid"); c.pop("min_weak_success_grid")
    ids = loaded[3].index[loaded[3].development_split.eq("validation")]
    if strong_miss:
        loaded[1].loc[ids[1], "strong"] = 0
    def evaluate(selected):
        return MODULE.evaluate_policy(selected, loaded[1].loc[ids], loaded[2].loc[ids], .75)["metrics"]
    actions = pd.DataFrame({"fixed_strong": True, "mixed": loaded[1].loc[ids, "weak"].eq(0), "fixed_weak": False}, index=ids)
    for name in ("bert101", "ensemble_v4"):
        directory = root / "reports" / name
        directory.mkdir(parents=True)
        actions.to_parquet(directory / "actions.parquet")
        curve = [{"parameter": i, "metrics": evaluate(actions[column].map({True: "strong", False: "weak"}))}
                 for i, column in enumerate(actions)]
        write_json(directory / "curve.json", {"curve": curve})
        write_json(directory / "summary.json", {"common_pair": c["common_pair"], "quality_score_threshold": .75,
                   "source_prepared_files_sha256": loaded[4]["files_sha256"], "outer_test_accessed": False,
                   "source_reference_probabilities_sha256": c["reference"]["development_probabilities_sha256"]})
        c["controls"][name] = {"directory": "reports/"+name, "summary": "summary.json", "curve": "curve.json", "actions": "actions.parquet"}
    directory = root / "reports/v6"
    directory.mkdir(parents=True)
    selected = pd.Series("strong", index=ids, name="selected_model")
    bert_selected = actions["mixed"].map({True: "strong", False: "weak"}).rename("selected_model")
    selected.to_frame().to_parquet(directory / "routes.parquet")
    bert_selected.to_frame().to_parquet(directory / "bert_routes.parquet")
    write_json(directory / "summary.json", {"configuration": {"common_pair": c["common_pair"], "quality_threshold": .75},
               "outer_test_accessed": False, "source_prepared_files_sha256": loaded[4]["files_sha256"],
               "selected_metrics": evaluate(selected), "routellm_bert": {"selected_point": {"metrics": evaluate(bert_selected)}}})
    write_json(directory / "bundle.json", {"schema_version": 1, "policy_mode": "fixed_strong"})
    c["controls"]["v6"] = {"directory": "reports/v6", "summary": "summary.json", "routes": "routes.parquet", "bert_routes": "bert_routes.parquet", "bundle": "bundle.json"}
    for control in c["controls"].values():
        control["files_sha256"] = {filename: sha256_file(root / control["directory"] / filename)
                                   for key, filename in control.items() if key != "directory"}
    return c, loaded


class V7RunnerTests(unittest.TestCase):
    def test_all_actions_precede_metrics_and_primary_remains_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, loaded = fixture(root)
            output = root / "reports/v7"
            models = []
            def factory(pair, params):
                model = FakeRouter(*pair, refuse_weak=params["use_neighbors"])
                models.append(model)
                return model
            original = MODULE.evaluate_policy
            def checked(*args, **kwargs):
                self.assertTrue((output / "actions_frozen.json").exists())
                for name in ("neighborhood_primary", "global_control"):
                    self.assertEqual(pd.read_parquet(output / (name+"_all_routes.parquet")).shape, (12, 101))
                return original(*args, **kwargs)
            with patch.object(MODULE, "load_development_inputs", return_value=loaded), patch.object(MODULE, "make_router", side_effect=factory), patch.object(MODULE, "evaluate_policy", side_effect=checked):
                result = MODULE.run_v7(root, c, output)
            self.assertEqual(result["primary_candidate_id"], "neighborhood_primary")
            self.assertEqual(result["selected_metrics"], result["candidates"]["neighborhood_primary"]["selected_metrics"])
            self.assertGreater(result["selected_metrics"]["cost_per_1000"], result["candidates"]["global_control"]["selected_metrics"]["cost_per_1000"])
            self.assertFalse(result["beats_bert_cost_on_development"])
            self.assertFalse(result["outer_test_accessed"])
            self.assertFalse(result["confirmation_of_superiority"])
            self.assertEqual(result["partition_counts"], {"fit": 18, "calibration": 12, "validation": 12})
            for model in models:
                self.assertEqual(len(model.fit_ids), 18)
                self.assertTrue(set(model.fit_ids).isdisjoint(model.calibration_ids))
            for name, digest in result["files_sha256"].items():
                self.assertEqual(sha256_file(output / name), digest)
            bundle = json.loads((output / "bundle.json").read_text())
            self.assertEqual(bundle["artifact"], "neighborhood_primary.joblib")
            self.assertEqual(bundle["policy_mode"], "pairwise_utility")
            self.assertTrue(np.isfinite(bundle["tradeoff"]))
            with self.assertRaises(FileExistsError):
                MODULE.run_v7(root, c, output)

    def test_fallback_below_quality_floor_does_not_promote_global_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, loaded = fixture(root, strong_miss=True)
            with patch.object(MODULE, "load_development_inputs", return_value=loaded), patch.object(MODULE, "make_router", side_effect=lambda pair, p: FakeRouter(*pair, refuse_weak=p["use_neighbors"])):
                result = MODULE.run_v7(root, c, root / "reports/v7")
            self.assertEqual(result["policy_mode"], "fixed_strong")
            self.assertFalse(result["development_gate_passed"])
            self.assertFalse(result["beats_bert_cost_on_development"])
            self.assertFalse(result["candidates"]["neighborhood_primary"]["quality_floor_met"])
            self.assertTrue(result["candidates"]["global_control"]["quality_floor_met"])

    def test_changed_control_hash_or_pair_rejected_before_training(self):
        for kind in ("hash", "pair", "test", "order"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                c, loaded = fixture(root)
                if kind == "hash":
                    (root / "reports/bert101/actions.parquet").write_bytes(b"modified")
                elif kind == "pair":
                    c["common_pair"]["weak"] = "different"
                elif kind == "test":
                    loaded[3].loc[100, "split"] = "test"
                else:
                    path = root / "reports/bert101/actions.parquet"
                    pd.read_parquet(path).iloc[::-1].to_parquet(path)
                    c["controls"]["bert101"]["files_sha256"]["actions.parquet"] = sha256_file(path)
                with patch.object(MODULE, "load_development_inputs", return_value=loaded), patch.object(MODULE, "make_router") as factory:
                    with self.assertRaises(ValueError):
                        MODULE.run_v7(root, c, root / "reports/v7")
                    factory.assert_not_called()
                self.assertFalse((root / "reports/v7").exists())

    def test_flags_primary_and_grid_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, _ = fixture(root)
            mutations = [{**c, flag: True} for flag in ("outer_test_evaluation_allowed", "automatic_live_promotion_allowed", "paid_api_calls_authorized")]
            mutations.extend([{**c, "primary_candidate_id": "global_control"}, {**c, "quantiles": 100},
                              {**c, "selection": {"quality_loss_margin": .01}}])
            for changed in mutations:
                with patch.object(MODULE, "load_development_inputs") as loader:
                    with self.assertRaises(ValueError):
                        MODULE.run_v7(root, changed, root / "reports/v7")
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
