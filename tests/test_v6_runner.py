"""Development boundaries, frozen routing, fixed fallback and provenance for v6."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from routerbench.data import sha256_file, write_json
from test_improvement_integration import fixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_v6", ROOT / "tools/train_v6.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixtureRouter:
    refuse_weak = False

    def __init__(self, strong, weak):
        self.strong, self.weak = strong, weak

    def fit(self, fit, scores, costs, calibration, calibration_scores):
        self.fit_ids = fit.index.tolist()
        self.calibration_ids = calibration.index.tolist()
        return self

    def predict_tables(self, inputs):
        simple = inputs["prompt_text"].str.contains("simple") if not self.refuse_weak else pd.Series(False, index=inputs.index)
        first = pd.DataFrame({"harm_probability": [.001 if value else .99 for value in simple],
                              "weak_success_probability": [.99 if value else .01 for value in simple]}, index=inputs.index)
        costs = pd.DataFrame({self.strong: .01, self.weak: .001}, index=inputs.index)
        return first, costs


def v6_fixture(root):
    config, loaded = fixture(root)
    config.update(experiment_id="v6-fixture", dataset={"path": "data/raw/fake.pkl", "sha256": "fixture"},
                  paid_api_calls_authorized=False, prepared_dir="data/processed",
                  learner={"seed": 42, "quality_threshold": .75},
                  max_harm_grid=[.01, .03, .05, .1], min_weak_success_grid=[.5, .65, .8, .9],
                  selection={"quality_loss_margin": 0}, latency_sample_size=2)
    config["reference"]["threshold_grid"] = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, 1]
    selection = root / config["reference"]["selection_file"]
    write_json(selection, {"test_outcomes_used_for_selection": False,
                          "configuration": {"common_pair": config["common_pair"], "quality_score_threshold": .75,
                                            "dataset": config["dataset"]}})
    config["reference"]["selection_sha256"] = sha256_file(selection)
    return config, loaded


class V6RunnerTests(unittest.TestCase):
    def test_previous_boolean_actions_decode_strong_and_reproduce_control_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = v6_fixture(root)
            validation = loaded[3].index[loaded[3]["development_split"].eq("validation")]
            directory = root / "reports/previous"
            directory.mkdir(parents=True)
            actions = pd.DataFrame({"point_000": True, "point_001": loaded[1].loc[validation, "weak"].eq(0)}, index=validation)
            actions.to_parquet(directory / "actions.parquet")
            curve = []
            for index, column in enumerate(actions):
                selected = actions[column].map({True: "strong", False: "weak"})
                metrics = MODULE.evaluate_policy(selected, loaded[1].loc[validation], loaded[2].loc[validation], .75)["metrics"]
                curve.append({"parameter": index * .5, "metrics": metrics})
            write_json(directory / "curve.json", {"curve": curve})
            write_json(directory / "summary.json", {"common_pair": config["common_pair"], "quality_score_threshold": .75,
                "source_prepared_files_sha256": loaded[4]["files_sha256"],
                "source_reference_probabilities_sha256": config["reference"]["development_probabilities_sha256"], "outer_test_accessed": False})
            control = {name: "reports/previous/" + filename for name, filename in
                       (("summary_file", "summary.json"), ("curve_file", "curve.json"), ("actions_file", "actions.parquet"))}
            control.update({name + "_sha256": sha256_file(root / path) for name, path in list(control.items())})
            config["previous_bert_control"] = control
            with patch.object(MODULE, "load_development_inputs", return_value=loaded), patch.object(MODULE, "make_router", return_value=FixtureRouter("strong", "weak")):
                result = MODULE.run_v6(root, config, root / "reports/v6")
            self.assertEqual(result["additional_controls"]["previous_bert_101"]["selected_point"]["index"], 1)
            persisted = pd.read_parquet(root / "reports/v6/previous_bert_all_routes.parquet")
            self.assertTrue(persisted.isin(["strong", "weak"]).all().all())
            self.assertEqual(result["comparisons"]["vs_previous_bert_101"]["success_rate_delta"], 0)

    def test_freezes_all_actions_before_metrics_and_exports_replayable_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = v6_fixture(root)
            output = root / "reports/v6"
            router = FixtureRouter("strong", "weak")
            original_evaluate = MODULE.evaluate_policy
            def inspect_evaluation(*args, **kwargs):
                for name in ("development_started.json", "source_bundle.zip", "actions_frozen.json", "relay_all_routes.parquet", "bert_all_routes.parquet"):
                    self.assertTrue((output / name).is_file(), name)
                self.assertEqual(pd.read_parquet(output / "relay_all_routes.parquet").shape, (12, 16))
                self.assertEqual(pd.read_parquet(output / "bert_all_routes.parquet").shape, (12, 16))
                return original_evaluate(*args, **kwargs)
            with patch.object(MODULE, "load_development_inputs", return_value=loaded), \
                    patch.object(MODULE, "make_router", return_value=router), \
                    patch.object(MODULE, "evaluate_policy", side_effect=inspect_evaluation):
                result = MODULE.run_v6(root, config, output)
            self.assertEqual(len(router.fit_ids), 18)
            self.assertEqual(len(router.calibration_ids), 12)
            self.assertTrue(set(router.fit_ids).isdisjoint(router.calibration_ids))
            validation_ids = set(loaded[3].index[loaded[3]["development_split"].eq("validation")])
            self.assertTrue(set(router.fit_ids + router.calibration_ids).isdisjoint(validation_ids))
            self.assertEqual(result["partition_counts"], {"fit": 18, "calibration": 12, "validation": 12})
            self.assertTrue(result["development_gate_passed"])
            self.assertTrue(result["quality_floor_met"])
            self.assertEqual(result["routing_diagnostics"]["strong_only_on_weak_routes"], 0)
            self.assertEqual(result["routing_diagnostics"]["weak_routes"], 6)
            self.assertEqual(result["policy_mode"], "guarded")
            self.assertFalse(result["outer_test_accessed"])
            self.assertFalse(result["confirmation_of_superiority"])
            bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["cost_unit"], "historical_cost")
            self.assertEqual(bundle["artifact_sha256"], sha256_file(output / bundle["artifact"]))
            for name, digest in result["files_sha256"].items():
                self.assertEqual(sha256_file(output / name), digest)
            with patch.object(MODULE, "load_development_inputs") as loader:
                with self.assertRaises(FileExistsError):
                    MODULE.run_v6(root, config, output)
                loader.assert_not_called()

    def test_fallback_is_not_a_pass_even_when_bert_quality_exceeds_fixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, loaded = v6_fixture(root)
            validation = loaded[3].index[loaded[3]["development_split"].eq("validation")]
            loaded[1].loc[validation[1], "strong"] = 0
            router = FixtureRouter("strong", "weak")
            router.refuse_weak = True
            with patch.object(MODULE, "load_development_inputs", return_value=loaded), patch.object(MODULE, "make_router", return_value=router):
                result = MODULE.run_v6(root, config, root / "reports/v6")
            self.assertEqual(result["policy_mode"], "fixed_strong")
            self.assertFalse(result["development_gate_passed"])
            self.assertFalse(result["quality_floor_met"])
            self.assertIsNone(result["selected_point"])

    def test_changed_sources_and_historical_pair_mismatch_fail_before_loader(self):
        for mutation in ("hash", "pair", "target", "dataset"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = v6_fixture(root)
                if mutation == "hash":
                    with (root / config["reference"]["development_probabilities_file"]).open("ab") as stream:
                        stream.write(b"changed")
                elif mutation == "pair":
                    config["common_pair"]["weak"] = "unrelated_model"
                elif mutation == "target":
                    config["quality_threshold"] = .5
                    config["learner"]["quality_threshold"] = .5
                else:
                    config["dataset"]["sha256"] = "other_dataset"
                with patch.object(MODULE, "load_development_inputs") as loader:
                    with self.assertRaises(ValueError):
                        MODULE.run_v6(root, config, root / "reports/v6")
                    loader.assert_not_called()

    def test_outer_test_rows_and_misaligned_cache_fail_before_training(self):
        for mutation in ("test", "order"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, loaded = v6_fixture(root)
                if mutation == "test":
                    loaded[3].loc[100, "split"] = "test"
                else:
                    path = root / config["reference"]["development_probabilities_file"]
                    pd.read_parquet(path).iloc[::-1].to_parquet(path)
                    config["reference"]["development_probabilities_sha256"] = sha256_file(path)
                with patch.object(MODULE, "load_development_inputs", return_value=loaded), patch.object(MODULE, "make_router") as factory:
                    with self.assertRaises(ValueError):
                        MODULE.run_v6(root, config, root / "reports/v6")
                    factory.assert_not_called()
                self.assertFalse((root / "reports/v6").exists())

    def test_forbidden_flags_grids_and_outside_output_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = v6_fixture(root)
            for name in ("outer_test_evaluation_allowed", "automatic_live_promotion_allowed", "paid_api_calls_authorized"):
                changed = copy.deepcopy(config)
                changed[name] = True
                with patch.object(MODULE, "load_development_inputs") as loader:
                    with self.assertRaises(ValueError):
                        MODULE.run_v6(root, changed, root / "reports/v6")
                    loader.assert_not_called()
            for changed in ({**config, "selection": {"quality_loss_margin": .01}}, {**config, "max_harm_grid": [.99]}):
                with self.assertRaises(ValueError):
                    MODULE.run_v6(root, changed, root / "reports/v6")
            with self.assertRaises(ValueError):
                MODULE.run_v6(root, config, root.parent / "outside-v6")


if __name__ == "__main__":
    unittest.main()
