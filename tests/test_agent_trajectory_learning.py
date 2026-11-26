"""Synthetic fixtures test contracts; none of these labels are live results."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import issparse

from relay_router.agent import AgentRequest, AgentStateV8
from routerbench.agent_trajectory_learning import (
    TARGET, TrajectorySuccessPredictor, TrajectoryTrainingRow,
    public_request_text, trajectory_state_features,
)


MODELS = ("fixture/small", "fixture/large", "fixture/third")


def rows(part, count=16, extra_tool=False):
    result = []
    for index in range(count):
        state = AgentStateV8(step=index % 4, attempt=index % 2,
            previous_errors=("BackendActionError:invalid_json",) if index % 2 else (),
            available_tools=("fit_tool", "calibration_only") if extra_tool else ("fit_tool",),
            remaining_calls=8 - index % 4, context_utf8_bytes=100 + index,
            progress={"tool_actions": index % 3, "source_group": "never_feature"})
        request = AgentRequest(f"{part} synthetic prompt {index}\n    preserve indentation", (), state)
        for tier, model in enumerate(MODELS):
            result.append(TrajectoryTrainingRow(request, model, bool((index + tier) % 2),
                .001 * (index + tier + 1), f"{part}-{index}", "fixture-strong-continuation-v1",
                "synthetic_fixture", TARGET, "fixture_list_price_usd"))
    return result


class AgentTrajectoryLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = TrajectorySuccessPredictor(MODELS, max_text_features=120).fit(
            rows("fit"), rows("calibrationexclusive", extra_tool=True))

    def test_public_text_preserves_raw_strings_but_removes_structured_provenance(self):
        request = rows("fit")[0].request
        history = ({"role": "tool", "content": {"task_id": "hidden_id", "family": "hidden_family",
            "source_group": "hidden_group", "model": "hidden_model", "cost_usd": 999,
            "state_id": "hidden_state", "anchor_id": "hidden_anchor",
            "observation": {"content": "def value():\n    return 7\n", "expected_spec": "hidden_gold"}}},
            {"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": "notes.txt"},
                                                       "returned_model": "hidden_returned"})})
        public = public_request_text(replace(request, history=history))
        self.assertIn("\n    preserve indentation", public)
        self.assertIn("def value():\n    return 7\n", public)
        self.assertIn("notes.txt", public)
        self.assertNotIn("hidden_", public)
        self.assertNotIn("never_feature", public)

    def test_features_exclude_transport_and_model_metadata_but_count_backend_action_errors(self):
        original = rows("fit")[0].request
        altered = replace(original, state=replace(original.state, infrastructure_attempt=9,
            infrastructure_errors=("BackendTimeoutError",), failed_tiers=(0, 1),
            progress={**original.state.progress, "model_id": "hidden_model", "source_group": "changed"}))
        self.assertEqual(trajectory_state_features(original), trajectory_state_features(altered))
        changed = replace(original, state=replace(original.state,
            previous_errors=("BackendActionError:format", "BackendActionError:malformed", "InvalidAction:json")))
        features = trajectory_state_features(changed)
        self.assertEqual(features["errors_BackendActionError"], 2)
        self.assertEqual(features["errors_InvalidAction"], 1)
        self.assertEqual(features["progress:tool_actions"], 0)
        self.assertFalse(any("hidden" in name or "source_group" in name for name in features))

    def test_vocabulary_and_scalers_fit_only_and_sparse(self):
        model = self.predictor
        union = dict(model._features.named_steps["union"].transformer_list)
        text = union["text"].named_steps["tfidf"]
        state = union["state"].named_steps["vectorizer"]
        scaler = union["state"].named_steps["scale"]
        self.assertNotIn("calibrationexclusive", text.vocabulary_)
        self.assertNotIn("tool:calibration_only", state.vocabulary_)
        self.assertNotIn("progress:source_group", state.vocabulary_)
        self.assertLessEqual(len(text.vocabulary_), 120)
        before_idf, before_scale = text.idf_.copy(), scaler.scale_.copy()
        inferred = replace(rows("neverfitword")[0].request,
            state=replace(rows("neverfitword")[0].request.state, context_utf8_bytes=10**9))
        self.assertTrue(issparse(model._features.transform([inferred])))
        model.predict_outcomes(inferred)
        self.assertNotIn("neverfitword", text.vocabulary_)
        np.testing.assert_array_equal(before_idf, text.idf_)
        np.testing.assert_array_equal(before_scale, scaler.scale_)

    def test_catalog_order_is_deterministic_and_forecasts_finite(self):
        request = rows("inference")[0].request
        outcomes = self.predictor.predict_outcomes(request)
        self.assertEqual(tuple(outcomes), MODELS)
        probabilities = self.predictor.predict_proba(request)
        costs = self.predictor.predict_costs(request)
        for model in MODELS:
            self.assertEqual(outcomes[model], {"success_probability": probabilities[model],
                "remaining_provider_cost_usd": costs[model]})
            self.assertTrue(0 <= probabilities[model] <= 1)
            self.assertTrue(np.isfinite(costs[model]) and costs[model] >= 0)
        self.assertEqual(self.predictor.predict_requests([]), [])
        self.assertEqual(self.predictor.predict_requests([request]), [probabilities])

    def test_missing_costs_are_not_zero_and_do_not_drop_success_targets(self):
        fitting = [replace(row, remaining_provider_cost_usd=None) if index >= 9 else row
                   for index, row in enumerate(rows("fit"))]
        model = TrajectorySuccessPredictor(MODELS).fit(fitting, rows("cal"))
        for support in model.manifest()["support"].values():
            self.assertEqual(support["fit"]["rows"], 16)
            self.assertEqual(support["fit"]["known_cost_rows"], 3)
        unknown = [replace(row, remaining_provider_cost_usd=None) for row in rows("development")]
        for diagnostic in model.evaluate(unknown).values():
            self.assertEqual(diagnostic["unknown_cost_rows"], 16)
            self.assertEqual(diagnostic["known_cost_rows"], 0)
            self.assertIsNone(diagnostic["remaining_cost_mae_usd"])
        with self.assertRaisesRegex(ValueError, "known fit costs"):
            TrajectorySuccessPredictor(MODELS).fit(
                [replace(row, remaining_provider_cost_usd=None) for row in rows("fit")], rows("cal"))

    def test_cost_head_uses_fit_targets_only(self):
        normal = TrajectorySuccessPredictor(MODELS).fit(rows("fit"), rows("cal"))
        altered = TrajectorySuccessPredictor(MODELS).fit(rows("fit"),
            [replace(row, remaining_provider_cost_usd=1e6) for row in rows("cal")])
        self.assertEqual(normal.predict_costs(rows("inference")[0].request),
                         altered.predict_costs(rows("inference")[0].request))

    def test_source_groups_and_normalized_prompts_cannot_cross_splits(self):
        calibration = rows("cal")
        calibration[0] = replace(calibration[0], source_group=rows("fit")[0].source_group)
        with self.assertRaisesRegex(ValueError, "groups"):
            TrajectorySuccessPredictor(MODELS).fit(rows("fit"), calibration)
        calibration = rows("cal")
        fitting = rows("fit")
        request = replace(fitting[0].request, prompt="  " + fitting[0].request.prompt.replace("\n", "  ") + " ")
        calibration[0] = replace(calibration[0], request=request)
        with self.assertRaisesRegex(ValueError, "Duplicate task prompts"):
            TrajectorySuccessPredictor(MODELS).fit(fitting, calibration)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.predictor.evaluate(fitting)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.predictor.evaluate(rows("calibrationexclusive"))

    def test_missing_classes_or_models_are_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, "Both classes"):
            TrajectorySuccessPredictor(MODELS).fit([replace(row, success=True) for row in rows("fit")], rows("cal"))
        with self.assertRaisesRegex(ValueError, "Both classes"):
            TrajectorySuccessPredictor(MODELS).fit(rows("fit"), [row for row in rows("cal") if row.model != MODELS[2]])

    def test_regime_and_cost_basis_are_explicit(self):
        for field, value in (("target", "observable_action_success"), ("success", 1),
                             ("assignment", "invented"), ("cost_basis", "")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(rows("fit")[0], **{field: value})
        for field, value in (("continuation_policy", "different-policy"), ("cost_basis", "different-pricing")):
            calibration = rows("cal")
            calibration[0] = replace(calibration[0], **{field: value})
            with self.assertRaisesRegex(ValueError, "All rows must share"):
                TrajectorySuccessPredictor(MODELS).fit(rows("fit"), calibration)
        with self.assertRaisesRegex(ValueError, "differ from fit"):
            self.predictor.evaluate([replace(row, cost_basis="other") for row in rows("development")])

    def test_paired_states_require_every_model_once_on_same_request(self):
        fitting = [replace(row, assignment="paired_state") for row in rows("fit")]
        calibration = [replace(row, assignment="paired_state") for row in rows("cal")]
        valid = TrajectorySuccessPredictor(MODELS).fit(fitting, calibration)
        self.assertEqual(valid.manifest()["assignment"], "paired_state")
        for broken in (fitting[1:], fitting + [fitting[0]],
                       [replace(fitting[0], request=replace(fitting[0].request, prompt="unpaired")), *fitting[1:]]):
            with self.assertRaisesRegex(ValueError, "Each paired state"):
                TrajectorySuccessPredictor(MODELS).fit(broken, calibration)

    def test_invalid_costs_and_requests_rejected_and_snapshots_are_independent(self):
        sample = rows("fit")[0]
        for cost in (True, -1, float("nan"), float("inf")):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                replace(sample, remaining_provider_cost_usd=cost)
        source = replace(sample.request, history=({"role": "tool", "content": {"text": "before"}},))
        row = replace(sample, request=source)
        source.history[0]["content"]["text"] = "after"
        source.state.progress["tool_actions"] = 99
        self.assertEqual(row.request.history[0]["content"]["text"], "before")
        self.assertEqual(row.request.state.progress["tool_actions"], 0)
        with self.assertRaises(ValueError):
            self.predictor.predict_proba({})

    def test_development_diagnostics_and_manifest_scope(self):
        diagnostics = self.predictor.evaluate(rows("development"))
        for value in diagnostics.values():
            self.assertEqual(value["rows"], 16)
            self.assertIsNotNone(value["brier"])
            self.assertIsNotNone(value["log_loss"])
            self.assertIsNotNone(value["remaining_cost_mae_usd"])
        manifest = self.predictor.manifest()
        self.assertEqual(manifest["target"], TARGET)
        self.assertIn("reference", manifest["scope"])
        self.assertIn("not full operating cost", manifest["cost_target_scope"])
        self.assertEqual(set(manifest["data_sha256"]), {"fit", "calibration"})
        self.assertEqual(manifest["cost_basis"], "fixture_list_price_usd")
        manifest["support"][MODELS[0]]["fit"]["rows"] = 0
        self.assertEqual(self.predictor.manifest()["support"][MODELS[0]]["fit"]["rows"], 16)

    def test_persistence_roundtrip_and_trusted_hash_precedes_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.joblib"
            sha = self.predictor.save(path)
            restored = TrajectorySuccessPredictor.load(path, expected_sha256=sha)
            self.assertEqual(restored.manifest(), self.predictor.manifest())
            self.assertEqual(restored.predict_outcomes(rows("inference")[0].request),
                             self.predictor.predict_outcomes(rows("inference")[0].request))
            with self.assertRaises(FileExistsError):
                self.predictor.save(path)
            path.write_bytes(path.read_bytes() + b"tamper")
            with patch("routerbench.agent_trajectory_learning.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    TrajectorySuccessPredictor.load(path, expected_sha256=sha)
                load.assert_not_called()

    def test_incompatible_versions_block_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.joblib"
            sha = self.predictor.save(path)
            sidecar = path.with_name(path.name + ".manifest.json")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            manifest["predictor"]["runtime_versions"]["scikit-learn"] = "incompatible"
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("routerbench.agent_trajectory_learning.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "runtime versions"):
                    TrajectorySuccessPredictor.load(path, expected_sha256=sha)
                load.assert_not_called()

    def test_task_audit_blocks_training_and_inspected_development_from_holdout(self):
        model = TrajectorySuccessPredictor(MODELS).fit(rows("fit"), rows("cal"))
        tasks = lambda part: [{"source_group": row.source_group, "prompt": row.request.prompt}
                              for row in rows(part) if row.model == MODELS[0]]
        with self.assertRaisesRegex(ValueError, "fit/calibration"):
            model.audit_task_groups(tasks("fit"), evaluation_split="development")
        model.evaluate(rows("development"))
        with self.assertRaisesRegex(ValueError, "previously inspected development"):
            model.audit_task_groups(tasks("development"))
        self.assertEqual(model.audit_task_groups(tasks("development"), evaluation_split="development")["tasks"], 16)
        self.assertEqual(model.audit_task_groups(tasks("unseen"))["evaluation_split"], "holdout")
        self.assertFalse(model.audit_task_groups(tasks("unseen"))["independent_validation"])
        self.assertEqual(len(model.manifest()["partition_provenance"]["development_group_sha256"]), 16)


if __name__ == "__main__":
    unittest.main()
