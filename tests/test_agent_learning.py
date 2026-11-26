"""Synthetic labels exercise offline fitting; they are not agent results."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from relay_router.agent import AgentRequest, AgentStateV8
from routerbench.agent_learning import StateSuccessPredictor, TrainingRow, TARGET, state_features


MODELS = ("fixture/small", "fixture/large")


def rows(part, count=24, extra_tool=False):
    result = []
    for index in range(count):
        state = AgentStateV8(step=index % 4, attempt=index % 2,
            previous_errors=("InvalidAction:ValueError",) if index % 2 else (),
            available_tools=("fit_tool", "calibration_only") if extra_tool else ("fit_tool",),
            remaining_calls=8 - index % 4, context_utf8_bytes=100 + index)
        request = AgentRequest(f"{part} synthetic prompt {index}\n    preserve indentation", (), state)
        for model in MODELS:
            result.append(TrainingRow(request, model, bool(index % 2), f"{part}-{index}", TARGET,
                                      "fixture-fixed-continuation-v1", "synthetic_fixture"))
    return result


class AgentLearningTests(unittest.TestCase):
    def fitted(self):
        return StateSuccessPredictor(MODELS, min_samples_leaf=2).fit(rows("fit"), rows("cal", extra_tool=True))

    def test_features_ignore_prompt_history_ids_and_infrastructure(self):
        original = rows("fit")[0].request
        altered = AgentRequest("provider IDs and scores are untrusted text", (
            {"role": "tool", "content": {"gold": "never vectorize", "model": "secret-model", "cost": 999}},),
            replace(original.state, infrastructure_attempt=9, infrastructure_errors=("BackendTimeoutError",)))
        self.assertEqual(state_features(original), state_features(altered))
        self.assertFalse(any("secret" in key or "gold" in key for key in state_features(altered)))

    def test_vocab_fit_only_and_probability_outputs_have_exact_catalog(self):
        model = self.fitted()
        manifest = model.manifest()
        for names in manifest["features_by_model"].values():
            self.assertIn("tool:fit_tool", names)
            self.assertNotIn("tool:calibration_only", names)
            self.assertFalse(any("prompt" in name or "model" in name for name in names))
        probabilities = model.predict_proba(rows("inference")[0].request)
        self.assertEqual(set(probabilities), set(MODELS))
        self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))
        self.assertEqual(manifest["target"], TARGET)
        self.assertEqual(manifest["assignment"], "synthetic_fixture")
        self.assertEqual(model.predict_requests([]), [])

    def test_missing_class_and_absent_model_are_not_fabricated(self):
        fitting = [replace(row, success=True) for row in rows("fit")]
        with self.assertRaisesRegex(ValueError, "Both classes"):
            StateSuccessPredictor(MODELS).fit(fitting, rows("cal"))
        calibration = [row for row in rows("cal") if row.model == MODELS[0]]
        with self.assertRaisesRegex(ValueError, "Both classes"):
            StateSuccessPredictor(MODELS).fit(rows("fit"), calibration)

    def test_source_groups_and_duplicate_prompts_cannot_cross_partitions(self):
        calibration = rows("cal")
        calibration[0] = replace(calibration[0], source_group=rows("fit")[0].source_group)
        with self.assertRaisesRegex(ValueError, "groups"):
            StateSuccessPredictor(MODELS).fit(rows("fit"), calibration)
        calibration = rows("cal")
        calibration[0] = replace(calibration[0], request=rows("fit")[0].request)
        with self.assertRaisesRegex(ValueError, "Duplicate prompts"):
            StateSuccessPredictor(MODELS).fit(rows("fit"), calibration)

    def test_regime_is_explicit_and_incompatible_continuations_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(rows("fit")[0], target="episode_success")
        with self.assertRaises(ValueError):
            replace(rows("fit")[0], success=1)
        calibration = rows("cal")
        calibration[0] = replace(calibration[0], continuation_policy="another-regime")
        with self.assertRaisesRegex(ValueError, "continuation"):
            StateSuccessPredictor(MODELS).fit(rows("fit"), calibration)

    def test_development_metrics_do_not_reuse_fit_or_calibration(self):
        model = self.fitted()
        diagnostics = model.evaluate(rows("development"))
        self.assertTrue(all(value["rows"] == 24 and value["brier"] is not None and value["log_loss"] is not None
                            for value in diagnostics.values()))
        with self.assertRaisesRegex(ValueError, "overlap"):
            model.evaluate(rows("fit"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            model.evaluate(rows("cal"))

    def test_trusted_artifact_hash_precedes_deserialization(self):
        model = self.fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.joblib"
            sha = model.save(path)
            restored = StateSuccessPredictor.load(path, expected_sha256=sha)
            request = rows("inference")[0].request
            self.assertEqual(restored.predict_proba(request), model.predict_proba(request))
            with self.assertRaises(FileExistsError):
                model.save(path)
            path.write_bytes(path.read_bytes() + b"tamper")
            with patch("routerbench.agent_learning.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    StateSuccessPredictor.load(path, expected_sha256=sha)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
