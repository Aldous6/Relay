"""Harness integration boundaries: identities, integrity and explicit execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import pandas as pd

from relay_router import RelayRouter, load_bundle


class FakeDecider:
    strong = "trained-large-v1"
    weak = "trained-small-v1"

    def __init__(self):
        self.calls = []

    def predict_tables(self, inputs):
        self.calls.append(inputs.copy())
        return (pd.DataFrame({"harm_probability": [.01], "weak_success_probability": [.95]}, index=inputs.index),
                pd.DataFrame({self.strong: [10.], self.weak: [1.]}, index=inputs.index))


def fixture(directory, **updates):
    artifact = directory / "router.joblib"
    joblib.dump(FakeDecider(), artifact)
    manifest = {"schema_version": 1, "artifact": artifact.name,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "strong": FakeDecider.strong, "weak": FakeDecider.weak,
                "max_harm": .05, "min_weak_success": .8, "policy_mode": "guarded",
                "cost_unit": "historical_cost",
                "scope": "development fixture, not a quality guarantee"}
    manifest.update(updates)
    path = directory / "bundle.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class SDKTests(unittest.TestCase):
    def make_router(self, **kwargs):
        return RelayRouter(FakeDecider(), {"strong": FakeDecider.strong, "weak": FakeDecider.weak}, **kwargs)

    def test_route_is_offline_and_full_input_reaches_predictor_unchanged(self):
        router = self.make_router()
        prompt = "SYSTEM:\nKeep indentation\n\nUSER:\n    x = 1\nLAST EVIDENCE"
        with patch("socket.create_connection") as network:
            decision = router.route(prompt)
        network.assert_not_called()
        self.assertEqual(decision.selected_model, FakeDecider.weak)
        self.assertEqual(decision.harm_probability, .01)
        self.assertEqual(decision.predicted_cost, 1.)
        self.assertEqual(router.decider.calls[0].to_dict("list"), {"prompt_text": [prompt]})
        json.dumps(decision.to_dict(), allow_nan=False)

    def test_run_routes_before_one_generation_and_propagates_errors_without_retry(self):
        router = self.make_router()
        calls = []
        def generate(model, prompt):
            self.assertEqual(len(router.decider.calls), 1)
            calls.append((model, prompt))
            return {"content": "answer"}
        result = router.run("  exact input\n", generate)
        self.assertEqual(calls, [(FakeDecider.weak, "  exact input\n")])
        self.assertEqual(result["response"], {"content": "answer"})
        failing = Mock(side_effect=RuntimeError("provider failed"))
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            router.run("next", failing)
        failing.assert_called_once()

    def test_availability_fallback_never_remaps_or_calls_predictor(self):
        router = self.make_router()
        for name in (FakeDecider.strong, FakeDecider.weak):
            decision = router.route("prompt", available_models=[name, "unrelated-model"])
            self.assertEqual(decision.selected_model, name)
            self.assertEqual(decision.reason, "availability_fallback")
            self.assertIsNone(decision.harm_probability)
        self.assertEqual(router.decider.calls, [])
        for models in ([], ["unrelated-model"], "trained-small-v1", [None]):
            with self.subTest(models=models), self.assertRaises(ValueError):
                router.route("prompt", available_models=models)

    def test_fixed_policy_never_runs_decider_and_catalog_copy_cannot_change_identity(self):
        router = self.make_router(policy_mode="fixed_strong")
        changed = router.model_catalog
        changed["strong"] = "different-model"
        self.assertEqual(router.route("prompt").selected_model, FakeDecider.strong)
        self.assertEqual(router.decider.calls, [])
        with self.assertRaisesRegex(ValueError, "identities"):
            RelayRouter(FakeDecider(), {"strong": "new-large", "weak": "new-small"})

    def test_invalid_input_fails_before_prediction_or_generation(self):
        router = self.make_router()
        generate = Mock()
        for text in (None, 4, "", "  \n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                router.run(text, generate)
        generate.assert_not_called()
        self.assertEqual(router.decider.calls, [])
        for name in ("max_harm", "min_weak_success"):
            for value in (True, -1, 2, float("nan"), float("inf"), "0.1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.make_router(**{name: value})
        with self.assertRaises(TypeError):
            router.run("prompt", None)

    def test_extra_prediction_rows_are_rejected_before_generation(self):
        router = self.make_router()
        router.decider.predict_tables = Mock(return_value=(
            pd.DataFrame({"harm_probability": [.01, .02], "weak_success_probability": [.9, .8]}),
            pd.DataFrame({FakeDecider.strong: [10., 10.], FakeDecider.weak: [1., 1.]})))
        generate = Mock()
        with self.assertRaisesRegex(ValueError, "requested row"):
            router.run("prompt", generate)
        generate.assert_not_called()

    def test_bundle_checks_hash_before_deserialization_and_accepts_directory_or_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = fixture(directory)
            for path in (directory, manifest):
                router = load_bundle(path)
                self.assertEqual(router.model_catalog["strong"], FakeDecider.strong)
                self.assertEqual(router.cost_unit, "historical_cost")
                self.assertEqual(router.evaluation_scope, "development fixture, not a quality guarantee")
            with (directory / "router.joblib").open("ab") as stream:
                stream.write(b"tampered")
            with patch("relay_router.client.joblib.load") as deserialize:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_bundle(manifest)
            deserialize.assert_not_called()

    def test_bundle_rejects_escape_and_bad_policy_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for updates in ({"artifact": "../outside.joblib"}, {"artifact": str(directory / "router.joblib")},
                            {"max_harm": float("nan")}, {"schema_version": True},
                            {"policy_mode": "unknown"}, {"artifact_sha256": "bad"}):
                manifest = fixture(directory, **updates)
                with self.subTest(updates=updates), patch("relay_router.client.joblib.load") as deserialize:
                    with self.assertRaises(ValueError):
                        load_bundle(manifest)
                    deserialize.assert_not_called()

    def test_bundle_cannot_relabel_historical_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = fixture(Path(temporary), strong="different-strong-version")
            with self.assertRaisesRegex(ValueError, "identities"):
                load_bundle(manifest)


if __name__ == "__main__":
    unittest.main()
