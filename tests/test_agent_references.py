"""Pinned transfer policies: identities, source integrity and honest telemetry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import pandas as pd

from relay_router.agent import AgentRequest, AgentState
from routerbench import agent_references as references
from routerbench.agent_ollama import routing_text


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return digest(path)


def specification():
    return {"mode": "pairwise", "policy": "signed_pairwise_utility", "tradeoff": .1,
            "cost_unit": "seconds", "input_template": "SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}",
            "system_sha256": references.LEGACY_SYSTEM_SHA256, "strong": references.STRONG,
            "weak": references.WEAK, "quality_threshold": .75}


def reference_config():
    return {"checkpoint": references.BERT_CHECKPOINT, "checkpoint_revision": references.BERT_REVISION,
            "weights_file": "model.safetensors", "weights_sha256": references.BERT_WEIGHTS_SHA256,
            "reference_tokenization": deepcopy(references.TOKENIZATION), "threads": 4,
            "packages": {"torch": "2.9.0", "transformers": "4.57.1"},
            "model_dir": "artifacts/routellm-bert", "source_repository": "https://github.com/lm-sys/RouteLLM",
            "source_revision": "0b64fdafe049e596a3f5657c219329f24af24198"}


def request():
    return AgentRequest("Tarea completa\n  con indentación", ({"role": "tool", "content": {"observed": "texto final\n  intacto"}},),
                        AgentState(1, 0, (), ("read_file",), 4))


class TinyDecider:
    strong, weak, quality_threshold = references.STRONG, references.WEAK, .75

    def __init__(self, probabilities=(.2, .1, .7), seconds=(2., 1.)):
        self.probabilities, self.seconds = probabilities, seconds
        self.last_frame = None

    def predict_tables(self, frame):
        self.last_frame = frame.copy()
        probabilities = pd.DataFrame([self.probabilities], columns=["strong_only", "weak_only", "equal"], index=frame.index)
        probabilities["expected_quality_gain"] = probabilities.strong_only-probabilities.weak_only
        return probabilities, pd.DataFrame([self.seconds], columns=[self.strong, self.weak], index=frame.index)


class TinyBERT:
    def __init__(self, probability=.5):
        self.record = {"probability": probability, "input_tokens": 900, "retained_tokens": 512, "truncated_tokens": 388}
        self.calls = []

    def predict_batch(self, prompts):
        self.calls.append(prompts.copy())
        return [deepcopy(self.record)]


def local_fixture(root):
    path = root / references.LOCAL_SUMMARY
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = path.parent / "adapter.joblib"
    joblib.dump(TinyDecider(), artifact)
    summary = {"models": {"strong": references.STRONG, "weak": references.WEAK},
               "partition_counts": {"fit": 48, "calibration": 16},
               "configuration": {"learner": {"representation": "structural"}, "expected_cases": 64},
               "selected_specification": specification(), "artifact": artifact.name, "artifact_sha256": digest(artifact)}
    return path, artifact, summary, write(path, summary)


def bert_fixture(root):
    config = reference_config()
    selection = {"configuration": config, "test_outcomes_used_for_selection": False}
    selection_path = root / references.BERT_SELECTION
    manifest = {"files": {"model.safetensors": {"sha256": references.BERT_WEIGHTS_SHA256}}}
    manifest_path = root / config["model_dir"] / "manifest.json"
    return selection_path, manifest_path, manifest, write(selection_path, selection), write(manifest_path, manifest)


class StructuralTransferTests(unittest.TestCase):
    def test_signed_policy_and_exact_tie_keep_frozen_rule(self):
        model = TinyDecider()
        policy = references.LocalStructuralTransferPolicy(model, specification())
        original = policy.manifest()
        self.assertEqual(policy.select(request()), references.WEAK)
        self.assertEqual(model.last_frame.columns.tolist(), ["prompt_text"])
        self.assertEqual(model.last_frame.iloc[0, 0], routing_text(request()))
        self.assertEqual(policy.last_audit["tradeoff"], .1)
        self.assertEqual(policy.manifest(), original)
        # Signed extra seconds can be negative; this must not become v7's ratio.
        model = TinyDecider((.1, .1, .8), (1., 2.))
        policy = references.LocalStructuralTransferPolicy(model, specification())
        self.assertEqual(policy.select(request()), references.STRONG)
        self.assertEqual(policy.last_audit["predicted_extra_generation_seconds"], -1.)

    def test_reject_remapping_changed_target_or_cutoff(self):
        for key, value in [("strong", "different-strong"), ("weak", "different-weak"),
                           ("tradeoff", .2), ("cost_unit", "USD"), ("quality_threshold", .8),
                           ("mode", "fixed_strong_fallback")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                references.LocalStructuralTransferPolicy(TinyDecider(), {**specification(), key: value})
        model = TinyDecider()
        policy = references.LocalStructuralTransferPolicy(model, specification())
        model.weak = "renamed"
        with self.assertRaisesRegex(ValueError, "remapping"):
            policy.select(request())

    def test_loader_checks_summary_and_artifact_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, artifact, summary, summary_hash = local_fixture(root)
            policy = references.LocalStructuralTransferPolicy.from_local(root, expected_summary_sha256=summary_hash)
            self.assertEqual(policy.manifest()["sources"]["artifact_sha256"], digest(artifact))
            self.assertEqual(policy.manifest()["training"]["known_templates"], 8)
            self.assertFalse(policy.manifest()["training"]["is_historical_v4_ensemble"])
            artifact.write_bytes(artifact.read_bytes()+b"changed")
            with patch.object(references.joblib, "load") as loader, self.assertRaisesRegex(ValueError, "SHA-256"):
                references.LocalStructuralTransferPolicy.from_local(root, expected_summary_sha256=summary_hash)
            loader.assert_not_called()
            path.write_text(path.read_text()+" ")
            with patch.object(references.joblib, "load") as loader, self.assertRaisesRegex(ValueError, "SHA-256"):
                references.LocalStructuralTransferPolicy.from_local(root, expected_summary_sha256=summary_hash)
            loader.assert_not_called()

    def test_invalid_source_pair_path_and_policy_rejected_before_loading(self):
        for mutate in [lambda s: s.update(artifact="../outside.joblib"),
                       lambda s: s["selected_specification"].update(tradeoff=.2),
                       lambda s: s["models"].update(weak="renamed")]:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, artifact, summary, _ = local_fixture(root)
                mutate(summary)
                checksum = write(path, summary)
                with patch.object(references.joblib, "load") as loader, self.assertRaises(ValueError):
                    references.LocalStructuralTransferPolicy.from_local(root, expected_summary_sha256=checksum)
                loader.assert_not_called()

    def test_deserializes_checked_bytes_and_manifest_is_independent_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, artifact, summary, checksum = local_fixture(root)
            payload = artifact.read_bytes()
            def load_checked(stream):
                self.assertEqual(stream.read(), payload)
                artifact.write_bytes(b"different later bytes")
                return TinyDecider()
            with patch.object(references.joblib, "load", side_effect=load_checked):
                policy = references.LocalStructuralTransferPolicy.from_local(root, expected_summary_sha256=checksum)
            manifest = policy.manifest()
            manifest["frozen_specification"]["tradeoff"] = 9
            self.assertEqual(policy.manifest()["frozen_specification"]["tradeoff"], .1)
            self.assertFalse(policy.manifest()["trained_on_agent_episodes"])

    def test_rejects_wrong_prediction_row_and_clears_stale_audit(self):
        model = TinyDecider()
        policy = references.LocalStructuralTransferPolicy(model, specification())
        policy.select(request())
        probabilities, costs = model.predict_tables(pd.DataFrame({"prompt_text": ["other"]}, index=[99]))
        with patch.object(model, "predict_tables", return_value=(probabilities, costs)), self.assertRaisesRegex(ValueError, "row/index"):
            policy.select(request())
        self.assertIsNone(policy.last_audit)


class PublishedTransferTests(unittest.TestCase):
    def test_threshold_tie_coverage_no_cache_and_stable_manifest(self):
        predictor = TinyBERT(.5)
        policy = references.PublishedBERTTransferPolicy(predictor, reference_config())
        original = policy.manifest()
        self.assertEqual(policy.select(request()), references.STRONG)
        self.assertEqual(policy.last_audit["truncated_tokens"], 388)
        self.assertAlmostEqual(policy.last_audit["token_coverage"], 512/900)
        predictor.record["probability"] = .499
        self.assertEqual(policy.select(request()), references.WEAK)
        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(predictor.calls[0], [routing_text(request())])
        self.assertEqual(policy.manifest(), original)
        self.assertNotIn("probability", policy.manifest())
        self.assertFalse(policy.manifest()["calibrated_for_execution_pair"])
        self.assertIn("not MF, SW", policy.manifest()["scope"])

    def test_invalid_pair_threshold_and_changed_tokenization_rejected(self):
        for kwargs in ({"strong": "qwen2.5:7b"}, {"weak": "mixtral"}, {"threshold": .6}, {"threshold": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                references.PublishedBERTTransferPolicy(TinyBERT(), reference_config(), **kwargs)
        config = reference_config()
        config["reference_tokenization"]["max_length"] = 1024
        with self.assertRaisesRegex(ValueError, "512"):
            references.PublishedBERTTransferPolicy(TinyBERT(), config)

    def test_invalid_probabilities_and_coverage_are_not_reported(self):
        for field, value in [("probability", True), ("probability", float("nan")), ("probability", 1.1),
                             ("input_tokens", 20), ("retained_tokens", 513), ("truncated_tokens", 0),
                             ("input_tokens", True), ("retained_tokens", 1.5)]:
            predictor = TinyBERT()
            policy = references.PublishedBERTTransferPolicy(predictor, reference_config())
            policy.select(request())
            predictor.record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                policy.select(request())
            self.assertIsNone(policy.last_audit)

    def test_local_factory_verifies_sources_before_injected_predictor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, manifest_path, manifest, selection_hash, manifest_hash = bert_fixture(root)
            factory = Mock(return_value=TinyBERT())
            with patch.object(references, "verify_local_model", return_value=manifest) as verifier:
                policy = references.PublishedBERTTransferPolicy.from_local(root,
                    expected_selection_sha256=selection_hash, expected_manifest_sha256=manifest_hash,
                    predictor_factory=factory)
            verifier.assert_called_once()
            factory.assert_called_once()
            self.assertTrue(policy.manifest()["sources"]["injected_predictor_factory"])
            self.assertEqual(policy.manifest()["sources"]["checkpoint_manifest_sha256"], manifest_hash)
            self.assertEqual(policy.select(request()), references.STRONG)
            manifest_path.write_text(manifest_path.read_text()+" ")
            factory.reset_mock()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                references.PublishedBERTTransferPolicy.from_local(root,
                    expected_selection_sha256=selection_hash, expected_manifest_sha256=manifest_hash,
                    predictor_factory=factory)
            factory.assert_not_called()
            selection.write_text(selection.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                references.PublishedBERTTransferPolicy.from_local(root,
                    expected_selection_sha256=selection_hash, expected_manifest_sha256=manifest_hash,
                    predictor_factory=factory)
            factory.assert_not_called()

    def test_non_agent_inputs_fail_without_prediction(self):
        predictor = TinyBERT()
        policy = references.PublishedBERTTransferPolicy(predictor, reference_config())
        with self.assertRaises(TypeError):
            policy.select("plain prompt")
        self.assertEqual(predictor.calls, [])


if __name__ == "__main__":
    unittest.main()
