"""Frozen local transfer controls for agent episodes; no fitting or generation.

The structural adapter already targets the exact Qwen identities. Published
BERT preferences are explicitly projected to that pair, without claiming Qwen
calibration, equal supervision or an MF/SW comparison.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import joblib
import pandas as pd

from relay_router.agent import AgentRequest
from .pairwise import choose_pairwise
from .routellm_reference import verify_local_model


ROOT = Path(__file__).resolve().parents[2]
STRONG = "qwen2.5:1.5b"
WEAK = "qwen2.5:0.5b"
LOCAL_SUMMARY = "reports/local-adapter-development-v1/summary.json"
LOCAL_SUMMARY_SHA256 = "201360a82b9f73c3cd469f3906098879fb0a016a151d5820ecc484b0fd86e280"
BERT_SELECTION = "reports/routellm-bert-secondary-v1/selection.json"
BERT_SELECTION_SHA256 = "cf6726654a0c5f772f056c5a24b9694f85a7db071e82b39dc8d2cd613c3a8fc3"
BERT_CHECKPOINT = "routellm/bert_gpt4_augmented"
BERT_REVISION = "86237e3df400762178ea98379477b8296e66d5e4"
BERT_WEIGHTS_SHA256 = "e2a92240bad1c7fb9befbf021a62fad5feb1e2a3990d53a28d160a39bb1cac7b"
BERT_MANIFEST_SHA256 = "3bf541749991e7a17d400040c0bba70adaf36e08a1464cd220c7c5bcf6be619b"
LEGACY_SYSTEM_SHA256 = "fa16732498a422fa6813cff372d210ff973264dda77099a7b572d3a2972b6450"
TOKENIZATION = {"max_length": 512, "truncation": True, "padding": True, "prefix": None}
INPUT_TEMPLATE = {"function": "routerbench.agent_ollama.routing_text",
                  "content": "current agent system, TASK first, state and complete observed history",
                  "format_and_target_transfer": True,
                  "private_reference_or_outcome_used": False}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _object_hash(value):
    return _hash(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            allow_nan=False, separators=(",", ":")).encode())


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("A lowercase SHA-256 digest is required.")
    return value


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Source paths must be relative to the trusted project root.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("The source must be an existing file within the project.")
    return path


def _checked_bytes(path, expected):
    payload = path.read_bytes()
    if _hash(payload) != _digest(expected):
        raise ValueError(f"Source SHA-256 changed: {path.name}")
    return payload


def _unique(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("Duplicate source JSON key.")
        value[name] = item
    return value


def _json(payload):
    def invalid(value):
        raise ValueError("Source JSON cannot contain nonfinite numbers.")
    result = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("Source JSON must be an object.")
    return result


def _pair(strong, weak):
    if strong != STRONG or weak != WEAK:
        raise ValueError("This transfer control requires the exact frozen Qwen model identities; no remapping.")
    return {"strong": STRONG, "weak": WEAK}


def _routing_text(request):
    if not isinstance(request, AgentRequest):
        raise TypeError("The policy requires AgentRequest.")
    from .agent_ollama import routing_text
    text = routing_text(request.snapshot())
    if not isinstance(text, str) or not text.strip():
        raise ValueError("routing_text must return complete nonempty text.")
    return text


def _input_template():
    from . import agent_ollama
    return {**INPUT_TEMPLATE, "format": agent_ollama.INPUT_FORMAT,
            "system_sha256": _hash(agent_ollama.AGENT_SYSTEM.encode()),
            "source_sha256": _hash(Path(agent_ollama.__file__).read_bytes())}


def _specification(specification):
    if not isinstance(specification, dict):
        raise ValueError("A frozen structural policy specification is required.")
    expected = {"mode": "pairwise", "policy": "signed_pairwise_utility", "tradeoff": .1,
                "cost_unit": "seconds", "input_template": "SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}",
                "system_sha256": LEGACY_SYSTEM_SHA256, "strong": STRONG, "weak": WEAK,
                "quality_threshold": .75}
    if specification != expected:
        raise ValueError("Preserve the exact frozen structural specification, pair and 0.1 cutoff.")
    return deepcopy(specification)


class LocalStructuralTransferPolicy:
    """Unchanged local structural model/cutoff applied to new episode text.

    Direct construction supports injected tiny predictors for contract tests.
    Use from_local for the verified historical artifact; do not trust arbitrary
    joblib files solely because their self-supplied checksum matches.
    """

    name = "local_structural_transfer"

    def __init__(self, model, specification, *, provenance=None):
        self._specification = _specification(specification)
        self._model = model
        self._check_model()
        self._provenance = deepcopy(provenance) if provenance is not None else {"origin": "injected predictor; no archived artifact verified"}
        self.last_audit = None

    @property
    def strong(self):
        return STRONG

    @property
    def weak(self):
        return WEAK

    def _check_model(self):
        _pair(getattr(self._model, "strong", None), getattr(self._model, "weak", None))
        if not callable(getattr(self._model, "predict_tables", None)) or getattr(self._model, "quality_threshold", None) != .75:
            raise ValueError("The predictor must retain its pairwise target and predict_tables contract.")

    @classmethod
    def from_local(cls, root=ROOT, *, expected_summary_sha256=LOCAL_SUMMARY_SHA256):
        root = Path(root).resolve()
        summary_path = _inside(root, LOCAL_SUMMARY)
        summary = _json(_checked_bytes(summary_path, expected_summary_sha256))
        specification = _specification(summary.get("selected_specification"))
        if (summary.get("models") != _pair(STRONG, WEAK)
                or summary.get("partition_counts") != {"fit": 48, "calibration": 16}
                or summary.get("configuration", {}).get("learner", {}).get("representation") != "structural"
                or summary.get("configuration", {}).get("expected_cases") != 64):
            raise ValueError("The source is not the frozen 64-case local structural adapter.")
        artifact = summary.get("artifact")
        if not isinstance(artifact, str) or Path(artifact).name != artifact or artifact != "adapter.joblib":
            raise ValueError("The structural artifact must be adapter.joblib beside its summary.")
        artifact_path = _inside(root, str(summary_path.parent.relative_to(root) / artifact))
        payload = _checked_bytes(artifact_path, summary.get("artifact_sha256"))
        # Deserialize exactly the bytes whose hash was checked, without a second read.
        model = joblib.load(BytesIO(payload))
        return cls(model, specification, provenance={"origin": "verified local structural adapter",
                   "summary": LOCAL_SUMMARY, "summary_sha256": expected_summary_sha256,
                   "artifact": artifact_path.relative_to(root).as_posix(),
                   "artifact_sha256": summary["artifact_sha256"],
                   "original_configuration_sha256": _object_hash(summary["configuration"])})

    def select(self, request):
        self.last_audit = None
        self._check_model()
        text = _routing_text(request)
        frame = pd.DataFrame({"prompt_text": [text]})
        probabilities, seconds = self._model.predict_tables(frame)
        if (not isinstance(probabilities, pd.DataFrame) or not isinstance(seconds, pd.DataFrame)
                or not probabilities.index.equals(frame.index) or not seconds.index.equals(frame.index)):
            raise ValueError("The adapter must return exactly the requested row/index.")
        chosen = choose_pairwise(probabilities, seconds, self.strong, self.weak, self._specification["tradeoff"]).iloc[0]
        self.last_audit = {"selected_model": str(chosen.selected_model),
                           "decision": str(chosen.decision), "tradeoff": .1,
                           "routing_text_sha256": _hash(text.encode()), "input_characters": len(text),
                           "predicted_quality_gain": float(chosen.predicted_quality_gain),
                           "predicted_extra_generation_seconds": float(chosen.predicted_extra_cost),
                           "predicted_generation_seconds": {name: float(seconds.iloc[0][name]) for name in (self.strong, self.weak)},
                           "probabilities": {name: float(probabilities.iloc[0][name]) for name in ("strong_only", "weak_only", "equal")},
                           "scope": "single-response local predictions transferred to episode actions; not calibrated episode success"}
        return str(chosen.selected_model)

    def manifest(self):
        self._check_model()
        return deepcopy({"name": self.name, "catalog": _pair(self.strong, self.weak),
                         "frozen_specification": self._specification, "sources": self._provenance,
                         "input_template": _input_template(),
                         "rule": "choose_pairwise with signed extra predicted generation seconds and frozen tradeoff 0.1; exact tie chooses weak",
                         "training": {"cases": 64, "fit": 48, "calibration": 16, "known_templates": 8,
                                      "features": "14 structural attributes", "method": "NonlinearRegretRouter local structural adapter",
                                      "is_historical_v4_ensemble": False},
                         "scope": "format and target transfer from single-response extraction to tool episodes; no retraining or Qwen identity remapping",
                         "independent_validation": False, "trained_on_agent_episodes": False})


def _reference_configuration(config):
    if not isinstance(config, dict):
        raise ValueError("A pinned BERT reference configuration is required.")
    expected = {"checkpoint": BERT_CHECKPOINT, "checkpoint_revision": BERT_REVISION,
                "weights_file": "model.safetensors", "weights_sha256": BERT_WEIGHTS_SHA256,
                "reference_tokenization": TOKENIZATION, "threads": 4,
                "packages": {"torch": "2.9.0", "transformers": "4.57.1"}}
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("Preserve the pinned published BERT checkpoint, CPU runtime and 512-token truncation.")
    return deepcopy(config)


class PublishedBERTTransferPolicy:
    """Published preference checkpoint with frozen .5 cut projected to Qwen.

    A predictor exposing predict_batch can be injected without importing Torch.
    Each select calls that predictor; no probability cache is maintained here.
    last_audit is mutable per-step telemetry and never part of the manifest.
    """

    name = "routellm_bert_transfer"

    def __init__(self, predictor, reference_config, *, strong=STRONG, weak=WEAK,
                 threshold=.5, provenance=None):
        _pair(strong, weak)
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not isfinite(threshold) or threshold != .5:
            raise ValueError("The published transfer comparison freezes threshold 0.5.")
        if not callable(getattr(predictor, "predict_batch", None)):
            raise TypeError("BERT predictor must expose predict_batch.")
        self._predictor = predictor
        self._reference_config = _reference_configuration(reference_config)
        self._provenance = deepcopy(provenance) if provenance is not None else {"origin": "injected predictor; no checkpoint verified"}
        self.last_audit = None

    @property
    def strong(self):
        return STRONG

    @property
    def weak(self):
        return WEAK

    @property
    def threshold(self):
        return .5

    @classmethod
    def from_local(cls, root=ROOT, *, expected_selection_sha256=BERT_SELECTION_SHA256,
                   expected_manifest_sha256=BERT_MANIFEST_SHA256, predictor_factory=None):
        root = Path(root).resolve()
        selection_path = _inside(root, BERT_SELECTION)
        selection = _json(_checked_bytes(selection_path, expected_selection_sha256))
        if selection.get("test_outcomes_used_for_selection") is not False:
            raise ValueError("The published reference must preserve its development-only source selection.")
        config = _reference_configuration(selection.get("configuration"))
        directory_name = config.get("model_dir")
        if not isinstance(directory_name, str) or Path(directory_name).is_absolute():
            raise ValueError("The BERT checkpoint must remain inside the project.")
        directory = (root / directory_name).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("The BERT checkpoint must remain inside the project.")
        manifest_path = _inside(root, str(directory.relative_to(root) / "manifest.json"))
        _checked_bytes(manifest_path, expected_manifest_sha256)
        manifest = verify_local_model(directory, config)
        injected = predictor_factory is not None
        if predictor_factory is None:
            from .routellm_reference import LocalBERTReference
            predictor_factory = LocalBERTReference
        if not callable(predictor_factory):
            raise TypeError("predictor_factory must be callable.")
        predictor = predictor_factory(directory, deepcopy(config))
        return cls(predictor, config, provenance={"origin": "pinned published preference checkpoint",
                   "selection": BERT_SELECTION, "selection_sha256": expected_selection_sha256,
                   "checkpoint_directory": directory.relative_to(root).as_posix(),
                   "checkpoint_manifest_sha256": expected_manifest_sha256,
                   "checkpoint_files_sha256": {name: entry["sha256"] for name, entry in manifest["files"].items()},
                   "injected_predictor_factory": injected})

    def select(self, request):
        self.last_audit = None
        text = _routing_text(request)
        records = self._predictor.predict_batch([text])
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            raise ValueError("BERT must return exactly one coverage/probability record.")
        record = records[0]
        probability = record.get("probability")
        if isinstance(probability, bool) or not isinstance(probability, Real) or not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("BERT returned an invalid preference probability.")
        counts = {}
        for name in ("input_tokens", "retained_tokens", "truncated_tokens"):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("BERT token coverage must contain nonnegative integer measurements.")
            counts[name] = int(value)
        if (counts["input_tokens"] < 1 or not 1 <= counts["retained_tokens"] <= 512
                or counts["retained_tokens"] > counts["input_tokens"]
                or counts["truncated_tokens"] != counts["input_tokens"]-counts["retained_tokens"]):
            raise ValueError("BERT token coverage is inconsistent with its fixed truncation.")
        chosen = self.strong if probability >= self.threshold else self.weak
        self.last_audit = {"probability": float(probability), **counts, "selected_model": chosen,
                           "routing_text_sha256": _hash(text.encode()), "input_characters": len(text),
                           "threshold": self.threshold, "token_coverage": counts["retained_tokens"]/counts["input_tokens"],
                           "scope": "published relative preference projected to Qwen; not a calibrated Qwen or episode-success probability",
                           "truncation": "official first 512 tokens including special tokens; no sliding windows"}
        return chosen

    def manifest(self):
        config = self._reference_config
        return deepcopy({"name": self.name, "catalog": _pair(self.strong, self.weak),
                         "threshold": self.threshold, "sources": self._provenance,
                         "reference_configuration_sha256": _object_hash(config),
                         "checkpoint": config["checkpoint"], "checkpoint_revision": config["checkpoint_revision"],
                         "weights_sha256": config["weights_sha256"], "tokenization": config["reference_tokenization"],
                         "runtime_packages": config["packages"], "cpu_threads": config["threads"],
                         "input_template": _input_template(),
                         "rule": "published preference probability >= 0.5 chooses strong; tie chooses strong; no probability cache",
                         "source_repository": config.get("source_repository"), "source_revision": config.get("source_revision"),
                         "supervision": "published external preference/augmentation supervision; no local Qwen or episode calibration",
                         "projection": {"published_strong_action": self.strong, "published_weak_action": self.weak},
                         "scope": "published RouteLLM-BERT transfer reference; not MF, SW, the adapted BERT head or an equal-supervision method comparison",
                         "independent_validation": False, "calibrated_for_execution_pair": False})
