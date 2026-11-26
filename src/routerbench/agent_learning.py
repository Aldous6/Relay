"""Offline state-only success heads with group-disjoint sigmoid calibration.

The target here is observable_action_success: a valid action whose tool/public
checks succeed (a terminal action uses its terminal grade OFFLINE as its label).
This is a step-level proxy, never a claim of final episode success from a
nonterminal state. No labels are extracted or inferred by this module.

Training rows explicitly declare the assignment and continuation regime. New
catalogs require new observations; historical RouterBench labels are unsuitable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from relay_router.agent import AgentRequest
from relay_router.v8 import model_order
from .features import normalize_for_dedup

TARGET = "observable_action_success"
ASSIGNMENTS = {"randomized_at_state", "paired_state", "synthetic_fixture"}
ERROR_KINDS = ("InvalidAction", "ToolError", "VerificationFailed", "ProgressError", "BackendActionError")


def state_features(request):
    """Read only explicit observable state; no text, outcomes or model IDs in X."""
    if not isinstance(request, AgentRequest):
        raise ValueError("state_features requires AgentRequest.")
    state = request.state
    result = {"step": state.step, "semantic_attempt": state.attempt,
              "remaining_calls": state.remaining_calls,
              "context_utf8_bytes": getattr(state, "context_utf8_bytes", 0),
              "n_tools": len(state.available_tools)}
    for kind in ERROR_KINDS:
        result["errors_" + kind] = sum(error.split(":", 1)[0] == kind for error in state.previous_errors)
    # Dictionary keys are encoded only from fit by DictVectorizer; unseen tools
    # during calibration/inference cannot extend that vocabulary.
    for tool in state.available_tools:
        result["tool:" + tool] = 1
    return result


@dataclass(frozen=True)
class TrainingRow:
    request: AgentRequest
    model: str
    success: bool
    source_group: str
    target: str
    continuation_policy: str
    assignment: str

    def __post_init__(self):
        if not isinstance(self.request, AgentRequest) or type(self.success) is not bool:
            raise ValueError("Rows require a real AgentRequest and an explicit boolean label.")
        for name in ("model", "source_group", "continuation_policy"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(f"{name} requires a nonempty exact string.")
        if self.target != TARGET or self.assignment not in ASSIGNMENTS:
            raise ValueError("Declare the supported action-success target and assignment regime.")
        object.__setattr__(self, "request", self.request.snapshot())


def _rows(rows, models):
    rows = list(rows)
    if not rows or any(not isinstance(row, TrainingRow) or row.model not in models for row in rows):
        raise ValueError("Nonempty TrainingRow inputs must use only the declared catalog.")
    return rows


def _logits(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class StateSuccessPredictor:
    """One local tree and one separately fitted sigmoid for each candidate.

    Insufficient class support raises an explicit error instead of fabricating
    counterfactual outcomes or claiming a constant output is calibrated.
    """

    def __init__(self, models, seed=42, max_depth=3, min_samples_leaf=4):
        self.models = model_order(models)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be uint32.")
        if type(max_depth) is not int or max_depth < 1 or type(min_samples_leaf) is not int or min_samples_leaf < 1:
            raise ValueError("Tree depth and minimum leaf size must be positive integers.")
        self.seed, self.max_depth, self.min_samples_leaf = seed, max_depth, min_samples_leaf

    def fit(self, fit_rows, calibration_rows):
        fitting, calibration = _rows(fit_rows, self.models), _rows(calibration_rows, self.models)
        fit_groups, calibration_groups = ({row.source_group for row in rows} for rows in (fitting, calibration))
        if fit_groups & calibration_groups:
            raise ValueError("Fit and calibration source groups must be disjoint.")
        fit_prompts, calibration_prompts = ({normalize_for_dedup(row.request.prompt) for row in rows}
                                             for rows in (fitting, calibration))
        if fit_prompts & calibration_prompts:
            raise ValueError("Duplicate prompts cannot cross fit and calibration, even under different group IDs.")
        regimes = {(row.target, row.continuation_policy, row.assignment) for row in fitting + calibration}
        if len(regimes) != 1:
            raise ValueError("All rows must share target, frozen continuation and assignment regime.")
        regime = next(iter(regimes))
        heads, support = {}, {}
        for model in self.models:
            fit_model, cal_model = ([row for row in rows if row.model == model] for rows in (fitting, calibration))
            if any({row.success for row in rows} != {False, True} for rows in (fit_model, cal_model)):
                raise ValueError(f"Both classes are required in fit and calibration for {model}; collect more observations.")
            pipeline = Pipeline([
                ("vectorizer", DictVectorizer(sparse=False)),
                ("tree", DecisionTreeClassifier(max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, random_state=self.seed)),
            ])
            pipeline.fit([state_features(row.request) for row in fit_model], [row.success for row in fit_model])
            raw = pipeline.predict_proba([state_features(row.request) for row in cal_model])[:, 1]
            sigmoid = LogisticRegression(random_state=self.seed, max_iter=1000)
            sigmoid.fit(_logits(raw), [row.success for row in cal_model])
            heads[model] = (pipeline, sigmoid)
            support[model] = {part: {"rows": len(rows), "positive": sum(row.success for row in rows),
                                    "groups": len({row.source_group for row in rows})}
                              for part, rows in (("fit", fit_model), ("calibration", cal_model))}
        self._heads = heads
        self._support = support
        self._regime = regime
        self._fit_groups = fit_groups
        self._calibration_groups = calibration_groups
        self._training_prompts = fit_prompts | calibration_prompts
        self._provenance = {part: self._data_hash(rows) for part, rows in (("fit", fitting), ("calibration", calibration))}
        self._versions = {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "joblib")}
        return self

    @staticmethod
    def _data_hash(rows):
        values = [{"request": row.request.to_dict(), "model": row.model, "success": row.success,
                   "source_group": row.source_group, "target": row.target,
                   "continuation_policy": row.continuation_policy, "assignment": row.assignment} for row in rows]
        return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False, separators=(",", ":")).encode()).hexdigest()

    def _fitted(self):
        if not hasattr(self, "_heads"):
            raise ValueError("Fit explicit observations before requesting probabilities.")

    def predict_requests(self, requests):
        self._fitted()
        requests = list(requests)
        if not requests:
            return []
        features = [state_features(request) for request in requests]
        outputs = [{} for _ in requests]
        for model, (pipeline, sigmoid) in self._heads.items():
            probabilities = sigmoid.predict_proba(_logits(pipeline.predict_proba(features)[:, 1]))[:, 1]
            for output, probability in zip(outputs, probabilities):
                output[model] = float(probability)
        return outputs

    def predict_proba(self, request):
        return self.predict_requests([request])[0]

    def evaluate(self, rows):
        """Development diagnostics only; all prior fitting groups stay excluded."""
        self._fitted()
        rows = _rows(rows, self.models)
        if ({row.source_group for row in rows} & (self._fit_groups | self._calibration_groups)
                or {normalize_for_dedup(row.request.prompt) for row in rows} & self._training_prompts):
            raise ValueError("Evaluation rows overlap fitting/calibration groups or prompts.")
        if any((row.target, row.continuation_policy, row.assignment) != self._regime for row in rows):
            raise ValueError("Evaluation target/continuation/assignment differ from training.")
        result = {}
        for model in self.models:
            selected = [row for row in rows if row.model == model]
            if not selected:
                result[model] = {"rows": 0, "brier": None, "log_loss": None}
                continue
            predicted = [self.predict_proba(row.request)[model] for row in selected]
            labels = [row.success for row in selected]
            result[model] = {"rows": len(selected), "brier": float(brier_score_loss(labels, predicted)),
                             "log_loss": float(log_loss(labels, predicted, labels=[False, True]))}
        return result

    def manifest(self):
        self._fitted()
        return {"kind": "state_success_tree_sigmoid_v1", "model_catalog": list(self.models),
                "target": self._regime[0], "continuation_policy": self._regime[1], "assignment": self._regime[2],
                "seed": self.seed, "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf,
                "support": json.loads(json.dumps(self._support)), "data_sha256": dict(self._provenance),
                "features_by_model": {model: list(head[0].named_steps["vectorizer"].get_feature_names_out())
                                      for model, head in self._heads.items()},
                "runtime_versions": dict(self._versions),
                "fitted_state_sha256": self._state_hash(),
                "scope": "action-success proxy with held-out sigmoid calibration; not final episode success or adoption evidence"}

    def _state_hash(self):
        # Hash the logical fitted state, rather than object serialization:
        # sklearn's tree pickle layout can differ after a faithful round trip.
        state = {}
        for model, (pipeline, sigmoid) in self._heads.items():
            tree = pipeline.named_steps["tree"]
            fitted = tree.tree_.__getstate__()
            state[model] = {
                "features": list(pipeline.named_steps["vectorizer"].get_feature_names_out()),
                "tree_params": tree.get_params(deep=False),
                "tree_nodes": fitted["nodes"].tolist(), "tree_values": fitted["values"].tolist(),
                "tree_classes": tree.classes_.tolist(),
                "sigmoid_coefficients": sigmoid.coef_.tolist(),
                "sigmoid_intercept": sigmoid.intercept_.tolist(),
                "sigmoid_classes": sigmoid.classes_.tolist(),
            }
        return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False,
                                         separators=(",", ":")).encode()).hexdigest()

    def save(self, path):
        """Save a new local artifact; return its hash for a separate trusted config."""
        self._fitted()
        path = Path(path)
        sidecar = path.with_name(path.name + ".manifest.json")
        if path.exists() or sidecar.exists():
            raise FileExistsError("Use a new artifact path; existing training artifacts are immutable.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps({"artifact_sha256": sha, "predictor": self.manifest()},
                                      ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        return sha

    @classmethod
    def load(cls, path, *, expected_sha256):
        """Only load a trusted local artifact after checking an external hash.

        A hash establishes file integrity, not trust in its author. Joblib can
        execute Python; an untrusted third-party artifact must not be loaded.
        """
        path = Path(path)
        if (type(expected_sha256) is not str or len(expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ValueError("Provide the SHA-256 from the trusted frozen configuration.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Artifact SHA-256 mismatch; deserialization was not attempted.")
        sidecar = json.loads(path.with_name(path.name + ".manifest.json").read_text(encoding="utf-8"))
        if sidecar.get("artifact_sha256") != expected_sha256:
            raise ValueError("Artifact sidecar does not match the trusted SHA-256.")
        expected_versions = {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "joblib")}
        if sidecar.get("predictor", {}).get("runtime_versions") != expected_versions:
            raise ValueError("Artifact runtime versions differ; deserialization was not attempted.")
        model = joblib.load(path)
        if not isinstance(model, cls) or model.manifest() != sidecar.get("predictor"):
            raise ValueError("Loaded predictor differs from its declared manifest.")
        return model
