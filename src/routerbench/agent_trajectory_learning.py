"""Local text/state predictors for explicitly observed reference continuations.

The label is final episode success after choosing a model at an anchor and
following a declared continuation policy. It is not action validity, nor a
guarantee for repeatedly applying a different learned policy. Remaining cost
includes the anchor call and all continuation calls, under a declared USD basis.
No dataset, API, credential, or historical agent label is loaded implicitly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from relay_router.agent import AgentRequest
from relay_router.v8 import model_order
from .agent_learning import state_features
from .features import normalize_for_dedup


TARGET = "episode_success_under_reference_continuation"
ASSIGNMENTS = {"paired_state", "synthetic_fixture"}
FEATURE_CONTRACT_VERSION = "trajectory_public_text_state_v1"
PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib")
_PROVENANCE_KEYS = {
    "id", "task_id", "episode_id", "state_id", "anchor_id", "trace_id", "snapshot_id",
    "source_id", "group_id", "source_group", "family", "benchmark", "partition", "split",
    "model", "model_id", "requested_model", "returned_model", "provider_metadata",
    "usage", "cost", "cost_usd", "reported_llm_cost_usd", "score", "success",
    "final_success", "private_reference", "expected_spec", "test_inputs", "target",
}
_PROGRESS_KEYS = {
    "listed_files", "records_read", "inspected", "revision", "tested_revision",
    "files_read", "output_written", "public_tests_passed", "tool_actions",
    "repeated_tool_actions", "repetitions_prove_loop",
}


def _text_parts(value):
    """Preserve string content, including indentation; remove structured metadata."""
    if isinstance(value, dict):
        return [part for key in sorted(value) if key not in _PROVENANCE_KEYS
                for part in (key, *_text_parts(value[key]))]
    if isinstance(value, (list, tuple)):
        return [part for item in value for part in _text_parts(item)]
    if value is None:
        return []
    return [value if type(value) is str else str(value)]


def public_request_text(request):
    """Only supplied pre-decision task/history, never row labels or state metadata.

    This does not establish that an input transcript is authentic. The collector
    must verify public prefixes against their archived environment observations.
    Ordinary free-form task/file content is retained, not interpreted as code.
    """
    if not isinstance(request, AgentRequest):
        raise ValueError("Expected AgentRequest.")
    request = request.snapshot()
    parts = ["TASK", request.prompt, "PUBLIC_HISTORY"]
    for entry in request.history:
        content = entry["content"]
        if entry["role"] == "assistant" and type(content) is str:
            try:
                parsed = json.loads(content)
            except (ValueError, TypeError):
                parsed = content
            content = parsed
        parts.extend([entry["role"], *_text_parts(content)])
    return "\n".join(parts)


def trajectory_state_features(request):
    """Observable state only; infrastructure and selected model/tier are omitted."""
    result = state_features(request)
    result["errors_BackendActionError"] = sum(
        error.split(":", 1)[0] == "BackendActionError" for error in request.state.previous_errors)
    for key, value in getattr(request.state, "progress", {}).items():
        if key in _PROGRESS_KEYS and type(value) in (bool, int):
            result["progress:" + key] = int(value)
    return result


class _PublicText(BaseEstimator, TransformerMixin):
    def fit(self, requests, y=None):
        return self

    def transform(self, requests):
        return [public_request_text(request) for request in requests]


class _PublicState(BaseEstimator, TransformerMixin):
    def fit(self, requests, y=None):
        return self

    def transform(self, requests):
        return [trajectory_state_features(request) for request in requests]


@dataclass(frozen=True)
class TrajectoryTrainingRow:
    request: AgentRequest
    model: str
    success: bool
    remaining_provider_cost_usd: float | None
    source_group: str
    continuation_policy: str
    assignment: str
    target: str
    cost_basis: str

    def __post_init__(self):
        if not isinstance(self.request, AgentRequest) or type(self.success) is not bool:
            raise ValueError("A trajectory row requires AgentRequest and an observed boolean outcome.")
        for name in ("model", "source_group", "continuation_policy", "cost_basis"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(f"{name} requires an explicit nonempty exact string.")
        if self.target != TARGET or self.assignment not in ASSIGNMENTS:
            raise ValueError("Declare the trajectory target and supported assignment regime.")
        cost = self.remaining_provider_cost_usd
        if cost is not None:
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                raise ValueError("Remaining provider cost must be finite nonnegative USD or None.")
            object.__setattr__(self, "remaining_provider_cost_usd", float(cost))
        object.__setattr__(self, "request", self.request.snapshot())


def _rows(rows, models):
    values = list(rows)
    if not values or any(not isinstance(row, TrajectoryTrainingRow) or row.model not in models for row in values):
        raise ValueError("Nonempty trajectory rows must use exactly the declared model catalog.")
    values = [replace(row) for row in values]
    paired = {}
    for row in values:
        if row.assignment == "paired_state":
            key = (row.source_group, row.request.serialize())
            paired.setdefault(key, []).append(row.model)
    if any(len(names) != len(models) or set(names) != set(models) for names in paired.values()):
        raise ValueError("Each paired state must contain exactly one observed continuation per model.")
    return values


def _regime(row):
    return (row.target, row.continuation_policy, row.assignment, row.cost_basis)


def _groups(rows):
    return {row.source_group for row in rows}


def _prompts(rows):
    return {normalize_for_dedup(row.request.prompt) for row in rows}


def _logits(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


def _versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def _parameter_state(value):
    """Canonical JSON of supported estimator options; never serialize code.

    Runtime mutation is not an extension mechanism for a frozen predictor.
    Unsupported callbacks/opaque objects are rejected before transformation.
    """
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, BaseEstimator):
        return {"estimator": type(value).__module__ + "." + type(value).__qualname__,
                "parameters": _parameter_state(value.get_params(deep=False))}
    if type(value) in (list, tuple):
        return [_parameter_state(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _parameter_state(item) for key, item in value.items()}
    if isinstance(value, type) and value in (float, int, bool, str, np.float32, np.float64, np.int32, np.int64):
        return {"dtype": value.__module__ + "." + value.__qualname__}
    raise ValueError("Unsupported frozen inference parameter; construct and fit a new supported predictor.")


def _source_hash():
    # Include the imported state/normalization/request contracts: an unchanged
    # artifact must not silently adopt a different feature implementation.
    source = Path(__file__).resolve()
    dependencies = (source, source.with_name("agent_learning.py"), source.with_name("features.py"),
                    source.parent.parent / "relay_router" / "agent.py")
    digest = hashlib.sha256()
    for path in dependencies:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class TrajectorySuccessPredictor:
    """Sparse lexical/state outcome and cost heads with separate calibration.

    The text representation is lexical TF-IDF, not a pretrained semantic encoder.
    It reads full public text but cannot establish semantic comprehension. Fit,
    calibration and diagnostic development must have disjoint source groups and
    normalized task prompts. No missing outcome or cost is imputed as success/0.
    """

    def __init__(self, models, seed=42, max_text_features=12000,
                 success_regularization=1.0, cost_regularization=1.0):
        self.models = model_order(models)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be uint32.")
        if type(max_text_features) is not int or max_text_features < 1:
            raise ValueError("max_text_features must be a positive integer.")
        for value in (success_regularization, cost_regularization):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Regularization values must be positive finite numbers.")
        self.seed, self.max_text_features = seed, max_text_features
        self.success_regularization = float(success_regularization)
        self.cost_regularization = float(cost_regularization)

    def _new_features(self):
        return Pipeline([("union", FeatureUnion([
            ("text", Pipeline([("public", _PublicText()), ("tfidf", TfidfVectorizer(
                max_features=self.max_text_features, ngram_range=(1, 2), sublinear_tf=True,
                lowercase=True, dtype=np.float64))])),
            ("state", Pipeline([("public", _PublicState()), ("vectorizer", DictVectorizer(sparse=True)),
                                 ("scale", MaxAbsScaler())])),
        ]))])

    def fit(self, fit_rows, calibration_rows):
        fitting, calibration = _rows(fit_rows, self.models), _rows(calibration_rows, self.models)
        if _groups(fitting) & _groups(calibration):
            raise ValueError("Fit and calibration source groups must be disjoint.")
        if _prompts(fitting) & _prompts(calibration):
            raise ValueError("Duplicate task prompts cannot cross fit and calibration.")
        regimes = {_regime(row) for row in fitting + calibration}
        if len(regimes) != 1:
            raise ValueError("All rows must share target, continuation, assignment and cost basis.")
        support = {}
        for model in self.models:
            model_fit = [row for row in fitting if row.model == model]
            model_cal = [row for row in calibration if row.model == model]
            if any({row.success for row in rows} != {False, True} for rows in (model_fit, model_cal)):
                raise ValueError(f"Both classes are required in fit and calibration for {model}; collect observations.")
            if sum(row.remaining_provider_cost_usd is not None for row in model_fit) < 2:
                raise ValueError(f"At least two known fit costs are required for {model}; unknown is not zero.")
            support[model] = {part: {"rows": len(rows), "positive": sum(row.success for row in rows),
                                    "groups": len(_groups(rows)),
                                    "known_cost_rows": sum(row.remaining_provider_cost_usd is not None for row in rows)}
                              for part, rows in (("fit", model_fit), ("calibration", model_cal))}
        # Deduplicate representation input by the public request, independently of
        # outcomes; paired candidates otherwise repeat the same state N times.
        unique_fit = {row.request.serialize(): row.request for row in fitting}
        features = self._new_features()
        features.fit([unique_fit[key] for key in sorted(unique_fit)])
        heads = {}
        for model in self.models:
            model_fit = [row for row in fitting if row.model == model]
            model_cal = [row for row in calibration if row.model == model]
            x_fit = features.transform([row.request for row in model_fit])
            x_cal = features.transform([row.request for row in model_cal])
            success = LogisticRegression(C=self.success_regularization, solver="liblinear",
                                         max_iter=1500, random_state=self.seed)
            success.fit(x_fit, [row.success for row in model_fit])
            calibration_head = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.seed)
            calibration_head.fit(_logits(success.predict_proba(x_cal)[:, 1]), [row.success for row in model_cal])
            known = [index for index, row in enumerate(model_fit) if row.remaining_provider_cost_usd is not None]
            cost = Ridge(alpha=self.cost_regularization, solver="lsqr")
            cost.fit(x_fit[known], [model_fit[index].remaining_provider_cost_usd for index in known])
            heads[model] = (success, calibration_head, cost)
        self._features, self._heads, self._support = features, heads, support
        self._regime = next(iter(regimes))
        self._fit_groups, self._calibration_groups = _groups(fitting), _groups(calibration)
        self._training_prompts = _prompts(fitting) | _prompts(calibration)
        self._development_groups, self._development_prompts = set(), set()
        self._provenance = {name: self._data_hash(rows) for name, rows in (("fit", fitting), ("calibration", calibration))}
        self._versions, self._feature_source_sha256 = _versions(), _source_hash()
        return self

    @staticmethod
    def _data_hash(rows):
        values = [{**{key: getattr(row, key) for key in TrajectoryTrainingRow.__dataclass_fields__ if key != "request"},
                   "request": row.request.to_dict()} for row in rows]
        return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False, separators=(",", ":")).encode()).hexdigest()

    def _fitted(self):
        if not hasattr(self, "_heads"):
            raise ValueError("Fit explicit observed continuations before inference.")

    def _predict(self, requests):
        self._fitted()
        requests = list(requests)
        if not requests:
            return [], []
        if any(not isinstance(request, AgentRequest) for request in requests):
            raise ValueError("Predictions require AgentRequest inputs.")
        matrix = self._features.transform([request.snapshot() for request in requests])
        probabilities, costs = [{} for _ in requests], [{} for _ in requests]
        for model in self.models:
            success, calibrated, cost = self._heads[model]
            p = calibrated.predict_proba(_logits(success.predict_proba(matrix)[:, 1]))[:, 1]
            usd = np.maximum(0., cost.predict(matrix))
            if not np.isfinite(p).all() or not np.isfinite(usd).all():
                raise ValueError("Predictor produced a nonfinite forecast.")
            for output_p, output_c, value_p, value_c in zip(probabilities, costs, p, usd):
                output_p[model], output_c[model] = float(value_p), float(value_c)
        return probabilities, costs

    def predict_proba(self, request):
        return self._predict([request])[0][0]

    def predict_costs(self, request):
        return self._predict([request])[1][0]

    def predict_outcomes(self, request):
        probabilities, costs = self._predict([request])
        return {model: {"success_probability": probabilities[0][model],
                        "remaining_provider_cost_usd": costs[0][model]} for model in self.models}

    def predict_requests(self, requests):
        return self._predict(requests)[0]

    def evaluate(self, rows):
        """Held-out development diagnostics; not a learned-policy episode rollout."""
        self._fitted()
        rows = _rows(rows, self.models)
        if (_groups(rows) & (self._fit_groups | self._calibration_groups)
                or _prompts(rows) & self._training_prompts):
            raise ValueError("Development rows overlap fit/calibration groups or task prompts.")
        if any(_regime(row) != self._regime for row in rows):
            raise ValueError("Development target/continuation/assignment/cost basis differ from fit.")
        result = {}
        for model in self.models:
            selected = [row for row in rows if row.model == model]
            if not selected:
                result[model] = {"rows": 0, "brier": None, "log_loss": None,
                                 "known_cost_rows": 0, "unknown_cost_rows": 0, "remaining_cost_mae_usd": None}
                continue
            probabilities, costs = self._predict([row.request for row in selected])
            p = [row[model] for row in probabilities]
            labels = [row.success for row in selected]
            known = [index for index, row in enumerate(selected) if row.remaining_provider_cost_usd is not None]
            result[model] = {"rows": len(selected), "brier": float(brier_score_loss(labels, p)),
                             "log_loss": float(log_loss(labels, p, labels=[False, True])),
                             "known_cost_rows": len(known), "unknown_cost_rows": len(selected) - len(known),
                             "remaining_cost_mae_usd": float(mean_absolute_error(
                                 [selected[i].remaining_provider_cost_usd for i in known],
                                 [costs[i][model] for i in known])) if known else None}
        self._development_groups.update(_groups(rows))
        self._development_prompts.update(_prompts(rows))
        return result

    def audit_task_groups(self, tasks, *, evaluation_split="holdout"):
        """Prevent training reuse and identify previously inspected development.

        A passing audit establishes only disjointness against supplied training
        provenance, not population independence or a publication claim.
        """
        self._fitted()
        if evaluation_split not in ("development", "holdout"):
            raise ValueError("evaluation_split must be development or holdout.")
        tasks = list(tasks)
        if not tasks or any(type(task) is not dict or any(
                type(task.get(key)) is not str or not task[key].strip()
                for key in ("source_group", "prompt")) for task in tasks):
            raise ValueError("Audit requires tasks with explicit source_group and prompt.")
        groups = {task["source_group"] for task in tasks}
        prompts = {normalize_for_dedup(task["prompt"]) for task in tasks}
        if groups & (self._fit_groups | self._calibration_groups) or prompts & self._training_prompts:
            raise ValueError("Evaluation tasks overlap fit/calibration groups or task prompts.")
        if evaluation_split == "holdout" and (
                groups & self._development_groups or prompts & self._development_prompts):
            raise ValueError("Holdout tasks overlap previously inspected development groups or prompts.")
        return {"evaluation_split": evaluation_split, "tasks": len(tasks), "source_groups": len(groups),
                "fit_calibration_overlap": False, "independent_validation": False,
                "scope": "supplied task groups and normalized prompt provenance only"}

    def _state_hash(self):
        union = dict(self._features.named_steps["union"].transformer_list)
        text, state = union["text"].named_steps["tfidf"], union["state"]
        digest = hashlib.sha256(json.dumps({
            "feature_parameters": _parameter_state(self._features),
            # TfidfVectorizer constructs this fitted transformer at fit time.
            # Its effective norm/sublinear_tf/use_idf may differ from the outer
            # constructor options after mutation; both belong to the freeze.
            "effective_tfidf_parameters": _parameter_state(text._tfidf),
            "head_parameters": {model: [_parameter_state(head) for head in self._heads[model]]
                                for model in self.models},
            "text_vocabulary": {key: int(value) for key, value in text.vocabulary_.items()},
            "state_vocabulary": {key: int(value) for key, value in state.named_steps["vectorizer"].vocabulary_.items()}},
            sort_keys=True, allow_nan=False).encode())
        def array(value):
            value = np.ascontiguousarray(value)
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        array(text.idf_)
        array(state.named_steps["scale"].scale_)
        for model in self.models:
            for head in self._heads[model]:
                array(head.coef_)
                array(head.intercept_)
                if hasattr(head, "classes_"):
                    array(head.classes_)
        return digest.hexdigest()

    def manifest(self):
        self._fitted()
        union = dict(self._features.named_steps["union"].transformer_list)
        return {"kind": "trajectory_text_state_logistic_ridge_v1", "model_catalog": list(self.models),
                "target": self._regime[0], "continuation_policy": self._regime[1],
                "assignment": self._regime[2], "cost_basis": self._regime[3],
                "seed": self.seed, "max_text_features": self.max_text_features,
                "success_regularization": self.success_regularization, "cost_regularization": self.cost_regularization,
                "features": {"contract": FEATURE_CONTRACT_VERSION, "representation": "word TF-IDF 1-2grams plus scaled public state",
                    "text_vocabulary_size": len(union["text"].named_steps["tfidf"].vocabulary_),
                    "state_feature_names": list(union["state"].named_steps["vectorizer"].get_feature_names_out()),
                    "input_text_truncated": False, "preprocessing_fit_only": True,
                    "raw_string_whitespace_preserved": True, "whitespace_not_encoded_by_word_tokenizer": True,
                    "excluded_structured_metadata_keys": sorted(_PROVENANCE_KEYS)},
                "support": deepcopy(self._support), "data_sha256": dict(self._provenance),
                "partition_provenance": {"fit_group_sha256": sorted(hashlib.sha256(value.encode()).hexdigest() for value in self._fit_groups),
                    "calibration_group_sha256": sorted(hashlib.sha256(value.encode()).hexdigest() for value in self._calibration_groups),
                    "fit_calibration_prompt_sha256": sorted(hashlib.sha256(value.encode()).hexdigest() for value in self._training_prompts),
                    "development_group_sha256": sorted(hashlib.sha256(value.encode()).hexdigest() for value in self._development_groups),
                    "development_prompt_sha256": sorted(hashlib.sha256(value.encode()).hexdigest() for value in self._development_prompts)},
                "runtime_versions": dict(self._versions), "feature_source_sha256": self._feature_source_sha256,
                "fitted_state_sha256": self._state_hash(),
                "cost_target_scope": "total provider USD remaining including current call under reference continuation; not full operating cost",
                "missing_cost_handling": "retain for success head; exclude unknown fit targets from cost regression; report coverage",
                "calibration_scope": "separate-group sigmoid for reference-continuation success; cost head is uncalibrated ridge clamped at zero",
                "scope": "reference-continuation forecasts; deploying a different repeated routing policy requires new episode evaluation"}

    def save(self, path):
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
        """Hash/version/source checks precede joblib; caller must trust the author."""
        path = Path(path)
        if (type(expected_sha256) is not str or len(expected_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha256)):
            raise ValueError("Provide an externally trusted SHA-256.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Artifact SHA-256 mismatch; deserialization was not attempted.")
        sidecar = json.loads(path.with_name(path.name + ".manifest.json").read_text(encoding="utf-8"))
        declared = sidecar.get("predictor", {})
        if sidecar.get("artifact_sha256") != expected_sha256:
            raise ValueError("Artifact sidecar does not match the trusted SHA-256.")
        if declared.get("runtime_versions") != _versions() or declared.get("feature_source_sha256") != _source_hash():
            raise ValueError("Artifact runtime versions or feature source differ; deserialization was not attempted.")
        model = joblib.load(path)
        if not isinstance(model, cls) or model.manifest() != declared:
            raise ValueError("Loaded predictor differs from its declared manifest.")
        return model
