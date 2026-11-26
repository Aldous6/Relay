"""N-model agent policies using explicit state and frozen cost forecasts.

Success probabilities refer to the predictor's declared target. They do not
guarantee final episode success. Retry costs are paid whether an attempt works
or fails; continuation forecasts contain only ADDITIONAL costs after failure.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

from .agent import AgentRequest


def model_order(models):
    if (type(models) is not tuple or len(models) < 2
            or any(type(name) is not str or not name.strip() or name != name.strip() for name in models)
            or len(set(models)) != len(models)):
        raise ValueError("models requires at least two unique exact names, ordered weak to strong.")
    return models


def _request(request):
    if not isinstance(request, AgentRequest):
        raise ValueError("A policy requires an AgentRequest.")
    return request


def _floor(request, models):
    state = _request(request).state
    failed = getattr(state, "failed_tiers", ())
    if any(type(tier) is not int or not 0 <= tier < len(models) for tier in failed):
        raise ValueError("failed_tiers contains an index outside the frozen catalog.")
    if failed:
        return min(max(failed) + 1, len(models) - 1)
    # Compatibility with the older state, which has no failed-tier metadata.
    if not hasattr(state, "failed_tiers"):
        semantic = sum(error.split(":", 1)[0] in {
            "InvalidAction", "ToolError", "VerificationFailed", "ProgressError", "BackendActionError"
        } for error in state.previous_errors)
        return min(semantic, len(models) - 1)
    return 0


def _finite(value, label, upper=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} requires a finite nonnegative number.")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
        raise ValueError(f"{label} is outside its declared range.")
    return value


class NReactiveCascade:
    """Stay at the next tier after a harness-observed semantic failure.

    Infrastructure errors never appear in failed_tiers and cannot escalate.
    This policy has no state shared between episodes.
    """

    name = "cheap_then_escalate"

    def __init__(self, models):
        self.models = model_order(models)

    @property
    def weak(self):
        return self.models[0]

    @property
    def strong(self):
        return self.models[-1]

    def select(self, request):
        return self.models[_floor(request, self.models)]

    def manifest(self):
        return {"name": self.name, "catalog": {"strong": self.strong, "weak": self.weak},
                "model_catalog": list(self.models), "rule": "next tier after observed semantic failure; infrastructure never escalates"}


class ExpectedCostPolicy:
    """Choose the lowest forecast cost among eligible probability estimates.

    E[C] = C_current + (1 - p_success) * C_additional_continuation.
    The horizon and continuation used for p and both forecasts must match.
    Estimates and the probability threshold are frozen development inputs.

    A declared protection rule raises the minimum eligible tier after every
    semantic failure. Learning anticipates failures ABOVE that fixed floor;
    it does not learn whether a failed tier should be retried. Compare against
    NReactiveCascade to isolate this predictive selection contribution.
    """

    name = "relay_v8_predictive"

    def __init__(self, predictor, expected_call_costs, expected_escalation_costs,
                 min_probability=.8, operating_point_source=""):
        self.predictor = predictor
        self.models = model_order(getattr(predictor, "models", None))
        if not callable(getattr(predictor, "predict_proba", None)) or not callable(getattr(predictor, "manifest", None)):
            raise ValueError("predictor requires predict_proba and manifest.")
        self.expected_call_costs = self._costs(expected_call_costs, "expected_call_costs")
        self.expected_escalation_costs = self._costs(expected_escalation_costs, "expected_escalation_costs")
        self.min_probability = _finite(min_probability, "min_probability", 1)
        if type(operating_point_source) is not str or not operating_point_source.strip():
            raise ValueError("Document the development source for the frozen threshold and forecasts.")
        self.operating_point_source = operating_point_source
        self._predictor_manifest = deepcopy(predictor.manifest())
        self.last_audit = None

    def _costs(self, values, label):
        if type(values) is not dict or set(values) != set(self.models):
            raise ValueError(f"{label} must cover exactly the frozen catalog.")
        return {model: _finite(values[model], label) for model in self.models}

    @property
    def weak(self):
        return self.models[0]

    @property
    def strong(self):
        return self.models[-1]

    def _check(self):
        if self.predictor.models != self.models or self.predictor.manifest() != self._predictor_manifest:
            raise ValueError("The fitted predictor changed after policy construction.")
        self._costs(self.expected_call_costs, "expected_call_costs")
        self._costs(self.expected_escalation_costs, "expected_escalation_costs")
        _finite(self.min_probability, "min_probability", 1)

    def select(self, request):
        self._check()
        floor = _floor(request, self.models)
        probabilities = self.predictor.predict_proba(_request(request).snapshot())
        if type(probabilities) is not dict or set(probabilities) != set(self.models):
            raise ValueError("The predictor must return one probability per exact model.")
        probabilities = {model: _finite(p, "success probability", 1) for model, p in probabilities.items()}
        eligible_models = self.models[floor:]
        expected = {model: self.expected_call_costs[model] +
                    (1 - probabilities[model]) * self.expected_escalation_costs[model]
                    for model in eligible_models}
        qualified = [model for model in eligible_models if probabilities[model] >= self.min_probability]
        fallback = not qualified
        if qualified:
            selected = min(qualified, key=lambda model: (expected[model], -probabilities[model], self.models.index(model)))
        else:
            selected = min(eligible_models, key=lambda model: (-probabilities[model], expected[model], self.models.index(model)))
        self.last_audit = {"selected_model": selected, "probabilities": probabilities,
                           "expected_cost_usd": expected, "minimum_tier": floor,
                           "min_probability": self.min_probability, "fallback": fallback,
                           "cost_formula": "current + (1-p) * additional_continuation",
                           "target": self._predictor_manifest.get("target")}
        return selected

    def manifest(self):
        self._check()
        return deepcopy({"name": self.name, "catalog": {"strong": self.strong, "weak": self.weak},
                         "model_catalog": list(self.models), "predictor": self._predictor_manifest,
                         "expected_call_costs_usd": self.expected_call_costs,
                         "expected_additional_continuation_costs_usd": self.expected_escalation_costs,
                         "min_probability": self.min_probability,
                         "operating_point_source": self.operating_point_source,
                         "cost_formula": "current + (1-p) * additional_continuation",
                         "scope": "frozen development forecasts; final episode quality requires independent measurement"})
