"""Choose from frozen forecasts of final success and TOTAL remaining cost.

The forecasts describe a probe followed by a fixed reference continuation.
Using this selector repeatedly is a policy change requiring episode evaluation.
"""
from copy import deepcopy

from .v8 import _finite, _floor, _request, model_order

TARGET = "episode_success_under_reference_continuation"


class TrajectoryValuePolicy:
    name = "relay_v8_trajectory"

    def __init__(self, predictor, min_probability=.8, operating_point_source=""):
        self.predictor = predictor
        self.models = model_order(getattr(predictor, "models", None))
        if not callable(getattr(predictor, "predict_outcomes", None)):
            raise ValueError("A trajectory predictor with joint outcome forecasts is required.")
        self._predictor_manifest = deepcopy(predictor.manifest())
        if self._predictor_manifest.get("target") != TARGET:
            raise ValueError("A step-success predictor cannot be used as a trajectory predictor.")
        if not self._predictor_manifest.get("continuation_policy") or not self._predictor_manifest.get("cost_basis"):
            raise ValueError("Declare the reference continuation and cost basis.")
        self.min_probability = _finite(min_probability, "min_probability", 1)
        if type(operating_point_source) is not str or not operating_point_source.strip():
            raise ValueError("Document the source of the frozen operating point.")
        self.operating_point_source = operating_point_source
        self._operating_point = (self.min_probability, self.operating_point_source)
        self.last_audit = None

    @property
    def weak(self):
        return self.models[0]

    @property
    def strong(self):
        return self.models[-1]

    def _check(self):
        if (self.min_probability, self.operating_point_source) != self._operating_point:
            raise ValueError("The frozen trajectory operating point changed after policy construction.")
        if self.predictor.models != self.models or self.predictor.manifest() != self._predictor_manifest:
            raise ValueError("The fitted trajectory predictor changed after policy construction.")
        _finite(self.min_probability, "min_probability", 1)

    def select(self, request):
        self._check()
        floor = _floor(request, self.models)
        forecasts = self.predictor.predict_outcomes(_request(request).snapshot())
        if type(forecasts) is not dict or set(forecasts) != set(self.models):
            raise ValueError("Forecasts must cover exactly the frozen model catalog.")
        probabilities, costs = {}, {}
        for model, row in forecasts.items():
            if type(row) is not dict or set(row) != {"success_probability", "remaining_provider_cost_usd"}:
                raise ValueError("Each forecast must contain probability and total remaining cost.")
            probabilities[model] = _finite(row["success_probability"], "success_probability", 1)
            costs[model] = _finite(row["remaining_provider_cost_usd"], "remaining_provider_cost_usd")
        candidates = self.models[floor:]
        qualified = [model for model in candidates if probabilities[model] >= self.min_probability]
        if qualified:
            selected = min(qualified, key=lambda model: (costs[model], -probabilities[model], self.models.index(model)))
        else:
            selected = min(candidates, key=lambda model: (-probabilities[model], costs[model], self.models.index(model)))
        self.last_audit = {"selected_model": selected, "probabilities": probabilities,
                           "expected_remaining_provider_cost_usd": costs,
                           "minimum_tier": floor, "min_probability": self.min_probability,
                           "fallback": not qualified, "target": TARGET,
                           "cost_formula": "predicted total remaining cost; current attempt already included"}
        return selected

    def manifest(self):
        self._check()
        return deepcopy({"name": self.name, "catalog": {"strong": self.strong, "weak": self.weak},
                         "model_catalog": list(self.models), "predictor": self._predictor_manifest,
                         "min_probability": self.min_probability,
                         "operating_point_source": self.operating_point_source,
                         "cost_formula": "predicted total remaining cost; current attempt already included",
                         "scope": "reference-continuation estimates; repeated learned routing requires live episode validation"})
