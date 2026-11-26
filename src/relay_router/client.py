"""Local routing; generation belongs to the caller's harness.

Bundles use joblib and must come from a trusted source. A SHA-256 check detects
changes relative to the manifest; it does not make an untrusted pickle safe or
authenticate its author. This module neither discovers nor calls providers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import joblib
import pandas as pd


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and within [0, 1].")
    return float(value)


def _catalog(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"strong", "weak"}:
        raise ValueError("model_catalog must contain exactly strong and weak.")
    result = dict(value)
    if any(not isinstance(name, str) or not name.strip() for name in result.values()) or result["strong"] == result["weak"]:
        raise ValueError("The catalog requires two distinct, nonempty model names.")
    return result


@dataclass(frozen=True)
class RoutingDecision:
    """An auditable decision; predicted values are estimates, not guarantees.

    ``predicted_cost`` retains the training bundle's unit. It must not be read
    as a current API price or measured expenditure.
    """

    selected_model: str
    reason: str
    policy_mode: str
    router_seconds: float
    cost_unit: str = "unspecified"
    evaluation_scope: str = "unvalidated"
    harm_probability: float | None = None
    weak_success_probability: float | None = None
    predicted_cost: float | None = None
    expected_quality_gain: float | None = None
    operating_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelayRouter:
    """Route full prompt text to the exact model identities used in training.

    The harness supplies already serialized instructions, history and material
    as ``prompt`` and manages providers, context limits, retries and billing.
    ``available_models`` may restrict the catalog after those capability checks.
    Availability fallback overrides the quality gate and is identified as such;
    it does not preserve any quality measured for the complete catalog.
    """

    def __init__(self, decider: Any, model_catalog: Mapping[str, str], *,
                 max_harm: float = .05, min_weak_success: float = .8,
                 policy_mode: str = "guarded", cost_unit: str = "unspecified",
                 evaluation_scope: str = "unvalidated", tradeoff: float | None = None):
        self._catalog = _catalog(model_catalog)
        self.max_harm = _probability(max_harm, "max_harm")
        self.min_weak_success = _probability(min_weak_success, "min_weak_success")
        if not isinstance(policy_mode, str) or policy_mode not in {"guarded", "fixed_strong", "pairwise_utility"}:
            raise ValueError("Unsupported policy_mode.")
        if policy_mode == "pairwise_utility" and (isinstance(tradeoff, bool) or not isinstance(tradeoff, Real) or not isfinite(tradeoff)):
            raise ValueError("pairwise_utility requires a finite tradeoff score.")
        if any(not isinstance(value, str) or not value.strip() for value in (cost_unit, evaluation_scope)):
            raise ValueError("cost_unit and evaluation_scope must be nonempty text.")
        if not callable(getattr(decider, "predict_tables", None)):
            raise TypeError("decider must implement predict_tables.")
        for role, name in self._catalog.items():
            if getattr(decider, role, None) != name:
                raise ValueError("The catalog must match the trained model identities; remapping is unsupported.")
        self.decider = decider
        self.policy_mode = policy_mode
        self.cost_unit = cost_unit
        self.evaluation_scope = evaluation_scope
        self.tradeoff = tradeoff

    @property
    def model_catalog(self) -> dict[str, str]:
        return self._catalog.copy()

    def route(self, prompt: str, *, available_models: Iterable[str] | None = None) -> RoutingDecision:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text.")
        start = time.perf_counter()
        catalog = set(self._catalog.values())
        if available_models is None:
            available = catalog
        else:
            if isinstance(available_models, (str, bytes)):
                raise ValueError("available_models must be a collection of model names.")
            names = list(available_models)
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("available_models contains an invalid name.")
            available = catalog.intersection(names)
        if not available:
            raise ValueError("No model from the trained catalog is available.")
        if len(available) == 1:
            return RoutingDecision(next(iter(available)), "availability_fallback", self.policy_mode,
                                   time.perf_counter()-start, self.cost_unit, self.evaluation_scope)
        if self.policy_mode == "fixed_strong":
            return RoutingDecision(self._catalog["strong"], "fixed_strong", self.policy_mode,
                                   time.perf_counter()-start, self.cost_unit, self.evaluation_scope)

        # Import lazily: fixed fallback never constructs representations.
        from routerbench.guarded import choose_guarded

        inputs = pd.DataFrame({"prompt_text": [prompt]})
        predictions, costs = self.decider.predict_tables(inputs)
        if (not isinstance(predictions, pd.DataFrame) or not isinstance(costs, pd.DataFrame)
                or not predictions.index.equals(inputs.index) or not costs.index.equals(inputs.index)):
            raise ValueError("Predictions must contain exactly the requested row and index.")
        if self.policy_mode == "pairwise_utility":
            from routerbench.utility import choose_cost_utility
            selected = choose_cost_utility(predictions, costs, self._catalog["strong"], self._catalog["weak"], self.tradeoff)
        else:
            selected = choose_guarded(predictions, costs, self._catalog["strong"], self._catalog["weak"],
                                      self.max_harm, self.min_weak_success)
        row = selected.iloc[0]
        model = str(row["selected_model"])
        if model not in available:
            raise ValueError("The decision selected a model outside the available trained catalog.")
        return RoutingDecision(model, str(row["decision"]), self.policy_mode, time.perf_counter()-start,
                               self.cost_unit, self.evaluation_scope,
                               float(predictions.iloc[0]["harm_probability"]) if self.policy_mode == "guarded" else None,
                               float(predictions.iloc[0]["weak_success_probability"]) if self.policy_mode == "guarded" else None,
                               float(costs.loc[inputs.index[0], model]),
                               float(row["predicted_quality_gain"]) if self.policy_mode == "pairwise_utility" else None,
                               float(row["operating_score"]) if self.policy_mode == "pairwise_utility" else None)

    def run(self, prompt: str, generate: Callable[[str, str], Any], *,
            available_models: Iterable[str] | None = None) -> dict[str, Any]:
        """Choose once, then invoke ``generate(model_name, unchanged_prompt)`` once.

        Exceptions propagate to the harness. Relay does not retry a generation
        or silently switch models after observing its response.
        """
        if not callable(generate):
            raise TypeError("generate must be callable.")
        decision = self.route(prompt, available_models=available_models)
        response = generate(decision.selected_model, prompt)
        return {"decision": decision.to_dict(), "response": response}


def load_bundle(path: str | Path) -> RelayRouter:
    """Load a trusted local bundle after validating its manifest and artifact.

    Accept a manifest file or a directory containing ``bundle.json``. Never
    load a bundle from an untrusted author: joblib deserialization can execute
    code, including when its checksum matches a supplied manifest.
    """
    path = Path(path).resolve()
    manifest_path = path / "bundle.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "artifact", "artifact_sha256", "strong", "weak", "policy_mode"}
    if (not isinstance(manifest, dict) or not required.issubset(manifest)
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1):
        raise ValueError("Unsupported or incomplete Relay bundle manifest.")
    catalog = _catalog({role: manifest[role] for role in ("strong", "weak")})
    if manifest["policy_mode"] == "guarded" and not {"max_harm", "min_weak_success"}.issubset(manifest):
        raise ValueError("Guarded bundles require both probability thresholds.")
    max_harm = _probability(manifest.get("max_harm", .05), "max_harm")
    min_success = _probability(manifest.get("min_weak_success", .8), "min_weak_success")
    if not isinstance(manifest["policy_mode"], str) or manifest["policy_mode"] not in {"guarded", "fixed_strong", "pairwise_utility"}:
        raise ValueError("Unsupported bundle policy_mode.")
    tradeoff = manifest.get("tradeoff")
    if manifest["policy_mode"] == "pairwise_utility" and (isinstance(tradeoff, bool) or not isinstance(tradeoff, Real) or not isfinite(tradeoff)):
        raise ValueError("Utility bundles require a finite tradeoff before loading.")
    cost_unit, scope = manifest.get("cost_unit", "unspecified"), manifest.get("scope", "unvalidated")
    if any(not isinstance(value, str) or not value.strip() for value in (cost_unit, scope)):
        raise ValueError("Bundle cost_unit and scope must be nonempty text.")
    name, digest = manifest["artifact"], manifest["artifact_sha256"]
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("The artifact must be a relative path inside the bundle.")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("Invalid artifact_sha256.")
    directory = manifest_path.parent.resolve()
    artifact = (directory / name).resolve()
    if not artifact.is_relative_to(directory) or not artifact.is_file():
        raise ValueError("The artifact must be a file inside the bundle.")
    # Deserialize the exact bytes checked, avoiding a second read after hashing.
    from io import BytesIO

    payload = artifact.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("The artifact SHA-256 differs from the frozen manifest.")
    decider = joblib.load(BytesIO(payload))
    return RelayRouter(decider, catalog, max_harm=max_harm, min_weak_success=min_success,
                       policy_mode=manifest["policy_mode"], cost_unit=cost_unit, evaluation_scope=scope, tradeoff=tradeoff)
