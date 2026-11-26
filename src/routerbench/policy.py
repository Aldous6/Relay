"""Select models from predictions and evaluate the resulting offline calls.

Selection never receives realized outcomes. Capability and context compatibility
belong in ``allowed_models``; an empty candidate set yields an abstention rather
than an implicit context trim or an unsupported call.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


def _threshold(value: float, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and within [0, 1].")
    return float(value)


def _validate_axes(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame.")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"{name} must have unique row and model identifiers.")
    if not len(frame.columns) or any(not isinstance(model, str) or not model for model in frame.columns):
        raise ValueError(f"{name} needs nonempty string model identifiers.")


def _same_axes(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    _validate_axes(other, name)
    if not reference.index.equals(other.index):
        raise ValueError(f"{name} row indices must match exactly, including order.")
    if set(reference.columns) != set(other.columns):
        raise ValueError(f"{name} model columns must match exactly.")


def _numeric_tables(
    values: pd.DataFrame, costs: pd.DataFrame, value_name: str, cost_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_axes(values, value_name)
    _same_axes(values, costs, cost_name)
    names = sorted(values.columns)
    try:
        numeric_values = values.loc[:, names].apply(pd.to_numeric, errors="raise").astype(float)
        numeric_costs = costs.loc[:, names].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions, scores and costs must be numeric.") from exc
    value_array = numeric_values.to_numpy()
    cost_array = numeric_costs.to_numpy()
    if not np.isfinite(value_array).all() or ((value_array < 0) | (value_array > 1)).any():
        raise ValueError(f"{value_name} must be finite and within [0, 1].")
    if not np.isfinite(cost_array).all() or (cost_array < 0).any():
        raise ValueError(f"{cost_name} must be finite and non-negative.")
    return numeric_values, numeric_costs


def choose_models(
    probabilities: pd.DataFrame,
    predicted_costs: pd.DataFrame,
    quality_probability_threshold: float = 0.75,
    allowed_models: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Choose the cheapest eligible model meeting a predicted-success target.

    If none meets the target, choose the eligible model with greatest predicted
    success, then lowest predicted cost. Model-name order breaks remaining ties.
    If none is eligible, return nullable predictions and ``decision='abstain'``.
    Probabilities must come from a separately fitted predictor; this rule does
    not establish their calibration or guarantee the requested success rate.
    Row indices are aligned strictly. Column order is immaterial.
    """
    target = _threshold(quality_probability_threshold, "quality_probability_threshold")
    probabilities, predicted_costs = _numeric_tables(
        probabilities, predicted_costs, "probabilities", "predicted_costs"
    )
    if allowed_models is None:
        allowed = np.ones(probabilities.shape, dtype=bool)
    else:
        _same_axes(probabilities, allowed_models, "allowed_models")
        if any(not is_bool_dtype(dtype) for dtype in allowed_models.dtypes) or allowed_models.isna().any().any():
            raise ValueError("allowed_models must contain only non-missing booleans.")
        allowed = allowed_models.loc[:, probabilities.columns].to_numpy(dtype=bool)

    probability_array = probabilities.to_numpy()
    cost_array = predicted_costs.to_numpy()
    model_names = probabilities.columns.to_numpy()
    selected: list[str | None] = []
    selected_probabilities: list[float | None] = []
    selected_costs: list[float | None] = []
    decisions: list[str] = []
    for row_number in range(len(probabilities)):
        candidates = np.flatnonzero(allowed[row_number])
        if not len(candidates):
            selected.append(None)
            selected_probabilities.append(None)
            selected_costs.append(None)
            decisions.append("abstain")
            continue
        above_target = candidates[probability_array[row_number, candidates] >= target]
        if len(above_target):
            finalists = above_target
            decision = "threshold_met"
        else:
            highest_probability = probability_array[row_number, candidates].max()
            finalists = candidates[probability_array[row_number, candidates] == highest_probability]
            decision = "fallback"
        # Columns and candidate indices are sorted by model name, so argmin
        # preserves deterministic model-name ties without consulting outcomes.
        chosen = finalists[np.argmin(cost_array[row_number, finalists])]
        selected.append(model_names[chosen])
        selected_probabilities.append(float(probability_array[row_number, chosen]))
        selected_costs.append(float(cost_array[row_number, chosen]))
        decisions.append(decision)

    return pd.DataFrame(
        {
            "selected_model": pd.array(selected, dtype="string"),
            "predicted_success": pd.array(selected_probabilities, dtype="Float64"),
            "predicted_cost": pd.array(selected_costs, dtype="Float64"),
            "decision": pd.array(decisions, dtype="string"),
        },
        index=probabilities.index,
    )


def evaluate_policy(
    selected_models: pd.Series,
    scores: pd.DataFrame,
    costs: pd.DataFrame,
    threshold: float = 0.75,
) -> dict:
    """Evaluate exactly the selected calls on every supplied prompt.

    Returns ``{'metrics': dict, 'rows': DataFrame}``. Outcomes have plain model
    names as columns, as returned by ``labels.outcome_tables``. Missing selected
    models are abstentions: they incur zero stored-call cost and count as failed
    requests in the overall success rate. Coverage and conditional success are
    reported separately. Costs exclude router compute, retries and all other
    calls; this is retrospective evaluation, not a net production savings claim.
    """
    target = _threshold(threshold, "threshold")
    scores, costs = _numeric_tables(scores, costs, "scores", "costs")
    if not len(scores):
        raise ValueError("Cannot evaluate an empty collection of prompts.")
    if not isinstance(selected_models, pd.Series):
        raise TypeError("selected_models must be a Series.")
    if not selected_models.index.equals(scores.index):
        raise ValueError("Selected-model row indices must match outcomes exactly, including order.")
    non_missing = selected_models.dropna()
    if any(not isinstance(model, str) for model in non_missing):
        raise ValueError("Selected models must be model names or missing values.")
    unknown = sorted(set(non_missing) - set(scores.columns))
    if unknown:
        raise ValueError(f"Unknown selected model names: {unknown}")

    executed = selected_models.notna().to_numpy()
    realized_scores = np.full(len(scores), np.nan)
    realized_costs = np.zeros(len(scores), dtype=float)
    model_positions = {name: position for position, name in enumerate(scores.columns)}
    row_positions = np.flatnonzero(executed)
    column_positions = np.array([model_positions[name] for name in selected_models.iloc[row_positions]], dtype=int)
    realized_scores[row_positions] = scores.to_numpy()[row_positions, column_positions]
    realized_costs[row_positions] = costs.to_numpy()[row_positions, column_positions]
    successes = executed & (realized_scores >= target)
    total = len(scores)
    calls = int(executed.sum())
    total_successes = int(successes.sum())
    total_cost = float(realized_costs.sum())
    rows = pd.DataFrame(
        {
            "selected_model": pd.array(selected_models, dtype="string"),
            "score": pd.array(realized_scores, dtype="Float64"),
            "success": successes,
            "cost": realized_costs,
            "executed": executed,
        },
        index=scores.index,
    )
    metrics = {
        "n_prompts": total,
        "n_routed": calls,
        "n_abstained": total - calls,
        "n_successes": total_successes,
        "coverage": calls / total,
        "success_rate": total_successes / total,
        "success_rate_on_routed": total_successes / calls if calls else None,
        "mean_score_on_routed": float(realized_scores[executed].mean()) if calls else None,
        "total_cost": total_cost,
        "cost_per_1000": 1000 * total_cost / total,
        "quality_score_threshold": target,
        "cost_scope": "stored selected-model call costs only; excludes router compute and retries",
    }
    return {"metrics": metrics, "rows": rows}
