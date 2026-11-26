"""Validate stored outcomes and construct retrospective cost-tier targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import pandas as pd

COST_SUFFIX = "|total_cost"


def discover_models(frame: pd.DataFrame) -> tuple[str, ...]:
    """Identify models from paired score and historical-cost columns."""
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate column names are ambiguous.")
    models = tuple(sorted(
        column[:-len(COST_SUFFIX)]
        for column in frame.columns
        if isinstance(column, str) and column.endswith(COST_SUFFIX)
    ))
    if not models:
        raise ValueError("No paired model cost columns were found.")
    missing = [model for model in models if model not in frame.columns]
    if missing:
        raise ValueError(f"Missing score columns for models: {missing}")
    return models


def outcome_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric score and cost tables with identical model columns.

    Reject missing, infinite, out-of-range scores and negative costs rather than
    silently treating corrupt observations as ineligible.
    """
    models = discover_models(frame)
    try:
        scores = frame.loc[:, list(models)].apply(pd.to_numeric, errors="raise").astype(float)
        costs = frame.loc[:, [model + COST_SUFFIX for model in models]].apply(
            pd.to_numeric, errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scores and costs must be numeric.") from exc
    costs.columns = models
    score_values = scores.to_numpy()
    cost_values = costs.to_numpy()
    if not np.isfinite(score_values).all() or not np.isfinite(cost_values).all():
        raise ValueError("Scores and costs must be finite and non-missing.")
    if ((score_values < 0) | (score_values > 1)).any():
        raise ValueError("Scores must be within [0, 1].")
    if (cost_values < 0).any():
        raise ValueError("Historical costs must be non-negative.")
    return scores, costs


def fit_tiers(
    train: pd.DataFrame, tier_sizes: Sequence[int] = (2, 6, 3)
) -> dict[str, int]:
    """Fit a model-to-tier map using training rows only.

    Call this after reserving test rows. Cost-median ties use model names in
    lexicographic order. The caller must persist this map and reuse it unchanged.
    """
    if len(train) == 0:
        raise ValueError("Cannot fit tiers on empty training data.")
    _, costs = outcome_tables(train)
    if (
        len(tier_sizes) != 3
        or any(not isinstance(size, Integral) or isinstance(size, bool) or size < 1 for size in tier_sizes)
        or sum(tier_sizes) != len(costs.columns)
    ):
        raise ValueError("Three positive tier sizes must sum to the number of models.")
    medians = costs.median(axis=0)
    ordered = sorted(costs.columns, key=lambda model: (medians[model], model))
    tier_map: dict[str, int] = {}
    offset = 0
    for tier, size in enumerate(tier_sizes):
        tier_map.update({model: tier for model in ordered[offset:offset + size]})
        offset += size
    return tier_map


def make_labels(
    frame: pd.DataFrame, tier_map: Mapping[str, int], threshold: float = 0.75
) -> pd.DataFrame:
    """Label the cheapest eligible model's tier, retaining ineligible rows.

    Per-row cost ties use model names in lexicographic order. The target is a
    retrospective oracle label, not a guarantee for a representative model.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The quality threshold must be finite and within [0, 1].")
    scores, costs = outcome_tables(frame)
    if set(tier_map) != set(scores.columns):
        raise ValueError("The tier map must match the observed model columns exactly.")
    if any(
        not isinstance(tier, Integral) or isinstance(tier, bool) or tier not in (0, 1, 2)
        for tier in tier_map.values()
    ):
        raise ValueError("Tier values must be integers 0, 1 or 2.")
    eligible_mask = scores.to_numpy() >= threshold
    has_eligible = eligible_mask.any(axis=1)
    ranked_costs = np.where(eligible_mask, costs.to_numpy(), np.inf)
    # np.argmin selects the first occurrence; outcome_tables sorts model names.
    choices = ranked_costs.argmin(axis=1)
    model_names = scores.columns.to_numpy()
    cheapest = pd.array(
        [model_names[choice] if valid else pd.NA for choice, valid in zip(choices, has_eligible)],
        dtype="string",
    )
    tiers = pd.array(
        [tier_map[model] if valid else pd.NA for model, valid in zip(cheapest, has_eligible)],
        dtype="Int64",
    )
    return pd.DataFrame(
        {"tier": tiers, "cheapest_model": cheapest, "eligible": has_eligible},
        index=frame.index,
    )
