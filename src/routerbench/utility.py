"""Shared inference policy for the fixed, positive-extra-cost v7 experiment."""
from numbers import Real

import numpy as np
import pandas as pd

from .pairwise import choose_pairwise


def cost_utility_score(predictions, costs, strong, weak):
    # Reuse the pairwise table boundary, including simplex and declared gain.
    checked = choose_pairwise(predictions, costs, strong, weak, 0.)
    extra = checked.predicted_extra_cost
    if (extra <= 0).any():
        raise ValueError("La política de cociente exige costo fuerte previsto estrictamente mayor.")
    result = checked.predicted_quality_gain / extra
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("El cociente de ganancia/costo no es finito.")
    return result.rename("operating_score")


def choose_cost_utility(predictions, costs, strong, weak, tradeoff):
    """Choose strong iff signed gain/positive extra cost exceeds the cutoff.

    The cutoff is an operating score, not a probability. Negative values are
    permitted to reproduce the outcome-free quantile grid, including endpoints.
    Predicted costs must retain the training unit; current tariffs are not used.
    """
    if (not isinstance(tradeoff, Real) or isinstance(tradeoff, (bool, np.bool_))
            or not np.isfinite(tradeoff)):
        raise ValueError("tradeoff debe ser un número real finito.")
    score = cost_utility_score(predictions, costs, strong, weak)
    strong_selected = score.gt(float(tradeoff))
    return pd.DataFrame({
        "selected_model": pd.array(np.where(strong_selected, strong, weak), dtype="string"),
        "decision": np.where(strong_selected, "utility_strong", "utility_weak"),
        "predicted_quality_gain": predictions.strong_only-predictions.weak_only,
        "predicted_extra_cost": costs[strong]-costs[weak],
        "operating_score": score,
    }, index=predictions.index)
