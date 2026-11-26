"""A bounded, explicit adaptation recipe; selection never sees validation."""
from __future__ import annotations
import time

from .adaptive_policy import train_policy


def train_adaptation(records, *, baseline_model, candidate_model, scope="openhands-python-v1", scorer=None):
    """Compare at most two fixed representations on historical calibration.

    A provided resident BERT engine adds its frozen prompt representation. Both
    candidates use the same fit/calibration groups, regularization and gate.
    All candidate artifacts are returned for audit; no validation is consumed.
    """
    records = list(records)
    started = time.perf_counter()
    candidates = [train_policy(records, baseline_model=baseline_model,
                               candidate_model=candidate_model, scope=scope)]
    if scorer is not None:
        candidates.append(train_policy(records, baseline_model=baseline_model, candidate_model=candidate_model,
                                       scope=scope, scorer=scorer, representation="routellm-bert"))
    qualified = [policy for policy in candidates if policy["calibration"]["reason"] == "candidate_qualified_on_calibration"]
    selected = min(qualified, key=lambda policy: policy["calibration"]["relay"]["cost_per_accepted_task_usd"]) if qualified else candidates[0]
    return selected, {
        "kind": "relay_automatic_adaptation_selection", "selected_artifact_id": selected["artifact_id"],
        "selection_scope": "historical_calibration_only_no_fresh_validation_outcomes",
        "selection_rule": "minimum_complete_cost_per_accepted_among_calibration_qualified_candidates; structural_first_on_tie",
        "reason": "calibration_candidate_selected" if qualified else "no_candidate_qualified_keep_baseline",
        "candidates": candidates, "elapsed_seconds": time.perf_counter() - started,
        "compute_cost_usd": None, "compute_cost_included_in_savings": False,
        "scorer_provenance": getattr(scorer, "provenance", None),
        "limitation": "Selection on development does not establish savings; a fresh independent validation is required.",
    }
