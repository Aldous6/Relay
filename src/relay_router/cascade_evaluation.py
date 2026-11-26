"""Offline, paired analysis of verified model cascades; never a runtime selector.

Routing consumes operational checks only. Independent acceptance labels are
read after the route is fixed. Historical replay is development evidence, not
authorization to activate a policy or a claim of end-to-end monetary savings.
This module deliberately uses only the Python standard library.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random


BOOTSTRAP_SEED = 20261118
BOOTSTRAP_RESAMPLES = 10000


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _probability(value, name):
    value = _number(value, name)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return value


def exact_harm_upper(losses, groups, *, confidence=.95):
    """One-sided Clopper-Pearson upper bound on P(baseline passes, policy fails).

    Groups must be independent Bernoulli sampling units for coverage to apply.
    An observed gain never cancels an observed loss in this estimand.
    """
    if type(groups) is not int or groups < 1 or type(losses) is not int or not 0 <= losses <= groups:
        raise ValueError("Require 0 <= integer losses <= integer groups, groups >= 1")
    confidence = _probability(confidence, "confidence")
    alpha = 1 - confidence
    if losses == groups:
        return 1.
    if losses == 0:
        return -math.expm1(math.log(alpha) / groups)
    coefficients = [math.lgamma(groups + 1) - math.lgamma(k + 1) - math.lgamma(groups - k + 1)
                    for k in range(losses + 1)]
    low, high = 0., 1.
    for _ in range(70):
        p = (low + high) / 2
        terms = [coefficient + k * math.log(p) + (groups - k) * math.log1p(-p)
                 for k, coefficient in enumerate(coefficients)]
        largest = max(terms)
        cdf = math.exp(largest) * math.fsum(math.exp(term - largest) for term in terms)
        if cdf > alpha:
            low = p
        else:
            high = p
    return (low + high) / 2


def zero_loss_required_groups(max_harm=.02, *, confidence=.95):
    """Minimum n with an exact one-sided bound <= max_harm, IF zero losses."""
    max_harm = _probability(max_harm, "max_harm")
    confidence = _probability(confidence, "confidence")
    return math.ceil(math.log1p(-confidence) / math.log1p(-max_harm))


def _cost(row):
    value = row.get("cost_usd")
    complete = row.get("cost_complete")
    basis = row.get("cost_basis")
    if type(complete) is not bool:
        raise ValueError("cost_complete must be an explicit boolean")
    if value is None:
        if complete:
            raise ValueError("Missing cost cannot be complete")
        return None
    value = _number(value, "cost_usd")
    if not complete or not isinstance(basis, str) or not basis or basis == "unknown":
        raise ValueError("Numeric episode cost requires a complete, explicit basis")
    return value


def _cost_range(row):
    """An unknown charge may have a documented ceiling on the same price basis."""
    point = _cost(row)
    lower = _number(row.get("known_cost_subtotal_usd", point if point is not None else 0.), "known_cost_subtotal_usd")
    upper = row.get("cost_upper_bound_usd", point)
    basis = row.get("cost_bound_basis", row.get("cost_basis") if point is not None else None)
    if upper is not None:
        upper = _number(upper, "cost_upper_bound_usd")
        if upper < lower or not isinstance(basis, str) or not basis or basis == "unknown":
            raise ValueError("A cost ceiling requires a comparable price basis and must cover known charges")
    if point is not None and (lower != point or upper != point or basis != row["cost_basis"]):
        raise ValueError("A complete charge must agree with its subtotal, bounds and price basis")
    return lower, upper, basis


def _sum_cost_ranges(rows):
    ranges = [_cost_range(row) for row in rows]
    lower = math.fsum(entry[0] for entry in ranges)
    bases = {entry[2] for entry in ranges if entry[2] is not None}
    mixed = len(bases) > 1 or any(row.get("mixed_cost_bases", False) for row in rows)
    basis = next(iter(bases)) if len(bases) == 1 and not mixed else None
    upper = (math.fsum(entry[1] for entry in ranges)
             if ranges and basis is not None and all(entry[1] is not None for entry in ranges) else None)
    return lower, upper, basis, mixed


def _index(records, candidate_model, baseline_model):
    if not all(isinstance(model, str) and model for model in (candidate_model, baseline_model)) or candidate_model == baseline_model:
        raise ValueError("Two distinct explicit models are required")
    by_task = {}
    for source in records:
        row = deepcopy(source)
        for field in ("task_id", "group_id", "model"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"An explicit {field} is required")
        if row["model"] not in (candidate_model, baseline_model):
            raise ValueError("Unexpected model in paired cohort")
        for field in ("accepted", "public_passed"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"{field} must be an explicit boolean; missing cases cannot be dropped")
        _cost(row)
        pair = by_task.setdefault(row["task_id"], {})
        if row["model"] in pair:
            raise ValueError("Duplicate task/model episode")
        pair[row["model"]] = row
    if not by_task:
        raise ValueError("The paired cohort is empty")
    groups = set()
    for pair in by_task.values():
        if set(pair) != {candidate_model, baseline_model}:
            raise ValueError("Every task requires both model episodes, including failures")
        group = pair[candidate_model]["group_id"]
        if group != pair[baseline_model]["group_id"] or group in groups:
            raise ValueError("Require matching pairs and one task per deduplicated prompt group")
        groups.add(group)
    return dict(sorted(by_task.items()))


def operational_cascade_route(*, candidate_passed):
    """Small auditable decision boundary: no acceptance label argument exists."""
    if type(candidate_passed) is not bool:
        raise ValueError("The operational verifier must return a boolean")
    return "candidate" if candidate_passed else "baseline"


def _attempt_row(attempts):
    final = attempts[-1]
    costs = [_cost(row) for row in attempts]
    bases = {row["cost_basis"] for row in attempts}
    complete = all(cost is not None for cost in costs) and len(bases) == 1
    lower, upper, basis, mixed = _sum_cost_ranges(attempts)
    result = {"task_id": final["task_id"], "group_id": final["group_id"],
              "accepted": final["accepted"], "operational_passed": final["public_passed"],
              "final_model": final["model"], "attempted_models": [row["model"] for row in attempts],
              "cost_usd": math.fsum(costs) if complete else None,
              "cost_complete": complete, "cost_basis": next(iter(bases)) if complete else "unknown",
              "known_cost_subtotal_usd": lower, "cost_upper_bound_usd": upper,
              "cost_bound_basis": basis, "mixed_cost_bases": mixed,
              "http_attempts_complete": all(row.get("http_attempts_complete", True) for row in attempts),
              "escalated": len(attempts) > 1}
    for field in ("http_attempts", "logical_calls"):
        values = [row.get(field) for row in attempts]
        result[field] = sum(values) if all(type(value) is int and value >= 0 for value in values) else None
    return result


def summarize_policy(rows):
    """All task attempts contribute to the numerator, including failed tasks."""
    costs = [_cost(row) for row in rows]
    bases = {row["cost_basis"] for row in rows}
    complete = bool(rows) and all(cost is not None for cost in costs) and len(bases) == 1
    accepted = sum(row["accepted"] for row in rows)
    total = math.fsum(costs) if complete else None
    lower, upper, basis, mixed = _sum_cost_ranges(rows)
    return {"tasks": len(rows), "accepted": accepted,
            "acceptance_rate": accepted / len(rows) if rows else None,
            "total_cost_usd": total, "cost_per_accepted_usd": total / accepted if complete and accepted else None,
            "cost_complete": complete, "cost_basis": next(iter(bases)) if complete else "unknown",
            "known_cost_subtotal_usd": lower, "cost_bound_basis": basis,
            "total_cost_bounds_usd": [lower, upper] if rows and not mixed else None,
            "cost_per_accepted_bounds_usd": ([lower / accepted, upper / accepted if upper is not None else None]
                                             if accepted and not mixed else None),
            "unknown_task_costs": sum(cost is None for cost in costs),
            "mixed_cost_bases": mixed,
            "escalated_tasks": sum(row.get("escalated", False) for row in rows)}


def _percentile(sorted_values, q):
    offset = (len(sorted_values) - 1) * q
    lower = math.floor(offset)
    upper = math.ceil(offset)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (offset - lower)


def _savings_bounds(own, base):
    if own is None or base is None or own[1] is None or base[1] is None or base[0] <= 0:
        return None
    return [1 - own[1] / base[0], 1 - own[0] / base[1]]


def paired_comparison(rows, baseline_rows, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Paired percentile bootstrap, descriptive after development-set reuse.

    Each input must have one row per independent group. Undefined resampled
    ratios are counted and make the saving interval unavailable, not discarded.
    """
    if type(resamples) is not int or resamples < 100 or type(seed) is not int:
        raise ValueError("Require an integer seed and at least 100 bootstrap resamples")
    own = {row["group_id"]: row for row in rows}
    base = {row["group_id"]: row for row in baseline_rows}
    if not own or len(own) != len(rows) or len(base) != len(baseline_rows) or set(own) != set(base):
        raise ValueError("Paired uncertainty requires identical unique prompt groups")
    if any(own[group]["task_id"] != base[group]["task_id"] for group in own):
        raise ValueError("Paired task identifiers differ")
    policy, baseline = summarize_policy(rows), summarize_policy(baseline_rows)
    comparable = (policy["cost_complete"] and baseline["cost_complete"] and policy["cost_basis"] == baseline["cost_basis"])
    savings = (1 - policy["cost_per_accepted_usd"] / baseline["cost_per_accepted_usd"]
               if comparable and policy["cost_per_accepted_usd"] is not None and baseline["cost_per_accepted_usd"] else None)
    bounds = None
    if policy["cost_bound_basis"] is not None and policy["cost_bound_basis"] == baseline["cost_bound_basis"]:
        bounds = _savings_bounds(policy["cost_per_accepted_bounds_usd"], baseline["cost_per_accepted_bounds_usd"])
    own_ranges = {group: _cost_range(row)[:2] for group, row in own.items()}
    base_ranges = {group: _cost_range(row)[:2] for group, row in base.items()}
    losses = sorted(row["task_id"] for group, row in own.items() if base[group]["accepted"] and not row["accepted"])
    gains = sorted(row["task_id"] for group, row in own.items() if row["accepted"] and not base[group]["accepted"])
    groups, rng = sorted(own), random.Random(seed)
    differences, saving_samples, undefined = [], [], 0
    lower_samples, upper_samples = [], []
    for _ in range(resamples):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        accepted_own = sum(own[group]["accepted"] for group in draw)
        accepted_base = sum(base[group]["accepted"] for group in draw)
        differences.append((accepted_own - accepted_base) / len(groups))
        if bounds is not None and not comparable and accepted_own and accepted_base:
            own_range = [math.fsum(own_ranges[group][end] for group in draw) / accepted_own for end in (0, 1)]
            base_range = [math.fsum(base_ranges[group][end] for group in draw) / accepted_base for end in (0, 1)]
            interval = _savings_bounds(own_range, base_range)
            if interval is not None:
                lower_samples.append(interval[0])
                upper_samples.append(interval[1])
        if not comparable or not accepted_own or not accepted_base:
            undefined += 1
            continue
        cost_own = math.fsum(own[group]["cost_usd"] for group in draw)
        cost_base = math.fsum(base[group]["cost_usd"] for group in draw)
        if cost_base == 0:
            undefined += 1
        else:
            saving_samples.append(1 - (cost_own / accepted_own) / (cost_base / accepted_base))
    differences.sort()
    saving_samples.sort()
    point_interval = ([_percentile(saving_samples, .025), _percentile(saving_samples, .975)]
                      if saving_samples and not undefined else None)
    envelope = point_interval if comparable else None
    bounded_samples = len(saving_samples) if comparable else len(lower_samples)
    if lower_samples and len(lower_samples) == resamples:
        envelope = [_percentile(sorted(lower_samples), .025), _percentile(sorted(upper_samples), .975)]
    return {"paired_losses": len(losses), "paired_gains": len(gains), "lost_task_ids": losses, "gained_task_ids": gains,
            "acceptance_difference": (len(gains) - len(losses)) / len(groups),
            "savings_per_accepted_task": savings,
            "savings_per_accepted_bounds": bounds,
            "harm_upper_95_one_sided_exact": exact_harm_upper(len(losses), len(groups)),
            "harm_estimand": "P(baseline accepted AND policy rejected), denominator all independent prompt groups",
            "bootstrap": {"method": "paired_percentile", "unit": "deduplicated_prompt_group", "resamples": resamples, "seed": seed,
                          "acceptance_difference_interval_95": [_percentile(differences, .025), _percentile(differences, .975)],
                          "savings_interval_95": point_interval,
                          "savings_bounds_envelope_95": envelope,
                          "undefined_savings_bound_resamples": resamples - bounded_samples,
                          "undefined_savings_resamples": undefined,
                          "acceptance_has_no_discordant_pairs": not losses and not gains,
                          "degenerate_quality_interval_caution": "a [0,0] bootstrap interval with no discordant observed pairs cannot establish zero population harm; inspect the separate exact bound",
                          "scope": "descriptive conditional on observed groups and outcomes; no correction for policy selection or benchmark dependence"}}


def replay_public_cascade(records, *, candidate_model, baseline_model, reference_routes=None, initial_public_checks=None,
                          resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Reconstruct fresh-start fallback from already acquired paired episodes.

    A model branch may include its internal retries. Every selected branch is
    charged in full. This is not replay of continuation over the cheap patch.
    Named reference routes must be fixed independently of these outcomes.
    """
    pairs = _index(records, candidate_model, baseline_model)
    policy_rows = {"fixed_baseline": [], "fixed_candidate": [], "public_cascade": []}
    initial_audit = None
    if initial_public_checks is not None:
        if set(initial_public_checks) != set(pairs):
            raise ValueError("Initial public checks must cover every paired task")
        for check in initial_public_checks.values():
            if check.get("status") not in {"passed", "failed", "inconclusive"}:
                raise ValueError("Initial public check status must be explicit")
            _number(check.get("seconds"), "initial public check seconds")
        inconclusive = sorted(task for task, check in initial_public_checks.items() if check["status"] == "inconclusive")
        initial_audit = {"checks": deepcopy(initial_public_checks), "inconclusive_task_ids": inconclusive,
                         "complete_replay_available": not inconclusive,
                         "initial_check_total_seconds": math.fsum(check["seconds"] for check in initial_public_checks.values()),
                         "scope": "measured public check on original source only; no private evaluator or generated response",
                         "local_compute_priced": False}
        if not inconclusive:
            policy_rows["initial_failure_guarded_cascade"] = []
    references = reference_routes or {}
    for name, routes in references.items():
        if not isinstance(name, str) or not name or name in policy_rows or set(routes) != set(pairs):
            raise ValueError("Named reference routes must cover the exact cohort and use unique names")
        if any(model not in (candidate_model, baseline_model) for model in routes.values()):
            raise ValueError("Reference selected a model outside the compared pair")
        policy_rows[name] = []
    diagnostics = {model: {"public_passes": 0, "false_passes": 0, "false_pass_task_ids": []}
                   for model in (candidate_model, baseline_model)}
    for task_id, pair in pairs.items():
        cheap, strong = pair[candidate_model], pair[baseline_model]
        # Freeze the route BEFORE consulting independent acceptance or cost.
        route = operational_cascade_route(candidate_passed=cheap["public_passed"])
        selected = [cheap] if route == "candidate" else [cheap, strong]
        if "initial_failure_guarded_cascade" in policy_rows:
            # This signal also exists before any model generation. Passing the
            # original means the tests do not reproduce the reported defect.
            initial_passed = initial_public_checks[task_id]["status"] == "passed"
            guarded = _attempt_row([strong] if initial_passed else selected)
            guarded["initial_operational_status"] = initial_public_checks[task_id]["status"]
            guarded["candidate_skipped_initial_tests_passed"] = initial_passed
            policy_rows["initial_failure_guarded_cascade"].append(guarded)
        policy_rows["public_cascade"].append(_attempt_row(selected))
        policy_rows["fixed_candidate"].append(_attempt_row([cheap]))
        policy_rows["fixed_baseline"].append(_attempt_row([strong]))
        for name, routes in references.items():
            policy_rows[name].append(_attempt_row([pair[routes[task_id]]]))
        for model, row in pair.items():
            diagnostic = diagnostics[model]
            if row["public_passed"]:
                diagnostic["public_passes"] += 1
                if not row["accepted"]:
                    diagnostic["false_passes"] += 1
                    diagnostic["false_pass_task_ids"].append(task_id)
    for diagnostic in diagnostics.values():
        passes, failures = diagnostic["public_passes"], diagnostic["false_passes"]
        diagnostic["false_pass_rate_given_pass"] = failures / passes if passes else None
        diagnostic["false_pass_upper_95_one_sided_exact"] = exact_harm_upper(failures, passes) if passes else None
    comparisons = {name: paired_comparison(rows, policy_rows["fixed_baseline"], resamples=resamples, seed=seed)
                   for name, rows in policy_rows.items() if name != "fixed_baseline"}
    return {"schema_version": 1, "kind": "cascade_development_replay", "independent_validation": False,
            "activation_eligible": False, "baseline_model": baseline_model, "candidate_model": candidate_model,
            "policies": {name: summarize_policy(rows) for name, rows in policy_rows.items()},
            "comparisons_to_baseline": comparisons, "operational_verifier_diagnostics": diagnostics,
            "initial_public_check_audit": initial_audit,
            "task_rows": policy_rows,
            "scope": {"decision_input": ("candidate public_passed; guarded variant also uses public check on unchanged original source"
                                          if initial_public_checks is not None else "candidate public_passed only"),
                      "acceptance_labels_used_for_routing": False,
                      "fallback": "fresh independent baseline episode from unchanged original state",
                      "cost": "sum of all selected episode costs including failures and internal retries",
                      "net_monetary_savings_verified": False, "local_compute_priced": False,
                      "end_to_end_cascade_executed": False,
                      "limitation": "historical cohort already inspected; uncertainty is descriptive, not confirmatory"}}
