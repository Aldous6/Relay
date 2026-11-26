"""Episode accounting and paired diagnostics, without providers or adoption.

Missing measurements remain unknown. A retry is a charged call already present
in the total; its subtotal is descriptive and is never added a second time.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from math import isclose, isfinite
from numbers import Integral, Real
from typing import Mapping

import numpy as np


COMPONENTS = ("llm_usd", "router_usd", "verification_usd", "tool_usd")
DEFAULT_ROI = {"min_total_cost_saving": .30, "max_success_rate_loss": .02,
               "max_p95_router_seconds": .2}


def _number(value, name, *, nullable=True, integer=False):
    if value is None and nullable:
        return None
    kind = Integral if integer else Real
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, kind) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, without booleans.")
    return int(value) if integer else float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text.")
    return value


def _event(event):
    if not isinstance(event, dict):
        raise ValueError("Each event must be a dictionary.")
    for field in ("step", "attempt"):
        _number(event.get(field), field, nullable=False, integer=True)
    if type(event.get("is_retry")) is not bool:
        raise ValueError("is_retry must be an explicit boolean.")
    _text(event.get("model"), "model")
    for phase in ("router", "generation", "verification", "tool"):
        _number(event.get(phase+"_seconds"), phase+"_seconds")
    _number(event.get("reported_llm_cost_usd"), "reported_llm_cost_usd")
    usage = event.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("usage must declare token measurements, including unknown values.")
    for field in ("input_tokens", "output_tokens", "cached_input_tokens"):
        _number(usage.get(field), field, integer=True)
    if (usage.get("cached_input_tokens") is not None and usage.get("input_tokens") is not None
            and usage["cached_input_tokens"] > usage["input_tokens"]):
        raise ValueError("cached_input_tokens is a subset of input_tokens.")


def _cost_record(components, **extra):
    known = [value for value in components.values() if value is not None]
    subtotal = _number(sum(known), "cost subtotal", nullable=False)
    return {**components, "total_usd": subtotal if len(known) == len(COMPONENTS) else None,
            "known_subtotal_usd": subtotal, "known_components": len(known),
            "total_components": len(COMPONENTS), **extra}


def annotate_costs(events, pricing, compute_rates):
    """Copy events and attach explicit monetary accounting under ``costs``.

    Tariffs apply to exact model identities. Input tokens include cached input;
    they are split before multiplication, never billed twice. Provider-reported
    LLM cost takes precedence over estimates. A missing cached-token count is
    unknown, and a positive cached count requires its own explicit tariff.
    """
    if not isinstance(events, list) or not isinstance(pricing, Mapping) or not isinstance(compute_rates, Mapping):
        raise ValueError("events, pricing and compute_rates require list/mapping contracts.")
    allowed_rates = {phase+"_usd_per_second" for phase in ("router", "verification", "tool")}
    if set(compute_rates)-allowed_rates:
        raise ValueError("Unknown compute rate; generation is billed as LLM cost.")
    for name, value in compute_rates.items():
        _number(value, name)
    for model, tariff in pricing.items():
        _text(model, "pricing model")
        if not isinstance(tariff, Mapping):
            raise ValueError("Each tariff must be a mapping.")
        for field in ("input_per_million", "output_per_million", "cached_input_per_million"):
            _number(tariff.get(field), field)
        for field in ("source", "as_of"):
            _text(tariff.get(field), field)
    result = deepcopy(events)
    for event in result:
        _event(event)
        llm = _number(event.get("reported_llm_cost_usd"), "reported_llm_cost_usd")
        basis = "reported" if llm is not None else "unknown"
        tariff, usage = pricing.get(event["model"]), event["usage"]
        tariff_provenance = None
        identity_known = event.get("backend_failure", {}).get("category") != "model_identity"
        if llm is None and tariff is not None and identity_known:
            incoming, outgoing, cached = (usage.get(name) for name in
                                          ("input_tokens", "output_tokens", "cached_input_tokens"))
            ready = (incoming is not None and outgoing is not None and cached is not None
                     and tariff.get("input_per_million") is not None and tariff.get("output_per_million") is not None
                     and (cached == 0 or tariff.get("cached_input_per_million") is not None))
            if ready:
                llm = ((incoming-cached)*tariff["input_per_million"] + outgoing*tariff["output_per_million"]
                       + (cached*tariff["cached_input_per_million"] if cached else 0))/1_000_000
                llm = _number(llm, "estimated llm cost", nullable=False)
                basis = "estimated"
                tariff_provenance = {"model": event["model"], **dict(tariff)}
        components = {"llm_usd": llm}
        for phase in ("router", "verification", "tool"):
            duration, rate = event.get(phase+"_seconds"), compute_rates.get(phase+"_usd_per_second")
            components[phase+"_usd"] = _number(duration*rate if duration is not None and rate is not None else None, phase+" cost")
        event["costs"] = _cost_record(components, llm_cost_basis=basis, pricing=tariff_provenance)
    return result


def _components(event):
    costs = event.get("costs", {})
    if not isinstance(costs, Mapping):
        raise ValueError("costs must be a mapping.")
    values = {name: _number(costs.get(name), name) for name in COMPONENTS}
    computed = _cost_record(values)
    for name in ("total_usd", "known_subtotal_usd"):
        if name in costs:
            supplied = _number(costs[name], name)
            expected = computed[name]
            if (supplied is None) != (expected is None) or (supplied is not None and not isclose(supplied, expected, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError("Stored cost totals disagree with their components.")
    return computed


def _latency(values):
    measured = [float(value) for value in values if value is not None]
    return {"n": len(values), "known": len(measured), "unknown": len(values)-len(measured),
            "complete": bool(values) and len(measured) == len(values),
            "p50_seconds": float(np.quantile(measured, .5)) if measured else None,
            "p95_seconds": float(np.quantile(measured, .95)) if measured else None,
            "quantile_scope": "observed measurements only; unknown values are not zero"}


def _episode(episode):
    if not isinstance(episode, dict):
        raise ValueError("Each episode must be a dictionary.")
    for field in ("episode_id", "source_group", "family", "policy", "termination"):
        _text(episode.get(field), field)
    if episode["termination"] in {"pending", "running"}:
        raise ValueError("Summaries require terminated episodes, not pending journals.")
    if type(episode.get("success")) is not bool or not isinstance(episode.get("events"), list):
        raise ValueError("Episodes require boolean success and a list of events.")
    _number(episode.get("end_to_end_seconds"), "end_to_end_seconds")
    provenance = episode.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("execution_mode") not in {"simulated", "external"}:
        raise ValueError("Declare simulated or external execution provenance.")
    if "independent_validation" in provenance and type(provenance["independent_validation"]) is not bool:
        raise ValueError("independent_validation must be a boolean when declared.")
    _text(provenance.get("evaluation_split", "development"), "evaluation_split")
    seen, records = set(), []
    for event in episode["events"]:
        _event(event)
        key = (event["step"], event["attempt"])
        if "infrastructure_attempt" in event:
            _number(event["infrastructure_attempt"], "infrastructure_attempt", nullable=False, integer=True)
            key += (event["infrastructure_attempt"],)
        if key in seen:
            raise ValueError("Duplicate step/attempt (including infrastructure retry) would double-count a call.")
        seen.add(key)
        records.append(_components(event))
    # An empty journal supplies no measurements; it is not a zero-cost episode.
    total = _number(sum(row["total_usd"] for row in records), "episode total") if records and all(row["total_usd"] is not None for row in records) else None
    return {"episode": episode, "cost_records": records, "total_usd": total}


def _policy(rows):
    episodes = [row["episode"] for row in rows]
    events = [event for episode in episodes for event in episode["events"]]
    records = [record for row in rows for record in row["cost_records"]]
    totals = [row["total_usd"] for row in rows]
    total = _number(sum(totals), "policy total") if all(value is not None for value in totals) else None
    successes = sum(episode["success"] for episode in episodes)
    retry_costs = [record["llm_usd"] for row in rows
                   for event, record in zip(row["episode"]["events"], row["cost_records"]) if event["is_retry"]]
    components = {}
    for name in COMPONENTS:
        known = [record[name] for record in records if record[name] is not None]
        subtotal = _number(sum(known), "component subtotal", nullable=False)
        components[name] = {"total": subtotal if records and len(known) == len(records) else None,
                            "known_subtotal": subtotal, "known_calls": len(known), "unknown_calls": len(records)-len(known)}
    tokens = {}
    for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
        values = [event["usage"].get(name) for event in events]
        known = [value for value in values if value is not None]
        tokens[name] = {"total": sum(known) if values and len(known) == len(values) else None,
                        "known_total": sum(known), "known_calls": len(known), "unknown_calls": len(values)-len(known)}
    diagnostics = {}
    if any("error_scope" in event for event in events):
        diagnostics = {"attempt_routing_latency": _latency([event.get("router_seconds") for event in events]),
            "routing_latency_scope": "policy decisions only; pinned infrastructure retries excluded",
            "failure_diagnostics": {
            "semantic": dict(sorted(Counter(e["error_type"] for e in events if e.get("error_scope") == "semantic").items())),
            "infrastructure": dict(sorted(Counter(e.get("backend_failure", {}).get("category", "unknown")
                                     for e in events if e.get("error_scope") == "infrastructure").items())),
            "pinned_infrastructure_retries": sum(e.get("routing_source") == "pinned_infrastructure_retry" for e in events)}}
    return {**diagnostics, "n_episodes": len(rows), "source_groups": len({e["source_group"] for e in episodes}),
            "successes": successes, "failures": len(rows)-successes, "success_rate": successes/len(rows),
            "terminations": dict(sorted(Counter(e["termination"] for e in episodes).items())),
            "calls": len(events), "retry_calls": len(retry_costs), "tokens": tokens,
            "costs": {"total_usd": total, "cost_per_task_usd": total/len(rows) if total is not None else None,
                      "cost_per_success_usd": total/successes if total is not None and successes else None,
                      "known_subtotal_usd": float(sum(record["known_subtotal_usd"] for record in records)),
                      "components": components,
                      "retry_llm_usd": float(sum(retry_costs)) if all(value is not None for value in retry_costs) else None,
                      "retry_llm_known_subtotal_usd": float(sum(value for value in retry_costs if value is not None)),
                      "retry_llm_unknown_calls": sum(value is None for value in retry_costs),
                      "retry_scope": "subset of llm_usd, already included exactly once in total_usd",
                      "denominator": "all episodes, including failed tasks"},
            "coverage": {"complete_cost_episodes": sum(value is not None for value in totals),
                         "unknown_cost_episodes": sum(value is None for value in totals),
                         "complete_cost_calls": sum(record["total_usd"] is not None for record in records),
                         "unknown_cost_calls": sum(record["total_usd"] is None for record in records),
                         "llm_cost_basis": dict(sorted(Counter(e.get("costs", {}).get("llm_cost_basis", "unknown") for e in events).items()))},
            "routing_latency": _latency([event.get("router_seconds") for event in events
                                          if event.get("routing_source", "policy") == "policy"]),
            "end_to_end_latency": _latency([episode.get("end_to_end_seconds") for episode in episodes]),
            "execution_modes": sorted({e["provenance"]["execution_mode"] for e in episodes}),
            "evaluation_splits": sorted({e["provenance"].get("evaluation_split", "development") for e in episodes})}


def _paired(candidate, baseline, replicates, seed):
    left = {row["episode"]["episode_id"]: row for row in candidate}
    right = {row["episode"]["episode_id"]: row for row in baseline}
    if set(left) != set(right):
        raise ValueError("Paired policies require exactly the same episode_id set; no silent complete-case selection.")
    ids = sorted(left)
    for key in ids:
        if any(left[key]["episode"][field] != right[key]["episode"][field] for field in ("source_group", "family")):
            raise ValueError("Paired episode metadata differs between policies.")
    groups = sorted({left[key]["episode"]["source_group"] for key in ids})
    membership = [[key for key in ids if left[key]["episode"]["source_group"] == group] for group in groups]
    sizes = np.asarray([len(keys) for keys in membership])
    success = np.asarray([sum(int(left[key]["episode"]["success"])-int(right[key]["episode"]["success"]) for key in keys) for keys in membership])
    complete_costs = all(left[key]["total_usd"] is not None and right[key]["total_usd"] is not None for key in ids)
    sums = np.asarray([[sum(left[key]["total_usd"] for key in keys), sum(right[key]["total_usd"] for key in keys)]
                       for keys in membership]) if complete_costs else None
    delta = float(success.sum()/len(ids))
    cost_delta = float((sums[:, 0].sum()-sums[:, 1].sum())/len(ids)) if complete_costs else None
    saving = float(1-sums[:, 0].sum()/sums[:, 1].sum()) if complete_costs and sums[:, 1].sum() > 0 else None
    success_samples, cost_samples, saving_samples = [], [], []
    if len(groups) >= 2:
        rng = np.random.default_rng(seed)
        for _ in range(replicates):
            sampled = rng.integers(0, len(groups), len(groups))
            n = sizes[sampled].sum()
            success_samples.append(float(success[sampled].sum()/n))
            if complete_costs:
                a, b = sums[sampled].sum(axis=0)
                cost_samples.append(float((a-b)/n))
                if b > 0:
                    saving_samples.append(float(1-a/b))
    def interval(values):
        return np.quantile(values, [.025, .975]).tolist() if len(values) == replicates else None
    return {"n_episodes": len(ids), "n_source_groups": len(groups), "paired_by": "episode_id",
            "bootstrap_unit": "source_group", "bootstrap_replicates": replicates, "seed": seed,
            "success_rate_delta": delta, "success_rate_delta_95ci": interval(success_samples),
            "cost_per_task_usd_delta": cost_delta, "cost_per_task_usd_delta_95ci": interval(cost_samples),
            "relative_total_cost_saving": saving, "relative_total_cost_saving_95ci": interval(saving_samples),
            "valid_saving_bootstrap_replicates": len(saving_samples), "complete_paired_costs": complete_costs,
            "scope": "paired source-group percentile bootstrap; all task outcomes, no claim of independent sampling"}


def _roi(config):
    if not isinstance(config, Mapping) or set(config)-set(DEFAULT_ROI):
        raise ValueError("Unknown ROI configuration fields.")
    result = {**DEFAULT_ROI, **dict(config)}
    for name, value in result.items():
        result[name] = _number(value, name, nullable=False)
    if any(result[name] > 1 for name in ("min_total_cost_saving", "max_success_rate_loss")) or result["max_p95_router_seconds"] <= 0:
        raise ValueError("ROI fractions must be in [0,1] and latency limit positive.")
    return result


def _assessment(comparison, candidate, baseline, policy, thresholds):
    provenance = [row["episode"]["provenance"] for row in [*candidate, *baseline]]
    simulated = any(p["execution_mode"] == "simulated" for p in provenance)
    independent = bool(provenance) and all(p.get("evaluation_split", "development") == "evaluation"
                                          and p.get("independent_validation") is True for p in provenance)
    missing = []
    if comparison is None:
        missing.append("baseline_policy_missing")
    elif comparison["relative_total_cost_saving"] is None:
        missing.append("paired_total_cost_or_positive_baseline_cost_unknown")
    baseline_latencies = [row["episode"].get("end_to_end_seconds") for row in baseline]
    baseline_routing = [event.get("router_seconds") for row in baseline for event in row["episode"]["events"]]
    if (not policy["routing_latency"]["complete"] or not policy["end_to_end_latency"]["complete"]
            or any(value is None for value in [*baseline_latencies, *baseline_routing])):
        missing.append("latency_measurements_incomplete")
    criteria = {"total_cost_saving": None, "success_loss": None, "router_p95": None}
    if comparison is not None:
        saving = comparison["relative_total_cost_saving"]
        criteria["total_cost_saving"] = saving >= thresholds["min_total_cost_saving"] if saving is not None else None
        criteria["success_loss"] = comparison["success_rate_delta"] >= -thresholds["max_success_rate_loss"]
    if policy["routing_latency"]["complete"]:
        criteria["router_p95"] = policy["routing_latency"]["p95_seconds"] < thresholds["max_p95_router_seconds"]
    complete = all(value is not None for value in criteria.values())
    point_pass = all(criteria.values()) if complete else None
    intervals = (comparison.get("relative_total_cost_saving_95ci"), comparison.get("success_rate_delta_95ci")) if comparison else (None, None)
    confidence_pass = (intervals[0][0] >= thresholds["min_total_cost_saving"] and intervals[1][0] >= -thresholds["max_success_rate_loss"]
                       and criteria["router_p95"] is True) if all(item is not None for item in intervals) and complete else None
    if simulated:
        status = "not_assessed_simulation"
    elif missing:
        status = "indeterminate"
    elif not independent:
        status = "not_assessed_development" if any(p.get("evaluation_split", "development") in {"development", "calibration", "fit"} for p in provenance) else "not_assessed_independence"
    elif confidence_pass:
        status = "meets_gates_pending_review"
    elif point_pass is False:
        status = "does_not_meet_gates"
    else:
        status = "inconclusive"
    return {"adoption_status": status, "point_criteria": criteria, "point_criteria_met": point_pass,
            "confidence_criteria_met": confidence_pass, "independent_validation_declared": independent,
            "missing_requirements": missing, "automatic_adoption": False,
            "scope": "proposed monetary/quality/latency gates; metadata and integrity checks do not certify independence"}


def summarize(episodes, roi_config, baseline_policy="always_strong", bootstrap_replicates=2000, seed=42):
    """Summarize complete episodes, including failed tasks and all retry calls.

    Policies must use identical episode IDs for paired comparison. Confidence
    intervals resample whole source groups and are unavailable with one group.
    Simulation and reused development cannot establish adoption or ROI.
    """
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Provide a nonempty list of completed episodes.")
    _text(baseline_policy, "baseline_policy")
    bootstrap_replicates = _number(bootstrap_replicates, "bootstrap_replicates", nullable=False, integer=True)
    seed = _number(seed, "seed", nullable=False, integer=True)
    if bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    thresholds, grouped, seen = _roi(roi_config), {}, set()
    for episode in episodes:
        row = _episode(episode)
        key = (episode["policy"], episode["episode_id"])
        if key in seen:
            raise ValueError("Duplicate policy/episode_id.")
        seen.add(key)
        grouped.setdefault(episode["policy"], []).append(row)
    policies = {name: _policy(rows) for name, rows in sorted(grouped.items())}
    baseline = grouped.get(baseline_policy, [])
    comparisons = {}
    for name, rows in sorted(grouped.items()):
        if name == baseline_policy:
            continue
        comparison = _paired(rows, baseline, bootstrap_replicates, seed) if baseline else None
        comparisons[name] = {"baseline_policy": baseline_policy, "paired": comparison,
                             "roi": _assessment(comparison, rows, baseline, policies[name], thresholds)}
    return {"schema_version": 1, "baseline_policy": baseline_policy, "roi_config": thresholds,
            "policies": policies, "comparisons": comparisons, "automatic_adoption": False,
            "cost_scope": "all recorded calls, including retries, plus router, verification and tool costs; missing components remain unknown",
            "interpretation": "episode success is the declared task verifier outcome; this module does not validate its rubric or independence"}
