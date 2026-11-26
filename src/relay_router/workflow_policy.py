"""Build small, empirical workflow policies from paired, whole-task outcomes.

Selection uses development only. Validation checks the frozen choice on new
groups; it never searches for another model. The gates are empirical product
criteria, not a statistical guarantee of noninferiority or future savings.
Only workflow names enter routing. Prompts, model answers and extra observation
metadata are neither predictors nor copied into the policy artifact.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


FIELDS = ("workflow", "task_id", "group_id", "split", "model", "accepted",
          "cost_usd", "cost_basis", "evaluation_id")
COST_BASES = {"provider_reported", "tariff_estimate", "unknown"}
SCOPE = "empirical_workflow_policy_separate_validation_not_statistical_guarantee"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _name(value, name):
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} requires a nonempty name without surrounding whitespace.")
    return value


def _number(value, name, maximum=None):
    try:
        valid = (type(value) in (int, float) and math.isfinite(value) and value >= 0
                 and (maximum is None or value <= maximum))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} requires a finite nonnegative number.")
    return float(value)


def _records(records, split):
    if type(records) is not list or not records:
        raise ValueError("Provide a nonempty list of whole-task observations.")
    result, seen, identities = [], set(), {}
    for row in records:
        if type(row) is not dict or any(key not in row for key in FIELDS):
            raise ValueError("Observation is missing a required workflow outcome field.")
        if (type(row.get("schema_version", 1)) is not int or row.get("schema_version", 1) != 1
                or row.get("kind", "relay_task_observation") != "relay_task_observation"):
            raise ValueError("Unsupported task observation format.")
        item = {key: deepcopy(row[key]) for key in FIELDS}
        for key in ("workflow", "task_id", "group_id", "model", "evaluation_id"):
            _name(item[key], key)
        if item["split"] != split:
            raise ValueError(f"Expected only {split} observations; never mix selection and validation.")
        if item["accepted"] is not None and type(item["accepted"]) is not bool:
            raise ValueError("accepted must be true, false, or null.")
        if type(item["cost_basis"]) is not str or item["cost_basis"] not in COST_BASES:
            raise ValueError("Unknown cost_basis.")
        if item["cost_usd"] is not None:
            item["cost_usd"] = _number(item["cost_usd"], "cost_usd")
            if item["cost_basis"] == "unknown":
                raise ValueError("An unknown cost basis cannot carry a known monetary cost.")
        if "cost_complete" in row:
            if type(row["cost_complete"]) is not bool:
                raise ValueError("cost_complete must be a boolean declaration when supplied.")
            if not row["cost_complete"] and item["cost_usd"] is not None:
                raise ValueError("An explicitly incomplete task cannot carry a complete monetary cost.")
        key = (item["workflow"], item["task_id"], item["model"])
        if key in seen:
            raise ValueError("Duplicate model observation for one workflow task.")
        seen.add(key)
        task = key[:2]
        if task in identities and identities[task] != item["group_id"]:
            raise ValueError("Paired model observations must share the same task group.")
        identities[task] = item["group_id"]
        result.append(item)
    if len({item["evaluation_id"] for item in result}) != 1:
        raise ValueError("Observations must share one declared harness/evaluation contract.")
    return sorted(result, key=lambda row: (row["workflow"], row["task_id"], row["model"]))


def _criteria(min_groups, min_savings, max_quality_loss):
    if type(min_groups) is not int or min_groups < 1:
        raise ValueError("min_groups must be a positive integer.")
    return {"min_groups": min_groups, "min_savings": _number(min_savings, "min_savings", 1),
            "max_quality_loss": _number(max_quality_loss, "max_quality_loss", 1)}


def _catalog(baseline, models):
    _name(baseline, "baseline_model")
    if (type(models) not in (list, tuple) or not models
            or any(type(model) is not str for model in models)):
        raise ValueError("candidate_models must be a nonempty list of model names.")
    for model in models:
        _name(model, "candidate_model")
    if baseline not in models or len(set(models)) != len(models):
        raise ValueError("The candidate catalog must include the baseline once, without duplicates.")
    return sorted(models)


def _mean(values):
    try:
        result = math.fsum(value / len(values) for value in values)
    except OverflowError as exc:
        raise ValueError("Observed costs exceed the supported numeric range.") from exc
    if not math.isfinite(result):
        raise ValueError("Observed costs exceed the supported numeric range.")
    return result


def _comparison(rows, baseline, candidate, criteria):
    tasks = sorted({row["task_id"] for row in rows})
    index = {(row["task_id"], row["model"]): row for row in rows}
    pairs = [(index.get((task, baseline)), index.get((task, candidate))) for task in tasks]
    complete = [(left, right) for left, right in pairs if left is not None and right is not None]
    groups = {left["group_id"] for left, _ in complete}
    result = {"candidate_model": candidate, "task_count": len(tasks), "paired_task_count": len(complete),
              "group_count": len(groups), "missing_pair_count": len(tasks) - len(complete),
              "unknown_outcome_count": sum(row["accepted"] is None for pair in complete for row in pair),
              "unknown_cost_count": sum(row["cost_usd"] is None for pair in complete for row in pair),
              "baseline_success_rate": None, "candidate_success_rate": None,
              "baseline_mean_cost_usd": None, "candidate_mean_cost_usd": None,
              "quality_delta": None, "relative_cost_savings": None, "qualified": False,
              "weighting": "equal weight per group; repetitions averaged within group"}
    if result["missing_pair_count"]:
        result["reason"] = "missing_pairs"
        return result
    if result["unknown_outcome_count"] or result["unknown_cost_count"]:
        result["reason"] = "unknown_outcome_or_cost"
        return result
    bases = {row["cost_basis"] for pair in complete for row in pair}
    if len(bases) != 1 or "unknown" in bases:
        result["reason"] = "incomparable_cost_bases"
        return result
    result["cost_basis"] = bases.pop()
    grouped = {group: [pair for pair in complete if pair[0]["group_id"] == group] for group in groups}
    for role, position in (("baseline", 0), ("candidate", 1)):
        result[role + "_success_rate"] = _mean([
            _mean([float(pair[position]["accepted"]) for pair in pairs]) for pairs in grouped.values()])
        result[role + "_mean_cost_usd"] = _mean([
            _mean([pair[position]["cost_usd"] for pair in pairs]) for pairs in grouped.values()])
    base_cost = result["baseline_mean_cost_usd"]
    result["quality_delta"] = result["candidate_success_rate"] - result["baseline_success_rate"]
    if base_cost > 0:
        ratio = result["candidate_mean_cost_usd"] / base_cost
        if math.isfinite(ratio):
            result["relative_cost_savings"] = 1 - ratio
    if len(groups) < criteria["min_groups"]:
        result["reason"] = "insufficient_groups"
    elif result["candidate_success_rate"] == 0:
        result["reason"] = "no_accepted_tasks"
    elif result["quality_delta"] + 1e-12 < -criteria["max_quality_loss"]:
        result["reason"] = "quality_below_baseline"
    elif result["relative_cost_savings"] is None or result["relative_cost_savings"] + 1e-12 < criteria["min_savings"]:
        result["reason"] = "insufficient_savings"
    else:
        result.update(qualified=True, reason="empirical_criteria_passed")
    return result


def _seal(value, field):
    value[field] = _digest(value)
    return value


def _verify(value, kind, field):
    if (type(value) is not dict or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1 or value.get("kind") != kind):
        raise ValueError("Unsupported workflow policy artifact.")
    result = deepcopy(value)
    supplied = result.pop(field, None)
    if type(supplied) is not str or supplied != _digest(result):
        raise ValueError("Policy integrity mismatch; use the original frozen JSON artifact.")
    _catalog(result.get("baseline_model"), result.get("candidate_models"))
    if type(result.get("criteria")) is not dict or set(result["criteria"]) != {"min_groups", "min_savings", "max_quality_loss"}:
        raise ValueError("The policy must preserve its complete selection criteria.")
    _criteria(**result["criteria"])
    if result.get("scope") != SCOPE:
        raise ValueError("A workflow policy must declare its empirical scope.")
    return result


def propose_policy(dev_records, *, baseline_model, candidate_models=None,
                   min_groups=20, min_savings=.10, max_quality_loss=0.):
    """Freeze one development choice per workflow, without activating it.

    Costs must include all calls/repairs/router costs in the declared task.
    The evaluation_id identifies that shared harness configuration. This
    declaration and user-supplied outcomes are not independently authenticated.
    """
    _name(baseline_model, "baseline_model")
    rows = _records(dev_records, "development")
    models = _catalog(baseline_model, candidate_models if candidate_models is not None
                      else sorted({baseline_model, *(row["model"] for row in rows)}))
    if any(row["model"] not in models for row in rows):
        raise ValueError("An observation uses a model outside the declared candidate catalog.")
    criteria = _criteria(min_groups, min_savings, max_quality_loss)
    workflows = {}
    for workflow in sorted({row["workflow"] for row in rows}):
        current = [row for row in rows if row["workflow"] == workflow]
        comparisons = [_comparison(current, baseline_model, candidate, criteria)
                       for candidate in models if candidate != baseline_model]
        qualified = [item for item in comparisons if item["qualified"]]
        best = min(qualified, key=lambda item: (item["candidate_mean_cost_usd"],
                   -item["candidate_success_rate"], item["candidate_model"])) if qualified else None
        workflows[workflow] = {"proposed_model": best["candidate_model"] if best else baseline_model,
                               "reason": "awaiting_separate_validation" if best else "no_qualified_alternative",
                               "development_comparisons": comparisons}
    return _seal({"schema_version": 1, "kind": "relay_workflow_proposal", "scope": SCOPE,
                  "baseline_model": baseline_model, "candidate_models": models, "criteria": criteria,
                  "evaluation_id": rows[0]["evaluation_id"], "workflows": workflows,
                  "development_data_sha256": _digest(rows),
                  "development_group_hashes": sorted({_digest(row["group_id"]) for row in rows}),
                  "development_task_hashes": sorted({_digest([row["workflow"], row["task_id"]]) for row in rows}),
                  "data_provenance": "user_supplied_observations_not_independently_verified", "cost_unit": "USD",
                  "cost_contract": "complete task costs including retries, repairs and router; supplied by harness",
                  "ready": False}, "proposal_id")


def validate_policy(proposal, validation_records):
    """Evaluate frozen choices on separate groups; fall back instead of reselecting."""
    frozen = _verify(proposal, "relay_workflow_proposal", "proposal_id")
    rows = _records(validation_records, "validation")
    if frozen["evaluation_id"] != rows[0]["evaluation_id"]:
        raise ValueError("Validation must preserve the declared harness/evaluation contract.")
    if any(row["model"] not in frozen["candidate_models"] for row in rows):
        raise ValueError("Validation changed the candidate catalog.")
    if (set(frozen["development_group_hashes"]) & {_digest(row["group_id"]) for row in rows}
            or set(frozen["development_task_hashes"]) & {_digest([row["workflow"], row["task_id"]]) for row in rows}):
        raise ValueError("Development and validation must use disjoint tasks and groups.")
    baseline, workflows, routes = frozen["baseline_model"], {}, {}
    for workflow, selection in frozen["workflows"].items():
        candidate = selection["proposed_model"]
        current = [row for row in rows if row["workflow"] == workflow]
        validation = None
        status = "baseline_no_development_alternative"
        if candidate != baseline:
            if not current:
                status = "baseline_missing_validation"
            else:
                validation = _comparison(current, baseline, candidate, frozen["criteria"])
                development = next(item for item in selection["development_comparisons"]
                                   if item["candidate_model"] == candidate)
                if validation.get("cost_basis") is not None and validation["cost_basis"] != development.get("cost_basis"):
                    validation.update(qualified=False, reason="cost_basis_changed_since_development")
                status = "approved_empirical" if validation["qualified"] else "baseline_validation_rejected"
        routes[workflow] = candidate if status == "approved_empirical" else baseline
        workflows[workflow] = {**selection, "status": status, "selected_model": routes[workflow],
                               "validation": validation}
    return _seal({"schema_version": 1, "kind": "relay_workflow_policy", "scope": SCOPE,
                  "baseline_model": baseline, "candidate_models": frozen["candidate_models"],
                  "criteria": frozen["criteria"], "evaluation_id": frozen["evaluation_id"],
                  "routes": routes, "workflows": workflows,
                  "ready": any(model != baseline for model in routes.values()),
                  "proposal_id": proposal["proposal_id"],
                  "development_data_sha256": frozen["development_data_sha256"],
                  "validation_data_sha256": _digest(rows),
                  "validation_group_count": len({row["group_id"] for row in rows}),
                  "data_provenance": frozen["data_provenance"], "cost_unit": "USD",
                  "cost_contract": frozen["cost_contract"],
                  "unknown_workflow_action": "baseline",
                  "limitation": "Empirical validation gate, not guaranteed savings or a statistical noninferiority test. "
                                "Integrity hashes do not authenticate the author or the supplied outcome data."}, "policy_id")


def validate_artifact(policy):
    """Return an independent, integrity-checked deployable JSON policy copy."""
    frozen = _verify(policy, "relay_workflow_policy", "policy_id")
    routes, workflows = frozen.get("routes"), frozen.get("workflows")
    if type(routes) is not dict or type(workflows) is not dict or set(routes) != set(workflows):
        raise ValueError("Routes and workflow validation evidence must match.")
    for workflow, model in routes.items():
        _name(workflow, "workflow")
        if model not in frozen["candidate_models"]:
            raise ValueError("A route selects a model outside the frozen catalog.")
        record = workflows[workflow]
        if type(record) is not dict or record.get("selected_model") != model:
            raise ValueError("A route differs from its validation decision.")
        if model != frozen["baseline_model"]:
            validation = record.get("validation")
            if (record.get("status") != "approved_empirical" or record.get("proposed_model") != model
                    or type(validation) is not dict or validation.get("qualified") is not True
                    or validation.get("candidate_model") != model):
                raise ValueError("Alternative routes require matching separate validation evidence.")
    expected = any(model != frozen["baseline_model"] for model in routes.values())
    if type(frozen.get("ready")) is not bool or frozen["ready"] != expected or frozen.get("unknown_workflow_action") != "baseline":
        raise ValueError("Invalid readiness or fallback declaration.")
    frozen["policy_id"] = policy["policy_id"]
    return frozen
