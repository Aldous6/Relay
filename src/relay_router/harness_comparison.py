"""Frozen-plan comparisons of complete paired agent episodes, including failures."""
from __future__ import annotations
import hashlib
import json
import math


def seal(value):
    result = dict(value)
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def verify(value):
    result = dict(value)
    supplied = result.pop("sha256", None)
    if seal(result)["sha256"] != supplied:
        raise ValueError("Frozen plan integrity mismatch")
    return result


def freeze_plan(suite, observations, reference=None):
    if any(row["split"] != "development" for row in observations):
        raise ValueError("Freeze using development only")
    models, baseline = suite["models"], suite["baseline_model"]
    weak = next(model for model in models if model != baseline)
    tasks = [row for row in suite["tasks"] if row["split"] == "development"]
    index = {(row["task_id"], row["model"]): row for row in observations}
    if len(index) != len(observations) or any((task["task_id"], model) not in index for task in tasks for model in models):
        raise ValueError("Development must contain each task/model exactly once")
    routes = {}
    for workflow in sorted({task["workflow"] for task in tasks}):
        subset = [task for task in tasks if task["workflow"] == workflow]
        left = [index[(task["task_id"], baseline)] for task in subset]
        right = [index[(task["task_id"], weak)] for task in subset]
        complete = all(row["cost_complete"] and row["cost_usd"] is not None and type(row["accepted"]) is bool for row in left + right)
        qualifies = (complete and sum(row["accepted"] for row in right) >= sum(row["accepted"] for row in left)
                     and sum(row["accepted"] for row in right) > 0
                     and sum(row["cost_usd"] for row in right) <= .9 * sum(row["cost_usd"] for row in left))
        routes[workflow] = weak if qualifies else baseline
    policies = {"fixed_strong": {task["task_id"]: baseline for task in suite["tasks"]},
                "fixed_cheap": {task["task_id"]: weak for task in suite["tasks"]},
                "relay_exploratory": {task["task_id"]: routes[task["workflow"]] for task in suite["tasks"]}}
    threshold = None
    if reference is not None:
        scores = {row["task_id"]: row["probability"] for row in reference["rows"]}
        def choices(cut):
            return {task["task_id"]: baseline if scores[task["task_id"]] >= cut else weak for task in suite["tasks"]}
        policies["bert_transferred_fixed_0.5"] = choices(.5)
        baseline_success = sum(index[(task["task_id"], baseline)]["accepted"] is True for task in tasks)
        candidates = []
        for cut in [i / 100 for i in range(101)]:
            plan = choices(cut)
            selected = [index[(task["task_id"], plan[task["task_id"]])] for task in tasks]
            if (all(row["cost_complete"] and row["cost_usd"] is not None for row in selected)
                    and sum(row["accepted"] is True for row in selected) >= baseline_success):
                candidates.append((sum(row["cost_usd"] for row in selected), cut))
        threshold = min(candidates)[1] if candidates else 0.
        policies["bert_transferred_development_calibrated"] = choices(threshold)
    return seal({"kind": "frozen_exploratory_route_plan", "evaluation_id": suite["evaluation_id"],
                 "production_ready": False, "selection_frozen_before_validation": True,
                 "workflow_routes": routes, "policies": policies, "bert_threshold": threshold,
                 "development_observations_sha256": seal({"rows": observations})["sha256"],
                 "limitation": "Six development groups per workflow; research evaluation override only; production gate remains 20."})


def compare(suite, plan, observations, *, split="validation"):
    frozen = verify(plan)
    tasks = [task for task in suite["tasks"] if task["split"] == split]
    ids = {task["task_id"] for task in tasks}
    index = {}
    for row in observations:
        key = (row["task_id"], row["model"])
        if key in index:
            raise ValueError("Duplicate episode: never silently choose a successful retry")
        index[key] = row
    result = {}
    for name, choices in frozen["policies"].items():
        rows = [index.get((task["task_id"], choices[task["task_id"]])) for task in tasks]
        present = [row for row in rows if row is not None]
        missing = len(tasks) - len(present)
        known = not missing and all(type(row["accepted"]) is bool for row in present)
        accepted = sum(row["accepted"] is True for row in present)
        cost_known = known and all(row.get("cost_complete") and row.get("cost_usd") is not None for row in present)
        cost = math.fsum(row["cost_usd"] for row in present) if cost_known else None
        result[name] = {"tasks_expected": len(tasks), "episodes_present": len(present), "missing": missing,
                        "accepted": accepted, "acceptance_rate": accepted / len(tasks) if known and tasks else None,
                        "llm_tariff_total_usd": cost,
                        "llm_tariff_cost_per_accepted_task_usd": cost / accepted if cost is not None and accepted else None,
                        "known_cost_subtotal_usd": math.fsum(row["cost_usd"] for row in present if row.get("cost_usd") is not None),
                        "unknown_cost_episodes": sum(row.get("cost_usd") is None for row in present),
                        "http_attempts": sum(row.get("http_attempts", 0) for row in present),
                        "input_tokens": sum(row.get("observed_input_tokens", 0) for row in present),
                        "output_tokens": sum(row.get("observed_output_tokens", 0) for row in present),
                        "model_counts": {model: sum(choices[task["task_id"]] == model for task in tasks) for model in suite["models"]}}
    baseline = result["fixed_strong"]
    for name, stats in result.items():
        cost, base_cost = stats["llm_tariff_cost_per_accepted_task_usd"], baseline["llm_tariff_cost_per_accepted_task_usd"]
        stats["cost_per_accepted_savings_vs_strong"] = 1 - cost / base_cost if cost is not None and base_cost else None
        stats["quality_delta_vs_strong"] = stats["acceptance_rate"] - baseline["acceptance_rate"] if stats["acceptance_rate"] is not None and baseline["acceptance_rate"] is not None else None
        harm = 0
        for task_id in ids:
            left = index.get((task_id, frozen["policies"]["fixed_strong"][task_id]))
            right = index.get((task_id, frozen["policies"][name][task_id]))
            if left is not None and right is not None:
                harm += left["accepted"] is True and right["accepted"] is False
        stats["paired_lost_successes_vs_strong"] = harm
        # A finite sample with no losses still has uncertainty. Never emit a
        # degenerate bootstrap interval as proof that population harm is zero.
        n, z = len(tasks), 1.6448536269514722
        p = harm / n if n else 0
        stats["wilson_95_one_sided_harm_upper"] = ((p + z*z/(2*n) + z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))) / (1+z*z/n)
                                                          if n and stats["quality_delta_vs_strong"] is not None else None)
    return {"kind": "paired_agent_branch_replay", "split": split, "plan_sha256": plan["sha256"], "policies": result,
            "cost_basis": "tariff_estimate_uncached_scenario_not_free_account_invoice",
            "quality_noninferiority_demonstrated": False, "production_ready": False,
            "method": "Each candidate executed once per task with identical host and budget; policies frozen before validation choose measured branches.",
            "limitations": ["Small historical public benchmarks; possible training contamination", "No customer adoption or production evidence",
                            "LLM token tariff scenario excludes local compute and acquisition costs", "Not Diamond not executed; BERT is a transferred checkpoint, not all RouteLLM"]}
