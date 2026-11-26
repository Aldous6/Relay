"""Paired replay of recorded maintenance episodes; never execute or choose Relay.

Both branches include the same inspection and allowed repair policy. ``weak``
means cheap-first followed by that repair, not a hypothetical cheap-only call.
USD values use recorded usage and scenario tariffs, not Free-account billing.
The reference is RouteLLM-BERT transferred to this pair, not all of RouteLLM.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import random


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261116
DEFAULT_THRESHOLD_GRID = tuple(i / 10 for i in range(11)) + (1.000001,)
_POLICIES = ("weak_first", "strong_first", "routellm_bert_transfer", "relay")


def _number(value, label, *, minimum=0., maximum=None):
    try:
        numeric = float(value) if type(value) in (int, float) else math.nan
    except (OverflowError, ValueError):
        numeric = math.nan
    if (not math.isfinite(numeric) or numeric < minimum
            or (maximum is not None and numeric > maximum)):
        raise ValueError(f"{label} debe ser finito y estar dentro de su rango.")
    return numeric


def _identifier(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} debe ser texto no vacío.")
    return value


def _validate(records, *, require_relay):
    if type(records) is not list or not records:
        raise ValueError("records debe ser una lista no vacía.")
    seen = set()
    groups = {}
    for row in records:
        if type(row) is not dict:
            raise ValueError("Cada registro debe ser un objeto.")
        task = _identifier(row.get("task_id"), "task_id")
        group = _identifier(row.get("group_id"), "group_id")
        if task in seen:
            raise ValueError("task_id duplicado: no se permiten episodios repetidos.")
        seen.add(task)
        partition = row.get("partition")
        if partition not in ("development", "holdout"):
            raise ValueError("partition debe ser development o holdout.")
        if group in groups and groups[group] != partition:
            raise ValueError("Un group_id no puede cruzar particiones.")
        groups[group] = partition
        _identifier(row.get("routing_text"), "routing_text")
        _number(row.get("reference_probability"), "reference_probability", maximum=1.)
        if require_relay and row.get("relay_choice") not in ("weak", "strong"):
            raise ValueError("relay_choice debe estar congelada como weak o strong.")
        branches = row.get("branches")
        if type(branches) is not dict or set(branches) != {"weak", "strong"}:
            raise ValueError("branches debe contener exactamente weak y strong.")
        for name, branch in branches.items():
            if type(branch) is not dict or type(branch.get("success")) is not bool:
                raise ValueError(f"{name}.success debe ser bool, no un score ni un número.")
            if "cost_usd" not in branch or "total_tokens" not in branch:
                raise ValueError("Cada branch debe declarar cost_usd y total_tokens, o null.")
            if branch["cost_usd"] is not None:
                _number(branch["cost_usd"], f"{name}.cost_usd")
            tokens = branch["total_tokens"]
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError(f"{name}.total_tokens debe ser un entero no negativo o null.")


def _choice(row, threshold, policy):
    if policy == "relay":
        return row["relay_choice"]
    if policy == "routellm_bert_transfer":
        return "strong" if row["reference_probability"] >= threshold else "weak"
    return "weak" if policy == "weak_first" else "strong"


def _metrics(branches):
    count = len(branches)
    successes = sum(branch["success"] for branch in branches)
    unknown_cost = sum(branch["cost_usd"] is None for branch in branches)
    unknown_tokens = sum(branch["total_tokens"] is None for branch in branches)
    try:
        known_cost = math.fsum(branch["cost_usd"] for branch in branches
                               if branch["cost_usd"] is not None)
    except OverflowError as exc:
        raise ValueError("El costo agregado excede el rango numérico finito.") from exc
    total_cost = None if unknown_cost else known_cost
    return {
        "task_count": count,
        "success_count": successes,
        "success_rate": successes / count,
        "total_cost_usd": total_cost,
        "known_cost_subtotal_usd": known_cost,
        "unknown_cost_count": unknown_cost,
        "cost_per_success_usd": total_cost / successes if total_cost is not None and successes else None,
        "total_tokens": None if unknown_tokens else sum(branch["total_tokens"] for branch in branches),
        "unknown_token_count": unknown_tokens,
    }


def _paired(relay, reference):
    relay_cost = relay["total_cost_usd"]
    reference_cost = reference["total_cost_usd"]
    relay_cps = relay["cost_per_success_usd"]
    reference_cps = reference["cost_per_success_usd"]
    ratio = (relay_cost / reference_cost if relay_cost is not None
             and reference_cost is not None and reference_cost > 0 else None)
    return {
        "success_delta": relay["success_rate"] - reference["success_rate"],
        "relative_cost_savings": 1. - ratio if ratio is not None and math.isfinite(ratio) else None,
        "delta_cost_per_success_usd": (relay_cps - reference_cps
                                       if relay_cps is not None and reference_cps is not None else None),
    }


def _interval(values, point):
    # Do not drop undefined bootstrap samples: that would select favorable draws.
    if point is None or any(value is None for value in values):
        return None
    ordered = sorted(values)
    def quantile(q):
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        fraction = index - lower
        return (1. - fraction) * ordered[lower] + fraction * ordered[upper]
    return [quantile(.025), quantile(.975)]


def _partition_report(rows, threshold, partition):
    grouped = defaultdict(list)
    selected = {}
    for row in rows:
        grouped[row["group_id"]].append(row["task_id"])
        selected[row["task_id"]] = {
            policy: row["branches"][_choice(row, threshold, policy)] for policy in _POLICIES
        }

    def aggregate(tasks):
        return {policy: _metrics([selected[task][policy] for task in tasks])
                for policy in _POLICIES}

    policy_metrics = aggregate([row["task_id"] for row in rows])
    controls = {"reference": "routellm_bert_transfer", "weak_first": "weak_first",
                "strong_first": "strong_first"}
    paired = {f"relay_vs_{name}": _paired(policy_metrics["relay"], policy_metrics[policy])
              for name, policy in controls.items()}
    samples = {name: {metric: [] for metric in result} for name, result in paired.items()}
    group_ids = sorted(grouped)
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        tasks = [task for group in rng.choices(group_ids, k=len(group_ids))
                 for task in grouped[group]]
        metrics = aggregate(tasks)
        for name, policy in controls.items():
            for metric, value in _paired(metrics["relay"], metrics[policy]).items():
                samples[f"relay_vs_{name}"][metric].append(value)
    for name, result in paired.items():
        result["bootstrap_ci95"] = {
            metric: _interval(samples[name][metric], value) for metric, value in list(result.items())
        }
    warnings = []
    if partition == "development":
        warnings.append("Desarrollo: estas cifras no son una evaluación independiente.")
    if len(grouped) < 30:
        warnings.append("Pocos grupos independientes; los intervalos pueden ser inestables o degenerados.")
    if any(result["unknown_cost_count"] for result in policy_metrics.values()):
        warnings.append("Hay costos desconocidos; no se sustituyen por cero ni se acredita ahorro con ellos.")
    return {
        "task_count": len(rows),
        "group_count": len(grouped),
        "policies": policy_metrics,
        "paired_comparisons": paired,
        "group_results": {group: aggregate(grouped[group]) for group in group_ids},
        "interpretation": {
            "scope": "exploratory_development" if partition == "development" else "declared_holdout",
            "superiority_claim": False,
            "reason": "El reporte describe evidencia pareada; no certifica potencia, congelación ni superioridad general.",
            "warnings": warnings,
        },
    }


def compare_records(records, reference_threshold: float, relay_policy: dict) -> dict:
    """Evaluate supplied choices; no fitting, threshold selection, or API calls.

    ``reference_probability`` is the reference's preference for strong; equality
    routes strong. A threshold above one permits the always-weak endpoint.
    Policy identity/freeze are declarations; this function cannot authenticate
    whether an upstream policy saw holdout outcomes.
    """
    _validate(records, require_relay=True)
    threshold = _number(reference_threshold, "reference_threshold", maximum=1.000001)
    if (type(relay_policy) is not dict or relay_policy.get("frozen") is not True
            or type(relay_policy.get("learned")) is not bool):
        raise ValueError("relay_policy debe declarar frozen=True y learned como bool.")
    _identifier(relay_policy.get("name"), "relay_policy.name")
    partitions = {}
    for partition in ("development", "holdout"):
        rows = [row for row in records if row["partition"] == partition]
        if rows:
            partitions[partition] = _partition_report(rows, threshold, partition)
    return {
        "schema_version": 1,
        "scope": "maintenance_paired_branch_replay",
        "reference_scope": "RouteLLM-BERT transfer; no representa RouteLLM-MF ni todo RouteLLM.",
        "cost_basis": "Uso observado con tarifas de escenario; no facturación de Groq Free.",
        "branch_contract": "weak y strong incluyen inspección, edición y la misma política de reparación permitida.",
        "relay_policy": deepcopy(relay_policy),
        "policy_freeze_provenance": "declared_by_caller_not_verified",
        "reference_threshold": threshold,
        "bootstrap": {"method": "paired_percentile_cluster", "unit": "group_id",
                      "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "confidence": .95, "estimand": "task_weighted"},
        "partitions": partitions,
        "decisions": [{"task_id": row["task_id"], "group_id": row["group_id"],
                       "partition": row["partition"], "relay_choice": row["relay_choice"],
                       "reference_choice": _choice(row, threshold, "routellm_bert_transfer")}
                      for row in records],
    }


def select_reference_threshold(devrecords, quality_loss_margin=0., grid=DEFAULT_THRESHOLD_GRID) -> dict:
    """Choose minimum observed cost on development at the strong quality floor.

    This is empirical development selection, not a noninferiority test. Relay's
    choices are irrelevant and are never selected or modified here.
    """
    _validate(devrecords, require_relay=False)
    if any(row["partition"] != "development" for row in devrecords):
        raise ValueError("La selección del umbral acepta únicamente development.")
    margin = _number(quality_loss_margin, "quality_loss_margin", maximum=1.)
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("grid debe contener umbrales explícitos.")
    thresholds = sorted(set(_number(value, "grid threshold", maximum=1.000001) for value in grid))
    strong = _metrics([row["branches"]["strong"] for row in devrecords])
    candidates = []
    for threshold in thresholds:
        result = _metrics([row["branches"][_choice(row, threshold, "routellm_bert_transfer")]
                           for row in devrecords])
        result["threshold"] = threshold
        result["eligible"] = (result["total_cost_usd"] is not None
                              and result["success_rate"] + 1e-12 >= strong["success_rate"] - margin)
        candidates.append(result)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("Ningún umbral tiene costo completo y cumple el piso de calidad en desarrollo.")
    selected = min(eligible, key=lambda row: (row["total_cost_usd"], -row["success_rate"], row["threshold"]))
    return {
        "threshold": selected["threshold"],
        "selection_scope": "development_only_empirical_not_noninferiority",
        "quality_loss_margin": margin,
        "strong_success_rate": strong["success_rate"],
        "task_count": len(devrecords),
        "group_count": len({row["group_id"] for row in devrecords}),
        "tie_break": "minimum cost, maximum success, minimum threshold",
        "candidates": candidates,
    }
