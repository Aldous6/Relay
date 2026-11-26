"""Offline exploratory sensitivity; exact arithmetic, no imputation or policy edits.

This is a post-evaluation diagnostic, NOT a new validation or an economic
confidence interval. Unknown usage remains unknown in the primary result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def number(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("A finite numeric value is required")
    result = F(str(value))
    if result < 0:
        raise ValueError("Costs cannot be negative")
    return result


def exact(value):
    return {"fraction": str(value), "decimal": float(value)}


def threshold(known, accepted, baseline_cost, baseline_accepted, target):
    """Largest unknown nonnegative cost giving savings >= target; may be < 0."""
    known, baseline_cost, target = number(known), number(baseline_cost), number(target)
    if type(accepted) is not int or type(baseline_accepted) is not int or min(accepted, baseline_accepted) < 1:
        raise ValueError("Positive integer acceptance counts are required")
    if baseline_cost == 0 or target >= 1:
        raise ValueError("Positive baseline cost and 0 <= target < 1 are required")
    return (1 - target) * baseline_cost * accepted / baseline_accepted - known


def binomial_percentile_quality_interval(n, losses):
    """Exact empirical-bootstrap distribution when all discordances are losses."""
    if type(n) is not int or type(losses) is not int or not 0 <= losses <= n or n < 1:
        raise ValueError("Require integer 0 <= losses <= n")
    p = F(losses, n)
    distribution = sorted((-F(k, n), math.comb(n, k) * p**k * (1 - p)**(n-k)) for k in range(n + 1))
    def quantile(q):
        cumulative = F(0)
        for value, mass in distribution:
            cumulative += mass
            if cumulative >= q:
                return value
        raise AssertionError("Distribution does not sum to one")
    return [quantile(F(1, 40)), quantile(F(39, 40))]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_prompt(prompt):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", prompt).casefold().split()).encode()).hexdigest()


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Independent arithmetic differs from stored result")


def make_report(root=ROOT):
    root = Path(root)
    run = root / "outputs/harness-cascade/v1/run"
    suite_path = root / "data/imported/harness-cascade-v1/fresh_suite.json"
    suite, freeze = load(suite_path), load(run / "frozen_policy.json")
    history = load(root / "data/imported/harness-cascade-v1/historical_exclusions.json")
    primary_path = root / "reports/openhands-cascade-validation-v1.json"
    primary = load(primary_path)
    assert digest(suite_path) == freeze["contract"]["suite_sha256"]
    for name, expected in freeze["contract"]["source_sha256"].items():
        assert digest(root / name) == expected, name
    tasks = suite["tasks"]
    for field, history_field in (("task_id", "task_ids"), ("group_id", "group_ids"), ("normalized_prompt_sha256", "normalized_prompt_sha256")):
        values = [task[field] for task in tasks]
        assert len(values) == len(set(values)) and not set(values).intersection(history[history_field]), field
    assert all(normalized_prompt(task["prompt_text"]) == task["normalized_prompt_sha256"] for task in tasks)
    observations = lines(run / "observations.jsonl")
    obs = {(row["task_id"], row["model"]): row for row in observations}
    assert len(obs) == len(observations) == len(tasks) * 2
    events = lines(run / "http_attempts.jsonl")
    by_episode = {}
    for event in events:
        inp, out = event["input_tokens"], event["output_tokens"]
        cost = None
        if inp is not None and out is not None:
            assert type(inp) is int and type(out) is int and min(inp, out) >= 0
            rates = freeze["contract"]["tariffs"][event["model"]]
            cost = (inp * number(rates["input_per_million_usd"]) + out * number(rates["output_per_million_usd"])) / 1000000
            close(cost, event["cost_usd"])
        else:
            assert event["cost_usd"] is None
        by_episode.setdefault(event["episode_id"], []).append((event["attempt_id"], cost))
    known, unknown, success = {}, {}, {}
    for key, row in obs.items():
        dispatches = by_episode.get(row["episode_id"], [])
        assert len(dispatches) == row["http_attempts"]
        known[key] = sum((cost for _, cost in dispatches if cost is not None), F(0))
        unknown[key] = [identity for identity, cost in dispatches if cost is None]
        success[key] = row["public_passed"] and row["private_passed"] and row["episode_status"] == "completed"
        assert success[key] == row["accepted"]
        if unknown[key]:
            assert row["cost_usd"] is None and row["cost_complete"] is False
        else:
            close(known[key], row["cost_usd"])
    baseline = suite["baseline_model"]
    candidate = next(model for model in suite["models"] if model != baseline)
    own_known, baseline_cost, cheap_known = F(0), F(0), F(0)
    own_accepted = baseline_accepted = cheap_accepted = 0
    own_unknown, cheap_unknown, losses, gains, actual_models = [], [], [], [], Counter()
    for task in tasks:
        identity = task["task_id"]
        result = load(run / "tasks" / identity / "cascade/result.json")
        keys = [(identity, attempt["model"]) for attempt in result["attempts"]]
        assert len(keys) == len(set(keys))
        own_known += sum((known[key] for key in keys), F(0))
        own_unknown.extend(identifier for key in keys for identifier in unknown[key])
        selected = result["selected_model"]
        accepted = success[(identity, selected)] if selected is not None else False
        strong_success = success[(identity, baseline)]
        own_accepted += accepted
        baseline_accepted += strong_success
        cheap_accepted += success[(identity, candidate)]
        baseline_cost += known[(identity, baseline)]
        assert not unknown[(identity, baseline)]
        cheap_known += known[(identity, candidate)]
        cheap_unknown.extend(unknown[(identity, candidate)])
        if strong_success and not accepted:
            losses.append(identity)
        if accepted and not strong_success:
            gains.append(identity)
        actual_models[tuple(key[1] for key in keys)] += 1
    n = len(tasks)
    assert n == 43 and own_accepted == 42 and baseline_accepted == 43 and cheap_accepted == 41
    assert own_unknown == [60] and cheap_unknown == [60, 81]
    p = primary["policies"]
    for calculated, recorded in [(own_known, p["relay_cascade_actual"]["known_cost_subtotal_usd"]),
                                 (baseline_cost, p["fixed_baseline"]["total_cost_usd"]),
                                 (cheap_known, p["fixed_candidate"]["known_cost_subtotal_usd"])]:
        close(calculated, recorded)
    assert not gains
    exact_interval = binomial_percentile_quality_interval(n, len(losses))
    recorded_interval = primary["comparisons_to_baseline"]["relay_cascade_actual"]["bootstrap"]["acceptance_difference_interval_95"]
    assert [float(x) for x in exact_interval] == recorded_interval
    groups, rng, draws_with_unknown = sorted(task["task_id"] for task in tasks), random.Random(20261118), 0
    unknown_tasks = {row["task_id"] for row in observations if row["stage"] == "cascade" and not row["cost_complete"]}
    for _ in range(10000):
        draw = [groups[rng.randrange(n)] for _ in groups]
        draws_with_unknown += bool(set(draw).intersection(unknown_tasks))
    historical = load(root / "outputs/cascade-final-audit/independent_audit.json")
    for name, expected in historical["sha256"].items():
        assert digest(root / name) == expected, name
    historical_rows = historical["task_rows"]
    hcost = sum((number(row["policy_cost_usd"]) for row in historical_rows), F(0))
    hbase = sum((number(row["baseline_cost_usd"]) for row in historical_rows), F(0))
    haccepted = sum(row["policy_accepted"] for row in historical_rows)
    hbaccepted = sum(row["baseline_accepted"] for row in historical_rows)
    hsaving = 1 - (hcost / haccepted) / (hbase / hbaccepted)
    close(hsaving, historical["saving_per_accepted"])
    # This deliberately wrong numerator demonstrates why failed tasks stay charged.
    wrong_hcost = sum((number(row["policy_cost_usd"]) for row in historical_rows if row["policy_accepted"]), F(0))
    wrong_hbase = sum((number(row["baseline_cost_usd"]) for row in historical_rows if row["baseline_accepted"]), F(0))
    wrong_saving = 1 - (wrong_hcost / haccepted) / (wrong_hbase / hbaccepted)
    files = [suite_path, run / "frozen_policy.json", run / "observations.jsonl", run / "http_attempts.jsonl", primary_path,
             root / "outputs/cascade-final-audit/independent_audit.json", Path(__file__)]
    return {
        "kind": "exploratory_scientific_contrast_not_new_validation", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0, "policy_or_original_results_modified": False, "unknown_cost_imputed": False,
        "primary_savings_estimable": False, "primary_quality_gate_met_for_any_unknown_cost": False,
        "cohort": {"tasks": n, "unique_normalized_prompts": n, "historical_id_group_prompt_overlap": False,
                   "families": dict(Counter(task["workflow"] for task in tasks)), "frozen_sources_verified": len(freeze["contract"]["source_sha256"])},
        "measurements": {"relay_accepted": own_accepted, "strong_accepted": baseline_accepted, "cheap_accepted": cheap_accepted,
            "paired_losses": losses, "paired_gains": gains, "relay_known_subtotal_usd": exact(own_known),
            "strong_total_usd": exact(baseline_cost), "cheap_known_subtotal_usd": exact(cheap_known),
            "relay_unknown_http_ids": own_unknown, "cheap_unknown_http_ids": cheap_unknown,
            "model_sequences": [{"models": list(models), "tasks": count} for models, count in actual_models.items()]},
        "economic_sensitivity": {"definition_of_x": "Nonnegative unobserved tariff cost of HTTP 60, not its reservation or an imputed value",
            "formula": "saving(x) = 1 - ((0.00631725+x)/42)/(0.01239765/43)",
            "condition_for_nonnegative_savings": {"x_less_or_equal_usd": exact(threshold(own_known, own_accepted, baseline_cost, baseline_accepted, F(0)))},
            "condition_for_at_least_10_percent_savings": {"x_less_or_equal_usd": exact(threshold(own_known, own_accepted, baseline_cost, baseline_accepted, F(1,10)))},
            "unknown_cost_measured_or_bounded_by_provider": False,
            "quality_gate_still_fails_under_both_conditions": True,
            "not_a_confidence_interval_or_verified_savings_claim": True},
        "bootstrap_contrast": {"method": "Exact empirical-bootstrap binomial distribution for loss count; no Monte Carlo for analytical interval",
            "exact_percentile_quality_interval_95": [exact(x) for x in exact_interval], "matches_10000_paired_report_interval": True,
            "resampled_draws_examined": 10000, "seed": 20261118, "draws_containing_unknown_cost_task": draws_with_unknown,
            "recorded_economic_resamples_blocked": 10000,
            "meaning": "All economic inference is gated by cohort-level accounting completeness, even draws that omit the unknown-cost task; no complete-case cost interval is supplied"},
        "historical_numerator_contrast": {"tasks": len(historical_rows), "accepted": haccepted,
            "saving_all_attempts": exact(hsaving), "incorrect_saving_if_failed_tasks_removed_from_numerator": exact(wrong_saving),
            "incorrect_numerator_is_for_demonstration_only": True, "retrospective_not_confirmatory": True,
            "original_input_hashes_verified": len(historical["sha256"])},
        "limitations": ["Benchmark task groups need not be representative or independent",
            "A one-sided harm bound above 2% and an interval touching zero do not establish equivalence",
            "Fixed control branches are reused or acquired after the cascade, not randomized independent policy executions",
            "No completed 20B-to-120B escalation is observed in this cohort",
            "No RouteLLM or Not Diamond comparison occurred in this fresh validation",
            "Tariff usage differs from invoiced cash savings and excludes unpriced local compute"],
        "provenance_sha256": {path.relative_to(root).as_posix(): digest(path) for path in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cascade-contrast/v1/science.json")
    args = parser.parse_args()
    report = make_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(args.output), "no_imputation": True, "primary_quality_gate_met": False,
                      "economic_sensitivity": report["economic_sensitivity"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
