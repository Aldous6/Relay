"""Independent retrospective audit from raw episodes and HTTP token ledgers.

Does not import cascade_evaluation or use its reconstructed policy rows. Reuses
the infrastructure-recovery audit only to verify eligibility/provenance. Initial
public-check states are read from the hash-verified development report; a fresh
preflight execution is a separate reproduction, not new model evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(first, second):
    return math.isclose(first, second, abs_tol=1e-12, rel_tol=1e-12)


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    fraction = position - left
    return ordered[left] + fraction * (ordered[min(left + 1, len(ordered) - 1)] - ordered[left])


def audit():
    import report_adaptive_recovery as infrastructure

    report_path = ROOT / "reports/openhands-cascade-development-v3.json"
    report = read(report_path)
    for collection in ("source_sha256", "input_sha256"):
        for relative, expected in report["provenance"][collection].items():
            if sha(ROOT / relative) != expected:
                raise ValueError("The development report no longer matches its recorded inputs/sources")
    run = ROOT / "outputs/harness-adaptive/v1/run"
    recovered = run / "pre_dispatch_recovery"
    state = infrastructure.recovery.load_original(
        run, ROOT / "data/imported/harness-adaptive-v1/fresh_suite.json",
        ROOT / "outputs/harness-adaptive/v1/policy-auto.json")
    # This verifies all write-ahead HTTP records, the zero-dispatch failure and
    # the uniformly defined recovery rule. Its reconstructed rows are unused.
    _, _, _, recovery_audit = infrastructure.audit_recovery(state, recovered)
    frozen = read(run / "frozen_routes.json")
    tariffs = frozen["contract"]["tariffs"]
    original = rows(run / "observations.jsonl")
    recovery = rows(recovered / "observations.jsonl")
    indexed = {(item["task_id"], item["model"]): item for item in original}
    if len(indexed) != len(original):
        raise ValueError("Duplicate original episodes")
    for item in recovery:
        key = item["task_id"], item["model"]
        before = indexed[key]
        if before["logical_calls"] != 0 or before["http_attempts"] != 0 or before["error_type"] != "PermissionError":
            raise ValueError("Recovery would replace an already dispatched model episode")
        indexed[key] = item
    events = rows(run / "http_attempts.jsonl") + rows(recovered / "http_attempts.jsonl")
    episode_costs, episode_http = {}, {}
    for event in events:
        rate = tariffs[event["model"]]
        inp, out = event["input_tokens"], event["output_tokens"]
        if type(inp) is not int or type(out) is not int or min(inp, out) < 0:
            raise ValueError("Unknown token usage cannot be priced as zero")
        price = (inp * rate["input_per_million_usd"] + out * rate["output_per_million_usd"]) / 1_000_000
        if event["cost_usd"] is None or not close(price, event["cost_usd"]):
            raise ValueError("HTTP cost differs from its frozen tariff and measured token usage")
        episode_costs.setdefault(event["episode_id"], []).append(price)
        episode_http[event["episode_id"]] = episode_http.get(event["episode_id"], 0) + 1
    strong, cheap = report["baseline_model"], report["candidate_model"]
    initial = report["initial_public_check_audit"]["checks"]
    suite = {item["task_id"]: item for item in state["suite"]["tasks"]}
    tasks = sorted({item["task_id"] for item in original})
    if set(tasks) != set(initial) or set(tasks) != set(suite) or len(indexed) != 2 * len(tasks):
        raise ValueError("Paired cohort coverage changed")
    rebuilt, seen_groups = [], set()
    for task_id in tasks:
        before = initial[task_id]
        for field, source in (("source_sha256", "initial_source"), ("public_tests_sha256", "public_tests")):
            if hashlib.sha256(suite[task_id][source].encode("utf-8")).hexdigest() != before[field]:
                raise ValueError("Initial public check refers to another task or verifier")
        if before["status"] not in {"passed", "failed"}:
            raise ValueError("Inconclusive initial checks cannot be silently excluded")
        weak_row, strong_row = indexed[task_id, cheap], indexed[task_id, strong]
        if weak_row["group_id"] != strong_row["group_id"] or weak_row["group_id"] in seen_groups:
            raise ValueError("The bootstrap requires unique paired prompt groups")
        seen_groups.add(weak_row["group_id"])
        # No acceptance label or cost is consulted until this route is fixed.
        if before["status"] == "passed":
            route = [strong]
        elif weak_row["public_passed"]:
            route = [cheap]
        else:
            route = [cheap, strong]
        def cost(model):
            item = indexed[task_id, model]
            episode_id = task_id + "-" + model.rsplit("/", 1)[-1]
            total = math.fsum(episode_costs[episode_id])
            if not item["cost_complete"] or item["cost_basis"] != "tariff_estimate" or not close(total, item["cost_usd"]):
                raise ValueError("An episode omits or misprices a dispatched request")
            return total
        accepted = indexed[task_id, route[-1]]["accepted"]
        rebuilt.append({"task_id": task_id, "group_id": weak_row["group_id"],
                        "initial_public_status": before["status"], "attempted_models": route,
                        "policy_accepted": accepted, "baseline_accepted": strong_row["accepted"],
                        "policy_cost_usd": math.fsum(cost(model) for model in route),
                        "baseline_cost_usd": cost(strong),
                        "policy_http_attempts": sum(episode_http[task_id + "-" + model.rsplit("/", 1)[-1]] for model in route)})
    rebuilt.sort(key=lambda item: item["group_id"])
    accepted_policy = sum(item["policy_accepted"] for item in rebuilt)
    accepted_baseline = sum(item["baseline_accepted"] for item in rebuilt)
    total_policy = math.fsum(item["policy_cost_usd"] for item in rebuilt)
    total_baseline = math.fsum(item["baseline_cost_usd"] for item in rebuilt)
    saving = 1 - (total_policy / accepted_policy) / (total_baseline / accepted_baseline)
    losses = [item["task_id"] for item in rebuilt if item["baseline_accepted"] and not item["policy_accepted"]]
    gains = [item["task_id"] for item in rebuilt if item["policy_accepted"] and not item["baseline_accepted"]]
    rng, samples, undefined = random.Random(20261118), [], 0
    for _ in range(10000):
        sample = [rebuilt[rng.randrange(len(rebuilt))] for _ in rebuilt]
        accept_policy = sum(item["policy_accepted"] for item in sample)
        accept_baseline = sum(item["baseline_accepted"] for item in sample)
        base_cost = math.fsum(item["baseline_cost_usd"] for item in sample)
        if not accept_policy or not accept_baseline or not base_cost:
            undefined += 1
            continue
        policy_cost = math.fsum(item["policy_cost_usd"] for item in sample)
        samples.append(1 - (policy_cost / accept_policy) / (base_cost / accept_baseline))
    interval = [percentile(samples, .025), percentile(samples, .975)] if samples and not undefined else None
    target = report["comparisons_to_baseline"]["initial_failure_guarded_cascade"]
    target_policy = report["policies"]["initial_failure_guarded_cascade"]
    comparisons = {
        "acceptance_matches": accepted_policy == target_policy["accepted"],
        "total_cost_matches": close(total_policy, target_policy["total_cost_usd"]),
        "saving_matches": close(saving, target["savings_per_accepted_task"]),
        "paired_losses_match": len(losses) == target["paired_losses"],
        "paired_gains_match": len(gains) == target["paired_gains"],
        "bootstrap_interval_matches": interval is not None and all(close(a, b) for a, b in zip(interval, target["bootstrap"]["savings_interval_95"])),
    }
    if "relay_router.cascade_evaluation" in sys.modules or not all(comparisons.values()):
        raise ValueError("Independent calculation did not reproduce the reported development result")
    paths = [report_path, run / "observations.jsonl", recovered / "observations.jsonl",
             run / "http_attempts.jsonl", recovered / "http_attempts.jsonl", run / "frozen_routes.json",
             recovered / "recovery_manifest.json", ROOT / "tools/report_adaptive_recovery.py",
             ROOT / "tools/recover_adaptive_initialization.py", Path(__file__)]
    return {"kind": "independent_retrospective_cascade_audit", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_kind": "same_cohort_recomputation_not_new_model_validation", "new_provider_calls": 0,
            "cascade_evaluation_imported": False, "replay_rows_from_previous_report_used": False,
            "initial_public_checks_source": str(report_path.relative_to(ROOT)),
            "initial_public_checks_reexecuted_in_this_script": False,
            "tasks": len(tasks), "paired_model_episodes": len(indexed), "original_episode_attempts": len(original),
            "recovery_episode_attempts": len(recovery), "acquisition_http_attempts": len(events),
            "policy_accepted": accepted_policy, "baseline_accepted": accepted_baseline,
            "policy_total_cost_usd": total_policy, "baseline_total_cost_usd": total_baseline,
            "policy_cost_per_accepted_usd": total_policy / accepted_policy,
            "baseline_cost_per_accepted_usd": total_baseline / accepted_baseline,
            "saving_per_accepted": saving, "lost_task_ids": losses, "gained_task_ids": gains,
            "direct_baseline_tasks": sum(item["initial_public_status"] == "passed" for item in rebuilt),
            "escalated_tasks": sum(len(item["attempted_models"]) == 2 for item in rebuilt),
            "policy_http_attempts": sum(item["policy_http_attempts"] for item in rebuilt),
            "bootstrap": {"replicates": 10000, "seed": 20261118, "method": "paired_percentile_linear_interpolation",
                          "unit": "unique_prompt_group", "saving_interval_95": interval, "undefined_replicates": undefined},
            "zero_loss_exact_upper_95": 1 - .05 ** (1 / len(tasks)) if not losses else None,
            "comparisons_to_report": comparisons, "recovery_provenance_verified": recovery_audit,
            "cash_or_net_saving_verified": False, "activation_eligible": False,
            "limitations": ["The policy was chosen after viewing this cohort; no new independent confirmation.",
                            "A percentile interval containing zero does not establish positive savings.",
                            "Observed zero paired losses and a degenerate quality bootstrap do not imply zero population harm.",
                            "Costs reconstruct frozen no-cache LLM tariffs; local compute, setup and bills are not measured.",
                            "Published benchmark tasks may be dependent or present in model pretraining.",
                            "An independent implementation verifies calculation, not statistical independence of the sample."],
            "task_rows": rebuilt,
            "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cascade-final-audit/independent_audit.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new output; audit reports are immutable")
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("tasks", "policy_accepted", "baseline_accepted", "saving_per_accepted", "bootstrap", "comparisons_to_report")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
