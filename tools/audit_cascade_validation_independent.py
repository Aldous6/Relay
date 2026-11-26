"""Independently audit completed real cascades; never analyze a live prefix.

Standard-library implementation. Does not import the validation reporter,
cascade_evaluation, provider SDK, prompts, or private evaluator code. The suite
is decoded only for frozen identities/configuration; text fields are unused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/harness-cascade/v1/run"
DEFAULT_SUITE = ROOT / "data/imported/harness-cascade-v1/fresh_suite.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonlines(path):
    path = Path(path)
    if not path.exists():
        return []
    # A live file may be between writes. Never treat an incomplete line as an
    # observation; final mode also checks coverage and exact start identities.
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(first, second):
    if first is None or second is None:
        return first is second
    return type(first) in (int, float) and type(second) in (int, float) and math.isclose(first, second, abs_tol=1e-12, rel_tol=1e-12)


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Audit timestamps require a timezone")
    return parsed


def percentile(samples, q):
    ordered = sorted(samples)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def harm_upper(losses, n):
    """Invert the finite binomial CDF directly; independent of reporter code."""
    if losses == n:
        return 1.
    lower, upper = 0., 1.
    for _ in range(80):
        p = (lower + upper) / 2
        cdf = math.fsum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(losses + 1))
        if cdf > .05:
            lower = p
        else:
            upper = p
    return (lower + upper) / 2


def load_integrity(run=DEFAULT_RUN, suite_path=DEFAULT_SUITE, *, require_complete=True):
    run, suite_path = Path(run).resolve(), Path(suite_path).resolve()
    suite, frozen = read(suite_path), read(run / "frozen_policy.json")
    unsigned = {key: value for key, value in frozen.items() if key != "freeze_id"}
    require(seal(unsigned) == frozen["freeze_id"], "Frozen policy seal changed")
    contract = frozen["contract"]
    require(sha(suite_path) == contract["suite_sha256"] and seal(suite) == contract["suite_content_sha256"], "Frozen cohort changed")
    private_path = suite_path.parent / suite["private_evaluator_file"]
    require(private_path.parent == suite_path.parent and sha(private_path) == contract["private_evaluators_sha256"], "Frozen final evaluator bytes changed")
    for relative, expected in contract["source_sha256"].items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT) and sha(source) == expected, "Frozen source fingerprint changed")
    frozen_at = timestamp(frozen["created_at_utc"])
    task_groups = {item["task_id"]: item["group_id"] for item in suite["tasks"]}
    require(len(task_groups) == len(suite["tasks"]) and len(set(task_groups.values())) == len(task_groups), "Task/group duplication")
    historical_path = suite_path.parent / "historical_exclusions.json"
    require(sha(historical_path) == suite["historical_exclusions_sha256"], "Historical exclusion snapshot changed")
    historical = read(historical_path)
    normalized = [item["normalized_prompt_sha256"] for item in suite["tasks"]]
    require(len(normalized) == len(set(normalized)), "Duplicate normalized task group")
    require(not set(task_groups).intersection(historical["task_ids"])
            and not set(task_groups.values()).intersection(historical["group_ids"])
            and not set(normalized).intersection(historical["normalized_prompt_sha256"]),
            "Fresh cohort overlaps its predeclared historical exclusions")
    completions = jsonlines(run / "task_completions.jsonl")
    done = {item["task_id"] for item in completions}
    require(len(done) == len(completions) and done.issubset(task_groups), "Invalid task completion identity")
    if require_complete:
        require(done == set(task_groups), f"Cohort incomplete: {len(done)}/{len(task_groups)}; do not calculate prefix savings")
    # All remaining reads restrict live audits to tasks already committed as
    # complete. No quality/cost aggregates are computed in integrity-only mode.
    starts = {path.stem: read(path) for path in (run / "task_starts").glob("*.json")}
    require(done.issubset(starts), "Completed task lacks a start record")
    if require_complete:
        require(set(starts) == done, "Unresolved extra task start")
    for item in completions:
        start = starts[item["task_id"]]
        require(start["task_id"] == item["task_id"] and start["freeze_id"] == frozen["freeze_id"], "Task start identity changed")
        require(frozen_at <= timestamp(start["created_at_utc"]) <= timestamp(item["completed_at_utc"]), "Task timestamps contradict prospective freeze")
    all_observations = jsonlines(run / "observations.jsonl")
    require(all(item["task_id"] in task_groups for item in all_observations), "Observation outside the frozen task set")
    observations = [item for item in all_observations if item["task_id"] in done]
    expected_pairs = {(task, model) for task in done for model in suite["models"]}
    indexed = {(item["task_id"], item["model"]): item for item in observations}
    require(len(indexed) == len(observations) and set(indexed) == expected_pairs, "Completed cohort lacks exact paired observations")
    by_episode = {}
    for row in observations:
        task_id, model = row["task_id"], row["model"]
        episode = task_id + "-" + model.rsplit("/", 1)[-1]
        require(row["episode_id"] == episode and row["group_id"] == task_groups[task_id], "Observation identity changed")
        counterpart = run / "tasks" / task_id / (model.rsplit("/", 1)[-1] + "-observation.json")
        require(read(counterpart) == row, "Observation differs from its per-branch artifact")
        require(all(type(row[name]) is bool for name in ("accepted", "public_passed", "private_passed", "cost_complete")), "Incomplete outcome/cost typing")
        require(row["episode_status"] in ("completed", "error"), "Unknown episode state")
        require(row["accepted"] == (row["public_passed"] and row["private_passed"] and row["episode_status"] == "completed"), "Acceptance is inconsistent with final checks")
        by_episode[episode] = row
    all_events = jsonlines(run / "http_attempts.jsonl")
    require([item["attempt_id"] for item in all_events] == list(range(1, len(all_events) + 1)), "Nonconsecutive global HTTP history")
    continuation_hashes = {}
    previous_tasks, previous_http, previous_time = 0, 0, frozen_at
    for path in sorted(run.glob("continuation-*.json")):
        continuation = read(path)
        tasks_before, http_before = continuation["completed_tasks"], continuation["completed_http_attempts"]
        created = timestamp(continuation["created_at_utc"])
        require(continuation["kind"] == "forward_only_acquisition_continuation"
                and continuation["repeat_completed_tasks"] is False
                and continuation["policy_or_parameters_changed"] is False,
                "Continuation changed its frozen acquisition contract")
        require(type(tasks_before) is int and previous_tasks <= tasks_before <= len(completions)
                and type(http_before) is int and previous_http <= http_before <= len(all_events)
                and previous_time <= created, "Invalid continuation boundary")
        prefix_lines = {"http_attempts.jsonl": http_before, "observations.jsonl": tasks_before * len(suite["models"]),
                        "cascade_results.jsonl": tasks_before, "task_completions.jsonl": tasks_before}
        require(set(continuation["prefix_sha256"]) == set(prefix_lines) | {"frozen_policy.json"}, "Unknown continuation fingerprint")
        require(sha(run / "frozen_policy.json") == continuation["prefix_sha256"]["frozen_policy.json"], "Continuation changed frozen manifest")
        for name, number in prefix_lines.items():
            lines = (run / name).read_bytes().splitlines(keepends=True)
            require(len(lines) >= number and hashlib.sha256(b"".join(lines[:number])).hexdigest()
                    == continuation["prefix_sha256"][name], "Pre-continuation evidence was rewritten")
        if tasks_before:
            require(timestamp(completions[tasks_before - 1]["completed_at_utc"]) <= created, "Continuation predates its committed prefix")
        prior_ids = {item["task_id"] for item in completions[:tasks_before]}
        require(all(timestamp(item["created_at_utc"]) >= created for task, item in starts.items() if task not in prior_ids),
                "A remaining task began before the continuation was documented")
        preserved = continuation.get("failure_preserved")
        if preserved:
            event = all_events[preserved["http_attempt_id"] - 1]
            require(event["attempt_id"] <= http_before and event["status"] == "error"
                    and event["model"] == preserved["model"] and event["cost_usd"] == preserved["cost_usd"]
                    and event["episode_id"] == preserved["task_id"] + "-" + preserved["model"].rsplit("/", 1)[-1],
                    "The documented failure was removed or reclassified")
        continuation_hashes[path.name] = sha(path)
        previous_tasks, previous_http, previous_time = tasks_before, http_before, created
    if require_complete:
        require(len(list((run / "attempt_starts").glob("*.json"))) == len(all_events), "Pending HTTP start remains")
        require(all(event["episode_id"] in by_episode for event in all_events), "Extra HTTP event outside completed cohort")
    events = [item for item in all_events if item["episode_id"] in by_episode]
    tariffs = contract["tariffs"]
    tokens = {model: 0 for model in suite["models"]}
    per_episode = {episode: [] for episode in by_episode}
    for event in events:
        start = read(run / "attempt_starts" / (str(event["attempt_id"]) + ".json"))
        require(all(start[key] == event[key] for key in ("attempt_id", "episode_id", "model", "started_at_utc")), "HTTP identity changed after dispatch")
        row = by_episode[event["episode_id"]]
        require(event["model"] == row["model"] and event["status"] in ("completed", "error"), "HTTP model/status mismatch")
        require(frozen_at <= timestamp(event["started_at_utc"]) <= timestamp(row["final_evaluation_at_utc"]), "A dispatch preceded freeze or followed its final evaluator")
        inp, out = event["input_tokens"], event["output_tokens"]
        known = type(inp) is int and inp >= 0 and type(out) is int and out >= 0
        if known:
            require(type(event["total_tokens"]) is int and event["total_tokens"] == inp + out, "Invalid token total")
            tokens[event["model"]] += inp + out
            rate = tariffs[event["model"]]
            price = (inp * rate["input_per_million_usd"] + out * rate["output_per_million_usd"]) / 1e6
            require(valid_number(event["cost_usd"]) and equal(price, event["cost_usd"]), "HTTP tariff does not reproduce from usage")
        else:
            require(event["cost_usd"] is None and event["total_tokens"] is None, "Unknown usage was replaced with a numeric cost")
        per_episode[event["episode_id"]].append(event)
    require(len(all_events) <= contract["limits"]["max_total_http_attempts"], "Global HTTP cap exceeded")
    require(all(count <= contract["limits"]["max_observed_tokens_per_model"] for count in tokens.values()), "Observed per-model token cap exceeded")
    for episode, row in by_episode.items():
        episode_events = per_episode[episode]
        known = all(item["cost_usd"] is not None for item in episode_events)
        subtotal = math.fsum(item["cost_usd"] for item in episode_events if item["cost_usd"] is not None)
        require(type(row["http_attempts"]) is int and row["http_attempts"] == len(episode_events), "Episode dispatches were dropped")
        require(row["cost_complete"] is known and equal(row["cost_usd"], subtotal if known else None), "Episode cost omits actual attempts")
        require(equal(row["known_cost_subtotal_usd"], subtotal), "Episode subtotal differs from HTTP evidence")
        require(row["cost_basis"] == ("tariff_estimate" if known else "unknown"), "Mixed/unsupported cost basis")
    all_entries = jsonlines(run / "cascade_results.jsonl")
    require(all(item["task_id"] in task_groups for item in all_entries), "Runtime result outside the frozen task set")
    entries = [item for item in all_entries if item["task_id"] in done]
    require(len(entries) == len(done) and {item["task_id"] for item in entries} == done, "Missing or duplicate completed runtime result")
    results = {}
    for entry in entries:
        task_id = entry["task_id"]
        path = (run / entry["result_path"]).resolve()
        require(path == run / "tasks" / task_id / "cascade/result.json" and sha(path) == entry["result_sha256"], "Runtime result path/hash mismatch")
        result = read(path)
        require(result["task_id"] == task_id and result["accepted"] is None and result["mode"] == "experimental", "Invalid runtime identity or private acceptance leakage")
        require(result["require_initial_failure"] is True and result["original_snapshot_unchanged"] is True, "Runtime policy/snapshot drift")
        manifest = read(path.parent / "manifest.json")
        require(manifest["task_id"] == task_id and manifest["require_initial_failure"] is True, "Runtime task manifest drift")
        for model in suite["models"]:
            require(indexed[task_id, model]["runtime_decision_sha256_before_final_evaluation"] == entry["result_sha256"], "Final evaluator did not reference the sealed decision")
        results[task_id] = result
    return {"run": run, "suite_path": suite_path, "suite": suite, "frozen": frozen, "done": done,
            "observations": indexed, "events": events, "per_episode": per_episode, "results": results,
            "input_sha256": {"suite": sha(suite_path), "frozen_policy": sha(run / "frozen_policy.json"),
                             "private_evaluators": sha(private_path), "historical_exclusions": sha(historical_path)},
            "continuation_sha256": continuation_hashes,
            "complete": done == set(task_groups)}


def analyze_complete(state):
    require(state["complete"], "Independent inference is forbidden before all frozen tasks finish")
    suite, indexed = state["suite"], state["observations"]
    strong = suite["baseline_model"]
    cheap = next(model for model in suite["models"] if model != strong)
    reconstructed = []
    for task in suite["tasks"]:
        task_id, result = task["task_id"], state["results"][task["task_id"]]
        attempts = result["attempts"]
        models = [attempt["model"] for attempt in attempts]
        initial = result["initial_verification"]["status"]
        allowed = {"passed": ([], [strong]), "failed": ([], [cheap], [cheap, strong]), "inconclusive": ([],)}
        require(initial in allowed and models in allowed[initial], "Actual route violates the initial public-check rule")
        require(len(attempts) <= 2 and len(set(models)) == len(models), "A task silently repeated a model branch")
        for index, attempt in enumerate(attempts, 1):
            require(attempt["attempt_id"] == index, "Attempt identifiers are not consecutive")
            path = state["run"] / "tasks" / task_id / "cascade" / f"attempt-{index}/attempt.json"
            require(read(path) == attempt, "Attempt artifact differs from sealed runtime result")
            observation = indexed[task_id, attempt["model"]]
            require(observation["stage"] == "cascade" and observation["episode_status"] == attempt["status"], "Actual cascade branch was mislabeled")
            source_path = path.parent / "workspace/solution.py"
            require(sha(source_path) == observation["source_sha256"], "Final branch source changed")
        for model in suite["models"]:
            require(indexed[task_id, model]["stage"] == ("cascade" if model in models else "control"), "Control acquisition was charged to a different stage")
        selected = result["selected_attempt"]
        if selected is None:
            require(result["selected_model"] is None and result["operational_passed"] is False, "No-output task cannot count as successful")
            accepted = False
        else:
            require(type(selected) is int and selected == len(attempts) and selected > 0, "Selected result is not the final actual attempt")
            require(result["selected_model"] == models[-1] and attempts[-1]["status"] == "completed"
                    and attempts[-1]["verification"]["status"] == "passed" and result["operational_passed"] is True,
                    "Selected attempt did not satisfy its operational contract")
            accepted = indexed[task_id, models[-1]]["accepted"]
        policy_events = [event for event in state["events"] if event["episode_id"] in
                         {indexed[task_id, model]["episode_id"] for model in models}]
        known = all(event["cost_usd"] is not None for event in policy_events)
        subtotal = math.fsum(event["cost_usd"] for event in policy_events if event["cost_usd"] is not None)
        accounting = result["accounting"]
        require(accounting["http_calls"] == len(policy_events) and accounting["cost_complete"] is known,
                "Actual-runtime call accounting differs from HTTP evidence")
        require(equal(accounting["cost_usd"], subtotal if known else None)
                and equal(accounting["known_cost_subtotal_usd"], subtotal), "Actual-runtime cost differs from all its requests")
        require(len(accounting["calls"]) == len(policy_events), "Runtime ledger omitted transport requests")
        for internal, outer in zip(accounting["calls"], policy_events):
            require(internal["model"] == outer["model"] and equal(internal["cost_usd"], outer["cost_usd"]), "Inner/outer transport accounting differs")
        baseline = indexed[task_id, strong]
        reconstructed.append({"task_id": task_id, "group_id": task["group_id"], "initial_public_status": initial,
                              "actual_models": models, "actual_selected_model": result["selected_model"],
                              "accepted": accepted, "baseline_accepted": baseline["accepted"],
                              "cost_usd": subtotal if known else None, "cost_complete": known,
                              "baseline_cost_usd": baseline["cost_usd"], "baseline_cost_complete": baseline["cost_complete"],
                              "actual_http_attempts": len(policy_events), "baseline_http_attempts": baseline["http_attempts"]})
    reconstructed.sort(key=lambda item: item["group_id"])
    count = len(reconstructed)
    accepted, base_accepted = sum(row["accepted"] for row in reconstructed), sum(row["baseline_accepted"] for row in reconstructed)
    complete_cost = all(row["cost_complete"] and row["baseline_cost_complete"] for row in reconstructed)
    total = math.fsum(row["cost_usd"] for row in reconstructed) if all(row["cost_complete"] for row in reconstructed) else None
    base_total = math.fsum(row["baseline_cost_usd"] for row in reconstructed) if all(row["baseline_cost_complete"] for row in reconstructed) else None
    cpa = total / accepted if total is not None and accepted else None
    base_cpa = base_total / base_accepted if base_total is not None and base_accepted else None
    savings = 1 - cpa / base_cpa if cpa is not None and base_cpa else None
    losses = [row["task_id"] for row in reconstructed if row["baseline_accepted"] and not row["accepted"]]
    gains = [row["task_id"] for row in reconstructed if row["accepted"] and not row["baseline_accepted"]]
    rng, bootstrap_savings, quality, undefined = random.Random(20261118), [], [], 0
    for _ in range(10000):
        sample = [reconstructed[rng.randrange(count)] for _ in range(count)]
        own_accept, base_accept = sum(row["accepted"] for row in sample), sum(row["baseline_accepted"] for row in sample)
        quality.append((own_accept - base_accept) / count)
        if not complete_cost or not own_accept or not base_accept:
            undefined += 1
            continue
        own_cost, base_cost = math.fsum(row["cost_usd"] for row in sample), math.fsum(row["baseline_cost_usd"] for row in sample)
        if not base_cost:
            undefined += 1
            continue
        bootstrap_savings.append(1 - (own_cost / own_accept) / (base_cost / base_accept))
    interval = [percentile(bootstrap_savings, .025), percentile(bootstrap_savings, .975)] if bootstrap_savings and not undefined else None
    return {"tasks": count, "accepted": accepted, "baseline_accepted": base_accepted,
            "total_cost_usd": total, "baseline_total_cost_usd": base_total, "cost_per_accepted_usd": cpa,
            "baseline_cost_per_accepted_usd": base_cpa, "savings_per_accepted_task": savings,
            "lost_task_ids": losses, "gained_task_ids": gains, "harm_upper_95_one_sided_exact": harm_upper(len(losses), count),
            "bootstrap": {"resamples": 10000, "seed": 20261118, "unit": "unique_prompt_group",
                          "method": "paired_percentile_linear_interpolation", "savings_interval_95": interval,
                          "acceptance_difference_interval_95": [percentile(quality, .025), percentile(quality, .975)],
                          "undefined_savings_resamples": undefined}, "task_rows": reconstructed,
            "activation_eligible": False, "net_monetary_savings_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cascade-final-audit/fresh-independent.json")
    parser.add_argument("--integrity-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.integrity_only and args.output.exists():
        raise FileExistsError("Use a new independent-audit destination")
    state = load_integrity(args.run, args.suite, require_complete=not args.integrity_only)
    if args.integrity_only:
        print(json.dumps({"integrity_checks_passed": True, "completed_tasks_checked": len(state["done"]),
                          "quality_or_savings_aggregates_computed": False, "sources_unchanged": True}))
        return 0
    metrics = analyze_complete(state)
    paths = {name: sha(args.run / name) for name in ("frozen_policy.json", "task_completions.jsonl",
             "observations.jsonl", "http_attempts.jsonl", "cascade_results.jsonl")}
    result = {"kind": "independent_complete_real_cascade_validation_audit", "complete": True,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_id": state["frozen"]["freeze_id"],
              "provider_calls_by_this_audit": 0, "reporter_or_cascade_evaluation_imported": False,
              "acceptance_used_for_routing": False, "measurement": metrics,
              "provenance": {"input_sha256": state["input_sha256"], "run_sha256": paths,
                             "continuation_sha256": state["continuation_sha256"],
                             "frozen_source_sha256": state["frozen"]["contract"]["source_sha256"],
                             "independent_auditor_sha256": sha(Path(__file__))},
              "limitations": ["Public tasks are new to Relay selection, not necessarily to model pretraining.",
                              "Independent arithmetic implementation does not create independent model samples.",
                              "Fixed controls reuse already-generated branches or are acquired after the cascade.",
                              "LLM tariffs exclude unpriced local compute and do not measure a Groq Free invoice.",
                              "Zero observed paired losses does not demonstrate zero population harm."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "report": str(args.output), "measurement": {key: metrics[key] for key in
                      ("tasks", "accepted", "baseline_accepted", "savings_per_accepted_task", "lost_task_ids", "bootstrap")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
