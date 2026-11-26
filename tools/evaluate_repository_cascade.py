"""Freeze and evaluate a new repository cohort against paired OpenHands controls.

score is offline. freeze and run take a trusted host adapter supplied by its
operator, never commands from an issue, dataset or model. See the v2 protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from relay_router.cascade import BudgetExceeded, SpendBudget, _manifest, run_cascade
from relay_router.cascade_evaluation import paired_comparison, summarize_policy
from relay_router.cascade_verifier import CommandVerifier
from relay_router.integrations.openhands_cascade import make_openhands_attempt
from repository_reference import CATALOG, duration, frozen_routes, score_cohort

CHEAP, STRONG = CATALOG
ARMS = ("relay", "fixed_cheap", "fixed_strong")
BERT = "routellm_bert_transfer"
BRANCH_CALLS = 8
SOURCES = (
    "tools/evaluate_repository_cascade.py", "tools/repository_reference.py", "src/relay_router/cascade.py",
    "src/relay_router/cascade_evaluation.py", "src/relay_router/cascade_verifier.py",
    "src/relay_router/integrations/openhands.py", "src/relay_router/integrations/openhands_io.py",
    "src/relay_router/integrations/groq_recovery.py",
    "src/relay_router/integrations/openhands_cascade.py", "src/relay_router/workflow.py",
    "src/relay_router/maintenance_reference.py", "src/routerbench/routellm_reference.py",
    "configs/groq_paid_tariffs_20261112.json", "configs/routellm_reference.json",
    "docs/openhands_repository_validation_v2.md",
)
OLD_SUITES = (
    "data/imported/harness-public-v1/frozen_suite.json",
    "data/imported/harness-public-v1/frozen_suite_v2.json",
    "data/imported/harness-adaptive-v1/fresh_suite.json",
    "data/imported/harness-cascade-v1/fresh_suite.json",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def normalized(prompt):
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def load_cohort(path):
    cohort = read(path)
    if set(cohort) != {"cohort_id", "tasks"} or not isinstance(cohort["cohort_id"], str) or not cohort["cohort_id"].strip():
        raise ValueError("Declare cohort_id and tasks")
    if not isinstance(cohort["tasks"], list) or not cohort["tasks"]:
        raise ValueError("Empty cohort")
    historical_ids, prompts = set(), set()
    for name in OLD_SUITES:
        for row in read(ROOT / name)["tasks"]:
            historical_ids.add(row["task_id"])
            prompts.add(normalized(row["instruction"]))
            if row.get("prompt_text"):
                prompts.add(normalized(row["prompt_text"]))
    seen = {k: set() for k in ("task_id", "group_id", "source")}
    for row in cohort["tasks"]:
        if set(row) != {"task_id", "group_id", "source", "revision", "prompt"}:
            raise ValueError("Only task_id, group_id, source, revision and the original prompt may enter the public cohort")
        if any(type(v) is not str or not v.strip() for v in row.values()):
            raise ValueError("All task metadata and prompts must be nonempty text")
        if (not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}", row["task_id"])
                or row["task_id"].endswith(".") or Path(row["task_id"]).is_reserved()):
            raise ValueError("Use portable task identifiers")
        if row["task_id"] in historical_ids or normalized(row["prompt"]) in prompts:
            raise ValueError("Historical or duplicate task")
        prompts.add(normalized(row["prompt"]))
        for field, values in seen.items():
            value = row[field].casefold().strip().rstrip("/")
            if value in values:
                raise ValueError(f"Expected one task per unique {field}")
            values.add(value)
    return cohort


def score(cohort_path, output):
    write(output, score_cohort(load_cohort(cohort_path), cohort_path))


def load_host(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("relay_repository_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_case(host, task):
    # Other host inputs include isolated verification images, private tests,
    # dependency locks and code used by the adapter, all outside the snapshot.
    snapshot, verifier, dependencies = host.prepare(task)
    snapshot = Path(snapshot).resolve(strict=True)
    if type(verifier) is not CommandVerifier or not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("Host must return a captured CommandVerifier and pinned evaluator dependencies")
    for path in dependencies.values():
        if Path(path).resolve(strict=True).is_relative_to(snapshot):
            raise ValueError("Keep private evaluator and host dependencies outside the agent workspace")
    contract = {"snapshot_sha256": digest(_manifest(snapshot)), "public_verifier_sha256": verifier.config_sha256,
                "host_inputs_sha256": {key: file_hash(path) for key, path in dependencies.items()}}
    return snapshot, verifier, contract


def freeze(cohort_path, scores_path, host_path, output, hourly_usd, rate_source):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Freeze into an empty campaign directory before any provider request")
    cohort = load_cohort(cohort_path)
    scores = read(scores_path)
    routes = frozen_routes(cohort, cohort_path, scores)
    if (scores.get("routes") != routes or scores.get("threshold") != .50 or scores.get("truncated_route") != "fixed_strong"):
        raise ValueError("BERT routing differs from the routes sealed at scoring")
    if hourly_usd is not None and (duration(hourly_usd) == 0 or not rate_source or not rate_source.strip()):
        raise ValueError("An economic scenario needs a positive dedicated-host rate and source")
    host = load_host(host_path)
    contract = {
        "cohort_sha256": file_hash(cohort_path), "scores_sha256": file_hash(scores_path),
        "host_sha256": file_hash(host_path), "cases": {}, "routes": routes,
        "code_sha256": {name: file_hash(ROOT / name) for name in SOURCES},
        "packages": {name: importlib.metadata.version(name) for name in ("openhands-sdk", "litellm")},
        "python": sys.version, "models": list(CATALOG), "usd_per_dedicated_host_hour": hourly_usd,
        "rate_source": rate_source, "max_admission_usd_per_arm": .05,
        "max_iterations_per_branch": BRANCH_CALLS, "max_http_calls_per_branch": BRANCH_CALLS,
        "max_http_calls": {"relay": 2 * BRANCH_CALLS, "fixed": BRANCH_CALLS},
        "timeout_seconds": 600, "max_output_tokens": 1024,
        "bert_threshold": .50, "bert_truncated_route": "fixed_strong", "arm_order": {},
    }
    for task in cohort["tasks"]:
        task_id = task["task_id"]
        contract["cases"][task_id] = public_case(host, task)[2]
        contract["arm_order"][task_id] = sorted(ARMS, key=lambda arm: digest([cohort["cohort_id"], task_id, arm]))
    write(output / "freeze.json", {"freeze_id": digest(contract), "contract": contract})


def verify_sources(freeze_doc, cohort_path, scores_path, host_path):
    contract = freeze_doc["contract"]
    if (digest(contract) != freeze_doc["freeze_id"]
            or (contract["cohort_sha256"], contract["scores_sha256"], contract["host_sha256"])
            != tuple(file_hash(p) for p in (cohort_path, scores_path, host_path))
            or contract["code_sha256"] != {name: file_hash(ROOT / name) for name in SOURCES}
            or contract["python"] != sys.version
            or any(importlib.metadata.version(k) != v for k, v in contract["packages"].items())):
        raise ValueError("Acquisition inputs, packages or code changed after freeze")
    return contract


def preflight_routes(cohort_path, host_path, output):
    """Observe the frozen first routing decision on the unchanged originals only."""
    output = Path(output)
    freeze_path = output / "freeze.json"
    frozen = read(freeze_path)
    contract = frozen["contract"]
    cohort = read(cohort_path)  # Enrolment was checked at freeze. No historical test files are needed.
    if (digest(contract) != frozen["freeze_id"] or file_hash(cohort_path) != contract["cohort_sha256"]
            or file_hash(host_path) != contract["host_sha256"]
            or contract["code_sha256"] != {name: file_hash(ROOT / name) for name in SOURCES}
            or contract["python"] != sys.version
            or any(importlib.metadata.version(k) != v for k, v in contract["packages"].items())
            or set(contract["cases"]) != {t["task_id"] for t in cohort["tasks"]}):
        raise ValueError("Cohort, host, execution environment or freeze changed")
    seal_path = output / "public-route-preflight.json"
    if seal_path.exists():
        sealed = read(seal_path)
        report = sealed["report"]
        if (digest(report) != sealed["report_sha256"] or report["freeze_sha256"] != file_hash(freeze_path)
                or report["freeze_id"] != frozen["freeze_id"]):
            raise ValueError("Previous public preflight seal changed")
        return report
    host = load_host(host_path)
    if not callable(getattr(host, "prepare_public", None)):
        raise ValueError("Supply a trusted host with a public-only prepare_public(task) method")
    rows = []
    for task in cohort["tasks"]:
        expected = contract["cases"][task["task_id"]]
        row = {"task_id": task["task_id"], "status": "inconclusive", "verifier_wall_seconds": None,
               "snapshot_sha256": None, "public_verifier_sha256": None, "reason": "public_check_unavailable"}
        try:
            original, verifier = host.prepare_public(task)
            if type(verifier) is not CommandVerifier:
                raise TypeError("A captured public CommandVerifier is required")
            row["snapshot_sha256"] = digest(_manifest(Path(original)))
            row["public_verifier_sha256"] = verifier.config_sha256
            if (row["snapshot_sha256"] != expected["snapshot_sha256"]
                    or row["public_verifier_sha256"] != expected["public_verifier_sha256"]):
                row["reason"] = "frozen_public_inputs_changed"
            else:
                with tempfile.TemporaryDirectory(prefix="relay-public-route-") as tmp:
                    copy = Path(tmp) / "workspace"
                    shutil.copytree(original, copy)
                    if digest(_manifest(copy)) != expected["snapshot_sha256"]:
                        raise ValueError("Public snapshot changed during copy")
                    started = time.perf_counter()
                    try:
                        check = verifier.verify(copy)
                    finally:
                        row["verifier_wall_seconds"] = time.perf_counter() - started
                    if digest(_manifest(Path(original))) != expected["snapshot_sha256"]:
                        raise ValueError("Original snapshot changed during check")
                    row.update(status=check.status, reason=check.reason, verification=check.to_dict())
        except (OSError, KeyError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            row.update(status="inconclusive", reason="public_preflight_" + type(exc).__name__)
        row["initial_route"] = {"failed": "candidate_20b", "passed": "direct_120b",
                                "inconclusive": "no_dispatch"}[row["status"]]
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("failed", "passed", "inconclusive")}
    report = {"kind": "frozen_initial_public_routes", "freeze_id": frozen["freeze_id"], "freeze_sha256": file_hash(freeze_path),
              "cohort_id": cohort["cohort_id"], "cohort_sha256": contract["cohort_sha256"],
              "host_sha256": contract["host_sha256"], "source_sha256": contract["code_sha256"],
              "tasks": len(rows), "public_verification_status": counts,
              "candidate_20b_opportunities": counts["failed"], "direct_120b": counts["passed"],
              "no_dispatch": counts["inconclusive"], "total_verifier_wall_seconds": math.fsum(
                  r["verifier_wall_seconds"] for r in rows if r["verifier_wall_seconds"] is not None),
              "provider_requests": 0, "private_evaluation": False, "acceptance": None, "savings": None,
              "finding": ("no_observed_opportunities_to_reduce_model_cost_by_routing_to_20b" if counts["failed"] == 0
                          else "candidate_20b_would_be_attempted_quality_and_cost_unmeasured"), "rows": rows}
    write(seal_path, {"report_sha256": digest(report), "report": report})
    return report


def run_arm(host, task, arm, snapshot, verifier, destination):
    # All three arms use the same fresh agent, tools, issue and final public check.
    # The two fixed arms do not pay for Relay's pre-routing check on the original.
    fixed_cheap = arm == "fixed_cheap"
    started = time.perf_counter()
    def build_agent(context):
        agent = host.build_agent(task, context)
        llm = agent.llm
        rates = read(ROOT / "configs/groq_paid_tariffs_20261112.json")["models"][context.model]
        if (llm.model != context.model or llm.max_output_tokens != 1024 or llm.temperature != 0
                or llm.reasoning_effort != "low"
                or llm.input_cost_per_token != rates["input_usd_per_million"] / 1e6
                or llm.output_cost_per_token != rates["output_usd_per_million"] / 1e6):
            raise ValueError("The host agent differs from the frozen generation contract")
        return agent

    sdk_attempt = make_openhands_attempt(build_agent, task["prompt"],
        reservation_usd_by_model={CHEAP: .005, STRONG: .010}, tools_are_local=True, max_iterations=BRANCH_CALLS,
        recover_tool_use_failed=True)

    def attempt(context):
        # Internal SDK repairs can consume more than one HTTP call per iteration.
        # Apply the same per-branch budget in all arms, also charging the shared task ledger.
        ledger = context.budget
        start_count = ledger.snapshot()["http_calls"]

        def begin_call(*args, **kwargs):
            if ledger.snapshot()["http_calls"] - start_count >= BRANCH_CALLS:
                raise BudgetExceeded("The fixed per-branch HTTP limit was reached")
            return ledger.begin_call(*args, **kwargs)

        branch_budget = SimpleNamespace(begin_call=begin_call, finish_call=ledger.finish_call,
                                        record_unobserved_transport=ledger.record_unobserved_transport)
        return sdk_attempt(replace(context, budget=branch_budget))

    run = run_cascade(snapshot_dir=snapshot, output_dir=destination, task_id=task["task_id"],
        candidate_model=STRONG if fixed_cheap else CHEAP, baseline_model=CHEAP if fixed_cheap else STRONG,
        run_attempt=attempt, verifier=verifier, budget=SpendBudget(.05, 2 * BRANCH_CALLS if arm == "relay" else BRANCH_CALLS, 600),
        mode="experimental" if arm == "relay" else "baseline", allow_experimental=arm == "relay",
        require_initial_failure=arm == "relay")
    wall = time.perf_counter() - started
    return run, wall


def observation(run, wall, private_status, private_seconds):
    accounting = run["accounting"]
    selected = run["selected_attempt"]
    selected_ids = set(run["attempts"][selected - 1]["call_ids"]) if selected else set()
    selected_post = any(call["call_id"] in selected_ids and call["model"] == run["selected_model"]
                        and call["status"] == "completed" and call["http_status"] == 200
                        for call in accounting["calls"])
    zero_call_errors = [row["error_type"] for row in run["attempts"] if row["status"] == "error" and row["transport_calls"] == 0]
    return {"accepted": bool(selected_post and private_status == "passed"),
            "selected_model": run["selected_model"], "escalated": run["escalated"],
            "operational_passed": run["operational_passed"], "private_status": private_status,
            "run_reason": run["reason"], "errors_before_dispatch": zero_call_errors,
            "selected_without_verified_response": bool(selected and not selected_post),
            "public_check_inconclusive": any(v.get("status") == "inconclusive" for v in
                [run["initial_verification"] or {}, *(a["verification"] for a in run["attempts"])]),
            "run_seconds": wall, "private_seconds": private_seconds,
            "operational_verifier_seconds": sum(v.get("duration_seconds", 0.) for v in
                [run["initial_verification"] or {}, *(a["verification"] for a in run["attempts"])]),
            "provider": {key: accounting.get(key) for key in ("cost_usd", "cost_complete", "cost_basis",
                "known_cost_subtotal_usd", "cost_upper_bound_usd", "cost_bound_basis", "http_calls",
                "http_calls_complete", "journal_complete")}}


def acquire(cohort_path, scores_path, host_path, output):
    output = Path(output)
    frozen = read(output / "freeze.json")
    contract = verify_sources(frozen, cohort_path, scores_path, host_path)
    cohort, host = load_cohort(cohort_path), load_host(host_path)
    # Exclusive task/arm directories plus the per-task seal protect continuations:
    # a crash may have dispatched a call. An unsealed task must be audited.
    acquisition_ready = False
    for task in cohort["tasks"]:
        task_id = task["task_id"]
        root = output / "episodes" / task_id
        if (root / "seal.json").is_file():
            load_episode(root, frozen, task_id)
            continue
        if root.exists():
            raise FileExistsError("Unsealed task exists; audit its calls before continuing acquisition")
        if not acquisition_ready:
            configure = getattr(host, "prepare_acquisition", None)
            if configure is not None:
                configure()
            acquisition_ready = True
        root.mkdir(parents=True, exist_ok=False)
        snapshot, verifier, inputs = public_case(host, task)
        if inputs != contract["cases"][task_id]:
            raise ValueError("Task snapshot or evaluator changed after freeze")
        results = {}
        for arm in contract["arm_order"][task_id]:
            if public_case(host, task)[2] != inputs:
                raise ValueError("Task inputs changed between arms")
            results[arm] = run_arm(host, task, arm, snapshot, verifier, root / arm)
            accounting = results[arm][0]["accounting"]
            if (not accounting["http_calls_complete"] or not accounting["journal_complete"]
                    or not accounting["cost_complete"] and accounting["cost_upper_bound_usd"] is None):
                write(root / "accounting_hold.json", {"freeze_id": frozen["freeze_id"], "arm": arm,
                      "task_id": task_id, "reason": "unobserved_dispatch_or_unbounded_spend",
                      "known_cost_subtotal_usd": accounting["known_cost_subtotal_usd"],
                      "http_calls_observed": accounting["http_calls"], "http_calls_complete": accounting["http_calls_complete"],
                      "unobserved_transport_incidents": len(accounting["unobserved_transport"])})
                raise ValueError("Incomplete transport trace or unbounded spending; audit before another acquisition")
        observations = {}
        for arm, (run, wall) in results.items():
            start = time.perf_counter()
            status = "no_selected_patch"
            if run["selected_workspace"] is not None:
                status = host.private_verdict(task, root / arm / run["selected_workspace"])
                if status not in ("passed", "failed", "inconclusive"):
                    raise ValueError("Private evaluator must distinguish pass, failure and infrastructure")
            observations[arm] = observation(run, wall, status, time.perf_counter() - start)
        if public_case(host, task)[2] != inputs:
            raise ValueError("Inputs changed during evaluation")
        content = {"freeze_id": frozen["freeze_id"], "task_id": task_id, "arms": observations,
                   "runtime_sha256": runtime_hashes(root)}
        write(root / "seal.json", {"record_sha256": digest(content), "record": content})
    return report(cohort, output, frozen)


def runtime_hashes(root):
    evidence = {}
    for arm in ARMS:
        directory = root / arm
        for path in [directory / "result.json", directory / "manifest.json",
                     *sorted(directory.glob("attempt-*/attempt.json"))]:
            evidence[path.relative_to(root).as_posix()] = file_hash(path)
        transport = directory / "transport.jsonl"
        accounting = read(directory / "result.json")["accounting"]
        if transport.exists() or accounting["http_calls"] or accounting.get("unobserved_transport"):
            evidence[transport.relative_to(root).as_posix()] = file_hash(transport)
    return evidence


def load_episode(root, frozen, task_id):
    sealed = read(root / "seal.json")
    rec = sealed["record"]
    if (sealed["record_sha256"] != digest(rec) or rec["freeze_id"] != frozen["freeze_id"]
            or rec["task_id"] != task_id or set(rec["arms"]) != set(ARMS)
            or rec["runtime_sha256"] != runtime_hashes(root)):
        raise ValueError("Sealed episode or transport evidence changed")
    return rec["arms"]


def cost_row(task, arm, raw, route, rate):
    provider = raw["provider"]
    router_seconds = route["router_seconds"] + route["amortized_load_seconds"] if arm == BERT else 0.
    seconds = duration(raw["run_seconds"]) + duration(raw["private_seconds"]) + router_seconds
    local = seconds * rate / 3600 if rate is not None else None
    # Without a host rate the LLM subtotal remains visible; no net saving claim.
    complete = provider["cost_complete"] and local is not None
    basis = "uncached_provider_tariff_plus_dedicated_host"
    upper = provider["cost_usd"] if provider["cost_complete"] else provider["cost_upper_bound_usd"]
    return {"task_id": task["task_id"], "group_id": task["group_id"], **raw, "router_seconds": router_seconds,
            "chargeable_host_seconds": seconds, "host_cost_scenario_usd": local,
            "cost_complete": complete, "cost_usd": provider["cost_usd"] + local if complete else None,
            "cost_basis": basis if complete else "unknown", "cost_bound_basis": basis,
            "cost_upper_bound_usd": upper + local if upper is not None and local is not None else None,
            "known_cost_subtotal_usd": provider["known_cost_subtotal_usd"] + (local or 0.)}


def report(cohort, output, frozen):
    contract = frozen["contract"]
    all_rows = {arm: [] for arm in (*ARMS, BERT)}
    for task in cohort["tasks"]:
        task_id = task["task_id"]
        observations = load_episode(output / "episodes" / task_id, frozen, task_id)
        route = contract["routes"][task_id]
        for arm in all_rows:
            raw = observations[route["arm"] if arm == BERT else arm]
            all_rows[arm].append(cost_row(task, arm, raw, route, contract["usd_per_dedicated_host_hour"]))
    summaries = {}
    for arm, rows in all_rows.items():
        summary = summarize_policy(rows)
        for field in ("chargeable_host_seconds", "operational_verifier_seconds", "private_seconds", "router_seconds"):
            summary[field] = math.fsum(row[field] for row in rows)
        summary["provider_dispatches"] = sum(row["provider"]["http_calls"] for row in rows)
        summaries[arm] = summary
    comparisons = {arm: paired_comparison(all_rows["relay"], rows) for arm, rows in all_rows.items() if arm != "relay"}
    primary = comparisons["fixed_strong"]
    ci = primary["bootstrap"]["savings_interval_95"]
    all_points_available = all(s["cost_complete"] for s in summaries.values())
    pilot = (len(cohort["tasks"]) >= 20 and primary["paired_losses"] == 0
             and all_points_available
             and primary["savings_per_accepted_task"] is not None and primary["savings_per_accepted_task"] >= .10)
    gate = pilot and ci is not None and ci[0] > 0 and primary["harm_upper_95_one_sided_exact"] <= .02
    infrastructure = sorted({row["task_id"] for rows in all_rows.values() for row in rows if
                            row["private_status"] == "inconclusive" or row["errors_before_dispatch"]
                            or row["public_check_inconclusive"] or row["selected_without_verified_response"]})
    accounting_interval = primary["savings_per_accepted_bounds"]
    conservative_threshold = accounting_interval is not None and accounting_interval[0] >= .10
    return {"kind": "repository_cascade_paired_evaluation", "freeze_id": frozen["freeze_id"],
            "cohort_id": cohort["cohort_id"], "summary": summaries, "relay_compared_to": comparisons,
            "pilot_criteria_met": pilot and not infrastructure, "strong_criteria_met": gate and not infrastructure,
            "verification_or_predispatch_infrastructure_failure_task_ids": infrastructure,
            "savings_at_least_10_percent_even_at_adverse_cost_endpoints": conservative_threshold,
            "all_operating_costs_point_identified": all_points_available,
            "evidence_scope": "held-out cohort and declared compute-price scenario, not an invoice or third-party usability test",
            "rows": all_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("score", "freeze", "run", "preflight-routes"):
        sub = commands.add_parser(name)
        sub.add_argument("--cohort", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        if name in ("freeze", "run"):
            sub.add_argument("--scores", required=True, type=Path)
        if name != "score":
            sub.add_argument("--host", required=True, type=Path)
        if name == "freeze":
            sub.add_argument("--host-usd-hour", type=float)
            sub.add_argument("--rate-source")
    args = parser.parse_args()
    if args.action == "score":
        score(args.cohort, args.output)
    elif args.action == "freeze":
        freeze(args.cohort, args.scores, args.host, args.output, args.host_usd_hour, args.rate_source)
    elif args.action == "preflight-routes":
        result = preflight_routes(args.cohort, args.host, args.output)
        print(json.dumps({k: result[k] for k in ("tasks", "candidate_20b_opportunities", "direct_120b", "no_dispatch",
                                                 "total_verifier_wall_seconds", "provider_requests")}))
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
        final = acquire(args.cohort, args.scores, args.host, args.output)
        out = args.output / "report.json"
        if not out.exists():
            write(out, final)
        elif read(out) != final:
            raise ValueError("A prior completed report disagrees with the sealed episodes")


if __name__ == "__main__":
    main()
