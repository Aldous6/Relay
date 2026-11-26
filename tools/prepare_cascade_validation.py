"""Freeze all unused eligible public tasks before any cascade model outcome.

No LLM calls. Public operational checks and private final evaluators are stored
separately. The original source eligibility and reference checks are preserved;
the cohort is not balanced or replaced after model outcomes are available.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_adaptive_pilot as prior

DEFAULT_OUTPUT = ROOT / "data/imported/harness-cascade-v1"
TASK_PATTERN = re.compile(r"HumanEval[-/]\d+|QuixBugs-[A-Za-z_]+")
HISTORICAL_SUITES = (
    "data/imported/harness-public-v1/frozen_suite.json",
    "data/imported/harness-public-v1/frozen_suite_v2.json",
    "data/imported/harness-adaptive-v1/fresh_suite.json",
)
EXPECTED_FRESH_COUNTS = {"function-implementation": 48, "function-repair": 2}
MAX_HTTP_ATTEMPTS = 312
TOKEN_BUDGET_PER_MODEL = 180000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_id(value):
    return value.replace("HumanEval/", "HumanEval-")


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _json_values(path, content):
    text = content.decode("utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def history_from_paths(paths, helper, *, root=ROOT):
    """Conservatively exclude every referenced source ID, not only successes."""
    ids, groups, prompts, provenance = set(), set(), set(), []
    for path in sorted(set(Path(path).resolve() for path in paths)):
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        referenced = {normalized_id(value) for value in TASK_PATTERN.findall(text)}
        ids.update(referenced)
        for value in _json_values(path, content):
            for row in _objects(value):
                task = row.get("task_id")
                if isinstance(task, str) and TASK_PATTERN.fullmatch(task):
                    ids.add(normalized_id(task))
                    if isinstance(row.get("group_id"), str):
                        groups.add(row["group_id"])
                    if isinstance(row.get("prompt_text"), str):
                        prompts.add(prior.normalized_prompt_digest(row["prompt_text"]))
                    if all(isinstance(row.get(key), str) for key in ("instruction", "initial_source", "public_tests")):
                        prompts.add(prior.normalized_prompt_digest(helper.task_prompt(row)))
                    existing = row.get("normalized_prompt_sha256")
                    if isinstance(existing, str) and re.fullmatch(r"[a-f0-9]{64}", existing):
                        prompts.add(existing)
        provenance.append({"path": path.relative_to(root.resolve()).as_posix(),
                           "sha256": digest(content), "bytes": len(content),
                           "referenced_task_ids": sorted(referenced)})
    return {"task_ids": sorted(ids), "group_ids": sorted(groups),
            "normalized_prompt_sha256": sorted(prompts), "files": provenance,
            "rule": "Exclude IDs mentioned anywhere in historical config/report/output JSON or JSONL, including unsuccessful preflights, plus exact normalized prompt overlap"}


def capture_history(helper):
    command = ["rg", "--no-ignore", "-l", r"HumanEval[-/][0-9]+|QuixBugs-",
               "outputs", "reports", "configs", "-g", "*.json", "-g", "*.jsonl"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode not in (0, 1):
        raise RuntimeError("Could not enumerate historical source-task references")
    paths = [ROOT / name for name in completed.stdout.splitlines()]
    paths.extend(ROOT / name for name in HISTORICAL_SUITES)
    return history_from_paths(paths, helper)


def fresh_inventory(rows, history, helper):
    historical_ids = set(history["task_ids"])
    historical_groups = set(history["group_ids"])
    historical_prompts = set(history["normalized_prompt_sha256"])
    fresh, excluded, seen = [], [], set()
    # This source-ID hash order is unchanged from the original acquisition.
    for row in sorted(rows, key=lambda item: helper.rank(item["task_id"])):
        prompt = helper.task_prompt(row)
        group = prior.normalized_prompt_digest(prompt)
        if row["task_id"] in historical_ids or row["task_id"] in historical_groups:
            reason = "historical_id_or_group"
        elif group in historical_prompts:
            reason = "historical_normalized_prompt"
        elif group in seen:
            reason = "duplicate_normalized_prompt_in_candidate_pool"
        else:
            fresh.append(row)
            seen.add(group)
            continue
        excluded.append({"task_id": row["task_id"], "reason": reason, "normalized_prompt_sha256": group})
    return fresh, excluded


def check_status(result):
    return ("passed" if result["passed"] else
            "failed" if result["category"] == "test_failure" else "inconclusive")


def reference_eligible(public_gold, private_gold, original_private):
    # Preserve the earlier source/reference eligibility exactly. A timeout of
    # the broken original is not used to select on any generated model result.
    return public_gold["passed"] and private_gold["passed"] and not original_private["passed"]


def preflight_all(rows, helper):
    tasks, private_rows, audit, excluded = [], {}, [], []
    for position, row in enumerate(rows):
        with tempfile.TemporaryDirectory(prefix="relay-cascade-preflight-") as temporary:
            directory = Path(temporary)
            gold_public = helper.run_check(row["gold_source"], row["public_tests"], directory=directory / "gold-public", timeout=8)
            gold_private = helper.run_check(row["gold_source"], row["private_tests"], directory=directory / "gold-private", timeout=8)
            original_private = helper.run_check(row["initial_source"], row["private_tests"], directory=directory / "original-private", timeout=2)
            # Operational policy input, using only public information. An
            # inconclusive initial check remains in the cohort, never omitted.
            original_public = helper.run_check(row["initial_source"], row["public_tests"], directory=directory / "original-public", timeout=8)
        def metadata(result, timeout):
            return {"passed": result["passed"], "category": result["category"],
                    "seconds": result["seconds"], "timeout_seconds": timeout}
        record = {"task_id": row["task_id"], "source_order_index": position,
                  "gold_public": metadata(gold_public, 8), "gold_private": metadata(gold_private, 8),
                  "original_private": metadata(original_private, 2), "original_public": metadata(original_public, 8)}
        audit.append(record)
        if not reference_eligible(gold_public, gold_private, original_private):
            excluded.append(record)
            continue
        item = {key: value for key, value in row.items() if key not in {"gold_source", "private_tests"}}
        prompt = helper.task_prompt(item)
        private_tests = row["private_tests"]
        item.update(group_id=item["task_id"], split="validation", prompt_text=prompt,
                    routing_text_sha256=digest(prompt.encode("utf-8")),
                    normalized_prompt_sha256=prior.normalized_prompt_digest(prompt),
                    initial_source_sha256=digest(item["initial_source"].encode("utf-8")),
                    public_tests_sha256=digest(item["public_tests"].encode("utf-8")),
                    private_tests_sha256=digest(private_tests.encode("utf-8")),
                    initial_public_check={"status": check_status(original_public), **metadata(original_public, 8)})
        tasks.append(item)
        private_rows[item["task_id"]] = {"private_tests": private_tests,
                                         "sha256": digest(private_tests.encode("utf-8"))}
    return tasks, private_rows, audit, excluded


def validate_disjoint(tasks, history):
    if not tasks:
        raise ValueError("No new reference-eligible tasks")
    for field, historical in (("task_id", history["task_ids"]), ("group_id", history["group_ids"]),
                              ("normalized_prompt_sha256", history["normalized_prompt_sha256"])):
        values = [row[field] for row in tasks]
        if len(set(values)) != len(values) or set(values).intersection(historical):
            raise ValueError(f"Frozen new tasks overlap or duplicate {field}")
    if any("private_tests" in row or "gold_source" in row for row in tasks):
        raise ValueError("Agent-facing tasks must not contain final evaluators or reference solutions")


def payload(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def exclusive(path, value):
    content = payload(value)
    with path.open("xb") as stream:
        stream.write(content)
    return {"file": path.name, "sha256": digest(content), "bytes": len(content)}


def prepare(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if not (output == DEFAULT_OUTPUT.resolve() or output.is_relative_to((ROOT / "outputs/harness-cascade-v1").resolve())):
        raise ValueError("Use the new cascade preparation directory or outputs/harness-cascade-v1")
    if output.exists():
        raise FileExistsError("A frozen preparation cannot be overwritten")
    values, pinned_provenance = prior.read_pinned_inputs()
    source_manifest = values["data/imported/harness-public-v1/sources.json"]
    source_verification = prior.verify_imports(source_manifest)
    helper = prior.helpers()
    history = capture_history(helper)
    candidates = prior.candidates(helper)
    inventory, excluded_history = fresh_inventory(candidates, history, helper)
    counts = dict(Counter(row["workflow"] for row in inventory))
    if counts != EXPECTED_FRESH_COUNTS:
        raise ValueError(f"The predeclared unused source pool changed: {counts}; do not silently choose substitutes")
    output.mkdir(parents=True, exist_ok=False)
    inventory_record = {"schema_version": 1, "kind": "cascade_validation_source_inventory_before_preflight",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "task_ids": [row["task_id"] for row in inventory], "workflow_counts": counts,
                        "source_candidates": dict(Counter(row["workflow"] for row in candidates)),
                        "selection": "ALL unused source-eligible tasks in original SHA256 rank order; no model outcomes",
                        "normalization": prior.NORMALIZATION, "historical_exclusions": excluded_history,
                        "model_calls": 0}
    files = [exclusive(output / "candidate_inventory.json", inventory_record),
             exclusive(output / "historical_exclusions.json", history)]
    tasks, private_rows, checks, excluded = preflight_all(inventory, helper)
    validate_disjoint(tasks, history)
    files.append(exclusive(output / "preflight.json", {"checks": checks, "excluded": excluded,
                                                       "model_calls": 0, "historical_eligibility_preserved": True}))
    private_file = exclusive(output / "private_evaluators.json", {
        "schema_version": 1, "kind": "host_only_final_evaluators", "evaluation_id": "openhands-cascade-validation-v1",
        "scope": "Never expose to agent tools, prompt or routing; use only after each branch completes",
        "tasks": private_rows})
    files.append(private_file)
    original = json.loads((ROOT / HISTORICAL_SUITES[-1]).read_text(encoding="utf-8"))
    suite = {key: original[key] for key in ("schema_version", "models", "baseline_model", "max_calls_per_episode",
                                           "max_output_tokens", "temperature", "reasoning_effort", "stop_on_public_pass",
                                           "source_manifest_sha256")}
    suite.update(kind="frozen_cascade_independent_validation", evaluation_id="openhands-cascade-validation-v1",
                 created_at_utc=datetime.now(timezone.utc).isoformat(),
                 max_total_http_attempts=MAX_HTTP_ATTEMPTS, max_observed_tokens_per_model=TOKEN_BUDGET_PER_MODEL,
                 scope="New to Relay model-outcome selection; public historical function tasks, not production generalization",
                 task_selection="All 50 remaining source-eligible candidates after excluding historical IDs including prior reference-preflight mentions, minus only original reference preflight failures; no balancing or outcome replacement",
                 source_candidate_counts=counts, workflow_counts=dict(Counter(row["workflow"] for row in tasks)),
                 candidate_inventory_sha256=files[0]["sha256"], historical_exclusions_sha256=files[1]["sha256"],
                 private_evaluator_file=private_file["file"], private_evaluator_sha256=private_file["sha256"],
                 source_manifest=source_manifest, source_verification=source_verification,
                 policy={"version": "initial-failure-guarded-cascade-v1", "require_initial_failure": True,
                         "initial_passed": "baseline_only", "initial_failed": "candidate_then_baseline_if_operational_not_passed",
                         "initial_inconclusive": "stop_without_generation_and_count_as_unaccepted",
                         "fallback": "fresh_snapshot", "max_model_attempts_per_task": 2,
                         "acceptance_used_for_routing": False},
                 criteria={"minimum_point_savings_per_accepted": .10, "maximum_observed_paired_losses": 0,
                           "cost_complete": True, "bootstrap_resamples": 10000, "bootstrap_seed": 20261118,
                           "confidence_level": .95, "strong_evidence_planning_harm_upper_max": .02,
                           "strong_evidence_planning_groups_if_zero_losses": 149,
                           "small_cohort_is_not_noninferiority_proof": True},
                 excluded_preflight=excluded, tasks=tasks,
                 limitation="The new available mixture differs from the earlier 12 HumanEval + 12 QuixBugs cohort; do not attribute differences solely to the router")
    files.append(exclusive(output / "fresh_suite.json", suite))
    provenance = {relative: digest((ROOT / relative).read_bytes()) for relative in (
        "tools/prepare_cascade_validation.py", "tools/prepare_adaptive_pilot.py",
        "tools/freeze_harness_suite.py", "src/relay_router/harness_episode.py",
        "docs/openhands_cascade_v1.md")}
    manifest = {"schema_version": 1, "kind": "cascade_validation_preparation",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
                "source_sha256": provenance, "pinned_historical_inputs": pinned_provenance,
                "model_calls": 0, "new_tasks": len(tasks), "preflight_excluded": len(excluded),
                "all_remaining_reference_eligible_tasks_selected": True,
                "historical_artifacts_modified": False, "private_evaluators_separate": True}
    exclusive(output / "preparation_manifest.json", manifest)
    return {"output": str(output), "candidate_count": len(inventory), "tasks": len(tasks),
            "workflow_counts": suite["workflow_counts"], "excluded_preflight": len(excluded),
            "initial_public_statuses": dict(Counter(row["initial_public_check"]["status"] for row in tasks)),
            "files": files, "model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
