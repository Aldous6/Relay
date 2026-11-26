"""Pinned, outcome-blind RouteLLM-BERT transfer for a cohort of repository issues."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from relay_router.maintenance_reference import CATALOG, PINNED, score_records

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_cohort(cohort, cohort_path):
    records = [{"task_id": t["task_id"], "routing_text": t["prompt"]} for t in cohort["tasks"]]
    sources = {name: file_hash(ROOT / name) for name in (
        "tools/repository_reference.py", "src/relay_router/maintenance_reference.py",
        "src/routerbench/routellm_reference.py", "configs/routellm_reference.json")}
    result = score_records(records)
    result.update(cohort_sha256=file_hash(cohort_path), score_source_sha256=sources)
    result["routes"] = frozen_routes(cohort, cohort_path, result)
    result["threshold"] = .50
    result["truncated_route"] = "fixed_strong"
    return result


def frozen_routes(cohort, cohort_path, scores):
    reference = scores["reference"]
    for name in ("tools/repository_reference.py", "src/relay_router/maintenance_reference.py", "src/routerbench/routellm_reference.py",
                 "configs/routellm_reference.json"):
        if scores["score_source_sha256"].get(name) != file_hash(ROOT / name):
            raise ValueError("The BERT scoring implementation or its configuration changed")
    if (scores["cohort_sha256"] != file_hash(cohort_path) or not reference["checkpoint_verified"]
            or reference["predictor_injected"] or reference["revision"] != PINNED["checkpoint_revision"]
            or reference["weights_sha256"] != PINNED["weights_sha256"] or reference["catalog"] != list(CATALOG)
            or reference["tokenization"] != PINNED["reference_tokenization"]):
        raise ValueError("A verified, pinned BERT score file for this exact cohort is required")
    tasks = cohort["tasks"]
    if len(scores["rows"]) != len(tasks):
        raise ValueError("BERT scores must cover the full cohort")
    routes = {}
    # A single declared cold start, amortized over the frozen campaign.
    load = duration(reference["load_seconds"]) / len(tasks)
    for task, row in zip(tasks, scores["rows"]):
        sha = hashlib.sha256(task["prompt"].encode()).hexdigest()
        if row["task_id"] != task["task_id"] or row["routing_text_sha256"] != sha:
            raise ValueError("BERT score does not correspond to the original agent message")
        probability = row["probability"]
        if type(probability) not in (float, int) or not 0 <= probability <= 1 or type(row["truncated_tokens"]) is not int or row["truncated_tokens"] < 0:
            raise ValueError("Invalid BERT score or truncation count")
        routes[task["task_id"]] = {
            "arm": "fixed_strong" if probability >= .50 or row["truncated_tokens"] > 0 else "fixed_cheap",
            "router_seconds": duration(row["router_seconds"]), "amortized_load_seconds": load,
        }
    return routes


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Duration or price must be finite and nonnegative")
    return value
