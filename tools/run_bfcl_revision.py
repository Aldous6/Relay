"""New BFCL experiment on the 19 conversations with no parent-run attempts.

Prepare by default. --execute uses an existing authorized environment key or
hidden terminal entry. The previous experiment is audited, never resumed or
pooled. Selection depends on any attempted conversation, never its outcome.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_bfcl
from report_bfcl import audit_bfcl
from routerbench.agent_evaluation import object_hash
from routerbench.bfcl_harness import (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade,
                                     prepare_bfcl_episodes, run_bfcl_episodes)
from routerbench.bfcl_litellm import BFCLPacedGenerator
from routerbench.bfcl_session import BFCLSessionGenerator

DEFAULT_PARENT = ROOT / "outputs/bfcl-comparison/20261112-development-v2/runplan/run"
DEFAULT_DIAGNOSTIC = DEFAULT_PARENT.parent.parent / "http400-diagnostic.json"
CLASSIFIER = ROOT / "src/routerbench/agent_litellm.py"
CLASSIFIER_KEY = "routerbench/agent_litellm.py"
ERROR_CODE = "output_parse_failed"
MODELS = ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")


def select_unattempted(manifest, report):
    """Exclude a whole conversation after any attempt, regardless of outcomes."""
    if (report.get("status") not in {"stopped", "complete"} or not report.get("summary_current")
            or not report.get("summary_present")
            or any(row.get("status") in {"pending", "in_progress"} for row in report["episodes"])):
        raise ValueError("The parent must have a closed current summary and no pending attempt.")
    cases = manifest["cases"]
    if len(cases) != 24 or len({case["id"] for case in cases}) != 24:
        raise ValueError("This revision requires the original 24 development conversations.")
    attempted = {row["episode_id"] for row in report["episodes"] if row["calls_started"] > 0}
    if len(attempted) != 5 or not attempted <= {case["id"] for case in cases}:
        raise ValueError("This declared revision excludes exactly five attempted conversations.")
    selected = [deepcopy(case) for case in cases if case["id"] not in attempted]
    if len(selected) != 19:
        raise ValueError("The declared untouched-by-generation cohort must contain 19 conversations.")
    return selected, sorted(attempted)


def _read_audited(path, report):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report["source_file_sha256"].get(str(path)):
        raise ValueError("An audited parent artifact changed before revision preparation.")
    return json.loads(raw)


def prepare_revision(parent, diagnostic, directory):
    parent, diagnostic, directory = Path(parent).resolve(), Path(diagnostic).resolve(), Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Use a new experiment directory; never overwrite or resume this revision.")
    report = audit_bfcl(parent)
    manifest = _read_audited(parent / "manifest.json", report)
    cases, excluded = select_unattempted(manifest, report)
    config = deepcopy(manifest["configuration"])
    if (config.get("evaluation_split") != "development" or config.get("independent_validation") is not False
            or config.get("execution_mode") != "external" or tuple(config["model_catalog"]) != MODELS
            or manifest["policy_sequence"] != run_bfcl.POLICIES or config.get("max_calls_per_turn") != 21):
        raise ValueError("Parent catalog, controls and development protocol must match the declared revision.")
    spend = config.get("spending", {})
    if spend.get("account_is_free_tier") is not True or spend.get("max_provider_spend_usd") != 0:
        raise ValueError("The confirmed Free account and zero provider budget must be preserved.")
    evidence_raw = diagnostic.read_bytes()
    evidence = json.loads(evidence_raw)
    if (evidence.get("benchmark_freeze_sha256") != manifest["freeze_sha256"]
            or evidence.get("outcome") != "error"
            or not any(row.get("code") == ERROR_CODE for row in evidence.get("provider_error_facts", []))):
        raise ValueError("The diagnostic must link this parent freeze to the exact output_parse_failed code.")
    before = manifest["source_files_sha256"][CLASSIFIER_KEY]
    after = hashlib.sha256(CLASSIFIER.read_bytes()).hexdigest()
    if before == after:
        raise ValueError("The new classifier source must differ from the frozen parent source.")
    revision = {"kind": "new_development_cohort_after_provider_error_classification_fix",
        "parent_run": str(parent), "parent_freeze_sha256": manifest["freeze_sha256"],
        "parent_artifact_sha256": deepcopy(report["source_file_sha256"]),
        "selection_rule": "exclude each whole conversation with any parent call_started, independent of outcome; retain parent order",
        "parent_conversations": 24, "excluded_attempted_ids": excluded,
        "selected_ids": [case["id"] for case in cases], "selected_conversations": 19,
        "classification_change": {"source": "src/" + CLASSIFIER_KEY, "parent_source_key": CLASSIFIER_KEY,
            "before_sha256": before, "after_sha256": after,
            "diagnosed_provider_code": ERROR_CODE, "previous_category": "http", "new_category": "model_action"},
        "diagnostic": {"path": str(diagnostic), "sha256": hashlib.sha256(evidence_raw).hexdigest(),
                       "scope": evidence.get("scope"), "provider_calls": evidence.get("provider_calls")},
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cohort_scope": "no generation attempts in parent; previously prepared development, not untouched holdout",
        "causal_evidence_scope": "separate reproduction of the same request exposed output_parse_failed; the original parent event did not retain that code; its censorship is preserved",
        "original_parent_error_code_observed": False, "original_parent_outcomes_reclassified": False,
        "pool_with_parent": False, "independent_validation": False}
    for stale in ("preflight_plan_sha256", "generation_manifest", "local_helper_source"):
        config.pop(stale, None)
    maximum = sum(len(case["entry"]["question"]) for case in cases) * config["max_calls_per_turn"] * len(run_bfcl.POLICIES)
    config.update(run_id="bfcl-development-error-classification-revision-v1", revision_provenance=revision,
        max_total_generation_calls=maximum, session_stop_on_infrastructure=True,
        session_limits={"max_tokens_per_model": 200000, "max_provider_calls": 250},
        scope="New 19-conversation development revision; report separately from parent; no independent validation.")
    run_bfcl._check_no_inline_secrets(config)
    base = run_bfcl.build_generator(config)
    paced = BFCLPacedGenerator(base, **config["pacing"], max_output_tokens=config["generation_options"]["max_output_tokens"],
                               show_progress=config.get("console_progress") is True)
    generate = BFCLSessionGenerator(paced, **config["session_limits"])
    preflight = run_bfcl.preflight(cases, config, base)
    if len(preflight["eligible"]) != len(cases) or preflight["excluded"]:
        raise ValueError("Every selected conversation must pass initial preflight; do not silently change this cohort.")
    config["generation_manifest"] = generate.manifest()
    plan = {"schema_version": 1, "configuration": config, "revision": revision, "preflight": preflight,
            "maximum_generation_calls": maximum, "provider_calls": 0, "mode": "prepare"}
    plan_sha = object_hash(plan)
    run_bfcl._write_frozen(directory / "plan.json", {"payload": plan, "payload_sha256": plan_sha})
    config["revision_plan_sha256"] = plan_sha
    run_bfcl._write_frozen(directory / "config.json", config)
    policies = [kind(config["model_catalog"]) for kind in (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade)]
    frozen = prepare_bfcl_episodes(cases, policies, directory / "run", config, generate=generate)
    result = {"directory": str(directory), "mode": "prepare", "selected_conversations": len(cases),
              "excluded_attempted_ids": excluded, "maximum_generation_calls": maximum,
              "session_limits": config["session_limits"], "plan_sha256": plan_sha,
              "freeze_sha256": frozen["freeze_sha256"], "provider_calls": 0, "pool_with_parent": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
    return {"result": result, "cases": cases, "policies": policies, "generate": generate,
            "config": config, "directory": directory}


def execute_revision(prepared):
    previous = os.environ.get("GROQ_API_KEY")
    try:
        key = previous
        if not key:
            if not sys.stdin.isatty():
                raise RuntimeError("Use a terminal for hidden key entry or the authorized GROQ_API_KEY environment variable.")
            key = getpass.getpass("Groq API key (oculta; solo en memoria): ")
        if not key or any(character.isspace() for character in key):
            raise ValueError("The API key must be nonempty and contain no whitespace.")
        os.environ["GROQ_API_KEY"] = key
        del key
        return run_bfcl_episodes(prepared["cases"], prepared["policies"], prepared["generate"],
            prepared["directory"] / "run", prepared["config"])
    finally:
        if previous is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    directory = args.output or ROOT / "outputs/bfcl-comparison" / ("revision-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    prepared = prepare_revision(args.parent_run, args.diagnostic, directory)
    if args.execute:
        executed = execute_revision(prepared)
        result = {**prepared["result"], "mode": "execute", "generation_calls": executed["generation_calls"],
                  "summary": str(prepared["directory"] / "run/summary.json")}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
        return result
    return prepared["result"]


if __name__ == "__main__":
    main()
