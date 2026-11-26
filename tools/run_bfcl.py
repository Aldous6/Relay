"""Preflight all 24 BFCL development groups; generation requires --execute.

UTF-8 bounds and optional local GPT-OSS estimates are distinct accounting modes.
No tools/history are trimmed. Neither mode measures the provider's template.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.v8 import model_order
from routerbench.agent_evaluation import object_hash
from routerbench.bfcl_data import COMMIT, read_snapshot, register_exposure_record, load_selected_cases
from routerbench.bfcl_environment import BFCLRuntime
from routerbench.bfcl_harness import (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade,
    build_initial_request, prepare_bfcl_episodes, run_bfcl_episodes)
from routerbench.bfcl_litellm import BFCLLiteLLMGenerator, BFCLPacedGenerator
from run_agent_api_v8 import _check_no_inline_secrets
from train_agent_v8 import read_json

POLICIES = ["always_strong", "always_weak", "cheap_then_escalate"]


def build_generator(config):
    settings = config.get("quota_input_accounting", {"method": "utf8_bytes_bound"})
    method = settings.get("method")
    estimator = None
    if method == "gptoss_local_token_estimate":
        from routerbench.bfcl_tokens import GPTOSSTokenEstimator
        if config["generation_options"].get("reasoning_effort") != settings["reasoning_effort"]:
            raise ValueError("Freeze the same reasoning effort for generation and the published template estimate.")
        estimator = GPTOSSTokenEstimator(margin_tokens=settings["margin_tokens"],
            template_date=settings["template_date"], reasoning_effort=settings["reasoning_effort"])
    elif method != "utf8_bytes_bound":
        raise ValueError("Unknown quota input accounting method.")
    return BFCLLiteLLMGenerator(config["models_metadata"], config["generation_options"], input_estimator=estimator)


def _absolute(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _config(path):
    value = read_json(path)
    _check_no_inline_secrets(value)
    if (type(value) is not dict or value.get("schema_version") != 1 or value.get("upstream_commit") != COMMIT
            or value.get("execution_mode") != "external" or value.get("evaluation_split") != "development"
            or value.get("independent_validation") is not False or value.get("policies") != POLICIES):
        raise ValueError("Use the explicit development-only BFCL config and all three fixed controls.")
    catalog = model_order(tuple(value.get("model_catalog", ())))
    if len(catalog) != 2 or set(value.get("models_metadata", {})) != set(catalog):
        raise ValueError("The binary experiment requires the same exact two models for every control.")
    for name in ("max_calls_per_turn", "max_total_generation_calls"):
        if type(value.get(name)) is not int or value[name] < 1:
            raise ValueError(f"{name} must be an explicitly declared positive integer.")
    value["snapshot_path"] = str(_absolute(value["snapshot_path"]))
    value["exposure_record_path"] = str(_absolute(value["exposure_record_path"]))
    return deepcopy(value)


def preflight(cases, config, base, *, runtime_factory=BFCLRuntime):
    """Observe only the first public request; no private checker or model call."""
    options, pacing = config["generation_options"], config["pacing"]
    bounds = [pacing["max_request_budget"], pacing["tokens_per_minute"]]
    bounds.extend(config["models_metadata"][model]["context_window"] for model in config["model_catalog"])
    quota_bound = min(bounds)
    results = []
    for case in cases:
        item = {"id": case["id"], "source_group": case["source_group"],
                "turns": len(case["entry"]["question"]), "quota_bound": quota_bound,
                "output_token_reservation": options["max_output_tokens"], "template_reserve": 512}
        try:
            request = build_initial_request(case, config, runtime_factory=runtime_factory)
            incoming = base.request_input_bytes(request)
            estimate = base.request_input_estimate(request)
            bound = estimate["input_budget_units"] + options["max_output_tokens"] + 512
            verified = bound <= quota_bound
            local_estimate = estimate["method"] == "gptoss_local_token_estimate"
            item.update(initial_input_utf8_bytes=incoming,
                        conservative_request_bound=None if local_estimate else bound,
                        input_accounting=estimate, request_budget_estimate=bound,
                        eligible=verified, status=("eligible_under_local_estimate" if local_estimate
                            else "eligible_under_conservative_bound") if verified else
                            ("unverified_under_local_estimate" if local_estimate else "unverified_under_bound"))
        except (ImportError, ValueError, TypeError, RuntimeError) as error:
            item.update(initial_input_utf8_bytes=None, conservative_request_bound=None, eligible=False,
                        status="runtime_unavailable", error_type=type(error).__name__)
        results.append(item)
    return {"denominator": len(cases), "eligible": [row for row in results if row["eligible"]],
            "excluded": [row for row in results if not row["eligible"]],
            "actual_token_counts_measured": False, "initial_request_only": True,
            "no_context_truncation": True, "provider_calls": 0,
            "scope": "Initial public request under the declared input accounting method + reserved output + 512; no server template verification, quality estimate or proof of actual token excess."}


def select_smoke_ids(eligible, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be a positive integer.")
    ordered = sorted(eligible, key=lambda row: (row["initial_input_utf8_bytes"],
        hashlib.sha256(f"{COMMIT}:{row['id']}".encode()).hexdigest(), row["id"]))
    return [row["id"] for row in (ordered if limit is None else ordered[:limit])]


def _blockers(config, selected_rows):
    blockers = []
    spend = config.get("spending", {})
    if type(spend.get("max_provider_spend_usd")) not in (int, float) or spend["max_provider_spend_usd"] != 0:
        blockers.append("explicit_zero_provider_budget_required")
    if spend.get("account_is_free_tier") is not True or type(spend.get("account_tier_source")) is not str or not spend["account_tier_source"].strip():
        blockers.append("free_account_declaration_and_source_required")
    maximum = sum(row["turns"] for row in selected_rows) * config["max_calls_per_turn"] * len(POLICIES)
    if maximum > config["max_total_generation_calls"]:
        blockers.append("declared_call_cap_below_selected_episode_bound")
    if not selected_rows:
        blockers.append("no_development_case_verified_under_bound")
    return blockers, maximum


def _write_frozen(path, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError("Existing plan differs; choose a new output directory.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bfcl_groq_free.example.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Deterministic smoke subset among eligible development groups.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    config = _config(args.config)
    register_exposure_record(config["snapshot_path"], config["exposure_record_path"])
    snapshot = read_snapshot(config["snapshot_path"])
    cases = load_selected_cases(config["snapshot_path"], include_reference=False)
    if len(cases) != 24 or {case["id"] for case in cases} != set(snapshot["selection"]["development"]):
        raise ValueError("Preflight must retain the full denominator of 24 frozen development groups.")
    base = build_generator(config)
    generate = BFCLPacedGenerator(base, **config["pacing"],
        max_output_tokens=config["generation_options"]["max_output_tokens"], show_progress=config.get("console_progress") is True)
    if "session_limits" in config:
        from routerbench.bfcl_session import BFCLSessionGenerator
        generate = BFCLSessionGenerator(generate, **config["session_limits"])
    report = preflight(cases, config, base, runtime_factory=BFCLRuntime)
    selected_ids = select_smoke_ids(report["eligible"], args.limit)
    selected_rows = [row for row in report["eligible"] if row["id"] in selected_ids]
    blockers, maximum = _blockers(config, selected_rows)
    config.update(generation_manifest=generate.manifest(), snapshot_manifest_sha256=snapshot["manifest_sha256"],
        selection_sha256=snapshot["manifest"]["selection_sha256"],
        exposure_ledger_sha256=snapshot["exposure_ledger_sha256"], exposure_ledger=deepcopy(snapshot["exposure_ledger"]))
    plan = {"schema_version": 1, "configuration": config, "preflight": report,
        "selected_ids": selected_ids, "requested_limit": args.limit,
        "selection_rule": "eligible only; initial public request UTF-8 bytes ascending, then sha256(commit:id); no outcomes",
        "maximum_generation_calls": maximum, "execution_blockers": blockers,
        "model_catalog": config["model_catalog"], "policies": POLICIES,
        "upstream_commit": COMMIT, "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "selection_sha256": config["selection_sha256"], "exposure_ledger_sha256": snapshot["exposure_ledger_sha256"],
        "exposure_ledger": deepcopy(snapshot["exposure_ledger"]),
        "source_files_sha256": {str(path.relative_to(ROOT).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT / "src/routerbench/bfcl_data.py", ROOT / "src/routerbench/bfcl_harness.py",
                         ROOT / "src/routerbench/bfcl_environment.py", ROOT / "src/routerbench/bfcl_litellm.py",
                         ROOT / "src/routerbench/bfcl_tokens.py", ROOT / "src/routerbench/bfcl_session.py",
                         ROOT / "configs/bfcl_tokenizer_gptoss_v1.json")},
        "provider_calls": 0, "independent_validation": False,
        "scope": "Exploratory BFCL adaptation; initial feasibility only, no benchmark quality, routing advantage or savings claim."}
    plan_sha = object_hash(plan)
    output = args.output or ROOT / "outputs/bfcl-preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output).resolve()
    _write_frozen(output / "plan.json", {"payload": plan, "payload_sha256": plan_sha})
    _write_frozen(output / "preflight.json", report)
    result = {"output": str(output), "plan_sha256": plan_sha, "preflight_denominator": len(cases),
        "eligible_groups": len(report["eligible"]), "excluded_groups": len(report["excluded"]),
        "selected_groups": len(selected_ids), "selected_ids": selected_ids,
        "maximum_generation_calls": maximum, "execution_blockers": blockers, "generation_calls": 0,
        "exposed_groups": len(snapshot["exposure_ledger"]["exposed_ids"]),
        "blocked_evaluation_groups": len(snapshot["exposure_ledger"]["blocked_evaluation_ids"]),
        "mode": "prepare", "independent_validation": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
    if args.execute and blockers:
        raise ValueError("Execution blocked: " + ", ".join(blockers))
    if not selected_ids or "declared_call_cap_below_selected_episode_bound" in blockers:
        return result
    selected = load_selected_cases(config["snapshot_path"], selected_ids, include_reference=True)
    config["preflight_plan_sha256"] = plan_sha
    policies = [BFCLAlwaysStrong(config["model_catalog"]), BFCLAlwaysWeak(config["model_catalog"]), BFCLReactiveCascade(config["model_catalog"])]
    frozen = prepare_bfcl_episodes(selected, policies, output / "run", config, generate=generate, runtime_factory=BFCLRuntime)
    result["freeze_sha256"] = frozen["freeze_sha256"]
    if args.execute:
        execution = run_bfcl_episodes(selected, policies, generate, output / "run", config, runtime_factory=BFCLRuntime)
        result.update(mode="execute", generation_calls=execution["generation_calls"], summary=str(output / "run/summary.json"))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
