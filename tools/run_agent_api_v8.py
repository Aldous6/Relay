"""Prepare an explicit API-agent development plan; generation requires --execute.

Preparation never imports LiteLLM or reads credentials. The distributed example
is deliberately incomplete and only displays a plan. No provider or contemporary
model is silently selected, and no account is created or upgraded by this CLI.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.agent import AlwaysStrong, AlwaysWeak
from routerbench.agent_evaluation import prepare_episodes, run_episodes
from routerbench.agent_tasks import build_tasks, validate_task_splits


def _read(path):
    def reject(value):
        raise ValueError("Configuration numbers must be finite.")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _check_no_inline_secrets(value):
    if isinstance(value, dict):
        if any(key.casefold() in {"api_key", "authorization", "password", "access_token", "secret"} for key in value):
            raise ValueError("Keep credentials out of configuration; use api_key_env.")
        for child in value.values():
            _check_no_inline_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _check_no_inline_secrets(child)


def _tasks(config):
    path = config.get("task_splits_path")
    if path is None:
        if config.get("evaluation_split") != "development":
            raise ValueError("Synthetic defaults are development only; supply task_splits_path for a reserved cohort.")
        if config.get("require_unseen_families"):
            raise ValueError("Synthetic defaults do not reserve unseen families.")
        tasks = build_tasks(config["seed"], config["per_family"])
        config["task_scope"] = "synthetic known-family development fixture, not a representative holdout"
        return tasks
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    splits = _read(path)
    if config.get("evaluation_split") == "evaluation" and ("evaluation" not in splits or len(splits) < 2):
        raise ValueError("Supply reserved evaluation and development partitions together for the overlap audit.")
    audit = validate_task_splits(splits, require_unseen_families=config.get("require_unseen_families", False))
    if config["evaluation_split"] not in splits:
        raise ValueError("The chosen partition is absent from task_splits_path.")
    config["task_splits_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config["task_split_audit"] = audit
    config["task_scope"] = "supplied whole episodes; overlap audit does not certify independence"
    return splits[config["evaluation_split"]]


def _catalog_ready(config):
    if config.get("catalog_status") != "selected":
        return False
    names = config.get("model_catalog", [])
    if not names or any(not isinstance(name, str) or "SELECT_" in name.upper() for name in names):
        return False
    metadata = config.get("models_metadata", {})
    return set(metadata) == set(names) and not any("SELECT_" in json.dumps(item).upper() for item in metadata.values())


def _spending_plan(config, maximum_calls):
    plan = config.get("spending", {})
    reasons = []
    budget = plan.get("max_provider_spend_usd")
    if not _number(budget):
        reasons.append("declare_nonnegative_provider_budget")
    cap = plan.get("max_total_generation_calls")
    if type(cap) is not int or cap < maximum_calls:
        reasons.append("generation_call_cap_below_frozen_maximum")
    bound = None
    if budget == 0:
        if plan.get("account_is_free_tier") is not True or not isinstance(plan.get("account_tier_source"), str) or not plan["account_tier_source"].strip():
            reasons.append("confirm_free_account_tier_and_source")
    elif _number(budget):
        per_call = []
        for model in config["model_catalog"]:
            tariff = config.get("pricing", {}).get(model, {})
            rates = [tariff.get(key) for key in ("input_per_million", "output_per_million", "cached_input_per_million")]
            context = config["models_metadata"][model].get("context_window")
            if (not all(_number(rate) for rate in rates) or type(context) is not int or context < 1
                    or not tariff.get("source") or not tariff.get("as_of")):
                reasons.append("complete_exact_model_tariffs_for_paid_execution")
                break
            per_call.append((context * max(rates[0], rates[2]) + config["generation_options"]["max_output_tokens"] * rates[1]) / 1_000_000)
        else:
            bound = maximum_calls * max(per_call)
            if bound > budget:
                reasons.append("conservative_provider_cost_bound_exceeds_budget")
    return {"maximum_generation_calls": maximum_calls, "provider_budget_usd": budget,
            "conservative_provider_cost_bound_usd": bound, "execution_blockers": sorted(set(reasons)),
            "budget_scope": "provider tokens only; account declarations and frozen tariff bounds are not a billing guarantee"}


def _policies(config, tasks=None):
    from relay_router.v8 import NReactiveCascade
    strong, weak = config["models"]["strong"], config["models"]["weak"]
    catalog = tuple(config["model_catalog"])
    factories = {"always_strong": lambda: AlwaysStrong(strong, weak),
                 "always_weak": lambda: AlwaysWeak(strong, weak),
                 "cheap_then_escalate": lambda: NReactiveCascade(catalog)}
    requested = config["policies"]
    if (not isinstance(requested, list) or not requested or len(requested) != len(set(requested))
            or "always_strong" not in requested):
        raise ValueError("Declare unique policies including always_strong.")
    if "relay_v8_predictive" in requested:
        from routerbench.agent_learning import StateSuccessPredictor
        from relay_router.v8 import ExpectedCostPolicy
        settings = config.get("predictive_policy")
        if not isinstance(settings, dict) or not settings.get("artifact_sha256"):
            raise ValueError("Predictive control requires its trained artifact and an external SHA-256.")
        artifact = Path(settings["artifact_path"])
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        predictor = StateSuccessPredictor.load(artifact, expected_sha256=settings["artifact_sha256"])
        if tuple(predictor.models) != catalog:
            raise ValueError("Predictor identities must equal the frozen catalog in order.")
        factories["relay_v8_predictive"] = lambda: ExpectedCostPolicy(
            predictor, settings["expected_call_costs"], settings["expected_escalation_costs"],
            min_probability=settings["min_probability"], operating_point_source=settings["operating_point_source"])
    if "relay_v8_trajectory" in requested:
        from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor
        from relay_router.trajectory import TrajectoryValuePolicy
        settings = config.get("trajectory_policy")
        if not isinstance(settings, dict) or not settings.get("artifact_sha256"):
            raise ValueError("Trajectory control requires a trained artifact and external SHA-256.")
        artifact = Path(settings["artifact_path"])
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        trajectory_predictor = TrajectorySuccessPredictor.load(artifact, expected_sha256=settings["artifact_sha256"])
        if trajectory_predictor.models != catalog:
            raise ValueError("Trajectory predictor identities must equal the frozen catalog in order.")
        if tasks is None:
            raise ValueError("Trajectory routing requires tasks for the fit/calibration overlap audit.")
        trajectory_predictor.audit_task_groups(tasks, evaluation_split=(
            "development" if config["evaluation_split"] == "development" else "holdout"))
        if trajectory_predictor.manifest()["assignment"] != "paired_state":
            raise ValueError("External execution requires paired live observations, not synthetic training fixtures.")
        factories["relay_v8_trajectory"] = lambda: TrajectoryValuePolicy(
            trajectory_predictor, min_probability=settings["min_probability"],
            operating_point_source=settings["operating_point_source"])
    if set(requested) - set(factories):
        raise ValueError("An unknown policy was requested.")
    return [factories[name]() for name in requested]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_api_v8.example.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/agent-api-v8-development-v1")
    parser.add_argument("--only-strong", action="store_true", help="Prepare the phase-0 fixed-strong quality control only.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = deepcopy(_read(args.config))
    _check_no_inline_secrets(config)
    if config.get("schema_version") != 2 or config.get("execution_mode") != "external" or config.get("independent_validation") is not False:
        raise ValueError("This CLI requires the explicit exploratory API schema v2.")
    if args.only_strong:
        config["policies"], config["primary_policy"] = ["always_strong"], "always_strong"
    tasks = _tasks(config)
    if config.get("primary_policy") not in config["policies"]:
        raise ValueError("The frozen primary policy must be included.")
    maximum_calls = sum(min(task["max_steps"], config["max_calls"]) for task in tasks) * len(config["policies"])
    plan = _spending_plan(config, maximum_calls)
    if not _catalog_ready(config):
        plan["execution_blockers"].append("select_exact_catalog_and_verified_model_metadata")
        plan.update(mode="template_plan_only", prepared_tasks=len(tasks), frozen=False, generation_calls=0,
                    policies=config["policies"], predictive_control_available="relay_v8_predictive" in config["policies"])
        if args.execute:
            raise ValueError("Cannot execute an incomplete catalog template.")
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return plan
    if args.execute and plan["execution_blockers"]:
        raise ValueError("Execution plan is incomplete: " + ", ".join(plan["execution_blockers"]))
    from routerbench.agent_litellm import LiteLLMEpisodeGenerator
    generate = LiteLLMEpisodeGenerator(config["models_metadata"], config["generation_options"])
    if "pacing" in config:
        from routerbench.agent_pacing import PacedEpisodeGenerator
        generate = PacedEpisodeGenerator(generate, **config["pacing"],
                                         max_output_tokens=config["generation_options"]["max_output_tokens"],
                                         show_progress=config.get("console_progress") is True)
    config["generation_manifest"] = generate.manifest()
    config["execution_plan"] = plan
    config["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policies = _policies(config, tasks)
    frozen = prepare_episodes(tasks, policies, args.output, config, generate=generate)
    result = {**plan, "prepared_tasks": len(tasks), "policies": [policy.name for policy in policies],
              "freeze_sha256": frozen["freeze_sha256"], "frozen": True,
              "mode": "execute" if args.execute else "prepare_only", "generation_calls": 0,
              "stage": "prepared_before_generation"}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.execute:
        summary = run_episodes(tasks, policies, generate, args.output, config)
        result = {"tasks": summary["n_tasks"], "episodes": summary["n_episodes"],
                  "summary": str(args.output / "summary.json"), "independent_validation": False,
                  "stage": "finished", "results": {
                      name: {"successes": row["successes"], "tasks": row["n_episodes"],
                             "recorded_attempts": row["calls"], "failures": row.get("failure_diagnostics", {})}
                      for name, row in summary.get("metrics", {}).get("policies", {}).items()}}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
