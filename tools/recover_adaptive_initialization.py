"""Recover only audited, zero-dispatch host initialization failures separately.

The complete primary cohort remains immutable. Eligibility depends on a
PermissionError and zero logical/HTTP calls, never a policy decision, model
answer or acceptance. Every eligible original episode gets at most one new
episode, recorded in a separate directory under the same frozen task contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_adaptive_pilot as original
import continue_adaptive_pilot as continuation

DEFAULT_RUN = ROOT / "outputs/harness-adaptive/v1/run"
DEFAULT_SUITE = ROOT / "data/imported/harness-adaptive-v1/fresh_suite.json"
DEFAULT_POLICY = ROOT / "outputs/harness-adaptive/v1/policy-auto.json"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ORIGINAL_INPUTS = (
    "frozen_routes.json", "run_manifest.json", "observations.jsonl", "http_attempts.jsonl",
    "comparison.json", "validation.json", "execution_amendment.json",
)
RECOVERY_SOURCES = (
    "tools/recover_adaptive_initialization.py", "tools/continue_adaptive_pilot.py",
    "src/relay_router/integrations/openhands_io.py",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in RECOVERY_SOURCES}


def _zero(value):
    return type(value) is int and value == 0


def eligible_primary_episodes(observations, events, run):
    """Derive eligibility without inspecting routes, answers or acceptance."""
    starts = [_read(path) for path in (Path(run) / "attempt_starts").glob("*.json")]
    eligible = []
    for row in observations:
        if (row.get("error_type") != "PermissionError" or not _zero(row.get("logical_calls"))
                or not _zero(row.get("http_attempts"))):
            continue
        episode_id = original._episode_id(row["task_id"], row["model"])
        dispatches = [event for event in events if event["episode_id"] == episode_id]
        dispatch_starts = [start for start in starts if start["episode_id"] == episode_id]
        if dispatches or dispatch_starts:
            raise ValueError("A claimed zero-dispatch failure has a recorded HTTP attempt")
        eligible.append({"episode_id": episode_id, "task_id": row["task_id"], "group_id": row["group_id"],
                         "model": row["model"], "original_error_type": "PermissionError",
                         "original_logical_calls": 0, "original_http_attempts": 0,
                         "original_http_start_records": 0,
                         "zero_dispatch_tariff_usd": 0.,
                         "evidence_scope": "consistent audited primary episode and HTTP write-ahead/completion logs"})
    return sorted(eligible, key=lambda row: row["episode_id"])


def token_budget(events):
    budget = {model: {"observed_tokens": 0, "unknown_usage_attempts": 0, "unknown_admission_reserved_tokens": 0}
              for model in original.MODELS}
    for event in events:
        item = budget[event["model"]]
        tokens = event.get("total_tokens")
        if tokens is None:
            reserve = event.get("admission_reserved_tokens")
            if type(reserve) is not int or reserve <= 0:
                raise ValueError("Unknown original/recovery usage needs its nonzero admission reservation")
            item["unknown_usage_attempts"] += 1
            item["unknown_admission_reserved_tokens"] += reserve
        elif type(tokens) is not int or tokens < 0:
            raise ValueError("Recorded token totals must be nonnegative integers or unknown")
        else:
            item["observed_tokens"] += tokens
    return budget


def load_original(run, suite_path, policy_path):
    """Verify existing primary artifacts; never create or modify primary data."""
    run = Path(run).resolve()
    if not all((run / name).is_file() for name in ORIGINAL_INPUTS):
        raise ValueError("Recovery requires the complete primary run, comparison and recorded IO amendment")
    suite, policy, frozen = original.prepare_run(suite_path, policy_path, run)
    observations = original.read_observations(suite, run)
    events = original.reconcile_http(suite, run, observations)
    if len(observations) != len(suite["tasks"]) * len(suite["models"]):
        raise ValueError("All primary episodes must finish before any isolated recovery")
    # This existing amendment is checked, never created retroactively. It binds
    # the scoped file-write repair and its preserved original observation prefix.
    amendment = continuation.prepare_amendment(suite_path, policy_path, run)
    inputs = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in ORIGINAL_INPUTS}
    for name in ("validation_records.json", "io_retry_events.jsonl"):
        inputs[name] = hashlib.sha256((run / name).read_bytes()).hexdigest() if (run / name).is_file() else None
    comparison = _read(run / "comparison.json")
    body = dict(comparison)
    seal = body.pop("report_id", None)
    if seal != original._digest(body) or comparison.get("freeze_id") != frozen["freeze_id"]:
        raise ValueError("Primary comparison integrity or freeze binding mismatch")
    if comparison.get("evaluation") != _read(run / "validation.json"):
        raise ValueError("Primary validation differs from its frozen comparison")
    return {"run": run, "suite": suite, "policy": policy, "frozen": frozen,
            "observations": observations, "events": events, "amendment": amendment,
            "eligible": eligible_primary_episodes(observations, events, run),
            "original_input_sha256": inputs,
            "suite_sha256": hashlib.sha256(Path(suite_path).read_bytes()).hexdigest(),
            "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()}


def _manifest_contract(state):
    return {
        "schema_version": 1, "kind": "isolated_zero_dispatch_initialization_recovery",
        "freeze_id": state["frozen"]["freeze_id"], "artifact_id": state["policy"]["artifact_id"],
        "suite_sha256": state["suite_sha256"], "policy_sha256": state["policy_sha256"],
        "original_input_sha256": state["original_input_sha256"],
        "original_execution_amendment_id": state["amendment"]["amendment_id"],
        "source_sha256": source_hashes(),
        "eligible_episode_ids": [row["episode_id"] for row in state["eligible"]],
        "eligible": state["eligible"], "original_completed_episodes": len(state["observations"]),
        "original_http_attempts": len(state["events"]), "original_token_budget": token_budget(state["events"]),
        "eligibility_rule": "ALL primary PermissionError episodes with audited logical_calls==0 and HTTP starts/completions==0",
        "reason": "separate infrastructure recovery after bounded local IO repair; no LLM answer existed in eligible original episodes",
        "execution": {**original.FIXED_LIMITS, "max_recovery_episodes_per_original_episode": 1,
                      "api_base": GROQ_BASE_URL,
                      "evaluation_id": state["suite"]["evaluation_id"] + "-recovery",
                      "same_prompts_tools_verifiers_models_and_policy": True,
                      "no_original_observations_replaced": True,
                      "IO_retry": continuation._write_contract(),
                      "io_retry_event_binding": "amendment_id field equals this recovery_id"},
        "combined_budget_scope": "primary acquisition plus this isolated recovery; other diagnostics remain separately reported",
        "recovery_result_scope": "secondary sensitivity analysis; cannot replace or approve the primary result",
    }


def verify_recovery_manifest(manifest, state):
    """Read-only integrity and exact-primary binding check for report tools."""
    body = dict(manifest)
    seal = body.pop("recovery_id", None)
    if seal != original._digest(body):
        raise ValueError("Recovery manifest integrity mismatch")
    expected = _manifest_contract(state)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery source, primary data, eligibility or execution contract changed")
    when = datetime.fromisoformat(manifest["created_at_utc"])
    if when.tzinfo is None:
        raise ValueError("Recovery manifest timestamp must contain a timezone")
    return manifest


def prepare_recovery(state, output):
    output = Path(output).resolve()
    if output == state["run"] or output.is_relative_to(state["run"] / "episodes"):
        raise ValueError("Recovery must use its own directory, outside original episode artifacts")
    path = output / "recovery_manifest.json"
    if path.exists():
        return verify_recovery_manifest(_read(path), state)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to start a recovery in an already used directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**_manifest_contract(state), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest["recovery_id"] = original._digest(manifest)
    original.write(path, manifest)
    return manifest


def recovery_suite(state):
    tasks = {row["task_id"] for row in state["eligible"]}
    return {**state["suite"], "tasks": [task for task in state["suite"]["tasks"] if task["task_id"] in tasks],
            "evaluation_id": state["suite"]["evaluation_id"] + "-recovery",
            "max_total_http_attempts": state["suite"]["max_total_http_attempts"] - len(state["events"])}


def read_recovery(state, manifest, output):
    suite = recovery_suite(state)
    rows = original.read_observations(suite, output)
    events = original.reconcile_http(suite, output, rows)
    eligible = {(row["task_id"], row["model"]): row for row in state["eligible"]}
    for row in rows:
        pair = row["task_id"], row["model"]
        if pair not in eligible:
            raise ValueError("A recovery observation is outside the independently derived eligibility set")
        if (row.get("recovery_id") != manifest["recovery_id"]
                or row.get("original_episode_id") != eligible[pair]["episode_id"]
                or not _zero(row.get("original_http_attempts_verified"))
                or row.get("original_llm_tariff_zero_dispatch_usd") != 0.):
            raise ValueError("Recovery observation lost its zero-dispatch original evidence")
        start = _read(output / "episode_starts" / (eligible[pair]["episode_id"] + ".json"))
        if start.get("recovery_id") != manifest["recovery_id"] or start.get("freeze_id") != manifest["freeze_id"]:
            raise ValueError("Recovery episode was not bound to its write-ahead manifest")
    combined = state["events"] + events
    if len(combined) > state["suite"]["max_total_http_attempts"]:
        raise ValueError("Combined primary/recovery HTTP cap exceeded")
    budget = token_budget(combined)
    limit = state["suite"]["max_observed_tokens_per_model"]
    if any(row["observed_tokens"] + row["unknown_admission_reserved_tokens"] > limit for row in budget.values()):
        raise ValueError("Combined primary/recovery observed usage and admission reserves exceed the token cap")
    return rows, events, budget


class RecoveryTransport(original.BoundedAuditedTransport):
    """Count recovery attempts locally and enforce the combined acquisition cap."""
    def __init__(self, provider, suite, output, original_events):
        super().__init__(provider, suite, output)
        budget = token_budget(original_events)
        now = datetime.now(timezone.utc)
        for model, row in budget.items():
            self.tokens[model] += row["observed_tokens"]
            self.unknown_reserved[model] += row["unknown_admission_reserved_tokens"]
        # Original attempts still within the 63s quota window are conservatively
        # aged from restart. Recovery attempts are already restored by the base.
        for event in original_events:
            age = (now - datetime.fromisoformat(event["started_at_utc"])).total_seconds()
            if age < 63:
                self.windows[event["model"]].append((time.monotonic(), event.get("total_tokens") or 7900))

    def __call__(self, *args, **kwargs):
        for name in ("api_base", "base_url"):
            base = kwargs.get(name)
            if base is not None and (type(base) is not str or base.rstrip("/") != GROQ_BASE_URL):
                raise ValueError("Recovery dispatch is restricted to the fixed Groq API endpoint")
        kwargs["api_base"] = GROQ_BASE_URL
        return super().__call__(*args, **kwargs)


def save_status(state, manifest, output):
    rows, events, budget = read_recovery(state, manifest, output)
    unknown = sum(event["cost_usd"] is None for event in events)
    result = {
        "schema_version": 1, "recovery_id": manifest["recovery_id"], "freeze_id": manifest["freeze_id"],
        "eligible_count": len(state["eligible"]), "completed_recovery_episodes": len(rows),
        "complete": len(rows) == len(state["eligible"]), "no_op": not state["eligible"],
        "recovery_http_attempts": len(events), "combined_http_attempts": len(state["events"]) + len(events),
        "combined_token_budget": budget, "recovery_unknown_http_costs": unknown,
        "recovery_known_tariff_subtotal_usd": sum(event["cost_usd"] for event in events if event["cost_usd"] is not None),
        "recovery_total_tariff_usd": sum(event["cost_usd"] for event in events) if not unknown else None,
        "primary_observations_unchanged": True, "primary_validation_approved_by_recovery": False,
        "net_monetary_savings_verified": False,
    }
    result["status_id"] = original._digest(result)
    original.write(output / "recovery_status.json", result, exclusive=False)
    return result


def execute_recovery(state, manifest, output, *, key, llm_class, llm_module, versions,
                     episode_runner=original.run_episode):
    output = Path(output).resolve()
    verify_recovery_manifest(manifest, state)
    if _read(output / "recovery_manifest.json") != manifest:
        raise ValueError("Recovery manifest must be persisted before dispatch")
    version_path = output / "recovery_packages.json"
    if version_path.exists():
        if _read(version_path) != versions:
            raise ValueError("Cannot resume recovery with changed SDK packages")
    else:
        original.write(version_path, versions)
    rows, events, _ = read_recovery(state, manifest, output)
    finished = {(row["task_id"], row["model"]) for row in rows}
    suite = recovery_suite(state)
    transport = RecoveryTransport(llm_module.litellm_completion, suite, output, state["events"])
    relay = original.Relay(suite["baseline_model"], log_path=output / "relay_events.jsonl")
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    for eligible in state["eligible"]:
        task_id, model = eligible["task_id"], eligible["model"]
        if (task_id, model) in finished:
            continue
        if len(transport.events) >= suite["max_total_http_attempts"]:
            raise ValueError("Combined acquisition HTTP budget exhausted before recovery dispatch")
        task, episode_id = tasks[task_id], eligible["episode_id"]
        original.write(output / "episode_starts" / (episode_id + ".json"), {
            "episode_id": episode_id, "task_id": task_id, "model": model,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "recovery_id": manifest["recovery_id"], "freeze_id": manifest["freeze_id"],
        })
        session = relay.task(task["workflow"], task_id=task_id, group_id=task["group_id"], split="validation",
                             evaluation_id=suite["evaluation_id"], model_override=model, max_calls=3)
        llm = llm_class(model=model, api_key=key, base_url=GROQ_BASE_URL, num_retries=0, api_mode="chat", timeout=90,
                        temperature=0, max_output_tokens=1024, reasoning_effort="low", log_completions=False,
                        usage_id="agent", caching_prompt=False,
                        capability_overrides={"supports_prompt_cache_key": False, "supports_prompt_cache_retention": False},
                        input_cost_per_token=original.TARIFFS[model]["input_per_million_usd"] / 1e6,
                        output_cost_per_token=original.TARIFFS[model]["output_per_million_usd"] / 1e6)
        transport.episode, transport.active = episode_id, []
        print(f"Recuperacion aislada {len(rows)+1}/{len(state['eligible'])}: {episode_id}", flush=True)
        with patch.object(llm_module, "litellm_completion", transport):
            row = episode_runner(task, llm, session, output / "episodes" / episode_id,
                                 max_calls=3, stop_on_public_pass=True)
        attempts = transport.active
        known = bool(attempts) and all(item["cost_usd"] is not None for item in attempts)
        row.update(cost_usd=sum(item["cost_usd"] for item in attempts) if known else None,
                   cost_complete=known, cost_basis="tariff_estimate" if known else "unknown",
                   accounting_source="all_HTTP_dispatches_audited_including_internal_repairs",
                   cost_scope="LLM_uncached_tariff_scenario_excludes_local_compute",
                   http_attempts=len(attempts), unknown_http_costs=sum(item["cost_usd"] is None for item in attempts),
                   observed_input_tokens=sum(item["input_tokens"] or 0 for item in attempts),
                   observed_output_tokens=sum(item["output_tokens"] or 0 for item in attempts),
                   quota_wait_seconds=sum(item["quota_wait_seconds"] for item in attempts),
                   transport_seconds=sum(item["transport_seconds"] for item in attempts),
                   original_episode_id=episode_id, original_http_attempts_verified=0,
                   original_llm_tariff_zero_dispatch_usd=0., recovery_id=manifest["recovery_id"])
        original.write(output / "episodes" / episode_id / "accounted_episode.json", row)
        with (output / "observations.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        print(json.dumps({field: row[field] for field in ("accepted", "http_attempts", "cost_usd", "error_type")}), flush=True)
        native_tool_failure = bool(attempts) and attempts[-1].get("provider_error_code") == "tool_use_failed"
        if (row["error_type"] or not known) and not native_tool_failure:
            save_status(state, manifest, output)
            print("Fallo de recuperacion conservado; no se repetira este episodio.", flush=True)
            return 2
    save_status(state, manifest, output)
    print("Recuperacion aislada cerrada; el resultado primario permanece intacto.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credential-pipe")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    output = (args.output or args.run / "pre_dispatch_recovery").resolve()
    state = load_original(args.run, args.suite, args.policy)
    manifest = prepare_recovery(state, output)
    rows, _, _ = read_recovery(state, manifest, output)
    if not state["eligible"] or len(rows) == len(state["eligible"]):
        save_status(state, manifest, output)
        print("No hay recuperaciones pendientes; no se solicita credencial ni se llama a modelos.")
        return 0
    if args.prepare_only:
        print("Manifiesto de recuperacion aislada preparado; ninguna llamada ejecutada.")
        return 0
    key = None
    try:
        import litellm
        from openhands.sdk.llm import LLM
        import openhands.sdk.llm.llm as llm_module
        from relay_router.integrations.openhands_io import retry_openhands_writes
        litellm.telemetry = False
        logging.getLogger().setLevel(logging.ERROR)
        versions = {name: importlib.metadata.version(name) for name in ("openhands-sdk", "litellm")}
        original_versions = _read(state["run"] / "run_manifest.json")["packages"]
        if versions != original_versions:
            raise ValueError("Recovery must use the same SDK packages as the primary acquisition")
        key = original._credential(args.credential_pipe)
        callback = continuation.retry_logger(output, {"amendment_id": manifest["recovery_id"]})
        with retry_openhands_writes(output, max_attempts=5, base_delay=.05, on_retry=callback):
            return execute_recovery(state, manifest, output, key=key, llm_class=LLM,
                                    llm_module=llm_module, versions=versions)
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
