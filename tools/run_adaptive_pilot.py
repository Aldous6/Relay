"""Freeze and measure a Relay policy in the existing bounded OpenHands harness.

This runner measures both candidate branches once per fresh task. Its report is
a paired replay of the policy frozen before any dispatch, not an additional run
of every policy. The accounting includes every HTTP attempt, even SDK repairs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

from relay_router import Relay
from relay_router.harness_episode import run_episode, task_prompt
from relay_harness_live import AuditedTransport, write

MODELS = ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")
TARIFFS = {
    MODELS[0]: {"input_per_million_usd": .075, "output_per_million_usd": .30},
    MODELS[1]: {"input_per_million_usd": .15, "output_per_million_usd": .60},
}
FIXED_LIMITS = {
    "max_calls_per_episode": 3, "max_total_http_attempts": 144,
    "max_output_tokens": 1024, "max_observed_tokens_per_model": 180000,
    "temperature": 0, "reasoning_effort": "low", "stop_on_public_pass": True,
}
# Admission estimate, not a measured token count or a billing guarantee. The
# full serialized request is reserved for the daily guard (unlike the shorter
# sliding-window estimate in the unchanged historical transport).
ADMISSION_OVERHEAD_TOKENS = 2048
BERT_REFERENCE_PATH = ROOT / "data/imported/harness-adaptive-v1/bert_reference.json"
SOURCES = (
    "tools/run_adaptive_pilot.py", "tools/relay_harness_live.py",
    "src/relay_router/harness_episode.py", "src/relay_router/adaptive_policy.py",
    "src/relay_router/adaptive_runtime.py", "docs/openhands_adaptation_v1.md",
    "src/relay_router/adaptive_training.py", "tools/adaptive_scoring.py",
    "src/relay_router/workflow.py", "src/relay_router/prompt_selection.py",
    "src/relay_router/integrations/openhands.py",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=True).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def reference_contract():
    path = BERT_REFERENCE_PATH
    if not path.exists():
        return None
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _policy_api():
    # Training dependencies are never imported by this executable.
    from relay_router import adaptive_policy
    return adaptive_policy


def _scorer_for(suite, policy, suite_sha256):
    if policy.get("representation", "structural") != "routellm-bert":
        return None
    reference = _read(BERT_REFERENCE_PATH)
    if reference.get("suite_sha256") != suite_sha256:
        raise ValueError("Cached BERT reference belongs to a different frozen suite")
    from adaptive_scoring import cached_bert_scorer
    return cached_bert_scorer(BERT_REFERENCE_PATH, suite["tasks"])


class BoundedAuditedTransport(AuditedTransport):
    """Add a conservative daily admission guard without changing old pilots.

    Unknown usage consumes its request reservation for admission only. It stays
    unknown in measurements and prices. A provider can exceed an estimate; any
    observed overrun is persisted by the parent transport before this wrapper
    stops the acquisition. This is not a hard provider-side billing limit.
    """

    def __init__(self, original, suite, output):
        super().__init__(original, suite, output)
        self.unknown_reserved = {model: 0 for model in suite["models"]}
        for event in self.events:
            if event.get("total_tokens") is None:
                reserved = event.get("admission_reserved_tokens")
                if type(reserved) is not int or reserved <= 0:
                    raise ValueError("Unknown usage lacks its preserved admission reservation")
                self.unknown_reserved[event["model"]] += reserved
        self._admission_reserve = None

    def save(self, row):
        row["admission_reserved_tokens"] = self._admission_reserve
        super().save(row)

    def __call__(self, *args, **kwargs):
        model = kwargs.get("model")
        if kwargs.get("custom_llm_provider") == "groq" and "groq/" + str(model) in self.windows:
            model = "groq/" + str(model)
        if model not in self.windows:
            raise ValueError("Transport model is outside the frozen catalog")
        request_bytes = len(json.dumps({"messages": kwargs.get("messages"), "tools": kwargs.get("tools")}, default=str).encode())
        reserved = request_bytes + self.suite["max_output_tokens"] + ADMISSION_OVERHEAD_TOKENS
        limit = self.suite["max_observed_tokens_per_model"]
        if self.tokens[model] + self.unknown_reserved[model] + reserved > limit:
            raise ValueError("Daily admission token reserve exhausted before dispatch")
        before = len(self.events)
        self._admission_reserve = reserved
        try:
            response = super().__call__(*args, **kwargs)
        finally:
            for event in self.events[before:]:
                if event.get("total_tokens") is None:
                    self.unknown_reserved[event["model"]] += reserved
            self._admission_reserve = None
        if self.tokens[model] + self.unknown_reserved[model] > limit:
            raise ValueError("Observed usage exceeded the daily admission budget; attempt persisted")
        return response


def validate_suite(suite):
    if suite.get("models") != list(MODELS) or suite.get("baseline_model") != MODELS[1]:
        raise ValueError("This integration has one fixed, exact Groq candidate pair")
    if any(suite.get(key) != value for key, value in FIXED_LIMITS.items()):
        raise ValueError("Frozen episode settings or acquisition limits changed")
    if suite.get("prior_setup_http_attempts", 0) != 0:
        raise ValueError("A new acquisition must not hide earlier setup attempts")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 24:
        raise ValueError("Expected one fresh cohort of at most 24 tasks")
    seen_ids, seen_groups, seen_prompts = set(), set(), set()
    for task in tasks:
        task_id, group_id = task.get("task_id"), task.get("group_id")
        if type(task_id) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", task_id):
            raise ValueError("Task IDs must be safe local directory names")
        if type(group_id) is not str or not group_id or task.get("split") != "validation":
            raise ValueError("Only fresh validation groups are admitted")
        if task_id in seen_ids or group_id in seen_groups:
            raise ValueError("Duplicate validation task/group")
        text = task_prompt(task)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if task.get("prompt_text", text) != text or task.get("routing_text_sha256", digest) != digest:
            raise ValueError("Routing text differs from the actual agent task")
        import unicodedata
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        normalized_digest = hashlib.sha256(normalized.encode()).hexdigest()
        if normalized_digest in seen_prompts:
            raise ValueError("Duplicate normalized validation prompt")
        seen_ids.add(task_id)
        seen_groups.add(group_id)
        seen_prompts.add(normalized_digest)


def _route_rows(suite, policy, scorer=None):
    api = _policy_api()
    result = []
    encoded_seconds = ({row["task_id"]: row["routing_seconds"] for row in _read(BERT_REFERENCE_PATH)["rows"]}
                       if scorer is not None else {})
    for task in sorted(suite["tasks"], key=lambda task: task["task_id"]):
        options = {"scorer": scorer} if scorer is not None else {}
        decision = api.select_model(policy, task_prompt(task), **options)
        if decision["model"] not in MODELS:
            raise ValueError("Policy selected a model outside the frozen catalog")
        result.append({"task_id": task["task_id"], "group_id": task["group_id"],
                       "routing_text_sha256": hashlib.sha256(task_prompt(task).encode()).hexdigest(),
                       "model": decision["model"], "score": decision["score"],
                       "routing_seconds": decision["routing_seconds"],
                       "encoder_seconds": encoded_seconds.get(task["task_id"], 0.),
                       "input_tokens": decision.get("input_tokens"),
                       "truncated_tokens": decision.get("truncated_tokens")})
    return result


def prepare_run(suite_path, policy_path, output):
    """Persist the policy decisions and acquisition contract before SDK/API use."""
    suite_raw, policy_raw = Path(suite_path).read_bytes(), Path(policy_path).read_bytes()
    suite, policy = json.loads(suite_raw), json.loads(policy_raw)
    validate_suite(suite)
    _policy_api().validate_artifact(policy)
    if policy.get("baseline_model") != MODELS[1] or policy.get("candidate_model") != MODELS[0]:
        raise ValueError("Policy candidates do not match this integration")
    _policy_api().validate_fresh_prompts(policy, [
        {"task_id": task["task_id"], "group_id": task["group_id"], "prompt_text": task_prompt(task)}
        for task in suite["tasks"]
    ])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "suite_sha256": hashlib.sha256(suite_raw).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "suite_content_sha256": _digest(suite), "policy_content_sha256": _digest(policy),
        "artifact_id": policy["artifact_id"], "source_sha256": source_hashes(),
        "bert_reference": reference_contract(),
        "tariffs": TARIFFS, "limits": FIXED_LIMITS,
        "cost_scope": "LLM_uncached_tariff_scenario_excludes_local_compute",
        "evaluation": "paired_branch_replay_of_frozen_routes",
    }
    routes_path = output / "frozen_routes.json"
    scorer = _scorer_for(suite, policy, contract["suite_sha256"])
    current_routes = _route_rows(suite, policy, scorer)
    if routes_path.exists():
        frozen = _read(routes_path)
        seal = frozen.pop("freeze_id", None)
        if seal != _digest(frozen):
            raise ValueError("Frozen route integrity mismatch")
        frozen["freeze_id"] = seal
        if frozen["contract"] != contract:
            raise ValueError("Cannot resume after changing policy, suite, sources, tariffs or limits")
        # Timing is measured only on the initial decision; model and score must
        # reproduce exactly before a report or a resumed dispatch is permitted.
        comparable = lambda rows: [{key: val for key, val in row.items() if key != "routing_seconds"} for row in rows]
        if comparable(frozen["routes"]) != comparable(current_routes):
            raise ValueError("Frozen routes no longer reproduce")
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing to freeze a new policy in a used output directory")
        frozen = {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                  "contract": contract, "routes": current_routes,
                  "feature_source": "precomputed_pinned_BERT" if scorer is not None else "resident_structural_features",
                  "encoder_original_load_seconds": scorer.provenance.get("original_load_seconds") if scorer is not None else 0.,
                  "local_routing_cost_usd": None, "net_monetary_savings": None}
        frozen["freeze_id"] = _digest(frozen)
        write(routes_path, frozen)
    return suite, policy, frozen


def _episode_id(task_id, model):
    return task_id + "-" + model.rsplit("/", 1)[-1]


def read_observations(suite, output):
    """Reject duplicates, out-of-cohort rows and unresolved episode starts."""
    rows = _rows(output / "observations.jsonl")
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    identities = set()
    for row in rows:
        identity = row.get("task_id"), row.get("model")
        if identity in identities or identity[0] not in tasks or identity[1] not in MODELS:
            raise ValueError("Duplicate or out-of-cohort observation")
        task = tasks[identity[0]]
        if row.get("group_id") != task["group_id"] or row.get("split") != "validation":
            raise ValueError("Observation group/split changed")
        if row.get("evaluation_id") != suite["evaluation_id"]:
            raise ValueError("Observation evaluation ID changed")
        episode_id = _episode_id(*identity)
        if _read(output / "episodes" / episode_id / "accounted_episode.json") != row:
            raise ValueError("Episode accounting differs from its observation")
        start = _read(output / "episode_starts" / (episode_id + ".json"))
        if any(start.get(key) != value for key, value in {
            "episode_id": episode_id, "task_id": row["task_id"], "model": row["model"]
        }.items()):
            raise ValueError("Episode no longer matches its write-ahead record")
        identities.add(identity)
    starts = {path.stem for path in (output / "episode_starts").glob("*.json")}
    accounted = {_episode_id(*identity) for identity in identities}
    directories = {path.name for path in (output / "episodes").iterdir()} if (output / "episodes").exists() else set()
    if starts != accounted or directories != accounted:
        raise ValueError("An episode is unresolved; never silently rerun or discard it")
    return rows


def reconcile_http(suite, output, observations):
    """Ensure every dispatch is costed, or explicitly retained as unknown."""
    events = _rows(output / "http_attempts.jsonl")
    starts = list((output / "attempt_starts").glob("*.json"))
    if len(starts) != len(events) or [row.get("attempt_id") for row in events] != list(range(1, len(events) + 1)):
        raise ValueError("Unresolved or duplicate HTTP attempts")
    if len(events) > suite["max_total_http_attempts"]:
        raise ValueError("Acquisition HTTP cap exceeded")
    by_episode = {_episode_id(row["task_id"], row["model"]): [] for row in observations}
    observed_tokens = {model: 0 for model in MODELS}
    for event in events:
        start = _read(output / "attempt_starts" / (str(event["attempt_id"]) + ".json"))
        if any(event.get(key) != start.get(key) for key in ("attempt_id", "episode_id", "model", "started_at_utc")):
            raise ValueError("Dispatch no longer matches its write-ahead record")
        if event.get("status") not in ("completed", "error") or event["episode_id"] not in by_episode:
            raise ValueError("Unaccounted HTTP dispatch")
        model = event["model"]
        if model not in TARIFFS or not event["episode_id"].endswith("-" + model.rsplit("/", 1)[-1]):
            raise ValueError("Dispatch model mismatch")
        inp, out = event.get("input_tokens"), event.get("output_tokens")
        known = type(inp) is int and inp >= 0 and type(out) is int and out >= 0
        total_tokens = event.get("total_tokens")
        if known:
            if type(total_tokens) is not int or total_tokens != inp + out:
                raise ValueError("HTTP total_tokens does not equal input_tokens plus output_tokens")
            observed_tokens[model] += total_tokens
        elif total_tokens is not None:
            raise ValueError("Unknown usage must retain an unknown total_tokens value")
        expected = ((inp * TARIFFS[model]["input_per_million_usd"] + out * TARIFFS[model]["output_per_million_usd"]) / 1e6
                    if known else None)
        if expected is None and event.get("cost_usd") is not None:
            raise ValueError("Missing token usage cannot be assigned zero cost")
        if expected is not None and (event.get("cost_usd") is None or not math.isclose(event["cost_usd"], expected, rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError("HTTP cost does not reproduce at the frozen tariffs")
        by_episode[event["episode_id"]].append(event)
    if any(total > suite["max_observed_tokens_per_model"] for total in observed_tokens.values()):
        raise ValueError("Per-model observed token acquisition cap exceeded")
    for row in observations:
        attempts = by_episode[_episode_id(row["task_id"], row["model"])]
        known = bool(attempts) and all(item["cost_usd"] is not None for item in attempts)
        total = sum(item["cost_usd"] for item in attempts) if known else None
        if row.get("cost_usd") != total or row.get("cost_complete") is not known or row.get("http_attempts") != len(attempts):
            raise ValueError("Episode does not include all its HTTP costs")
    return events


def build_report(suite, policy, frozen, output):
    observations = read_observations(suite, output)
    events = reconcile_http(suite, output, observations)
    if len(observations) != len(suite["tasks"]) * len(MODELS):
        raise ValueError("The complete paired cohort is required; missing tasks cannot be dropped")
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    records = [{**{key: row[key] for key in (
        "task_id", "group_id", "model", "accepted", "cost_usd", "cost_complete", "cost_basis"
    )}, "prompt_text": task_prompt(tasks[row["task_id"]])} for row in observations]
    scorer = _scorer_for(suite, policy, frozen["contract"]["suite_sha256"])
    options = {"scorer": scorer} if scorer is not None else {}
    evaluation = _policy_api().evaluate_policy(policy, records, **options)
    policy_seconds = sum(row["routing_seconds"] for row in frozen["routes"])
    encoder_seconds = sum(row["encoder_seconds"] for row in frozen["routes"])
    report = {
        "schema_version": 1, "evaluation": evaluation,
        "freeze_id": frozen["freeze_id"], "contract": frozen["contract"],
        "design": "paired_branch_replay; each candidate ran once on each fresh task",
        "fresh_tasks": len(tasks), "completed_episodes": len(observations), "http_attempts": len(events),
        "unknown_http_costs": sum(row["cost_usd"] is None for row in events),
        "acquisition_known_tariff_subtotal_usd": sum(row["cost_usd"] or 0 for row in events),
        "acquisition_total_tariff_usd": sum(row["cost_usd"] for row in events) if all(row["cost_usd"] is not None for row in events) else None,
        "routing_seconds": policy_seconds + encoder_seconds,
        "routing_policy_seconds": policy_seconds, "routing_encoder_seconds": encoder_seconds,
        "routing_encoder_load_seconds": frozen["encoder_original_load_seconds"],
        "routing_measurement": ("sum of earlier real encoder timings and cached head replay; excludes loading; not end-to-end latency"
                                if scorer is not None else "resident structural policy inference; no encoder"),
        "router_compute_price_usd": None, "net_monetary_savings": None,
        "scope": "public Python function tasks with OpenHands; empirical pilot, not production guarantee",
        "credential_persisted": False,
    }
    report["report_id"] = _digest(report)
    write(output / "comparison.json", report, exclusive=False)
    # Keep the evaluator's sealed artifact unmodified for the runtime loader.
    write(output / "validation.json", evaluation, exclusive=False)
    # Public task text stays in this local input file, never in routing ledgers.
    # Re-evaluation of the same frozen policy is reproducible without an API.
    records_path = output / "validation_records.json"
    if records_path.exists():
        if _read(records_path) != records:
            raise ValueError("Validation input file differs from the audited observations")
    else:
        write(records_path, records)
    return report


def execute_cohort(suite, policy, frozen, output, *, key, llm_class, llm_module, versions,
                   max_new_episodes=48, episode_runner=run_episode):
    """Dependency injection is for offline transport tests, never policy search."""
    if type(max_new_episodes) is not int or not 0 <= max_new_episodes <= 48:
        raise ValueError("max_new_episodes must be within [0,48]")
    output = Path(output)
    if _read(output / "frozen_routes.json") != frozen:
        raise ValueError("Routes must have been persisted before SDK use")
    if frozen["contract"]["source_sha256"] != source_hashes():
        raise ValueError("Sources changed after route freezing")
    if frozen["contract"]["bert_reference"] != reference_contract():
        raise ValueError("BERT reference changed after route freezing")
    if frozen["contract"]["suite_content_sha256"] != _digest(suite) or frozen["contract"]["policy_content_sha256"] != _digest(policy):
        raise ValueError("Suite or policy changed after route freezing")
    manifest = {"freeze_id": frozen["freeze_id"], "packages": versions, "credential_persisted": False}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        previous = _read(manifest_path)
        if any(previous.get(key) != value for key, value in manifest.items()):
            raise ValueError("Cannot resume with changed packages or frozen policy")
    else:
        manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(manifest_path, manifest)
    observations = read_observations(suite, output)
    reconcile_http(suite, output, observations)
    transport = BoundedAuditedTransport(llm_module.litellm_completion, suite, output)
    complete = {(row["task_id"], row["model"]) for row in observations}
    relay = Relay(suite["baseline_model"], log_path=output / "relay_events.jsonl")
    completed_now = 0
    tasks = sorted(suite["tasks"], key=lambda task: hashlib.sha256(task["task_id"].encode()).hexdigest())
    for task in tasks:
        models = list(MODELS)
        if int(hashlib.sha256(task["task_id"].encode()).hexdigest(), 16) % 2:
            models.reverse()
        for model in models:
            if (task["task_id"], model) in complete:
                continue
            if completed_now >= max_new_episodes:
                print("Lote terminado; politica y observaciones conservadas.", flush=True)
                return 0
            if len(transport.events) >= suite["max_total_http_attempts"]:
                raise ValueError("Acquisition HTTP cap exhausted; no further episode is dispatched")
            episode_id = _episode_id(task["task_id"], model)
            write(output / "episode_starts" / (episode_id + ".json"), {
                "episode_id": episode_id, "task_id": task["task_id"], "model": model,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_id": frozen["freeze_id"],
            })
            session = relay.task(task["workflow"], task_id=task["task_id"], group_id=task["group_id"],
                                 split="validation", evaluation_id=suite["evaluation_id"],
                                 model_override=model, max_calls=suite["max_calls_per_episode"])
            llm = llm_class(model=model, api_key=key, num_retries=0, api_mode="chat", timeout=90,
                            temperature=0, max_output_tokens=1024, reasoning_effort="low", log_completions=False,
                            usage_id="agent", caching_prompt=False,
                            capability_overrides={"supports_prompt_cache_key": False, "supports_prompt_cache_retention": False},
                            input_cost_per_token=TARIFFS[model]["input_per_million_usd"] / 1e6,
                            output_cost_per_token=TARIFFS[model]["output_per_million_usd"] / 1e6)
            transport.episode, transport.active = episode_id, []
            print(f"{len(observations)+1}/{len(tasks)*2} validation {task['task_id']} {model}", flush=True)
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
                       transport_seconds=sum(item["transport_seconds"] for item in attempts))
            write(output / "episodes" / episode_id / "accounted_episode.json", row)
            with (output / "observations.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            observations.append(row)
            completed_now += 1
            print(json.dumps({name: row[name] for name in ("accepted", "http_attempts", "cost_usd", "error_type")}), flush=True)
            native_tool_failure = bool(attempts) and attempts[-1].get("provider_error_code") == "tool_use_failed"
            if (row["error_type"] or not known) and not native_tool_failure:
                print("Fallo conservado con su costo desconocido; adquisicion detenida para revisar.", flush=True)
                return 2
    build_report(suite, policy, frozen, output)
    print("Adaptacion evaluada; comparison.json y validation.json guardados.", flush=True)
    return 0


def _credential(pipe):
    if pipe:
        if not re.fullmatch(r"\\\\\.\\pipe\\relay-groq-[a-f0-9]{32}", pipe):
            raise ValueError("Expected a one-use Windows memory pipe")
        with open(pipe, "rb", buffering=0) as stream:
            key = stream.readline(256).decode("ascii").strip()
    else:
        key = getpass.getpass("Clave Groq (oculta, solo en memoria): ").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("Missing credential")
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-pipe")
    parser.add_argument("--max-new-episodes", type=int, default=48)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.max_new_episodes <= 48:
        parser.error("--max-new-episodes must be within [0,48]")
    suite, policy, frozen = prepare_run(args.suite, args.policy, args.output)
    output = args.output.resolve()
    if args.freeze_only or args.max_new_episodes == 0:
        print("Rutas congeladas antes de cualquier llamada; frozen_routes.json.")
        return 0
    if args.report_only:
        build_report(suite, policy, frozen, output)
        return 0
    # Check interrupted acquisition before requesting or importing a credential.
    observations = read_observations(suite, output)
    reconcile_http(suite, output, observations)
    key = None
    try:
        import litellm
        from openhands.sdk.llm import LLM
        import openhands.sdk.llm.llm as llm_module
        litellm.telemetry = False
        logging.getLogger().setLevel(logging.ERROR)
        key = _credential(args.credential_pipe)
        versions = {name: importlib.metadata.version(name) for name in ("openhands-sdk", "litellm")}
        return execute_cohort(suite, policy, frozen, output, key=key, llm_class=LLM, llm_module=llm_module,
                              versions=versions, max_new_episodes=args.max_new_episodes)
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
