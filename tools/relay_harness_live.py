"""Run the frozen public OpenHands pilot with a memory-only Groq credential."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
from relay_router import Relay
from relay_router.harness_episode import run_episode
from relay_router.harness_comparison import compare, freeze_plan


def write(path, value, *, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class AuditedTransport:
    """Count every actual SDK transport call, including internal cache repairs."""
    def __init__(self, original, suite, output):
        self.original, self.suite, self.output = original, suite, output
        self.events = []
        self.active = []
        self.windows = {model: deque() for model in suite["models"]}
        self.tokens = {model: 0 for model in suite["models"]}
        path = output / "http_attempts.jsonl"
        if path.exists():
            self.events = [json.loads(line) for line in path.read_text().splitlines()]
            for row in self.events:
                self.tokens[row["model"]] += row.get("total_tokens") or 0
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["started_at_utc"])).total_seconds()
                if age < 180:
                    self.windows[row["model"]].append((time.monotonic(), row.get("total_tokens") or 7900))
        self.episode = None

    def save(self, row):
        with (self.output / "http_attempts.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def __call__(self, *args, **kwargs):
        model = kwargs.get("model")
        # OpenHands splits the provider from the model before calling LiteLLM.
        # Only accept this exact equivalent representation of the frozen pair.
        if kwargs.get("custom_llm_provider") == "groq" and "groq/" + str(model) in self.windows:
            model = "groq/" + model
        if model not in self.windows:
            raise ValueError("Transport model is outside the frozen catalog")
        # Request bodies and credentials never enter the accounting log.
        request_chars = len(json.dumps({"messages": kwargs.get("messages"), "tools": kwargs.get("tools")}, default=str).encode())
        reserved = min(7900, request_chars + self.suite["max_output_tokens"])
        if request_chars > 18000:
            raise ValueError("Task context exceeds the fixed 18 KiB admission limit")
        if len(self.events) + self.suite.get("prior_setup_http_attempts", 0) >= self.suite["max_total_http_attempts"]:
            raise ValueError("Total HTTP attempt cap reached before dispatch")
        if self.tokens[model] + reserved > self.suite["max_observed_tokens_per_model"]:
            raise ValueError("Per-model daily token budget reached before dispatch")
        wait_started = time.monotonic()
        window = self.windows[model]
        while window and (sum(row[1] for row in window) + reserved > 7900 or len(window) >= 28):
            now = time.monotonic()
            while window and now - window[0][0] >= 63:
                window.popleft()
            if not window or (sum(row[1] for row in window) + reserved <= 7900 and len(window) < 28):
                break
            time.sleep(min(3., max(.05, 63 - (now - window[0][0]))))
        row = {"attempt_id": len(self.events) + 1, "episode_id": self.episode, "model": model,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
               "quota_wait_seconds": time.monotonic() - wait_started, "input_tokens": None, "output_tokens": None,
               "total_tokens": None, "cost_usd": None, "cost_lower_cached_usd": None}
        self.events.append(row)
        self.active.append(row)
        # A separate write-ahead record preserves an interrupted dispatch.
        write(self.output / "attempt_starts" / (str(row["attempt_id"]) + ".json"), row)
        start = time.monotonic()
        try:
            response = self.original(*args, **kwargs)
            usage = response.usage
            inp, out = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
            if type(inp) is int and inp >= 0 and type(out) is int and out >= 0:
                rate = .075 if model.rsplit("/", 1)[-1] == "gpt-oss-20b" else .15
                output_rate = .30 if model.rsplit("/", 1)[-1] == "gpt-oss-20b" else .60
                cached_rate = .037 if model.rsplit("/", 1)[-1] == "gpt-oss-20b" else .075
                row.update(input_tokens=inp, output_tokens=out, total_tokens=inp + out,
                           cost_usd=(inp * rate + out * output_rate) / 1_000_000,
                           cost_lower_cached_usd=(inp * cached_rate + out * output_rate) / 1_000_000)
                self.tokens[model] += inp + out
            row["status"] = "completed"
            return response
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
            row["provider_error_code"] = "tool_use_failed" if "tool_use_failed" in str(exc) else None
            raise
        finally:
            row["transport_seconds"] = time.monotonic() - start
            window.append((time.monotonic(), row["total_tokens"] or reserved))
            self.save(row)


def load_suite(config_path):
    config = json.loads(config_path.read_text())
    raw = (ROOT / config["task_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["task_file_sha256"]:
        raise ValueError("Task source integrity mismatch")
    return config, json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=ROOT / "configs/harness_public_pilot_v2.json")
    parser.add_argument("--credential-pipe")
    parser.add_argument("--max-new-episodes", type=int, default=48)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--prior-setup-http-attempts", type=int, default=0)
    args = parser.parse_args(argv)
    config, suite = load_suite(args.suite)
    if not 0 <= args.prior_setup_http_attempts <= suite["max_total_http_attempts"]:
        raise ValueError("Invalid prior setup attempt count")
    suite["prior_setup_http_attempts"] = args.prior_setup_http_attempts
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = ROOT / "data/imported/harness-public-v1/bert_scores.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    observation_path = output / "observations.jsonl"
    observations = [json.loads(line) for line in observation_path.read_text().splitlines()] if observation_path.exists() else []
    if args.report_only:
        plan = json.loads((output / "frozen_plan.json").read_text())
        write(output / "comparison.json", compare(suite, plan, observations), exclusive=False)
        return 0
    key = None
    if args.credential_pipe:
        if not re.fullmatch(r"\\\\\.\\pipe\\relay-groq-[a-f0-9]{32}", args.credential_pipe):
            raise ValueError("Expected one-use Windows memory pipe")
        with open(args.credential_pipe, "rb", buffering=0) as stream:
            key = stream.readline(256).decode("ascii").strip()
    else:
        key = getpass.getpass("Clave Groq (oculta, solo en memoria): ").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("Missing credential")
    try:
        import litellm
        from openhands.sdk.llm import LLM
        import openhands.sdk.llm.llm as llm_module
        litellm.telemetry = False
        logging.getLogger().setLevel(logging.ERROR)
        manifest_path = output / "run_manifest.json"
        if not manifest_path.exists():
            write(manifest_path, {"started_at_utc": datetime.now(timezone.utc).isoformat(), "suite": config,
                                  "packages": {name: importlib.metadata.version(name) for name in ("openhands-sdk", "litellm")},
                                  "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (Path(__file__), ROOT / "src/relay_router/harness_episode.py", ROOT / "src/relay_router/harness_comparison.py")},
                                  "credential_persisted": False, "prior_setup_http_attempts": args.prior_setup_http_attempts,
                                  "reference": reference["reference"] if reference else None})
        elif json.loads(manifest_path.read_text())["suite"] != config:
            raise ValueError("Cannot resume with a changed cohort")
        suite["prior_setup_http_attempts"] = json.loads(manifest_path.read_text())["prior_setup_http_attempts"]
        transport = AuditedTransport(llm_module.litellm_completion, suite, output)
        complete_ids = {(row["task_id"], row["model"]) for row in observations}
        started_ids = {path.stem for path in (output / "episodes").glob("*")} if (output / "episodes").exists() else set()
        if len(list((output / "attempt_starts").glob("*.json"))) != len(transport.events):
            raise ValueError("Unresolved HTTP attempt; recover accounting before resuming")
        relay = Relay(suite["baseline_model"], log_path=output / "relay_events.jsonl")
        count = 0
        for split in ("development", "validation"):
            if split == "validation":
                plan_path = output / "frozen_plan.json"
                if not plan_path.exists():
                    plan = freeze_plan(suite, [row for row in observations if row["split"] == "development"], reference)
                    write(plan_path, plan)
                    print("Selecciones congeladas antes de evaluar validacion.", flush=True)
            tasks = sorted((row for row in suite["tasks"] if row["split"] == split),
                           key=lambda row: hashlib.sha256(row["task_id"].encode()).hexdigest())
            for task in tasks:
                models = list(suite["models"])
                if int(hashlib.sha256(task["task_id"].encode()).hexdigest(), 16) % 2:
                    models.reverse()
                for model in models:
                    if (task["task_id"], model) in complete_ids:
                        continue
                    if count >= args.max_new_episodes:
                        print("Lote terminado con observaciones persistidas.", flush=True)
                        return 0
                    episode_id = task["task_id"] + "-" + model.rsplit("/", 1)[-1]
                    if episode_id in started_ids:
                        raise ValueError("An episode was interrupted; do not silently rerun or discard it")
                    session = relay.task(task["workflow"], task_id=task["task_id"], group_id=task["group_id"], split=split,
                                         evaluation_id=suite["evaluation_id"], model_override=model, max_calls=suite["max_calls_per_episode"])
                    llm = LLM(model=model, api_key=key, num_retries=0, api_mode="chat", timeout=90, temperature=0,
                              max_output_tokens=suite["max_output_tokens"], reasoning_effort=suite["reasoning_effort"],
                              log_completions=False, usage_id="agent", caching_prompt=False,
                              capability_overrides={"supports_prompt_cache_key": False, "supports_prompt_cache_retention": False},
                              input_cost_per_token=(.075 if model.rsplit("/", 1)[-1] == "gpt-oss-20b" else .15) / 1e6,
                              output_cost_per_token=(.30 if model.rsplit("/", 1)[-1] == "gpt-oss-20b" else .60) / 1e6)
                    transport.episode, transport.active = episode_id, []
                    print(f"{len(observations)+1}/48 {split} {task['task_id']} {model}", flush=True)
                    with patch.object(llm_module, "litellm_completion", transport):
                        row = run_episode(task, llm, session, output / "episodes" / episode_id, max_calls=suite["max_calls_per_episode"],
                                          stop_on_public_pass=suite.get("stop_on_public_pass", False))
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
                    with observation_path.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    observations.append(row)
                    count += 1
                    print(json.dumps({key: row[key] for key in ("accepted", "http_attempts", "cost_usd", "error_type")}), flush=True)
                    model_action_failure = bool(attempts) and attempts[-1].get("provider_error_code") == "tool_use_failed"
                    if (row["error_type"] or not known) and not model_action_failure:
                        print("Episodio conservado. Se detiene para revisar el fallo antes de consumir mas llamadas.", flush=True)
                        return 2
        plan = json.loads((output / "frozen_plan.json").read_text())
        write(output / "comparison.json", compare(suite, plan, observations), exclusive=False)
        print("Comparacion completada; resultados en comparison.json.", flush=True)
        return 0
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
