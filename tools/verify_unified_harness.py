"""Check the installed Relay, real local BERT, sklearn and OpenHands together.

Only historical development is fitted. Frozen prompts check numerical/runtime
compatibility, not new quality or savings. All generation uses a mocked SDK
transport, and outbound sockets are blocked throughout the verification.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[name] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/unified-harness-environment-v1.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new output; previous verifications are retained")
    pins = {"torch": "2.9.0", "transformers": "4.57.1", "scikit-learn": "1.8.0",
            "openhands-sdk": "1.62.0", "relay-harness": "0.1.0.dev3"}
    packages = {name: importlib.metadata.version(name) for name in pins}
    require(packages == pins, "Installed package versions differ from the verified integration")
    for name in ("numpy", "scipy", "joblib", "threadpoolctl", "tokenizers", "huggingface-hub", "safetensors", "litellm", "pydantic"):
        packages[name] = importlib.metadata.version(name)

    run = ROOT / "outputs/harness-adaptive/v1/run"
    policy_path = run.parent / "policy-auto.json"
    suite_path = ROOT / "data/imported/harness-adaptive-v1/fresh_suite.json"
    reference_path = suite_path.parent / "bert_reference.json"
    development_path = suite_path.parent / "development.json"
    frozen, policy = read(run / "frozen_routes.json"), read(policy_path)
    suite, reference = read(suite_path), read(reference_path)
    require(sha(policy_path) == frozen["contract"]["policy_sha256"], "Frozen policy changed")
    require(sha(suite_path) == frozen["contract"]["suite_sha256"] == reference["suite_sha256"], "Frozen task input changed")
    require(sha(reference_path) == frozen["contract"]["bert_reference"]["sha256"], "Frozen BERT reference changed")
    require(sha(development_path) == "dbabcf423c5aa53aedf4bb91af9320aac274ec2a02c37be4a2e82454b7a13fc2", "Historical development changed")
    before = {str(path.relative_to(ROOT)): sha(path) for path in
              (policy_path, suite_path, reference_path, development_path, run / "validation.json",
               run / "pre_dispatch_recovery/validation-recovered.json")}
    reference_rows = {row["task_id"]: row for row in reference["rows"]}
    frozen_routes = {row["task_id"]: row for row in frozen["routes"]}

    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline unified verification forbids network")):
        import relay_harness
        from relay_harness import Relay, load_adapted_relay
        from relay_harness.adaptive_policy import select_model
        from relay_harness.adaptive_training import train_adaptation
        from relay_harness.integrations.routellm import LocalBERTScorer
        from relay_harness.integrations.openhands import attach_openhands_agent
        from relay_harness.prompt_selection import PairwisePromptSelector
        from litellm import ModelResponse
        from openhands.sdk import Agent, LLM
        from openhands.sdk.llm import Message, TextContent

        # Prove that the distributed wheel, not the source checkout, is running.
        installed = Path(relay_harness.__file__).resolve().parent
        require(installed.is_relative_to(Path(sys.prefix).resolve()), "Expected the installed wheel in this Python environment")
        manifest = read(installed / "source_manifest.json")
        for relative, expected in manifest["source_sha256"].items():
            shipped = installed / relative.removeprefix("src/relay_router/")
            require(sha(shipped) == expected, "An installed runtime source differs from its wheel manifest")

        print("Cargando BERT local real; red bloqueada...", flush=True)
        resident = LocalBERTScorer(ROOT / "artifacts/routellm-bert")
        print("Comprobando adaptación con desarrollo histórico...", flush=True)
        adapted, selection = train_adaptation(read(development_path)["records"],
                                               baseline_model=policy["baseline_model"],
                                               candidate_model=policy["candidate_model"], scope=policy["scope"], scorer=resident)
        require(adapted["representation"] == policy["representation"], "Representation selection changed in the combined environment")
        for name in ("fit_group_ids", "calibration_group_ids", "input_digest"):
            require(adapted["training"][name] == policy["training"][name], "Adaptation used different historical data or groups")

        print("Verificando las 24 decisiones congeladas...", flush=True)
        rows, examples = [], {}
        for task in suite["tasks"]:
            text = task["prompt_text"]
            expected_score = reference_rows[task["task_id"]]
            require(hashlib.sha256(text.encode()).hexdigest() == expected_score["prompt_sha256"], "Prompt hash changed")
            scored = resident(text)
            encoder_delta = abs(scored["score"] - expected_score["score"])
            require(encoder_delta <= 1e-7, "BERT score differs from its frozen reference")
            require(all(scored[name] == expected_score[name] for name in ("input_tokens", "truncated_tokens")), "Token coverage changed")
            cached = lambda incoming, original=text, value=scored: value if incoming == original else None
            decision = select_model(policy, text, scorer=cached)
            reproduced = select_model(adapted, text, scorer=cached)
            expected = frozen_routes[task["task_id"]]
            head_delta = abs(decision["score"] - expected["score"])
            require(head_delta <= 1e-7, "Frozen policy numerical score changed")
            require(decision["model"] == reproduced["model"] == expected["model"], "A frozen route changed")
            rows.append({"task_id": task["task_id"], "model": decision["model"],
                         "encoder_score_absolute_difference": encoder_delta,
                         "frozen_head_score_absolute_difference": head_delta,
                         "refitted_head_score_absolute_difference": abs(reproduced["score"] - decision["score"])})
            examples.setdefault(decision["model"], text)
        require(set(examples) == {policy["baseline_model"], policy["candidate_model"]}, "Expected both model routes in the frozen fixture")

        # This is an explicitly experimental software fixture, not promotion of
        # the real policy whose observed savings did not pass its activation gate.
        selector = PairwisePromptSelector(weak_model=policy["candidate_model"], strong_model=policy["baseline_model"],
                                          threshold=policy["threshold"],
                                          scorer=lambda text: select_model(policy, text, scorer=resident),
                                          selector_id="offline-unified-fixture")
        relay = Relay(policy["baseline_model"], mode="apply", prompt_selector=selector, allow_experimental=True)
        fixture_routes = []
        for expected, text in examples.items():
            clients = {model: LLM(model=model, api_key="offline-verification-only", num_retries=0, api_mode="chat",
                                  caching_prompt=False, usage_id="agent", input_cost_per_token=.000001,
                                  output_cost_per_token=.000002, capability_overrides={
                                      "supports_prompt_cache_key": False, "supports_prompt_cache_retention": False})
                       for model in examples}
            agent = Agent(llm=clients[policy["baseline_model"]], tools=[], condenser=None)
            task, routed = attach_openhands_agent(agent, relay, workflow="offline-engineering-fixture",
                                                   models=clients, routing_text=text, max_calls=2)
            require(task.model == expected, "The real encoder/SDK integration selected a different model")
            response = ModelResponse(id="offline-fixture", model=expected,
                                      choices=[{"message": {"role": "assistant", "content": "Synthetic response"}}],
                                      usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
            messages = [Message(role="user", content=[TextContent(text=text)])]
            with patch("openhands.sdk.llm.llm.litellm_completion", return_value=response) as transport:
                for _ in range(2):
                    routed.llm.generate(messages)
            require(transport.call_count == 2, "Unexpected transport count in the synthetic fixture")
            result = task.finish(accepted=None, cost_complete=True)
            require(result["logical_calls"] == 2 and task.model == expected, "The model was not sticky across calls")
            fixture_routes.append({"model": task.model, "synthetic_transport_calls": transport.call_count})

        gates = []
        for file in (run / "validation.json", run / "pre_dispatch_recovery/validation-recovered.json"):
            deployed = load_adapted_relay(policy_path, file, scope=policy["scope"], checkpoint_dir=ROOT / "artifacts/routellm-bert")
            task = deployed.task("function-implementation", routing_text=next(iter(examples.values())))
            require(task.model == policy["baseline_model"] and task.decision["reason"] == "baseline_adaptation_gate_not_met",
                    "Installing dependencies must not activate rejected evidence")
            task.finish(accepted=None)
            gates.append({"file": str(file.relative_to(ROOT)), "model": task.model, "alternative_activated": False})

    require(all(sha(ROOT / path) == digest for path, digest in before.items()), "Frozen evidence was modified")
    result = {"kind": "unified_harness_environment_verification", "at_utc": datetime.now(timezone.utc).isoformat(),
              "packages": packages, "installed_runtime_manifest": manifest,
              "python_executable": sys.executable, "python_version": sys.version,
              "network_blocked": True, "model_api_calls": 0, "new_quality_evidence": False,
              "savings_reestimated": False, "frozen_evidence_unchanged": True,
              "checkpoint": resident.provenance, "historical_training_reproduced": True,
              "original_policy_artifact_id": policy["artifact_id"], "refitted_artifact_id": adapted["artifact_id"],
              "threshold_absolute_difference": abs(adapted["threshold"] - policy["threshold"]),
              "training_seconds_including_real_encoder_inference": selection["elapsed_seconds"],
              "frozen_routes_reproduced": rows, "openhands_synthetic_transport_fixtures": fixture_routes,
              "real_policy_gate_checks": gates, "input_sha256": before,
              "combined_optional_dependency_installation": "installed_and_verified_in_one_environment"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "frozen_routes": len(rows), "sdk_model_routes": len(fixture_routes), "model_api_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
