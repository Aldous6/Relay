"""Verify the installed wheel against the real frozen evidence, without APIs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/adaptive-runtime-delivery.json")
    destination = parser.parse_args().output
    if destination.exists():
        raise FileExistsError("Delivery verification already exists")
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline delivery check")):
        import relay_harness
        from relay_harness import load_adapted_relay
        from relay_harness.integrations.openhands import attach_openhands_agent
        from openhands.sdk import Agent, LLM
        policy_path = ROOT / "outputs/harness-adaptive/v1/policy-auto.json"
        policy = json.loads(policy_path.read_text())
        model = policy["baseline_model"]
        outcomes = []
        for report_path in (ROOT / "outputs/harness-adaptive/v1/run/validation.json",
                            ROOT / "outputs/harness-adaptive/v1/run/pre_dispatch_recovery/validation-recovered.json"):
            report = json.loads(report_path.read_text())
            assert report["activation_eligible"] is False
            relay = load_adapted_relay(policy_path, report_path, scope=policy["scope"])
            llm = LLM(model=model, api_key="offline-verification-only", num_retries=0, api_mode="chat",
                      caching_prompt=False, capability_overrides={"supports_prompt_cache_key": False,
                                                                  "supports_prompt_cache_retention": False})
            agent = Agent(llm=llm, tools=[], condenser=None)
            task, routed = attach_openhands_agent(agent, relay, workflow="function-implementation",
                                                   models={model: llm}, routing_text="Offline delivery wiring check")
            assert task.model == model and routed.llm.relay_session is task and agent.llm is llm
            assert task.decision["reason"] == "baseline_adaptation_gate_not_met"
            task.finish(accepted=None, cost_complete=False)
            outcomes.append({"validation_report_id": report["validation_report_id"],
                             "validation_file": str(report_path.relative_to(ROOT)),
                             "activated": False, "selected_model": task.model,
                             "reason": task.decision["reason"]})
        assert "torch" not in sys.modules and "sklearn" not in sys.modules
    wheel = ROOT / f"dist/relay-harness/relay_harness-{relay_harness.__version__}-py3-none-any.whl"
    optional = {}
    requested = {"torch": "2.9.0", "transformers": "4.57.1", "scikit-learn": "1.8.0"}
    for name in requested:
        try:
            optional[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional[name] = None
    result = {"kind": "installed_adaptive_runtime_offline_verification", "version": relay_harness.__version__,
              "openhands_version": importlib.metadata.version("openhands-sdk"),
              "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
              "policy_artifact_id": policy["artifact_id"], "outcomes": outcomes,
              "remote_calls": 0, "synthetic_wiring_only_not_quality_evidence": True,
              "optional_encoder_not_loaded_when_gate_fails": True,
              "optional_dependency_versions": optional,
              "combined_optional_dependency_installation": "installed_requested_versions" if optional == requested else "missing_or_different_versions"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"version": result["version"], "report": str(destination), "checks": len(outcomes), "remote_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
