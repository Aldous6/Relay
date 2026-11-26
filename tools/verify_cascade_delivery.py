"""Verify the installed dev4 wheel, unchanged evidence and real SDK offline.

This is integration evidence with simulated replies. It does not acquire model
outcomes, re-evaluate a holdout, or certify product savings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cascade-runtime-delivery-v1.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a fresh delivery report path")
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline delivery check")):
        import relay_harness
        package_root = Path(relay_harness.__file__).resolve().parent
        require("site-packages" in str(package_root), "Run against the installed wheel")
        require(relay_harness.__version__ == "0.1.0.dev4", "This delivery verifies dev4")
        manifest = json.loads((package_root / "source_manifest.json").read_text())
        for relative, expected in manifest["source_sha256"].items():
            installed = package_root / Path(relative).relative_to("src/relay_router")
            require(digest(installed) == expected == digest(ROOT / relative), "Installed/source manifest mismatch")
        development_path = ROOT / "reports/openhands-cascade-development-v3.json"
        development = json.loads(development_path.read_text(encoding="utf-8"))
        for collection in ("source_sha256", "input_sha256"):
            for relative, expected in development["provenance"][collection].items():
                require(digest(ROOT / relative) == expected, "Development evidence changed")
        require(development["activation_eligible"] is False, "Development must not activate a policy")
        frozen = json.loads((ROOT / "outputs/harness-adaptive/v1/run/frozen_routes.json").read_text())
        for relative, expected in frozen["contract"]["source_sha256"].items():
            require(digest(ROOT / relative) == expected, "A frozen historical source changed")
        policy_path = ROOT / "outputs/harness-adaptive/v1/policy-auto.json"
        policy = json.loads(policy_path.read_text())
        old_gates = []
        for relative in ("outputs/harness-adaptive/v1/run/validation.json",
                         "outputs/harness-adaptive/v1/run/pre_dispatch_recovery/validation-recovered.json"):
            relay = relay_harness.load_adapted_relay(policy_path, ROOT / relative, scope=policy["scope"])
            task = relay.task("python-repair", routing_text="Offline gate compatibility fixture")
            require(task.model == policy["baseline_model"], "The rejected old policy was activated")
            task.finish(accepted=None)
            old_gates.append({"path": relative, "sha256": digest(ROOT / relative), "baseline_preserved": True})
        example_path = ROOT / "examples/relay_openhands_cascade.py"
        spec = importlib.util.spec_from_file_location("relay_cascade_delivery_example", example_path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        # Force installed modules even though the example lives in this checkout.
        example.NAMESPACE = "relay_harness"
        artifacts = ROOT / "outputs/cascade-demo" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-installed")
        result = example.run_offline_demo(artifacts)
        require(result["initial_verification"]["status"] == "failed", "Fixture failed to reproduce the original defect")
        require(result["escalated"] and result["selected_attempt"] == 2, "Clean escalation did not complete")
        require(result["accounting"]["http_calls"] == 2 and result["accounting"]["cost_complete"], "Both simulated dispatches must be counted")
        require(result["accepted"] is None and result["savings_usd"] is None, "Fixture must not claim quality or savings")
        require(result["original_snapshot_unchanged"], "The input snapshot was modified")
        require("torch" not in sys.modules and "sklearn" not in sys.modules, "Cascade should not need an ML model")
    wheel = ROOT / "dist/relay-harness/relay_harness-0.1.0.dev4-py3-none-any.whl"
    report = {"kind": "installed_cascade_runtime_offline_verification", "version": relay_harness.__version__,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
              "package_path": str(package_root), "wheel_sha256": digest(wheel),
              "installed_sources_verified": len(manifest["source_sha256"]),
              "historical_frozen_sources_verified": len(frozen["contract"]["source_sha256"]),
              "openhands_version": importlib.metadata.version("openhands-sdk"),
              "provider_requests": 0, "model_replies_simulated": True,
              "quality_or_savings_evidence": False, "old_adaptation_gates": old_gates,
              "development_report": str(development_path.relative_to(ROOT)),
              "development_sha256": digest(development_path),
              "example_sha256": digest(example_path), "verifier_script_sha256": digest(Path(__file__)),
              "fixture_artifacts": str(artifacts.relative_to(ROOT)),
              "fixture_result_sha256": digest(artifacts / "run/result.json"),
              "fixture_checks": {"initial_defect_reproduced": True, "clean_fallback": True,
                                 "both_attempts_counted": True, "accepted_unknown": True,
                                 "input_snapshot_unchanged": True, "no_ml_encoder_loaded": True}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"version": report["version"], "report": str(args.output),
                      "fixture_artifacts": str(artifacts), "provider_requests": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
