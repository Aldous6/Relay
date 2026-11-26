"""Check real local scorer/runtime equivalence, not a new quality evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from relay_router import Relay
from relay_router.harness_episode import task_prompt
from relay_router.integrations.routellm import load_groq_pilot_selector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/prompt-runtime-v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output; verification reports are retained.")
    config = json.loads((ROOT / "configs/harness_public_pilot_v2.json").read_text())
    raw = (ROOT / config["task_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["task_file_sha256"]:
        raise ValueError("The frozen cohort changed.")
    suite = json.loads(raw)
    plan = json.loads((ROOT / "outputs/harness-live/public-v2/frozen_plan.json").read_text())
    reference = json.loads((ROOT / "data/imported/harness-public-v1/bert_scores.json").read_text())
    scores = {row["task_id"]: row for row in reference["rows"]}
    output = []
    print("Cargando selector BERT real, con red bloqueada...", flush=True)
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline verification forbids network")):
        selector, provenance = load_groq_pilot_selector(ROOT / "artifacts/routellm-bert")
        relay = Relay(selector.strong_model, prompt_selector=selector, mode="apply", allow_experimental=True)
        for task in suite["tasks"]:
            text = task_prompt(task)
            session = relay.task(task["workflow"], routing_text=text,
                                 task_id=task["task_id"], eligible_models=suite["models"])
            selected = session.model
            expected = plan["policies"]["bert_transferred_development_calibrated"][task["task_id"]]
            details = session.decision["prompt_selection"]
            delta = abs(details["preference_score"] - scores[task["task_id"]]["probability"])
            if selected != expected or delta > 1e-7:
                raise ValueError("Runtime differs from the frozen scorer or routing rule.")
            seen = []
            def client(model, *, messages):
                seen.append(model)
                assert messages == [{"role": "user", "content": text}]
                return {"model": model, "fixture": True}
            for _ in range(2):
                session.call(client, messages=[{"role": "user", "content": text}])
            observation = session.finish(accepted=None)
            assert seen == [expected, expected]
            assert observation["cost_usd"] is None and observation["routing_cost_usd"] is None
            assert observation["elapsed_seconds"] >= details["routing_seconds"]
            output.append({"task_id": task["task_id"], "split": task["split"], "model": selected,
                           "score_absolute_difference": delta, "routing_seconds": details["routing_seconds"],
                           "input_tokens": details["input_tokens"], "truncated_tokens": details["truncated_tokens"]})
            print(f"{len(output)}/24 {task['task_id']}: {selected}", flush=True)
    result = {"kind": "prompt_runtime_equivalence_check", "at_utc": datetime.now(timezone.utc).isoformat(),
              "real_local_scorer": True, "generation_transport": "synthetic_client_only",
              "model_api_calls": 0, "new_quality_evidence": False, "net_savings_measured": False,
              "all_frozen_decisions_match": True, "sticky_client_calls_per_task": 2,
              "provenance": provenance, "rows": output,
              "runtime_sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                  ["src/relay_router/workflow.py", "src/relay_router/prompt_selection.py", "src/relay_router/integrations/routellm.py"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Verificación guardada: {args.output}")


if __name__ == "__main__":
    main()
