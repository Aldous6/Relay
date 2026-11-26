"""Freeze the historical BERT controls on new public prompts, without APIs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from relay_router.harness_episode import task_prompt
from relay_router.integrations.routellm import LocalBERTScorer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "data/imported/harness-adaptive-v1/fresh_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/imported/harness-adaptive-v1/bert_reference.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Reference already frozen; do not overwrite it")
    raw = args.suite.read_bytes()
    suite = json.loads(raw)
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline reference forbids network")):
        scorer = LocalBERTScorer(ROOT / "artifacts/routellm-bert")
        rows = []
        for task in suite["tasks"]:
            text = task_prompt(task)
            start = time.perf_counter()
            score = scorer(text)
            elapsed = time.perf_counter() - start
            rows.append({"task_id": task["task_id"], "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         **score, "routing_seconds": elapsed})
    result = {"kind": "frozen_historical_BERT_controls", "at_utc": datetime.now(timezone.utc).isoformat(),
              "suite_sha256": hashlib.sha256(raw).hexdigest(), "reference": scorer.provenance,
              "thresholds": [0.50, 0.58], "truncation_policy": "keep_baseline",
              "threshold_provenance": "historical_public_v2_development_frozen_no_new_outcomes",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), ROOT / "src/relay_router/integrations/routellm.py")},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"path": str(args.output), "tasks": len(rows), "load_seconds": scorer.load_seconds,
                      "routing_seconds": sum(r["routing_seconds"] for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
