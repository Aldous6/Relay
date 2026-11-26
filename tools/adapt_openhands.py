"""Reproduce automatic adaptation from the pinned historical public episodes.

No LLM calls. BERT representations were computed earlier with verified local
weights; this command verifies their task/text binding before reusing them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from relay_router.adaptive_training import train_adaptation
from relay_router.workflow_cli import _write
from adaptive_scoring import cached_bert_scorer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/harness-adaptive/v1/policy-auto.json")
    args = parser.parse_args(argv)
    selection_path = Path(str(args.output) + ".selection.json")
    if args.output.exists() or selection_path.exists():
        raise FileExistsError("An adaptation is already frozen; use a new output path")
    source = ROOT / "data/imported/harness-adaptive-v1/development.json"
    # This exact snapshot was extracted and source-verified before adaptation.
    if hashlib.sha256(source.read_bytes()).hexdigest() != "dbabcf423c5aa53aedf4bb91af9320aac274ec2a02c37be4a2e82454b7a13fc2":
        raise ValueError("Historical development snapshot changed")
    records = json.loads(source.read_text(encoding="utf-8"))["records"]
    reference = ROOT / "data/imported/harness-public-v1/bert_scores.json"
    scorer = cached_bert_scorer(reference, records)
    policy, selection = train_adaptation(records, baseline_model="groq/openai/gpt-oss-120b",
                                        candidate_model="groq/openai/gpt-oss-20b", scorer=scorer)
    _write(selection_path, selection)
    _write(args.output, policy)
    print(json.dumps({"policy": str(args.output), "representation": policy["representation"],
                      "threshold": policy["threshold"], "calibration_savings": policy["calibration"]["relative_savings"],
                      "artifact_id": policy["artifact_id"], "validation_pending": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
