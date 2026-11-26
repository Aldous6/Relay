"""Score the frozen task prompts with the pinned local BERT checkpoint."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.harness_episode import task_prompt
from relay_router.maintenance_reference import score_records

if __name__ == "__main__":
    config = json.loads((ROOT / "configs/harness_public_pilot_v1.json").read_text())
    raw = (ROOT / config["task_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config["task_file_sha256"]
    suite = json.loads(raw)
    result = score_records([{"task_id": row["task_id"], "routing_text": task_prompt(row)} for row in suite["tasks"]])
    out = ROOT / "data/imported/harness-public-v1/bert_scores.json"
    if out.exists():
        raise ValueError("Scores already frozen")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(result["rows"]), "verified": result["reference"]["checkpoint_verified"], "path": str(out)}))
