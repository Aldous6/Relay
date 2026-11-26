"""Development-only catalog audit; reuse prior 1.5B answers, call 0.5B once."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_transfer_pilot import SYSTEM
from routerbench.data import sha256_file
from routerbench.live import _request_json, _usage, discover_ollama
from routerbench.prospective import _hash, run_prospective


def main():
    import pandas as pd

    previous = ROOT / "reports/local-transfer-pilot-v1"
    source = json.loads((previous / "evaluation/summary.json").read_text(encoding="utf-8"))
    if source["result_sha256"] != _hash({k: v for k, v in source.items() if k != "result_sha256"}):
        raise ValueError("La referencia local cambió.")
    responses_path = previous / "evaluation/responses.json"
    if sha256_file(responses_path) != source["responses_sha256"]:
        raise ValueError("Las respuestas anteriores cambiaron.")
    responses = json.loads(responses_path.read_text(encoding="utf-8"))["payload"]
    reusable = {r["prompt_sha256"]: r for r in responses.values() if r["model"] == "weak"}
    small = "qwen2.5:0.5b"
    available = {r["model"]: r for r in discover_ollama(include_context=True)}
    if small not in available or available[small].get("context_length", 0) < 8192:
        raise ValueError("Falta el Qwen pequeño instalado; no se descarga ni sustituye.")
    output = ROOT / "reports/local-catalog-audit-v1"
    manifest = {"scope": "development catalog audit after viewing the prior synthetic pilot; not a new independent evaluation",
        "prior_summary_sha256": sha256_file(previous / "evaluation/summary.json"),
        "models": {"strong": "qwen2.5:1.5b cached prior responses", "weak": small},
        "small_digest": available[small]["digest"], "generation_options": {"num_ctx": 8192, "num_predict": 96, "temperature": 0, "seed": 42},
        "paid_api_calls": False, "code_sha256": sha256_file(Path(__file__).resolve()),
        "new_calls_expected": 64, "prior_responses_reused": 64, "monetary_cost_usd": None}
    known = set(pd.read_parquet(ROOT / "data/processed/rows.parquet")["prompt_group"])
    count = 0

    def generate(alias, prompt):
        nonlocal count
        import hashlib

        if alias == "strong":
            value = reusable[hashlib.sha256(prompt.encode()).hexdigest()]
            if value["status"] != "ok":
                raise ValueError("La respuesta reutilizada no estaba completa.")
            return {key: value[key] for key in ("content", "usage", "duration_seconds")}
        if {r["model"]: r for r in discover_ollama()}[small]["digest"] != manifest["small_digest"]:
            raise ValueError("El modelo cambió durante la colección.")
        start = time.perf_counter()
        value = _request_json("http://127.0.0.1:11434/api/chat", {"model": small,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "stream": False, "options": manifest["generation_options"]}, timeout=120)
        count += 1
        print(f"Qwen 0.5B: nueva respuesta {count}/64.", flush=True)
        return {"content": value.get("message", {}).get("content"),
                "usage": _usage(value.get("prompt_eval_count"), value.get("eval_count")),
                "duration_seconds": time.perf_counter()-start}

    result = run_prospective(previous / "cohort.jsonl", output, manifest, lambda _: "weak", lambda _: "strong", generate,
                             known_prompt_groups=known, collect_both_models=True)
    print(json.dumps({"scope": manifest["scope"], "new_calls_this_execution": count, "policies": result["policies"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
