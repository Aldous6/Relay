"""Recommend one historical catalog model using a verified local artifact.

This command does not call an LLM. Joblib artifacts must originate from this
project's trusted development run; a hash is an integrity check, not a signature.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import sha256_file
from .features import parse_prompt
from .policy import choose_models


def recommend(run_dir: Path, variant: str, prompt: str) -> dict:
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if variant not in report["variants"]:
        raise ValueError("La variante no está en el manifiesto de esta ejecución.")
    entry = report["variants"][variant]
    artifact = run_dir / entry["artifact"]
    if artifact.resolve().parent != run_dir.resolve() or sha256_file(artifact) != entry["artifact_sha256"]:
        raise ValueError("El artefacto no coincide con su manifiesto local.")
    parsed_prompt = parse_prompt(prompt)
    if not parsed_prompt.strip():
        raise ValueError("El prompt está vacío.")
    router = joblib.load(artifact)
    if hasattr(router.transformer_, "use_cache"):
        router.transformer_.use_cache = False
    inputs = pd.DataFrame({"prompt_text": [parsed_prompt]})
    probabilities, costs = router.predict_tables(inputs)
    decision = choose_models(probabilities, costs, report["configuration"]["policy"]["min_success_probability"]).iloc[0]
    return {"selected_model": decision["selected_model"], "predicted_success": float(decision["predicted_success"]),
            "predicted_historical_cost": float(decision["predicted_cost"]), "decision": decision["decision"],
            "all_predicted_success": probabilities.iloc[0].to_dict(),
            "llm_called": False, "production_validated": False,
            "scope": "experimental model recommendation in the historical RouterBench catalog, not current provider pricing or guaranteed success"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", default="lexical_hybrid")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file", type=Path)
    args = parser.parse_args(argv)
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    print(json.dumps(recommend(args.run_dir, args.variant, prompt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
