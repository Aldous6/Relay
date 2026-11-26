"""Freeze or execute the bounded local agent pilot with existing verified models.

The default only prepares. --execute explicitly runs the already available
Ollama models. No downloads, server startup, fitting, paid APIs or threshold
selection occur. A completed output is verified and returned without generation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.agent import AlwaysStrong, AlwaysWeak, EscalateOnError
from routerbench.agent_tasks import build_tasks
from routerbench.agent_evaluation import prepare_episodes, run_episodes
from routerbench.agent_ollama import OllamaEpisodeGenerator
from routerbench.agent_references import LocalStructuralTransferPolicy, PublishedBERTTransferPolicy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_local_competition_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/agent-local-competition-v2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config.get("execution_mode") != "external" or config.get("evaluation_split") != "development"
            or config.get("independent_validation") is not False
            or config.get("primary_policy") != "cheap_then_escalate"
            or config.get("policy_order") != "cyclic_by_task"):
        raise ValueError("This CLI only runs the declared exploratory local protocol.")
    generate = OllamaEpisodeGenerator(config["models_metadata"], config["generation_options"],
                                      config["base_url"], config["timeout_seconds"])
    if set(config["models_metadata"]) != set(config["models"].values()):
        raise ValueError("Catalog metadata must match the two execution models exactly.")
    generate.verify_models()  # Read-only inventory, no inference or model startup.
    config["generation_manifest"] = generate.manifest()
    config["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["additional_runtime_versions"] = {name: importlib.metadata.version(name)
                                             for name in ("torch", "transformers")}
    tasks = build_tasks(config["seed"], config["per_family"])
    strong, weak = config["models"]["strong"], config["models"]["weak"]
    policies = [AlwaysStrong(strong, weak), AlwaysWeak(strong, weak), EscalateOnError(strong, weak),
                LocalStructuralTransferPolicy.from_local(ROOT), PublishedBERTTransferPolicy.from_local(ROOT)]
    frozen = prepare_episodes(tasks, policies, args.output, config, generate=generate)
    maximum_calls = sum(min(t["max_steps"], config["max_calls"]) for t in tasks) * len(policies)
    print(json.dumps({"prepared_tasks": len(tasks), "policies": [p.name for p in policies],
                      "maximum_generation_calls": maximum_calls, "freeze_sha256": frozen["freeze_sha256"],
                      "mode": "execute" if args.execute else "prepare_only"}), flush=True)
    if not args.execute:
        return
    result = run_episodes(tasks, policies, generate, args.output, config)
    generate.verify_models()
    print(json.dumps({"tasks": result["n_tasks"], "episodes": result["n_episodes"],
                      "summary": str(args.output / "summary.json"),
                      "independent_validation": False, "paid_api_calls": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
