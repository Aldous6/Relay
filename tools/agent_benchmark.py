"""Prepare or run only the explicitly simulated episode fixture; no providers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.agent import AlwaysStrong, AlwaysWeak, EscalateOnError
from routerbench.agent_tasks import build_tasks
from routerbench.agent_evaluation import prepare_episodes, run_episodes
from routerbench.agent_simulation import ScriptedToolFixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "simulate"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_benchmark_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["execution_mode"] != "simulated" or config["generation_manifest"].get("kind") != "scripted_mock":
        raise ValueError("Este CLI solo ejecuta simulación explícita, nunca modelos reales.")
    tasks = build_tasks(config["seed"], config["per_family"])
    strong, weak = config["models"]["strong"], config["models"]["weak"]
    policies = [AlwaysStrong(strong, weak), AlwaysWeak(strong, weak), EscalateOnError(strong, weak)]
    generate = ScriptedToolFixture(weak)
    if args.mode == "prepare":
        frozen = prepare_episodes(tasks, policies, args.output, config, generate=generate)
        print(json.dumps({"prepared_tasks": len(tasks), "freeze_sha256": frozen["freeze_sha256"], "generation_calls": 0}))
    else:
        result = run_episodes(tasks, policies, generate, args.output, config)
        print(json.dumps({"tasks": result["n_tasks"], "episodes": result["n_episodes"],
                          "scripted_calls_this_invocation": generate.calls,
                          "real_model_calls": 0, "execution_mode": "simulated",
                          "summary": str(args.output / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
