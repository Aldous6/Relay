"""Select a frozen probability threshold on paired development continuations.

This is offline reference-policy selection, NOT an end-to-end evaluation of
repeated learned routing. Development is grouped by source task, not by step.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from relay_router.trajectory import TrajectoryValuePolicy
from relay_router.v8 import NReactiveCascade
from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor
from train_agent_trajectory import parse_dataset, read_json

THRESHOLDS = (.5, .65, .8, .9, .95, .99)


def select_threshold(predictor, rows, *, max_success_loss=.02):
    if type(max_success_loss) not in (float, int) or not 0 <= max_success_loss <= 1:
        raise ValueError("max_success_loss must be a probability difference in [0,1].")
    # Register/audit development BEFORE constructing policies with fixed manifests.
    diagnostics = predictor.evaluate(rows)
    states = defaultdict(dict)
    for row in rows:
        key = (row.source_group, row.request.serialize())
        if row.model in states[key]:
            raise ValueError("A paired state has duplicate model observations.")
        states[key][row.model] = row
    if any(set(values) != set(predictor.models) for values in states.values()):
        raise ValueError("Threshold selection requires every model at every development state.")
    if any(row.remaining_provider_cost_usd is None for row in rows):
        raise ValueError("Unknown development costs block cost-based operating-point selection.")

    def measure(policy):
        groups, latency, selections = defaultdict(list), [], {}
        for (group, signature), values in states.items():
            request = next(iter(values.values())).request
            started = perf_counter()
            model = policy.select(request)
            latency.append(perf_counter() - started)
            # Only after selection, inspect the branch's observed outcome/cost.
            observed = values[model]
            groups[group].append((int(observed.success), observed.remaining_provider_cost_usd))
            selections[model] = selections.get(model, 0) + 1
        per_group = [np.mean(values, axis=0) for values in groups.values()]
        means = np.mean(per_group, axis=0)
        return {"success_rate": float(means[0]), "mean_remaining_provider_cost_usd": float(means[1]),
                "routing_p95_seconds": float(np.quantile(latency, .95)), "model_selections": selections}

    class StrongProbe:
        def select(self, request):
            return predictor.models[-1]

    strong = measure(StrongProbe())
    reactive = measure(NReactiveCascade(predictor.models))
    grid = []
    for threshold in THRESHOLDS:
        policy = TrajectoryValuePolicy(predictor, threshold, "predeclared paired-development threshold grid")
        metrics = measure(policy)
        loss = strong["success_rate"] - metrics["success_rate"]
        grid.append({"min_probability": threshold, **metrics, "success_loss_vs_strong_probe": loss,
                     "eligible": loss <= max_success_loss})
    candidates = [point for point in grid if point["eligible"]]
    selected = min(candidates, key=lambda point: (point["mean_remaining_provider_cost_usd"],
        -point["success_rate"], -point["min_probability"])) if candidates else None
    return {"threshold_grid": list(THRESHOLDS), "max_success_loss": max_success_loss,
            "groups": len({group for group, _ in states}), "states": len(states), "grid": grid,
            "strong_probe_reference": strong, "reactive_reference": reactive, "selected": selected,
            "development_diagnostics": diagnostics, "independent_validation": False,
            "scope": "group-balanced paired reference continuations; no guarantee for repeated learned routing, no adoption claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new operating-point report.")
    models, partitions = parse_dataset(read_json(args.data))
    predictor = TrajectorySuccessPredictor.load(args.artifact, expected_sha256=args.artifact_sha256)
    if predictor.models != models:
        raise ValueError("Artifact and development catalog differ.")
    training = read_json(args.artifact.with_name(args.artifact.name + ".training.json"))
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if training["artifact_sha256"] != args.artifact_sha256 or training["data_file_sha256"] != data_sha:
        raise ValueError("Selection must use the development partition declared in the artifact training record.")
    result = select_threshold(predictor, partitions["development"])
    result.update(artifact_sha256=args.artifact_sha256, data_file_sha256=data_sha,
                  selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if result["selected"]:
        result["trajectory_policy"] = {"artifact_path": str(args.artifact.resolve()),
            "artifact_sha256": args.artifact_sha256, "min_probability": result["selected"]["min_probability"],
            "operating_point_source": str(args.output.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "selected": result["selected"], "provider_calls": 0}, indent=2))
    return result


if __name__ == "__main__":
    main()
