"""Train text/state episode-reference forecasts from explicit paired labels offline.

Schema 2 is deliberately distinct from v8's action-success proxy. This command
does not call providers, generate labels, optimize policy thresholds, or claim
that reference-continuation diagnostics establish repeated-policy performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.v8 import model_order
from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor, TrajectoryTrainingRow
from train_agent_v8 import PARTITIONS, _unique_object, _nonfinite, parse_request, read_json

ROW_FIELDS = set(TrajectoryTrainingRow.__dataclass_fields__)


def parse_dataset(payload, models=None):
    expected = {"schema_version", "model_catalog", *PARTITIONS}
    if (type(payload) is not dict or set(payload) != expected
            or type(payload.get("schema_version")) is not int or payload["schema_version"] != 2):
        raise ValueError("Trajectory dataset requires schema_version=2, model_catalog and exactly fit/calibration/development.")
    if type(payload["model_catalog"]) is not list:
        raise ValueError("model_catalog must be an ordered JSON list.")
    catalog = model_order(tuple(payload["model_catalog"]))
    if models is not None and tuple(models) != catalog:
        raise ValueError("The requested catalog/order differs from the dataset catalog.")
    partitions = {}
    for part in PARTITIONS:
        if type(payload[part]) is not list or not payload[part]:
            raise ValueError(f"{part} requires explicit nonempty observed trajectory rows.")
        parsed = []
        for row in payload[part]:
            if type(row) is not dict or set(row) != ROW_FIELDS:
                raise ValueError(f"{part} rows require exactly TrajectoryTrainingRow fields; no labels or costs are inferred.")
            values = {key: value for key, value in row.items() if key != "request"}
            parsed.append(TrajectoryTrainingRow(request=parse_request(row["request"]), **values))
        partitions[part] = parsed
    return catalog, partitions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New joblib path; artifacts are immutable.")
    requested_args = parser.add_mutually_exclusive_group()
    requested_args.add_argument("--models", nargs="+", help="Exact ordered catalog, matching the dataset.")
    requested_args.add_argument("--config", type=Path, help="Frozen config with matching model_catalog.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-features", type=int, default=12000)
    parser.add_argument("--success-regularization", type=float, default=1.0)
    parser.add_argument("--cost-regularization", type=float, default=1.0)
    args = parser.parse_args(argv)
    requested = args.models
    if args.config is not None:
        config = read_json(args.config)
        requested = config.get("model_catalog") if type(config) is dict else None
        if type(requested) is not list:
            raise ValueError("The config must declare its ordered model_catalog.")
    raw = args.data.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    models, partitions = parse_dataset(payload, requested)
    training_path = args.output.with_name(args.output.name + ".training.json")
    if any(path.exists() for path in (args.output, training_path, args.output.with_name(args.output.name + ".manifest.json"))):
        raise FileExistsError("Use a new output path; existing artifacts/training records are immutable.")
    predictor = TrajectorySuccessPredictor(models, seed=args.seed, max_text_features=args.max_text_features,
        success_regularization=args.success_regularization, cost_regularization=args.cost_regularization)
    predictor.fit(partitions["fit"], partitions["calibration"])
    diagnostics = predictor.evaluate(partitions["development"])
    # Evaluate disjointness and all diagnostics before writing the artifact.
    sha = predictor.save(args.output)
    sources = (Path(__file__).resolve(), ROOT / "tools/train_agent_v8.py",
        ROOT / "src/routerbench/agent_trajectory_learning.py", ROOT / "src/routerbench/agent_learning.py",
        ROOT / "src/routerbench/features.py", ROOT / "src/relay_router/agent.py", ROOT / "src/relay_router/v8.py")
    record = {"schema_version": 2, "artifact_sha256": sha,
        "data_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files_sha256": {str(path.relative_to(ROOT).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources},
        "model_catalog": list(models), "predictor": predictor.manifest(),
        "partition_rows": {part: len(values) for part, values in partitions.items()},
        "development_diagnostics": diagnostics, "providers_called": False,
        "threshold_selection_executed": False, "policy_thresholds_frozen": False,
        "next_required_step": "Choose policy parameters using development episodes, then freeze configuration before reserved evaluation.",
        "independent_validation": False,
        "scope": "offline reference-continuation success/cost training; not repeated-policy episode performance or adoption evidence"}
    with training_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    result = {"artifact": str(args.output), "artifact_sha256": sha, "training_record": str(training_path),
        "target": predictor.manifest()["target"], "development_diagnostics": diagnostics, "provider_calls": 0,
        "policy_thresholds_frozen": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return result


if __name__ == "__main__":
    main()
