"""Fit v8 from explicitly labelled, group-separated JSON observations offline.

No data is collected or relabelled. This command never invokes a provider,
opens RouterBench, chooses a probability threshold or claims final-episode ROI.
The development partition reports diagnostics and is never passed to fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.agent import AgentRequest, AgentState, AgentStateV8
from relay_router.v8 import model_order
from routerbench.agent_learning import StateSuccessPredictor, TrainingRow

PARTITIONS = ("fit", "calibration", "development")
ROW_FIELDS = {"request", "model", "success", "source_group", "target", "continuation_policy", "assignment"}
LEGACY_STATE = {"step", "attempt", "previous_errors", "available_tools", "remaining_calls"}
V8_STATE = {"infrastructure_errors", "infrastructure_attempt", "context_utf8_bytes", "progress", "failed_tiers"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object, parse_constant=_nonfinite)


def parse_request(payload):
    """Restore only the public contract; unknown/private fields are rejected."""
    if type(payload) is not dict or set(payload) != {"prompt", "history", "state"}:
        raise ValueError("request requires exactly prompt/history/state.")
    if type(payload["history"]) is not list or type(payload["state"]) is not dict:
        raise ValueError("request history must be a JSON list and state a JSON object.")
    state = dict(payload["state"])
    if not LEGACY_STATE.issubset(state) or set(state) - LEGACY_STATE - V8_STATE - {"context_length_unit"}:
        raise ValueError("Unknown/private state fields or missing base state fields.")
    extended = bool(set(state) & (V8_STATE | {"context_length_unit"}))
    if "context_length_unit" in state:
        if state.pop("context_length_unit") != "utf8_bytes_prompt_and_history":
            raise ValueError("context_length_unit must preserve the explicit UTF-8-byte unit.")
    for name in ("previous_errors", "available_tools", "infrastructure_errors", "failed_tiers"):
        if name in state:
            if type(state[name]) is not list:
                raise ValueError(f"{name} must be a JSON list.")
            state[name] = tuple(state[name])
    restored_state = (AgentStateV8 if extended else AgentState)(**state)
    return AgentRequest(payload["prompt"], tuple(payload["history"]), restored_state)


def parse_dataset(payload, models=None):
    expected = {"schema_version", "model_catalog", *PARTITIONS}
    if type(payload) is not dict or set(payload) != expected or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Dataset requires schema_version=1, model_catalog and exactly fit/calibration/development.")
    if type(payload["model_catalog"]) is not list:
        raise ValueError("model_catalog must be an ordered JSON list.")
    catalog = model_order(tuple(payload["model_catalog"]))
    if models is not None and tuple(models) != catalog:
        raise ValueError("The requested catalog/order differs from the dataset catalog.")
    partitions = {}
    for part in PARTITIONS:
        raw = payload[part]
        if type(raw) is not list or not raw:
            raise ValueError(f"{part} must contain explicit nonempty labelled rows.")
        parsed = []
        for row in raw:
            if type(row) is not dict or set(row) != ROW_FIELDS:
                raise ValueError(f"{part} rows require exactly the declared TrainingRow fields; no labels are inferred.")
            values = {key: value for key, value in row.items() if key != "request"}
            parsed.append(TrainingRow(request=parse_request(row["request"]), **values))
        partitions[part] = parsed
    return catalog, partitions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local joblib artifact path; existing files are never overwritten.")
    catalog_args = parser.add_mutually_exclusive_group()
    catalog_args.add_argument("--models", nargs="+", help="Exact model IDs ordered weak to strong, matching the dataset.")
    catalog_args.add_argument("--config", type=Path, help="Frozen config containing the same ordered model_catalog.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
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
    if training_path.exists():
        raise FileExistsError("Use a new output path; the training record already exists.")
    predictor = StateSuccessPredictor(models, seed=args.seed, max_depth=args.max_depth,
                                      min_samples_leaf=args.min_samples_leaf)
    predictor.fit(partitions["fit"], partitions["calibration"])
    diagnostics = predictor.evaluate(partitions["development"])
    # All partition checks and diagnostics finish before creating artifacts.
    sha = predictor.save(args.output)
    record = {
        "schema_version": 1, "artifact_sha256": sha,
        "data_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files_sha256": {str(path.relative_to(ROOT).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__).resolve(), ROOT / "src/routerbench/agent_learning.py", ROOT / "src/relay_router/v8.py")},
        "model_catalog": list(models), "predictor": predictor.manifest(),
        "partition_rows": {part: len(rows) for part, rows in partitions.items()},
        "development_diagnostics": diagnostics,
        "providers_called": False, "threshold_selection_executed": False,
        "scope": "offline state/action-success training; not final episode quality, independent validation, or adoption evidence",
    }
    with training_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    output = {"artifact": str(args.output), "artifact_sha256": sha,
              "training_record": str(training_path), "target": predictor.manifest()["target"],
              "development_diagnostics": diagnostics, "provider_calls": 0}
    print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
    return output


if __name__ == "__main__":
    main()
