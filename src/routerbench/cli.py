"""Audit and prepare reproducible artifacts; never trains or scores the final test."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from .data import audit_frame, hash_partition, load_verified_pickle, numeric_results, prompt_metadata, sha256_file, write_json
from .features import FEATURE_NAMES, extract_features
from .labels import fit_tiers, make_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit(frame, config: dict, output: Path, source: Path) -> None:
    report = audit_frame(frame, config["sensitivity_thresholds"])
    report.update({"dataset_sha256": sha256_file(source), "dataset_revision": config["dataset"]["revision"],
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(), "kind": "descriptive audit; no model performance"})
    write_json(output / "audit.json", report)
    print(f"Auditoría: {report['rows']} filas, {report['models_count']} modelos; {output / 'audit.json'}")


def run_prepare(frame, config: dict, output: Path, source: Path) -> None:
    artifact_names = ("features.parquet", "rows.parquet", "labels.parquet", "manifest.json")
    if any((output / name).exists() for name in artifact_names):
        raise FileExistsError("La salida ya contiene artefactos. Usa --output con una carpeta nueva para conservar la preparación anterior.")
    if config["split"]["strategy"] != "prompt_hash":
        raise ValueError("Estrategia de partición no implementada.")
    models, _, _ = numeric_results(frame)
    if len(models) != config["dataset"]["expected_models"]:
        raise ValueError("Cantidad inesperada de modelos.")
    metadata = prompt_metadata(frame)
    keep = ~metadata["prompt_group"].duplicated(keep="first")
    metadata = metadata.loc[keep].reset_index(drop=True)
    dedup = frame.iloc[metadata["source_row"].to_numpy()].reset_index(drop=True)
    metadata["split"] = hash_partition(metadata["prompt_group"], config["seed"], config["split"]["test_fraction"])
    train_mask = metadata["split"].eq("train")
    if not train_mask.any() or train_mask.all():
        raise ValueError("La partición produjo un conjunto vacío.")
    tier_map = fit_tiers(dedup.loc[train_mask], tier_sizes=tuple(config["tier_sizes"]))
    labels = make_labels(dedup, tier_map, threshold=config["quality_threshold"])
    features = extract_features(metadata["prompt_text"])
    if tuple(features.columns) != tuple(FEATURE_NAMES) or features.isna().any().any():
        raise ValueError("Las 14 variables no cumplen el contrato.")
    output.mkdir(parents=True, exist_ok=True)
    # Features are stored alone to keep labels, outcomes and identifiers out of X.
    features.to_parquet(output / "features.parquet", index=False)
    metadata.drop(columns="prompt_text").to_parquet(output / "rows.parquet", index=False)
    labels.to_parquet(output / "labels.parquet", index=False)
    counts = {part: {"all": int(metadata["split"].eq(part).sum()),
                     "eligible": int((metadata["split"].eq(part) & labels["eligible"]).sum())}
              for part in ("train", "test")}
    write_json(output / "manifest.json", {
        "dataset_sha256": sha256_file(source), "configuration": config,
        "tier_map_fitted_on": "training rows only, including ineligible rows",
        "tier_map": tier_map, "feature_names": list(FEATURE_NAMES),
        "deduplicated_rows": len(metadata), "counts": counts,
        "final_test_evaluated": False,
        "files_sha256": {name: sha256_file(output / name) for name in artifact_names if name.endswith(".parquet")},
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow")}},
    })
    print(f"Preparación: {len(metadata)} prompts únicos, 14 variables; {output / 'manifest.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiment.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    source = args.data or PROJECT_ROOT / config["dataset"]["path"]
    frame = load_verified_pickle(source, config["dataset"]["sha256"])
    if frame.shape != (config["dataset"]["expected_rows"], config["dataset"]["expected_columns"]):
        raise ValueError(f"Forma inesperada: {frame.shape}.")
    if args.command == "audit":
        run_audit(frame, config, args.output or PROJECT_ROOT / "reports", source)
    else:
        run_prepare(frame, config, args.output or PROJECT_ROOT / "data/processed", source)


if __name__ == "__main__":
    main()
