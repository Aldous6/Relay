"""Paired development comparisons from saved policy decisions, without refit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256_file, write_json
from .development import ROOT, load_development_inputs
from .policy import evaluate_policy


def paired_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame,
                     replicates: int = 2000, seed: int = 42) -> dict:
    if not candidate.index.equals(reference.index) or not candidate.index.is_unique or len(candidate) < 2:
        raise ValueError("Las filas deben ser las mismas, únicas y estar alineadas.")
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError("Se requieren al menos 100 réplicas.")
    values = []
    for frame in (candidate, reference):
        if not {"success", "cost"}.issubset(frame.columns):
            raise ValueError("Faltan success o cost.")
        array = frame.loc[:, ["success", "cost"]].to_numpy(dtype=float)
        if not np.isfinite(array).all() or not np.isin(array[:, 0], [0, 1]).all() or (array[:, 1] < 0).any():
            raise ValueError("Resultados no válidos.")
        values.append(array)
    left, right = values
    difference = left - right
    rng = np.random.default_rng(seed)
    deltas, savings = [], []
    for start in range(0, replicates, 128):
        indices = rng.integers(0, len(candidate), size=(min(128, replicates-start), len(candidate)))
        deltas.append(difference[indices].mean(axis=1))
        reference_cost = right[indices, 1].mean(axis=1)
        candidate_cost = left[indices, 1].mean(axis=1)
        savings.extend((1-candidate_cost[reference_cost > 0]/reference_cost[reference_cost > 0]).tolist())
    samples = np.concatenate(deltas)
    mean = difference.mean(axis=0)
    return {
        "n": len(candidate), "replicates": replicates, "seed": seed,
        "success_rate_delta": float(mean[0]),
        "success_rate_delta_95ci": np.quantile(samples[:, 0], [0.025, 0.975]).tolist(),
        "cost_per_1000_delta": float(mean[1] * 1000),
        "cost_per_1000_delta_95ci": (np.quantile(samples[:, 1], [0.025, 0.975])*1000).tolist(),
        "relative_call_cost_saving": float(1-left[:, 1].mean()/right[:, 1].mean()) if right[:, 1].mean() > 0 else None,
        "relative_call_cost_saving_95ci": np.quantile(savings, [0.025, 0.975]).tolist() if savings else None,
        "scope": "paired prompt bootstrap on exploratory development; historical selected-call cost only; no router overhead or confirmatory superiority claim",
    }


def compare_run(run_dir: Path, replicates: int = 2000, reference_run_dir: Path | None = None) -> dict:
    report = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    _, scores, costs, metadata, _ = load_development_inputs(ROOT, report["configuration"])
    observed = {}
    for variant in report["variants"]:
        name = f"{variant}_observed_results.parquet"
        if sha256_file(run_dir / name) != report["files_sha256"][name]:
            raise ValueError("Los resultados guardados fueron modificados.")
        observed[variant] = pd.read_parquet(run_dir / name)
        if not metadata.loc[observed[variant].index, "development_split"].eq("validation").all():
            raise ValueError("La comparación contiene filas ajenas a validación de desarrollo.")
    reference_model = report["fixed_models_selected_from_fit"]["best_success"]
    comparisons = {}
    for variant, rows in observed.items():
        reference = evaluate_policy(pd.Series(reference_model, index=rows.index), scores.loc[rows.index], costs.loc[rows.index], report["configuration"]["quality_threshold"])["rows"]
        comparisons[f"{variant}_vs_best_fixed_fit"] = paired_bootstrap(rows, reference, replicates)
    if "structural" in observed and "lexical_hybrid" in observed:
        comparisons["lexical_hybrid_vs_structural"] = paired_bootstrap(observed["lexical_hybrid"], observed["structural"], replicates)
    if "semantic" in observed and "semantic_hybrid" in observed:
        comparisons["semantic_hybrid_vs_semantic"] = paired_bootstrap(observed["semantic_hybrid"], observed["semantic"], replicates)
    if reference_run_dir is not None:
        reference_report = json.loads((reference_run_dir / "summary.json").read_text(encoding="utf-8"))
        if (reference_report["dataset_sha256"] != report["dataset_sha256"] or
            reference_report["source_prepared_files_sha256"] != report["source_prepared_files_sha256"] or
            reference_report["configuration"]["quality_threshold"] != report["configuration"]["quality_threshold"]):
            raise ValueError("Las ejecuciones usan datos u objetivos diferentes.")
        for reference_variant in reference_report["variants"]:
            name = f"{reference_variant}_observed_results.parquet"
            if sha256_file(reference_run_dir / name) != reference_report["files_sha256"][name]:
                raise ValueError("Los resultados de referencia fueron modificados.")
            reference_rows = pd.read_parquet(reference_run_dir / name)
            for variant, rows in observed.items():
                comparisons[f"{variant}_vs_reference_{reference_variant}"] = paired_bootstrap(rows, reference_rows, replicates)
    result = {"kind": "exploratory paired development comparisons", "fixed_reference_model": reference_model,
              "outer_test_evaluated": False, "routellm_evaluated": False, "comparisons": comparisons}
    write_json(run_dir / "paired_comparisons.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(compare_run(args.run_dir, args.replicates, args.reference_run), indent=2))


if __name__ == "__main__":
    main()
