"""Complete the original fourteen-feature, three-tier experiment separately.

Model selection and permutation interpretation use outer training rows only.
The prepared, training-defined tier map is reused unchanged. The final test is
scored once after the CV winner and the interpretation procedure are frozen.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             make_scorer, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold

from .data import sha256_file, write_json
from .features import FEATURE_NAMES
from .models import build_models

ROOT = Path(__file__).resolve().parents[2]
LABELS = [0, 1, 2]


def classification_metrics(target, predicted) -> dict:
    """Keep all three classes in macro F1 even if one is never predicted."""
    truth = np.asarray(target)
    guess = np.asarray(predicted)
    if truth.ndim != 1 or guess.shape != truth.shape or not len(truth):
        raise ValueError("Las etiquetas y predicciones deben ser vectores no vacíos y alineados.")
    if not np.isin(truth, LABELS).all() or not np.isin(guess, LABELS).all():
        raise ValueError("Solo se admiten las tres etiquetas 0, 1 y 2.")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, guess, labels=LABELS, zero_division=0
    )
    return {
        "n": len(truth),
        "f1_macro": float(f1_score(truth, guess, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, guess)),
        "tier2_recall": float(recall[2]), "tier2_f1": float(f1[2]),
        "per_class": {str(label): {"precision": float(precision[label]), "recall": float(recall[label]),
                                   "f1": float(f1[label]), "support": int(support[label])} for label in LABELS},
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(truth, guess, labels=LABELS).tolist(),
    }


def load_tier_inputs(root: Path, config: dict, prepared_dir: str = "data/processed"):
    """Check every provenance file before loading the immutable prepared tables."""
    prepared = root / prepared_dir
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    expected_raw = config["dataset"]["sha256"]
    if manifest["dataset_sha256"] != expected_raw or sha256_file(root / config["dataset"]["path"]) != expected_raw:
        raise ValueError("El archivo original no coincide con la preparación y configuración.")
    required = ("features.parquet", "rows.parquet", "labels.parquet")
    for name in required:
        if name not in manifest.get("files_sha256", {}) or sha256_file(prepared / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Archivo preparado modificado o no fijado: {name}.")
    if "configuration" in manifest and manifest["configuration"]["quality_threshold"] != config["quality_threshold"]:
        raise ValueError("Las etiquetas preparadas usan otro umbral de calidad.")
    features = pd.read_parquet(prepared / "features.parquet")
    rows = pd.read_parquet(prepared / "rows.parquet")
    labels = pd.read_parquet(prepared / "labels.parquet")
    if tuple(features.columns) != FEATURE_NAMES or len(features) != len(rows) or len(labels) != len(rows):
        raise ValueError("La tabla de predictores no tiene exactamente las 14 columnas y filas esperadas.")
    if not {"source_row", "split", "prompt_group"}.issubset(rows.columns) or not {"tier", "eligible"}.issubset(labels.columns):
        raise ValueError("Faltan metadatos de filas o etiquetas.")
    if not rows["source_row"].is_unique or not rows["prompt_group"].is_unique or not rows["split"].isin(["train", "test"]).all():
        raise ValueError("Hay duplicados o particiones desconocidas en la preparación.")
    if not pd.api.types.is_integer_dtype(rows["source_row"]) or rows["source_row"].lt(0).any():
        raise ValueError("Los identificadores de fila original deben ser enteros no negativos.")
    if not pd.api.types.is_bool_dtype(labels["eligible"]) or labels["eligible"].isna().any():
        raise ValueError("eligible debe ser booleano completo.")
    if not labels["tier"].notna().equals(labels["eligible"]) or not labels.loc[labels["eligible"], "tier"].isin(LABELS).all():
        raise ValueError("Las etiquetas de tier e indicador de elegibilidad no coinciden.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Las 14 variables deben ser finitas.")
    source_index = pd.Index(rows["source_row"].to_numpy(), name="source_row")
    features.index = source_index
    rows.index = source_index
    labels.index = source_index
    for part in ("train", "test"):
        expected = manifest.get("counts", {}).get(part)
        mask = rows["split"].eq(part)
        if expected and (int(mask.sum()) != expected["all"] or int((mask & labels["eligible"]).sum()) != expected["eligible"]):
            raise ValueError("Los soportes de la partición difieren del manifiesto.")
    return features, rows, labels, manifest


def _permutation_on_training(model, features: pd.DataFrame, target: pd.Series,
                             seed: int, repeats: int = 5):
    """Interpret a fixed model with three training-only validation partitions."""
    if target.value_counts().min() < 3:
        raise ValueError("Se necesitan al menos tres ejemplos de cada clase para interpretación.")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1)
    scorer = make_scorer(f1_score, labels=[2], average="macro", zero_division=0)
    draws = []
    records = []
    assignments = []
    fold_ranks = []
    for fold, (fit_positions, validation_positions) in enumerate(splitter.split(features, target), 1):
        fitted = clone(model).fit(features.iloc[fit_positions], target.iloc[fit_positions])
        validation_x, validation_y = features.iloc[validation_positions], target.iloc[validation_positions]
        importance = permutation_importance(fitted, validation_x, validation_y, scoring=scorer,
                                            n_repeats=repeats, random_state=seed + fold, n_jobs=1)
        means = importance.importances_mean
        ordered_features = sorted(range(len(FEATURE_NAMES)), key=lambda index: (-means[index], FEATURE_NAMES[index]))
        rank_by_feature = {FEATURE_NAMES[index]: rank for rank, index in enumerate(ordered_features, 1)}
        ranks = pd.Series([rank_by_feature[name] for name in FEATURE_NAMES], index=FEATURE_NAMES)
        fold_ranks.append(ranks)
        draws.append(importance.importances)
        records.append({"fold": fold, "train_n": len(fit_positions), "validation_n": len(validation_positions),
                        "baseline_tier2_f1": float(scorer(fitted, validation_x, validation_y))})
        assignments.extend({"source_row": int(source), "interpretation_fold": fold} for source in validation_x.index)
    combined = np.concatenate(draws, axis=1)
    ranks = pd.concat(fold_ranks, axis=1)
    table = pd.DataFrame({"feature": list(FEATURE_NAMES),
                          "mean_f1_tier2_drop": combined.mean(axis=1),
                          "std_f1_tier2_drop": combined.std(axis=1, ddof=1),
                          "mean_rank_across_folds": ranks.mean(axis=1).to_numpy(),
                          "top3_fold_fraction": ranks.le(3).mean(axis=1).to_numpy()})
    table = table.sort_values(["mean_f1_tier2_drop", "feature"], ascending=[False, True]).reset_index(drop=True)
    evidence = {"population": "outer_training_eligible_only", "folds": 3, "repeats_per_fold": repeats,
                "scoring": "F1 of tier 2, zero_division=0", "fold_records": records,
                "top_three": table.head(3)["feature"].tolist(),
                "ranking_tie_break": "feature_name; zero-importance ties do not establish useful predictors",
                "features_with_positive_mean_drop": int(table["mean_f1_tier2_drop"].gt(0).sum()),
                "interpretation": "predictive reliance of the fixed model; not causal effects; fold variability is descriptive, not independent confidence intervals"}
    return table, pd.DataFrame(assignments), evidence


def run_tier_experiment(root: Path, config: dict, output: Path,
                        prepared_dir: str = "data/processed") -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Usa una carpeta de salida nueva; no se sobrescribe una evaluación final.")
    started = time.perf_counter()
    features, rows, labels, manifest = load_tier_inputs(root, config, prepared_dir)
    train_ids = rows.index[rows["split"].eq("train") & labels["eligible"]]
    test_ids = rows.index[rows["split"].eq("test") & labels["eligible"]]
    train_x = features.loc[train_ids]
    train_y = labels.loc[train_ids, "tier"].astype(int)
    folds = int(config.get("cv_folds", 5))
    if folds != 5 or set(train_y.unique()) != set(LABELS) or train_y.value_counts().min() < folds or len(test_ids) == 0:
        raise ValueError("La comparación requiere cinco folds, las tres clases con soporte suficiente y test elegible.")
    seed = int(config["seed"])
    models = build_models(seed)
    if set(models) != {"dummy", "length", "logistic", "random_forest"}:
        raise ValueError("La fábrica debe aportar exactamente los cuatro modelos preespecificados.")
    cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(train_x, train_y))
    output.mkdir(parents=True, exist_ok=True)
    assignments = []
    for fold, (_, validation_positions) in enumerate(cv, 1):
        assignments.extend({"source_row": int(source), "cv_fold": fold} for source in train_x.iloc[validation_positions].index)
    pd.DataFrame(assignments).to_parquet(output / "cv_folds.parquet", index=False)
    fold_records, predictions = [], []
    cv_summary = {}
    for name in sorted(models):
        print(f"Tiers v1: validación cruzada {name}.", flush=True)
        local_records = []
        for fold, (fit_positions, validation_positions) in enumerate(cv, 1):
            fitted = clone(models[name]).fit(train_x.iloc[fit_positions], train_y.iloc[fit_positions])
            prediction = fitted.predict(train_x.iloc[validation_positions])
            metrics = classification_metrics(train_y.iloc[validation_positions], prediction)
            record = {"model": name, "fold": fold, **metrics}
            local_records.append(record)
            fold_records.append(record)
            predictions.extend({"source_row": int(source), "model": name, "fold": fold,
                                "tier": int(target), "predicted_tier": int(guess)}
                               for source, target, guess in zip(train_x.iloc[validation_positions].index,
                                                               train_y.iloc[validation_positions], prediction))
        cv_summary[name] = {
            "metrics_mean": {metric: float(np.mean([item[metric] for item in local_records])) for metric in ("f1_macro", "balanced_accuracy", "tier2_recall", "tier2_f1")},
            "metrics_std": {metric: float(np.std([item[metric] for item in local_records], ddof=1)) for metric in ("f1_macro", "balanced_accuracy", "tier2_recall", "tier2_f1")},
            "std_scope": "descriptive variability across correlated CV folds; not an independent-sample confidence interval",
        }
    winner = sorted(models, key=lambda name: (-cv_summary[name]["metrics_mean"]["f1_macro"], name))[0]
    pd.DataFrame(predictions).to_parquet(output / "cv_predictions.parquet", index=False)
    write_json(output / "cv_results.json", {"folds": fold_records, "summary": cv_summary})
    selection = {"selected_model": winner, "selected_by": "maximum training five-fold mean macro-F1; model-name tie break",
                 "selected_at_utc": datetime.now(timezone.utc).isoformat(), "outer_test_metrics_used": False,
                 "cv_summary": cv_summary, "configuration": config,
                 "dataset_sha256": manifest["dataset_sha256"], "prepared_files_sha256": manifest["files_sha256"]}
    write_json(output / "selection.json", selection)
    selection_sha256 = sha256_file(output / "selection.json")
    print(f"Tiers v1: elegido {winner}; importancia por permutación solo en train.", flush=True)
    permutation, permutation_folds, permutation_evidence = _permutation_on_training(models[winner], train_x, train_y, seed)
    permutation.to_parquet(output / "permutation_importance.parquet", index=False)
    permutation_folds.to_parquet(output / "permutation_folds.parquet", index=False)
    test_predictions = pd.DataFrame({"source_row": test_ids.to_numpy()})
    artifacts = {}
    for name in sorted(models):
        print(f"Tiers v1: ajuste completo y predicción final {name}.", flush=True)
        fitted = clone(models[name]).fit(train_x, train_y)
        test_predictions[name] = fitted.predict(features.loc[test_ids])
        artifact_name = "model.joblib" if name == winner else f"{name}.joblib"
        joblib.dump(fitted, output / artifact_name, compress=3)
        artifacts[name] = {"file": artifact_name, "sha256": sha256_file(output / artifact_name)}
    # Persist outcomes-independent decisions before computing any test metric.
    test_predictions.to_parquet(output / "test_predictions.parquet", index=False)
    test_y = labels.loc[test_ids, "tier"].astype(int)
    final_metrics = {name: classification_metrics(test_y, test_predictions[name].to_numpy()) for name in sorted(models)}
    support = {part: {str(label): int(labels.loc[rows["split"].eq(part) & labels["eligible"], "tier"].eq(label).sum())
                      for label in LABELS} for part in ("train", "test")}
    report = {
        "experiment_id": "relay-original-tier-experiment-final-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": winner, "selection_sha256": selection_sha256,
        "selection_used_test": False, "final_test_evaluated": True,
        "configuration": config, "dataset_sha256": manifest["dataset_sha256"],
        "prepared_files_sha256": manifest["files_sha256"],
        "tier_map": manifest.get("tier_map"),
        "tier_map_scope": "prepared model-to-tier map estimated in original outer train and reused unchanged",
        "feature_names": list(FEATURE_NAMES), "train_eligible_rows": len(train_ids), "test_eligible_rows": len(test_ids),
        "excluded_ineligible": {part: int((rows["split"].eq(part) & ~labels["eligible"]).sum()) for part in ("train", "test")},
        "support": support, "cv": cv_summary, "test": final_metrics,
        "primary_test_result": final_metrics[winner],
        "secondary_test_results": "remaining three models are predeclared references; do not reselect using test",
        "permutation_importance": permutation_evidence, "artifacts": artifacts,
        "policy_cost_quality_simulation_completed_here": False,
        "quality_threshold_sensitivity_completed_here": False,
        "new_benchmark_family_holdout_completed_here": False,
        "routellm_comparison_completed_here": False,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["files_sha256"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "summary.json", report)
    print(f"Tiers v1: test final {winner}, F1-macro={final_metrics[winner]['f1_macro']:.4f}.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--prepared-dir", default="data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final-tiers")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_tier_experiment(ROOT, config, args.output, args.prepared_dir)


if __name__ == "__main__":
    main()
