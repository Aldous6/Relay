"""Evaluate the frozen original tier classifier as an offline call policy.

The classifier and representatives never receive held-out outcomes. Sensitivity
changes only the success definition and descriptive labels, never the routes.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .comparison import paired_bootstrap
from .data import OFFICIAL_SHA256, load_verified_pickle, sha256_file, write_json
from .features import FEATURE_NAMES
from .labels import outcome_tables
from .policy import _numeric_tables, _threshold, evaluate_policy

ROOT = Path(__file__).resolve().parents[2]
QUALITY_THRESHOLD = 0.75
SENSITIVITY_THRESHOLDS = (0.5, 0.75, 1.0)
REPRESENTATIVE_RULE = "maximum train success rate at score >= 0.75; ties: lower train mean cost, then model name"


def _tier_map(tier_map: Mapping[str, int], columns) -> dict[str, int]:
    if set(tier_map) != set(columns):
        raise ValueError("El mapa de tiers debe coincidir con todos los modelos observados.")
    if any(not isinstance(tier, Integral) or isinstance(tier, bool) or tier not in (0, 1, 2)
           for tier in tier_map.values()) or set(tier_map.values()) != {0, 1, 2}:
        raise ValueError("El mapa debe contener los tres tiers enteros 0, 1 y 2.")
    return dict(tier_map)


def select_representatives(scores: pd.DataFrame, costs: pd.DataFrame,
                           tier_map: Mapping[str, int]) -> dict:
    """Apply the protocol's deterministic rule to outer-training rows only."""
    scores, costs = _numeric_tables(scores, costs, "train scores", "train costs")
    if not len(scores):
        raise ValueError("No se pueden elegir representantes sin filas de entrenamiento.")
    tier_map = _tier_map(tier_map, scores.columns)
    statistics = {model: {"tier": int(tier_map[model]),
                          "train_success_rate": float(scores[model].ge(QUALITY_THRESHOLD).mean()),
                          "train_mean_cost": float(costs[model].mean())}
                  for model in scores.columns}
    representatives = {
        tier: min((model for model in scores.columns if tier_map[model] == tier),
                  key=lambda model: (-statistics[model]["train_success_rate"],
                                     statistics[model]["train_mean_cost"], model))
        for tier in (0, 1, 2)
    }
    return {"representatives": representatives, "all_train_statistics": statistics,
            "n_train": len(scores), "threshold": QUALITY_THRESHOLD, "rule": REPRESENTATIVE_RULE}


def route_predicted_tiers(predicted_tiers: pd.Series, representatives: Mapping[int, str]) -> pd.Series:
    """Map classifier predictions to fixed representatives without outcomes."""
    if not isinstance(predicted_tiers, pd.Series) or not predicted_tiers.index.is_unique:
        raise ValueError("Las predicciones deben ser una Series con índice único.")
    if set(representatives) != {0, 1, 2} or any(not isinstance(name, str) or not name for name in representatives.values()):
        raise ValueError("Se necesita un representante por cada uno de los tres tiers.")
    if len(set(representatives.values())) != 3:
        raise ValueError("Los representantes de tiers distintos deben ser modelos distintos.")
    if any(not isinstance(tier, Integral) or isinstance(tier, (bool, np.bool_)) or tier not in (0, 1, 2)
           for tier in predicted_tiers):
        raise ValueError("El clasificador debe predecir tiers enteros 0, 1 o 2 sin valores ausentes.")
    return predicted_tiers.map(representatives).astype("string").rename("selected_model")


def retrospective_labels(scores: pd.DataFrame, costs: pd.DataFrame, tier_map: Mapping[str, int],
                         threshold: float) -> pd.DataFrame:
    """Descriptive cheapest-successful labels, including explicit failures."""
    target = _threshold(threshold, "threshold")
    scores, costs = _numeric_tables(scores, costs, "scores", "costs")
    tier_map = _tier_map(tier_map, scores.columns)
    eligible = scores.to_numpy() >= target
    has_eligible = eligible.any(axis=1)
    choices = np.where(eligible, costs.to_numpy(), np.inf).argmin(axis=1)
    names = scores.columns.to_numpy()
    selected = pd.Series(pd.array([names[position] if valid else pd.NA
                                   for position, valid in zip(choices, has_eligible)], dtype="string"), index=scores.index)
    return pd.DataFrame({"cheapest_model": selected, "tier": selected.map(tier_map).astype("Int64"),
                         "eligible": has_eligible}, index=scores.index)


def label_balance(labels: pd.DataFrame) -> dict:
    eligible = int(labels["eligible"].sum())
    return {"n_all": len(labels), "n_eligible": eligible, "n_ineligible": len(labels)-eligible,
            "tier_counts": {str(tier): int(labels["tier"].eq(tier).sum()) for tier in (0, 1, 2)},
            "tier_proportions_among_eligible": {
                str(tier): float(labels["tier"].eq(tier).sum()/eligible) if eligible else None
                for tier in (0, 1, 2)}}


def evaluate_frozen_routes(routes: pd.Series, scores: pd.DataFrame, costs: pd.DataFrame,
                           representatives: Mapping[int, str], tier_map: Mapping[str, int]) -> dict:
    """Re-score unchanged calls at all thresholds; no retraining or retuning."""
    results = {}
    for threshold in SENSITIVITY_THRESHOLDS:
        labels = retrospective_labels(scores, costs, tier_map, threshold)
        policy = evaluate_policy(routes, scores, costs, threshold)
        fixed = {str(tier): evaluate_policy(pd.Series(model, index=scores.index, dtype="string"), scores, costs, threshold)["metrics"]
                 for tier, model in representatives.items()}
        oracle = evaluate_policy(labels["cheapest_model"], scores, costs, threshold)
        results[str(threshold)] = {"policy": policy["metrics"], "fixed_representatives": fixed,
                                   "retrospective_oracle": oracle["metrics"], "label_balance": label_balance(labels)}
    return results


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 distinto del manifiesto: {path}")


def run_tier_policy(root: Path = ROOT, output: Path | None = None,
                    bootstrap_replicates: int = 5000, seed: int = 42) -> dict:
    """Evaluate only artifacts whose hashes match the completed v1 study."""
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / "reports/final-tier-policy-v1"
    protected = (root / "reports/final", root / "reports/final-tiers", root / "data")
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError("La política debe escribirse fuera de los datos y resultados finales existentes.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("El directorio de salida ya contiene archivos; no se sobreescribe una evaluación.")
    prepared = root / "data/processed"
    tiers_directory = root / "reports/final-tiers"
    manifest_path, summary_path = prepared / "manifest.json", tiers_directory / "summary.json"
    manifest, tiers = _read_json(manifest_path), _read_json(summary_path)
    if tiers["selection_used_test"] is not False or tiers["final_test_evaluated"] is not True:
        raise ValueError("Se requiere un modelo original congelado sin selección por test.")
    if manifest["dataset_sha256"] != OFFICIAL_SHA256 or tiers["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("El conjunto de datos no coincide con el estudio original.")
    if manifest["tier_map"] != tiers["tier_map"]:
        raise ValueError("El mapa de tiers difiere entre preparación y estudio original.")
    if manifest["configuration"]["quality_threshold"] != QUALITY_THRESHOLD:
        raise ValueError("P3 original usa la definición de éxito congelada en 0.75.")
    if list(tiers["feature_names"]) != list(FEATURE_NAMES) or list(manifest["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError("P3 original requiere exactamente las 14 variables estructurales.")
    for name in ("features.parquet", "rows.parquet", "labels.parquet"):
        checksum = manifest["files_sha256"][name]
        if tiers["prepared_files_sha256"][name] != checksum:
            raise ValueError(f"La preparación no coincide con el clasificador: {name}")
        _verify(prepared / name, checksum)
    artifact = tiers["artifacts"][tiers["selected_model"]]
    model_path = tiers_directory / artifact["file"]
    if model_path.parent != tiers_directory or model_path.suffix != ".joblib":
        raise ValueError("Ruta de modelo no autorizada en el manifiesto original.")
    _verify(model_path, artifact["sha256"])
    model = joblib.load(model_path)
    features, rows = pd.read_parquet(prepared / "features.parquet"), pd.read_parquet(prepared / "rows.parquet")
    if not features.index.equals(rows.index) or list(features.columns) != list(FEATURE_NAMES):
        raise ValueError("Variables y metadatos desalineados.")
    if not rows["source_row"].is_unique or not rows["split"].isin(["train", "test"]).all():
        raise ValueError("La partición preparada contiene identificadores ambiguos o splits desconocidos.")
    train_mask, test_mask = rows["split"].eq("train"), rows["split"].eq("test")
    if not train_mask.any() or test_mask.sum() < 2:
        raise ValueError("Se requieren entrenamiento no vacío y al menos dos prompts de test.")
    raw = load_verified_pickle(root / manifest["configuration"]["dataset"]["path"], manifest["dataset_sha256"])
    if (rows["source_row"] < 0).any() or (rows["source_row"] >= len(raw)).any():
        raise ValueError("Los identificadores preparados exceden el archivo verificado.")
    train_source = rows.loc[train_mask, "source_row"].to_numpy(dtype=int)
    test_source = rows.loc[test_mask, "source_row"].to_numpy(dtype=int)
    train_raw = raw.iloc[train_source].copy()
    train_raw.index = pd.Index(train_source, name="source_row")
    train_scores, train_costs = outcome_tables(train_raw)
    choice = select_representatives(train_scores, train_costs, manifest["tier_map"])
    representatives = choice["representatives"]
    x_test = features.loc[test_mask].copy()
    x_test.index = pd.Index(test_source, name="source_row")
    start = time.perf_counter()
    predicted_tiers = pd.Series(model.predict(x_test), index=x_test.index, name="predicted_tier")
    selected_models = route_predicted_tiers(predicted_tiers, representatives)
    prediction_seconds = time.perf_counter()-start
    routes = pd.concat([predicted_tiers, selected_models], axis=1)
    output.mkdir(parents=True, exist_ok=True)
    routes_path = output / "routes.parquet"
    routes.to_parquet(routes_path)
    # This checkpoint is durable before constructing or validating test outcomes.
    write_json(output / "routing_frozen.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "routes_sha256": sha256_file(routes_path),
        "source_tier_summary_sha256": sha256_file(summary_path), "prepared_manifest_sha256": sha256_file(manifest_path),
        "model_sha256": artifact["sha256"], "selected_classifier": tiers["selected_model"],
        "representative_selection": choice, "n_test": len(routes), "test_outcomes_consulted_for_routing": False,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS), "sensitivity_reuses_routes_and_representatives": True})
    # No test scores/costs are extracted until the calls above are persisted.
    test_raw = raw.iloc[test_source].copy()
    test_raw.index = routes.index
    scores, costs = outcome_tables(test_raw)
    evaluated = evaluate_policy(selected_models, scores, costs, QUALITY_THRESHOLD)
    labels = retrospective_labels(scores, costs, manifest["tier_map"], QUALITY_THRESHOLD)
    premium_model = representatives[2]
    premium = evaluate_policy(pd.Series(premium_model, index=scores.index, dtype="string"), scores, costs)
    fixed_results = {model: evaluate_policy(pd.Series(model, index=scores.index, dtype="string"), scores, costs)["metrics"]
                     for model in scores.columns}
    oracle = evaluate_policy(labels["cheapest_model"], scores, costs)
    eligible_mask = labels["eligible"]
    eligible_result = evaluate_policy(selected_models.loc[eligible_mask], scores.loc[eligible_mask], costs.loc[eligible_mask])["metrics"] if eligible_mask.any() else None
    eligible_premium = evaluate_policy(pd.Series(premium_model, index=scores.index[eligible_mask], dtype="string"), scores.loc[eligible_mask], costs.loc[eligible_mask])["metrics"] if eligible_mask.any() else None
    sensitivity = evaluate_frozen_routes(selected_models, scores, costs, representatives, manifest["tier_map"])
    train_balances = {str(threshold): label_balance(retrospective_labels(train_scores, train_costs, manifest["tier_map"], threshold))
                      for threshold in SENSITIVITY_THRESHOLDS}
    if (train_balances["0.75"]["n_eligible"] != tiers["train_eligible_rows"] or
            sensitivity["0.75"]["label_balance"]["n_eligible"] != tiers["test_eligible_rows"]):
        raise ValueError("Las etiquetas a 0.75 no reproducen la población del experimento original.")
    evaluated["rows"].to_parquet(output / "policy_outcomes.parquet")
    premium["rows"].to_parquet(output / "premium_outcomes.parquet")
    oracle["rows"].to_parquet(output / "oracle_outcomes.parquet")
    comparison = paired_bootstrap(evaluated["rows"], premium["rows"], replicates=bootstrap_replicates, seed=seed)
    comparison["scope"] = "secondary original-tier policy comparison on held-out prompts; paired prompt bootstrap; historical selected-call cost excludes router overhead and retries"
    report = {
        "experiment_id": "relay-original-tier-policy-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": manifest["dataset_sha256"], "prepared_files_sha256": manifest["files_sha256"],
        "source_tier_summary_sha256": sha256_file(summary_path), "source_model_sha256": artifact["sha256"],
        "selected_classifier": tiers["selected_model"], "classifier_retrained": False, "selection_used_test": False,
        "n_test": len(scores), "n_train": len(train_scores), "quality_score_threshold": QUALITY_THRESHOLD,
        "representative_selection": choice, "premium_reference_model": premium_model,
        "policy": evaluated["metrics"], "policy_on_eligible_subset": eligible_result,
        "premium_on_eligible_subset": eligible_premium,
        "premium_reference": premium["metrics"],
        "fixed_representatives": {str(tier): {"model": model, "metrics": fixed_results[model]}
                                  for tier, model in representatives.items()},
        "all_fixed_models": fixed_results,
        "comparison_vs_premium": comparison,
        "retrospective_oracle": {"metrics": oracle["metrics"],
                                 "definition": "cheapest stored-cost model attaining score >= threshold; abstain if none succeeds",
                                 "deployable": False, "uses_test_outcomes": True,
                                 "scope": "retrospective success-aware bound; impossible prompts abstain at zero selected-call cost; not a fair deployable cost policy"},
        "sensitivity": {"scope": "evaluation-only; same classifier, tier map, representatives and per-prompt calls at every threshold",
                        "retrained": False, "routes_sha256": sha256_file(routes_path), "thresholds": sensitivity},
        "label_balance": {"scope": "descriptive only, no relabeling or retraining of the fitted classifier",
                          "train": train_balances,
                          "test": {threshold: entry["label_balance"] for threshold, entry in sensitivity.items()}},
        "batch_prediction_seconds": prediction_seconds, "routing_frozen_before_test_outcomes": True,
        "predicted_tier_counts": {str(tier): int(predicted_tiers.eq(tier).sum()) for tier in (0, 1, 2)},
        "limitations": ["stored historical model-call costs exclude router compute, retries and training",
                        "a correct tier does not imply that the fixed representative succeeds",
                        "sensitivity changes evaluation thresholds only; it is not a retrained threshold-specific study",
                        "bootstrap resamples prompts and does not establish generalization to unseen benchmark families",
                        "no claims about production token savings, current pricing or local demo model quality"],
        "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()},
    }
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_tier_policy(args.root, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "n_test": result["n_test"],
                      "policy": result["policy"], "premium_reference_model": result["premium_reference_model"],
                      "comparison_vs_premium": result["comparison_vs_premium"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
