"""Development-only training for direct model selection; outer test is reserved."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .data import load_verified_pickle, sha256_file, write_json
from .features import FEATURE_NAMES, parse_prompt
from .labels import outcome_tables
from .learner import SuccessCostRouter
from .policy import choose_models, evaluate_policy

ROOT = Path(__file__).resolve().parents[2]


def development_partition(groups: pd.Series, seed: int = 42,
                          fit_fraction: float = 0.70, calibration_fraction: float = 0.15) -> pd.Series:
    if not np.isfinite([fit_fraction, calibration_fraction]).all() or not (0 < fit_fraction < 1 and 0 < calibration_fraction < 1 and fit_fraction + calibration_fraction < 1):
        raise ValueError("Las tres particiones internas deben tener fracciones positivas.")
    def assign(group):
        value = int(hashlib.sha256(f"development:{seed}:{group}".encode()).hexdigest(), 16) / 2**256
        return "fit" if value < fit_fraction else "calibration" if value < fit_fraction + calibration_fraction else "validation"
    return groups.map(assign)


def load_development_inputs(root: Path, config: dict):
    prepared = root / config["prepared_dir"]
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_sha256"] != config["dataset"]["sha256"]:
        raise ValueError("La preparación y la configuración usan datos distintos.")
    for name, checksum in manifest["files_sha256"].items():
        if sha256_file(prepared / name) != checksum:
            raise ValueError(f"Artefacto modificado: {name}.")
    rows = pd.read_parquet(prepared / "rows.parquet")
    features = pd.read_parquet(prepared / "features.parquet")
    if tuple(features.columns) != FEATURE_NAMES or len(rows) != len(features) or not rows["source_row"].is_unique:
        raise ValueError("La preparación no cumple el contrato de alineación.")
    train_mask = rows["split"].eq("train")
    train_rows = rows.loc[train_mask].copy()
    raw = load_verified_pickle(root / config["dataset"]["path"], config["dataset"]["sha256"])
    # Slice before inspecting scores, text or costs for this new experiment.
    train = raw.iloc[train_rows["source_row"].to_numpy()].copy()
    train.index = pd.Index(train_rows["source_row"].to_numpy(), name="source_row")
    inputs = features.loc[train_mask].copy()
    inputs.index = train.index
    inputs.insert(0, "prompt_text", train["prompt"].map(parse_prompt))
    train_rows.index = train.index
    scores, costs = outcome_tables(train)
    if scores.shape[1] != config["dataset"]["expected_models"]:
        raise ValueError("Número inesperado de modelos.")
    split = config["internal_split"]
    if split["strategy"] != "prompt_hash" or not np.isclose(sum(split[name] for name in ("fit_fraction", "calibration_fraction", "validation_fraction")), 1):
        raise ValueError("Configuración de partición interna inválida.")
    train_rows["development_split"] = development_partition(train_rows["prompt_group"], config["seed"], split["fit_fraction"], split["calibration_fraction"])
    if config.get("reference_split_run"):
        reference_dir = root / config["reference_split_run"]
        reference = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
        reference_rows_path = reference_dir / "development_rows.parquet"
        if sha256_file(reference_rows_path) != reference["files_sha256"]["development_rows.parquet"]:
            raise ValueError("Las particiones de referencia fueron modificadas.")
        if reference["dataset_sha256"] != manifest["dataset_sha256"] or reference["source_prepared_files_sha256"] != manifest["files_sha256"]:
            raise ValueError("La referencia usa otra preparación de datos.")
        reference_rows = pd.read_parquet(reference_rows_path).set_index("source_row")
        if not reference_rows.index.is_unique or set(reference_rows.index) != set(train_rows.index):
            raise ValueError("Las particiones de referencia no corresponden al train reservado.")
        if not reference_rows["development_split"].isin(["fit", "calibration", "validation", "unused_fit_pilot"]).all():
            raise ValueError("Partición de referencia desconocida.")
        if not train_rows["prompt_group"].equals(reference_rows.loc[train_rows.index, "prompt_group"]):
            raise ValueError("Las huellas de prompt no coinciden con la referencia.")
        train_rows["development_split"] = reference_rows.loc[train_rows.index, "development_split"]
    return inputs, scores, costs, train_rows, manifest


def calibration_diagnostics(probabilities: pd.DataFrame, scores: pd.DataFrame, threshold: float) -> dict:
    if not probabilities.index.equals(scores.index) or not probabilities.index.is_unique or set(probabilities.columns) != set(scores.columns):
        raise ValueError("Probabilidades y scores deben tener las mismas filas alineadas y modelos.")
    if probabilities.empty or not probabilities.columns.is_unique or not np.isfinite(probabilities.to_numpy()).all() or not probabilities.ge(0).all().all() or not probabilities.le(1).all().all():
        raise ValueError("Probabilidades inválidas.")
    if not np.isfinite(scores.to_numpy()).all() or not scores.ge(0).all().all() or not scores.le(1).all().all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Scores o umbral inválidos.")
    target = scores.loc[:, probabilities.columns].ge(threshold).astype(int)
    per_model = {}
    for model in probabilities:
        y = target[model].to_numpy()
        p = probabilities[model].to_numpy()
        per_model[model] = {"brier": float(brier_score_loss(y, p)),
                            "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
                            "observed_success": float(y.mean()), "mean_predicted_success": float(p.mean())}
    p_all, y_all = probabilities.to_numpy().ravel(), target.to_numpy().ravel()
    bins = []
    for index in range(10):
        mask = (p_all >= index / 10) & ((p_all < (index+1) / 10) if index < 9 else (p_all <= 1))
        bins.append({"lower": index/10, "upper": (index+1)/10, "n": int(mask.sum()),
                     "mean_prediction": float(p_all[mask].mean()) if mask.any() else None,
                     "observed_success": float(y_all[mask].mean()) if mask.any() else None})
    return {"per_model": per_model, "mean_brier": float(np.mean([value["brier"] for value in per_model.values()])),
            "reliability_bins_pooled_diagnostic_only": bins,
            "note": "Brier combines reliability and discrimination; fitted calibration does not guarantee reliability."}


def run_development(root: Path, config: dict, output: Path, max_fit_rows: int | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Usa una carpeta --output nueva; no se sobrescriben experimentos.")
    if max_fit_rows is not None and max_fit_rows < 2:
        raise ValueError("max_fit_rows debe ser al menos 2.")
    if max_fit_rows is not None and config.get("reference_split_run"):
        raise ValueError("No se permite cambiar el subconjunto de ajuste de la referencia congelada.")
    started = time.perf_counter()
    inputs, scores, costs, rows, source_manifest = load_development_inputs(root, config)
    fit_ids = rows.loc[rows["development_split"].eq("fit")].sort_values("prompt_group").index
    if max_fit_rows is not None and len(fit_ids) > max_fit_rows:
        fit_ids = rows.loc[fit_ids].sort_values("prompt_group").index[:max_fit_rows]
        rows.loc[rows["development_split"].eq("fit") & ~rows.index.isin(fit_ids), "development_split"] = "unused_fit_pilot"
    calibration_ids = rows.index[rows["development_split"].eq("calibration")]
    validation_ids = rows.index[rows["development_split"].eq("validation")]
    if min(len(fit_ids), len(calibration_ids), len(validation_ids)) < 2:
        raise ValueError("No hay suficientes filas en las particiones internas.")
    output.mkdir(parents=True, exist_ok=True)
    rows.drop(columns=["sample_id", "eval_name"], errors="ignore").to_parquet(output / "development_rows.parquet", index=False)
    validation_scores, validation_costs = scores.loc[validation_ids], costs.loc[validation_ids]
    fixed = {model: evaluate_policy(pd.Series(model, index=validation_ids), validation_scores, validation_costs, config["quality_threshold"])["metrics"] for model in scores}
    best_fixed_fit = sorted(scores.columns, key=lambda model: (-scores.loc[fit_ids, model].ge(config["quality_threshold"]).mean(), costs.loc[fit_ids, model].mean(), model))[0]
    cheapest_fixed_fit = sorted(scores.columns, key=lambda model: (costs.loc[fit_ids, model].mean(), model))[0]
    report = {
        "experiment_id": config["experiment_id"], "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "kind": "exploratory development; not a RouteLLM comparison or final result",
        "pilot_max_fit_rows": max_fit_rows, "dataset_sha256": source_manifest["dataset_sha256"],
        "source_prepared_files_sha256": source_manifest["files_sha256"],
        "partition_counts": rows["development_split"].value_counts().to_dict(),
        "fixed_model_baselines": fixed,
        "fixed_models_selected_from_fit": {"best_success": best_fixed_fit, "cheapest_mean_cost": cheapest_fixed_fit},
        "variants": {}, "outer_test_performance_evaluated": False,
        "routellm_official_evaluated": False,
        "production_capability_metadata_available": False,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
    }
    for variant in config["variants"]:
        print(f"Entrenando {variant}: {len(fit_ids)} ajuste, {len(calibration_ids)} calibración, {len(validation_ids)} validación.", flush=True)
        semantic = variant in ("semantic", "semantic_hybrid")
        transformer = None
        if semantic:
            from .semantic_representation import SemanticPromptRepresentation
            spec = json.loads((root / config["encoder_config"]).read_text(encoding="utf-8"))
            transformer = SemanticPromptRepresentation(mode=variant, encoder_spec=spec)
            representation = variant
        else:
            representation = {"structural": "structural", "lexical": "lexical", "lexical_hybrid": "hybrid"}[variant]
        router = SuccessCostRouter(representation=representation, seed=config["seed"], quality_threshold=config["quality_threshold"],
                                   max_features=config["lexical"]["max_features_per_block"], min_df=config["lexical"]["min_df"], feature_transformer=transformer)
        training_start = time.perf_counter()
        router.fit(inputs.loc[fit_ids], scores.loc[fit_ids], costs.loc[fit_ids], inputs.loc[calibration_ids], scores.loc[calibration_ids])
        training_seconds = time.perf_counter() - training_start
        inference_start = time.perf_counter()
        probabilities, predicted_costs = router.predict_tables(inputs.loc[validation_ids])
        batch_inference_seconds = time.perf_counter() - inference_start
        semantic_evidence = None
        if semantic:
            encoder = router.transformer_.encoder_
            coverage = pd.DataFrame(encoder.last_stats_, index=validation_ids)
            coverage.to_parquet(output / f"{variant}_coverage.parquet", index=True)
            semantic_evidence = {"encoder_spec": spec, "validation_encoder_call": encoder.last_call_,
                                 "validation_max_input_tokens": int(coverage["input_tokens"].max()),
                                 "validation_prompts_with_multiple_windows": int(coverage["n_windows"].gt(1).sum()),
                                 "validation_truncated_tokens": int(coverage["truncated_tokens"].sum()),
                                 "encoder_artifact_files": encoder.assets_manifest_["files"]}
        curves = []
        for probability_threshold in config["policy"]["probability_threshold_grid"]:
            routes = choose_models(probabilities, predicted_costs, probability_threshold)
            evaluation = evaluate_policy(routes["selected_model"], validation_scores, validation_costs, config["quality_threshold"])
            curves.append({"minimum_predicted_success": probability_threshold,
                           "fallback_rate": float(routes["decision"].eq("fallback").mean()), **evaluation["metrics"]})
        routes = choose_models(probabilities, predicted_costs, config["policy"]["min_success_probability"])
        evaluation = evaluate_policy(routes["selected_model"], validation_scores, validation_costs, config["quality_threshold"])
        artifact = output / f"{variant}.joblib"
        joblib.dump(router, artifact)
        probabilities.to_parquet(output / f"{variant}_probabilities.parquet", index=True)
        predicted_costs.to_parquet(output / f"{variant}_predicted_costs.parquet", index=True)
        routes.to_parquet(output / f"{variant}_routes.parquet", index=True)
        evaluation["rows"].to_parquet(output / f"{variant}_observed_results.parquet", index=True)
        # Single-request pilot includes representation and policy, no feature
        # cache or network. It measures this dataset/hardware, not a serving SLA.
        latency_sample_size = config.get("latency_sample_size", 100)
        latency_ids = list(validation_ids[:min(latency_sample_size, len(validation_ids))])
        if semantic:
            router.transformer_.use_cache = False
        for row_id in latency_ids[:min(10, len(latency_ids))]:
            p, c = router.predict_tables(inputs.loc[[row_id], ["prompt_text"]])
            choose_models(p, c, config["policy"]["min_success_probability"])
        elapsed = []
        for row_id in latency_ids:
            tick = time.perf_counter()
            p, c = router.predict_tables(inputs.loc[[row_id], ["prompt_text"]])
            choose_models(p, c, config["policy"]["min_success_probability"])
            elapsed.append((time.perf_counter() - tick) * 1000)
        if semantic:
            router.transformer_.use_cache = True
        report["variants"][variant] = {
            "training_seconds": training_seconds, "batch_validation_inference_seconds": batch_inference_seconds,
            "feature_count": router.transformer_.n_features_out_,
            "calibration_notes": router.calibration_notes_, "diagnostics": calibration_diagnostics(probabilities, validation_scores, config["quality_threshold"]),
            "predicted_cost_mae": float(np.abs(predicted_costs - validation_costs).to_numpy().mean()),
            "policy_at_configured_probability": {**evaluation["metrics"], "fallback_rate": float(routes["decision"].eq("fallback").mean()),
                                                  "selected_model_counts": routes["selected_model"].value_counts().to_dict()},
            "development_cost_quality_curve": curves,
            "single_request_latency_pilot_ms": {"n": len(elapsed), "p50": float(np.percentile(elapsed, 50)), "p95": float(np.percentile(elapsed, 95)),
                                                 "scope": "representation plus cost/success prediction and selection; warm process, batch=1, first validation rows, no truncation"},
            "artifact": artifact.name, "artifact_sha256": sha256_file(artifact),
        }
        if semantic_evidence is not None:
            report["variants"][variant]["semantic_evidence"] = semantic_evidence
        print(f"{variant}: éxito validación={evaluation['metrics']['success_rate']:.4f}, costo histórico/1000={evaluation['metrics']['cost_per_1000']:.6f}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["files_sha256"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/competitor.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/competitor-development-v1")
    parser.add_argument("--max-fit-rows", type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_development(ROOT, config, args.output, args.max_fit_rows)


if __name__ == "__main__":
    main()
