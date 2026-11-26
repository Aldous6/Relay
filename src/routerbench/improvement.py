"""Development-only v3 ablations. Never evaluate the consumed outer test."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .comparison import paired_bootstrap
from .data import sha256_file, write_json
from .development import ROOT, load_development_inputs
from .learner import SuccessCostRouter
from .policy import choose_models, evaluate_policy


def choose_development_point(points: list[dict], quality_floor: float) -> dict | None:
    if not np.isfinite(quality_floor) or not 0 <= quality_floor <= 1:
        raise ValueError("Piso de calidad inválido.")
    feasible = []
    for point in points:
        success, cost = point["metrics"]["success_rate"], point["metrics"]["cost_per_1000"]
        if not np.isfinite([success, cost, point["parameter"]]).all() or not 0 <= success <= 1 or cost < 0:
            raise ValueError("Punto de desarrollo inválido.")
        if success >= quality_floor:
            feasible.append(point)
    return min(feasible, key=lambda p: (p["metrics"]["cost_per_1000"], -p["metrics"]["success_rate"],
                                       p["candidate_id"], p["parameter"])) if feasible else None


def activate_full_fit(rows: pd.DataFrame) -> pd.DataFrame:
    """Activate the original fit rows omitted by the pilot, preserving all groups."""
    result = rows.copy()
    if not result.index.is_unique or not result["prompt_group"].is_unique or not result["split"].eq("train").all():
        raise ValueError("V3 admite únicamente prompts únicos del train exterior original.")
    if not result["development_split"].isin(["fit", "unused_fit_pilot", "calibration", "validation"]).all():
        raise ValueError("Partición de desarrollo desconocida.")
    result["original_development_split"] = result["development_split"]
    result.loc[result["development_split"].eq("unused_fit_pilot"), "development_split"] = "fit"
    return result


def _verify(root, relative, digest):
    path = root / relative
    if sha256_file(path) != digest:
        raise ValueError(f"Fuente modificada: {relative}")
    return path


def _decisions(spec, first, costs, pair, parameter):
    if spec["objective"] == "absolute":
        return choose_models(first.loc[:, pair], costs.loc[:, pair], parameter)["selected_model"]
    from .pairwise import choose_pairwise
    return choose_pairwise(first, costs, pair[0], pair[1], parameter)["selected_model"]


def _point(candidate_id, parameter, selected, scores, costs, quality):
    return {"candidate_id": candidate_id, "parameter": parameter,
            "weak_fraction": float(selected.eq(costs.columns[1]).mean()),
            "metrics": evaluate_policy(selected, scores, costs, quality)["metrics"]}


def run_improvement(root: Path, config: dict, output: Path) -> dict:
    if config["outer_test_evaluation_allowed"] is not False or config["automatic_live_promotion_allowed"] is not False:
        raise ValueError("Este experimento no permite evaluar test ni promover a producción.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Conserva los experimentos existentes; usa otra salida para otra revisión.")
    _verify(root, config["reference_split_run"]+"/summary.json", config["reference_split_summary_sha256"])
    reference = config["reference"]
    selection_path = _verify(root, reference["selection_file"], reference["selection_sha256"])
    probability_path = _verify(root, reference["development_probabilities_file"], reference["development_probabilities_sha256"])
    old_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if old_selection["test_outcomes_used_for_selection"] is not False:
        raise ValueError("La referencia no seleccionó exclusivamente con desarrollo.")
    inputs, scores, costs, rows, manifest = load_development_inputs(root, config)
    rows = activate_full_fit(rows)
    ids = {part: rows.index[rows["development_split"].eq(part)] for part in ("fit", "calibration", "validation")}
    if {part: len(value) for part, value in ids.items()} != config["expected_population"]:
        raise ValueError("La población v3 no corresponde al diseño fijado.")
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    quality = config["quality_threshold"]
    val_scores, val_costs = scores.loc[ids["validation"], pair], costs.loc[ids["validation"], pair]
    bert = pd.read_parquet(probability_path)
    if not bert.index.equals(ids["validation"]) or not np.isfinite(bert["probability"]).all() or not bert["probability"].between(0, 1).all():
        raise ValueError("Predicciones BERT desalineadas o inválidas.")
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(output / "development_started.json", {"configuration": config, "configuration_sha256": config_hash,
        "outer_test_access_allowed": False, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    rows.to_parquet(output / "development_rows.parquet")
    fixed_selected = pd.Series(pair[0], index=ids["validation"], dtype="string")
    fixed = evaluate_policy(fixed_selected, val_scores, val_costs, quality)
    floor = fixed["metrics"]["success_rate"] - config["selection"]["quality_loss_margin"]
    bert_curve = []
    for parameter in reference["threshold_grid"]:
        selected = pd.Series(np.where(bert["probability"] >= parameter, pair[0], pair[1]), index=bert.index, dtype="string")
        bert_curve.append(_point("routellm_bert", parameter, selected, val_scores, val_costs, quality))
    reference_point = choose_development_point(bert_curve, floor)
    if reference_point is None:
        raise ValueError("La grilla BERT no tiene punto factible; no se cambia el piso después de observar resultados.")
    bert_selected = pd.Series(np.where(bert["probability"] >= reference_point["parameter"], pair[0], pair[1]), index=bert.index, dtype="string")
    bert_evaluation = evaluate_policy(bert_selected, val_scores, val_costs, quality)
    bert_selected.to_frame("selected_model").to_parquet(output / "bert_selected_routes.parquet")
    all_points, candidates = [], {}
    started = time.perf_counter()
    for spec in config["candidates"]:
        name = spec["id"]
        print(f"V3 {name}: fit={len(ids['fit'])}, calibración={len(ids['calibration'])}, validación={len(ids['validation'])}.", flush=True)
        common = dict(representation=spec["representation"], seed=config["seed"], quality_threshold=quality,
                      max_features=config["lexical"]["max_features_per_block"], min_df=config["lexical"]["min_df"])
        tick = time.perf_counter()
        if spec["objective"] == "absolute":
            model = SuccessCostRouter(alpha=spec["alpha"], **common)
            model.fit(inputs.loc[ids["fit"]], scores.loc[ids["fit"]], costs.loc[ids["fit"]], inputs.loc[ids["calibration"]], scores.loc[ids["calibration"]])
            grid = config["absolute_probability_grid"]
        else:
            if spec.get("backend") == "hist_gradient_boosting":
                from .nonlinear import NonlinearRegretRouter
                model = NonlinearRegretRouter(strong=pair[0], weak=pair[1],
                    n_components=spec["n_components"], max_leaf_nodes=spec["max_leaf_nodes"],
                    max_iter=spec["max_iter"], min_samples_leaf=spec["min_samples_leaf"],
                    l2_regularization=spec["l2_regularization"], **common)
            else:
                from .pairwise import PairwiseRegretRouter
                model = PairwiseRegretRouter(strong=pair[0], weak=pair[1], alpha=spec["alpha"], **common)
            model.fit(inputs.loc[ids["fit"]], scores.loc[ids["fit"], pair], costs.loc[ids["fit"], pair],
                      inputs.loc[ids["calibration"]], scores.loc[ids["calibration"], pair])
            grid = config["pairwise_tradeoff_grid"]
        training_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        first, predicted_costs = model.predict_tables(inputs.loc[ids["validation"]])
        batch_seconds = time.perf_counter()-tick
        first.to_parquet(output / f"{name}_predictions.parquet")
        predicted_costs.to_parquet(output / f"{name}_predicted_costs.parquet")
        artifact = output / f"{name}.joblib"
        joblib.dump(model, artifact)
        curve = []
        for parameter in grid:
            selected = _decisions(spec, first, predicted_costs, pair, parameter)
            curve.append(_point(name, parameter, selected, val_scores, val_costs, quality))
        all_points.extend(curve)
        best = choose_development_point(curve, floor)
        candidate = {"specification": spec, "curve": curve, "selected_point": best, "training_seconds": training_seconds,
                     "validation_batch_inference_seconds": batch_seconds, "artifact": artifact.name,
                     "artifact_sha256": sha256_file(artifact)}
        if best:
            selected = _decisions(spec, first, predicted_costs, pair, best["parameter"])
            selected.to_frame("selected_model").to_parquet(output / f"{name}_selected_routes.parquet")
            evaluation = evaluate_policy(selected, val_scores, val_costs, quality)
            evaluation["rows"].to_parquet(output / f"{name}_observed_results.parquet")
            candidate["paired_development_comparison_vs_bert"] = paired_bootstrap(evaluation["rows"], bert_evaluation["rows"], config["bootstrap_replicates"], config["seed"])
            timings = []
            for row_id in ids["validation"][:min(config["latency_sample_size"], len(ids["validation"]))]:
                t = time.perf_counter()
                one, one_cost = model.predict_tables(inputs.loc[[row_id], ["prompt_text"]])
                _decisions(spec, one, one_cost, pair, best["parameter"])
                timings.append(time.perf_counter()-t)
            candidate["single_request_latency"] = {"n": len(timings), "p50_seconds": float(np.median(timings)),
                                                  "p95_seconds": float(np.quantile(timings, .95)),
                                                  "scope": "prompt representation + policy on this CPU, warm model, no cached feature vectors"}
        candidates[name] = candidate
        write_json(output / f"{name}_development.json", candidate)
        print(f"V3 {name}: punto factible={best}; entrenamiento {training_seconds:.1f}s.", flush=True)
    best = choose_development_point(all_points, floor)
    if best:
        winner = candidates[best["candidate_id"]]
        selection = {"candidate_id": best["candidate_id"], "point": best, "artifact": winner["artifact"],
                     "artifact_sha256": winner["artifact_sha256"], "requires_independent_evaluation": True}
    else:
        selection = {"candidate_id": "fixed_strong_fallback", "artifact": None, "requires_independent_evaluation": True}
    report = {"experiment_id": config["experiment_id"], "kind": "exploratory repeated development after the original outer test was consumed",
              "configuration": config, "configuration_sha256": config_hash, "source_prepared_files_sha256": manifest["files_sha256"],
              "partition_counts": {part: len(value) for part, value in ids.items()}, "quality_floor": floor,
              "fixed_strong": fixed["metrics"], "routellm_bert": {"curve": bert_curve, "selected_point": reference_point},
              "candidates": candidates, "selected_for_future_evaluation": selection,
              "outer_test_accessed_for_v3": False, "confirmation_of_superiority": False, "automatic_live_promotion": False,
              "source_reference_selection_sha256": reference["selection_sha256"], "source_reference_probabilities_sha256": reference["development_probabilities_sha256"],
              "elapsed_seconds": time.perf_counter()-started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "limitations": ["validation was reused for selection, including hyperparameters; intervals are optimistic development diagnostics",
                              "Relay uses local training while BERT has external supervision; this is not equal-supervision method isolation",
                              "historical selected-call cost excludes router compute, training and retries; no production savings claim",
                              "no independent untouched evaluation cohort is available in this run"],
              "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/relay_v3.json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=config.get("native_threads")):
        result = run_improvement(ROOT, config, ROOT / config["output_dir"])
    print(json.dumps(result["selected_for_future_evaluation"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
