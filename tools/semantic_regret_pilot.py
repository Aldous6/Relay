"""Fixed-encoder nonlinear pilot; train/calibration/validation only, no APIs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.comparison import paired_bootstrap
from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs
from routerbench.improvement import choose_development_point
from routerbench.nonlinear import NonlinearRegretRouter
from routerbench.policy import evaluate_policy
from routerbench.semantic_dense import DenseSemanticRepresentation
from report_v3 import paired_diagnostics
from sweep_v3 import object_hash, quantile_actions, read_json, routes, utility_score, verify


def run(resume=False):
    config_path = ROOT / "configs/relay_v5_semantic_pilot.json"
    cfg = read_json(config_path)
    if cfg["outer_test_allowed"] is not False or cfg["automatic_live_promotion"] is not False:
        raise ValueError("El piloto solo admite desarrollo sin promoción automática.")
    output = ROOT / cfg["output_dir"]
    has_previous = output.exists() and any(output.iterdir())
    if has_previous and not resume:
        raise FileExistsError("Conserva el piloto existente; no se sobreescribe.")
    if resume:
        if (output / "summary.json").exists() or not (output / "started.json").exists():
            raise ValueError("Solo se reanuda una ejecución iniciada e incompleta.")
        if read_json(output / "started.json")["configuration_sha256"] != object_hash(cfg):
            raise ValueError("No se permite cambiar la configuración al reanudar.")
        recovery_path = output / "recovery_manifest.json"
        if not recovery_path.exists():
            write_json(recovery_path, {"reason": "Runtime removed from live object during pickle; metrics had not been calculated. Preserve completed predictions without retraining.",
                "interrupted_source_archive_sha256": sha256_file(output / "interrupted_source.zip"),
                "completed_files_sha256": {path.name: sha256_file(path) for path in output.iterdir()
                    if path.suffix in (".parquet", ".joblib")}})
        for name, digest in read_json(recovery_path)["completed_files_sha256"].items():
            verify(output, name, digest)
    base = read_json(ROOT / cfg["source_config"])
    control_dir = ROOT / cfg["semantic_control_run"]
    if sha256_file(control_dir / "summary.json") != cfg["semantic_control_summary_sha256"]:
        raise ValueError("La referencia semántica cambió.")
    control = read_json(control_dir / "summary.json")
    reference_dir = ROOT / cfg["reference_operating_run"]
    reference = read_json(reference_dir / "summary.json")
    if reference["outer_test_accessed"] is not False or reference["confirmation_of_superiority"] is not False:
        raise ValueError("La referencia debe proceder de desarrollo.")
    for name, digest in reference["files_sha256"].items():
        verify(reference_dir, name, digest)
    inputs, scores, costs, rows, manifest = load_development_inputs(ROOT, base)
    # Preserve the original pilot identities. Do not activate unused_fit_pilot.
    ids = {part: rows.index[rows["development_split"].eq(part)] for part in ("fit", "calibration", "validation")}
    if {part: len(index) for part, index in ids.items()} != cfg["expected_population"]:
        raise ValueError("La población no corresponde al piloto semántico de 6,000 filas.")
    if manifest["files_sha256"] != control["source_prepared_files_sha256"] or manifest["files_sha256"] != reference["source_prepared_files_sha256"]:
        raise ValueError("Preparaciones incompatibles.")
    control_rows = pd.read_parquet(verify(control_dir, "development_rows.parquet", control["files_sha256"]["development_rows.parquet"])).set_index("source_row")
    if not rows["development_split"].equals(control_rows.loc[rows.index, "development_split"]):
        raise ValueError("Los grupos del piloto semántico no coinciden.")
    pair = [base["common_pair"]["strong"], base["common_pair"]["weak"]]
    quality = base["quality_threshold"]
    encoder_spec = read_json(ROOT / cfg["encoder_config"])
    output.mkdir(parents=True, exist_ok=True)
    if not has_previous:
        write_json(output / "started.json", {"configuration": cfg, "configuration_sha256": object_hash(cfg),
            "source_config_sha256": sha256_file(ROOT / cfg["source_config"]),
            "encoder_config_sha256": sha256_file(ROOT / cfg["encoder_config"]),
            "reference_summary_sha256": sha256_file(reference_dir / "summary.json"), "outer_test_allowed": False})
        rows.to_parquet(output / "development_rows.parquet")
    predictions = {}
    for spec in cfg["candidates"]:
        name = spec["id"]
        print(f"Piloto semántico {name}: 6000 fit, 4383 calibración, 4406 validación.", flush=True)
        model_path = output / f"{name}.joblib"
        files = [model_path, output / f"{name}_probabilities.parquet", output / f"{name}_predicted_costs.parquet", output / f"{name}_actions.parquet"]
        if resume and all(path.is_file() for path in files):
            recovered = read_json(output / "recovery_manifest.json")["completed_files_sha256"]
            for path in files:
                verify(output, path.name, recovered[path.name])
            model = joblib.load(model_path)
            first = pd.read_parquet(files[1])
            estimates = pd.read_parquet(files[2])
            if not first.index.equals(ids["validation"]) or not estimates.index.equals(ids["validation"]):
                raise ValueError("El checkpoint recuperado no corresponde a validación.")
            score, _ = utility_score(first, estimates, pair, "pairwise")
            parameters, actions = quantile_actions(score, cfg["quantiles"])
            pd.testing.assert_frame_equal(actions, pd.read_parquet(files[3]))
            predictions[name] = {"probabilities": first, "costs": estimates, "parameters": parameters, "actions": actions,
                "artifact": model_path.name, "artifact_sha256": recovered[model_path.name], "training_seconds": None,
                "target_counts": model.target_counts_, "validation_encoder_call": None,
                "reused_completed_checkpoint": True, "missing_timing_reason": "interrupted before timing metadata was persisted"}
            print(f"{name}: checkpoint verificado reutilizado sin reentrenar.", flush=True)
            continue
        transformer = DenseSemanticRepresentation(mode=cfg["semantic_mode"], encoder_spec=encoder_spec)
        model = NonlinearRegretRouter(strong=pair[0], weak=pair[1], max_leaf_nodes=spec["max_leaf_nodes"],
            max_iter=cfg["max_iter"], min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=cfg["l2_regularization"],
            seed=cfg["seed"], quality_threshold=quality, feature_transformer=transformer)
        tick = time.perf_counter()
        model.fit(inputs.loc[ids["fit"]], scores.loc[ids["fit"], pair], costs.loc[ids["fit"], pair],
                  inputs.loc[ids["calibration"]], scores.loc[ids["calibration"], pair])
        elapsed = time.perf_counter()-tick
        first, estimates = model.predict_tables(inputs.loc[ids["validation"]])
        first.to_parquet(output / f"{name}_probabilities.parquet")
        estimates.to_parquet(output / f"{name}_predicted_costs.parquet")
        encoder = model.transformer_.semantic_._get_encoder()
        validation_encoder_call = dict(encoder.last_call_)
        joblib.dump(model, model_path)
        score, _ = utility_score(first, estimates, pair, "pairwise")
        parameters, actions = quantile_actions(score, cfg["quantiles"])
        actions.to_parquet(output / f"{name}_actions.parquet")
        predictions[name] = {"probabilities": first, "costs": estimates, "parameters": parameters, "actions": actions,
            "artifact": model_path.name, "artifact_sha256": sha256_file(model_path), "training_seconds": elapsed,
            "target_counts": model.target_counts_, "validation_encoder_call": validation_encoder_call}
        print(f"{name}: predicciones y acciones guardadas; entrenamiento {elapsed:.1f}s.", flush=True)
    # Same-representation old linear model: control, not a new fitted candidate.
    cp = pd.read_parquet(verify(control_dir, "semantic_hybrid_probabilities.parquet", control["files_sha256"]["semantic_hybrid_probabilities.parquet"]))
    cc = pd.read_parquet(verify(control_dir, "semantic_hybrid_predicted_costs.parquet", control["files_sha256"]["semantic_hybrid_predicted_costs.parquet"]))
    if not cp.index.equals(ids["validation"]):
        raise ValueError("Control semántico desalineado.")
    score, _ = utility_score(cp, cc, pair, "absolute")
    parameters, actions = quantile_actions(score, cfg["quantiles"])
    actions.to_parquet(output / "semantic_linear_control_actions.parquet")
    predictions["semantic_linear_control"] = {"parameters": parameters, "actions": actions, "training_seconds": 0,
        "scope": "stored 6000-fit absolute-success semantic hybrid; objective and head differ"}
    write_json(output / "actions_frozen.json", {"all_actions_persisted_before_new_metrics": True,
        "parameters": {name: entry["parameters"] for name, entry in predictions.items()},
        "files_sha256": {path.name: sha256_file(path) for path in output.glob("*_actions.parquet")}})
    val_scores, val_costs = scores.loc[ids["validation"], pair], costs.loc[ids["validation"], pair]
    fixed = evaluate_policy(pd.Series(pair[0], index=ids["validation"]), val_scores, val_costs, quality)
    floor = fixed["metrics"]["success_rate"]-cfg["quality_loss_margin"]
    ref_name = "routellm_bert_selected_routes.parquet"
    ref_routes = pd.read_parquet(verify(reference_dir, ref_name, reference["files_sha256"][ref_name]))["selected_model"]
    bert = evaluate_policy(ref_routes, val_scores, val_costs, quality)
    candidates, all_points = {}, []
    for name, entry in predictions.items():
        curve = []
        for position, parameter in enumerate(entry["parameters"]):
            selected = routes(entry["actions"], position, pair)
            curve.append({"candidate_id": name, "parameter": parameter, "grid_position": position,
                "metrics": evaluate_policy(selected, val_scores, val_costs, quality)["metrics"]})
        point = choose_development_point(curve, floor)
        evaluated = evaluate_policy(routes(entry["actions"], point["grid_position"], pair), val_scores, val_costs, quality)
        evaluated["rows"].to_parquet(output / f"{name}_selected_observed_results.parquet")
        comparison = paired_bootstrap(evaluated["rows"], bert["rows"], cfg["bootstrap_replicates"], cfg["seed"])
        comparison["scope"] = "post-selection reused-validation diagnostic; no independent superiority claim"
        candidates[name] = {key: value for key, value in entry.items() if key not in ("probabilities", "costs", "parameters", "actions")}
        candidates[name].update(curve=curve, selected_point=point, comparison_vs_bert=comparison)
        if name != "semantic_linear_control":
            candidates[name]["calibration_diagnostics"] = paired_diagnostics(entry["probabilities"], val_scores, pair[0], pair[1], quality)
            all_points.extend(curve)
    selected = choose_development_point(all_points, floor)
    table = ["| Router | Éxito | Costo histórico / 1,000 |", "| --- | ---: | ---: |",
        f"| BERT · 101 puntos | {100*bert['metrics']['success_rate']:.2f} % | {bert['metrics']['cost_per_1000']:.6f} |"]
    for name, candidate in candidates.items():
        metric = candidate["selected_point"]["metrics"]
        table.append(f"| {name} | {100*metric['success_rate']:.2f} % | {metric['cost_per_1000']:.6f} |")
    text = ["Relay · piloto semántico no lineal. Los resultados reutilizan validación y no son una prueba independiente.",
        "Se conservan exactamente las 6,000 filas del piloto, 4,383 de calibración y 4,406 de validación. El encoder E5 está congelado, usa los pesos y la tokenización previamente fijados y cubre el texto mediante ventanas; su caché contiene atributos sin outcomes. Los nuevos árboles se ajustan solo en fit y su calibración es disjunta.",
        "\n".join(table),
        "Los puntos se eligen por menor costo entre medias de éxito que cumplen el piso del fuerte fijo menos 0.5 pp. Se guardaron todas las acciones antes de calcular métricas nuevas. El control semántico lineal usa las mismas filas y representación, pero también cambia el objetivo; no se aísla exclusivamente el clasificador. Los modelos v3/v4 de 20,368 filas son comparaciones de pipelines con cantidades de ajuste distintas.",
        "Los intervalos completos, soportes y diagnósticos están en summary.json. Son posteriores a selección y no confirman no inferioridad. El costo es histórico de llamadas y excluye encoder, entrenamiento y reintentos. No hay ahorro productivo ni promoción automática. No se consultó el test exterior para este piloto."]
    (output / "semantic_pilot.md").write_text("\n\n".join(text)+"\n", encoding="utf-8")
    result = {"configuration": cfg, "configuration_sha256": object_hash(cfg), "source_prepared_files_sha256": manifest["files_sha256"],
        "partition_counts": cfg["expected_population"], "quality_floor": floor, "bert": bert["metrics"],
        "candidates": candidates, "selected_for_future_evaluation": selected, "outer_test_accessed": False,
        "automatic_live_promotion": False, "confirmation_of_superiority": False,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", result)
    print(json.dumps({"selected": selected, "comparison": candidates[selected["candidate_id"]]["comparison_vs_bert"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(resume=arguments.resume)
