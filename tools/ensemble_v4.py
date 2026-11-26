"""Evaluate a predeclared fixed blend of fitted regret routers on development."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from routerbench.comparison import paired_bootstrap
from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs
from routerbench.improvement import activate_full_fit, choose_development_point
from routerbench.pairwise import OUTCOME_CLASSES, pairwise_targets
from routerbench.policy import evaluate_policy
from report_v3 import paired_diagnostics
from report_v4 import comparison_table, pct, policy_table
from sweep_v3 import object_hash, quantile_actions, read_json, routes, utility_score, verify


def _manifest(directory: Path, summary: dict) -> None:
    for name, digest in summary["files_sha256"].items():
        verify(directory, name, digest)


def _source(root: Path, component: dict) -> tuple[Path, dict, dict]:
    directory = (root / component["run"]).resolve()
    if not directory.is_relative_to(root / "reports"):
        raise ValueError("Cada componente debe proceder de reports/ del proyecto.")
    summary = read_json(directory / "summary.json")
    for name in ("outer_test_accessed_for_v3", "confirmation_of_superiority", "automatic_live_promotion"):
        if summary[name] is not False:
            raise ValueError("Los componentes deben proceder exclusivamente de desarrollo.")
    if object_hash(summary["configuration"]) != summary["configuration_sha256"]:
        raise ValueError("La configuración fuente no coincide con su hash.")
    _manifest(directory, summary)
    candidate = summary["candidates"][component["candidate_id"]]
    if candidate["specification"]["objective"] != "pairwise":
        raise ValueError("La mezcla requiere componentes de las mismas tres clases pareadas.")
    if summary["files_sha256"].get(candidate["artifact"]) != candidate["artifact_sha256"]:
        raise ValueError("La huella del componente difiere del manifiesto fuente.")
    return directory, summary, candidate


def _probabilities(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if not frame.index.equals(index) or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Predicciones de componentes desalineadas o repetidas.")
    values = frame.loc[:, list(OUTCOME_CLASSES)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Las probabilidades de cada componente deben ser finitas y sumar uno.")
    result = pd.DataFrame(values, columns=OUTCOME_CLASSES, index=index)
    result["expected_quality_gain"] = result["strong_only"]-result["weak_only"]
    if "expected_quality_gain" in frame and not np.allclose(frame["expected_quality_gain"], result["expected_quality_gain"], atol=1e-8, rtol=0):
        raise ValueError("La ganancia almacenada no coincide con las clases pareadas.")
    return result


def _costs(frame: pd.DataFrame, index: pd.Index, pair: list[str]) -> pd.DataFrame:
    if not frame.index.equals(index) or not frame.index.is_unique or not frame.columns.is_unique or set(frame.columns) != set(pair):
        raise ValueError("Costos de componentes desalineados o con otro par.")
    values = frame.loc[:, pair].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Los costos deben ser finitos y no negativos.")
    return pd.DataFrame(values, columns=pair, index=index)


def _counts(scores: pd.DataFrame, rows: pd.DataFrame, pair: list[str], quality: float) -> dict:
    return {part: pairwise_targets(scores.loc[rows["development_split"].eq(part), pair], pair[0], pair[1], quality)
                  .value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict()
            for part in ("fit", "calibration")}


def _same_metrics(actual: dict, expected: dict) -> None:
    if set(actual) != set(expected):
        raise ValueError("El esquema de métricas BERT difiere de su referencia.")
    for name, value in actual.items():
        other = expected[name]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite([value, other]).all() or not np.isclose(value, other, rtol=1e-10, atol=1e-12):
                raise ValueError(f"La referencia BERT no reproduce su métrica {name}.")
        elif value != other:
            raise ValueError(f"La referencia BERT no reproduce su métrica {name}.")


def run_ensemble(root: Path, config: dict) -> dict:
    root = root.resolve()
    for name in ("outer_test_allowed", "weights_learned", "automatic_live_promotion"):
        if config[name] is not False:
            raise ValueError("La revisión exige mezcla fija, sin test ni promoción automática.")
    components = config["components"]
    if len(components) != 2 or [entry["weight"] for entry in components] != [.5, .5] or config["quantiles"] != 101:
        raise ValueError("Esta revisión fija dos componentes 50/50 y 101 cuantiles.")
    expected = [("reports/relay-v3-development-v1", "pairwise_hybrid_a5"),
                ("reports/relay-v4-development-v1", "svd64_hgb31")]
    if [(entry["run"], entry["candidate_id"]) for entry in components] != expected:
        raise ValueError("Los componentes no son los predeclarados para esta revisión.")
    output = (root / config["output_dir"]).resolve()
    component_directories = [(root / entry["run"]).resolve() for entry in components]
    if not output.is_relative_to(root / "reports") or any(output == path or output.is_relative_to(path) for path in component_directories):
        raise ValueError("La mezcla debe usar su propio directorio de informes.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("La mezcla ya se inició; conserva sus resultados y usa otra revisión.")
    sources = [_source(root, entry) for entry in components]
    base_config = sources[1][1]["configuration"]
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    quality = config["quality_threshold"]
    for _, source, _ in sources:
        if (source["configuration"]["common_pair"] != config["common_pair"] or
                source["configuration"]["quality_threshold"] != quality or
                source["configuration"]["selection"]["quality_loss_margin"] != config["quality_loss_margin"] or
                source["source_prepared_files_sha256"] != sources[1][1]["source_prepared_files_sha256"] or
                source["partition_counts"] != sources[1][1]["partition_counts"]):
            raise ValueError("Los componentes no comparten par, calidad, población y preparación.")
    operating_directory = root / "reports/relay-v4-operating-v1"
    operating = read_json(operating_directory / "summary.json")
    for name in ("outer_test_accessed", "models_refit", "confirmation_of_superiority", "automatic_live_promotion"):
        if operating[name] is not False:
            raise ValueError("La referencia operativa debe ser desarrollo exclusivo.")
    if (object_hash(operating["configuration"]) != operating["configuration_sha256"] or
            operating["source_summary_sha256"] != sha256_file(sources[1][0] / "summary.json") or
            operating["common_pair"] != config["common_pair"]):
        raise ValueError("La referencia operativa no procede del entrenamiento v4 fijado.")
    _manifest(operating_directory, operating)
    # The loader slices outer train before constructing outcomes.
    inputs, scores, costs, rows, manifest = load_development_inputs(root, base_config)
    rows = activate_full_fit(rows)
    counts = rows["development_split"].value_counts().to_dict()
    if counts != sources[1][1]["partition_counts"] or manifest["files_sha256"] != sources[1][1]["source_prepared_files_sha256"]:
        raise ValueError("La población actual no coincide con los componentes entrenados.")
    validation = rows.index[rows["development_split"].eq("validation")]
    if len(validation) != operating["n_validation"] or operating["source_prepared_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("La referencia BERT usa otra población o preparación.")
    for directory, source, _ in sources:
        stored_rows = pd.read_parquet(verify(directory, "development_rows.parquet", source["files_sha256"]["development_rows.parquet"]))
        if not stored_rows.index.equals(rows.index) or any(not stored_rows[name].equals(rows[name]) for name in ("prompt_group", "split", "development_split")):
            raise ValueError("Los componentes no contienen las mismas identidades de ajuste, calibración y validación.")
    targets = _counts(scores, rows, pair, quality)
    models, predictions, predicted_costs, evidence = [], [], [], []
    for entry, (directory, source, candidate) in zip(components, sources):
        name = entry["candidate_id"]
        prediction_name, cost_name = f"{name}_predictions.parquet", f"{name}_predicted_costs.parquet"
        prediction = _probabilities(pd.read_parquet(verify(directory, prediction_name, source["files_sha256"][prediction_name])), validation)
        estimate = _costs(pd.read_parquet(verify(directory, cost_name, source["files_sha256"][cost_name])), validation, pair)
        # Only trusted, checksummed, local experiment objects are deserialized.
        model = joblib.load(verify(directory, candidate["artifact"], candidate["artifact_sha256"]))
        if (tuple(model.models_) != tuple(pair) or model.quality_threshold != quality or model.target_counts_ != targets or
                model.fit_rows_ != counts["fit"] or model.calibration_rows_ != counts["calibration"]):
            raise ValueError("La semántica de clases o los soportes del modelo difieren de la población fijada.")
        models.append(model)
        predictions.append(prediction)
        predicted_costs.append(estimate)
        evidence.append({**entry, "source_summary_sha256": sha256_file(directory / "summary.json"),
            "artifact": candidate["artifact"], "artifact_sha256": candidate["artifact_sha256"],
            "predictions_sha256": source["files_sha256"][prediction_name],
            "predicted_costs_sha256": source["files_sha256"][cost_name], "target_counts": model.target_counts_})
    weights = [entry["weight"] for entry in components]
    blended = sum(weight*frame.loc[:, list(OUTCOME_CLASSES)] for weight, frame in zip(weights, predictions))
    blended["expected_quality_gain"] = blended["strong_only"]-blended["weak_only"]
    blended = _probabilities(blended, validation)
    combined_cost = _costs(sum(weight*frame for weight, frame in zip(weights, predicted_costs)), validation, pair)
    score, policy_specification = utility_score(blended, combined_cost, pair, "pairwise")
    parameters, actions = quantile_actions(score, config["quantiles"])
    from routerbench.ensemble import FrozenRegretEnsemble
    ensemble = FrozenRegretEnsemble(models, weights)
    replay = []
    for row_id in validation[:3]:
        actual, actual_cost = ensemble.predict_tables(inputs.loc[[row_id]])
        actual = _probabilities(actual, pd.Index([row_id], name=validation.name))
        actual_cost = _costs(actual_cost, actual.index, pair)
        if not np.allclose(actual.to_numpy(), blended.loc[[row_id]].to_numpy(), atol=1e-6, rtol=1e-5) or not np.allclose(actual_cost.to_numpy(), combined_cost.loc[[row_id]].to_numpy(), atol=1e-12, rtol=1e-7):
            raise ValueError("La inferencia del ensemble no reproduce la mezcla de predicciones guardadas.")
        replay.append({"source_row": int(row_id), "maximum_probability_error": float(np.abs(actual.to_numpy()-blended.loc[[row_id]].to_numpy()).max()),
                       "maximum_predicted_cost_error": float(np.abs(actual_cost.to_numpy()-combined_cost.loc[[row_id]].to_numpy()).max())})
    output.mkdir(parents=True, exist_ok=True)
    started = {"configuration": config, "configuration_sha256": object_hash(config), "components": evidence,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "outer_test_allowed": False,
               "models_refit": False, "weights_learned": False}
    write_json(output / "started.json", started)
    blended.to_parquet(output / "ensemble_probabilities.parquet")
    combined_cost.to_parquet(output / "ensemble_predicted_costs.parquet")
    score.to_frame().to_parquet(output / "ensemble_operating_scores.parquet")
    actions.to_parquet(output / "ensemble_actions.parquet")
    artifact = output / "weighted_ensemble.joblib"
    joblib.dump(ensemble, artifact)
    write_json(output / "inference_replay.json", {"scope": "first three validation prompts, fixed before any ensemble policy metrics; no generation",
               "n": len(replay), "records": replay, "models_refit": False})
    write_json(output / "actions_frozen.json", {"configuration_sha256": object_hash(config), "parameters": parameters,
               "class_order": list(OUTCOME_CLASSES), "actions_sha256": sha256_file(output / "ensemble_actions.parquet"),
               "probabilities_sha256": sha256_file(output / "ensemble_probabilities.parquet"),
               "predicted_costs_sha256": sha256_file(output / "ensemble_predicted_costs.parquet"),
               "all_actions_persisted_before_metrics": True, "strong_comparator": ">", "ties": "weak"})
    # All ensemble predictions and actions above are durable before scoring.
    val_scores, val_costs = scores.loc[validation, pair], costs.loc[validation, pair]
    reference_name = "routellm_bert_selected_routes.parquet"
    reference_routes = pd.read_parquet(verify(operating_directory, reference_name, operating["files_sha256"][reference_name]))["selected_model"]
    if not reference_routes.index.equals(validation):
        raise ValueError("Las acciones BERT no corresponden a los prompts de validación.")
    reference = evaluate_policy(reference_routes, val_scores, val_costs, quality)
    _same_metrics(reference["metrics"], operating["candidates"]["routellm_bert"]["selected_point"]["metrics"])
    fixed = evaluate_policy(pd.Series(pair[0], index=validation, dtype="string"), val_scores, val_costs, quality)
    floor = fixed["metrics"]["success_rate"]-config["quality_loss_margin"]
    if not np.isclose(floor, operating["quality_floor"], atol=1e-12, rtol=0):
        raise ValueError("El piso de calidad cambió respecto a la referencia BERT.")
    curve = []
    for position, parameter in enumerate(parameters):
        selected = routes(actions, position, pair)
        metric = evaluate_policy(selected, val_scores, val_costs, quality)["metrics"]
        curve.append({"candidate_id": "fixed_ensemble", "parameter": parameter, "grid_position": position,
                      "weak_fraction": float(selected.eq(pair[1]).mean()), "metrics": metric})
    best = choose_development_point(curve, floor)
    if best is None:
        raise ValueError("La grilla incluía el fuerte fijo, que debe cumplir el piso.")
    selected = routes(actions, best["grid_position"], pair)
    evaluation = evaluate_policy(selected, val_scores, val_costs, quality)
    selected.to_frame("selected_model").to_parquet(output / "ensemble_selected_routes.parquet")
    evaluation["rows"].to_parquet(output / "ensemble_selected_observed_results.parquet")
    comparison = paired_bootstrap(evaluation["rows"], reference["rows"], config["bootstrap_replicates"], config["seed"])
    comparison["scope"] = "post-selection diagnostic on repeatedly reused development validation; fixed blend, historical calls only; no confirmatory coverage or superiority claim"
    diagnostics = paired_diagnostics(blended, val_scores, pair[0], pair[1], quality)
    diagnostics["note"] = "A fixed weighted average of calibrated components does not guarantee calibration of the blend; reused-validation diagnostics only."
    diagnostic_pass = comparison["success_rate_delta_95ci"][0] > -config["quality_loss_margin"] and comparison["cost_per_1000_delta_95ci"][1] < 0
    selection = {"candidate_id": "fixed_ensemble", "parameter": best["parameter"], "point": best,
                 "artifact": artifact.name, "artifact_sha256": sha256_file(artifact),
                 "policy": policy_specification, "requires_independent_evaluation": True,
                 "automatic_live_promotion": False, "confirmation_of_superiority": False}
    write_json(output / "selection.json", selection)
    write_json(output / "curve.json", {"curve": curve, "selected_point": best})
    write_json(output / "diagnostics.json", diagnostics)
    lines = ["Relay v4 · ensemble fijo 50/50. Esta mezcla se declaró antes de completar los resultados v4 y no aprende pesos nuevos. Sus componentes ya fueron ajustados y seleccionados durante desarrollo; la mezcla y el corte no constituyen una evaluación independiente.",
        f"El par es {pair[0]} frente a {pair[1]}, con score de éxito ≥ {quality:g}. Se reutilizan {len(validation):,} prompts de validación; los componentes comparten {counts['fit']:,} filas de ajuste y {counts['calibration']:,} de calibración. Se verificaron preparación, orden de prompts, clases y soportes antes de mezclar.",
        "Se promedian las tres probabilidades y los costos previstos de los componentes con pesos 0.5/0.5. La ganancia esperada es P(strong_only)−P(weak_only), dividida entre el costo previsto adicional estrictamente positivo. Ambos fallos permanecen como equal. Las predicciones, el artefacto y las 101 acciones se guardaron antes de calcular métricas de la mezcla.",
        policy_table([("GPT-4 fijo", {"metrics": fixed["metrics"]}), ("RouteLLM BERT · búsqueda 101 v4", operating["candidates"]["routellm_bert"]["selected_point"]), ("Ensemble fijo 50/50", best)]),
        f"El piso de selección es {pct(floor)}, correspondiente al fuerte fijo menos {100*config['quality_loss_margin']:g} pp. Se seleccionó por menor costo con media de éxito al menos igual al piso, sin usar intervalos para escoger el corte.",
        comparison_table([("Ensemble fijo 50/50", comparison)], config["quality_loss_margin"]),
        f"El diagnóstico conjunto del punto ya elegido {'cumple descriptivamente' if diagnostic_pass else 'no cumple'} los límites de calidad y costo. Los intervalos son posteriores a selección sobre validación reutilizada: no prueban una victoria, ni garantizan cobertura nominal. No se reentrena, se ajustan pesos, se ensancha el margen ni se buscan otros intervalos para conseguir una conclusión favorable.",
        "Las dos representaciones y predictores se ejecutan al inferir; el costo de router y su latencia no se suponen gratuitos. Esta ejecución no mide una ventaja de latencia ni gasto neto de producción. Los costos de la tabla son llamadas históricas seleccionadas y excluyen entrenamiento, cómputo del router y reintentos.",
        "No se consulta ni reevalúa el test exterior. Una afirmación independiente requiere congelar artefacto, componentes y corte y evaluarlos sobre una cohorte nueva. La adaptación local de Relay y la supervisión externa de BERT siguen siendo distintas, y no se ha comprobado transferencia a familias nuevas o documentos productivos.",
        "Fuentes: [resumen](summary.json), [curva completa](curve.json), [acciones congeladas](actions_frozen.json), [reproducción de inferencia](inference_replay.json), [diagnósticos](diagnostics.json), [v4](../relay-v4-development-v1/summary.json) y [referencia BERT](../relay-v4-operating-v1/summary.json)."]
    report_path = output / "relay_v4_ensemble.md"
    report_path.write_text("\n\n".join(lines)+"\n", encoding="utf-8")
    report = {**started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_training_configuration": base_config, "source_prepared_files_sha256": manifest["files_sha256"],
        "source_operating_summary_sha256": sha256_file(operating_directory / "summary.json"),
        "source_bert_routes_sha256": operating["files_sha256"][reference_name],
        "partition_counts": counts, "common_pair": config["common_pair"], "quality_score_threshold": quality,
        "quality_floor": floor, "fixed_strong": fixed["metrics"], "routellm_bert": {"selected_point": operating["candidates"]["routellm_bert"]["selected_point"]},
        "candidates": {"fixed_ensemble": {"specification": {"objective": "pairwise", "backend": "frozen_fixed_ensemble"},
                                          "curve": curve, "selected_point": best, "artifact": artifact.name,
                                          "artifact_sha256": sha256_file(artifact), "paired_development_comparison_vs_bert": comparison}},
        "selected_for_future_evaluation": selection, "comparison_vs_bert": comparison,
        "conservative_criterion_diagnostic": {"point_already_selected": True, "used_for_selection": False,
                                              "both_bounds_satisfied": diagnostic_pass, "confirmation": False},
        "outer_test_accessed": False, "confirmation_of_superiority": False, "automatic_live_promotion": False,
        "all_actions_persisted_before_metrics": True,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ensemble_class_source_sha256": sha256_file(Path(inspect.getfile(FrozenRegretEnsemble))),
        "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/relay_v4_ensemble.json")
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        report = run_ensemble(args.root, read_json(args.config))
    print(json.dumps({"output": report["configuration"]["output_dir"],
                      "selected_for_future_evaluation": report["selected_for_future_evaluation"],
                      "comparison_vs_bert": report["comparison_vs_bert"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
