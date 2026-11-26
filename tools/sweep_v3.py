"""Additional exploratory utility-policy ablation on saved v3 predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.comparison import paired_bootstrap
from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs
from routerbench.improvement import activate_full_fit, choose_development_point
from routerbench.policy import evaluate_policy


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Se requiere un objeto JSON.")
    return value


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(directory, name, digest):
    if Path(name).name != name:
        raise ValueError("El manifiesto requiere archivos directos de su carpeta.")
    path = directory / name
    if sha256_file(path) != digest:
        raise ValueError(f"Artefacto modificado: {name}")
    return path


def quantile_actions(values: pd.Series, count: int = 101):
    """Strict score > threshold; endpoints include the two fixed policies."""
    if not isinstance(values, pd.Series) or values.empty or not values.index.is_unique:
        raise ValueError("Se requieren puntuaciones no vacías con índice único.")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or count != 101:
        raise ValueError("Esta revisión fija 101 cuantiles de puntuaciones finitas.")
    thresholds = np.quantile(array, np.linspace(0, 1, count))
    thresholds[0] = np.nextafter(array.min(), -np.inf)
    thresholds[-1] = np.nextafter(array.max(), np.inf)
    if not np.isfinite(thresholds).all():
        raise ValueError("Los extremos de la grilla no son representables como finitos.")
    actions = pd.DataFrame(array[:, None] > thresholds[None, :], index=values.index,
                           columns=[f"point_{i:03d}" for i in range(count)])
    assert actions.iloc[:, 0].all() and not actions.iloc[:, -1].any()
    return thresholds.tolist(), actions


def utility_score(predictions, predicted_costs, pair, objective):
    if objective not in {"absolute", "pairwise"}:
        raise ValueError("Objetivo no admitido.")
    if not predictions.index.equals(predicted_costs.index) or not predictions.index.is_unique:
        raise ValueError("Predicciones y costos desalineados.")
    if not predicted_costs.columns.is_unique or not predictions.columns.is_unique or not set(pair).issubset(predicted_costs.columns):
        raise ValueError("Columnas de modelos inválidas.")
    costs = predicted_costs.loc[:, pair].to_numpy(dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Costos previstos inválidos.")
    extra = costs[:, 0]-costs[:, 1]
    if (extra <= 0).any():
        raise ValueError("Esta ablación requiere costo previsto fuerte estrictamente mayor en cada prompt; no se omiten casos.")
    columns = pair if objective == "absolute" else ["strong_only", "weak_only", "equal"]
    values = predictions.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilidades inválidas.")
    if objective == "pairwise" and not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Las probabilidades pareadas no suman uno.")
    gain = values[:, 0]-values[:, 1]
    ratio = gain/extra
    if not np.isfinite(ratio).all():
        raise ValueError("Cociente de ganancia/costo no finito.")
    return pd.Series(ratio, index=predictions.index, name="operating_score"), {
        "objective": objective, "minimum_predicted_extra_cost": float(extra.min()),
        "maximum_predicted_extra_cost": float(extra.max()),
        "score_definition": "expected threshold-success gain divided by positive predicted extra historical call cost"}


def routes(actions, position, pair):
    return pd.Series(np.where(actions.iloc[:, position].to_numpy(), pair[0], pair[1]),
                     index=actions.index, dtype="string", name="selected_model")


def sweep(root, config):
    for name in ("outer_test_allowed", "independent_superiority_claim_allowed", "retraining_allowed"):
        if config[name] is not False:
            raise ValueError("Esta revisión solo permite desarrollo sin reentrenamiento.")
    if config["quantiles"] != 101 or config["require_predicted_strong_extra_cost_positive"] is not True:
        raise ValueError("No coincide la política predeclarada.")
    source = root / config["source_run"]
    output = root / config["output_dir"]
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("La revisión no se escribe sobre el experimento original.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("El barrido ya se inició; conserva los resultados existentes.")
    original_path = source / "summary.json"
    original = read_json(original_path)
    if original["outer_test_accessed_for_v3"] is not False or original["confirmation_of_superiority"] is not False:
        raise ValueError("La fuente no es desarrollo exclusivo.")
    original_config = original["configuration"]
    if object_hash(original_config) != original["configuration_sha256"]:
        raise ValueError("Configuración fuente modificada.")
    if config["quality_loss_margin"] != original_config["selection"]["quality_loss_margin"]:
        raise ValueError("No se modifica el margen del estudio base.")
    for name, digest in original["files_sha256"].items():
        verify(source, name, digest)
    _, scores, costs, rows, manifest = load_development_inputs(root, original_config)
    rows = activate_full_fit(rows)
    if manifest["files_sha256"] != original["source_prepared_files_sha256"]:
        raise ValueError("La preparación no coincide con el estudio base.")
    validation = rows.index[rows["development_split"].eq("validation")]
    if len(validation) != original["partition_counts"]["validation"]:
        raise ValueError("Cambió la población de validación.")
    pair = [original_config["common_pair"]["strong"], original_config["common_pair"]["weak"]]
    reference = original_config["reference"]
    if original["source_reference_probabilities_sha256"] != reference["development_probabilities_sha256"]:
        raise ValueError("Hash de referencia incoherente en el estudio fuente.")
    bert_path = root / reference["development_probabilities_file"]
    if sha256_file(bert_path) != reference["development_probabilities_sha256"]:
        raise ValueError("Predicciones de validación BERT modificadas.")
    bert = pd.read_parquet(bert_path)
    if not bert.index.equals(validation) or not bert["probability"].between(0, 1).all():
        raise ValueError("Predicciones BERT inválidas o desalineadas.")
    output.mkdir(parents=True, exist_ok=True)
    predeclared = config.get("scope") == "prespecified_v4_policy_ablation_before_v4_outcomes"
    label = "v4" if config["experiment_id"].startswith("relay-v4-") else "v3"
    started = {"configuration": config, "configuration_sha256": object_hash(config),
               "source_summary_sha256": sha256_file(original_path),
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "additional_search_after_stage_one_results": not predeclared,
               "policy_grid_predeclared_before_source_run": predeclared, "outer_test_allowed": False}
    write_json(output / "started.json", started)
    operating, frozen = {}, {}
    candidates = {"routellm_bert": {"score": bert["probability"], "specification": {"objective": "relative_preference_bert"}}}
    for name, candidate in original["candidates"].items():
        verify(source, candidate["artifact"], candidate["artifact_sha256"])
        if original["files_sha256"].get(candidate["artifact"]) != candidate["artifact_sha256"]:
            raise ValueError("Hash de modelo incoherente con el manifiesto fuente.")
        prediction = pd.read_parquet(verify(source, f"{name}_predictions.parquet", original["files_sha256"][f"{name}_predictions.parquet"]))
        predicted_costs = pd.read_parquet(verify(source, f"{name}_predicted_costs.parquet", original["files_sha256"][f"{name}_predicted_costs.parquet"]))
        if not prediction.index.equals(validation):
            raise ValueError("Un candidato contiene filas ajenas a validación.")
        score, specification = utility_score(prediction, predicted_costs, pair, candidate["specification"]["objective"])
        candidates[name] = {"score": score, "specification": specification,
                            "source_artifact": f"{config['source_run']}/{candidate['artifact']}",
                            "source_artifact_sha256": candidate["artifact_sha256"]}
    # Persist every candidate's complete action matrix before computing metrics.
    for name, candidate in candidates.items():
        parameters, actions = quantile_actions(candidate["score"], config["quantiles"])
        path = output / f"{name}_actions.parquet"
        actions.to_parquet(path)
        candidate["score"].to_frame().to_parquet(output / f"{name}_operating_scores.parquet")
        frozen[name] = {"parameters": parameters, "actions_sha256": sha256_file(path),
                        "unique_action_patterns": int(np.unique(actions.to_numpy().T, axis=0).shape[0]),
                        "specification": candidate["specification"]}
        operating[name] = {"parameters": parameters, "actions": actions}
    write_json(output / "actions_frozen.json", {**started, "candidates": frozen,
               "all_actions_persisted_before_metrics": True, "strong_comparator": "score > parameter; ties choose weak"})
    val_scores, val_costs = scores.loc[validation, pair], costs.loc[validation, pair]
    quality = original_config["quality_threshold"]
    fixed = evaluate_policy(pd.Series(pair[0], index=validation, dtype="string"), val_scores, val_costs, quality)
    floor = fixed["metrics"]["success_rate"]-config["quality_loss_margin"]
    selected, evaluations, all_relay_points = {}, {}, []
    for name, stored in operating.items():
        curve = []
        for position, parameter in enumerate(stored["parameters"]):
            chosen = routes(stored["actions"], position, pair)
            metric = evaluate_policy(chosen, val_scores, val_costs, quality)["metrics"]
            curve.append({"candidate_id": name, "parameter": parameter, "grid_position": position,
                          "weak_fraction": float(chosen.eq(pair[1]).mean()), "metrics": metric})
        best = choose_development_point(curve, floor)
        if best is None:
            raise ValueError("Los extremos debían incluir el fuerte fijo, que cumple el piso.")
        chosen = routes(stored["actions"], best["grid_position"], pair)
        evaluations[name] = evaluate_policy(chosen, val_scores, val_costs, quality)
        evaluations[name]["rows"].to_parquet(output / f"{name}_selected_observed_results.parquet")
        chosen.to_frame().to_parquet(output / f"{name}_selected_routes.parquet")
        selected[name] = {"curve": curve, "selected_point": best,
                          "unique_action_patterns": frozen[name]["unique_action_patterns"],
                          "specification": candidates[name]["specification"]}
        if name != "routellm_bert":
            all_relay_points.extend(curve)
        write_json(output / f"{name}_curve.json", selected[name])
    comparisons = {}
    for name in original["candidates"]:
        comparison = paired_bootstrap(evaluations[name]["rows"], evaluations["routellm_bert"]["rows"],
                                      config["bootstrap_replicates"], config["seed"])
        comparison["scope"] = "post-selection diagnostic on repeatedly reused validation; historical calls only; no confirmatory coverage or superiority claim"
        comparisons[name] = comparison
    best = choose_development_point(all_relay_points, floor)
    winner = candidates[best["candidate_id"]]
    selection = {"status": "selected_on_reused_development_requires_new_cohort", "candidate_id": best["candidate_id"],
                 "parameter": best["parameter"], "point": best, "common_pair": original_config["common_pair"],
                 "policy": config["policy"], "strong_comparator": ">", "ties": "weak",
                 "objective": winner["specification"]["objective"],
                 "score_definition": winner["specification"]["score_definition"],
                 "source_artifact": winner["source_artifact"], "source_artifact_sha256": winner["source_artifact_sha256"],
                 "configuration_sha256": object_hash(config), "source_summary_sha256": started["source_summary_sha256"],
                 "quality_floor": floor, "requires_independent_evaluation": True,
                 "automatic_live_promotion": False, "confirmation_of_superiority": False}
    write_json(output / "selection.json", selection)
    table = ["| Candidato con política común | Éxito | Costo histórico / 1,000 | Fracción débil |",
             "| --- | ---: | ---: | ---: |",
             f"| GPT-4 fijo | {100*fixed['metrics']['success_rate']:.2f} % | {fixed['metrics']['cost_per_1000']:.6f} | 0.00 % |"]
    for name, candidate in selected.items():
        point = candidate["selected_point"]
        table.append(f"| {name} | {100*point['metrics']['success_rate']:.2f} % | {point['metrics']['cost_per_1000']:.6f} | {100*point['weak_fraction']:.2f} % |")
    introduction = (f"Relay {label} · ablación exploratoria de política. Esta grilla de 101 puntos se declaró antes de entrenar los nuevos candidatos, junto a la comparación inicial de nueve puntos. La validación ya había sido reutilizada en etapas anteriores: la predeclaración no la convierte en una evaluación independiente. Se conservan todos los candidatos y los estudios anteriores."
                    if predeclared else "Relay v3 · ablación exploratoria adicional de política. Este barrido se decidió después de observar que los primeros puntos de desarrollo eran conservadores. Amplía de nueve a 101 puntos por router; no es una regla confirmatoria fijada antes de aquellos resultados. Se conservan todos los candidatos y el estudio anterior intacto.")
    paragraphs = [introduction,
                  f"Se reutilizan las mismas {len(validation):,} filas de validación del train exterior, sin reentrenar. El par es {pair[0]} frente a {pair[1]}. El piso de selección es {100*floor:.2f} % de éxito, correspondiente al fuerte fijo menos {100*config['quality_loss_margin']:g} pp.",
                  "La política Relay divide la ganancia esperada de éxito por el costo previsto adicional positivo. La ganancia es P(éxito fuerte)−P(éxito débil) en el objetivo absoluto y P(strong_only)−P(weak_only) en el pareado. Así se compara el objetivo bajo la misma política. Se comprobaron costos adicionales positivos para todos los prompts; no se eliminaron casos que violaran la condición.",
                  "Cada grilla usa 101 cuantiles de predicciones, sin outcomes para fijar sus posiciones, con extremos nextafter que incluyen siempre fuerte y siempre débil. Pueden existir cuantiles o acciones duplicadas y se informa su número efectivo. Para BERT y Relay se usa score>umbral, con igualdad enviada al débil; esta convención explícita difiere del ≥ usado en la primera referencia BERT. Algunos umbrales de utilidad pueden ser negativos: son cortes de esta puntuación operativa, no el λ no negativo de la política anterior. Las matrices completas de acciones se guardaron antes de calcular métricas.",
                  "\n".join(table)]
    ci_table = ["| Relay frente al BERT seleccionado aquí | Δéxito (pp), IC 95 % | Reducción relativa de costo, IC 95 % |",
                "| --- | ---: | ---: |"]
    for name, comparison in comparisons.items():
        q = comparison["success_rate_delta_95ci"]
        c = comparison["relative_call_cost_saving_95ci"]
        ci_table.append(f"| {name} | {100*comparison['success_rate_delta']:+.2f} [{100*q[0]:+.2f}, {100*q[1]:+.2f}] | {100*comparison['relative_call_cost_saving']:+.2f} % [{100*c[0]:+.2f}, {100*c[1]:+.2f}] |")
    paragraphs.extend(["\n".join(ci_table),
                       f"El candidato seleccionado solo para una evaluación futura es {selection['candidate_id']}, con corte {selection['parameter']:.12g}. Su artefacto fuente es {selection['source_artifact']}. La selección no cambia el router de la demo.",
                       "La misma validación eligió candidatos y cortes tras una búsqueda adicional guiada por resultados anteriores. Los intervalos pareados son descriptivos después de selección y no respaldan una victoria confirmatoria ni su cobertura nominal. No se descartan los candidatos negativos. Las probabilidades de BERT tienen un objetivo de preferencia externo y Relay recibió ajuste local; no se aísla el algoritmo bajo supervisión idéntica.",
                       "Los costos son llamadas históricas seleccionadas, sin cómputo del router, entrenamiento o reintentos. No hay aquí medición comparable de overhead de ambos ni ahorro monetario productivo. El test exterior anterior no se consulta ni se reevalúa. Para afirmar victoria se debe congelar un candidato y corte y usar una cohorte nueva con criterio compuesto de calidad y costo, como describe el protocolo v3.",
                       "Fuentes y curvas: [summary.json](summary.json), [selection.json](selection.json), [acciones congeladas](actions_frozen.json) y [etapa v3 original](../relay-v3-development-v1/summary.json). Los hashes de fuentes y artefactos quedan en el resumen."])
    (output / f"relay_{label}_operating.md").write_text("\n\n".join(paragraphs)+"\n", encoding="utf-8")
    report = {**started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_prepared_files_sha256": manifest["files_sha256"],
              "source_reference_probabilities_sha256": reference["development_probabilities_sha256"],
              "common_pair": original_config["common_pair"], "quality_score_threshold": quality,
              "n_validation": len(validation), "quality_floor": floor, "fixed_strong": fixed["metrics"],
              "candidates": selected, "comparisons_vs_bert": comparisons, "selected_for_future_evaluation": selection,
              "source_artifacts": {name: {key: candidate[key] for key in ("source_artifact", "source_artifact_sha256")}
                                   for name, candidate in candidates.items() if name != "routellm_bert"},
              "all_actions_persisted_before_metrics": True, "outer_test_accessed": False,
              "models_refit": False, "confirmation_of_superiority": False, "automatic_live_promotion": False,
              "script_sha256": sha256_file(Path(__file__).resolve()),
              "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/relay_v3_operating_curves.json")
    args = parser.parse_args()
    result = sweep(args.root.resolve(), read_json(args.config))
    print(json.dumps(result["selected_for_future_evaluation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
