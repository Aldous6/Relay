"""Render frozen v7 development evidence without refitting or deserializing models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTROL_LABELS = {
    "bert101": "RouteLLM BERT publicado · 101 puntos",
    "ensemble_v4": "Relay · ensamble v4",
    "relay_v6": "Relay v6 · política archivada",
    "bert16_v6": "RouteLLM BERT · selección archivada de v6 (16 puntos)",
}
COMPARISON_LABELS = {"fixed_strong": "Fuerte fijo", **CONTROL_LABELS,
                     "global_control": "Ablación global v7 · sin vecinos"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def percentage(value):
    return "no disponible" if value is None else f"{100 * value:.2f} %"


def interval(values, scale=1, suffix="", digits=2):
    return ("no disponible" if values is None else
            f"[{values[0] * scale:+.{digits}f}, {values[1] * scale:+.{digits}f}]{suffix}")


def verify_file(directory, name, expected):
    path = directory / name
    if (not isinstance(name, str) or Path(name).name != name or
            not path.resolve().is_relative_to(directory) or digest(path) != expected):
        raise ValueError(f"Artefacto alterado o ruta inválida: {name}")
    return path


def validate_metrics(metric, n, quality):
    if (metric["n_prompts"] != n or metric["n_routed"] != n or metric["n_abstained"] != 0 or
            type(metric["n_successes"]) is not int or not 0 <= metric["n_successes"] <= n or
            metric["quality_score_threshold"] != quality):
        raise ValueError("Población, cobertura u objetivo incompatibles.")
    for key in ("success_rate", "total_cost", "cost_per_1000"):
        if not math.isfinite(metric[key]) or metric[key] < 0:
            raise ValueError("Métricas no finitas o negativas.")
    if (not math.isclose(metric["success_rate"], metric["n_successes"] / n, abs_tol=1e-12) or
            not math.isclose(metric["cost_per_1000"], 1000 * metric["total_cost"] / n, rel_tol=1e-12)):
        raise ValueError("Métricas inconsistentes con sus conteos o costos.")


def best_point(curve, floor):
    feasible = [point for point in curve if point["metrics"]["success_rate"] >= floor]
    return min(feasible, key=lambda p: (p["metrics"]["cost_per_1000"], -p["metrics"]["success_rate"], p["index"])) if feasible else None


def validate_report(directory):
    summary = read(directory / "summary.json")
    flags = ("outer_test_accessed", "independent_validation", "confirmation_of_superiority",
             "production_validation", "automatic_live_promotion", "paid_api_calls")
    if any(summary.get(key) is not False for key in flags):
        raise ValueError("El informe admite únicamente desarrollo histórico sin promoción ni confirmación.")
    for name, expected in summary["files_sha256"].items():
        verify_file(directory, name, expected)
    started = read(directory / "development_started.json")
    config = summary["configuration"]
    configuration_hash = object_hash(config)
    if (configuration_hash != summary["configuration_sha256"] or configuration_hash != started["configuration_sha256"] or
            config != started["configuration"] or digest(directory / "source_bundle.zip") != started["source_bundle_sha256"]):
        raise ValueError("La configuración o su snapshot fue modificado.")
    frozen = read(directory / "actions_frozen.json")
    if frozen["configuration_sha256"] != configuration_hash or frozen["selection_and_metrics_computed"] is not False:
        raise ValueError("Congelación de acciones incompatible.")
    for name, expected in frozen["files_sha256"].items():
        if summary["files_sha256"].get(name) != expected:
            raise ValueError("Las acciones finales difieren de las congeladas.")
    primary_name = summary["primary_candidate_id"]
    if (primary_name != "neighborhood_primary" or config["primary_candidate_id"] != primary_name or
            set(summary["candidates"]) != {primary_name, "global_control"} or
            set(summary["controls"]) != set(CONTROL_LABELS) or
            set(summary["comparisons"]) != set(COMPARISON_LABELS)):
        raise ValueError("Falta una candidata o referencia predeclarada.")
    if summary["partition_counts"] != config["expected_population"]:
        raise ValueError("Población diferente de la predeclarada.")
    n, quality = summary["partition_counts"]["validation"], config["quality_threshold"]
    fixed, controls, candidates = summary["fixed_models"], summary["controls"], summary["candidates"]
    metrics = [("Relay v7 · primaria con vecinos", candidates[primary_name]["selected_metrics"]),
               ("Ablación global v7 · sin vecinos", candidates["global_control"]["selected_metrics"]),
               *[(label, controls[name]["metrics"]) for name, label in CONTROL_LABELS.items()],
               ("Modelo fuerte fijo", fixed["strong"]), ("Modelo barato fijo", fixed["weak"])]
    for _, metric in metrics:
        validate_metrics(metric, n, quality)
    if summary["quality_floor"] != max(fixed["strong"]["success_rate"], controls["bert101"]["metrics"]["success_rate"]):
        raise ValueError("Piso de calidad inconsistente con BERT101 y fuerte fijo.")
    development_rows = pd.read_parquet(directory / "development_rows.parquet")
    validation_index = development_rows.index[development_rows.development_split.eq("validation")]
    if len(validation_index) != n or not validation_index.is_unique:
        raise ValueError("Filas de validación incompatibles.")
    pair = list(config["common_pair"].values())
    for name, candidate in candidates.items():
        curve = candidate["curve"]
        if len(curve) != config["quantiles"] or [point["index"] for point in curve] != list(range(config["quantiles"])):
            raise ValueError("Curva candidata incompleta o desordenada.")
        for point in curve:
            validate_metrics(point["metrics"], n, quality)
            if not math.isfinite(point["parameter"]):
                raise ValueError("Punto operativo no finito.")
        chosen = best_point(curve, summary["quality_floor"])
        expected = fixed["strong"] if chosen is None else chosen["metrics"]
        if (candidate["selected_point"] != chosen or candidate["selected_metrics"] != expected or
                candidate["development_gate_passed"] != (chosen is not None) or
                candidate["quality_floor_met"] != (expected["success_rate"] >= summary["quality_floor"])):
            raise ValueError("La selección candidata no reproduce la regla predeclarada.")
        rows = pd.read_parquet(directory / (name + "_observed_results.parquet"))
        selected = pd.read_parquet(directory / (name + "_selected_routes.parquet"))
        actions = pd.read_parquet(directory / (name + "_all_routes.parquet"))
        if (not rows.index.equals(validation_index) or not selected.index.equals(validation_index) or
                not actions.index.equals(validation_index) or actions.shape[1] != config["quantiles"] or
                not selected.selected_model.isin(pair).all() or not rows.executed.all() or
                not rows.selected_model.astype(str).equals(selected.selected_model.astype(str))):
            raise ValueError("Rutas seleccionadas desalineadas o inválidas.")
        expected_routes = (pd.Series(config["common_pair"]["strong"], index=validation_index) if chosen is None
                           else actions.iloc[:, chosen["index"]])
        if not selected.selected_model.astype(str).equals(expected_routes.astype(str)):
            raise ValueError("Las rutas no pertenecen al punto seleccionado.")
        if (int(rows.success.sum()) != expected["n_successes"] or not rows.success.eq(rows.score.ge(quality)).all() or
                not math.isclose(float(rows.cost.sum()), expected["total_cost"], rel_tol=1e-12)):
            raise ValueError("Los resultados por prompt no reproducen las métricas.")
        diagnostic = candidate["diagnostics"]
        if (int(selected.selected_model.eq(config["common_pair"]["weak"]).sum()) != diagnostic["weak_routes"] or
                expected["n_successes"] != fixed["strong"]["n_successes"] - diagnostic["strong_only_on_weak_routes"] + diagnostic["weak_only_on_weak_routes"]):
            raise ValueError("El diagnóstico no reproduce las rutas y el cambio de aciertos.")
        verify_file(directory, candidate["artifact"], candidate["artifact_sha256"])
    primary = candidates[primary_name]
    if any(summary[key] != primary[key] for key in ("selected_point", "selected_metrics", "development_gate_passed", "beats_bert_cost_on_development")):
        raise ValueError("El resumen no conserva el candidato primario.")
    cheaper = primary["selected_point"] is not None and primary["selected_metrics"]["cost_per_1000"] < controls["bert101"]["metrics"]["cost_per_1000"]
    if primary["beats_bert_cost_on_development"] != cheaper:
        raise ValueError("La comparación de costos con BERT101 es inconsistente.")
    bundle = read(directory / "bundle.json")
    expected_mode = "pairwise_utility" if primary["selected_point"] is not None else "fixed_strong"
    expected_cut = primary["selected_point"]["parameter"] if primary["selected_point"] is not None else None
    if (bundle["policy_mode"] != expected_mode or summary["policy_mode"] != expected_mode or bundle["tradeoff"] != expected_cut or
            bundle["strong"] != config["common_pair"]["strong"] or bundle["weak"] != config["common_pair"]["weak"] or
            bundle["artifact"] != primary["artifact"] or bundle["artifact_sha256"] != primary["artifact_sha256"] or
            bundle["independent_validation"] is not False):
        raise ValueError("El bundle no corresponde a la primaria congelada.")
    references = {"fixed_strong": fixed["strong"], **{name: value["metrics"] for name, value in controls.items()},
                  "global_control": candidates["global_control"]["selected_metrics"]}
    for name, comparison in summary["comparisons"].items():
        reference, selected = references[name], primary["selected_metrics"]
        if comparison["n"] != n or comparison["replicates"] != config["bootstrap_replicates"] or comparison["seed"] != config["seed"]:
            raise ValueError("Bootstrap distinto del predeclarado.")
        expected_deltas = {"success_rate_delta": selected["success_rate"] - reference["success_rate"],
                           "cost_per_1000_delta": selected["cost_per_1000"] - reference["cost_per_1000"],
                           "relative_call_cost_saving": 1-selected["total_cost"]/reference["total_cost"] if reference["total_cost"] else None}
        for key, expected in expected_deltas.items():
            actual, bounds = comparison[key], comparison[key + "_95ci"]
            if expected is None:
                if actual is not None:
                    raise ValueError("Reducción de costo indefinida con referencia de costo cero.")
            elif actual is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Diferencia pareada inconsistente con métricas absolutas.")
            if bounds is not None and (len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]):
                raise ValueError("Intervalo inválido.")
    return summary, bundle, metrics


def render_report(directory):
    directory = Path(directory).resolve()
    summary, bundle, metrics = validate_report(directory)
    config = summary["configuration"]
    reference_path = (ROOT / config["reference"]["selection_file"]).resolve()
    if not reference_path.is_relative_to(ROOT) or digest(reference_path) != config["reference"]["selection_sha256"]:
        raise ValueError("La procedencia del checkpoint BERT cambió.")
    reference = read(reference_path)["configuration"]
    primary = summary["candidates"][summary["primary_candidate_id"]]
    point, selected = primary["selected_point"], primary["selected_metrics"]
    bert_comparison = summary["comparisons"]["bert101"]
    if point is None:
        status = "Ningún punto de la primaria alcanzó el piso de calidad; el bundle usa el fuerte fijo. Este respaldo no cuenta como mejora de v7."
    elif primary["beats_bert_cost_on_development"]:
        status = (f"La primaria alcanzó el piso de calidad y redujo {percentage(bert_comparison['relative_call_cost_saving'])} "
                  "el costo histórico de llamadas frente a BERT101 en esta validación de desarrollo.")
    else:
        status = "La primaria alcanzó el piso de calidad, pero no consiguió menor costo histórico que BERT101; el objetivo competitivo no se alcanzó en esta ejecución."
    if bert_comparison["relative_call_cost_saving"] is not None and bert_comparison["relative_call_cost_saving"] < 0:
        status += f" Frente a BERT101, el costo fue {percentage(-bert_comparison['relative_call_cost_saving'])} mayor."
    lines = [
        "Relay v7 · evidencia local por similitud. Informe de desarrollo reutilizado.", "",
        status + " Esto no confirma superioridad, no inferioridad ni ahorro neto en producción.", "",
        f"La primaria obtuvo {selected['n_successes']:,}/{selected['n_prompts']:,} aciertos ({percentage(selected['success_rate'])}) con costo histórico {selected['cost_per_1000']:.6f} por 1,000 prompts. Éxito significa score guardado ≥ {config['quality_threshold']}; estos son porcentajes absolutos de éxito, no PGR ni CPT del paper.", "",
        f"Se conservaron {summary['partition_counts']['fit']:,} filas de ajuste, {summary['partition_counts']['calibration']:,} de calibración y {summary['partition_counts']['validation']:,} de validación del train exterior. El test histórico consumido quedó fuera de esta ejecución. La validación intervino antes en v2–v6 y volvió a usarse para seleccionar: sus métricas e intervalos son exploratorios.", "",
        f"Los candidatos mantienen las identidades históricas {bundle['strong']} y {bundle['weak']}. El resultado no permite sustituirlos por otros modelos ni extrapolarlo a tareas completas de agentes sin evaluación nueva.", "",
        f"V7 combina un predictor global multiclase con los {config['learner']['n_neighbors']} vecinos de mayor similitud coseno de una representación híbrida de TF-IDF y 14 atributos del prompt. Vocabulario, escalado, predictor global, banco de vecinos y costos se ajustan únicamente en fit. Los objetivos son strong_only, weak_only y equal; equal conserva tanto ambos aciertos como ambos fallos. El combinador logístico se ajusta solo en calibración. Las filas de calibración y validación nunca se incorporan al banco de vecinos.", "",
        "La idea está inspirada en la ponderación por similitud del [paper RouteLLM](https://arxiv.org/abs/2406.18665). V7 es una adaptación propia con top-k léxico y outcomes umbralizados; no reproduce el SW semántico con Bradley–Terry del artículo. La referencia BERT conserva el checkpoint publicado de [RouteLLM](https://github.com/lm-sys/RouteLLM) y sus predicciones archivadas, con selección local de umbral. Su supervisión externa difiere de los outcomes locales de Relay. No se evaluaron MF ni SW oficiales ni se igualaron los regímenes de entrenamiento.", "",
        f"La procedencia de BERT fija el checkpoint {reference['checkpoint']}, revisión {reference['checkpoint_revision']}, con máximo {reference['reference_tokenization']['max_length']} tokens y truncamiento {'activado' if reference['reference_tokenization']['truncation'] else 'desactivado'}. V7 procesa el texto completo con el vocabulario aprendido; esto no equivale a comprensión semántica completa. La posible contaminación del checkpoint externo respecto a benchmarks no está resuelta.", "",
        f"Se predeclararon una primaria con vecinos y una ablación global sin vecinos. Cada una usa {config['quantiles']} cuantiles de la puntuación de validación calculados sin outcomes. La puntuación es la ganancia prevista de éxito dividida por la diferencia positiva de costo previsto; se elige el fuerte únicamente cuando supera el corte. Las acciones se persistieron antes de calcular métricas. El corte no es una probabilidad.", "",
        f"BERT101 se seleccionó por menor costo entre puntos con éxito al menos igual al fuerte fijo. El piso exigido a v7 fue {percentage(summary['quality_floor'])}, el máximo entre BERT101 y fuerte fijo. Se eligió el punto de menor costo que alcanzaba ese piso, sin usar intervalos para seleccionar. Alcanzar el piso y ser más barato que BERT son resultados separados. El ensamble v4 se reeligió con ese piso; v6 y su BERT de 16 puntos conservan las políticas archivadas. La ablación global no sustituye a la primaria aunque resulte mejor.", "",
    ]
    if point is not None:
        lines += [f"Punto de la primaria: índice {point['index']}, corte {point['parameter']:.12g}. Modo del bundle: {bundle['policy_mode']}. Piso de calidad cumplido: {'sí' if primary['quality_floor_met'] else 'no'}.", ""]
    else:
        lines += [f"Modo del bundle: {bundle['policy_mode']}. Piso de calidad cumplido por el respaldo: {'sí' if primary['quality_floor_met'] else 'no'}.", ""]
    lines += ["| Política | Aciertos / prompts | Éxito | Costo histórico / 1,000 |",
              "| --- | ---: | ---: | ---: |"]
    for label, metric in metrics:
        lines.append(f"| {label} | {metric['n_successes']}/{metric['n_prompts']} | {percentage(metric['success_rate'])} | {metric['cost_per_1000']:.6f} |")
    lines += ["", "Todas las políticas seleccionan una llamada sobre los mismos prompts y el mismo par histórico. Los costos suman exclusivamente las llamadas seleccionadas guardadas en el dataset; excluyen representación, búsqueda, entrenamiento, reintentos y precios vigentes. No equivalen a facturación actual ni ahorro neto del harness.", "",
              "| Primaria v7 frente a | Diferencia de éxito | IC 95 % de diferencia | Reducción relativa del costo histórico | IC 95 % de reducción |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, label in COMPARISON_LABELS.items():
        comparison = summary["comparisons"][name]
        lines.append(f"| {label} | {100 * comparison['success_rate_delta']:+.2f} pp | {interval(comparison['success_rate_delta_95ci'], 100, ' pp')} | {percentage(comparison['relative_call_cost_saving'])} | {interval(comparison['relative_call_cost_saving_95ci'], 100, ' %')} |")
    lines += ["", f"Los intervalos usan {config['bootstrap_replicates']:,} réplicas de bootstrap pareado por prompt, semilla {config['seed']}, después de elegir con esta misma validación. No corrigen la selección repetida entre versiones, no son intervalos confirmatorios de superioridad y no validan no inferioridad. Una reducción negativa significa mayor costo de v7.", "",
              "| Primaria v7 frente a | Diferencia de costo histórico / 1,000 | IC 95 % de diferencia |",
              "| --- | ---: | ---: |"]
    for name, label in COMPARISON_LABELS.items():
        comparison = summary["comparisons"][name]
        lines.append(f"| {label} | {comparison['cost_per_1000_delta']:+.6f} | {interval(comparison['cost_per_1000_delta_95ci'], digits=6)} |")
    lines += ["", "| Candidato v7 | Rutas baratas | Aciertos perdidos frente al fuerte | Aciertos recuperados | Aciertos del barato seleccionado | MSE de ganancia prevista |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in ((summary["primary_candidate_id"], "Primaria con vecinos"), ("global_control", "Ablación global")):
        diagnostic = summary["candidates"][name]["diagnostics"]
        lines.append(f"| {label} | {diagnostic['weak_routes']} ({percentage(diagnostic['weak_fraction'])}) | {diagnostic['strong_only_on_weak_routes']} | {diagnostic['weak_only_on_weak_routes']} | {diagnostic['weak_successes_on_weak_routes']} | {diagnostic['expected_gain_mse']:.6f} |")
    lines += ["", "Aciertos perdidos son casos enviados al barato donde solo acertaba el fuerte; recuperados, casos donde solo acertaba el barato. Elegir barato cuando ambos fallan puede ahorrar costo, pero conserva el fallo. El MSE mide la diferencia prevista de éxito frente a la diferencia observada; no garantiza calibración individual ni detección de tareas desconocidas.", ""]
    latency = summary["single_request_latency"]
    if latency["n"]:
        lines += [f"En {latency['n']} solicitudes residentes, la mediana fue {latency['p50_seconds'] * 1000:.3f} ms y p95 {latency['p95_seconds'] * 1000:.3f} ms. En modo pairwise_utility incluye representación completa, búsqueda de vecinos sin caché, predicción y política; el respaldo fijo omite el predictor. Excluye carga inicial, generación e infraestructura. No permite comparar latencias con BERT medidas en otro proceso ni constituye un SLA.", ""]
    lines += ["| Candidato v7 | Ajuste (s) | Predicción por lote de validación (s) |",
              "| --- | ---: | ---: |"]
    for name, label in ((summary["primary_candidate_id"], "Primaria con vecinos"), ("global_control", "Ablación global")):
        candidate = summary["candidates"][name]
        lines.append(f"| {label} | {candidate['training_seconds']:.2f} | {candidate['batch_inference_seconds']:.2f} |")
    lines += ["", "El bundle conserva la primaria y delega generación al harness. La matriz dispersa y las clases almacenadas contienen información derivada de fit, aunque no guarden los prompts originales como strings. La licencia del autor sigue pendiente y la procedencia del dataset y de sus artefactos requiere revisión antes de distribuirlos; la licencia del código original no se adjudica a los datos.", "",
              "Una conclusión competitiva independiente requiere congelar el candidato y compararlo en familias y documentos nuevos del catálogo de destino, con criterio predeclarado, incertidumbre, tareas completas y recursos del router incluidos. Esta ejecución no hizo llamadas pagadas, no volvió a evaluar el test histórico y no promocionó automáticamente ninguna política.", "",
              "Evidencia: [resumen](summary.json), [configuración y snapshot](development_started.json), [acciones congeladas antes de métricas](actions_frozen.json), [bundle del SDK](bundle.json), [procedencia BERT](../routellm-bert-secondary-v1/selection.json) y [procedencia del informe](report_v7_provenance.json). El informe verifica los artefactos guardados y su coherencia sin reentrenar, deserializar modelos ni cambiar la selección.", ""]
    report_path = directory / "relay_v7.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    provenance = {"summary_sha256": digest(directory / "summary.json"), "report_sha256": digest(report_path),
                  "script_sha256": digest(Path(__file__)), "verified_artifacts": len(summary["files_sha256"]),
                  "training_executed": False, "pickle_deserialized": False, "generation_executed": False,
                  "selection_changed": False, "candidate_rows_and_paired_deltas_checked": True}
    (directory / "report_v7_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/relay-v7-development-v1")
    args = parser.parse_args(argv)
    print(render_report(args.run_dir))


if __name__ == "__main__":
    main()
