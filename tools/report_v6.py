"""Render the completed v6 development evidence; never train, unpickle or generate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def percentage(value):
    return "desconocido" if value is None else f"{100 * value:.2f} %"


def interval(values, scale=1, suffix=""):
    return "no disponible" if values is None else f"[{values[0] * scale:+.2f}, {values[1] * scale:+.2f}]{suffix}"


def render_report(directory):
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    summary = read(summary_path)
    if any(summary.get(key) is not False for key in
           ("outer_test_accessed", "independent_validation", "confirmation_of_superiority", "production_validation", "paid_api_calls")):
        raise ValueError("Este informe solo admite desarrollo histórico sin afirmaciones confirmatorias.")
    for name, expected in summary["files_sha256"].items():
        path = directory / name
        if Path(name).name != name or not path.resolve().is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Artefacto alterado o ruta inválida: {name}")
    started = read(directory / "development_started.json")
    config_hash = hashlib.sha256(json.dumps(summary["configuration"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if config_hash != summary["configuration_sha256"] or config_hash != started["configuration_sha256"]:
        raise ValueError("Configuración o congelación modificada.")
    if digest(directory / "source_bundle.zip") != started["source_bundle_sha256"]:
        raise ValueError("Snapshot fuente modificado.")
    bundle = read(directory / "bundle.json")
    if (bundle["policy_mode"] != summary["policy_mode"] or bundle["strong"] != summary["configuration"]["common_pair"]["strong"] or
            bundle["weak"] != summary["configuration"]["common_pair"]["weak"] or
            Path(bundle["artifact"]).name != bundle["artifact"] or digest(directory / bundle["artifact"]) != bundle["artifact_sha256"]):
        raise ValueError("El bundle no corresponde al resultado congelado.")
    selected = summary["selected_metrics"]
    fixed = summary["fixed_models"]["strong"]
    controls = summary["additional_controls"]
    metrics = [("Relay v6 · política seleccionada", selected),
               ("Modelo fuerte fijo", fixed),
               ("Modelo barato fijo", summary["fixed_models"]["weak"]),
               ("RouteLLM BERT · grilla de 16", summary["routellm_bert"]["selected_point"]["metrics"]),
               ("RouteLLM BERT · umbral publicado 0.5", controls["published_bert_threshold_0_5"]["metrics"])]
    if "previous_bert_101" in controls:
        metrics.append(("RouteLLM BERT · control previo de 101", controls["previous_bert_101"]["selected_point"]["metrics"]))
    for _, metric in metrics:
        if (metric["n_prompts"] != selected["n_prompts"] or not math.isfinite(metric["cost_per_1000"]) or
                not math.isfinite(metric["success_rate"]) or not 0 <= metric["success_rate"] <= 1):
            raise ValueError("Métricas inválidas o poblaciones distintas.")
    point = summary["selected_point"]
    status = ("La política pasó el criterio de selección sobre las medias de desarrollo."
              if summary["development_gate_passed"] else
              "Ninguno de los puntos cumplió el criterio de selección; el bundle utiliza el modelo fuerte fijo. Este respaldo no es una mejora conseguida por v6.")
    bert_comparison = summary["comparisons"]["vs_routellm_bert"]
    if bert_comparison["relative_call_cost_saving"] is not None and bert_comparison["relative_call_cost_saving"] < 0:
        status += f" Frente al BERT seleccionado, Relay costó {percentage(-bert_comparison['relative_call_cost_saving'])} más; no se consiguió el objetivo de ahorrar frente a esa referencia."
    lines = [
        "Relay v6 · riesgo de perder calidad al elegir el modelo barato. Informe de desarrollo reutilizado.", "",
        status + " Este resultado no confirma superioridad, no inferioridad ni ahorro neto en producción.", "",
        f"Se conservaron {summary['partition_counts']['fit']:,} filas de ajuste, {summary['partition_counts']['calibration']:,} de calibración y {summary['partition_counts']['validation']:,} de validación del train exterior. El test histórico consumido no se volvió a evaluar. Esta validación ya había intervenido en versiones anteriores: los resultados e intervalos son exploratorios.", "",
        f"El par conserva sus identidades históricas: {bundle['strong']} frente a {bundle['weak']}. Éxito significa score guardado ≥ {summary['configuration']['quality_threshold']}. Este bundle no está validado para reemplazar esos candidatos por modelos locales o de API actuales.", "",
        "V6 reutiliza la representación híbrida existente de TF-IDF de palabras/caracteres y 14 atributos. Aprende dos probabilidades marginales calibradas por separado: que solo acierte el fuerte y que acierte el barato. La política solo permite el barato con riesgo previsto suficientemente bajo, éxito previsto suficientemente alto y costo previsto menor. Estas probabilidades no constituyen una garantía individual ni una distribución conjunta calibrada.", "",
        f"Se fijaron 16 combinaciones para Relay y 16 umbrales para BERT antes de entrenar. La grilla BERT incluye 0.5; se conserva además el control de 101 puntos de la búsqueda anterior. Igualar el número de puntos no iguala la supervisión ni elimina las diferencias del historial de búsqueda. Relay aprende con outcomes locales de RouterBench; el checkpoint BERT mantiene supervisión externa. No se evalúa RouteLLM MF.", "",
        f"El piso de selección v6 fue {percentage(summary['quality_floor'])}: el máximo entre el éxito del fuerte fijo y el BERT seleccionado en la grilla de 16. Se exigió costo estrictamente menor que el fuerte fijo y se eligió por menor costo entre medias factibles, sin utilizar intervalos para elegir. Piso cumplido por la política exportada: {'sí' if summary['quality_floor_met'] else 'no'}.", "",
    ]
    if point:
        lines += [f"Punto seleccionado: riesgo máximo {point['max_harm']}, éxito mínimo previsto del barato {point['min_weak_success']}. Modo del bundle: {bundle['policy_mode']}.", ""]
    lines += ["| Política | Aciertos / prompts | Éxito | Costo histórico / 1,000 |",
              "| --- | ---: | ---: | ---: |"]
    for label, metric in metrics:
        lines.append(f"| {label} | {metric['n_successes']}/{metric['n_prompts']} | {percentage(metric['success_rate'])} | {metric['cost_per_1000']:.6f} |")
    lines += ["", "Las políticas se miden con los mismos prompts y resultados históricos de sus llamadas seleccionadas. Los costos excluyen cómputo del router, entrenamiento, reintentos y precios vigentes. No son una medición de facturación ni ahorro total en un harness.", "",
              "| Relay v6 frente a | Diferencia de éxito | IC 95 % de diferencia | Reducción relativa del costo histórico | IC 95 % de reducción |",
              "| --- | ---: | ---: | ---: | ---: |"]
    labels = {"vs_fixed_strong": "Fuerte fijo", "vs_routellm_bert": "BERT · 16 puntos", "vs_previous_bert_101": "BERT · control previo 101"}
    for name, comparison in summary["comparisons"].items():
        lines.append(f"| {labels.get(name, name)} | {100 * comparison['success_rate_delta']:+.2f} pp | {interval(comparison['success_rate_delta_95ci'], 100, ' pp')} | {percentage(comparison['relative_call_cost_saving'])} | {interval(comparison['relative_call_cost_saving_95ci'], 100, ' %')} |")
    lines += ["", "Una reducción negativa significa mayor costo para Relay. Los intervalos proceden de bootstrap pareado por prompt después de seleccionar con esta misma validación: no tienen interpretación confirmatoria de superioridad.", "",
              "| Relay v6 frente a | Diferencia de costo histórico / 1,000 | IC 95 % de diferencia |",
              "| --- | ---: | ---: |"]
    for name, comparison in summary["comparisons"].items():
        bounds = comparison["cost_per_1000_delta_95ci"]
        lines.append(f"| {labels.get(name, name)} | {comparison['cost_per_1000_delta']:+.6f} | [{bounds[0]:+.6f}, {bounds[1]:+.6f}] |")
    diagnostic = summary["routing_diagnostics"]
    losses = diagnostic['strong_only_on_weak_routes']
    lines += ["", f"La política eligió el barato en {diagnostic['weak_routes']} prompts ({percentage(diagnostic['weak_route_fraction'])}). En esas rutas perdió {losses} {'acierto' if losses == 1 else 'aciertos'} que el fuerte sí conseguía y recuperó {diagnostic['weak_only_on_weak_routes']} que solo conseguía el barato; este último acertó {diagnostic['weak_successes_on_weak_routes']} de las llamadas seleccionadas. Brier marginal de daño: {diagnostic['brier_harm']:.6f}; Brier de éxito barato: {diagnostic['brier_weak_success']:.6f}.", ""]
    latency = summary["single_request_latency"]
    if latency["n"]:
        lines += [f"En {latency['n']} solicitudes con el modelo residente, la política exportada tuvo mediana {latency['p50_seconds'] * 1000:.3f} ms y p95 {latency['p95_seconds'] * 1000:.3f} ms. La medición incluye representación, predicción y decisión cuando el modo es guarded; el modo fijo omite el predictor. No incluye carga inicial, generación ni infraestructura de servicio, y no permite comparar latencia de BERT en otro proceso o medición.", ""]
    lines += [f"El ajuste tardó {summary['training_seconds']:.2f} segundos y la predicción por lote de validación {summary['batch_inference_seconds']:.2f} segundos en esta ejecución. No se realizaron llamadas pagadas ni se inició una aplicación.", "",
              "El siguiente criterio de promoción requiere congelar este candidato y evaluarlo en una cohorte independiente representativa de los modelos y tareas del harness, incluyendo el overhead completo. Reutilizar el test histórico o las mismas familias ya observadas no convierte este resultado en confirmación.", "",
              "Evidencia: [resumen](summary.json), [configuración y snapshot](development_started.json), [rutas persistidas antes de métricas](actions_frozen.json), [bundle para el SDK](bundle.json) y [procedencia del informe](report_v6_provenance.json). Este informe se deriva de los artefactos guardados, sin reentrenar ni modificar la selección.", ""]
    report_path = directory / "relay_v6.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    provenance = {"summary_sha256": digest(summary_path), "report_sha256": digest(report_path),
                  "script_sha256": digest(Path(__file__)), "verified_artifacts": len(summary["files_sha256"]),
                  "training_executed": False, "pickle_deserialized": False, "generation_executed": False}
    (directory / "report_v6_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/relay-v6-development-v1")
    args = parser.parse_args(argv)
    print(render_report(args.run_dir))


if __name__ == "__main__":
    main()
