"""Describe completed nonlinear development runs without refitting or test use."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from routerbench.data import sha256_file
from routerbench.development import load_development_inputs
from routerbench.improvement import activate_full_fit
from report_v3 import paired_diagnostics, read_latency_evidence


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Se requiere un objeto JSON: {path}")
    return value


def finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("El informe no admite métricas no finitas.")
    return result


def pct(value) -> str:
    return f"{100*finite(value):.2f} %"


def pp(value) -> str:
    return f"{100*finite(value):+.2f} pp"


def interval(bounds, percentage_points=True) -> str:
    if len(bounds) != 2 or finite(bounds[0]) > finite(bounds[1]):
        raise ValueError("Intervalo inválido.")
    formatter = pp if percentage_points else lambda value: f"{finite(value):+.6f}"
    return f"[{formatter(bounds[0])}, {formatter(bounds[1])}]"


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verified_file(directory: Path, name: str, digest: str) -> Path:
    if Path(name).name != name:
        raise ValueError("El manifiesto solo admite archivos directos de su carpeta.")
    path = directory / name
    if sha256_file(path) != digest:
        raise ValueError(f"Artefacto modificado: {path}")
    return path


def verify_manifest(directory: Path, report: dict) -> None:
    for name, digest in report["files_sha256"].items():
        verified_file(directory, name, digest)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name+".part")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def policy_table(entries) -> str:
    rows = ["| Política y corte de desarrollo | Éxito | Costo histórico / 1,000 |",
            "| --- | ---: | ---: |"]
    for name, point in entries:
        if point is None:
            rows.append(f"| {cell(name)} · sin punto factible | — | — |")
        else:
            metric = point["metrics"]
            parameter = point.get("parameter")
            suffix = "" if parameter is None else f" · {finite(parameter):.8g}"
            rows.append(f"| {cell(name)}{suffix} | {pct(metric['success_rate'])} | {finite(metric['cost_per_1000']):.6f} |")
    return "\n".join(rows)


def comparison_table(entries, margin: float) -> str:
    rows = ["| Punto Relay frente al BERT de su búsqueda | Δéxito, IC 95 % | Δcosto / 1,000, IC 95 % | Reducción relativa de costo | Diagnóstico conjunto |",
            "| --- | ---: | ---: | ---: | --- |"]
    for name, comparison in entries:
        q_ci, c_ci = comparison["success_rate_delta_95ci"], comparison["cost_per_1000_delta_95ci"]
        # A diagnostic of the already selected point, never another selector.
        diagnostic = finite(q_ci[0]) > -margin and finite(c_ci[1]) < 0
        saving = comparison["relative_call_cost_saving"]
        savings_text = "no estimable" if saving is None else pct(saving)
        savings_ci = comparison.get("relative_call_cost_saving_95ci")
        if savings_ci is not None:
            savings_text += f" [{pct(savings_ci[0])}, {pct(savings_ci[1])}]"
        rows.append(f"| {cell(name)} | {pp(comparison['success_rate_delta'])}, {interval(q_ci)} | {finite(comparison['cost_per_1000_delta']):+.6f}, {interval(c_ci, False)} | {savings_text} | {'cumple descriptivamente' if diagnostic else 'no cumple'} |")
    return "\n".join(rows)


def build_report(root: Path = ROOT, directory: Path | None = None,
                 operating_directory: Path | None = None) -> dict:
    root = root.resolve()
    directory = (root / "reports/relay-v4-development-v1") if directory is None else directory.resolve()
    operating_directory = (root / "reports/relay-v4-operating-v1") if operating_directory is None else operating_directory.resolve()
    if not directory.is_relative_to(root / "reports") or not operating_directory.is_relative_to(root / "reports"):
        raise ValueError("Los resultados deben permanecer dentro de reports/ del proyecto.")
    summary_path, operating_path = directory / "summary.json", operating_directory / "summary.json"
    # Both summaries must be complete before any presentation output is written.
    summary, operating = read_json(summary_path), read_json(operating_path)
    for name in ("outer_test_accessed_for_v3", "confirmation_of_superiority", "automatic_live_promotion"):
        if summary[name] is not False:
            raise ValueError("El entrenamiento no conserva el contrato de desarrollo exclusivo.")
    for name in ("outer_test_accessed", "models_refit", "confirmation_of_superiority", "automatic_live_promotion"):
        if operating[name] is not False:
            raise ValueError("La ablación de política no conserva el contrato de desarrollo exclusivo.")
    config = summary["configuration"]
    if object_hash(config) != summary["configuration_sha256"] or config["outer_test_evaluation_allowed"] is not False:
        raise ValueError("Configuración del entrenamiento modificada o con test autorizado.")
    if object_hash(operating["configuration"]) != operating["configuration_sha256"]:
        raise ValueError("Configuración de política modificada.")
    if operating["source_summary_sha256"] != sha256_file(summary_path):
        raise ValueError("La ablación de política no procede del entrenamiento v4 actual.")
    verify_manifest(directory, summary)
    verify_manifest(operating_directory, operating)
    if set(operating["candidates"]) != set(summary["candidates"]) | {"routellm_bert"}:
        raise ValueError("Los candidatos de la ablación no coinciden con el entrenamiento.")
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    if operating["common_pair"] != config["common_pair"] or not math.isclose(finite(operating["quality_floor"]), finite(summary["quality_floor"]), abs_tol=1e-12):
        raise ValueError("El par o piso de calidad cambió entre las dos búsquedas.")
    # The loader selects outer-training rows before constructing outcome tables.
    _, scores, _, rows, manifest = load_development_inputs(root, config)
    rows = activate_full_fit(rows)
    counts = rows["development_split"].value_counts().to_dict()
    if counts != summary["partition_counts"] or manifest["files_sha256"] != summary["source_prepared_files_sha256"] or manifest["files_sha256"] != operating["source_prepared_files_sha256"]:
        raise ValueError("La preparación o las particiones cambiaron después del entrenamiento.")
    validation = rows.index[rows["development_split"].eq("validation")]
    if operating["n_validation"] != len(validation):
        raise ValueError("La población de la ablación de política es distinta.")
    quality, margin = config["quality_threshold"], config["selection"]["quality_loss_margin"]
    diagnostics = {"scope": "post-training diagnostics on repeatedly reused development validation; no refit or new selection",
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_summary_sha256": sha256_file(summary_path),
                   "operating_summary_sha256": sha256_file(operating_path),
                   "outer_test_evaluated_here": False, "models_refit": False, "candidates": {}}
    for name, candidate in summary["candidates"].items():
        if Path(name).name != name or candidate["specification"]["objective"] != "pairwise":
            raise ValueError("Se espera un identificador directo con objetivo pareado.")
        artifact = candidate["artifact"]
        if summary["files_sha256"].get(artifact) != candidate["artifact_sha256"]:
            raise ValueError("La huella del modelo no coincide con el manifiesto.")
        model = joblib.load(verified_file(directory, artifact, candidate["artifact_sha256"]))
        prediction_name = f"{name}_predictions.parquet"
        prediction = pd.read_parquet(verified_file(directory, prediction_name, summary["files_sha256"][prediction_name]))
        if not prediction.index.equals(validation):
            raise ValueError("Las predicciones no corresponden exactamente a validación.")
        metric = paired_diagnostics(prediction, scores.loc[validation, pair], pair[0], pair[1], quality)
        metric["note"] = "A multiclass sigmoid calibrator and probability normalization do not guarantee reliability of the final class probabilities. These are reused-validation diagnostics."
        diagnostics["candidates"][name] = {"specification": candidate["specification"],
            "model_sha256": candidate["artifact_sha256"], "prediction_sha256": summary["files_sha256"][prediction_name],
            "fit_rows": int(model.fit_rows_), "calibration_rows": int(model.calibration_rows_),
            "target_supports": model.target_counts_, "calibration_notes": model.calibration_notes_, "validation": metric}

    reference9 = summary["routellm_bert"]["selected_point"]
    reference101 = operating["candidates"]["routellm_bert"]["selected_point"]
    lines = ["Relay v4 · desarrollo no lineal. Esta ejecución prueba otra familia de modelos después de conocer resultados anteriores. No es una cohorte independiente y no confirma superioridad sobre RouteLLM.",
        f"El par permanece {pair[0]} frente a {pair[1]}. Se usaron {counts['fit']:,} prompts de ajuste, {counts['calibration']:,} de calibración y {counts['validation']:,} de validación. Éxito significa score ≥ {quality:g}; el piso común es {pct(summary['quality_floor'])}, el éxito del fuerte fijo menos {100*margin:g} pp.",
        "Los nuevos candidatos usan HistGradientBoosting con las 14 variables estructurales que conservan magnitud y, en tres variantes, TF-IDF reducido por SVD. Vocabulario, proyección, escaladores, clasificadores y estimaciones de costo se ajustan solamente en fit; la calibración usa su bloque disjunto. Esta arquitectura prueba interacciones de contenido, formato y longitud; no demuestra comprensión semántica ni que esas interacciones sean la causa de una mejora.",
        "La primera tabla conserva la búsqueda inicial de nueve puntos por modelo. El candidato y corte se eligen por menor costo entre puntos cuya media cumple el piso. No se eligen usando los intervalos."]
    coarse = [("GPT-4 fijo", {"metrics": summary["fixed_strong"]}), ("RouteLLM BERT · búsqueda 9", reference9)]
    coarse.extend((name, candidate["selected_point"]) for name, candidate in summary["candidates"].items())
    lines.append(policy_table(coarse))
    coarse_comparisons = [(name, candidate["paired_development_comparison_vs_bert"])
                          for name, candidate in summary["candidates"].items()
                          if candidate["selected_point"] is not None]
    if coarse_comparisons:
        lines.append(comparison_table(coarse_comparisons, margin))
    lines.extend(["La segunda búsqueda utiliza 101 cuantiles de puntuaciones por router, con una política común de ganancia prevista dividida entre costo adicional previsto positivo. Las matrices completas de acciones se guardan antes de calcular sus métricas. El presupuesto de búsqueda no elimina el sesgo de reutilizar validación ni la diferencia entre supervisión local de Relay y supervisión externa de BERT.",
                  policy_table([(name, candidate["selected_point"]) for name, candidate in operating["candidates"].items()]),
                  comparison_table(list(operating["comparisons_vs_bert"].items()), margin)])
    lines.append(f"El diagnóstico conjunto del punto ya elegido exige límite inferior del IC de Δéxito > −{100*margin:g} pp y límite superior del IC de Δcosto < 0. Cumplirlo aquí es descriptivo: los intervalos se calculan después de elegir con la misma validación y no tienen cobertura confirmatoria garantizada. No se busca otro de los 101 intervalos para conseguir una conclusión favorable.")
    control_path = root / "reports/relay-v3-operating-v1/summary.json"
    sources = [{"path": str(summary_path.relative_to(root)), "sha256": sha256_file(summary_path)},
               {"path": str(operating_path.relative_to(root)), "sha256": sha256_file(operating_path)}]
    if control_path.is_file():
        control = read_json(control_path)
        verify_manifest(control_path.parent, control)
        if (control["outer_test_accessed"] is not False or control["confirmation_of_superiority"] is not False or
                control["common_pair"] != operating["common_pair"] or control["n_validation"] != len(validation) or
                control["source_prepared_files_sha256"] != manifest["files_sha256"] or
                not math.isclose(finite(control["quality_floor"]), finite(operating["quality_floor"]), abs_tol=1e-12) or
                control["candidates"]["routellm_bert"]["selected_point"] != reference101):
            raise ValueError("El control v3 no tiene el mismo par, población, piso o punto BERT.")
        lines.extend(["Como control histórico se conservan todos los puntos seleccionados en el barrido v3 de 101 cuantiles. Comparten población, piso, par y el mismo punto BERT, pero proceden de una búsqueda previa y mayor historial acumulado de selección. No se borran los controles que resulten desfavorables ni se atribuye una diferencia exclusivamente al algoritmo.",
                      policy_table([(f"Control v3 · {name}", candidate["selected_point"])
                                    for name, candidate in control["candidates"].items() if name != "routellm_bert"]),
                      comparison_table([(f"Control v3 · {name}", value) for name, value in control["comparisons_vs_bert"].items()], margin)])
        sources.append({"path": str(control_path.relative_to(root)), "sha256": sha256_file(control_path)})
    else:
        lines.append("El control histórico v3 no está disponible en esta copia; no se inventa una cifra para compararlo.")
    selected = summary["selected_for_future_evaluation"]
    operating_selected = operating["selected_for_future_evaluation"]
    lines.append(f"La búsqueda inicial registró {cell(selected['candidate_id'])} para evaluación futura; la búsqueda de 101 cuantiles registró {cell(operating_selected['candidate_id'])} con corte {finite(operating_selected['parameter']):.12g}. Son selecciones de desarrollo, no autorizaciones de promoción automática. Los candidatos, curvas, artefactos y reglas completos están en los resúmenes enlazados.")
    cpu = read_latency_evidence(directory, summary)
    if cpu is not None:
        evidence = cpu["evidence"]
        table = ["| CPU residente, candidato de la búsqueda inicial | p50 | p95 | n |", "| --- | ---: | ---: | ---: |"]
        for name, label in (("relay", f"Relay · {evidence['selected_candidate']}"), ("routellm_bert", "RouteLLM BERT")):
            metric = evidence["latency"][name]
            table.append(f"| {cell(label)} | {1000*finite(metric['median_seconds']):.2f} ms | {1000*finite(metric['p95_seconds']):.2f} ms | {metric['n']} |")
        lines.extend(["\n".join(table),
            f"La medición compara los mismos {evidence['sample']['n']} prompts, con modelos cargados, {evidence['warmups_per_backend']} calentamientos y orden alternado en un proceso CPU. Recalcula representación y predicción sin caché de atributos. BERT conserva su límite de {evidence['bert_tokenization']['max_length']} tokens y Relay procesa el texto completo. Se mide el candidato y punto iniciales, no todos los candidatos o todos los cortes de 101 cuantiles.",
            "Son tiempos de esta muestra pequeña y hardware, no un SLA ni ahorro monetario neto. Las inicializaciones tienen fronteras distintas y las cachés del sistema operativo no se reinician; no se comparan como arranques en frío controlados."])
        diagnostics["latency_evidence"] = {"path": cpu["path"], "sha256": cpu["sha256"], "selected_candidate": evidence["selected_candidate"], "latency": evidence["latency"]}
        sources.append({"path": str((directory / cpu["path"]).relative_to(root)), "sha256": cpu["sha256"]})
    else:
        table = ["| Relay, medición individual del runner | p50 | p95 | n |", "| --- | ---: | ---: | ---: |"]
        for name, candidate in summary["candidates"].items():
            metric = candidate.get("single_request_latency")
            if metric is not None:
                table.append(f"| {cell(name)} | {1000*finite(metric['p50_seconds']):.2f} ms | {1000*finite(metric['p95_seconds']):.2f} ms | {metric['n']} |")
        lines.extend(["\n".join(table), "No hay todavía una medición BERT individual bajo el mismo protocolo para v4. Los tiempos por lote anteriores no son una comparación equivalente; no se afirma una ventaja de latencia."])
    diagnostic_table = ["| Candidato no lineal | Brier multiclase | Log-loss multiclase | MSE de ganancia prevista |", "| --- | ---: | ---: | ---: |"]
    for name, candidate in diagnostics["candidates"].items():
        metric = candidate["validation"]
        diagnostic_table.append(f"| {cell(name)} | {finite(metric['multiclass_brier_sum_per_row']):.6f} | {finite(metric['multiclass_log_loss']):.6f} | {finite(metric['expected_quality_gain_mse']):.6f} |")
    lines.extend(["\n".join(diagnostic_table),
        "El Brier es la media por prompt de la suma de errores cuadrados de tres clases, rango [0,2]. Estos diagnósticos y soportes no garantizan calibración individual ni se comparan como si fueran el mismo objetivo que la preferencia externa de BERT. Las notas y soportes completos están en [diagnostics.json](diagnostics.json).",
        "La mayor flexibilidad de los árboles puede aprender formatos de benchmark y sobreajustar la validación reutilizada. No se ha evaluado generalización a familias nuevas, conversaciones o documentos productivos. Los costos son estimaciones históricas de las llamadas elegidas y excluyen cómputo del router, entrenamiento y reintentos. El test exterior anterior no se usa ni se reevalúa en este informe.",
        "Una afirmación futura de mejora sustancial exige congelar un candidato y un corte y evaluarlos en una cohorte nueva, con criterio compuesto de calidad y costo y overhead contabilizado si se afirma utilidad productiva. No se ensancha el margen ni se eliminan candidatos negativos para fabricar una victoria.",
        "Fuentes: [entrenamiento y búsqueda inicial](summary.json), [ablación de política de 101 cuantiles](../relay-v4-operating-v1/summary.json), [protocolo v4](../../docs/relay_v4_protocolo.md) y [control v3](../relay-v3-operating-v1/summary.json). El informe presenta los valores almacenados, sin reentrenar, recalibrar, cambiar cortes ni generar respuestas nuevas."])
    report_path, diagnostics_path = directory / "relay_v4.md", directory / "diagnostics.json"
    atomic_text(diagnostics_path, json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    atomic_text(report_path, "\n\n".join(lines)+"\n")
    provenance = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "script_sha256": sha256_file(Path(__file__).resolve()), "diagnostic_helper_sha256": sha256_file(ROOT / "tools/report_v3.py"),
        "report_sha256": sha256_file(report_path), "diagnostics_sha256": sha256_file(diagnostics_path),
        "outer_test_evaluated_here": False, "models_refit": False, "policy_or_candidate_reselected_here": False}
    atomic_text(directory / "report_provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return {"report": str(report_path), "diagnostics": str(diagnostics_path), "source_count": len(sources),
            "latency_evidence_included": cpu is not None, "outer_test_evaluated_here": False, "models_refit": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--operating-directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root, args.directory, args.operating_directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
