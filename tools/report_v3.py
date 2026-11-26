"""Report completed v3 development artifacts without refitting or using test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.data import sha256_file
from routerbench.development import calibration_diagnostics, load_development_inputs
from routerbench.improvement import activate_full_fit
from routerbench.pairwise import OUTCOME_CLASSES, pairwise_targets


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Se requiere un objeto JSON: {path}")
    return value


def finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("El informe no admite valores no finitos.")
    return result


def pct(value) -> str:
    return f"{100*finite(value):.2f} %"


def pp(value) -> str:
    return f"{100*finite(value):+.2f} pp"


def interval(values, percent_points=True) -> str:
    if len(values) != 2 or finite(values[0]) > finite(values[1]):
        raise ValueError("Intervalo inválido.")
    convert = pp if percent_points else lambda value: f"{finite(value):+.6f}"
    return f"[{convert(values[0])}, {convert(values[1])}]"


def atomic_text(path: Path, content: str):
    temporary = path.with_name(path.name+".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def verified_file(directory: Path, name: str, checksum: str) -> Path:
    if Path(name).name != name:
        raise ValueError("Solo se admiten archivos directos del experimento.")
    path = directory / name
    if sha256_file(path) != checksum:
        raise ValueError(f"Artefacto modificado: {name}")
    return path


def paired_diagnostics(probabilities: pd.DataFrame, scores: pd.DataFrame,
                       strong: str, weak: str, quality: float) -> dict:
    if not probabilities.index.equals(scores.index) or not probabilities.index.is_unique:
        raise ValueError("Probabilidades pareadas y outcomes desalineados.")
    if not probabilities.columns.is_unique or not set(OUTCOME_CLASSES).issubset(probabilities.columns):
        raise ValueError("Faltan probabilidades de las tres clases pareadas.")
    matrix = probabilities.loc[:, list(OUTCOME_CLASSES)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any() or not np.allclose(matrix.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Las probabilidades pareadas deben sumar uno y estar en [0,1].")
    target = pairwise_targets(scores, strong, weak, quality)
    codes = target.map({name: position for position, name in enumerate(OUTCOME_CLASSES)}).to_numpy(dtype=int)
    truth = np.eye(len(OUTCOME_CLASSES))[codes]
    brier_per_class = np.mean((matrix-truth)**2, axis=0)
    gain = matrix[:, 0]-matrix[:, 1]
    observed_gain = scores[strong].ge(quality).astype(int)-scores[weak].ge(quality).astype(int)
    return {
        "class_order": list(OUTCOME_CLASSES),
        "validation_class_counts": target.value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict(),
        "multiclass_brier_sum_per_row": float(brier_per_class.sum()),
        "multiclass_brier_definition": "mean over prompts of sum over three classes of squared probability error; range [0,2]",
        "mean_one_vs_rest_brier": float(brier_per_class.mean()),
        # Numeric labels preserve the explicit probability-column order; passing
        # strings to sklearn could alphabetically reorder the target classes.
        "multiclass_log_loss": float(log_loss(codes, matrix, labels=list(range(len(OUTCOME_CLASSES))))),
        "per_class": {name: {"brier": float(brier_per_class[position]),
                              "observed_prevalence": float(truth[:, position].mean()),
                              "mean_predicted_probability": float(matrix[:, position].mean())}
                      for position, name in enumerate(OUTCOME_CLASSES)},
        "expected_quality_gain_mse": float(np.mean((gain-observed_gain.to_numpy())**2)),
        "note": "Normalization after three one-vs-rest sigmoid calibrators need not preserve calibration. These diagnostics are not evidence of individual reliability or superiority.",
    }


def read_latency_evidence(directory: Path, summary: dict) -> dict | None:
    path = directory / "latency_cpu.json"
    if not path.is_file():
        return None
    evidence = read_json(path)
    selected = summary["selected_for_future_evaluation"]
    if (evidence["source_summary_sha256"] != sha256_file(directory / "summary.json") or
            evidence["source_configuration_sha256"] != summary["configuration_sha256"] or
            evidence["selected_candidate"] != selected["candidate_id"] or
            evidence["selected_parameter"] != selected["point"]["parameter"] or
            evidence["selected_artifact_sha256"] != selected["artifact_sha256"] or
            evidence["bert_threshold"] != summary["routellm_bert"]["selected_point"]["parameter"]):
        raise ValueError("La medición CPU no corresponde a la selección v3 y referencia declaradas.")
    if evidence["outer_test_evaluated"] is not False or evidence["model_responses_generated"] is not False:
        raise ValueError("La medición de latencia contiene otra evaluación o generación.")
    if evidence["feature_or_prediction_cache_used"] is not False:
        raise ValueError("La medición no debe usar vectores o predicciones precalculadas.")
    count = evidence["sample"]["n"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("Tamaño de muestra de latencia inválido.")
    for name in ("relay", "routellm_bert"):
        metric = evidence["latency"][name]
        if metric["n"] != count or finite(metric["median_seconds"]) < 0 or finite(metric["p95_seconds"]) < metric["median_seconds"]:
            raise ValueError("Métricas de latencia incoherentes.")
    return {"path": path.name, "sha256": sha256_file(path), "evidence": evidence}


def build_report(root: Path = ROOT, directory: Path | None = None) -> dict:
    directory = root / "reports/relay-v3-development-v1" if directory is None else directory.resolve()
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Falta el resumen completo de v3; no se presentan resultados parciales.")
    summary = read_json(summary_path)
    if summary["outer_test_accessed_for_v3"] is not False or summary["confirmation_of_superiority"] is not False or summary["automatic_live_promotion"] is not False:
        raise ValueError("El experimento no cumple el contrato de desarrollo exclusivo.")
    config = summary["configuration"]
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if config_hash != summary["configuration_sha256"]:
        raise ValueError("La configuración guardada no coincide con su hash.")
    for name, checksum in summary["files_sha256"].items():
        verified_file(directory, name, checksum)
    # This loader slices outer train before constructing the outcome tables.
    _, scores, _, rows, manifest = load_development_inputs(root, config)
    rows = activate_full_fit(rows)
    if manifest["files_sha256"] != summary["source_prepared_files_sha256"]:
        raise ValueError("La preparación cambió después de entrenar v3.")
    counts = rows["development_split"].value_counts().to_dict()
    if counts != summary["partition_counts"]:
        raise ValueError("Las particiones del informe no coinciden con la ejecución.")
    validation_ids = rows.index[rows["development_split"].eq("validation")]
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    quality = config["quality_threshold"]
    diagnostics = {"scope": "post-training diagnostics on reused development validation only; no refit or new selection",
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_summary_sha256": sha256_file(summary_path),
                   "configuration_sha256": config_hash,
                   "outer_test_evaluated_here": False,
                   "candidates": {}}
    for name, candidate in summary["candidates"].items():
        if Path(name).name != name:
            raise ValueError("Identificador de candidato inválido.")
        artifact_name = candidate["artifact"]
        model_path = verified_file(directory, artifact_name, candidate["artifact_sha256"])
        if summary["files_sha256"].get(artifact_name) != candidate["artifact_sha256"]:
            raise ValueError("Hash de modelo incoherente con el manifiesto.")
        # Local trusted experiment artifacts only, after checksum verification.
        model = joblib.load(model_path)
        prediction_name = f"{name}_predictions.parquet"
        prediction_path = verified_file(directory, prediction_name, summary["files_sha256"][prediction_name])
        prediction = pd.read_parquet(prediction_path)
        if not prediction.index.equals(validation_ids):
            raise ValueError("Las predicciones no corresponden exactamente a validación.")
        record = {"objective": candidate["specification"]["objective"],
                  "model_sha256": candidate["artifact_sha256"],
                  "prediction_sha256": summary["files_sha256"][prediction_name],
                  "fit_rows": int(model.fit_rows_), "calibration_rows": int(model.calibration_rows_),
                  "calibration_notes": model.calibration_notes_}
        if record["objective"] == "absolute":
            record["all_models"] = calibration_diagnostics(prediction, scores.loc[validation_ids, prediction.columns], quality)
            record["common_pair"] = calibration_diagnostics(prediction.loc[:, pair], scores.loc[validation_ids, pair], quality)
            record["target_supports"] = {part: {model_name: {"success": int(scores.loc[rows["development_split"].eq(part), model_name].ge(quality).sum()),
                                                            "failure": int(scores.loc[rows["development_split"].eq(part), model_name].lt(quality).sum())}
                                                for model_name in prediction.columns}
                                        for part in ("fit", "calibration", "validation")}
        else:
            record["target_supports"] = model.target_counts_
            record["validation"] = paired_diagnostics(prediction, scores.loc[validation_ids, pair], pair[0], pair[1], quality)
        diagnostics["candidates"][name] = record

    reference_point = summary["routellm_bert"]["selected_point"]
    selected = summary["selected_for_future_evaluation"]
    lines = [
        "Relay v3 · resultados de desarrollo. Esta ejecución amplía el ajuste y prueba el objetivo pareado después de haber consumido el test original. No es una nueva evaluación independiente ni confirma superioridad.",
        f"El par común es {pair[0]} frente a {pair[1]}. Se usaron {counts['fit']:,} prompts de ajuste, {counts['calibration']:,} de calibración y {counts['validation']:,} de validación. El umbral del score que define éxito fue {quality:g}. El piso de selección de desarrollo fue {pct(summary['quality_floor'])}, calculado con el modelo fuerte fijo menos {100*config['selection']['quality_loss_margin']:g} pp.",
    ]
    table = ["| Candidato y punto de desarrollo | Éxito | Costo histórico / 1,000 |", "| --- | ---: | ---: |"]
    table.append(f"| GPT-4 fijo | {pct(summary['fixed_strong']['success_rate'])} | {finite(summary['fixed_strong']['cost_per_1000']):.6f} |")
    table.append(f"| RouteLLM BERT · umbral {reference_point['parameter']:g} | {pct(reference_point['metrics']['success_rate'])} | {finite(reference_point['metrics']['cost_per_1000']):.6f} |")
    for name, candidate in summary["candidates"].items():
        point = candidate["selected_point"]
        if point is None:
            table.append(f"| {name} · sin punto factible | — | — |")
        else:
            table.append(f"| {name} · parámetro {point['parameter']:g} | {pct(point['metrics']['success_rate'])} | {finite(point['metrics']['cost_per_1000']):.6f} |")
    lines.append("\n".join(table))
    if selected["candidate_id"] == "fixed_strong_fallback":
        lines.append("Ningún candidato Relay cumplió el piso de desarrollo. El respaldo declarado es GPT-4 fijo; no se cambian el margen ni la grilla para producir un ganador.")
    else:
        winner = summary["candidates"][selected["candidate_id"]]
        comparison = winner["paired_development_comparison_vs_bert"]
        savings = comparison["relative_call_cost_saving"]
        savings_ci = comparison["relative_call_cost_saving_95ci"]
        saving_text = "no estimable" if savings is None else pct(savings)
        saving_interval = "no estimable" if savings_ci is None else f"[{pct(savings_ci[0])}, {pct(savings_ci[1])}]"
        lines.append(f"El candidato seleccionado para una futura evaluación es {selected['candidate_id']}, con parámetro {selected['point']['parameter']:g}. Frente al punto BERT de desarrollo tuvo Δéxito={pp(comparison['success_rate_delta'])}, IC pareado del 95 % {interval(comparison['success_rate_delta_95ci'])}; Δcosto por 1,000={finite(comparison['cost_per_1000_delta']):+.6f}, IC {interval(comparison['cost_per_1000_delta_95ci'], False)}. La reducción relativa de costo histórico fue {saving_text}, con intervalo descriptivo {saving_interval}; un valor negativo significa mayor costo.")
    lines.append("Estos intervalos reutilizan la validación que seleccionó modelo, regularización y punto operativo. Son diagnósticos descriptivos con sesgo de selección; no prueban no inferioridad ni victoria confirmatoria. Se exploraron cuatro modelos Relay con nueve puntos cada uno frente a un checkpoint BERT con nueve umbrales. Igualar puntos por modelo no iguala búsqueda, entrenamiento ni supervisión.")
    cpu = read_latency_evidence(directory, summary)
    if cpu is None:
        latency = ["| Router Relay | Inferencia individual p50 | Inferencia individual p95 | n |", "| --- | ---: | ---: | ---: |"]
        for name, candidate in summary["candidates"].items():
            metric = candidate.get("single_request_latency")
            if metric is not None:
                latency.append(f"| {name} | {1000*finite(metric['p50_seconds']):.2f} ms | {1000*finite(metric['p95_seconds']):.2f} ms | {metric['n']} |")
        lines.extend(["\n".join(latency),
                      "La latencia Relay incluye representación del prompt y política, con el modelo cargado y sin vectores de atributos precalculados. No existe aquí una medición individual BERT bajo el mismo protocolo; su ejecución anterior por lotes no es una comparación equivalente. No se afirma una victoria de latencia ni ahorro monetario neto."])
    else:
        evidence = cpu["evidence"]
        latency = ["| Medición CPU individual con modelo residente | p50 | p95 | n |", "| --- | ---: | ---: | ---: |"]
        for name, label in (("relay", f"Relay · {evidence['selected_candidate']}"), ("routellm_bert", "RouteLLM BERT")):
            metric = evidence["latency"][name]
            latency.append(f"| {label} | {1000*finite(metric['median_seconds']):.2f} ms | {1000*finite(metric['p95_seconds']):.2f} ms | {metric['n']} |")
        lines.extend(["\n".join(latency),
                      f"La [medición CPU emparejada](latency_cpu.json) usa los mismos {evidence['sample']['n']} prompts de desarrollo, inferencia individual, orden alternado de backends y {evidence['warmups_per_backend']} calentamientos por backend. Se recomputan representación y predicción, sin caché de vectores ni resultados; los modelos permanecen cargados. El límite de pools nativos y los hilos de Torch registrados son {evidence['hardware_environment']['native_pool_limit']} y {evidence['hardware_environment']['torch_num_threads']}, respectivamente. BERT conserva su truncamiento oficial a {evidence['bert_tokenization']['max_length']} tokens mientras Relay procesa el texto completo; recibir los mismos prompts no implica retener el mismo contexto.",
                      "Estos tiempos describen esta muestra pequeña y no aleatoria, hardware y carga de CPU. No constituyen un SLA, una garantía para otros documentos o hardware ni ahorro monetario neto. Se excluyen red, colas, generación del LLM, reintentos y renderizado. Las inicializaciones tienen fronteras diferentes —la de BERT incluye verificaciones e imports adicionales— y las cachés del sistema operativo no se reiniciaron: no se comparan sus cifras como arranque en frío controlado."])
        diagnostics["latency_evidence"] = {"path": cpu["path"], "sha256": cpu["sha256"],
                                           "selected_candidate": evidence["selected_candidate"],
                                           "selected_artifact_sha256": evidence["selected_artifact_sha256"],
                                           "latency": evidence["latency"],
                                           "scope": evidence["kind"]}
    lines.append("El costo histórico de las políticas excluye cómputo del router, entrenamiento y reintentos. Una diferencia medida de latencia no confirma superioridad de calidad ni gasto neto de producción.")
    diagnostic_table = ["| Objetivo | Candidato | Brier diagnóstico | Log-loss diagnóstico |", "| --- | --- | ---: | ---: |"]
    for name, record in diagnostics["candidates"].items():
        if record["objective"] == "absolute":
            metric = record["common_pair"]
            brier = metric["mean_brier"]
            loss = np.mean([value["log_loss"] for value in metric["per_model"].values()])
            label = "Dos éxitos absolutos · promedio binario"
        else:
            metric = record["validation"]
            brier, loss = metric["multiclass_brier_sum_per_row"], metric["multiclass_log_loss"]
            label = "Tres resultados pareados · multicategoría"
        diagnostic_table.append(f"| {label} | {name} | {finite(brier):.6f} | {finite(loss):.6f} |")
    lines.extend(["\n".join(diagnostic_table),
                  "Los Brier y log-loss de objetivos distintos no se comparan como una misma escala de rendimiento: el Brier pareado suma los errores de tres clases por prompt y tiene rango [0,2]; el absoluto promedia dos problemas binarios. Soportes de ajuste/calibración/validación, notas de calibración y detalles están en [diagnostics.json](diagnostics.json). La normalización de tres calibradores one-vs-rest no garantiza calibración conjunta.",
                  "La trazabilidad de esta ejecución conserva predicciones y costos previstos antes de calcular las curvas. Las acciones de cada parámetro son deterministas y reconstruibles a partir de esos archivos; solo las rutas de los puntos elegidos se guardaron después de seleccionar. No se afirma que todas las rutas se persistieron antes de consultar outcomes. Esta nota documenta la implementación y no cambia las reglas de selección ni las métricas.",
                  "El [barrido operativo de 101 puntos](../relay-v3-operating-v1/relay_v3_operating.md) es una ablación secundaria posterior a estos primeros resultados. Conserva los mismos artefactos y reutiliza validación con búsqueda adicional; no reemplaza el candidato de la grilla inicial ni se promueve automáticamente a la demo.",
                  "No se usó ni reevaluó el test exterior consumido para esta revisión. La demo conserva la opción anterior y permite activar de forma explícita el candidato v3 de la grilla inicial como experimento; esa activación no equivale a validación productiva ni a promoción automática. BERT tiene supervisión externa y Relay ajuste local; la comparación no aísla el efecto del método. Una victoria requiere congelar un candidato y su punto y evaluarlos en una cohorte nueva, con límite inferior del IC de Δéxito>−0.005 y superior del IC de Δcosto<0, además del overhead para una afirmación productiva.",
                  "Fuentes locales: [resumen y curvas completos](summary.json), [protocolo v3](../../docs/relay_v3_protocolo.md), [diagnósticos](diagnostics.json). Los diagnósticos se calcularon después del entrenamiento con las predicciones almacenadas, sin reajustar probabilidades, cambiar el punto elegido ni generar respuestas nuevas."])
    diagnostics["source_files_sha256"] = summary["files_sha256"]
    diagnostics["script_sha256"] = sha256_file(Path(__file__).resolve())
    atomic_text(directory / "diagnostics.json", json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    atomic_text(directory / "relay_v3.md", "\n\n".join(lines)+"\n")
    return {"report": str(directory / "relay_v3.md"), "diagnostics": str(directory / "diagnostics.json"),
            "source_summary_sha256": diagnostics["source_summary_sha256"],
            "outer_test_evaluated_here": False, "models_refit": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root.resolve(), args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
