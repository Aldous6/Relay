"""Audit and report the completed local adapter pilot; never train or generate."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.data import sha256_file
from routerbench.features import normalize_for_dedup
from routerbench.prospective import _aggregate, _hash, _read_cases, _write

LABELS = {"candidate": "Relay · 14 atributos adaptados", "reference": "BERT adaptado local",
          "published_bert": "RouteLLM BERT publicado · transferencia",
          "baseline_strong": "Siempre Qwen 1.5B", "baseline_weak": "Siempre Qwen 0.5B"}
TASKS = ["Consulta de un registro", "Suma de lotes", "Filtro y suma", "Actualización hasta una fecha",
         "Relación de dos pasos", "Conflicto de fuentes", "Dato ausente", "Corrección en historial"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def shadow_outcomes(shadow, rows, saved_routes, responses):
    if shadow["payload_sha256"] != _hash(shadow["routes"]) or shadow["threshold"] != .5:
        raise ValueError("Shadow alterado o umbral no congelado.")
    if len(shadow["routes"]) != len(rows) or len(saved_routes) != len(rows):
        raise ValueError("Faltan rutas shadow o principales.")
    result = []
    for row, route, saved in zip(rows, shadow["routes"], saved_routes):
        probability = route["probability"]
        if not isinstance(probability, (int, float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilidad shadow inválida.")
        expected = "strong" if probability >= .5 else "weak"
        if (route["id"] != row["id"] or route["prompt_sha256"] != row["prompt_sha256"] or
                route["model"] != expected or saved["id"] != row["id"] or
                saved["prompt_sha256"] != row["prompt_sha256"] or
                route["adapted_route"] != saved["routes"]["reference"]):
            raise ValueError("Las rutas shadow no coinciden con la cohorte y el control.")
        key = _hash([row["id"], row["prompt_sha256"], expected])
        response = responses[key]
        if response["id"] != row["id"] or response["prompt_sha256"] != row["prompt_sha256"] or response["model"] != expected:
            raise ValueError("La respuesta shadow pertenece a otra llamada.")
        success = response["status"] == "ok" and normalize_for_dedup(response["content"] or "") == normalize_for_dedup(row["reference"])
        result.append({"id": row["id"], "model": expected, "grade": row["grade"],
                       "success": success, "response_key": key})
    return result


def main():
    directory = ROOT / "reports/local-adapter-evaluation-v1"
    evaluation = directory / "evaluation"
    summary, freeze = read(evaluation / "summary.json"), read(evaluation / "freeze.json")
    lock = read(directory / "policy_lock.json")
    if summary["result_sha256"] != _hash({key: value for key, value in summary.items() if key != "result_sha256"}):
        raise ValueError("Resumen alterado.")
    if freeze["freeze_sha256"] != _hash({key: value for key, value in freeze.items() if key != "freeze_sha256"}):
        raise ValueError("Congelación alterada.")
    if (freeze["freeze_sha256"] != summary["freeze_sha256"] or freeze["policy_manifest"] != lock or
            summary["dataset_sha256"] != freeze["dataset_sha256"]):
        raise ValueError("La evaluación no corresponde a las políticas congeladas.")
    rows, _, dataset_hash = _read_cases(directory / "cohort.jsonl")
    if dataset_hash != summary["dataset_sha256"] or len(rows) != 64 or any(row["grade"] != "exact" for row in rows):
        raise ValueError("Cohorte alterada.")
    payloads = {}
    for filename, field in [("responses.json", "responses_sha256"), ("routes.json", "routes_sha256")]:
        checkpoint = read(evaluation / filename)
        if (sha256_file(evaluation / filename) != summary[field] or checkpoint["freeze_sha256"] != freeze["freeze_sha256"] or
                checkpoint["payload_sha256"] != _hash(checkpoint["payload"])):
            raise ValueError("Checkpoint alterado.")
        payloads[filename] = checkpoint["payload"]
    responses, routes = payloads["responses.json"], payloads["routes.json"]
    if len(responses) != 128 or any(value["status"] == "pending" for value in responses.values()):
        raise ValueError("La colección no está completa.")
    for policy, outcomes in summary["outcomes"].items():
        if len(outcomes) != 64:
            raise ValueError("Política incompleta.")
        for row, outcome, route in zip(rows, outcomes, routes):
            model = route["routes"].get(policy, policy.removeprefix("baseline_"))
            key = _hash([row["id"], row["prompt_sha256"], model])
            answer = responses[key]
            correct = answer["status"] == "ok" and normalize_for_dedup(answer["content"] or "") == normalize_for_dedup(row["reference"])
            if outcome != {"id": row["id"], "model": model, "grade": "exact", "success": correct, "response_key": key}:
                raise ValueError("Resultados reconstruidos no coinciden.")
    if _aggregate(summary["outcomes"], responses) != summary["policies"]:
        raise ValueError("Métricas no reproducibles desde las respuestas.")
    shadow = read(directory / "published_bert_routes.json")
    barrier = read(directory / "all_routes_before_generation.json")
    if (shadow["dataset_sha256"] != dataset_hash or shadow["policy_lock_sha256"] != sha256_file(directory / "policy_lock.json") or
            barrier["dataset_sha256"] != dataset_hash or barrier["n_cases"] != 64 or
            barrier["published_bert_routes_sha256"] != sha256_file(directory / "published_bert_routes.json") or
            barrier["evaluation_routes_sha256"] != sha256_file(evaluation / "routes.json")):
        raise ValueError("La barrera previa a generación no coincide con las rutas conservadas.")
    outcomes = {**summary["outcomes"], "published_bert": shadow_outcomes(shadow, rows, routes, responses)}
    metrics = _aggregate(outcomes, responses)
    for name, digest in lock["source_files_sha256"].items():
        with zipfile.ZipFile(directory / "source_bundle.zip") as archive:
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Fuente archivada alterada.")
    if sha256_file(directory / "source_bundle.zip") != lock["source_bundle_sha256"]:
        raise ValueError("Snapshot alterado.")
    source = ROOT / "reports/local-adapter-development-v1"
    development = read(source / "summary.json")
    if sha256_file(source / "summary.json") != lock["adapter_source_summary_sha256"]:
        raise ValueError("El entrenamiento fuente cambió.")
    for key, entry in [("candidate", development), ("control", development["controls"]["bert_adapted"])]:
        if sha256_file(source / entry["artifact"]) != lock[key+"_artifact_sha256"]:
            raise ValueError("Artefacto fuente alterado.")
    a = outcomes["baseline_strong"]
    b = outcomes["baseline_weak"]
    complementarity = {name: sum((x["success"], y["success"]) == pair for x, y in zip(a, b)) for name, pair in
                      {"both_correct": (True, True), "strong_only": (True, False), "weak_only": (False, True), "both_wrong": (False, False)}.items()}
    timing = read(directory / "route_timings.json")
    callback_seconds = {
        "candidate": [row["candidate_seconds"] for row in timing["candidate"]],
        "reference": [row["adapted_callback_seconds"] for row in shadow["routes"]],
        "published_bert_shared_native": [row["native_bert_seconds"] for row in shadow["routes"]]}
    resident = {name: {"n": len(values), "median_ms": float(np.median(values)*1000),
                       "p95_ms": float(np.quantile(values, .95)*1000), "total_seconds": sum(values)}
                for name, values in callback_seconds.items() if values}
    result = {"scope": lock["scope"], "policies": metrics, "complementarity": complementarity,
              "resident_routing": resident, "router_timing_scope": timing["scope"],
              "production_validation": False, "superiority_test_performed": False,
              "automatic_live_promotion": False, "monetary_cost_usd": None,
              "source_result_sha256": summary["result_sha256"]}
    _write(directory / "comparison.json", result)
    table = ["| Política | Exactas | Qwen 1.5B / 0.5B | Tokens entrada / salida | Segundos de generación seleccionados |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in LABELS:
        metric = metrics[name]
        exact, count, tokens = metric["by_grade"]["exact"], metric["selected_model_counts"], metric["tokens"]
        table.append(f"| {LABELS[name]} | {exact['successes']}/{exact['n']} ({100*exact['success_rate']:.2f} %) | {count['strong']} / {count['weak']} | {tokens['input_tokens']['known_total']} / {tokens['output_tokens']['known_total']} | {metric['selected_generation_seconds']:.3f} |")
    categories = ["| Plantilla | Relay | BERT adaptado | Qwen 1.5B fijo | Qwen 0.5B fijo |", "| --- | ---: | ---: | ---: | ---: |"]
    for task, title in enumerate(TASKS):
        values = [sum(row["success"] for row in outcomes[name] if row["id"].startswith(f"task{task}_")) for name in
                  ["candidate", "reference", "baseline_strong", "baseline_weak"]]
        categories.append(f"| {title} | "+" | ".join(f"{value}/8" for value in values)+" |")
    savings = 1-metrics["candidate"]["selected_generation_seconds"]/metrics["baseline_strong"]["selected_generation_seconds"]
    lines = [
        "Relay · evaluación de adaptación local, cohorte nueva de 64 casos. La política primaria quedó congelada antes de generar esta cohorte. No superó a usar siempre Qwen 1.5B: acertó 38/64 frente a 46/64. BERT adaptado acertó 39/64. No se modifica la demo ni se elige retrospectivamente otra arquitectura.",
        "El desarrollo utilizó los 64 casos sintéticos ya observados en la auditoría local, con 48 filas de ajuste y 16 de calibración por hash independiente de resultados. Relay usa los 14 atributos estructurales; el control usa 768 atributos CLS del BERT congelado. Comparten filas, hiperparámetros del árbol y grilla de utilidad, y cada política eligió su corte con calibración: 0.1 y 0.2. Calibrar y escoger el corte en esas mismas 16 filas produce diagnósticos de desarrollo optimistas; no se confunden con esta cohorte nueva.",
        "Se fijaron artefactos, digests de Qwen, instrucciones, parámetros y snapshot del código antes de muestrear la nueva semilla. Se verificó ausencia de coincidencias exactas normalizadas con los casos de desarrollo y RouterBench. Los 64 casos son otras instancias de las mismas ocho plantillas; la prueba no evalúa familias desconocidas, conversaciones reales ni documentos extensos. No acredita no inferioridad o superioridad general.",
        "\n".join(table),
        f"Las 128 respuestas únicas se comparten entre las cinco políticas; no son cinco ejecuciones de servicio. No hubo fallos de generación. Relay redujo {100*savings:.2f} % la suma de duraciones de generación seleccionadas frente a Qwen 1.5B, pero perdió ocho aciertos, equivalentes a 12.50 puntos porcentuales. Esa pérdida impide presentar el ahorro como calidad conservada. Los tokens totales fueron 15,950 frente a 15,957: esta prueba no respalda un ahorro sustancial de tokens.",
        "RouteLLM BERT publicado conserva su checkpoint de preferencias y umbral 0.5, proyectados al nuevo par. Eligió Qwen 1.5B en todos los casos. Es distinto del control BERT adaptado localmente: este último usa el backbone congelado y un nuevo árbol entrenado con las respuestas locales. Las preferencias publicadas se calcularon en el mismo forward que extrae CLS, sin entrar en los atributos del nuevo árbol. Todas las rutas, incluidas las shadow, quedaron persistidas antes de la primera generación.",
        "\n".join(categories),
        f"Complementariedad observada: ambos modelos acertaron {complementarity['both_correct']} casos, solo Qwen 1.5B acertó {complementarity['strong_only']}, solo Qwen 0.5B acertó {complementarity['weak_only']} y ambos fallaron {complementarity['both_wrong']}. El oráculo que conociera las respuestas habría acertado 47/64; es un límite retrospectivo, no una política ejecutable. Los modelos fallaron los ocho ejercicios de suma de lotes. Un router que solo escoge entre estas respuestas no puede corregir esos fallos compartidos.",
        "La exactitud exige igualdad normalizada de la respuesta completa. Puede penalizar explicaciones aunque incluyan el dato correcto, y no es una evaluación semántica amplia. La colección fija temperatura 0, seed 42, contexto 8192 y salida máxima 96; conserva el texto completo del prompt y recoge una sola respuesta por caso/modelo. No estima variabilidad entre generaciones ni usa APIs pagadas.",
        "La medición de routing usa CPU residente, cuatro hilos, calentamiento con dos textos neutrales y ejecución secuencial previa a generación. Los milisegundos no incluyen carga inicial ni calentamiento. El tiempo nativo del BERT publicado se comparte con la extracción del control, por lo que no es una ejecución independiente alternada. La suma de generación seleccionada excluye router y conserva efectos de caché, carga y orden. No representa dinero, ahorro energético, latencia completa de producción ni SLA. El costo monetario permanece desconocido.",
        "La siguiente mejora requiere evidencia local representativa y una selección más conservadora: identificar dónde el modelo pequeño conserva calidad, reservar nuevas familias y medir errores de enrutado. Los casos nuevos de este informe quedan consumidos para evaluar esta política; si se usan para adaptar, pasan a desarrollo. Para tareas que ambos modelos fallan, habrá que ampliar capacidades o candidatos y evaluar de nuevo antes de atribuir una mejora al router. Esta es la interpretación de estos resultados, no un nuevo experimento ejecutado.",
        "Evidencia: [comparación verificable](comparison.json), [resumen original](evaluation/summary.json), [cohorte](cohort.jsonl), [políticas](policy_lock.json), [rutas publicadas](published_bert_routes.json), [barrera previa a generación](all_routes_before_generation.json), [respuestas](evaluation/responses.json) y [snapshot ejecutado](source_bundle.zip). Parámetros y contadores: [chat de Ollama](https://docs.ollama.com/api/chat) y [parámetros de Ollama](https://docs.ollama.com/modelfile)."]
    report = directory / "local_adapter.md"
    report.write_text("\n\n".join(lines)+"\n", encoding="utf-8")
    sources = ["evaluation/summary.json", "evaluation/freeze.json", "evaluation/routes.json", "evaluation/responses.json",
               "policy_lock.json", "cohort.jsonl", "published_bert_routes.json", "all_routes_before_generation.json",
               "route_timings.json", "source_bundle.zip", "comparison.json"]
    _write(directory / "report_provenance.json", {"source_sha256": {name: sha256_file(directory / name) for name in sources},
        "report_sha256": sha256_file(report), "script_sha256": sha256_file(Path(__file__).resolve()),
        "models_retrained_or_policy_reselected": False, "production_validation": False})
    print(json.dumps({"report": str(report), "complementarity": complementarity, "resident_routing": resident}, ensure_ascii=False))


if __name__ == "__main__":
    main()
