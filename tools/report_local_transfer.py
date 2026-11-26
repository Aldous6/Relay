"""Report the completed frozen local pilot without selecting or training."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.data import sha256_file
from routerbench.prospective import _hash

LABELS = {"candidate": "Relay v3 congelado", "reference": "RouteLLM BERT congelado",
          "baseline_strong": "Siempre Llama 3", "baseline_weak": "Siempre Qwen 2.5 1.5B"}
TASKS = ["Consulta de un registro", "Suma de lotes", "Filtro y suma", "Actualización hasta una fecha",
         "Relación de dos pasos", "Conflicto de fuentes", "Dato ausente", "Corrección en historial"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    directory = ROOT / "reports/local-transfer-pilot-v1"
    evaluation = directory / "evaluation"
    summary = read(evaluation / "summary.json")
    body = {name: value for name, value in summary.items() if name != "result_sha256"}
    if summary["result_sha256"] != _hash(body):
        raise ValueError("Resumen modificado.")
    for field, name in [("responses_sha256", "responses.json"), ("routes_sha256", "routes.json")]:
        if summary[field] != sha256_file(evaluation / name):
            raise ValueError(f"Registro modificado: {name}")
    freeze = read(evaluation / "freeze.json")
    if freeze["freeze_sha256"] != summary["freeze_sha256"] or freeze["dataset_sha256"] != sha256_file(directory / "cohort.jsonl"):
        raise ValueError("Cohorte o congelación modificada.")
    if freeze["policy_manifest"] != read(directory / "policy_lock.json"):
        raise ValueError("La política no corresponde a la congelada antes de crear los casos.")
    responses = read(evaluation / "responses.json")
    routes = read(evaluation / "routes.json")
    for checkpoint in [responses, routes]:
        if checkpoint["freeze_sha256"] != summary["freeze_sha256"] or checkpoint["payload_sha256"] != _hash(checkpoint["payload"]):
            raise ValueError("Checkpoint modificado.")
    if any(value["status"] == "pending" for value in responses["payload"].values()):
        raise ValueError("No se informa como completa una evaluación con llamadas pendientes.")
    metrics = summary["policies"]
    table = ["| Política | Respuestas exactas | Llama / Qwen | Tokens de entrada reportados | Tokens de salida reportados | Fallos de generación |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, metric in metrics.items():
        exact = metric["by_grade"]["exact"]
        counts = metric["selected_model_counts"]
        tokens = metric["tokens"]
        table.append(f"| {LABELS[name]} | {exact['successes']}/{exact['n']} ({100*exact['success_rate']:.2f} %) | {counts['strong']} / {counts['weak']} | {tokens['input_tokens']['known_total']} | {tokens['output_tokens']['known_total']} | {metric['generation_failures']} |")
    categories = ["| Familia sintética | Relay | BERT | Siempre Llama | Siempre Qwen |", "| --- | ---: | ---: | ---: | ---: |"]
    for task, title in enumerate(TASKS):
        cells = []
        for name in LABELS:
            subset = [row for row in summary["outcomes"][name] if row["id"].startswith(f"task{task}_")]
            cells.append(f"{sum(row['success'] for row in subset)}/{len(subset)}")
        categories.append(f"| {title} | "+" | ".join(cells)+" |")
    best_fixed = max(("baseline_strong", "baseline_weak"), key=lambda name: metrics[name]["by_grade"]["exact"]["success_rate"])
    first, second = summary["outcomes"]["candidate"], summary["outcomes"]["reference"]
    if [row["id"] for row in first] != [row["id"] for row in second]:
        raise ValueError("Outcomes de políticas desalineados.")
    only_relay = sum(a["success"] and not b["success"] for a, b in zip(first, second))
    only_bert = sum(b["success"] and not a["success"] for a, b in zip(first, second))
    lines = ["Relay · piloto prospectivo sintético local. Los modelos, políticas y parámetros de generación se fijaron antes de crear esta cohorte. Es una prueba controlada de transferencia; no representa validación productiva ni una victoria general sobre RouteLLM.",
        "Se generaron 64 casos nuevos con valores aleatorios y ocho plantillas, sobre documentos e historial. Las respuestas de referencia se calculan al crear los casos; nunca entran en los callbacks de enrutado o generación. Se rechazaron duplicados normalizados internos y solapamientos exactos con las huellas conocidas de RouterBench. Esto no certifica ausencia de similitudes de formato ni independencia respecto al preentrenamiento general.",
        "Se mantuvo Relay pairwise_hybrid_a5 con corte 20 y BERT con umbral 0.5, ambos provenientes del catálogo histórico. En esta prueba, la acción strong se proyecta a llama3:latest y weak a qwen2.5:1.5b. Esos nombres heredados no garantizan que Llama tenga mayor calidad local. Los digests de ambos modelos instalados y la semilla de la cohorte están congelados en policy_lock.json.",
        "\n".join(table),
        f"Se conservaron {summary['unique_generation_calls']} llamadas únicas de colección, compartidas entre las cuatro políticas. Cada política selecciona una respuesta por caso; los totales por política son reconstrucciones sobre esos mismos resultados observados, no cuatro ejecuciones independientes. La comparación pareada tuvo {only_relay} casos correctos solo con Relay y {only_bert} solo con BERT.",
        f"La mejor referencia fija por exactitud en esta muestra fue {LABELS[best_fixed]}. Esta descripción no selecciona una nueva política ni prueba que un router aporte valor frente a un modelo fijo. Los resultados por plantilla son:",
        "\n".join(categories),
        "La métrica exige igualdad textual después de normalizar Unicode, mayúsculas y espacios. Penaliza respuestas con explicación o formato extra aunque contengan un dato correcto; mide salida exacta para tareas restringidas, no calidad semántica general. Las ocho plantillas y el tamaño pequeño limitan la generalización; no se calcula ni afirma no inferioridad de 0.5 pp con este piloto.",
        "Los tokens son los contadores devueltos por cada modelo, con tokenizadores distintos. No son dólares ni una medida uniforme de cómputo. El costo monetario permanece desconocido. La colección incluye posibles cargas de modelos y efectos de caché; sumar duraciones seleccionadas no simula fielmente una política de servicio ni acredita un SLA o ahorro neto. Los tiempos de routing guardados incluyen además auditoría de cobertura BERT y no son el benchmark de latencia emparejado anterior.",
        "Se usa una sola respuesta por caso y modelo, con temperatura 0 y seed 42 configuradas; no se ha estimado variabilidad entre generaciones. Un fallo de proveedor se conserva como fallo y no se reintenta ocultamente. Las políticas no se entrenan ni reajustan con este resultado. Para adaptar el router al catálogo local, estos casos tendrían que declararse desarrollo y reservar otra cohorte antes de volver a medir.",
        "Fuentes locales: [resumen](evaluation/summary.json), [cohorte](cohort.jsonl), [políticas congeladas](policy_lock.json), [rutas](evaluation/routes.json), [respuestas](evaluation/responses.json) y [código archivado](execution_sources.zip). Parámetros y contadores del proveedor: [API chat de Ollama](https://docs.ollama.com/api/chat) y [parámetros de Ollama](https://docs.ollama.com/modelfile)."]
    path = directory / "local_transfer.md"
    path.write_text("\n\n".join(lines)+"\n", encoding="utf-8")
    provenance = {"source_sha256": {name: sha256_file(directory / name) for name in
        ["evaluation/summary.json", "evaluation/routes.json", "evaluation/responses.json", "evaluation/freeze.json",
         "cohort.jsonl", "policy_lock.json", "execution_sources.zip", "execution_sources.json"]},
        "report_sha256": sha256_file(path), "script_sha256": sha256_file(Path(__file__).resolve()),
        "models_retrained_or_policy_reselected": False, "production_validation": False}
    (directory / "report_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "policies": {name: metric["by_grade"] for name, metric in metrics.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
