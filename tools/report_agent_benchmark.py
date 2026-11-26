"""Audit and report saved episode journals, without providers or environment execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.agent_metrics import annotate_costs, summarize


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Clave JSON duplicada: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Número JSON no finito: {value}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique, parse_constant=_constant)


def unseal(path):
    envelope = read(path)
    if (type(envelope) is not dict or set(envelope) != {"payload", "payload_sha256"} or
            object_hash(envelope["payload"]) != envelope["payload_sha256"]):
        raise ValueError(f"Sobre o hash alterado: {path}")
    return envelope["payload"]


def _pending(started):
    result = deepcopy(started)
    result.update(generation_seconds=None, verification_seconds=None, tool_seconds=None,
                  usage={}, reported_llm_cost_usd=None, error_type="InterruptedGeneration",
                  action=None, final_success=None, response=None)
    return result


def audit_run(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    if type(manifest) is not dict or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Manifiesto de episodios no reconocido.")
    body = {key: value for key, value in manifest.items() if key != "freeze_sha256"}
    if object_hash(body) != manifest["freeze_sha256"] or object_hash(manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("La congelación o las tareas fueron alteradas.")
    archive_path = directory / "source_bundle.zip"
    if digest(archive_path) != manifest["source_bundle_sha256"]:
        raise ValueError("El snapshot de fuentes fue alterado.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {*manifest["source_files_sha256"], "configuration.json"}:
            raise ValueError("El snapshot contiene fuentes distintas de las declaradas o nombres duplicados.")
        for name, expected in manifest["source_files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"La fuente congelada {name} fue alterada.")
        archived_config = json.loads(archive.read("configuration.json"), object_pairs_hook=_unique, parse_constant=_constant)
        if archived_config != manifest["configuration"]:
            raise ValueError("La configuración del snapshot difiere del manifiesto.")
    journal, summary = unseal(directory / "journal.json"), unseal(directory / "summary.json")
    config, freeze = manifest["configuration"], manifest["freeze_sha256"]
    if (journal["freeze_sha256"] != freeze or summary["freeze_sha256"] != freeze or
            summary["configuration"] != config or summary["journal_sha256"] != object_hash(journal["records"])):
        raise ValueError("Manifiesto, journal y resumen pertenecen a configuraciones distintas.")
    if (config["execution_mode"] not in {"simulated", "external"} or config["independent_validation"] is not False or
            summary["execution_mode"] != config["execution_mode"] or summary["independent_validation"] is not False):
        raise ValueError("Procedencia o alcance incompatible con este informe exploratorio.")
    tasks = {task["id"]: task for task in manifest["tasks"]}
    if len(tasks) != len(manifest["tasks"]) or not tasks or not manifest["policies"]:
        raise ValueError("Tareas duplicadas o población vacía.")
    policies = manifest["policies"]
    if "always_strong" not in policies:
        raise ValueError("Falta el control fuerte fijo.")
    for name, policy in policies.items():
        if policy["name"] != name or policy["catalog"] != config["models"]:
            raise ValueError("Política o catálogo diferente del congelado.")
    allowed = {(task_id, policy) for task_id in tasks for policy in policies}
    active, calls, episodes = {}, {key: [] for key in allowed}, {}
    previous = freeze
    for sequence, record in enumerate(journal["records"]):
        record_body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (type(record.get("sequence")) is not int or record["sequence"] != sequence or
                record["previous_sha256"] != previous or object_hash(record_body) != record["record_sha256"]):
            raise ValueError("Cadena de eventos alterada o desordenada.")
        previous = record["record_sha256"]
        key = (record["episode_id"], record["policy"])
        if key not in allowed or key in episodes:
            raise ValueError("Evento ajeno a la población o posterior al cierre.")
        kind, data = record["kind"], record["data"]
        if kind == "call_started":
            if (key in active or type(data["call_index"]) is not int or data["call_index"] != len(calls[key]) or
                    data["call_index"] >= min(config["max_calls"], tasks[key[0]]["max_steps"]) or
                    (calls[key] and calls[key][-1].get("final_success") is not None) or
                    data["model"] not in config.get("model_catalog", list(config["models"].values())) or
                    data["request_sha256"] != object_hash(data["request"]) or
                    set(data["request"]) != {"prompt", "history", "state"} or
                    data["request"]["prompt"] != tasks[key[0]]["prompt"]):
                raise ValueError("Inicio de llamada inconsistente con la decisión o solicitud congelada.")
            active[key] = data
        elif kind == "call_completed":
            if key not in active or any(data.get(name) != value for name, value in active[key].items()):
                raise ValueError("Llamada completada sin inicio o con una decisión modificada.")
            active.pop(key)
            calls[key].append(data)
        elif kind == "episode_completed":
            task = tasks[key[0]]
            if (data["episode_id"] != key[0] or data["policy"] != key[1] or
                    data["source_group"] != task["source_group"] or data["family"] != task["family"] or
                    data["provenance"] != {name: config[name] for name in
                                            ("execution_mode", "evaluation_split", "independent_validation")}):
                raise ValueError("Cierre con identidad o procedencia incompatible.")
            raw_events = list(calls[key])
            if key in active:
                if data["termination"] != "interrupted_not_retried" or data["success"] is not False:
                    raise ValueError("Un intento pendiente no puede reportarse como éxito o llamada completa.")
                raw_events.append(_pending(active.pop(key)))
            expected = annotate_costs(raw_events, config["pricing"], config["compute_rates"])
            if expected != data["events"]:
                raise ValueError("El cierre cambió llamadas, resultados o componentes de costo del journal.")
            final_success = raw_events[-1].get("final_success") if raw_events else None
            if final_success is not None:
                if type(final_success) is not bool or data["success"] != final_success or data["termination"] != "final_answer":
                    raise ValueError("El cierre no conserva el resultado final ya registrado.")
            elif data["success"] is not False or data["termination"] == "final_answer":
                raise ValueError("Se declaró éxito o respuesta final sin un evaluador final registrado.")
            episodes[key] = data
        else:
            raise ValueError("Tipo de evento desconocido.")
    completed_pairs = [tuple(key) for key in summary["completed_pairs"]]
    if (active or set(episodes) != allowed or len(completed_pairs) != len(allowed) or set(completed_pairs) != allowed or
            summary["n_episodes"] != len(allowed) or summary["n_tasks"] != len(tasks)):
        raise ValueError("No están todos los episodios terminados exactamente una vez.")
    ordered_episodes = [episodes[key] for key in completed_pairs]
    recomputed = summarize(ordered_episodes, config["roi"], bootstrap_replicates=config["bootstrap_replicates"], seed=config["seed"])
    if recomputed != summary["metrics"]:
        raise ValueError("Las métricas del resumen no se reproducen desde los episodios persistidos.")
    if recomputed["automatic_adoption"] is not False:
        raise ValueError("El informe no admite promoción automática.")
    if config["execution_mode"] == "simulated" and any(
            value["roi"]["adoption_status"] != "not_assessed_simulation" for value in recomputed["comparisons"].values()):
        raise ValueError("Una simulación no puede evaluar adopción o ROI real.")
    return manifest, journal, summary, ordered_episodes


def number(value, digits=6):
    return "desconocido" if value is None else f"{value:.{digits}f}"


def milliseconds(value):
    return "desconocido" if value is None else f"{value * 1000:.3f}"


def percentage(value):
    return "desconocido" if value is None else f"{100 * value:.2f} %"


def interval(values, scale=1, suffix=""):
    return "no disponible" if values is None else f"[{values[0] * scale:+.2f}, {values[1] * scale:+.2f}]{suffix}"


def label(value):
    names = {"always_strong": "Siempre fuerte", "always_weak": "Siempre barato",
             "cheap_then_escalate": "Barato y escalar tras error", "relay": "Relay · adaptador SDK"}
    return names.get(value, value.replace("|", "\\|").replace("\n", " ").replace("\r", " "))


def render_report(directory):
    directory = Path(directory).resolve()
    manifest, journal, summary, episodes = audit_run(directory)
    config, metrics = manifest["configuration"], summary["metrics"]
    simulated = config["execution_mode"] == "simulated"
    policies = metrics["policies"]
    scope = ("Esta ejecución usa respuestas programadas y entornos sintéticos en memoria. Comprueba contratos, escalado, persistencia y contabilidad; no mide calidad de LLMs, ahorro real ni capacidad general de agentes. Los importes LLM son ficticios de la simulación y no corresponden a facturas. El ROI no se evalúa."
             if simulated else
             "Esta ejecución declara callbacks externos y evaluación exploratoria. El informe reproduce lo guardado, sin llamar a proveedores ni evaluar de nuevo las respuestas; la procedencia declarada no certifica independencia ni permite afirmar adopción o ROI confirmado.")
    families = sorted({episode["family"] for episode in episodes})
    lines = ["Relay · evaluación de episodios. Informe de integridad y prueba de infraestructura.", "", scope, "",
             f"Se registraron {summary['n_tasks']} tareas completas y {summary['n_episodes']} episodios tarea–política, con los mismos identificadores de tarea para todos los controles. Partición declarada: {config['evaluation_split']}. Familias: {', '.join(families)}. Cada política comenzó con un entorno propio; la unidad de éxito es el episodio completo según su verificador, incluyendo los fallos en el denominador.", "",
             f"Catálogo declarado: {config['models']['strong']} y {config['models']['weak']}. El control principal es {label(metrics['baseline_policy'])}. La cascada usa errores anteriores registrados por el harness; no puede consultar la referencia privada para decidir otra llamada después de una respuesta final.", "",
             "Se verificaron el hash de congelación, el hash de tareas, el snapshot ZIP y sus fuentes, los sobres de journal/resumen y toda la cadena de eventos. Cada llamada completada conserva la decisión y solicitud persistidas antes de generación; los cierres conservan las llamadas y sus costos. Se repitieron annotate_costs y summarize desde los eventos guardados con la implementación local identificada en la procedencia, sin generación, ejecución de herramientas ni evaluación del contenido de respuestas. No se extrajo ni ejecutó código del ZIP.", "",
             "| Política | Episodios | Éxitos / episodios | Éxito en esta prueba | Llamadas registradas | Reintentos |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, policy in policies.items():
        lines.append(f"| {label(name)} | {policy['n_episodes']} | {policy['successes']}/{policy['n_episodes']} | {percentage(policy['success_rate'])} | {policy['calls']} | {policy['retry_calls']} |")
    lines += ["", "Las llamadas registradas incluyen intentos interrumpidos de resultado desconocido cuando existen; no se reejecutan al reanudar. Los reintentos son un subconjunto de las llamadas, no un cargo adicional sumado dos veces. Una respuesta final incorrecta termina el episodio y no autoriza escalado con información del evaluador privado.", ""]
    llm_label = "LLM ficticio (USD)" if simulated else "LLM registrado (USD)"
    lines += [f"| Política | {llm_label} | LLM de reintentos (subconjunto) | Costo total (USD) | Total / tarea (USD) | Episodios con costo completo |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, policy in policies.items():
        costs = policy["costs"]
        lines.append(f"| {label(name)} | {number(costs['components']['llm_usd']['total'])} | {number(costs['retry_llm_usd'])} | {number(costs['total_usd'])} | {number(costs['cost_per_task_usd'])} | {policy['coverage']['complete_cost_episodes']}/{policy['n_episodes']} |")
    lines += ["", "El costo total requiere LLM, router, verificación y herramientas. Un componente sin medición o tarifa permanece desconocido: no se reemplaza por cero. Medir segundos no establece su costo monetario. El total por tarea incluye episodios fallidos; un subtotal LLM no constituye costo total ni demuestra ahorro del harness.", "",
              "| Política | Llamadas con costo LLM conocido | Llamadas con costo router desconocido | Verificación desconocida | Herramientas desconocidas |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, policy in policies.items():
        components = policy["costs"]["components"]
        lines.append(f"| {label(name)} | {components['llm_usd']['known_calls']}/{policy['calls']} | {components['router_usd']['unknown_calls']} | {components['verification_usd']['unknown_calls']} | {components['tool_usd']['unknown_calls']} |")
    lines += ["", "| Política | Routing p50 (ms) | Routing p95 (ms) | Llamadas con tiempo routing | Episodio completo p50 (ms) | Episodio completo p95 (ms) | Episodios con tiempo total |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, policy in policies.items():
        routing, end = policy["routing_latency"], policy["end_to_end_latency"]
        lines.append(f"| {label(name)} | {milliseconds(routing['p50_seconds'])} | {milliseconds(routing['p95_seconds'])} | {routing['known']}/{routing['n']} | {milliseconds(end['p50_seconds'])} | {milliseconds(end['p95_seconds'])} | {end['known']}/{end['n']} |")
    lines += ["", "Routing mide la selección de política; el tiempo completo mide el episodio ejecutado. Los cuantiles usan únicamente mediciones disponibles y la cobertura aparece al lado. En una simulación con respuestas programadas, estos tiempos no estiman latencia de proveedores ni desempeño de un router aprendido que no haya participado.", "",
              "| Política frente al fuerte | Diferencia de éxito de esta prueba | IC 95 % por grupos | Reducción de costo total | Estado de adopción / ROI |",
              "| --- | ---: | ---: | ---: | --- |"]
    for name, comparison in metrics["comparisons"].items():
        paired = comparison["paired"]
        lines.append(f"| {label(name)} | {100 * paired['success_rate_delta']:+.2f} pp | {interval(paired['success_rate_delta_95ci'], 100, ' pp')} | {percentage(paired['relative_total_cost_saving'])} | {comparison['roi']['adoption_status']} |")
    roi = metrics["roi_config"]
    lines += ["", f"Los intervalos son diagnósticos del conjunto guardado: {config['bootstrap_replicates']} réplicas, semilla {config['seed']}, emparejadas por episode_id y remuestreadas por source_group. En respuestas programadas no son evidencia estadística sobre calidad de LLMs. Con costo total desconocido, la reducción de costo total también permanece desconocida.", "",
              f"Los criterios configurados son ahorro total mínimo {percentage(roi['min_total_cost_saving'])}, pérdida máxima de éxito {100 * roi['max_success_rate_loss']:.2f} puntos porcentuales y p95 de routing menor que {milliseconds(roi['max_p95_router_seconds'])} ms. Son parámetros para una evaluación futura: la simulación no puede validar ROI y no hay adopción automática.", "",
              "| Política | Cierres registrados |",
              "| --- | --- |"]
    for name, policy in policies.items():
        endings = "; ".join(f"{key}: {value}" for key, value in policy["terminations"].items())
        lines.append(f"| {label(name)} | {endings} |")
    lines += ["", "El siguiente uso evaluable requiere fijar modelos, tareas y grupos independientes, callbacks reales autorizados, presupuestos y medición completa de recursos. Los resultados de esta prueba no sustituyen una comparación de Relay con RouteLLM ni modifican los resultados históricos de routing de una sola llamada.", "",
              "Evidencia: [manifiesto](manifest.json), [snapshot de fuentes](source_bundle.zip), [journal íntegro](journal.json), [resumen](summary.json) y [procedencia del informe](report_provenance.json). Los hashes comprueban consistencia respecto a los archivos declarados; no autentican a su autor. Este informe no reentrena, no abre datos crudos ni deserializa modelos.", ""]
    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    provenance = {"script_sha256": digest(Path(__file__)), "manifest_sha256": digest(directory / "manifest.json"),
                  "journal_sha256": digest(directory / "journal.json"), "summary_sha256": digest(directory / "summary.json"),
                  "report_sha256": digest(path), "freeze_sha256": manifest["freeze_sha256"],
                  "source_bundle_sha256": digest(directory / "source_bundle.zip"),
                  "source_files_sha256": manifest["source_files_sha256"],
                  "recomputed_metrics_source_sha256": digest(ROOT / "src/routerbench/agent_metrics.py"),
                  "frozen_runtime_versions": manifest["runtime_versions"],
                  "journal_records_verified": len(journal["records"]),
                  "completed_episodes_verified": len(episodes), "costs_recomputed": True, "metrics_recomputed": True,
                  "generation_executed": False, "tools_executed": False, "final_evaluation_executed": False,
                  "training_executed": False, "raw_data_opened": False, "models_deserialized": False}
    (directory / "report_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/agent-benchmark-smoke-v1")
    args = parser.parse_args(argv)
    print(render_report(args.run_dir))


if __name__ == "__main__":
    main()
