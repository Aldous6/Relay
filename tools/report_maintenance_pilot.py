"""Render an existing eight-case maintenance comparison, entirely offline.

Never evaluates a policy, selects thresholds, runs tests, loads BERT, reads
credentials, or contacts a provider. Raw generations are not copied into reports.
Only delivery_report.json and delivery_report.md are written in --run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.maintenance_competition import load_run

EXPECTED_TASKS = 8
POLICIES = (("relay", "Relay: cascada 20B → 120B"),
            ("strong_first", "120B desde la primera edición"),
            ("routellm_bert_transfer", "RouteLLM-BERT transferido"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    return value if type(value) is int and value >= 0 else None


def model_name(value, tariffs):
    if value in tariffs["models"]:
        return value
    candidate = "groq/" + value if isinstance(value, str) else None
    return candidate if candidate in tariffs["models"] else None


def safe_call(call, tariffs):
    usage = call.get("usage") if type(call.get("usage")) is dict else {}
    incoming, outgoing, reported_total = (count(usage.get(k)) for k in
                                           ("input_tokens", "output_tokens", "total_tokens"))
    total = incoming + outgoing if incoming is not None and outgoing is not None else reported_total
    if reported_total is not None and incoming is not None and outgoing is not None:
        require(reported_total == incoming + outgoing, "El uso registrado tiene un total inconsistente.")
    model = model_name(call.get("model") or call.get("returned_model") or
                       call.get("decision", {}).get("selected_model"), tariffs)
    cost = None
    if model is not None and incoming is not None and outgoing is not None:
        price = tariffs["models"][model]
        cost = (incoming * price["input_usd_per_million"] + outgoing * price["output_usd_per_million"]) / 1e6
    return {"model": model, "activity": call.get("activity"), "state": call.get("state"),
            "category": call.get("category"), "input_tokens": incoming, "output_tokens": outgoing,
            "local_preflight_rejection": call.get("category") in {"request_budget", "call_budget"},
            "total_tokens": total, "scenario_cost_usd": cost}


def summarize_calls(calls, tariffs):
    ledger = [safe_call(call, tariffs) for call in calls]
    unknown_tokens = sum(row["total_tokens"] is None for row in ledger)
    unknown_cost = sum(row["scenario_cost_usd"] is None for row in ledger)
    tokens = sum(row["total_tokens"] for row in ledger if row["total_tokens"] is not None)
    costs = math.fsum(row["scenario_cost_usd"] for row in ledger if row["scenario_cost_usd"] is not None)
    return {"registered_calls": len(ledger), "known_token_subtotal": tokens,
            "local_preflight_rejections": sum(row["local_preflight_rejection"] for row in ledger),
            "unknown_token_call_count": unknown_tokens, "total_tokens": None if unknown_tokens else tokens,
            "known_cost_subtotal_usd": costs, "unknown_cost_call_count": unknown_cost,
            "total_scenario_cost_usd": None if unknown_cost else costs, "calls": ledger}


def acquisition(directory, tariffs):
    manifest, journal = read(directory / "manifest.json"), read(directory / "journal.json")
    require(journal.get("manifest_sha256") == digest(directory / "manifest.json"),
            "Cambió un manifiesto de adquisición.")
    calls, evidence = [], []
    for task in manifest["suite"]["tasks"]:
        if task["partition"] != manifest["partition"]:
            continue
        location = directory / task["task_id"] / "collection.json"
        if not location.exists():
            continue
        pair = read(location)
        entry = journal["tasks"].get(task["task_id"], {})
        if entry.get("collection_sha256"):
            require(entry["collection_sha256"] == digest(location), "Cambió una colección registrada.")
        recorded = pair["acquisition"]["calls"]
        require(pair["acquisition"]["calls_started"] == len(recorded), "Conteo de llamadas inconsistente.")
        calls.extend(recorded)
        evidence.append({"task_id": task["task_id"], "collection_sha256": digest(location),
                         "registered_calls": len(recorded)})
    declared = sum(invocation["calls_started"] for invocation in journal["invocations"])
    require(declared == len(calls), "El diario y las colecciones difieren; espera a que termine la escritura.")
    return {"directory": str(directory), "manifest_sha256": digest(directory / "manifest.json"),
            "journal_sha256": digest(directory / "journal.json"), "evidence": evidence,
            **summarize_calls(calls, tariffs)}, calls


def diagnostic_acquisition(path, tariffs):
    diagnostic = read(path)
    require(diagnostic.get("kind") == "explicit_single_request_diagnostic"
            and diagnostic.get("not_part_of_comparison") is True
            and diagnostic.get("automatic_retry") is False,
            "El diagnóstico debe identificar una llamada separada y sin reintentos.")
    response = diagnostic.get("response") if type(diagnostic.get("response")) is dict else {}
    # Access metadata only: never access response.content or copy the response.
    call = {"usage": response.get("usage", diagnostic.get("usage")),
            "model": (response.get("returned_model") or response.get("model")
                      or diagnostic.get("model") or diagnostic.get("requested_model")),
            "state": diagnostic.get("status"), "activity": "separate_diagnostic"}
    return {"path": str(path), "sha256": digest(path), **summarize_calls([call], tariffs)}, [call]


def policy_details(rows, decisions, policy, metric, pairs):
    selected = []
    for row in rows:
        decision = decisions[row["task_id"]]
        role = (decision["relay_choice"] if policy == "relay" else decision["reference_choice"]
                if policy == "routellm_bert_transfer" else "strong")
        selected.append((row["branches"][role], pairs[row["task_id"]]["branches"][role]))
    require(metric["task_count"] == len(rows) and metric["success_count"] == sum(b["success"] for b, _ in selected),
            "comparison.json no corresponde a los resultados congelados.")
    require(math.isclose(metric["success_rate"], metric["success_count"] / len(rows), abs_tol=1e-12),
            "La tasa de éxito no corresponde a los conteos registrados.")
    known_tokens = [b["total_tokens"] for b, _ in selected if b["total_tokens"] is not None]
    require(metric["unknown_token_count"] == len(rows) - len(known_tokens)
            and metric["total_tokens"] == (sum(known_tokens) if len(known_tokens) == len(rows) else None),
            "comparison.json no corresponde a los tokens registrados.")
    known_costs = [b["cost_usd"] for b, _ in selected if b["cost_usd"] is not None]
    require(metric["unknown_cost_count"] == len(rows) - len(known_costs)
            and math.isclose(metric["known_cost_subtotal_usd"], math.fsum(known_costs), rel_tol=1e-9, abs_tol=1e-12),
            "comparison.json no corresponde a los costos registrados.")
    if len(known_costs) == len(rows):
        require(metric["total_cost_usd"] is not None and math.isclose(metric["total_cost_usd"], math.fsum(known_costs),
                rel_tol=1e-9, abs_tol=1e-12), "Total tarifario inconsistente.")
    else:
        require(metric["total_cost_usd"] is None, "Un costo desconocido no puede sustituirse por cero.")
    cost_per_success = (metric["total_cost_usd"] / metric["success_count"]
                        if metric["total_cost_usd"] is not None and metric["success_count"] else None)
    require((metric["cost_per_success_usd"] is None and cost_per_success is None)
            or (metric["cost_per_success_usd"] is not None and cost_per_success is not None
                and math.isclose(metric["cost_per_success_usd"], cost_per_success, rel_tol=1e-9, abs_tol=1e-12)),
            "El costo por tarea resuelta no corresponde a los resultados registrados.")
    return {**metric, "public_success_count": sum(b["visible_success"] for b, _ in selected),
            "private_success_count": sum(b["acceptance_success"] is True for b, _ in selected),
            "private_unknown_count": sum(b["acceptance_success"] is None for b, _ in selected),
            "repair_calls": sum(call["activity"] == "repair" for _, pair in selected for call in pair["calls"])}


def comparison_block(rows, decisions, section, pairs):
    return {"policies": {policy: policy_details(rows, decisions, policy, section["policies"][policy], pairs)
                         for policy, _ in POLICIES},
            "paired_comparisons": section["paired_comparisons"],
            "interpretation": section["interpretation"]}


def build_report(directory, prior_directory, diagnostic_path, additional_prior_directories=(), additional_diagnostics=()):
    directory, prior_directory, diagnostic_path = (Path(path).resolve() for path in
                                                   (directory, prior_directory, diagnostic_path))
    additional_prior_directories = [Path(path).resolve() for path in additional_prior_directories]
    diagnostic_paths = [diagnostic_path, *(Path(path).resolve() for path in additional_diagnostics)]
    require(directory != prior_directory, "La ejecución primaria y la anterior deben ser distintas.")
    require(len({directory, prior_directory, *additional_prior_directories}) == 2 + len(additional_prior_directories),
            "Una ejecución previa está repetida o coincide con la primaria.")
    require(len(set(diagnostic_paths)) == len(diagnostic_paths), "Un diagnóstico está repetido.")
    manifest, journal = read(directory / "manifest.json"), read(directory / "journal.json")
    require(manifest["partition"] == "development", "Este informe solo admite el piloto de desarrollo.")
    tasks = [task for task in manifest["suite"]["tasks"] if task["partition"] == "development"]
    require(len(tasks) == EXPECTED_TASKS and len({task["task_id"] for task in tasks}) == EXPECTED_TASKS
            and len({task["group_id"] for task in tasks}) == EXPECTED_TASKS,
            "El piloto requiere ocho casos y grupos diferentes.")
    plan = manifest.get("analysis_plan") or {}
    require(plan.get("reference_threshold") == .5 and plan.get("operational_mode") == "verified",
            "La ejecución no fijó cascada y umbral 0.5 antes de adquirir.")
    tariffs = manifest["suite"]["tariffs"]
    current, current_calls = acquisition(directory, tariffs)
    prior, prior_calls = acquisition(prior_directory, tariffs)
    additional, additional_calls = [], []
    for previous in additional_prior_directories:
        summary, calls = acquisition(previous, tariffs)
        additional.append(summary)
        additional_calls.extend(calls)
    # A copied run under a different directory is still the same expenditure.
    # Compare frozen identities, not just names or the (possibly shared) suite.
    seen_runs, seen_collections = set(), set()
    for summary in (current, prior, *additional):
        identity = (summary["manifest_sha256"], summary["journal_sha256"])
        require(identity not in seen_runs, "La misma adquisición aparece más de una vez con otra ruta.")
        seen_runs.add(identity)
        for item in summary["evidence"]:
            if not item["registered_calls"]:
                continue
            require(item["collection_sha256"] not in seen_collections,
                    "Una colección con llamadas está duplicada entre ejecuciones.")
            seen_collections.add(item["collection_sha256"])
    diagnostics, diagnostic_calls, seen_diagnostics = [], [], set()
    for path in diagnostic_paths:
        summary, calls = diagnostic_acquisition(path, tariffs)
        require(summary["sha256"] not in seen_diagnostics, "El mismo diagnóstico aparece más de una vez con otra ruta.")
        seen_diagnostics.add(summary["sha256"])
        diagnostics.append(summary)
        diagnostic_calls.extend(calls)
    require(len(prior_calls) == 3, "Se esperaban tres llamadas registradas en la primera ejecución.")
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "incomplete", "scope": "exploratory_eight_self_authored_development_cases",
              "independent_holdout": False, "learned_router": False, "general_superiority_claim": False,
              "primary_run": str(directory), "expected_task_count": EXPECTED_TASKS,
              "analysis_plan": plan, "cost_basis": "noncached_list_tariff_scenario_not_Groq_Free_billing",
              "tariffs_sha256": manifest["suite"]["tariffs_sha256"], "tariffs": tariffs,
              "acquisition": {"current_run": current, "prior_run": prior, "additional_prior_runs": additional,
                              "separate_diagnostic": diagnostics[0], "additional_diagnostics": diagnostics[1:],
                              "all_research": summarize_calls(current_calls + prior_calls + additional_calls + diagnostic_calls, tariffs)},
              "completed_task_count": sum(journal["tasks"].get(task["task_id"], {}).get("state") == "completed" for task in tasks),
              "task_states": {task["task_id"]: journal["tasks"].get(task["task_id"], {}).get("state", "unstarted") for task in tasks},
              "limitations": ["Ocho problemas autocreados, no una muestra industrial representativa.",
                              "No hay holdout independiente, entrenamiento de router ni réplicas estocásticas.",
                              "La aceptación reservada no fue feedback de reparación; comparte autoría con las tareas.",
                              "RouteLLM-BERT es un wrapper local del checkpoint publicado transferido a GPT-OSS, no MF ni todo RouteLLM.",
                              "La probabilidad de BERT representa preferencia, no probabilidad calibrada de resolver mantenimiento.",
                              "Los costos excluyen cómputo local, entrenamiento y operación; no demuestran ventaja de latencia ni ROI.",
                              "Los IC son bootstrap pareado exploratorio por ocho grupos; pueden ser inestables o degenerados.",
                              "El umbral secundario usa los mismos ocho resultados: sus IC no corrigen selección ni son confirmatorios."]}
    if result["completed_task_count"] != EXPECTED_TASKS:
        result["incomplete_reason"] = "Faltan casos completos; no se calculan comparaciones ni porcentajes de ahorro."
        return result
    if not (directory / "comparison.json").is_file() or not (directory / "reference.json").is_file():
        result["incomplete_reason"] = "Falta la referencia o comparación ya ejecutada; este informe no las recalcula."
        return result
    rows, verified_manifest, _, reference = load_run(directory)
    require(verified_manifest == manifest and len(rows) == EXPECTED_TASKS, "La adquisición cambió o está incompleta.")
    comparison = read(directory / "comparison.json")
    require(comparison.get("reference_threshold") == .5
            and comparison.get("reference_selection", {}).get("selection_scope") == "fixed_before_acquisition"
            and comparison.get("relay_policy", {}).get("name") == "relay_verified_cascade_v1"
            and comparison.get("relay_policy", {}).get("learned") is False
            and comparison.get("incomplete_or_unstarted_tasks") == 0
            and comparison.get("analysis_plan") == plan, "La comparación no es la primaria predeclarada.")
    require(comparison.get("reference") == reference["reference"]
            and comparison.get("acquisition_calls") == current["registered_calls"],
            "La comparación y el registro de referencia/adquisición no coinciden.")
    decisions = {row["task_id"]: row for row in comparison["decisions"]}
    require(set(decisions) == {row["task_id"] for row in rows}, "Las decisiones no cubren los ocho casos.")
    scores = {row["task_id"]: row for row in reference["rows"]}
    pairs = {row["task_id"]: read(directory / row["task_id"] / "collection.json") for row in rows}
    cases = []
    for row in rows:
        task_id, decision, score = row["task_id"], decisions[row["task_id"]], scores[row["task_id"]]
        require(decision["relay_choice"] == "weak" and decision["reference_choice"] ==
                ("strong" if score["probability"] >= .5 else "weak"), "Cambió una decisión primaria registrada.")
        for branch in row["branches"].values():
            require("visible_success" in branch and "acceptance_success" in branch,
                    "Falta distinguir pruebas públicas de aceptación.")
        cases.append({"task_id": task_id, "group_id": row["group_id"],
                      "branches": {role: {**branch, "repair_calls": sum(call["activity"] == "repair" for call in
                                  pairs[task_id]["branches"][role]["calls"])} for role, branch in row["branches"].items()},
                      "relay_choice": decision["relay_choice"], "reference_choice": decision["reference_choice"],
                      "bert": {key: score[key] for key in ("probability", "input_tokens", "retained_tokens", "truncated_tokens", "router_seconds")}})
    primary = comparison_block(rows, decisions, comparison["partitions"]["development"], pairs)
    calibrated = comparison.get("development_calibrated_reference")
    require(type(calibrated) is dict, "Falta declarar la referencia secundaria calibrada o su indisponibilidad.")
    secondary = {"available": False, "reason": calibrated.get("reason")}
    if "comparison" in calibrated:
        threshold = calibrated["selection"]["threshold"]
        require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1.000001
                and calibrated["selection"].get("selection_scope") == "development_only_empirical_not_noninferiority",
                "La referencia secundaria no declara correctamente su selección en desarrollo.")
        calibrated_decisions = {row["task_id"]: {**decisions[row["task_id"]],
            "reference_choice": "strong" if row["reference_probability"] >= threshold else "weak"} for row in rows}
        secondary = {"available": True, "selection": calibrated["selection"],
                     **comparison_block(rows, calibrated_decisions, calibrated["comparison"], pairs)}
    result.update(status="completed", primary=primary, development_calibrated_reference=secondary, cases=cases,
                  reference=reference["reference"], bootstrap=comparison["bootstrap"],
                  comparison_sha256=digest(directory / "comparison.json"), reference_sha256=digest(directory / "reference.json"),
                  private_verdicts_complete=all(row.get("independent_quality_validation") for row in rows),
                  outcome_scope="Only current run; all prior runs and diagnostics included solely in research expenditure.")
    recovery_path = directory / "recovery-audit.json"
    if recovery_path.exists():
        recovery = read(recovery_path)
        require(recovery.get("status") == "completed" and recovery.get("original_run_unchanged") is True,
                "La continuación derivada no terminó o modificó el original.")
        result["recovery"] = {"path": str(recovery_path), "sha256": digest(recovery_path),
                              "new_calls_started": recovery["new_calls_started"],
                              "inherited_calls": current["registered_calls"] - recovery["new_calls_started"],
                              "original_run_unchanged": True,
                              "source_amendment": recovery["source_amendment"]}
    return result


def fmt(value, places=6):
    return "desconocido" if value is None else f"{value:.{places}f}"


def verdict(value):
    return "sí" if value is True else "no" if value is False else "desconocido"


def policy_table(block):
    lines = ["| Política | Públicas | Privadas | Éxito conjunto | USD escenario | USD/tarea resuelta | Tokens | Reparaciones |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, label in POLICIES:
        row = block["policies"][policy]
        unknown = f"; {row['private_unknown_count']} desconocidas" if row["private_unknown_count"] else ""
        lines.append(f"| {label} | {row['public_success_count']}/8 | {row['private_success_count']}/8{unknown} | "
                     f"{row['success_count']}/8 | {fmt(row['total_cost_usd'])} | {fmt(row['cost_per_success_usd'])} | "
                     f"{row['total_tokens'] if row['total_tokens'] is not None else 'desconocido'} | {row['repair_calls']} |")
    return lines


def paired_text(block):
    lines = []
    for key, label in (("relay_vs_reference", "RouteLLM-BERT"), ("relay_vs_strong_first", "120B primero")):
        value = block["paired_comparisons"][key]
        def interval(metric, multiplier=1):
            ci = value["bootstrap_ci95"].get(metric)
            return "no disponible" if ci is None else f"[{fmt(ci[0] * multiplier, 2)}, {fmt(ci[1] * multiplier, 2)}]"
        savings = value["relative_cost_savings"]
        cost_text = ("Ahorro relativo no disponible por costo desconocido o denominador no válido."
                     if savings is None else f"Ahorro relativo de escenario: {100 * savings:.2f} %; "
                     f"IC95 {interval('relative_cost_savings', 100)} %.")
        lines.append(f"Relay frente a {label}: diferencia de éxito {100 * value['success_delta']:.2f} puntos porcentuales; "
                     f"IC95 {interval('success_delta', 100)} puntos. {cost_text} "
                     f"Diferencia de USD por tarea resuelta: {fmt(value['delta_cost_per_success_usd'])}. "
                     f"IC95: {value['bootstrap_ci95'].get('delta_cost_per_success_usd') or 'no disponible'}.\n")
    return lines


def render(result):
    direct = result["analysis_plan"].get("inspection_mode", "model") == "direct"
    inspection_text = (
        "La ejecución primaria usa un plan determinista de edición directa; no hay llamada de inspección. "
        if direct else
        "Todas las trayectorias incluyen inspección 20B; su respuesta se comparte durante adquisición y su uso "
        "se contabiliza en cada trayectoria lógica. ")
    lines = ["# Piloto exploratorio: ocho problemas propios de mantenimiento", "",
             "Desarrollo autocreado; sin holdout independiente ni router aprendido. La métrica primaria exige éxito "
             "en las pruebas públicas y en la aceptación reservada. Esta evidencia no acredita superioridad general.", "",
             "Solo la ejecución indicada como primaria determina los resultados primarios. Todas las ejecuciones previas "
             "y los diagnósticos se conservan como gasto de investigación; no se mezclan sus resultados con los ocho pares actuales.", ""]
    if result["status"] != "completed":
        lines.extend([f"Adquisición incompleta: {result['completed_task_count']}/8 casos. {result['incomplete_reason']}", ""])
    else:
        lines.extend(["## Comparación primaria: BERT con umbral fijo 0.5", "",
                      "Relay comienza la edición con 20B y permite como máximo una reparación 120B según el contrato fijado. "
                      "El control comienza con 120B y permite la misma reparación. " + inspection_text +
                      "La aceptación reservada se aplica al terminar y no provoca reparaciones.", "",
                      *policy_table(result["primary"]), "", *paired_text(result["primary"]),
                      "## Comparación secundaria: BERT calibrado en estos mismos ocho casos", ""])
        secondary = result["development_calibrated_reference"]
        if secondary["available"]:
            lines.extend([f"Umbral seleccionado ya registrado: {secondary['selection']['threshold']}. "
                          "La selección busca el menor costo que alcanza el éxito observado de 120B primero. "
                          "Es ajuste sobre los mismos resultados; no constituye validación independiente ni prueba de no inferioridad.",
                          "", *policy_table(secondary), "", *paired_text(secondary)])
        else:
            lines.extend([f"No disponible: {secondary['reason']}", ""])
        lines.extend(["## Resultados por caso", "",
                      "P/Pr/C significa públicas, privadas y éxito conjunto. Los costos corresponden a la trayectoria completa. "
                      "La decisión BERT es la primaria, con umbral fijo 0.5. Su score es preferencia por la acción fuerte, "
                      "no una probabilidad calibrada de éxito en mantenimiento.", "",
                      "| Caso | 20B primero P/Pr/C | USD 20B primero | 120B primero P/Pr/C | USD 120B primero | BERT p(fuerte) | Decide BERT | Tokens BERT retenidos/entrada | Truncados |",
                      "|---|---|---:|---|---:|---:|---|---:|---:|"])
        for row in result["cases"]:
            weak, strong, bert = row["branches"]["weak"], row["branches"]["strong"], row["bert"]
            def outcomes(branch):
                return "/".join(verdict(branch[key]) for key in ("visible_success", "acceptance_success", "success"))
            lines.append(f"| {row['task_id']} | {outcomes(weak)} | {fmt(weak['cost_usd'])} | {outcomes(strong)} | "
                         f"{fmt(strong['cost_usd'])} | {bert['probability']:.6f} | "
                         f"{'120B' if row['reference_choice'] == 'strong' else '20B'} primero | "
                         f"{bert['retained_tokens']}/{bert['input_tokens']} | {bert['truncated_tokens']} |")
        lines.append("")
    lines.extend(["## Adquisición y gasto total de investigación", "",
                  "Escenario de tarifa de lista sin caché aplicado al uso registrado, con las tarifas congeladas de la "
                  "ejecución primaria. No es facturación ni ahorro monetario realizado en Groq Free. Los costos por política "
                  "son trayectorias lógicas; esta tabla cuenta intentos registrados. La inspección compartida, cuando existe, "
                  "se cuenta una sola vez por par; la edición directa no genera esa llamada. "
                  "Un rechazo local por presupuesto puede haberse registrado antes de enviar HTTP y se indica por separado.", "",
                  "| Origen | Intentos | Rechazos locales | Subtotal tokens conocidos | Intentos sin tokens | Subtotal USD conocido | Intentos sin costo | Total USD |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    ledger = result["acquisition"]
    origins = [("Ejecución primaria", ledger["current_run"]), ("Ejecución previa 1", ledger["prior_run"])]
    origins.extend((f"Ejecución previa {index}", row)
                   for index, row in enumerate(ledger.get("additional_prior_runs", []), start=2))
    diagnostics = [ledger["separate_diagnostic"], *ledger.get("additional_diagnostics", [])]
    for index, diagnostic in enumerate(diagnostics, start=1):
        model = diagnostic["calls"][0]["model"] or "modelo desconocido"
        origins.append((f"Diagnóstico separado {index} ({model})", diagnostic))
    origins.append(("Toda la investigación", ledger["all_research"]))
    for label, row in origins:
        if "directory" in row:
            name = Path(row["directory"]).name.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            label += f" ({name})"
        lines.append(f"| {label} | {row['registered_calls']} | {row['local_preflight_rejections']} | {row['known_token_subtotal']} | "
                     f"{row['unknown_token_call_count']} | {fmt(row['known_cost_subtotal_usd'])} | "
                     f"{row['unknown_cost_call_count']} | {fmt(row['total_scenario_cost_usd'])} |")
    lines.extend(["", "Un subtotal conocido excluye consumos sin reporte: no se imputan como cero. "
                  "Si queda cualquier uso o costo desconocido, el total correspondiente permanece desconocido.", "",
                  "## Alcance y trazabilidad", "", " ".join(result["limitations"]), "",
                  "El generador conserva los puntos, decisiones e intervalos de comparison.json; no vuelve a seleccionar "
                  "umbrales, ejecutar modelos o evaluar pruebas. Los hashes, tarifas, ledger de adquisición y datos por caso "
                  "están en delivery_report.json. La evidencia primaria permanece en manifest.json, journal.json, "
                  "reference.json, comparison.json y las colecciones/aceptaciones de cada caso.", ""])
    if result.get("recovery"):
        recovery = result["recovery"]
        lines.extend([f"Esta ejecución deriva de v3: conserva sus {recovery['inherited_calls']} llamadas y añade "
                      f"solo {recovery['new_calls_started']} pendientes. El original permaneció intacto. "
                      "La corrección del colector reclasificó una reparación inválida conocida como fallo del contrato, "
                      "sin repetir la llamada ni modificar su respuesta. Los hashes y fuentes antes/después están en "
                      "[recovery-audit.json](recovery-audit.json). La copia original no se suma otra vez como gasto.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Informe offline del piloto de ocho casos; no llama APIs ni recalibra.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prior-run", required=True, type=Path)
    parser.add_argument("--additional-prior-run", action="append", default=[], type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--additional-diagnostic", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    try:
        directory = args.run.resolve()
        result = build_report(directory, args.prior_run.resolve(), args.diagnostic.resolve(),
                              [path.resolve() for path in args.additional_prior_run],
                              [path.resolve() for path in args.additional_diagnostic])
        for name, content in (("delivery_report.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                              ("delivery_report.md", render(result))):
            destination = directory / name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(destination)
        print(f"Informe {result['status']}: {directory / 'delivery_report.md'}")
        return 0 if result["status"] == "completed" else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"No se generó el informe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
