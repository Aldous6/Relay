"""Report frozen local-agent competition; never generate, train or load model weights."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_agent_benchmark import (audit_run, digest, interval, milliseconds, number,
                                    object_hash, percentage, read, unseal)
from routerbench.agent_metrics import summarize

PRIMARY = "cheap_then_escalate"
LABELS = {"cheap_then_escalate": "Cascada · primaria",
          "always_strong": "Siempre fuerte", "always_weak": "Siempre barato",
          "local_structural_transfer": "Adaptador estructural local · transferencia",
          "routellm_bert_transfer": "RouteLLM BERT publicado · transferencia a Qwen"}
REFERENCES = ("always_strong", "always_weak", "routellm_bert_transfer", "local_structural_transfer")


def _sum_complete(values):
    return sum(values) if values and all(value is not None for value in values) else None


def _resource_totals(episodes):
    events = [event for episode in episodes for event in episode["events"]]
    metadata = [event["usage"].get("provider_metadata") for event in events]
    return {"n_episodes": len(episodes), "calls": len(events),
            "input_tokens": _sum_complete([event["usage"].get("input_tokens") for event in events]),
            "output_tokens": _sum_complete([event["usage"].get("output_tokens") for event in events]),
            "episode_seconds": _sum_complete([episode.get("end_to_end_seconds") for episode in episodes]),
            "generation_seconds": _sum_complete([event.get("generation_seconds") for event in events]),
            "provider_reply_count": sum(isinstance(item, dict) and item.get("provider") == "ollama" for item in metadata),
            "provider_finish_reasons": dict(sorted(Counter(item.get("finish_reason") or "unknown" for item in metadata if isinstance(item, dict)).items())),
            "provider_output_limit_replies": sum(isinstance(item, dict) and item.get("finish_reason") == "length" for item in metadata),
            "backend_error_calls": sum(event.get("error_type") == "LiveBackendError" for event in events),
            "backend_affected_episodes": sum(any(event.get("error_type") == "LiveBackendError" for event in episode["events"]) for episode in episodes),
            "provider_load_seconds": _sum_complete([item.get("durations", {}).get("load_seconds") if isinstance(item, dict) else None for item in metadata]),
            "selected_models": dict(sorted(Counter(event["model"] for event in events).items())),
            "errors": dict(sorted(Counter(event["error_type"] for event in events if event.get("error_type") is not None).items()))}


def _bert_coverage(episodes, policy_manifest):
    events = [event for episode in episodes for event in episode["events"]]
    rows = []
    threshold = policy_manifest["threshold"]
    maximum = policy_manifest["tokenization"]["max_length"]
    for event in events:
        audit = event.get("routing_audit")
        if not isinstance(audit, dict) or audit.get("selected_model") != event["model"] or audit.get("threshold") != threshold:
            raise ValueError("Falta la auditoría BERT de una decisión o cambió su umbral/modelo.")
        incoming, retained, truncated = [audit.get(name) for name in ("input_tokens", "retained_tokens", "truncated_tokens")]
        if (any(type(value) is not int for value in (incoming, retained, truncated)) or
                not 1 <= retained <= maximum or incoming < retained or truncated != incoming-retained or
                not math.isclose(audit["token_coverage"], retained/incoming, rel_tol=1e-12)):
            raise ValueError("La cobertura BERT no coincide con sus contadores congelados.")
        probability = audit.get("probability")
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("El score de preferencia BERT es inválido.")
        selected = policy_manifest["catalog"]["strong" if probability >= threshold else "weak"]
        if selected != event["model"]:
            raise ValueError("La decisión BERT no reproduce su regla congelada.")
        rows.append(audit)
    incoming_total = sum(row["input_tokens"] for row in rows)
    retained_total = sum(row["retained_tokens"] for row in rows)
    return {"decisions": len(rows), "decisions_with_truncation": sum(row["truncated_tokens"] > 0 for row in rows),
            "input_tokens_total": incoming_total, "retained_tokens_total": retained_total,
            "truncated_tokens_total": incoming_total-retained_total,
            "weighted_token_coverage": retained_total/incoming_total if incoming_total else None,
            "minimum_token_coverage": min((row["token_coverage"] for row in rows), default=None),
            "maximum_token_coverage": max((row["token_coverage"] for row in rows), default=None),
            "minimum_input_tokens": min((row["input_tokens"] for row in rows), default=None),
            "maximum_input_tokens": max((row["input_tokens"] for row in rows), default=None),
            "scope": "router-input coverage per decision, not generator tokens or proof of relevant-information coverage"}


def _audit_predecessor(manifest):
    link = manifest["configuration"].get("technical_predecessor")
    if link is None:
        return None
    directory = (ROOT / link["run_directory"]).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("La ejecución predecesora debe quedar dentro del proyecto.")
    for name in ("manifest", "journal"):
        if digest(directory / (name + ".json")) != link[name + "_sha256"]:
            raise ValueError("Cambió un archivo de la ejecución interrumpida enlazada.")
    previous = read(directory / "manifest.json")
    journal = unseal(directory / "journal.json")
    if (object_hash({key: value for key, value in previous.items() if key != "freeze_sha256"}) != previous["freeze_sha256"] or
            previous["freeze_sha256"] != link["freeze_sha256"] or journal["freeze_sha256"] != link["freeze_sha256"] or
            digest(directory / "source_bundle.zip") != previous["source_bundle_sha256"]):
        raise ValueError("La congelación o snapshot previo no coincide con el enlace técnico.")
    preserved = ("execution_mode", "evaluation_split", "independent_validation", "primary_policy", "policy_order",
                 "models", "models_metadata", "generation_options", "generation_manifest", "base_url", "timeout_seconds",
                 "max_calls", "pricing", "compute_rates", "roi", "bootstrap_replicates", "seed", "per_family")
    if (link.get("same_tasks_models_parameters_thresholds") is not True or
            any(previous["configuration"].get(key) != manifest["configuration"].get(key) for key in preserved) or
            previous["tasks_sha256"] != manifest["tasks_sha256"] or
            previous["policies"] != manifest["policies"] or previous["policy_sequence"] != manifest["policy_sequence"]):
        raise ValueError("La repetición cambió tareas, modelos, políticas o parámetros científicos.")
    allowed = {(task["id"], policy) for task in previous["tasks"] for policy in previous["policies"]}
    active, finished, calls = {}, set(), []
    per_pair_counts = Counter()
    prior_hash = previous["freeze_sha256"]
    for index, record in enumerate(journal["records"]):
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (record["sequence"] != index or record["previous_sha256"] != prior_hash or
                object_hash(body) != record["record_sha256"]):
            raise ValueError("La cadena del journal interrumpido fue alterada.")
        prior_hash = record["record_sha256"]
        key, data = (record["episode_id"], record["policy"]), record["data"]
        if key not in allowed or key in finished:
            raise ValueError("Identidad inválida en el journal interrumpido.")
        if record["kind"] == "call_started":
            if key in active or data["call_index"] != per_pair_counts[key]:
                raise ValueError("Orden inválido de intentos previos.")
            active[key] = data
        elif record["kind"] == "call_completed":
            if key not in active or any(data.get(name) != value for name, value in active[key].items()):
                raise ValueError("La llamada previa no conserva su decisión inicial.")
            active.pop(key)
            per_pair_counts[key] += 1
            calls.append(data)
        elif record["kind"] == "episode_completed":
            if key in active or (data["episode_id"], data["policy"]) != key:
                raise ValueError("Cierre previo incompatible con un intento pendiente.")
            finished.add(key)
        else:
            raise ValueError("Evento previo desconocido.")
    if len(calls) != link["previous_durable_calls"] or len(active) != link["previous_pending_calls"]:
        raise ValueError("El enlace técnico no reproduce el estado pendiente del journal previo.")
    interruption = read(directory / "interruption.json")
    if (interruption["manifest_sha256"] != link["manifest_sha256"] or
            interruption["journal_sha256"] != link["journal_sha256"] or interruption["reason"] != link["reason"] or
            interruption["durable_completed_calls"] != len(calls) or interruption["pending_calls"] != len(active) or
            interruption["complete_episodes"] != len(finished)):
        raise ValueError("La nota de interrupción contradice los archivos conservados.")
    temporary_digest = None
    if interruption.get("temporary_journal_preserved"):
        temporary_digest = digest(directory / "journal.json.tmp")
        if temporary_digest != interruption["temporary_journal_sha256"]:
            raise ValueError("El journal temporal previo cambió.")
    changed_sources = sorted(name for name in set(previous["source_files_sha256"]) | set(manifest["source_files_sha256"])
                             if previous["source_files_sha256"].get(name) != manifest["source_files_sha256"].get(name))
    return {"link": link, "durable_completed_calls": len(calls), "pending_calls": len(active),
            "completed_episodes": len(finished), "same_scientific_configuration_verified": True,
            "changed_source_files": changed_sources,
            "durable_calls_input_tokens": _sum_complete([event["usage"].get("input_tokens") for event in calls]),
            "durable_calls_output_tokens": _sum_complete([event["usage"].get("output_tokens") for event in calls]),
            "durable_calls_generation_seconds": _sum_complete([event.get("generation_seconds") for event in calls]),
            "interruption_sha256": digest(directory / "interruption.json"),
            "temporary_journal_sha256": temporary_digest,
            "scope": "additional interrupted work reported separately; pending call consumption unknown; no pooling into completed paired comparison"}


def _resource_delta(candidate, reference):
    if candidate["n_episodes"] != reference["n_episodes"]:
        raise ValueError("La comparación de recursos requiere los mismos episodios completos.")
    result = {}
    for name in ("calls", "input_tokens", "output_tokens", "episode_seconds", "generation_seconds"):
        left, right = candidate[name], reference[name]
        result[name + "_delta"] = left-right if left is not None and right is not None else None
    delta = result["episode_seconds_delta"]
    result["mean_episode_seconds_delta"] = delta/candidate["n_episodes"] if delta is not None else None
    result["scope"] = "descriptive primary-minus-reference resource differences across all paired episodes; no monetary conversion or independent claim"
    return result


def _matched_requests(episodes):
    grouped = {}
    for episode in episodes:
        for event in episode["events"]:
            key = (event["model"], event["request_sha256"])
            response = event.get("response")
            grouped.setdefault(key, []).append({"policy": episode["policy"], "episode_id": episode["episode_id"],
                    "call_index": event["call_index"], "error_type": event.get("error_type"),
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest() if isinstance(response, str) else None,
                    "response_characters": len(response) if isinstance(response, str) else None,
                    "generation_seconds": event.get("generation_seconds"),
                    "input_tokens": event["usage"].get("input_tokens"), "output_tokens": event["usage"].get("output_tokens")})
    matched = []
    for (model, request_hash), records in grouped.items():
        if len({record["policy"] for record in records}) < 2:
            continue
        responses = {record["response_sha256"] for record in records if record["response_sha256"] is not None}
        errors = {record["error_type"] for record in records}
        bert_fixed = [record for record in records if record["policy"] in {"routellm_bert_transfer", "always_strong"}]
        matched.append({"requested_model": model, "request_sha256": request_hash, "records": records,
                        "different_text_responses": len(responses) > 1, "different_error_status": len(errors) > 1,
                        "bert_and_fixed_strong_present": len({record["policy"] for record in bert_fixed}) == 2,
                        "bert_fixed_outcomes_differ": len({(record["response_sha256"], record["error_type"]) for record in bert_fixed}) > 1})
    return {"cross_policy_request_groups": len(matched),
            "groups_with_different_text_responses": sum(group["different_text_responses"] for group in matched),
            "groups_with_different_error_status": sum(group["different_error_status"] for group in matched),
            "bert_fixed_shared_groups": sum(group["bert_and_fixed_strong_present"] for group in matched),
            "bert_fixed_different_outcome_groups": sum(group["bert_and_fixed_strong_present"] and group["bert_fixed_outcomes_differ"] for group in matched),
            "all_matched_groups": matched,
            "scope": "same requested model and exact canonical request hash, observed across policies; descriptive variability without causal attribution"}


def _trajectories(episodes, catalog):
    """Summarize recorded behavior, without comparing answers to private references."""
    result = {}
    for episode in episodes:
        policy = episode["policy"]
        seen_tools, seen_error = set(), False
        repeated, invalid, tool_errors, failed_tests = 0, 0, 0, 0
        action_names = []
        for event in episode["events"]:
            if policy in ("always_strong", "always_weak", PRIMARY):
                expected_role = ("strong" if policy == "always_strong" or policy == PRIMARY and seen_error else "weak")
                if event["model"] != catalog[expected_role]:
                    raise ValueError("Una política fija o la cascada no reproduce su regla congelada.")
            error = event.get("error_type")
            invalid += isinstance(error, str) and error.startswith("InvalidAction:")
            tool_errors += isinstance(error, str) and error.startswith("ToolError:")
            seen_error = seen_error or error is not None
            action = event.get("action")
            if isinstance(action, dict) and action.get("type") == "tool":
                signature = json.dumps(action, sort_keys=True, ensure_ascii=False, allow_nan=False)
                repeated += signature in seen_tools
                seen_tools.add(signature)
                action_names.append(action["name"])
            elif isinstance(action, dict) and action.get("type") == "final":
                action_names.append("final")
            else:
                action_names.append("invalid_or_missing_action")
            observation = event.get("tool_result", {}).get("observation")
            failed_tests += isinstance(observation, dict) and observation.get("all_passed") is False
        result.setdefault(policy, []).append({"episode_id": episode["episode_id"], "family": episode["family"],
                    "success": episode["success"], "termination": episode["termination"],
                    "calls": len(episode["events"]), "invalid_action_calls": int(invalid),
                    "tool_error_calls": int(tool_errors), "public_failed_test_calls": int(failed_tests),
                    "repeated_identical_tool_actions": int(repeated), "actions": action_names,
                    "final_incorrect": episode["termination"] == "final_answer" and not episode["success"]})
    return result


def _observed_patterns(episodes):
    repeated_reads, missing_inspection, schema_arguments = [], [], []
    for episode in episodes:
        parsed = [event["action"] for event in episode["events"] if isinstance(event.get("action"), dict)]
        names = [action.get("name") for action in parsed if action.get("type") == "tool"]
        identity = {"episode_id": episode["episode_id"], "policy": episode["policy"]}
        if (episode["family"] == "aggregate_records" and names.count("read_records") >= 2
                and "write_json" not in names and episode["termination"] == "call_budget_exhausted"):
            repeated_reads.append({**identity, "read_calls": names.count("read_records")})
        if episode["family"] == "repair_function" and "inspect_function" not in names:
            missing_inspection.append(identity)
        for event in episode["events"]:
            action = event.get("action")
            if isinstance(action, dict) and action.get("type") == "tool":
                arguments = action["arguments"]
                if arguments.get("type") == "object" and "properties" in arguments:
                    schema_arguments.append({**identity, "call_index": event["call_index"], "tool": action["name"],
                                             "error_type": event.get("error_type")})
    return {"aggregate_repeated_reads_without_write": repeated_reads,
            "repair_without_inspection": missing_inspection,
            "schema_objects_submitted_as_arguments": schema_arguments,
            "scope": "observable action patterns only; no hidden-answer comparison or causal diagnosis"}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _count(value):
    return "desconocido" if value is None else f"{value:,}"


def _signed(value, digits=0):
    return "desconocido" if value is None else f"{value:+,.{digits}f}"


def render_report(directory):
    directory = Path(directory).resolve()
    manifest, journal, summary, episodes = audit_run(directory)
    config, metrics = manifest["configuration"], summary["metrics"]
    if config["execution_mode"] != "external" or config["independent_validation"] is not False:
        raise ValueError("Este informe requiere ejecución externa exploratoria, no una simulación.")
    if config.get("primary_policy") != PRIMARY:
        raise ValueError("El manifiesto debe predeclarar cheap_then_escalate como primary_policy.")
    if config.get("policy_order") != "cyclic_by_task":
        raise ValueError("Esta comparación predeclara rotación de políticas por tarea.")
    if set(manifest["policies"]) != set(LABELS) or set(metrics["policies"]) != set(LABELS):
        raise ValueError("Deben conservarse la primaria y los cuatro controles predeclarados.")
    if config["evaluation_split"] != "development":
        raise ValueError("Esta comparación local fue diseñada como desarrollo.")
    sequence = manifest.get("policy_sequence")
    if not isinstance(sequence, list) or len(sequence) != len(LABELS) or set(sequence) != set(LABELS):
        raise ValueError("Congelar la secuencia base de políticas como policy_sequence, no solo un objeto sin orden.")
    expected_pairs = []
    for index, task in enumerate(manifest["tasks"]):
        offset = index % len(sequence)
        expected_pairs.extend([[task["id"], name] for name in sequence[offset:] + sequence[:offset]])
    observed_pairs = [[record["episode_id"], record["policy"]] for record in journal["records"]
                      if record["kind"] == "episode_completed"]
    if summary["completed_pairs"] != expected_pairs or observed_pairs != expected_pairs:
        raise ValueError("El orden registrado no reproduce la rotación congelada.")
    predecessor = _audit_predecessor(manifest)
    protocol = config.get("protocol_document")
    if protocol is not None:
        protocol_path = (ROOT / protocol["path"]).resolve()
        if not protocol_path.is_relative_to(ROOT) or digest(protocol_path) != protocol["sha256"]:
            raise ValueError("El protocolo enlazado cambió respecto al manifiesto.")
    grouped = {name: [episode for episode in episodes if episode["policy"] == name] for name in LABELS}
    resources = {name: _resource_totals(rows) for name, rows in grouped.items()}
    bert_manifest = manifest["policies"]["routellm_bert_transfer"]
    structural_manifest = manifest["policies"]["local_structural_transfer"]
    bert_coverage = _bert_coverage(grouped["routellm_bert_transfer"], bert_manifest)
    trajectories = _trajectories(episodes, config["models"])
    matched_requests = _matched_requests(episodes)
    observed_patterns = _observed_patterns(episodes)
    comparisons = {}
    for reference in REFERENCES:
        rerun = summarize(episodes, config["roi"], baseline_policy=reference,
                          bootstrap_replicates=config["bootstrap_replicates"], seed=config["seed"])
        if rerun["policies"] != metrics["policies"]:
            raise ValueError("Cambiar la referencia no debe modificar métricas absolutas de las políticas.")
        comparison = rerun["comparisons"][PRIMARY]
        if comparison["roi"]["automatic_adoption"] is not False:
            raise ValueError("La comparación no permite promoción automática.")
        comparisons[reference] = {**comparison, "resources": _resource_delta(resources[PRIMARY], resources[reference])}
    if comparisons["always_strong"]["paired"] != metrics["comparisons"][PRIMARY]["paired"]:
        raise ValueError("La comparación fuerte no reproduce el resumen original.")
    extended = {"schema_version": 1, "freeze_sha256": manifest["freeze_sha256"],
                "summary_sha256": digest(directory / "summary.json"), "primary_policy": PRIMARY,
                "execution_mode": "external", "evaluation_split": "development", "independent_validation": False,
                "policy_manifests": manifest["policies"], "resources": resources, "comparisons": comparisons,
                "bert_input_coverage": bert_coverage,
                "technical_predecessor_audit": predecessor,
                "trajectory_diagnostics": trajectories,
                "matched_request_diagnostics": matched_requests,
                "observed_action_patterns": observed_patterns,
                "n_tasks": summary["n_tasks"], "n_episodes": summary["n_episodes"],
                "monetary_conversion_applied": False, "generation_executed": False, "training_executed": False,
                "automatic_adoption": False}
    policies = metrics["policies"]
    primary = policies[PRIMARY]
    backend_calls = sum(row["backend_error_calls"] for row in resources.values())
    backend_episodes = sum(row["backend_affected_episodes"] for row in resources.values())
    families = sorted({episode["family"] for episode in episodes})
    lines = ["Relay · competencia local entre políticas de agentes. Desarrollo exploratorio.", "",
             f"La primaria predeclarada fue la cascada barato→fuerte tras error. Completó {primary['successes']}/{primary['n_episodes']} tareas ({percentage(primary['success_rate'])}) con {primary['calls']} intentos de generación registrados y {primary['retry_calls']} reintentos. Las cifras siguientes conservan todos los controles; ningún resultado cambia la primaria ni acredita adopción o ROI.", "",
             f"La ejecución contiene {summary['n_tasks']} tareas y {summary['n_episodes']} episodios tarea–política con los modelos locales {config['models']['strong']} y {config['models']['weak']}. Fuerte y barato nombran los roles y no presuponen un orden de aciertos. Las familias son {', '.join(families)}. Son instancias de familias ya conocidas, no una prueba de generalización a familias nuevas. Con {primary['n_episodes']} tareas, un solo resultado cambia {100/primary['n_episodes']:.2f} puntos porcentuales de éxito. La muestra es demasiado pequeña para una conclusión competitiva general.", "",
             "La cascada es una regla sobre errores observados del harness, no un router aprendido nuevo. El adaptador estructural conserva entrenamiento local previo y se transfiere ahora a entradas con estado e historial de episodios. BERT conserva el checkpoint publicado y proyecta su decisión fuerte/barato al catálogo Qwen: no fue calibrado con preferencias de ese par. Su resultado aquí no representa el rendimiento oficial con GPT-4/Mixtral ni evalúa MF o SW. Los regímenes de supervisión y las representaciones difieren; comparar resultados de episodios no los vuelve equivalentes.", "",
             f"El control estructural conserva {structural_manifest['training']['fit']} casos de ajuste, {structural_manifest['training']['calibration']} de calibración y corte {structural_manifest['frozen_specification']['tradeoff']} sobre utilidad con segundos previstos; ese corte no es una probabilidad ni un precio. BERT conserva {bert_manifest['checkpoint']}, revisión {bert_manifest['checkpoint_revision']}, corte {bert_manifest['threshold']} y máximo {bert_manifest['tokenization']['max_length']} tokens de entrada. Los scores internos tienen objetivos distintos y no se interpretan como probabilidades comparables de éxito del episodio.", "",
             "Los parámetros, identidades de modelos y procedencia exacta de cada política están preservados en el manifiesto y en comparisons.json. No se reentrenó ni se eligieron nuevos umbrales para redactar este informe. El verificador final cierra el episodio tanto al acertar como al fallar: su resultado privado no alimenta una nueva decisión de escalado.", "",
             "| Política | Éxitos / tareas | Éxito | Intentos registrados | Reintentos | Respuestas con metadatos Ollama |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    if predecessor is not None:
        predecessor_lines = [f"Esta ejecución repite las mismas tareas tras una interrupción técnica de {predecessor['link']['run_directory']}: PermissionError al reemplazar el journal; no se identificó la causa concreta del bloqueo. Se verificó que tareas, modelos, parámetros de generación, políticas y cortes permanecieron iguales. Las fuentes que cambiaron fueron {', '.join(predecessor['changed_source_files'])}. La cohorte es desarrollo reutilizado tras esa interrupción, no una nueva observación independiente.", "",
                             f"La ejecución previa conserva {predecessor['durable_completed_calls']} llamadas completadas en el journal canónico, {predecessor['completed_episodes']} episodios cerrados y {predecessor['pending_calls']} intento pendiente. Sus llamadas completadas registran {_count(predecessor['durable_calls_input_tokens'])} tokens de entrada, {_count(predecessor['durable_calls_output_tokens'])} de salida y {number(predecessor['durable_calls_generation_seconds'], 3)} s de generación. Son recursos adicionales separados de las comparaciones siguientes; no representan todo el consumo previo porque el intento pendiente y otros componentes siguen desconocidos. El archivo temporal se preservó y verificó, sin incorporarlo como resultado confirmado ni reanudar esa ejecución.", ""]
        lines[4:4] = predecessor_lines
    if backend_calls:
        lines[4:4] = [f"La comparación está afectada por {backend_calls} fallos LiveBackendError repartidos en {backend_episodes} episodios. Esa clase agrupa problemas de transporte, HTTP, respuesta del proveedor y contrato/identidad; el journal no identifica la causa concreta. No es evidencia de incapacidad del modelo. Los fallos y su consumo desconocido se mantienen en tasas, tiempo y cobertura, pero impiden aislar la calidad de routing o de los LLMs a partir de esta ejecución. Las acciones JSON inválidas se registran por separado.", ""]
    for name in LABELS:
        policy = policies[name]
        lines.append(f"| {LABELS[name]} | {policy['successes']}/{policy['n_episodes']} | {percentage(policy['success_rate'])} | {policy['calls']} | {policy['retry_calls']} | {resources[name]['provider_reply_count']} |")
    lines += ["", "| Política | Llamadas con LiveBackendError | Episodios afectados por backend/contrato |",
              "| --- | ---: | ---: |"]
    for name in LABELS:
        lines.append(f"| {LABELS[name]} | {resources[name]['backend_error_calls']} | {resources[name]['backend_affected_episodes']} |")
    bert_models = resources["routellm_bert_transfer"]["selected_models"]
    if bert_models.get(config["models"]["strong"], 0) == resources["routellm_bert_transfer"]["calls"]:
        lines += ["", "BERT eligió el fuerte en todas sus decisiones. Su selección de candidatos coincide con siempre fuerte; cualquier diferencia observada entre sus aciertos no puede atribuirse a escoger modelos distintos. Permanecen variaciones de respuestas, backend, orden y tiempo de ejecución que esta muestra no separa causalmente.", ""]
    lines += ["", f"Se identificaron {matched_requests['cross_policy_request_groups']} grupos entre políticas con el mismo modelo solicitado y request_sha256 exacto. En {matched_requests['groups_with_different_text_responses']} grupos aparecieron textos de respuesta distintos y en {matched_requests['groups_with_different_error_status']} estados de error distintos. BERT y siempre fuerte compartieron {matched_requests['bert_fixed_shared_groups']} solicitudes exactas; en {matched_requests['bert_fixed_different_outcome_groups']} difirieron respuesta o error. Todos los grupos, no solo los discrepantes, quedan en comparisons.json. Esto registra variabilidad observada bajo parámetros congelados; no identifica su causa ni permite atribuirla al routing.", ""]
    lines += ["", "Un intento del harness puede fallar antes de obtener respuesta, incluso antes de invocar el modelo por un límite local. Las respuestas con metadatos no incluyen intentos interrumpidos o fallidos sin respuesta utilizable. Los reintentos ya están dentro del total y no se suman dos veces. No se eliminan tareas fallidas para mejorar tasas o recursos medios.", "",
              f"BERT truncó su entrada en {bert_coverage['decisions_with_truncation']}/{bert_coverage['decisions']} decisiones. Retuvo {bert_coverage['retained_tokens_total']:,} de {bert_coverage['input_tokens_total']:,} tokens acumulados por decisión ({percentage(bert_coverage['weighted_token_coverage'])}); la cobertura individual fue de {percentage(bert_coverage['minimum_token_coverage'])} a {percentage(bert_coverage['maximum_token_coverage'])}. Las entradas completas tenían entre {_count(bert_coverage['minimum_input_tokens'])} y {_count(bert_coverage['maximum_input_tokens'])} tokens. La tarea precede al historial, pero esta cobertura no demuestra retención de las observaciones relevantes: BERT puede perder historial que la cascada sí utiliza. Son tokens del router, no del generador.", "",
              "| Política | Respuestas que alcanzaron el límite de salida | Motivos de terminación del proveedor | Errores observados del harness |",
              "| --- | ---: | --- | --- |"]
    for name in LABELS:
        resource = resources[name]
        reasons = "; ".join(f"{key}: {value}" for key, value in resource["provider_finish_reasons"].items()) or "sin metadatos"
        errors = "; ".join(f"{key}: {value}" for key, value in resource["errors"].items()) or "ninguno registrado"
        lines.append(f"| {LABELS[name]} | {resource['provider_output_limit_replies']} | {reasons} | {errors} |")
    lines += ["", "El límite de salida se contabiliza solo cuando Ollama informa finish_reason=length. No se confunde ese truncamiento de una respuesta con la truncación de entrada de BERT. Una salida limitada o inválida permanece en los recursos y resultados; no se genera otra respuesta para corregir este informe.", "",
              "| Política | Intentos con fuerte | Intentos con barato | Tokens de entrada totales | Tokens de salida totales | Cobertura entrada | Cobertura salida |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in LABELS:
        policy, resource = policies[name], resources[name]
        incoming, outgoing = policy["tokens"]["input_tokens"], policy["tokens"]["output_tokens"]
        lines.append(f"| {LABELS[name]} | {resource['selected_models'].get(config['models']['strong'], 0)} | {resource['selected_models'].get(config['models']['weak'], 0)} | {_count(incoming['total'])} | {_count(outgoing['total'])} | {incoming['known_calls']}/{policy['calls']} | {outgoing['known_calls']}/{policy['calls']} |")
    lines += ["", "Los tokens se acumulan por llamada, incluido el contexto repetido en reintentos. Son contadores de recursos, no unidades monetarias. Un contador faltante deja desconocido el total; los subtotales y la cobertura permanecen en summary.json. Tokens cacheados desconocidos no se convierten en cero ni se infieren del total de entrada.", "",
              "| Política | Routing p50 (ms) | Routing p95 (ms) | Routing medido | Episodio p50 (s) | Episodio p95 (s) | Episodios medidos |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in LABELS:
        policy = policies[name]
        routing, end = policy["routing_latency"], policy["end_to_end_latency"]
        lines.append(f"| {LABELS[name]} | {milliseconds(routing['p50_seconds'])} | {milliseconds(routing['p95_seconds'])} | {routing['known']}/{routing['n']} | {number(end['p50_seconds'], 3)} | {number(end['p95_seconds'], 3)} | {end['known']}/{end['n']} |")
    lines += ["", f"El tiempo por episodio es el observado durante ejecución del entorno y sus llamadas; los cuantiles excluyen mediciones desconocidas y muestran su cobertura. Se comprobó la rotación contra la secuencia congelada: {', '.join(sequence)}. Rotar políticas reduce parte del efecto del orden, pero no elimina diferencias por carga inicial, caché, calentamiento o competencia de recursos. Con {summary['n_tasks']} tareas y {len(sequence)} políticas, los resultados no equivalen a un ensayo de latencia aleatorizado. No se interpreta menor duración como menor costo USD ni como un SLA.", "",
              "| Política | Generación acumulada (s) | Carga del proveedor acumulada (s) | Costo LLM (USD) | Costo total (USD) | Episodios con costo completo |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in LABELS:
        policy, resource = policies[name], resources[name]
        lines.append(f"| {LABELS[name]} | {number(resource['generation_seconds'], 3)} | {number(resource['provider_load_seconds'], 3)} | {number(policy['costs']['components']['llm_usd']['total'])} | {number(policy['costs']['total_usd'])} | {policy['coverage']['complete_cost_episodes']}/{policy['n_episodes']} |")
    lines += ["", "La carga informada por el proveedor es un componente diagnóstico de sus llamadas, no tiempo adicional que se sume otra vez al episodio. Los USD permanecen desconocidos cuando no hay gasto medido ni tarifa declarada; ejecutar localmente no convierte el costo en cero. Faltan, entre otros, energía y valoración del hardware/cómputo. No se convierten segundos o tokens en dinero mediante precios inventados.", "",
              "| Cascada primaria frente a | Diferencia de éxito | IC 95 % por grupos | Reducción del costo total | Estado ROI/adopción |",
              "| --- | ---: | ---: | ---: | --- |"]
    for reference in REFERENCES:
        comparison = comparisons[reference]
        paired = comparison["paired"]
        lines.append(f"| {LABELS[reference]} | {100 * paired['success_rate_delta']:+.2f} pp | {interval(paired['success_rate_delta_95ci'], 100, ' pp')} | {percentage(paired['relative_total_cost_saving'])} | {comparison['roi']['adoption_status']} |")
    lines += ["", "| Cascada primaria menos referencia | Diferencia de intentos | Diferencia de tokens entrada | Diferencia de tokens salida | Diferencia de tiempo medio por episodio (s) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for reference in REFERENCES:
        delta = comparisons[reference]["resources"]
        lines.append(f"| {LABELS[reference]} | {_signed(delta['calls_delta'])} | {_signed(delta['input_tokens_delta'])} | {_signed(delta['output_tokens_delta'])} | {_signed(delta['mean_episode_seconds_delta'], 3)} |")
    lines += ["", f"Las diferencias son primaria menos referencia: una diferencia temporal negativa indica menos segundos observados. Los intervalos de éxito usan {config['bootstrap_replicates']} réplicas, semilla {config['seed']}, emparejadas por episodio y remuestreadas por source_group. Son diagnósticos descriptivos con pocos grupos, plantillas conocidas y desarrollo; no permiten confirmar superioridad, no inferioridad o transferencia general. Las diferencias de tokens y tiempo son descriptivas y no se convierten a ahorro económico.", "",
              "| Familia conocida | Tareas por política | Cascada | Siempre fuerte | Siempre barato | BERT transferido | Estructural transferido |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family in families:
        supports = {name: [episode for episode in grouped[name] if episode["family"] == family] for name in LABELS}
        if len({len(rows) for rows in supports.values()}) != 1:
            raise ValueError("El soporte por familia cambió entre políticas.")
        count = len(supports[PRIMARY])
        cells = [f"{sum(episode['success'] for episode in supports[name])}/{count}" for name in
                 (PRIMARY, "always_strong", "always_weak", "routellm_bert_transfer", "local_structural_transfer")]
        lines.append(f"| {family} | {count} | " + " | ".join(cells) + " |")
    lines += ["", "| Política | Finales incorrectos | Presupuesto agotado | Acciones inválidas | Errores de herramienta | Pruebas públicas fallidas | Repeticiones exactas de herramientas |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in LABELS:
        rows = trajectories[name]
        counts = [sum(row["final_incorrect"] for row in rows), sum(row["termination"] == "call_budget_exhausted" for row in rows)]
        counts += [sum(row[key] for row in rows) for key in ("invalid_action_calls", "tool_error_calls", "public_failed_test_calls", "repeated_identical_tool_actions")]
        lines.append(f"| {LABELS[name]} | " + " | ".join(str(value) for value in counts) + " |")
    lines += ["", "Finales incorrectos y presupuesto agotado cuentan episodios; las otras columnas cuentan acciones. Una repetición exacta puede ser legítima: el conteo por sí solo no demuestra un bucle. Los errores se solapan con las acciones, no son fallos adicionales que deban sumarse. Las trayectorias ordenadas por episodio están en comparisons.json y describen hechos observados, sin volver a puntuar respuestas privadas.", "",
              f"Se observaron {len(observed_patterns['aggregate_repeated_reads_without_write'])} episodios de agregación con lecturas repetidas, ninguna acción write_json y presupuesto agotado; {len(observed_patterns['repair_without_inspection'])} episodios de reparación sin inspect_function; y {len(observed_patterns['schema_objects_submitted_as_arguments'])} llamadas donde se pasó un objeto schema como arguments. Son patrones registrados, no una causa inferida de los errores de backend. La hipótesis siguiente es que un contrato de acciones más explícito y un estado compacto de progreso observable podrían reducir errores de formato y repetición. Habría que probarlo como intervención nueva, común a todos los controles, después de instrumentar los fallos de backend y comprobar la variabilidad; no se cambiaron estos resultados ni cortes.", "",
              "Las filas por familia muestran los conteos completos y no se usan para seleccionar otra primaria. La implementación de reparación cambia campos de una plantilla aritmética restringida; no representa programación libre de agentes. La siguiente evidencia útil exige una cohorte mayor y nuevas familias, catálogo y presupuesto predeclarados, y costos completos antes de evaluar ROI.", "",
              "Evidencia: [manifiesto y políticas](manifest.json), [journal](journal.json), [resumen original](summary.json), [comparaciones extendidas](comparisons.json), [snapshot de fuentes](source_bundle.zip) y [procedencia del informe](competition_report_provenance.json). Se verificó la cadena completa y se reprodujo summarize con las referencias fuerte, barato, BERT y estructural sin nuevos callbacks ni acceso a pesos del modelo.", ""]
    report_path, comparisons_path = directory / "competition_report.md", directory / "comparisons.json"
    _write_json(comparisons_path, extended)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    provenance = {"script_sha256": digest(Path(__file__)),
                  "audit_script_sha256": digest(ROOT / "tools/report_agent_benchmark.py"),
                  "metrics_source_sha256": digest(ROOT / "src/routerbench/agent_metrics.py"),
                  "manifest_sha256": digest(directory / "manifest.json"), "journal_sha256": digest(directory / "journal.json"),
                  "summary_sha256": digest(directory / "summary.json"), "source_bundle_sha256": digest(directory / "source_bundle.zip"),
                  "report_sha256": digest(report_path), "comparisons_sha256": digest(comparisons_path),
                  "freeze_sha256": manifest["freeze_sha256"], "journal_records_verified": len(journal["records"]),
                  "episodes_verified": len(episodes), "recomputed_baselines": list(REFERENCES),
                  "frozen_policy_rotation_verified": True,
                  "technical_predecessor_audit": predecessor,
                  "protocol_document": protocol,
                  "generation_executed": False, "training_executed": False, "tools_executed": False,
                  "models_deserialized": False, "raw_data_opened": False, "monetary_conversion_applied": False,
                  "primary_policy_changed": False}
    _write_json(directory / "competition_report_provenance.json", provenance)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/agent-local-competition-v2")
    args = parser.parse_args(argv)
    print(render_report(args.run_dir))


if __name__ == "__main__":
    main()
