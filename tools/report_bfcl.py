"""Audit saved BFCL controls without generation, tool replay or private grading.

Reports completed and in-progress runs. Missing attempts/results never become
model failures. Optional prices define a no-cache token scenario, not invoices,
Free-tier savings, full operating cost, or evidence of predictive routing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from routerbench.agent_evaluation import _Journal, object_hash
from routerbench.bfcl_harness import (
    BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade, BFCLRequest, BFCLState,
    _audit_journal, _session_summary, PROTOCOL, UPSTREAM_COMMIT,
)

POLICIES = {p.name: p for p in (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade)}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Clave JSON duplicada: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Numero JSON no finito.")


class Snapshot:
    def __init__(self):
        self.hashes = {}

    def raw(self, path):
        path = str(Path(path).resolve())
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in self.hashes and self.hashes[path] != digest:
            raise ValueError("La corrida cambio durante la auditoria; repetir la lectura.")
        self.hashes[path] = digest
        return raw

    def read(self, path, *, sealed=False):
        value = json.loads(self.raw(path).decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
        if sealed:
            if (type(value) is not dict or set(value) != {"payload", "payload_sha256"}
                    or object_hash(value["payload"]) != value["payload_sha256"]):
                raise ValueError("Sobre o hash JSON alterado.")
            return value["payload"]
        return value

    def verify(self):
        for path, expected in self.hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("La corrida cambio durante la auditoria; repetir la lectura.")


def _manifest(directory, snapshot):
    manifest = snapshot.read(directory / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "freeze_sha256"}
    if (manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL
            or manifest.get("upstream_commit") != UPSTREAM_COMMIT
            or object_hash(body) != manifest.get("freeze_sha256")
            or object_hash(manifest["cases"]) != manifest["cases_sha256"]):
        raise ValueError("Manifiesto BFCL o poblacion alterados.")
    archive_path = directory / "source_bundle.zip"
    if hashlib.sha256(snapshot.raw(archive_path)).hexdigest() != manifest["source_bundle_sha256"]:
        raise ValueError("Snapshot de fuentes alterado.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {*manifest["source_files_sha256"], "configuration.json"}:
            raise ValueError("Inventario del snapshot incompatible.")
        for name, expected in manifest["source_files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Contenido del snapshot alterado.")
        if json.loads(archive.read("configuration.json"), object_pairs_hook=_unique, parse_constant=_nonfinite) != manifest["configuration"]:
            raise ValueError("Configuracion del snapshot incompatible.")
    return manifest


def _prices(tariffs, models):
    if tariffs is None:
        return None
    if type(tariffs) is not dict or tariffs.get("currency") != "USD" or type(tariffs.get("models")) is not dict:
        raise ValueError("Declarar tarifas USD por millon de tokens, con models.")
    prices = {}
    for model in models:
        entry = tariffs["models"].get(model, {})
        values = tuple(entry.get(k) for k in ("input_usd_per_million", "output_usd_per_million"))
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Cada modelo requiere tarifas finitas input/output.")
        prices[model] = values
    return prices


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def _account(events, prices):
    result = {"calls": len(events), "known_input_tokens": 0, "known_output_tokens": 0,
        "input_unknown_calls": 0, "output_unknown_calls": 0, "known_tariff_subtotal_usd": 0.,
        "tariff_unknown_calls": 0, "known_reported_provider_subtotal_usd": 0., "reported_provider_unknown_calls": 0,
        "known_phase_seconds": 0., "phase_time_incomplete_calls": 0, "routing_decisions": 0,
        "pinned_infrastructure_attempts": 0, "routing_seconds_unknown": 0,
        "no_provider_call_attempts": 0, "provider_calls_confirmed": 0, "provider_call_status_unknown": 0}
    routing = []
    for event in events:
        usage = event.get("usage") or {}
        metadata = usage.get("provider_metadata") or {}
        no_call = metadata.get("provider_call_started") is False
        if no_call:
            result["no_provider_call_attempts"] += 1
        elif metadata.get("provider_call_started") is True or metadata.get("returned_model"):
            result["provider_calls_confirmed"] += 1
        else:
            result["provider_call_status_unknown"] += 1
        # An explicit local rejection is evidence of no provider consumption,
        # unlike an absent measurement after a possibly transmitted request.
        counts = [0, 0] if no_call else [usage.get("input_tokens"), usage.get("output_tokens")]
        for kind, count in zip(("input", "output"), counts):
            if type(count) is int and count >= 0:
                result[f"known_{kind}_tokens"] += count
            else:
                result[f"{kind}_unknown_calls"] += 1
        identity_unknown = (event.get("backend_failure") or {}).get("category") == "model_identity"
        if prices is None or identity_unknown or any(type(c) is not int or c < 0 for c in counts):
            result["tariff_unknown_calls"] += 1
        else:
            result["known_tariff_subtotal_usd"] += sum(c*p/1e6 for c, p in zip(counts, prices[event["model"]]))
        reported = 0. if no_call else event.get("reported_llm_cost_usd")
        if reported is None:
            result["reported_provider_unknown_calls"] += 1
        else:
            result["known_reported_provider_subtotal_usd"] += reported
        durations = [event.get(k) for k in ("router_seconds", "generation_seconds", "tool_seconds", "verification_seconds", "retry_wait_seconds")]
        result["known_phase_seconds"] += sum(v for v in durations if v is not None)
        result["phase_time_incomplete_calls"] += any(v is None for v in durations)
        if event.get("routing_source") == "pinned_infrastructure_retry":
            result["pinned_infrastructure_attempts"] += 1
        else:
            result["routing_decisions"] += 1
            if event.get("router_seconds") is None:
                result["routing_seconds_unknown"] += 1
            else:
                routing.append(event["router_seconds"])
    result.update(input_tokens=None if result["input_unknown_calls"] else result["known_input_tokens"],
        output_tokens=None if result["output_unknown_calls"] else result["known_output_tokens"],
        tariff_no_cache_usd=None if prices is None or result["tariff_unknown_calls"] else result["known_tariff_subtotal_usd"],
        reported_provider_usd=None if result["reported_provider_unknown_calls"] else result["known_reported_provider_subtotal_usd"],
        measured_phase_seconds=None if result["phase_time_incomplete_calls"] else result["known_phase_seconds"],
        wall_clock_total_seconds=None, routing_p95_seconds=_percentile(routing, .95),
        provider_calls=None if result["provider_call_status_unknown"] else result["provider_calls_confirmed"],
        routing_p95_scope="policy decisions only; pinned infrastructure retries excluded; known timings only")
    return result


def _public_request(value):
    state = deepcopy(value["state"])
    for key in ("previous_errors", "infrastructure_errors", "failed_tiers"):
        state[key] = tuple(state[key])
    return BFCLRequest(tuple(value["messages"]), tuple(value["tools"]), BFCLState(**state))


def _response_variation(events):
    """Compare completed attempts across policies without exporting response text.

    Arguments are raw strings: different whitespace is textual variation, not
    evidence of different meaning. Provider call IDs are deliberately omitted.
    """
    grouped = {}
    for event in events:
        grouped.setdefault((event["model"], event["request_sha256"]), []).append(event)
    groups = []
    for (model, request_hash), attempts in sorted(grouped.items()):
        policies = sorted({event["policy"] for event in attempts})
        if len(policies) < 2:
            continue
        signatures = []
        for event in attempts:
            response = event.get("response")
            signature = {"policy": event["policy"], "episode_id": event["episode_id"],
                "call_index": event["call_index"], "response_present": response is not None,
                "text_sha256": None, "native_call_signature_sha256": None, "response_signature_sha256": None}
            if response is not None:
                native = []
                for call in response.get("tool_calls", []):
                    function = call.get("function")
                    function = function if type(function) is dict else {}
                    native.append({"name": function.get("name"), "arguments_raw": function.get("arguments")})
                text_value = response.get("content")
                signature.update(text_sha256=object_hash(text_value),
                    native_call_signature_sha256=object_hash(native),
                    response_signature_sha256=object_hash({"content": text_value, "native_function_calls": native}))
            failure = event.get("backend_failure") or {}
            signature["error_signature_sha256"] = object_hash({"error_type": event.get("error_type"),
                "error_scope": event.get("error_scope"), "backend_category": failure.get("category"),
                "backend_status_code": failure.get("status_code"), "tool_errors": [
                    {"ok": result.get("ok"), "error_type": result.get("error_type")}
                    for result in event.get("tool_results", []) if result.get("ok") is not True or result.get("error_type")]})
            signatures.append(signature)
        responses = [row for row in signatures if row["response_present"]]
        response_policies = sorted({row["policy"] for row in responses})
        counts = {key: len({row[field] for row in responses}) for key, field in (
            ("distinct_texts", "text_sha256"), ("distinct_native_call_signatures", "native_call_signature_sha256"),
            ("distinct_response_signatures", "response_signature_sha256"))}
        errors = len({row["error_signature_sha256"] for row in signatures})
        comparable = len(response_policies) >= 2
        groups.append({"model": model, "request_sha256": request_hash, "policies": policies,
            "completed_attempts": len(attempts), "responses_observed": len(responses),
            "missing_responses": len(attempts)-len(responses), "response_policies": response_policies,
            "response_comparable_across_policies": comparable, **counts, "distinct_error_signatures": errors,
            "response_varies": comparable and counts["distinct_response_signatures"] > 1,
            "text_varies": comparable and counts["distinct_texts"] > 1,
            "native_calls_vary": comparable and counts["distinct_native_call_signatures"] > 1,
            "errors_vary": errors > 1, "attempt_signatures": signatures})
    return {"grouping": ["model", "request_sha256"], "cross_policy_groups": len(groups),
        "groups_with_cross_policy_responses": sum(g["response_comparable_across_policies"] for g in groups),
        "groups_with_response_variation": sum(g["response_varies"] for g in groups),
        "groups_with_text_variation": sum(g["text_varies"] for g in groups),
        "groups_with_native_call_variation": sum(g["native_calls_vary"] for g in groups),
        "groups_with_error_variation": sum(g["errors_vary"] for g in groups), "groups": groups,
        "request_scope": "exact full public-request hash, including state and prior tool-call IDs; no input normalization; different hashes are not compared",
        "signature_scope": "exact content plus ordered function name/raw argument strings; tool-call IDs, usage, timings and provider metadata excluded; textual, not semantic equivalence",
        "error_scope": "error type/scope, backend category/status and ordered tool error status only; absent responses remain absent",
        "interpretation": "observed variation only; no causal attribution to routing, sampling-mechanism diagnosis or independence claim; temperature zero is not used as proof of identical responses"}


def audit_bfcl(directory, *, tariffs_path=None):
    """Produce an audited snapshot, including unstarted and unknown outcomes."""
    directory, snapshot = Path(directory).resolve(), Snapshot()
    manifest = _manifest(directory, snapshot)
    config, cases = manifest["configuration"], manifest["cases"]
    sequence = manifest["policy_sequence"]
    if not sequence or len(set(sequence)) != len(sequence) or set(sequence) != set(manifest["policies"]) or set(sequence)-set(POLICIES):
        raise ValueError("Solo controles BFCL declarados y unicos son compatibles.")
    policies = [POLICIES[name](config["model_catalog"]) for name in sequence]
    if manifest["policies"] != {policy.name: policy.manifest() for policy in policies}:
        raise ValueError("Las politicas no corresponden a sus controles declarados.")
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Poblacion vacia o conversaciones duplicadas.")
    allowed = {(case["id"], policy.name) for case in cases for policy in policies}
    if (directory / "journal.json").exists():
        snapshot.read(directory / "journal.json", sealed=True)
    journal = _Journal(directory, manifest["freeze_sha256"], allowed)
    _audit_journal(cases, policies, journal, config)
    tariffs = snapshot.read(tariffs_path) if tariffs_path else None
    prices = _prices(tariffs, config["model_catalog"])
    episodes, all_events, closed, events_by_pair, response_events = [], [], [], {}, []
    for case in cases:
        for policy in policies:
            records = journal.for_key((case["id"], policy.name))
            starts = [r["data"] for r in records if r["kind"] == "call_started"]
            completed = [r["data"] for r in records if r["kind"] == "call_completed"]
            response_events.extend({**event, "episode_id": case["id"], "policy": policy.name} for event in completed)
            for index, event in enumerate(starts):
                if event.get("routing_source") == "pinned_infrastructure_retry":
                    if index == 0 or event["model"] != starts[index-1]["model"]:
                        raise ValueError("El retry de infraestructura cambio de modelo.")
                elif event["model"] != policy.select(_public_request(event["request"])):
                    raise ValueError("La llamada no respeta la politica congelada.")
            events = completed + [{**event, "usage": {}, "reported_llm_cost_usd": None,
                "generation_seconds": None, "verification_seconds": None, "tool_seconds": None} for event in starts[len(completed):]]
            all_events.extend(events)
            events_by_pair[(case["id"], policy.name)] = events
            outcome = records[-1]["data"] if records and records[-1]["kind"] == "episode_completed" else None
            if outcome is not None:
                closed.append(outcome)
            status = ("censored" if outcome["censored_reason"] else "completed") if outcome else (
                     "pending" if len(starts) > len(completed) else "in_progress" if starts else "not_started")
            episodes.append({"episode_id": case["id"], "source_group": case["source_group"], "policy": policy.name,
                "status": status, "success": outcome["success"] if outcome else None,
                "operational_success": outcome["operational_success"] if outcome else None,
                "termination": outcome["termination"] if outcome else None,
                "censored_reason": outcome["censored_reason"] if outcome else None,
                "calls_started": len(starts), "calls_completed": len(completed), "calls_without_result": len(starts)-len(completed),
                "semantic_failures": dict(Counter(e["error_type"] for e in completed if e.get("error_scope") == "semantic")),
                "infrastructure_failures": dict(Counter((e.get("backend_failure") or {}).get("category", "local_resource")
                    for e in completed if e.get("error_scope") == "infrastructure")),
                "accounting": _account(events, prices)})
    summary, summary_current = None, False
    if (directory / "summary.json").exists():
        summary = snapshot.read(directory / "summary.json", sealed=True)
        prefix_count = summary.get("journal_records", len(journal.records))
        if type(prefix_count) is not int or not 0 <= prefix_count <= len(journal.records):
            raise ValueError("Prefijo del resumen fuera del journal.")
        prefix = journal.records[:prefix_count]
        prefix_closed = [r["data"] for r in prefix if r["kind"] == "episode_completed"]
        expected = {"schema_version": 1, "protocol": PROTOCOL, "freeze_sha256": manifest["freeze_sha256"],
            "journal_sha256": object_hash(prefix), "episodes": prefix_closed,
            "generation_calls": sum(r["kind"] == "call_started" for r in prefix),
            "maximum_generation_calls": manifest["maximum_generation_calls"], "independent_validation": False,
            "execution_mode": config["execution_mode"], "scope": manifest["scope"]}
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError("El resumen no reproduce los episodios del journal.")
        planned = [[case["id"], policy.name] for case in cases for policy in policies]
        completed_pairs = [[episode["episode_id"], episode["policy"]] for episode in prefix_closed]
        if "planned_pairs" in summary and summary["planned_pairs"] != planned:
            raise ValueError("El resumen cambio la poblacion planificada.")
        if "completed_pairs" in summary and summary["completed_pairs"] != completed_pairs:
            raise ValueError("El resumen cambio sus cierres declarados.")
        if "execution_complete" in summary and summary["execution_complete"] != (len(prefix_closed) == len(planned)):
            raise ValueError("El resumen cambio su cobertura declarada.")
        summary_current = prefix_count == len(journal.records)
        if "planned_pairs" in summary:
            # Current BFCL sessions, including partial ones, are immutable once
            # their summary commits. The helper recomputes closure/coverage only.
            expected_session = _session_summary(manifest, journal, planned, config, summary.get("session_stop_reason"))
            if summary != expected_session:
                raise ValueError("El resumen parcial cambio sus pares, causa o journal cerrado.")
    by_policy = {}
    for policy in policies:
        rows = [row for row in episodes if row["policy"] == policy.name]
        done = [row for row in rows if row["status"] in {"completed", "censored"}]
        successes = sum(row["success"] is True for row in done)
        known_count = sum(row["success"] is not None for row in done)
        actual_events = [event for case in cases for event in events_by_pair[(case["id"], policy.name)]]
        by_policy[policy.name] = {"planned_conversations": len(rows), "closed_conversations": len(done),
            "successes": successes, "known_failures": sum(row["success"] is False for row in done),
            "censored_conversations": sum(row["status"] == "censored" for row in rows),
            "not_started_conversations": sum(row["status"] == "not_started" for row in rows),
            "unfinished_conversations": sum(row["status"] in {"pending", "in_progress"} for row in rows),
            "operational_success_rate": successes/len(rows) if len(done) == len(rows) else None,
            "success_rate_among_closed": successes/len(done) if done else None,
            "success_rate_among_closed_scope": "operational; censored closures remain in denominator, not capability negatives",
            "known_outcome_conversations": known_count,
            "success_rate_with_known_outcome": successes/known_count if known_count else None,
            "accounting": _account(actual_events, prices)}
    paired = {}
    for first_index, first in enumerate(sequence):
        for second in sequence[first_index+1:]:
            first_rows = {e["episode_id"]: e for e in episodes if e["policy"] == first}
            second_rows = {e["episode_id"]: e for e in episodes if e["policy"] == second}
            pairs = [(first_rows[case["id"]], second_rows[case["id"]]) for case in cases
                     if first_rows[case["id"]]["status"] in {"completed", "censored"}
                     and second_rows[case["id"]]["status"] in {"completed", "censored"}]
            known = [(a, b) for a, b in pairs if a["success"] is not None and b["success"] is not None]
            first_only = sum(a["success"] and not b["success"] for a, b in known)
            second_only = sum(b["success"] and not a["success"] for a, b in known)
            accounting_first = _account([event for a, _ in pairs
                for event in events_by_pair[(a["episode_id"], first)]], prices)
            accounting_second = _account([event for _, b in pairs
                for event in events_by_pair[(b["episode_id"], second)]], prices)
            first_cost, second_cost = accounting_first["tariff_no_cache_usd"], accounting_second["tariff_no_cache_usd"]
            cost_known = bool(pairs) and first_cost is not None and second_cost is not None
            ratio = second_cost/first_cost if cost_known and first_cost > 0 else None
            paired[f"{first}__{second}"] = {"first_policy": first, "second_policy": second,
                "planned_pairs": len(cases), "closed_pairs": len(pairs), "uncensored_pairs": len(known),
                "censored_pairs": len(pairs)-len(known), "first_only_success": first_only,
                "second_only_success": second_only, "both_success": sum(a["success"] and b["success"] for a, b in known),
                "both_failure": sum(not a["success"] and not b["success"] for a, b in known),
                "known_outcome_success_delta": (first_only-second_only)/len(known) if known else None,
                "known_outcome_success_delta_scope": "uncensored pairs only; first minus second",
                "accounting_first": accounting_first, "accounting_second": accounting_second,
                "resource_pair_count": len(pairs), "resource_episode_ids": [a["episode_id"] for a, _ in pairs],
                "resource_scope": "same closed conversations on both sides; includes failures, censures and their attempts; no success filtering",
                "tariff_no_cache_cost_difference_second_minus_first_usd": second_cost-first_cost if cost_known else None,
                "tariff_no_cache_cost_ratio_second_to_first": ratio,
                "tariff_no_cache_savings_fraction_second_vs_first": 1-ratio if ratio is not None else None,
                "cost_comparison_scope": "no-cache token tariff scenario only; complete usage required; ratio requires positive first cost; not real Free savings or ROI",
                "source_groups": len({a["source_group"] for a, _ in known}), "confidence_interval": None,
                "uncertainty_scope": "exploratory paired discordance counts; no noninferiority or two-percentage-point claim"}
    snapshot.verify()
    complete = len(closed) == len(allowed)
    return {"schema_version": 1, "protocol": PROTOCOL, "directory": str(directory),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": manifest["freeze_sha256"],
        "status": "complete" if complete else "stopped" if summary_current else "in_progress" if all_events else "prepared",
        "summary_present": summary is not None, "summary_current": summary_current,
        "planned_conversations": len(cases), "policies": by_policy,
        "episodes": episodes, "paired_comparisons": paired, "accounting": _account(all_events, prices),
        "same_request_response_variation": _response_variation(response_events),
        "tariffs": tariffs, "execution_mode": config["execution_mode"], "independent_validation": False,
        "declared_free_account": config.get("spending", {}).get("account_is_free_tier") is True,
        "session_stop_reason": summary.get("session_stop_reason") if summary else None,
        "session_stop_reason_scope": "current summary" if summary_current else "previous summary" if summary else "unavailable",
        "source_file_sha256": snapshot.hashes, "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cost_scope": "optional no-cache provider token tariff scenario; every attempt once; not invoice, Free savings or full cost",
        "latency_scope": "sum of recorded phases and transport wait; quota pacing already in generation; no measured episode wall-clock",
        "provider_call_scope": "harness attempts differ from provider calls; explicit local rejections are zero provider consumption, missing status stays unknown",
        "scope": "local BFCL control comparison; exploratory selected cohort; no learned router or competitor superiority"}


def _number(value, digits=6):
    return "desconocido" if value is None else f"{value:.{digits}f}"


def render_report(report):
    lines = ["Comparacion exploratoria de controles BFCL de Relay.", "",
        f"Estado: {report['status']}. Poblacion congelada: {report['planned_conversations']} conversaciones; "
        f"{report['accounting']['calls']} intentos registrados. Es una adaptacion local de BFCL, no una puntuacion oficial.", "",
        "| Politica | Exitos / cerradas | Fallos conocidos | Censuradas | Sin iniciar | En curso | Intentos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in report["policies"].items():
        lines.append(f"| {name} | {row['successes']}/{row['closed_conversations']} | {row['known_failures']} | "
            f"{row['censored_conversations']} | {row['not_started_conversations']} | {row['unfinished_conversations']} | {row['accounting']['calls']} |")
    lines += ["", "Las censuras conservan sus intentos y costos; no son etiquetas negativas de capacidad. "
        "Exitos/cerradas es una fraccion operativa que incluye censuras en el denominador; la tasa con desenlace conocido se distingue en JSON. "
        "Los episodios sin iniciar o en curso no se convierten en fallos. Las tasas de politicas con distinta cobertura no son una comparacion pareada.", "",
        "| Politica | Input conocido | Output conocido | Intentos con tokens faltantes (input/output) | USD escenario sin cache | Routing p95 (ms) | Segundos de fases registradas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in report["policies"].items():
        a = row["accounting"]
        lines.append(f"| {name} | {a['known_input_tokens']} | {a['known_output_tokens']} | "
            f"{a['input_unknown_calls']}/{a['output_unknown_calls']} | {_number(a['tariff_no_cache_usd'])} | "
            f"{_number(None if a['routing_p95_seconds'] is None else a['routing_p95_seconds']*1000, 3)} | {_number(a['measured_phase_seconds'], 3)} |")
    lines += ["", "Los USD son un escenario con las tarifas declaradas y toda la entrada sin cache; no son facturacion ni ahorro real de una cuenta Free. "
        "Faltan costos de computo, herramientas y verificacion. Un uso ausente deja desconocido el total. No se calcula ROI.", "",
        "El p95 de routing incluye decisiones de politica y excluye los retries de infraestructura con modelo fijado. "
        "Los segundos suman fases registradas y esperas de transporte; el pacing de cuota ya esta dentro de generacion. "
        "El harness no registro wall-clock completo por episodio, por lo que no se presenta esa suma como latencia end-to-end.", ""]
    a = report["accounting"]
    lines += [f"Intentos con rechazo local explicito sin llamada al proveedor: {a['no_provider_call_attempts']}; "
        f"llamadas al proveedor confirmadas por metadatos: {a['provider_calls_confirmed']}; "
        f"intentos con estado de envio desconocido: {a['provider_call_status_unknown']}. "
        "Ausencia de consumo tras un rechazo local explicito se registra como cero; uso desconocido despues de un posible envio permanece desconocido.", ""]
    for row in report["paired_comparisons"].values():
        lines.append(f"{row['first_policy']} frente a {row['second_policy']}: {row['closed_pairs']} pares cerrados de "
            f"{row['planned_pairs']}; {row['uncensored_pairs']} con ambos desenlaces conocidos. Solo acierta la primera en "
            f"{row['first_only_success']} y solo la segunda en {row['second_only_success']}. "
            f"Pares censurados: {row['censored_pairs']}.")
    lines += ["", "Los recursos siguientes usan exactamente las mismas conversaciones cerradas en ambos lados, "
        "incluyendo fallos y censuras; no se filtran por exito. La comparacion de exito anterior usa solo los pares con ambos desenlaces conocidos.", "",
        "| Primera / segunda politica | Pares cerrados comunes | Intentos primera / segunda | USD primera | USD segunda | Razon segunda/primera | Reduccion del escenario (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["paired_comparisons"].values():
        first, second = row["accounting_first"], row["accounting_second"]
        saving = row["tariff_no_cache_savings_fraction_second_vs_first"]
        lines.append(f"| {row['first_policy']} / {row['second_policy']} | {row['resource_pair_count']} | "
            f"{first['calls']}/{second['calls']} | {_number(first['tariff_no_cache_usd'])} | {_number(second['tariff_no_cache_usd'])} | "
            f"{_number(row['tariff_no_cache_cost_ratio_second_to_first'], 4)} | {_number(None if saving is None else 100*saving, 2)} |")
    lines += ["", "La reduccion se refiere exclusivamente al escenario de tokens sin cache de la segunda politica frente a la primera. "
        "Si falta algun consumo o el costo de la primera es cero, la razon y la reduccion quedan desconocidas. "
        "No es ahorro real en Free ni ROI; un menor costo puede acompanarse de fallos o censuras."]
    variation = report["same_request_response_variation"]
    causal_note = ("Se observaron respuestas distintas ante el mismo modelo y solicitud; no se atribuye toda diferencia de resultado al routing. "
        if variation["groups_with_response_variation"] else
        "No se observaron respuestas distintas en los grupos comparables de este snapshot; esto no prueba determinismo ni causalidad. ")
    lines += ["", f"Respuestas al mismo modelo y request_sha256: {variation['cross_policy_groups']} grupos presentes en al menos dos politicas; "
        f"{variation['groups_with_cross_policy_responses']} tienen respuestas observadas en ambas. "
        f"Entre estos, {variation['groups_with_response_variation']} muestran respuestas distintas "
        f"({variation['groups_with_text_variation']} en texto; {variation['groups_with_native_call_variation']} en firmas de funciones). "
        f"Hay {variation['groups_with_error_variation']} grupos con estados de error distintos.", "",
        "La firma compara contenido y la lista ordenada de nombres de funcion/argumentos sin interpretar el JSON; "
        "excluye IDs de llamadas, uso, tiempos y metadatos del proveedor. No demuestra equivalencia ni diferencia semantica. "
        "El reporte conserva hashes, no copia prompts ni respuestas. Las respuestas ausentes no se cuentan como texto distinto. "
        "El hash de entrada es exacto e incluye estado e IDs previos: solicitudes con hashes distintos quedan fuera de esta comparacion. "
        + causal_note +
        "Temperatura 0 no se usa aqui como prueba de respuestas identicas ni de independencia; esta comparacion no aisla un efecto causal del router."]
    lines += ["", "La incertidumbre se muestra mediante cobertura y discordancias pareadas sobre la misma conversacion. "
        "Estos conteos exploratorios no demuestran no inferioridad a dos puntos ni superioridad de un router aprendido o de Relay frente a otros productos. "
        "Los fixtures simulados verifican contratos; no miden capacidad de modelos.", "",
        "Se comprobaron manifiesto, snapshot de fuentes, cadena de journals, solicitudes previas a cada llamada, politicas y costos/desenlaces persistidos. "
        "El resumen, cuando existe, debe reproducirse desde el journal. No se llamaron APIs, herramientas ni evaluadores privados. "
        "Los hashes comprueban consistencia local, no autoria."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tariffs", type=Path)
    parser.add_argument("--output", type=Path, help="New directory for immutable report.json/report.md; otherwise print summary only.")
    args = parser.parse_args(argv)
    report = audit_bfcl(args.run, tariffs_path=args.tariffs)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.json").write_text(json.dumps({"payload": report, "payload_sha256": object_hash(report)},
            ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
        (args.output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "planned_conversations": report["planned_conversations"],
        "policies": report["policies"], "paired_comparisons": report["paired_comparisons"],
        "output": str(args.output) if args.output else None}, ensure_ascii=False, allow_nan=False, indent=2))
    return report


if __name__ == "__main__":
    main()
