"""Audit a completed Groq pilot and report a separate paid-token counterfactual."""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from report_agent_benchmark import audit_run, digest, milliseconds, read


LABELS = {"always_strong": "Siempre fuerte", "always_weak": "Siempre barato",
          "cheap_then_escalate": "Cascada reactiva"}
ARTIFACTS = ("manifest.json", "journal.json", "summary.json", "source_bundle.zip")
DEFAULT_TARIFFS = ROOT / "configs/groq_paid_tariffs_20261112.json"


def _count(value):
    return type(value) is int and value >= 0


def validate_tariffs(document):
    if (document.get("schema_version") != 1 or document.get("currency") != "USD"
            or document.get("unit") != "one million tokens" or not document.get("as_of")
            or type(document.get("models")) is not dict or not document["models"]):
        raise ValueError("El escenario requiere tarifas USD por millón, fecha y modelos explícitos.")
    for model, row in document["models"].items():
        if type(model) is not str or not model.startswith("groq/") or not row.get("source", "").startswith("https://"):
            raise ValueError("Cada tarifa requiere identidad Groq y fuente HTTPS.")
        for key in ("input_usd_per_million", "cached_input_usd_per_million", "output_usd_per_million"):
            value = row.get(key)
            if type(value) not in (int, float) or not Decimal(str(value)).is_finite() or value < 0:
                raise ValueError("Las tres tarifas deben ser finitas y no negativas.")
    return document


def event_scenario(event, tariffs):
    """Price one observed attempt once; unknown cache yields bounds, not imputation."""
    usage = event.get("usage", {})
    incoming, outgoing, cached = (usage.get(key) for key in
                                  ("input_tokens", "output_tokens", "cached_input_tokens"))
    result = {"model": event["model"], "call_index": event.get("call_index"),
              "is_retry": event.get("is_retry", False), "input_tokens": incoming,
              "output_tokens": outgoing, "cached_input_tokens": cached,
              "paid_token_cost_usd_range": None, "cache_basis": "unknown", "unknown_reason": None}
    if event.get("backend_failure", {}).get("category") == "model_identity":
        result["unknown_reason"] = "model_identity_unverified"
        return result
    tariff = tariffs["models"].get(event["model"])
    if tariff is None:
        result["unknown_reason"] = "tariff_unavailable"
        return result
    if not _count(incoming) or not _count(outgoing):
        result["unknown_reason"] = "input_or_output_usage_unknown"
        return result
    if cached is not None and (not _count(cached) or cached > incoming):
        result["unknown_reason"] = "cached_usage_invalid"
        return result
    input_rate, cached_rate, output_rate = (Decimal(str(tariff[key])) for key in
        ("input_usd_per_million", "cached_input_usd_per_million", "output_usd_per_million"))

    def cost(cached_count):
        return ((incoming - cached_count) * input_rate + cached_count * cached_rate
                + outgoing * output_rate) / Decimal(1_000_000)

    endpoints = [cost(incoming), cost(0)] if cached is None else [cost(cached), cost(cached)]
    result["paid_token_cost_usd_range"] = {"lower": float(min(endpoints)), "upper": float(max(endpoints))}
    result["cache_basis"] = "all_cached_to_none_cached_bounds" if cached is None else "observed_cache_count"
    return result


def _sum_ranges(rows):
    return {bound: float(sum((Decimal(str(row["paid_token_cost_usd_range"][bound]))
                             for row in rows), Decimal(0))) for bound in ("lower", "upper")}


def build_scenario(episodes, tariffs):
    validate_tariffs(tariffs)
    policies = {}
    for name in sorted({episode["policy"] for episode in episodes}):
        selected = [episode for episode in episodes if episode["policy"] == name]
        events = [{**event_scenario(event, tariffs), "episode_id": episode["episode_id"],
                   "episode_success": episode["success"]}
                  for episode in selected for event in episode["events"]]
        known = [event for event in events if event["paid_token_cost_usd_range"] is not None]
        subtotal = _sum_ranges(known)
        complete = bool(events) and len(known) == len(events)
        policies[name] = {
            "episodes": len(selected), "successes": sum(episode["success"] for episode in selected),
            "attempts": len(events), "retry_attempts": sum(event["is_retry"] for event in events),
            "priced_attempts": len(known), "unknown_attempts": len(events) - len(known),
            "known_cache_attempts": sum(event["cache_basis"] == "observed_cache_count" for event in known),
            "bounded_unknown_cache_attempts": sum(event["cache_basis"] == "all_cached_to_none_cached_bounds" for event in known),
            "paid_token_cost_usd_range": subtotal if complete else None,
            "known_subtotal_paid_token_cost_usd_range": subtotal,
            "full_operating_cost_usd": None, "observed_invoice_usd": None,
            "events": events,
        }
    return {"schema_version": 1, "nature": "CONTRAFACTUAL paid token scenario",
            "currency": "USD", "independent_validation": False, "roi_assessed": False,
            "tariffs": deepcopy(tariffs), "policies": policies,
            "scope": "All recorded attempts once, including failures and retries. No invoice or full operating cost.",
            "unknown_usage_rule": "Any unpriced attempt leaves the complete policy token cost unknown; known subtotal is not a total.",
            "unknown_cache_rule": "Bounds assume all observed input cached versus none cached; no measured cache count is imputed.",
            "excluded": ["router compute", "tool compute", "verification compute", "training", "data acquisition"]}


def _range(value):
    return "desconocido" if value is None else f"{value['lower']:.8f}–{value['upper']:.8f}"


def _tokens(metric, name):
    tokens = metric["tokens"][name]
    return f"{tokens['known_total']} ({tokens['known_calls']}/{metric['calls']})"


def render_report(directory, tariffs_path=DEFAULT_TARIFFS, context_note=None):
    directory, tariffs_path = Path(directory).resolve(), Path(tariffs_path).resolve()
    # audit_run refuses unfinished runs and never executes archived code or tools.
    manifest, journal, summary, episodes = audit_run(directory)
    config, metrics = manifest["configuration"], summary["metrics"]
    if (set(metrics["policies"]) != set(LABELS) or config["execution_mode"] != "external"
            or config.get("predictive_policy") is not None or config["independent_validation"] is not False):
        raise ValueError("Este informe requiere los tres controles externos, sin predictor v8 ni validación independiente.")
    core_hashes = {name: digest(directory / name) for name in ARTIFACTS}
    tariffs = validate_tariffs(read(tariffs_path))
    scenario = build_scenario(episodes, tariffs)
    scenario.update(freeze_sha256=manifest["freeze_sha256"], run_artifacts_sha256=core_hashes,
                    tariffs_sha256=digest(tariffs_path))
    note_path = Path(context_note).resolve() if context_note is not None else None
    note_text = note_path.read_text(encoding="utf-8") if note_path is not None else None
    note_hash = digest(note_path) if note_path is not None else None
    lines = [
        f"Relay · piloto Groq de desarrollo: {summary['n_tasks']} tareas y {summary['n_episodes']} episodios completos.", "",
        "La cohorte se reutilizó durante el ajuste del contrato de herramientas. No es un holdout ni permite atribuir causalmente los cambios de éxito a una intervención aislada. "
        "El resultado conserva todos los fallos y no valida una fase científica, generalización o publicación. "
        "No se entrenó ni evaluó un predictor Relay v8; se comparan tres controles con el mismo harness.", "",
        f"Se auditaron los manifiestos, fuentes congeladas, cadena del journal y métricas sin nuevas llamadas API. "
        f"Formato declarado: {config.get('generation_options', {}).get('action_format', 'json_object')}. "
        f"independent_validation=false. Con {summary['n_tasks']} tareas, un resultado mueve "
        f"{100 / summary['n_tasks']:.2f} puntos porcentuales; esta muestra no establece un margen de pérdida de 2 puntos.", "",
    ]
    if note_text:
        lines += [note_text.rstrip(), ""]
    lines += ["| Política | Éxitos/tareas | Intentos | Errores semánticos / infraestructura | Tokens entrada (cobertura) | Tokens salida (cobertura) | Routing p95 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in LABELS.items():
        metric = metrics["policies"][name]
        failures = metric.get("failure_diagnostics", {})
        semantic = sum(failures.get("semantic", {}).values())
        infrastructure = sum(failures.get("infrastructure", {}).values())
        lines.append(f"| {label} | {metric['successes']}/{metric['n_episodes']} | {metric['calls']} | {semantic} / {infrastructure} | "
                     f"{_tokens(metric, 'input_tokens')} | {_tokens(metric, 'output_tokens')} | {milliseconds(metric['routing_latency']['p95_seconds'])} |")
    lines += ["", "Los tokens son subtotales observados; entre paréntesis aparece la cobertura de llamadas. Los errores de formato/ejecución no cuentan respuestas finales incorrectas. "
              "Routing p95 mide decisiones de política, excluye reintentos de infraestructura fijados al mismo modelo y no mide la latencia API o las esperas de cuota.", "",
              "El siguiente escenario es CONTRAFACTUAL: aplica tarifas pagadas a los tokens observados. No es una factura, un cargo del plan Free, costo operacional completo ni ROI. "
              "La declaración de cuenta Free no sustituye una factura observada. Los costos locales de router, herramientas y verificación permanecen desconocidos.", "",
              "| Política | Tokens a tarifa pagada, rango USD | Subtotal conocido, rango USD | Intentos valorados / total | Caché desconocida acotada |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, label in LABELS.items():
        row = scenario["policies"][name]
        lines.append(f"| {label} | {_range(row['paid_token_cost_usd_range'])} | {_range(row['known_subtotal_paid_token_cost_usd_range'])} | "
                     f"{row['priced_attempts']}/{row['attempts']} | {row['bounded_unknown_cache_attempts']} |")
    lines += ["", "Con caché medida se aplica su cantidad exacta, sin cobrarla otra vez como entrada ordinaria. Si falta ese contador, la banda va de toda la entrada en caché a ninguna. "
              "Si falta uso o identidad en algún intento, el total de la política queda desconocido; su subtotal conocido no lo reemplaza. Cada intento fallido o reintento se suma una sola vez.", ""]
    references = []
    for model, tariff in tariffs["models"].items():
        references.append(f"[{model}]({tariff['source']}): entrada {tariff['input_usd_per_million']}, "
                          f"entrada en caché {tariff['cached_input_usd_per_million']} y salida {tariff['output_usd_per_million']} USD por millón")
    lines += [f"Tarifas de referencia fechadas {tariffs['as_of']}: " + "; ".join(references) + ".", "",
              "Evidencia: [manifiesto](manifest.json), [journal](journal.json), [resumen original](summary.json), "
              "[escenario pagado separado](paid_token_scenario.json) y [procedencia y hashes](pilot_report_provenance.json). "
              "Los archivos originales no se modifican y los hashes acreditan consistencia, no autoría.", ""]
    if {name: digest(directory / name) for name in ARTIFACTS} != core_hashes:
        raise ValueError("La evidencia cambió durante la lectura; no se escribe el informe.")
    report_path = directory / "pilot_report.md"
    scenario_path = directory / "paid_token_scenario.json"
    scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    provenance = {
        "schema_version": 1, "freeze_sha256": manifest["freeze_sha256"], "run_artifacts_sha256": core_hashes,
        "renderer_sha256": digest(Path(__file__)), "audit_renderer_sha256": digest(ROOT / "tools/report_agent_benchmark.py"),
        "metrics_source_sha256": digest(ROOT / "src/routerbench/agent_metrics.py"),
        "tariffs_path": str(tariffs_path), "tariffs_sha256": digest(tariffs_path),
        "context_note_path": str(note_path) if note_path else None, "context_note_sha256": note_hash,
        "pilot_report_sha256": digest(report_path), "paid_token_scenario_sha256": digest(scenario_path),
        "episodes_audited": len(episodes), "journal_records_audited": len(journal["records"]),
        "api_calls_executed": False, "tools_executed": False, "training_executed": False,
        "raw_data_opened": False, "original_run_files_modified": False, "independent_validation": False,
    }
    (directory / "pilot_report_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tariffs", type=Path, default=DEFAULT_TARIFFS)
    parser.add_argument("--context-note", type=Path, help="Optional development-history note; its contents and hash enter the report.")
    args = parser.parse_args(argv)
    print(render_report(args.run_dir, args.tariffs, args.context_note))


if __name__ == "__main__":
    main()
