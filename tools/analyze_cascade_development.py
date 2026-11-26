"""Audit existing paired observations and write a NEW exploratory cascade report.

No generation, policy fitting, new held-out outcomes, or edits to old artifacts.
The public-only cascade is a retrospective hypothesis, not a validated product.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from relay_router.cascade_evaluation import replay_public_cascade, exact_harm_upper, zero_loss_required_groups
from relay_router.harness_episode import run_check
import report_adaptive_recovery as recovery_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_report():
    primary = recovery_report.primary
    run, suite_path, policy_path = primary.DEFAULT_RUN, primary.DEFAULT_SUITE, primary.DEFAULT_POLICY
    # Reuse the audited reader and its write-ahead HTTP/source-hash checks.
    original = primary.make_report(run, suite_path, policy_path)
    state = recovery_report.recovery.load_original(run, suite_path, policy_path)
    combined, provenance, events, audit = recovery_report.audit_recovery(state, run / "pre_dispatch_recovery")
    reference_path = primary.runner.BERT_REFERENCE_PATH
    if digest(reference_path) != state["frozen"]["contract"]["bert_reference"]["sha256"]:
        raise ValueError("The historical BERT reference differs from its frozen manifest")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    strong = state["suite"]["baseline_model"]
    cheap = next(model for model in state["suite"]["models"] if model != strong)
    reference_routes = {"historical_relay_adaptive": {row["task_id"]: row["model"] for row in state["frozen"]["routes"]}}
    for threshold in (.50, .58):
        reference_routes[f"historical_bert_{threshold:.2f}"] = {
            row["task_id"]: strong if row["truncated_tokens"] > 0 or row["score"] >= threshold else cheap
            for row in reference["rows"]}
    # A second DEVELOPMENT hypothesis: only try the cheap model when the
    # operational tests reproduce a failure in the unchanged original source.
    # No private tests, model outputs or acceptance labels enter this check.
    initial_checks = {}
    with tempfile.TemporaryDirectory(prefix="relay-cascade-initial-") as temporary:
        for position, task in enumerate(state["suite"]["tasks"]):
            result = run_check(task["initial_source"], task["public_tests"], directory=Path(temporary) / str(position), timeout=8)
            status = "passed" if result["passed"] else "failed" if result["category"] == "test_failure" else "inconclusive"
            initial_checks[task["task_id"]] = {
                "status": status, "category": result["category"], "seconds": result["seconds"], "timeout_seconds": 8,
                "source_sha256": hashlib.sha256(task["initial_source"].encode("utf-8")).hexdigest(),
                "public_tests_sha256": hashlib.sha256(task["public_tests"].encode("utf-8")).hexdigest(),
                "private_tests_used": False,
            }
    report = replay_public_cascade(combined, candidate_model=cheap, baseline_model=strong,
                                   reference_routes=reference_routes, initial_public_checks=initial_checks)
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["provenance"] = {
        "original_freeze_id": state["frozen"]["freeze_id"], "original_primary_report_id": original["report_id"],
        "recovery_audit": audit, "recovered_episode_provenance": provenance,
        "source_sha256": {name: digest(ROOT / name) for name in (
            "src/relay_router/cascade_evaluation.py", "tools/analyze_cascade_development.py", "docs/openhands_cascade_v1.md")},
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in (
            run / "observations.jsonl", run / "http_attempts.jsonl", run / "frozen_routes.json",
            run / "pre_dispatch_recovery/observations.jsonl", run / "pre_dispatch_recovery/http_attempts.jsonl",
            reference_path, suite_path)},
        "all_eligible_initialization_recoveries_included": True, "original_artifacts_modified": False,
        "historical_benchmark_split_now_used_for_cascade_development": True,
        "bert_reference": reference["reference"],
        "methodology_skill": {"path": ".agents/skills/scientific-critical-thinking/SKILL.md",
                              "sha256": digest(ROOT / ".agents/skills/scientific-critical-thinking/SKILL.md"),
                              "role": "local methodological review guidance; not a primary empirical source"},
        "supersedes_for_development_only": "reports/openhands-cascade-development-v2.json",
        "preceding_development_report_sha256": digest(ROOT / "reports/openhands-cascade-development-v2.json"),
        "initial_guard_hypothesis_selected_after_historical_outcomes_inspected": True,
    }
    report["prospective_design"] = {
        "status": "not_run; fresh cohort and final verifier not yet frozen",
        "minimum_point_savings_per_accepted": .10,
        "maximum_observed_paired_losses": 0,
        "planning_max_harm_upper_bound": .02,
        "independent_groups_needed_if_zero_losses": zero_loss_required_groups(.02),
        "zero_loss_upper_bound_at_24": exact_harm_upper(0, 24),
        "caution": "149 is precision planning under independent Bernoulli sampling, not a powered guarantee of success or universal noninferiority",
    }
    guarded = report["policies"].get("initial_failure_guarded_cascade")
    seconds = report["initial_public_check_audit"]["initial_check_total_seconds"]
    baseline = report["policies"]["fixed_baseline"]
    additional_compute_budget = (baseline["cost_per_accepted_usd"] * guarded["accepted"] - guarded["total_cost_usd"]
                                 if guarded is not None and guarded["cost_complete"] and baseline["cost_per_accepted_usd"] is not None else None)
    report["initial_guard_compute_scenario"] = {
        "preflight_wall_seconds": seconds, "measured_cpu_seconds": None, "compute_price_usd_per_hour": None,
        "additional_local_cost_budget_at_equal_cost_per_accepted_usd": additional_compute_budget,
        "break_even_allocated_compute_hourly_usd_preflight_only": (
            additional_compute_budget / seconds * 3600 if additional_compute_budget is not None and additional_compute_budget > 0 and seconds > 0 else None),
        "scope": "conditional allocated-resource price applied to measured preflight wall time only; other deployment overhead unpriced",
        "net_monetary_savings_verified": False,
    }
    report["scope"].update(
        reference="published RouteLLM BERT checkpoint scored locally; not official MF or Not Diamond",
        resource_cost="LLM tariff scenario only; no invoice savings, no measured net CPU saving",
        replay_latencies="not reported: old episode verification includes final evaluation; replay cannot identify deployment verifier overhead",
        selection_bias="cascade was chosen after inspecting this cohort; bootstrap does not undo adaptive selection")
    return report


def markdown(report):
    cascade = report["policies"]["public_cascade"]
    comparison = report["comparisons_to_baseline"]["public_cascade"]
    bootstrap = comparison["bootstrap"]
    lines = [
        "Análisis exploratorio de desarrollo de la cascada, 18 de noviembre de 2026. No es una nueva validación ni activa el ahorro.",
        "Se auditaron las observaciones originales y la recuperación de infraestructura. La decisión usa únicamente public_passed: si falla, suma un episodio 120B desde el estado original. Se incluyen costos de intentos fallidos y reintentos internos. Las etiquetas de aceptación independientes se consultan después de fijar la ruta.",
    ]
    table = ["| Política | Aceptadas | Costo tarifario total USD | USD por aceptada | Pérdidas / ganancias frente a 120B |",
             "|---|---:|---:|---:|---:|"]
    for name, policy in report["policies"].items():
        paired = report["comparisons_to_baseline"].get(name, {})
        table.append(f"| {name} | {policy['accepted']}/{policy['tasks']} | {policy['total_cost_usd']:.8f} | {policy['cost_per_accepted_usd']:.8f} | {paired.get('paired_losses', 0)} / {paired.get('paired_gains', 0)} |")
    lines.append("\n".join(table))
    lines.extend([
        f"La cascada pública reconstruida reduce el costo tarifario por aceptada en {comparison['savings_per_accepted_task']:.2%}, con {cascade['escalated_tasks']} escalamiento(s). Pierde {', '.join(comparison['lost_task_ids'])} y gana {', '.join(comparison['gained_task_ids'])}. Igualar la aceptación agregada no elimina la pérdida pareada. El gate de cero pérdidas no se cumple.",
        f"Bootstrap pareado por grupo: {bootstrap['resamples']} réplicas, semilla {bootstrap['seed']}; intervalo percentil descriptivo del 95% para ahorro {bootstrap['savings_interval_95']}; para diferencia de aceptación {bootstrap['acceptance_difference_interval_95']}. Límite binomial exacto unilateral del 95% para pérdida pareada: {comparison['harm_upper_95_one_sided_exact']:.2%}. Estos intervalos no corrigen selección posterior de la política, contaminación de benchmarks ni dependencia entre familias.",
        f"Con cero pérdidas en 24 grupos, el límite exacto aún sería {report['prospective_design']['zero_loss_upper_bound_at_24']:.2%}. Para que ese límite sea como máximo 2%, se necesitan al menos 149 grupos independientes sin pérdidas. Es un cálculo de precisión condicionado a cero pérdidas, no una promesa de potencia ni de éxito.",
        "No se usa accepted o private_passed para decidir el escalamiento. No se reconstruye una cascada oracular para anunciar ahorros. El verificador operativo deja pasar una solución incorrecta; debe desarrollarse y congelarse antes de una reserva independiente.",
        "Los costos corresponden a tarifas simuladas con uso observado; Groq Free no generó una factura con ese importe. No se descuenta el cómputo local como si fuera gratuito. La latencia neta y el ahorro monetario completo siguen sin medirse. Las ramas históricas comparten resultados entre controles y no son ejecuciones independientes de cada política.",
        "BERT es el checkpoint publicado de RouteLLM con scoring local y umbrales históricos congelados; no se ejecutó RouteLLM MF ni Not Diamond. Los resultados anteriores permanecen intactos. El protocolo nuevo está en docs/openhands_cascade_v1.md.",
        "Métodos: [NIST, límites binomiales exactos](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm) y [SciPy, bootstrap pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).",
    ])
    guarded = report["policies"].get("initial_failure_guarded_cascade")
    audit = report["initial_public_check_audit"]
    if guarded is None:
        lines.append("La comprobación pública del original fue inconclusa para " + ", ".join(audit["inconclusive_task_ids"])
                     + ". No se publica una estimación económica agregada de la variante protegida excluyendo esos casos.")
    else:
        paired = report["comparisons_to_baseline"]["initial_failure_guarded_cascade"]
        direct = [row["task_id"] for row in report["task_rows"]["initial_failure_guarded_cascade"]
                  if row["candidate_skipped_initial_tests_passed"]]
        lines.append(f"Hipótesis adicional de desarrollo: comprobar primero si las pruebas públicas detectan el defecto en el original. "
                     f"Cuando ya pasan, se conserva 120B directamente. La variante reconstruida acepta {guarded['accepted']}/{guarded['tasks']}, "
                     f"con ahorro tarifario por aceptada {paired['savings_per_accepted_task']:.2%}, {paired['paired_losses']} pérdidas y "
                     f"{paired['paired_gains']} ganancias. Conserva 120B por falta de reproducción en: {', '.join(direct) or 'ninguna'}. "
                     "Esta regla se propuso después de ver los resultados históricos: es desarrollo posterior, no confirmación independiente.")
        lines.append(f"La comprobación inicial tardó {audit['initial_check_total_seconds']:.6f} segundos de pared en esta máquina. "
                     "Su precio de cómputo no se ha medido; el JSON publica un punto de equilibrio condicional que cubre solamente ese preflight. "
                     "La verificación de integridad evita cambios no autorizados en las pruebas, pero no demuestra suficiencia semántica ni elimina todos los falsos pases.")
        lines.append(f"Para la variante protegida, el intervalo bootstrap descriptivo de ahorro es {paired['bootstrap']['savings_interval_95']}; "
                     f"el límite exacto de pérdida es {paired['harm_upper_95_one_sided_exact']:.2%}. "
                     "El intervalo bootstrap de diferencia de aceptación [0,0] es degenerado porque no hubo pares discordantes en esta muestra; "
                     "no prueba ausencia de daño poblacional. No se cambia la regla al observar estos intervalos.")
    lines.append("Procedencia metodológica local: `.agents/skills/scientific-critical-thinking/SKILL.md`, hash registrado en el JSON. Se usó como guía de revisión; no se le atribuyen resultados empíricos ni un DOI sin verificar.")
    return "\n\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/openhands-cascade-development-v3")
    args = parser.parse_args(argv)
    paths = [args.output.with_suffix(extension) for extension in (".json", ".md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("A development report already exists; choose a new output path")
    report = make_report()
    payloads = [json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", markdown(report)]
    for path, payload in zip(paths, payloads):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(json.dumps({"paths": [str(path) for path in paths], "activation_eligible": False,
                      "public_cascade": report["policies"]["public_cascade"],
                      "comparison": report["comparisons_to_baseline"]["public_cascade"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
