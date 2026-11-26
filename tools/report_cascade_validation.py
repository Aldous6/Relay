"""Audit a prospectively frozen, real-runtime cascade validation.

The primary policy is the ACTUAL runtime result. A paired branch replay is a
secondary diagnostic only and cannot replace failed/no-output runtime tasks.
No provider calls, policy fitting, cohort trimming, or artifact overwrites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from relay_router.cascade_evaluation import paired_comparison, replay_public_cascade, summarize_policy

DEFAULT_RUN = ROOT / "outputs/harness-cascade/v1/run"
DEFAULT_SUITE = ROOT / "data/imported/harness-cascade-v1/fresh_suite.json"
DEFAULT_EVALUATORS = ROOT / "data/imported/harness-cascade-v1/private_evaluators.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).exists() else []


def close(left, right):
    return left == right if left is None or right is None else math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _branch_bounds(accounting, branches, tariffs):
    """Verify recorded ceilings against each dispatch, keeping v1 reservations unpriced."""
    if "cost_upper_bound_usd" not in accounting:
        return
    calls = accounting["calls"]
    if not accounting.get("journal_complete") or len(calls) != accounting["http_calls"]:
        raise ValueError("Cannot audit cost bounds without a complete dispatch journal")
    call_ids = [call["call_id"] for call in calls]
    if len(set(call_ids)) != len(call_ids):
        raise ValueError("Duplicate dispatch in bounded ledger")
    missing = {entry["model"] for entry in accounting.get("unobserved_transport", [])}
    if (not missing) != accounting.get("http_calls_complete", True) or not missing <= {b["model"] for b in branches}:
        raise ValueError("Unobserved transport must have an identified model branch")
    uncertain_contract = (any(call["cost_usd"] is None for call in calls)
                          and any(call.get("reservation_exceeded") for call in calls))
    ceilings = []
    for branch in branches:
        model = branch["model"]
        selected = [call for call in calls if call["model"] == model]
        if len(selected) != branch["http_attempts"]:
            raise ValueError("Branch dispatch count differs from the bounded ledger")
        if model not in missing and all(call["cost_usd"] is not None for call in selected) != branch["cost_complete"]:
            raise ValueError("Branch completeness differs from the bounded ledger")
        known = math.fsum(number(call["cost_usd"], "known call") for call in selected if call["cost_usd"] is not None)
        if not close(known, branch.get("known_cost_subtotal_usd", branch["cost_usd"] or 0.)):
            raise ValueError("Branch known charges differ from the bounded ledger")
        limits = []
        for call in selected:
            point = call["cost_usd"]
            upper = call["cost_upper_bound_usd"]
            if point is not None:
                if not close(point, upper) or call["cost_basis"] != "tariff_estimate":
                    raise ValueError("Known dispatch differs from its tariff bound")
            elif upper is not None:
                bound = call.get("tariff_bound")
                if (not bound or not isinstance(bound.get("source"), str) or not bound["source"].startswith("https://")
                        or call.get("cost_bound_basis") != "tariff_estimate" or call.get("http_status") != 400
                        or call.get("provider_error_code") != "tool_use_failed" or call.get("status") != "error"):
                    raise ValueError("A bounded unknown dispatch requires documented tariff limits")
                expected = 0.
                for side in ("input", "output"):
                    cap = bound["max_" + side + "_tokens"]
                    if type(cap) is not int or cap < 1:
                        raise ValueError("Invalid dispatch token ceiling")
                    rate = number(bound[side + "_usd_per_token"], "tariff rate")
                    if rate == 0 or (tariffs is not None and not close(rate, tariffs[model][side + "_per_million_usd"] / 1e6)):
                        raise ValueError("Cost ceiling differs from the frozen tariff")
                    expected += cap * rate
                if not close(expected, upper):
                    raise ValueError("Cost ceiling differs from the recorded token limits and tariffs")
            limits.append(upper)
        if model in missing:
            branch.update(cost_usd=None, cost_basis="unknown", cost_complete=False,
                          known_cost_subtotal_usd=known, http_attempts_complete=False)
        ceiling = branch["cost_usd"]
        if ceiling is None and model not in missing and not uncertain_contract and all(limit is not None for limit in limits):
            ceiling = math.fsum(limits)
        if "cost_upper_bound_usd" in branch and not close(branch["cost_upper_bound_usd"], ceiling):
            raise ValueError("Branch cost ceiling disagrees with its actual dispatches")
        branch.update(cost_upper_bound_usd=ceiling, cost_bound_basis="tariff_estimate")
        ceilings.append(ceiling)
    combined = math.fsum(ceilings) if all(ceiling is not None for ceiling in ceilings) else None
    if not close(accounting["cost_upper_bound_usd"], combined):
        raise ValueError("Task ceiling must include all dispatches")


def summarize_actual(suite, observations, cascades, *, resamples=10000, seed=20261118, tariffs=None):
    """Analyze complete pairs, preserving runtime stopping and both attempt costs."""
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    if len(tasks) != len(suite["tasks"]) or set(cascades) != set(tasks):
        raise ValueError("Each frozen task needs exactly one actual cascade result")
    baseline = suite["baseline_model"]
    candidate = next(model for model in suite["models"] if model != baseline)
    index = {(row["task_id"], row["model"]): dict(row) for row in observations}
    expected = {(task, model) for task in tasks for model in (candidate, baseline)}
    if len(index) != len(observations) or set(index) != expected:
        raise ValueError("A complete comparison needs both model branches of every frozen task")
    if any(row["group_id"] != tasks[row["task_id"]]["group_id"] for row in observations):
        raise ValueError("Observed groups differ from the frozen cohort")
    # Reuse strict record validation and secondary controls; no primary result
    # is inferred from a replay's selected route.
    actual_rows, initial = [], {}
    for task_id, task in tasks.items():
        result = cascades[task_id]
        if result.get("task_id") != task_id or result.get("mode") != "experimental":
            raise ValueError("Cascade identity/mode differs from the frozen experiment")
        if result.get("accepted") is not None:
            raise ValueError("Runtime must not receive independent acceptance labels")
        if result.get("require_initial_failure") is not True:
            raise ValueError("The frozen initial-failure guard must remain enabled")
        attempts = result.get("attempts")
        if not isinstance(attempts, list) or len(attempts) > 2:
            raise ValueError("At most two actual attempts are allowed")
        models = [attempt["model"] for attempt in attempts]
        if models not in ([], [candidate], [baseline], [candidate, baseline]):
            raise ValueError("Actual model sequence is outside the frozen cascade")
        branch_rows = [index[(task_id, model)] for model in models]
        if any(row["group_id"] != task["group_id"] for row in branch_rows):
            raise ValueError("Actual branches differ from frozen prompt groups")
        selected = result.get("selected_attempt")
        if selected is not None and (type(selected) is not int or selected != len(attempts) or selected < 1):
            raise ValueError("A selected result must be the last actual attempt")
        if result.get("operational_passed") is not (selected is not None):
            raise ValueError("Operational result differs from selected attempt")
        if selected is not None:
            if result.get("selected_model") != models[-1] or attempts[-1]["status"] != "completed":
                raise ValueError("Selected model/status differs from its actual branch")
            if attempts[-1]["verification"]["status"] != "passed" or not branch_rows[-1]["public_passed"]:
                raise ValueError("A selected branch did not pass its operational checks")
        elif result.get("selected_model") is not None:
            raise ValueError("A task without selection cannot name a selected model")
        accepted = branch_rows[-1]["accepted"] if selected is not None else False
        accounting = result["accounting"]
        _branch_bounds(accounting, branch_rows, tariffs)
        bases = {row["cost_basis"] for row in branch_rows}
        complete = all(row["cost_complete"] and row["cost_usd"] is not None for row in branch_rows) and len(bases) <= 1
        subtotal = math.fsum(number(row["cost_usd"] if row["cost_usd"] is not None else row.get("known_cost_subtotal_usd", 0.),
                                   "known branch cost") for row in branch_rows)
        total = subtotal if complete else None
        http_count = sum(row["http_attempts"] for row in branch_rows)
        if (accounting["http_calls"] != http_count or accounting["cost_complete"] is not complete
                or not close(accounting["cost_usd"], total)
                or not close(accounting["known_cost_subtotal_usd"], subtotal)):
            raise ValueError("Runtime accounting differs from all actual branch HTTP costs")
        if not attempts and (accounting["calls"] or http_count):
            raise ValueError("Zero LLM cost requires explicit absence of dispatch")
        row = {"task_id": task_id, "group_id": task["group_id"], "accepted": accepted,
               "operational_passed": selected is not None, "final_model": result.get("selected_model"),
               "attempted_models": models, "cost_usd": total, "cost_complete": complete,
               "cost_basis": next(iter(bases)) if complete and bases else "tariff_estimate" if complete else "unknown",
               "known_cost_subtotal_usd": subtotal, "http_attempts": http_count,
               "http_attempts_complete": accounting.get("http_calls_complete", True),
               "escalated": len(attempts) == 2, "runtime_reason": result.get("reason"),
               "route_reason": result.get("route_reason"), "wall_seconds": number(result["wall_seconds"], "runtime wall seconds")}
        if "cost_upper_bound_usd" in accounting:
            row.update(cost_upper_bound_usd=total if complete else accounting["cost_upper_bound_usd"],
                       cost_bound_basis="tariff_estimate")
        actual_rows.append(row)
        check = result["initial_verification"]
        initial[task_id] = {"status": check["status"], "seconds": number(check["duration_seconds"], "initial check seconds")}
    controls = replay_public_cascade(index.values(), candidate_model=candidate, baseline_model=baseline,
                                     resamples=resamples, seed=seed)
    control_rows = controls["task_rows"]
    policy_rows = {"relay_cascade_actual": actual_rows, "fixed_baseline": control_rows["fixed_baseline"],
                   "fixed_candidate": control_rows["fixed_candidate"], "naive_public_cascade_replay": control_rows["public_cascade"]}
    summaries = {name: summarize_policy(values) for name, values in policy_rows.items()}
    comparisons = {name: paired_comparison(values, policy_rows["fixed_baseline"], resamples=resamples, seed=seed)
                   for name, values in policy_rows.items() if name != "fixed_baseline"}
    primary = comparisons["relay_cascade_actual"]
    selected = summaries["relay_cascade_actual"]
    reasons = []
    if len(tasks) < 20:
        reasons.append("fewer_than_20_fresh_groups")
    if primary["paired_losses"]:
        reasons.append("observed_paired_quality_loss")
    if not selected["cost_complete"] or not summaries["fixed_baseline"]["cost_complete"]:
        reasons.append("incomplete_cost_accounting")
    if primary["savings_per_accepted_task"] is None:
        reasons.append("savings_per_accepted_not_estimable")
    elif primary["savings_per_accepted_task"] < .10:
        reasons.append("less_than_10_percent_savings_per_accepted")
    interval = primary["bootstrap"]["savings_interval_95"]
    strong_reasons = list(reasons)
    if interval is None or interval[0] <= 0:
        strong_reasons.append("savings_interval_does_not_exclude_zero")
    if primary["harm_upper_95_one_sided_exact"] > .02:
        strong_reasons.append("harm_upper_bound_exceeds_two_percent")
    times = sorted(row["wall_seconds"] for row in actual_rows)
    seconds = math.fsum(check["seconds"] for check in initial.values())
    return {"policies": summaries, "comparisons_to_baseline": comparisons, "task_rows": policy_rows,
            "operational_verifier_diagnostics": controls["operational_verifier_diagnostics"],
            "initial_public_checks": initial,
            "pilot_criteria": {"met": not reasons, "reasons": reasons, "minimum_savings": .10,
                               "maximum_observed_paired_losses": 0, "minimum_groups": 20},
            "strong_evidence_criteria": {"met": not strong_reasons, "reasons": strong_reasons,
                                         "maximum_harm_upper_95": .02},
            "activation_eligible": False, "production_ready": False,
            "timing": {"actual_cascade_wall_seconds_sum": math.fsum(times),
                       "actual_cascade_wall_seconds_median": statistics.median(times),
                       "actual_cascade_wall_seconds_p95_nearest_rank": times[math.ceil(.95 * len(times)) - 1],
                       "initial_verification_wall_seconds_sum": seconds,
                       "private_final_evaluation_in_runtime_timing": False,
                       "net_resource_cost_measured": False},
            "comparison_design": "actual cascades; fixed model controls reuse a branch if already generated, else acquired afterwards",
            "net_monetary_savings_verified": False}


def audit_http(suite, observations, events, tariffs):
    """Reprice every acquired dispatch and reconcile every branch, including zero."""
    expected_ids = list(range(1, len(events) + 1))
    if [event.get("attempt_id") for event in events] != expected_ids:
        raise ValueError("HTTP attempt IDs must be unique and consecutive")
    indexed = {row["episode_id"]: row for row in observations}
    if len(indexed) != len(observations):
        raise ValueError("Duplicate episode IDs")
    by_episode = {episode: [] for episode in indexed}
    tokens = {model: 0 for model in suite["models"]}
    for event in events:
        if event.get("episode_id") not in indexed or event.get("status") not in {"completed", "error"}:
            raise ValueError("Unaccounted or unresolved HTTP event")
        row = indexed[event["episode_id"]]
        model = event["model"]
        if model != row["model"] or model not in tariffs:
            raise ValueError("HTTP model differs from episode model")
        inp, out = event.get("input_tokens"), event.get("output_tokens")
        known = type(inp) is int and inp >= 0 and type(out) is int and out >= 0
        if known:
            if event.get("total_tokens") != inp + out:
                raise ValueError("HTTP usage total is inconsistent")
            tokens[model] += inp + out
        rate = tariffs[model]
        cost = (inp * rate["input_per_million_usd"] + out * rate["output_per_million_usd"]) / 1e6 if known else None
        if not close(event.get("cost_usd"), cost):
            raise ValueError("HTTP price differs from frozen tariffs or unknown usage")
        by_episode[row["episode_id"]].append(event)
    for episode, row in indexed.items():
        selected = by_episode[episode]
        known = all(event["cost_usd"] is not None for event in selected)
        total = math.fsum(event["cost_usd"] for event in selected) if known else None
        if row.get("http_attempts") != len(selected) or row.get("cost_complete") is not known or not close(row.get("cost_usd"), total):
            raise ValueError("Episode cost must include every dispatch, with verified zero when none")
        if row.get("cost_basis") != ("tariff_estimate" if known else "unknown"):
            raise ValueError("Episode cost basis differs from HTTP accounting")
        subtotal = math.fsum(event["cost_usd"] for event in selected if event["cost_usd"] is not None)
        if "known_cost_subtotal_usd" in row and not close(row["known_cost_subtotal_usd"], subtotal):
            raise ValueError("Known cost subtotal differs from its HTTP events")
        row["known_cost_subtotal_usd"] = subtotal
    return {"http_attempts": len(events), "unknown_cost_http_attempts": sum(event["cost_usd"] is None for event in events),
            "known_acquisition_subtotal_usd": math.fsum(event["cost_usd"] for event in events if event["cost_usd"] is not None),
            "observed_tokens_by_model": tokens}


def make_report(run, suite_path, evaluators_path):
    import run_cascade_validation as runner
    run, suite_path, evaluators_path = Path(run).resolve(), Path(suite_path), Path(evaluators_path)
    suite, frozen = read(suite_path), read(run / "frozen_policy.json")
    runner.verify_frozen(suite, frozen)
    payload = dict(frozen)
    freeze_id = payload.pop("freeze_id", None)
    if seal(payload) != freeze_id:
        raise ValueError("Frozen policy integrity mismatch")
    contract = frozen["contract"]
    if contract["suite_sha256"] != digest(suite_path) or contract["private_evaluators_sha256"] != digest(evaluators_path):
        raise ValueError("Suite/evaluator bytes differ from prospective freeze")
    for relative, expected in contract["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Frozen source changed")
    observations = runner.read_observations(suite, run)
    events = runner.reconcile_http(suite, run, observations)
    runtime_rows = rows(run / "cascade_results.jsonl")
    for row in observations:
        if (type(row.get("private_passed")) is not bool
                or row.get("episode_status") not in {"completed", "error"}
                or row["accepted"] != (row["public_passed"] and row["private_passed"] and row["episode_status"] == "completed")):
            raise ValueError("Final acceptance requires public/private checks and a completed agent attempt")
    audit = audit_http(suite, observations, events, contract["tariffs"])
    frozen_at = datetime.fromisoformat(frozen["created_at_utc"])
    if frozen_at.tzinfo is None:
        raise ValueError("The prospective freeze must have a timezone-aware timestamp")
    for event in events:
        started = datetime.fromisoformat(event["started_at_utc"])
        if started.tzinfo is None or started < frozen_at:
            raise ValueError("A model dispatch preceded the policy freeze")
    if len(events) > contract["limits"]["max_total_http_attempts"]:
        raise ValueError("The declared acquisition HTTP cap was exceeded")
    if any(count > contract["limits"]["max_observed_tokens_per_model"] for count in audit["observed_tokens_by_model"].values()):
        raise ValueError("The declared per-model token cap was exceeded")
    starts = list((run / "attempt_starts").glob("*.json"))
    if len(starts) != len(events):
        raise ValueError("An HTTP start has no completed accounting record")
    for event in events:
        start = read(run / "attempt_starts" / (str(event["attempt_id"]) + ".json"))
        if any(start.get(name) != event.get(name) for name in ("attempt_id", "episode_id", "model", "started_at_utc")):
            raise ValueError("HTTP write-ahead identity changed")
    expected_pairs = {(task["task_id"], model) for task in suite["tasks"] for model in suite["models"]}
    observed_pairs = {(row["task_id"], row["model"]) for row in observations}
    if len(observed_pairs) != len(observations) or not observed_pairs.issubset(expected_pairs):
        raise ValueError("Duplicate or out-of-cohort model observations")
    cascades = {}
    for entry in runtime_rows:
        task_id = entry["task_id"]
        result_path = (run / entry["result_path"]).resolve()
        if not result_path.is_relative_to(run) or task_id in cascades or digest(result_path) != entry["result_sha256"]:
            raise ValueError("Unsafe, duplicate or changed runtime result")
        cascades[task_id] = read(result_path)
    if not set(cascades).issubset({task["task_id"] for task in suite["tasks"]}):
        raise ValueError("Runtime result outside the frozen cohort")
    complete = observed_pairs == expected_pairs and set(cascades) == {task["task_id"] for task in suite["tasks"]}
    report = {"schema_version": 1, "kind": "prospective_real_cascade_validation", "freeze_id": freeze_id,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
              "expected_tasks": len(suite["tasks"]), "expected_branches": len(expected_pairs),
              "completed_branches": len(observations), "completed_cascades": len(cascades),
              "missing_branches": [{"task_id": task, "model": model} for task, model in sorted(expected_pairs - observed_pairs)],
              "acquisition_audit": audit, "activation_eligible": False, "production_ready": False,
              "provenance": {"suite_sha256": digest(suite_path), "private_evaluators_sha256": digest(evaluators_path),
                             "frozen_policy_sha256": digest(run / "frozen_policy.json"),
                             "source_sha256": contract["source_sha256"], "reporter_sha256": digest(Path(__file__)),
                             "input_sha256": {name: digest(run / name) for name in ("observations.jsonl", "http_attempts.jsonl", "cascade_results.jsonl") if (run / name).exists()}},
              "limitations": ["Public benchmarks may have been seen during pretraining",
                              "Controls reuse model branches and are acquired after the operational cascade when missing",
                              "Tariff scenario is not the Groq Free invoice and excludes unpriced local compute",
                              "No direct comparison against RouteLLM MF or Not Diamond"]}
    families = {}
    for task in suite["tasks"]:
        family = task.get("workflow", "unspecified")
        families[family] = families.get(family, 0) + 1
    report["cohort_by_workflow"] = families
    report["historical_cohort_comparability"] = (
        "This cohort excludes prior tasks; its workflow mix may differ from the historical 12 HumanEval / 12 QuixBugs development cohort. "
        "A different savings point estimate is not an isolated effect of changing sample size.")
    if complete:
        for row in observations:
            if row.get("stage") == "control":
                control = read(run / "tasks" / row["task_id"] / ("control-" + row["model"].rsplit("/", 1)[-1]) / "result.json")
                _branch_bounds(control["accounting"], [row], contract["tariffs"])
        report.update(summarize_actual(suite, observations, cascades, tariffs=contract["tariffs"]))
    else:
        report.update(pilot_criteria={"met": False, "reasons": ["incomplete_frozen_cohort"]},
                      strong_evidence_criteria={"met": False, "reasons": ["incomplete_frozen_cohort"]})
    report["report_id"] = seal(report)
    return report


def markdown(report):
    first = "Informe de validación prospectiva de Relay, 18 de noviembre de 2026."
    if not report["complete"]:
        return (first + "\n\nLa campaña está incompleta: " + str(report["completed_branches"]) + "/" + str(report["expected_branches"])
                + " ramas y " + str(report["completed_cascades"]) + "/" + str(report["expected_tasks"])
                + " cascadas. No se calcula un ahorro de validación sobre el prefijo observado ni se activa la política. "
                "El JSON conserva tareas faltantes, despachos, costos desconocidos y hashes.\n")
    own, base = report["policies"]["relay_cascade_actual"], report["policies"]["fixed_baseline"]
    comparison = report["comparisons_to_baseline"]["relay_cascade_actual"]
    savings = comparison["savings_per_accepted_task"]
    interval = comparison["bootstrap"]["savings_interval_95"]
    savings_text = f"{savings:.2%}" if savings is not None else "indefinido por costo/aceptación insuficiente"
    interval_text = f"[{interval[0]:.2%}, {interval[1]:.2%}]" if interval is not None else "no disponible"

    def amount(point, limits):
        if point is not None:
            return f"USD {point:.8f}"
        if limits is None:
            return "indeterminado"
        if limits[1] is None:
            return f"al menos USD {limits[0]:.8f}, con límite superior desconocido"
        return f"entre USD {limits[0]:.8f} y USD {limits[1]:.8f}"

    saving_bounds = comparison.get("savings_per_accepted_bounds")
    accounting_text = (f" Las cotas tarifarias sitúan el ahorro entre {saving_bounds[0]:.2%} y {saving_bounds[1]:.2%}."
                       if savings is None and saving_bounds is not None else "")
    paragraphs = [first,
        f"Se completaron {report['expected_tasks']} tareas nuevas para Relay con la política congelada antes de adquirir respuestas. "
        f"La cascada real aceptó {own['accepted']}/{own['tasks']} y 120B fijo {base['accepted']}/{base['tasks']}. "
        f"Hubo {comparison['paired_losses']} pérdidas y {comparison['paired_gains']} ganancias pareadas; una ganancia no compensa una pérdida para el criterio de protección.",
        f"El ahorro puntual de costo tarifario por tarea aceptada frente a 120B fue {savings_text}.{accounting_text} "
        f"El gasto de Relay fue {amount(own['total_cost_usd'], own.get('total_cost_bounds_usd'))} "
        f"y el de la referencia fue {amount(base['total_cost_usd'], base.get('total_cost_bounds_usd'))}. "
        f"Por tarea aceptada fueron {amount(own['cost_per_accepted_usd'], own.get('cost_per_accepted_bounds_usd'))} "
        f"y {amount(base['cost_per_accepted_usd'], base.get('cost_per_accepted_bounds_usd'))}, respectivamente. "
        "El total de cada política incluye todos sus intentos, incluidos los fallidos.",
        f"El intervalo percentil pareado del 95%, con 10000 remuestreos y semilla 20261118, es {interval_text}. "
        f"El límite binomial exacto unilateral del 95% de pérdida pareada es {comparison['harm_upper_95_one_sided_exact']:.2%}. "
        "Cero pérdidas observadas o un intervalo bootstrap de calidad [0,0] no prueban daño poblacional nulo.",
        "Criterio empírico del piloto: " + ("cumplido." if report["pilot_criteria"]["met"] else "no cumplido: " + ", ".join(report["pilot_criteria"]["reasons"]) + ".")
        + " Evidencia fuerte: " + ("cumple los criterios fijados." if report["strong_evidence_criteria"]["met"] else "insuficiente: " + ", ".join(report["strong_evidence_criteria"]["reasons"]) + "."),
        "La decisión utilizó pruebas públicas del original y del parche. La aceptación final se evaluó después de cerrar la ruta, fuera del agente. "
        "La referencia reutiliza 120B cuando ese modelo ya se ejecutó en la cascada; de lo contrario se adquirió después. No son ejecuciones independientes de cada política.",
        "Los importes son un escenario tarifario con uso observado, no ahorro facturado por Groq Free. El costo local no está monetizado y no se declara ahorro neto. "
        "Estos benchmarks públicos no demuestran eficacia en cualquier repositorio ni superioridad frente a RouteLLM o Not Diamond.",
        "Composición de la cohorte nueva: " + ", ".join(f"{name}: {count}" for name, count in sorted(report["cohort_by_workflow"].items()))
        + ". La cohorte histórica tenía 12 tareas HumanEval y 12 QuixBugs; una mezcla distinta limita la comparación directa de sus porcentajes de ahorro.",
        "El JSON adjunto conserva cada tarea, pérdidas, ganancias, costos, usos y hashes de código, suite, evaluadores y bitácoras. "
        "Los resultados históricos de 12,68% se mantienen separados; esta cohorte no se usa para reajustar la política y luego presentarla como una nueva validación."]
    return "\n\n".join(paragraphs) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--evaluators", type=Path, default=DEFAULT_EVALUATORS)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/openhands-cascade-validation-v1")
    args = parser.parse_args(argv)
    paths = [args.output.with_suffix(extension) for extension in (".json", ".md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a new report destination; historical reports are immutable")
    report = make_report(args.run, args.suite, args.evaluators)
    contents = [json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", markdown(report)]
    for path, content in zip(paths, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    print(json.dumps({"complete": report["complete"], "report_id": report["report_id"],
                      "paths": [str(path) for path in paths], "pilot_criteria": report["pilot_criteria"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
