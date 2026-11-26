"""Offline CLI for metadata ledgers and separately validated workflow policies."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from .workflow_policy import COST_BASES, propose_policy, validate_policy


def _bad_constant(value):
    raise ValueError("JSON no admite NaN ni Infinity.")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON contiene un número fuera de rango.")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contiene una clave duplicada.")
        result[key] = value
    return result


def _json(text):
    return json.loads(text, parse_constant=_bad_constant, parse_float=_float, object_pairs_hook=_object)


def read_ledger(path):
    """Read JSON arrays or JSONL events without interpreting task outcomes."""
    source = Path(path).read_text(encoding="utf-8-sig")
    if not source.strip():
        raise ValueError("El archivo de observaciones está vacío.")
    try:
        value = _json(source)
    except json.JSONDecodeError:
        # A JSONL ledger consists of one complete JSON object per nonblank line.
        value = [_json(line) for line in source.splitlines() if line.strip()]
        kindless = False
    else:
        kindless = type(value) is list
        if type(value) is dict:
            value = [value]
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError("Se requiere una lista JSON de observaciones o eventos JSONL.")
    observations = [row for row in value if row.get("kind") == "relay_task_observation"
                    or (kindless and "kind" not in row)]
    return value, observations


def _identity(row):
    if type(row.get("session_id")) is str and row["session_id"]:
        return (row["session_id"],)
    fields = ("evaluation_id", "split", "workflow", "task_id", "model")
    if all(type(row.get(key)) is str and row[key] for key in fields):
        return tuple(row[key] for key in fields)
    raise ValueError("Un evento de tarea carece de session_id o identidad completa.")


def _pending(events, observations):
    started, finished, calls, closed_calls = set(), set(), set(), set()
    for row in events:
        kind = row.get("kind")
        if kind not in {"relay_task_started", "relay_call_started", "relay_call_finished"}:
            continue
        identity = _identity(row)
        started.add(identity)
        if kind.startswith("relay_call_"):
            call_id = row.get("call_id")
            if type(call_id) is not str or not call_id:
                raise ValueError("Un evento de llamada carece de call_id.")
            key = (identity, call_id)
            if kind == "relay_call_started":
                calls.add(key)
            else:
                closed_calls.add(key)
    for row in observations:
        identity = _identity(row)
        finished.add(identity)
        # Observation-only exports can close recorded calls after a partial log.
        for call in row.get("calls", []):
            if type(call) is not dict:
                raise ValueError("Las llamadas registradas deben ser objetos JSON.")
            if call.get("status") in {"completed", "error", "unknown"} and type(call.get("call_id")) is str:
                closed_calls.add((identity, call["call_id"]))
    return {"unfinished_tasks": len(started - finished), "unfinished_calls": len(calls - closed_calls)}


def _observation_summary(observations):
    """Descriptive task totals; keep cost sources and missingness visible."""
    accepted, rejected, unknown = 0, 0, 0
    costs = {basis: [] for basis in sorted(COST_BASES)}
    for row in observations:
        if "accepted" not in row or (row["accepted"] is not None and type(row["accepted"]) is not bool):
            raise ValueError("accepted debe ser true, false o null.")
        accepted += row["accepted"] is True
        rejected += row["accepted"] is False
        unknown += row["accepted"] is None
        basis, cost = row.get("cost_basis"), row.get("cost_usd")
        if type(basis) is not str or basis not in COST_BASES or "cost_usd" not in row:
            raise ValueError("Cada observación requiere cost_usd y una cost_basis válida.")
        if cost is not None:
            try:
                valid = type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
            except OverflowError:
                valid = False
            if not valid or basis == "unknown":
                raise ValueError("El costo debe ser finito, no negativo y tener procedencia conocida.")
        if "cost_complete" in row and (type(row["cost_complete"]) is not bool
                or (row["cost_complete"] is False and cost is not None)):
            raise ValueError("Un costo declarado incompleto no puede figurar como costo total.")
        costs[basis].append(row)
    by_basis = {}
    for basis, values in costs.items():
        if not values:
            continue
        known = [row["cost_usd"] for row in values if row["cost_usd"] is not None]
        subtotal = math.fsum(known) if known else None
        if subtotal is not None and not math.isfinite(subtotal):
            raise ValueError("El costo agregado excede el rango admitido.")
        accepted_in_basis = sum(row["accepted"] is True for row in values)
        unknown_in_basis = sum(row["accepted"] is None for row in values)
        complete = len(known) == len(values) and unknown_in_basis == 0
        by_basis[basis] = {"observation_count": len(values), "known_cost_count": len(known),
                           "unknown_cost_count": len(values) - len(known),
                           "accepted_count": accepted_in_basis,
                           "rejected_count": sum(row["accepted"] is False for row in values),
                           "unknown_outcome_count": unknown_in_basis,
                           "known_cost_subtotal_usd": subtotal,
                           "total_cost_usd": subtotal if len(known) == len(values) else None,
                           "cost_per_accepted_task_usd": subtotal / accepted_in_basis
                           if complete and accepted_in_basis > 0 else None}
    return {"observation_count": len(observations), "accepted_count": accepted,
            "rejected_count": rejected, "unknown_outcome_count": unknown,
            "unknown_cost_count": sum(row["cost_usd"] is None for row in observations),
            "costs_by_basis": by_basis}


def report_ledger(events, observations):
    """Describe supplied observations; never infer a counterfactual saving."""
    pending = _pending(events, observations)
    summary = _observation_summary(observations)
    groups = {}
    for row in observations:
        if any(type(row.get(key)) is not str or not row[key] for key in ("workflow", "model")):
            raise ValueError("Cada observación requiere workflow y model para desglosar los resultados.")
        groups.setdefault((row["workflow"], row["model"]), []).append(row)
    breakdown = [{"workflow": workflow, "model": model, **_observation_summary(rows)}
                 for (workflow, model), rows in sorted(groups.items())]
    return {"schema_version": 1, "kind": "relay_harness_report",
            "data_provenance": "user_supplied_observations_not_independently_verified",
            "event_count": len(events), "event_kinds": dict(sorted(Counter(
                row.get("kind", "untyped_observation") for row in events).items())),
            **summary, **pending, "cost_unit": "USD", "by_workflow_model": breakdown,
            "metrics_scope": "descriptive_observed_cohorts_not_causal",
            "savings_usd": None,
            "limitation": "Descripción de registros suministrados; sin ahorro contrafactual. "
                          "Los subtotales no incluyen intentos sin finalizar ni costos desconocidos. "
                          "Tarifas estimadas y costos reportados se mantienen separados. "
                          "Costo por tarea aceptada = costo de todas las tareas observadas, incluidas fallidas, "
                          "dividido entre aceptadas; solo con costos y resultados completos dentro de cada procedencia. "
                          "El desglose describe poblaciones observadas, no una comparación causal entre modelos."}


def _write(path, value):
    destination = Path(path)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _observations(path):
    events, observations = read_ledger(path)
    pending = _pending(events, observations)
    if any(pending.values()):
        raise ValueError("Hay tareas o llamadas sin finalizar; cierra esos intentos antes de comparar.")
    return observations


def _adaptation_records(path):
    """Paired local training exports intentionally contain authorized task text.

    They are separate from the runtime's metadata-only activity ledger.
    """
    source = Path(path).read_text(encoding="utf-8-sig")
    try:
        value = _json(source)
    except json.JSONDecodeError:
        value = [_json(line) for line in source.splitlines() if line.strip()]
    if type(value) is dict:
        value = value.get("records")
    if type(value) is not list or not value or any(type(row) is not dict for row in value):
        raise ValueError("La adaptación necesita registros pareados con prompt_text, aceptación y costo.")
    return value


def parser():
    result = argparse.ArgumentParser(description="Relay: costos observados y políticas por workflow, sin llamadas a APIs.")
    commands = result.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Resumir un registro JSONL o una lista JSON sin inferir ahorro.")
    report.add_argument("input", help="Archivo de eventos u observaciones.")
    report.add_argument("--output", help="Guardar el informe JSON en un archivo nuevo.")
    propose = commands.add_parser("propose", help="Congelar candidatos usando solo desarrollo.")
    propose.add_argument("development", help="Observaciones pareadas con split=development.")
    propose.add_argument("--baseline", required=True, help="Modelo base del harness.")
    propose.add_argument("--output", required=True, help="Archivo JSON nuevo para la propuesta.")
    propose.add_argument("--min-groups", type=int, default=20, help="Grupos independientes mínimos (20).")
    propose.add_argument("--min-savings", type=float, default=.10, help="Ahorro relativo mínimo observado (0.10).")
    propose.add_argument("--max-quality-loss", type=float, default=0., help="Pérdida máxima de aceptación observada (0).")
    validate = commands.add_parser("validate", help="Validar la selección congelada en grupos separados.")
    validate.add_argument("proposal", help="Propuesta JSON congelada.")
    validate.add_argument("validation", help="Observaciones pareadas con split=validation.")
    validate.add_argument("--output", required=True, help="Archivo JSON nuevo para la política.")
    adapt = commands.add_parser("adapt", help="Aprender una política por prompt a partir de resultados pareados locales.")
    adapt.add_argument("development", help="JSON/JSONL con prompts y dos resultados por tarea; solo desarrollo.")
    adapt.add_argument("--baseline", required=True)
    adapt.add_argument("--candidate", required=True)
    adapt.add_argument("--scope", default="openhands-python-v1")
    adapt.add_argument("--checkpoint", help="Directorio del BERT local verificado; añade su representación a la selección automática.")
    adapt.add_argument("--output", required=True, help="Artefacto JSON nuevo; no habilita rutas sin validación.")
    verify = commands.add_parser("validate-adaptation", help="Comprobar ahorro por aceptada en tareas nuevas.")
    verify.add_argument("policy")
    verify.add_argument("validation", help="Registros pareados de una reserva independiente.")
    verify.add_argument("--output", required=True)
    verify.add_argument("--checkpoint", help="BERT local requerido si la política usa su representación.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "report":
            value = report_ledger(*read_ledger(args.input))
            if args.output:
                _write(args.output, value)
                print(f"Informe guardado: {args.output}")
            else:
                print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
        elif args.command == "propose":
            value = propose_policy(_observations(args.development), baseline_model=args.baseline,
                                   min_groups=args.min_groups, min_savings=args.min_savings,
                                   max_quality_loss=args.max_quality_loss)
            _write(args.output, value)
            print(f"Propuesta congelada: {args.output}. Falta validación con grupos separados.")
        elif args.command == "adapt":
            from .adaptive_training import train_adaptation
            selection_path = str(args.output) + ".selection.json"
            if Path(args.output).exists() or Path(selection_path).exists():
                raise FileExistsError(args.output)
            scorer = None
            if args.checkpoint:
                from .integrations.routellm import LocalBERTScorer
                scorer = LocalBERTScorer(args.checkpoint)
            value, selection = train_adaptation(_adaptation_records(args.development), baseline_model=args.baseline,
                                                candidate_model=args.candidate, scope=args.scope, scorer=scorer)
            _write(selection_path, selection)
            _write(args.output, value)
            print(f"Adaptación congelada: {args.output}. Falta validación independiente.")
        elif args.command == "validate-adaptation":
            from .adaptive_policy import evaluate_policy
            policy = _json(Path(args.policy).read_text(encoding="utf-8-sig"))
            scorer = None
            if policy.get("representation") == "routellm-bert":
                if not args.checkpoint:
                    raise ValueError("Esta política necesita --checkpoint con el BERT local verificado.")
                from .integrations.routellm import LocalBERTScorer
                scorer = LocalBERTScorer(args.checkpoint)
            value = evaluate_policy(policy, _adaptation_records(args.validation), scorer=scorer)
            _write(args.output, value)
            print(f"Validación guardada: {args.output}.")
            if value["activation_eligible"]:
                print("La política cumple el criterio empírico del piloto en el alcance declarado.")
            else:
                print("Se conserva el modelo base: " + ", ".join(value["reasons"]))
        else:
            proposal = _json(Path(args.proposal).read_text(encoding="utf-8-sig"))
            value = validate_policy(proposal, _observations(args.validation))
            _write(args.output, value)
            alternatives = sum(model != value["baseline_model"] for model in value["routes"].values())
            print(f"Política guardada: {args.output}. Workflows con alternativa validada: {alternatives}.")
            if not value["ready"]:
                print("Se conserva el modelo base: la evidencia no habilita una alternativa.")
        return 0
    except FileExistsError:
        print("No se sobrescribió el archivo existente. Usa otra ruta de salida.", file=sys.stderr)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, ImportError) as exc:
        print(f"No se pudo completar: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
