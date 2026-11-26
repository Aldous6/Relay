"""Audit closed BFCL continuation sessions without generation or private grading."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import report_bfcl as base


def _sum_accounts(accounts, prices):
    """Add episode totals; quantiles cannot be added or averaged."""
    result = base._account([], prices)
    for key in result:
        if key in {"routing_p95_seconds", "wall_clock_total_seconds", "routing_p95_scope"}:
            continue
        values = [account[key] for account in accounts]
        if values:
            result[key] = None if any(value is None for value in values) else sum(values)
    result.update(routing_p95_seconds=None, wall_clock_total_seconds=None,
        routing_p95_scope="not pooled from episode or session quantiles; consult per-session reports")
    return result


def _normalized_config(config):
    result = deepcopy(config)
    if type(result.get("session_limits")) is dict:
        result["session_limits"].pop("max_tokens_per_model", None)
    generation = result.get("generation_manifest", {})
    if generation.get("kind") == "bfcl_session_budget":
        generation.pop("max_tokens_per_model", None)
    return result


def _compare_manifests(parent, child):
    if _normalized_config(parent["configuration"]) != _normalized_config(child["configuration"]):
        raise ValueError("La continuacion cambio configuracion cientifica o recursos fuera de los limites de sesion.")
    for key in ("protocol", "upstream_commit", "source_files_sha256", "runtime_source", "generation_source", "budget_contract"):
        if parent[key] != child[key]:
            raise ValueError(f"La continuacion cambio {key}.")
    parent_cases = {case["id"]: case for case in parent["cases"]}
    parent_runtime = dict(zip((case["id"] for case in parent["cases"]), parent["runtime_provenance"]))
    for case, provenance in zip(child["cases"], child["runtime_provenance"]):
        if case["id"] not in parent_cases or base.object_hash(case) != base.object_hash(parent_cases[case["id"]]):
            raise ValueError("La continuacion cambio un caso, su grupo o su referencia privada.")
        if provenance != parent_runtime[case["id"]]:
            raise ValueError("La continuacion cambio la procedencia del entorno.")
    for name in child["policy_sequence"]:
        if (name not in parent["policies"] or child["policies"][name] != parent["policies"][name]
                or child["policy_sources"][name] != parent["policy_sources"][name]):
            raise ValueError("La continuacion cambio una politica.")
    expected = [name for name in parent["policy_sequence"] if name in child["policy_sequence"]]
    if child["policy_sequence"] != expected:
        raise ValueError("La continuacion cambio el orden de politicas.")


def _aggregate(parent, reports, prices):
    planned = [(case["id"], policy) for case in parent["cases"] for policy in parent["policy_sequence"]]
    selected = {}
    for report in reports:
        for row in report["episodes"]:
            key = (row["episode_id"], row["policy"])
            if key not in planned:
                raise ValueError("Un episodio no pertenece al estudio original.")
            if row["status"] == "not_started":
                continue
            if row["status"] not in {"completed", "censored"}:
                raise ValueError("El estudio solo admite sesiones cerradas, sin intentos pendientes.")
            if key in selected:
                raise ValueError("Un par caso/politica fue intentado mas de una vez.")
            selected[key] = {**deepcopy(row), "run_directory": report["directory"]}
    # The inherited order must remain a prefix. No outcome-dependent skipping.
    closed_order = [key for key in planned if key in selected]
    if closed_order != planned[:len(closed_order)]:
        raise ValueError("Las sesiones omiten pares previos del orden original.")
    episodes = []
    groups = {case["id"]: case["source_group"] for case in parent["cases"]}
    for case_id, policy in planned:
        episodes.append(selected.get((case_id, policy), {"episode_id": case_id, "policy": policy,
            "source_group": groups[case_id], "status": "not_started", "success": None,
            "operational_success": None, "termination": None, "censored_reason": None,
            "run_directory": None, "accounting": _sum_accounts([], prices)}))
    by_policy = {}
    for policy in parent["policy_sequence"]:
        rows = [row for row in episodes if row["policy"] == policy]
        done = [row for row in rows if row["status"] != "not_started"]
        known = [row for row in done if row["success"] is not None]
        successes = sum(row["success"] is True for row in known)
        by_policy[policy] = {"planned_conversations": len(rows), "closed_conversations": len(done),
            "successes": successes, "known_failures": sum(row["success"] is False for row in known),
            "censored_conversations": len(done)-len(known), "not_started_conversations": len(rows)-len(done),
            "known_outcome_conversations": len(known),
            "success_rate_among_closed": successes/len(done) if done else None,
            "success_rate_with_known_outcome": successes/len(known) if known else None,
            "operational_success_rate": successes/len(rows) if len(done) == len(rows) else None,
            "accounting": _sum_accounts([row["accounting"] for row in done], prices)}
    paired = {}
    for index, first in enumerate(parent["policy_sequence"]):
        for second in parent["policy_sequence"][index+1:]:
            pairs = [(selected[(case["id"], first)], selected[(case["id"], second)]) for case in parent["cases"]
                if (case["id"], first) in selected and (case["id"], second) in selected]
            known = [(a, b) for a, b in pairs if a["success"] is not None and b["success"] is not None]
            a = _sum_accounts([a["accounting"] for a, _ in pairs], prices)
            b = _sum_accounts([b["accounting"] for _, b in pairs], prices)
            c1, c2 = a["tariff_no_cache_usd"], b["tariff_no_cache_usd"]
            ratio = c2/c1 if pairs and c1 is not None and c1 > 0 and c2 is not None else None
            first_only = sum(x["success"] and not y["success"] for x, y in known)
            second_only = sum(y["success"] and not x["success"] for x, y in known)
            paired[f"{first}__{second}"] = {"first_policy": first, "second_policy": second,
                "closed_pairs": len(pairs), "uncensored_pairs": len(known), "censored_pairs": len(pairs)-len(known),
                "resource_episode_ids": [x["episode_id"] for x, _ in pairs],
                "accounting_first": a, "accounting_second": b,
                "both_success": sum(x["success"] and y["success"] for x, y in known),
                "both_failure": sum(not x["success"] and not y["success"] for x, y in known),
                "first_only_success": first_only, "second_only_success": second_only,
                "known_outcome_success_delta": (first_only-second_only)/len(known) if known else None,
                "tariff_no_cache_cost_ratio_second_to_first": ratio,
                "tariff_no_cache_savings_fraction_second_vs_first": None if ratio is None else 1-ratio,
                "confidence_interval": None}
    return {"status": "complete" if len(selected) == len(planned) else "partial",
        "planned_conversations": len(parent["cases"]), "planned_pairs": [list(pair) for pair in planned],
        "closed_pairs": len(selected), "unattempted_pairs": [list(pair) for pair in planned if pair not in selected],
        "episodes": episodes, "policies": by_policy, "paired_comparisons": paired,
        "accounting": _sum_accounts([row["accounting"] for row in selected.values()], prices)}


def render_study(report):
    lines = ["Comparacion acumulada de controles BFCL.", "",
        f"Estado {report['status']}: {report['closed_pairs']}/{len(report['planned_pairs'])} pares conversacion/politica cerrados "
        f"sobre {report['planned_conversations']} conversaciones originales. Los pares pendientes conservan resultado desconocido.", "",
        "| Politica | Exitos / cerradas | Fallos conocidos | Censuras | Pendientes | Intentos | USD escenario sin cache |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in report["policies"].items():
        lines.append(f"| {name} | {row['successes']}/{row['closed_conversations']} | {row['known_failures']} | "
            f"{row['censored_conversations']} | {row['not_started_conversations']} | {row['accounting']['calls']} | "
            f"{base._number(row['accounting']['tariff_no_cache_usd'])} |")
    lines += ["", "Exitos/cerradas es una fraccion operativa, con censuras en el denominador. "
        "El JSON distingue la tasa con desenlace conocido. Las censuras no son fallos de capacidad. "
        "Los totales por politica pueden cubrir distintas conversaciones; para comparar recursos se usa la tabla pareada.", "",
        "| Primera / segunda | Pares comunes cerrados | Pares con desenlace conocido | Acierta solo primera / segunda | USD primera / segunda | Reduccion del escenario (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["paired_comparisons"].values():
        saving = row["tariff_no_cache_savings_fraction_second_vs_first"]
        lines.append(f"| {row['first_policy']} / {row['second_policy']} | {row['closed_pairs']} | {row['uncensored_pairs']} | "
            f"{row['first_only_success']}/{row['second_only_success']} | "
            f"{base._number(row['accounting_first']['tariff_no_cache_usd'])} / {base._number(row['accounting_second']['tariff_no_cache_usd'])} | "
            f"{base._number(None if saving is None else saving*100, 2)} |")
    lines += ["", "Los recursos pareados incluyen todos los intentos de las mismas conversaciones cerradas, incluidos fallos y censuras. "
        "La reduccion corresponde a la segunda politica frente a la primera, con tarifas fijas de tokens sin cache. "
        "Consumo ausente o costo base cero deja la razon desconocida. No es factura, ahorro real de Free ni ROI completo.", "",
        "No se promedian p95: el p95 acumulado queda desconocido y cada informe de sesion conserva sus mediciones. "
        "Tampoco se inventa latencia wall-clock. Las discordancias son exploratorias; no prueban el margen de dos puntos, "
        "un efecto causal del routing ni superioridad de Relay.", ""]
    for index, row in enumerate(report["runs"]):
        lines.append(f"Sesion {index+1}: [{row['status']}; {row['closed_pairs']} pares cerrados](reports/run-{index:03d}.md).")
    lines += ["", "Se verificaron sesiones cerradas, hashes, procedencia de continuacion, casos y referencias identicos, "
        "configuracion cientifica conservada y ausencia de pares intentados duplicados. No se ejecutaron APIs ni evaluadores privados."]
    return "\n".join(lines)+"\n"


def audit_study(parent_directory, run_directories=(), *, tariffs_path=None):
    """Read closed chains only; never build a generator, replay tools or grade."""
    # This helper recursively validates sealed continuation plans and summaries.
    # It only reads artifacts; prepare/execute/build_session are not invoked.
    from continue_bfcl import _load_previous

    parent_directory = Path(parent_directory).resolve()
    snapshot = base.Snapshot()
    parent_report = base.audit_bfcl(parent_directory, tariffs_path=tariffs_path)
    if not parent_report["summary_present"] or not parent_report["summary_current"]:
        raise ValueError("El padre original requiere un summary cerrado y vigente.")
    parent = base._manifest(parent_directory, snapshot)
    base_summary = snapshot.read(parent_directory / "summary.json", sealed=True)
    if "unattempted_pairs" not in base_summary:
        raise ValueError("El padre debe declarar sus pares no intentados.")
    original_pending = {tuple(pair) for pair in base_summary["unattempted_pairs"]}
    reports, directories, lineage, chains = [parent_report], [parent_directory], {}, []
    explicit_runs = []
    for requested in run_directories:
        requested = Path(requested).resolve()
        is_root = (requested / "continuation.json").is_file()
        root = requested if is_root else requested.parent
        if not (root / "continuation.json").is_file():
            raise ValueError("Cada RUN adicional requiere lineage de continuation.json.")
        if root not in lineage:
            state = _load_previous(root, snapshot)
            if state["manifest"]["freeze_sha256"] != parent["freeze_sha256"]:
                raise ValueError("La continuacion no desciende del padre original.")
            plan = snapshot.read(root / "continuation.json", sealed=True)
            root_summary = snapshot.read(root / "summary.json", sealed=True)
            actual_chain = [Path(path).resolve() for path in state["run_directories"]]
            current_runs = [Path(path).resolve() for path in root_summary["run_directories"]]
            expected_ancestors = actual_chain[:-len(current_runs)] if current_runs else actual_chain
            if ([str(path) for path in expected_ancestors] != plan["ancestor_run_directories"]
                    or not actual_chain or actual_chain[0] != parent_directory
                    or plan["source_files_sha256"] != parent["source_files_sha256"]):
                raise ValueError("El lineage cambio sus antecesores o fuentes.")
            lineage[root] = {"directory": str(root), "plan_sha256": base.object_hash(plan),
                "parent_directory": plan["parent"]["directory"], "closed_run_directories": [str(path) for path in current_runs]}
            chains.append(actual_chain)
        if is_root:
            explicit_runs.extend(Path(path) for path in lineage[root]["closed_run_directories"])
        else:
            if str(requested) not in lineage[root]["closed_run_directories"]:
                raise ValueError("El RUN no esta declarado cerrado por su continuacion.")
            explicit_runs.append(requested)
    if len(explicit_runs) != len(set(explicit_runs)) or parent_directory in explicit_runs:
        raise ValueError("Una sesion fue incluida mas de una vez.")
    included = {parent_directory, *explicit_runs}
    if any(path not in included for chain in chains for path in chain):
        raise ValueError("Faltan sesiones antecesoras cerradas en --runs.")
    parent_order = {(case["id"], name): index for index, (case, name) in enumerate(
        (case, name) for case in parent["cases"] for name in parent["policy_sequence"])}
    additions = []
    for directory in explicit_runs:
        report = base.audit_bfcl(directory, tariffs_path=tariffs_path)
        if not report["summary_present"] or not report["summary_current"] or report["status"] not in {"complete", "stopped"}:
            raise ValueError("Solo se agregan RUN con summary cerrado y vigente.")
        manifest = base._manifest(directory, snapshot)
        _compare_manifests(parent, manifest)
        summary = snapshot.read(directory / "summary.json", sealed=True)
        if any(tuple(pair) not in original_pending for pair in summary["planned_pairs"]):
            raise ValueError("La continuacion repite un par cerrado en el padre original.")
        additions.append((min(parent_order[tuple(pair)] for pair in summary["planned_pairs"]), directory, report))
    for _, directory, report in sorted(additions, key=lambda value: value[0]):
        directories.append(directory)
        reports.append(report)
    # Recheck all files audited by each report after reading the complete study.
    for report in reports:
        for path, expected in report["source_file_sha256"].items():
            if hashlib.sha256(snapshot.raw(path)).hexdigest() != expected:
                raise ValueError("Una sesion cambio durante la auditoria acumulada.")
    tariffs = reports[0]["tariffs"]
    if any(report["tariffs"] != tariffs for report in reports):
        raise ValueError("Las tarifas deben ser identicas para todas las sesiones.")
    prices = base._prices(tariffs, parent["configuration"]["model_catalog"])
    result = _aggregate(parent, reports, prices)
    snapshot.verify()
    result.update(schema_version=1, kind="bfcl_closed_study_report_v1", protocol=parent["protocol"],
        audited_at_utc=datetime.now(timezone.utc).isoformat(), parent_directory=str(parent_directory),
        parent_freeze_sha256=parent["freeze_sha256"], tariffs=tariffs, lineage=list(lineage.values()),
        source_file_sha256=snapshot.hashes,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=[{"directory": report["directory"], "status": report["status"], "freeze_sha256": report["freeze_sha256"],
            "closed_pairs": sum(row["status"] in {"completed", "censored"} for row in report["episodes"]),
            "report": report} for report in reports],
        independent_validation=False, execution_mode=parent["configuration"]["execution_mode"],
        cost_scope="fixed no-cache tariff scenario; same closed conversations in each paired comparison; no Free savings or full ROI",
        latency_scope="recorded phases summed; p95 not pooled from quantiles; per-session measurements retained",
        scope="exploratory local BFCL controls; no independence, causal routing, noninferiority or competitor superiority claim")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Original closed BFCL RUN with the full study population.")
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="Closed continuation RUNs or roots, including all ancestor sessions.")
    parser.add_argument("--tariffs", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory for cumulative and per-session reports.")
    args = parser.parse_args(argv)
    report = audit_study(args.parent, args.runs, tariffs_path=args.tariffs)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "reports").mkdir()
    for index, run in enumerate(report["runs"]):
        value = run["report"]
        (args.output / "reports" / f"run-{index:03d}.json").write_text(json.dumps({"payload": value,
            "payload_sha256": base.object_hash(value)}, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
        (args.output / "reports" / f"run-{index:03d}.md").write_text(base.render_report(value), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps({"payload": report, "payload_sha256": base.object_hash(report)},
        ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_study(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "planned_conversations": report["planned_conversations"],
        "closed_pairs": report["closed_pairs"], "unattempted_pairs": len(report["unattempted_pairs"]),
        "calls": report["accounting"]["calls"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
