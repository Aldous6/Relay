"""Report isolated infrastructure recovery without replacing the primary trial.

The same frozen policy is evaluated on complete task/model episodes including
their audited zero-dispatch initialization failure and one recovery attempt.
Only a demonstrated absence of logical calls and HTTP starts/completions can
establish zero LLM tariff cost; missing usage on an actual dispatch stays null.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import report_adaptive_pilot as primary
import recover_adaptive_initialization as recovery

DEFAULT_OUTPUT = ROOT / "reports/openhands-adaptation-recovery-v1"


def combine_episodes(original_rows, recovered_rows, eligible, recovery_zero_dispatch=()):
    """Return new rows; originals are untouched and their initial failures kept."""
    original = {(row["task_id"], row["model"]): row for row in original_rows}
    retried = {(row["task_id"], row["model"]): row for row in recovered_rows}
    allowed = {(row["task_id"], row["model"]): row for row in eligible}
    if (len(original) != len(original_rows) or len(retried) != len(recovered_rows)
            or len(allowed) != len(eligible) or set(retried) != set(allowed) or not set(allowed).issubset(original)):
        raise ValueError("Secondary analysis requires every independently eligible episode exactly once")
    verified_zero = {row["episode_id"] for row in recovery_zero_dispatch}
    combined, provenance = [], []
    for before in original_rows:
        pair = before["task_id"], before["model"]
        if pair not in retried:
            combined.append(deepcopy(before))
            continue
        evidence, after = allowed[pair], retried[pair]
        if (before.get("error_type") != "PermissionError" or not recovery._zero(before.get("logical_calls"))
                or not recovery._zero(before.get("http_attempts")) or not recovery._zero(evidence.get("original_logical_calls"))
                or not recovery._zero(evidence.get("original_http_attempts")) or not recovery._zero(evidence.get("original_http_start_records"))
                or evidence.get("zero_dispatch_tariff_usd") != 0):
            raise ValueError("Zero initial LLM cost requires explicit audited zero-dispatch evidence")
        if (type(after.get("accepted")) is not bool or type(after.get("public_passed")) is not bool
                or type(after.get("private_passed")) is not bool
                or after["accepted"] != (after["public_passed"] and after["private_passed"])
                or type(after.get("logical_calls")) is not int or not 0 <= after["logical_calls"] <= 3):
            raise ValueError("Recovery acceptance or logical-call bound differs from the original verifier contract")
        item = deepcopy(after)
        episode_id = evidence["episode_id"]
        retry_zero = episode_id in verified_zero
        if retry_zero:
            if after.get("logical_calls") != 0 or after.get("http_attempts") != 0 or after.get("error_type") != "PermissionError":
                raise ValueError("A recovery with requests cannot be priced as zero dispatch")
            item.update(cost_usd=0., cost_complete=True, cost_basis="tariff_estimate")
        elif after.get("cost_usd") is not None:
            primary.number(after["cost_usd"], "recovery cost")
            if after.get("cost_complete") is not True or after.get("cost_basis") != "tariff_estimate":
                raise ValueError("Numeric recovery cost requires complete reconciled HTTP accounting")
        elif after.get("cost_complete") is not False or after.get("cost_basis") != "unknown":
            raise ValueError("Unknown recovery cost cannot be declared complete")
        item.update(evaluation_id=before["evaluation_id"], original_error_type=before["error_type"],
                    original_accepted=before["accepted"], initial_llm_tariff_verified_usd=0.,
                    task_episode_attempts=2, cost_scope="initial_verified_zero_dispatch_plus_one_recovery_all_HTTP_excludes_local_compute")
        times = {}
        for field in ("wall_seconds", "elapsed_seconds", "verification_seconds", "transport_seconds", "quota_wait_seconds"):
            first, second = before.get(field), after.get(field)
            total = (primary.number(first, "initial " + field) + primary.number(second, "recovery " + field)
                     if first is not None and second is not None else None)
            item[field] = total
            times[field] = {"initial": first, "recovery": second, "combined": total}
        provenance.append({"episode_id": episode_id, "task_id": pair[0], "model": pair[1],
                           "initial_error_type": before["error_type"], "initial_accepted": before["accepted"],
                           "initial_raw_cost_usd": before["cost_usd"], "initial_llm_tariff_verified_usd": 0.,
                           "initial_zero_dispatch_evidence": deepcopy(evidence), "recovery_accepted": after["accepted"],
                           "recovery_raw_cost_usd": after["cost_usd"], "recovery_zero_dispatch_verified": retry_zero,
                           "combined_episode_cost_usd": item["cost_usd"], "times": times,
                           "timing_scope": "sum of measured attempt times, excluding delay between runs; local compute unpriced"})
        combined.append(item)
    return combined, provenance


def audit_recovery(state, output):
    manifest, manifest_hash = primary.read(output / "recovery_manifest.json")
    recovery.verify_recovery_manifest(manifest, state)
    rows, events, budget = recovery.read_recovery(state, manifest, output)
    if len(rows) != len(state["eligible"]):
        raise ValueError("Recovery is incomplete; do not drop an eligible initialization failure")
    status, status_hash = primary.read(output / "recovery_status.json")
    primary.check_seal(status, "status_id")
    unknown = sum(event["cost_usd"] is None for event in events)
    subtotal = math.fsum(event["cost_usd"] for event in events if event["cost_usd"] is not None)
    checks = {"recovery_id": manifest["recovery_id"], "freeze_id": state["frozen"]["freeze_id"],
              "eligible_count": len(state["eligible"]), "completed_recovery_episodes": len(rows), "complete": True,
              "no_op": not state["eligible"], "recovery_http_attempts": len(events),
              "combined_http_attempts": len(events) + len(state["events"]), "combined_token_budget": budget,
              "recovery_unknown_http_costs": unknown, "primary_observations_unchanged": True,
              "primary_validation_approved_by_recovery": False, "net_monetary_savings_verified": False}
    if (any(status.get(key) != value for key, value in checks.items())
            or not primary.close(status.get("recovery_known_tariff_subtotal_usd"), subtotal)
            or not primary.close(status.get("recovery_total_tariff_usd"), None if unknown else subtotal)):
        raise ValueError("Recovery status differs from the complete reconciled retry records")
    when = primary.utc(manifest["created_at_utc"])
    if any(primary.utc(event["started_at_utc"]) > when for event in state["events"]):
        raise ValueError("Recovery was not declared after the original acquisition")
    if any(primary.utc(event["started_at_utc"]) < when for event in events):
        raise ValueError("A recovery dispatch preceded its manifest")
    for row in rows:
        start, _ = primary.read(output / "episode_starts" / (primary.runner._episode_id(row["task_id"], row["model"]) + ".json"))
        if primary.utc(start["started_at_utc"]) < when:
            raise ValueError("A recovery episode preceded its manifest")
    io = primary.audit_io_events(output, manifest["recovery_id"], manifest["created_at_utc"])
    packages, package_hash = primary.read(output / "recovery_packages.json") if rows else (None, None)
    original_packages, _ = primary.read(state["run"] / "run_manifest.json")
    if rows and packages != original_packages["packages"]:
        raise ValueError("Recovery used different SDK package versions")
    zero_retry = recovery.eligible_primary_episodes(rows, events, output)
    combined, provenance = combine_episodes(state["observations"], rows, state["eligible"], zero_retry)
    snapshots = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() if (output / name).exists() else None
                 for name in ("observations.jsonl", "http_attempts.jsonl", "io_retry_events.jsonl")}
    audit = {"recovery_manifest_sha256": manifest_hash, "recovery_status_sha256": status_hash,
             "secondary_reporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "recovery_packages_sha256": package_hash, "recovery_snapshot_sha256": snapshots,
             "recovery_id": manifest["recovery_id"], "created_at_utc": manifest["created_at_utc"],
             "source_sha256": manifest["source_sha256"], "original_input_sha256": state["original_input_sha256"],
             "eligible_episode_ids": manifest["eligible_episode_ids"], "all_eligible_episodes_recovered_once": True,
             "combined_token_budget": budget, "io_retry_events": io["event_count"],
             "original_primary_files_unchanged": True, "policy_and_routes_unchanged": True,
             "eligibility_scope": manifest["eligibility_rule"]}
    return combined, provenance, events, audit


def make_report(run=primary.DEFAULT_RUN, suite_path=primary.DEFAULT_SUITE, policy_path=primary.DEFAULT_POLICY,
                output=None, reference_path=None, selection_path=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output is not None else run / "pre_dispatch_recovery"
    # This independently audits the entire primary result before considering
    # retries; the returned object is a new dictionary and no primary file writes.
    primary_report = primary.make_report(run, suite_path, policy_path, reference_path, selection_path)
    state = recovery.load_original(run, suite_path, policy_path)
    combined, episode_provenance, events, audit = audit_recovery(state, output)
    policy, suite, frozen = state["policy"], state["suite"], state["frozen"]
    reference_path = primary.runner.BERT_REFERENCE_PATH if reference_path is None else reference_path
    reference, _ = primary.read(reference_path)
    scorer = primary.cached_bert_scorer(reference_path, suite["tasks"])
    options = {"scorer": scorer} if policy["representation"] == "routellm-bert" else {}
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    records = [{**{key: row[key] for key in ("task_id", "group_id", "model", "accepted", "cost_usd", "cost_complete", "cost_basis")},
                "prompt_text": primary.runner.task_prompt(tasks[row["task_id"]])} for row in combined]
    validation = primary.policy_api.evaluate_policy(policy, records, **options)
    # Bind the runtime-compatible sealed report to the explicitly secondary
    # infrastructure analysis. Its primary counterpart is not rewritten.
    validation.pop("validation_report_id")
    validation["infrastructure_recovery"] = {"recovery_id": audit["recovery_id"],
        "primary_validation_report_id": primary.read(run / "validation.json")[0]["validation_report_id"],
        "original_failed_episodes_preserved": True, "primary_result_replaced": False,
        "scope": "secondary infrastructure recovery; same frozen policy and cohort, not an independent new policy trial"}
    validation = primary.policy_api.validate_report(primary.policy_api._seal(validation, "validation_report_id"), policy)
    policies, comparisons = primary.summarize_controls(suite, combined, frozen["routes"], reference["rows"])
    if any(policies["relay_adaptive"][key] != value for key, value in validation["metrics"]["relay"].items()):
        raise ValueError("Recovered paired summary differs from the sealed core metrics")
    report = deepcopy(primary_report)
    report.pop("report_id")
    report.update(kind="audited_openhands_adaptation_recovery", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  policies=policies, relay_comparisons=comparisons,
                  primary_result={"validation_report_id": validation["infrastructure_recovery"]["primary_validation_report_id"],
                                  "activation": primary_report["activation"], "policies": primary_report["policies"],
                                  "comparison_sha256": primary_report["audit"]["comparison_sha256"],
                                  "original_observations_preserved": True},
                  infrastructure_recovery={"recovery_id": audit["recovery_id"], "episode_provenance": episode_provenance,
                                           "eligible_count": len(state["eligible"]), "secondary_validation_report_id": validation["validation_report_id"],
                                           "policy_retuned": False, "independent_new_trial": False,
                                           "primary_result_replaced": False, "initial_llm_zero_basis": "verified zero logical calls AND zero HTTP starts/completions; not a missing-usage imputation"})
    report["activation"].update(eligible=validation["activation_eligible"], reasons=validation["reasons"],
                                result_scope="secondary infrastructure recovery only; does not approve or replace primary result")
    report["routing_compute"]["break_even_cpu_price_usd_per_hour_vs_strong"] = primary.break_even_cpu_hourly(
        policies["fixed_strong"], policies["relay_adaptive"], report["routing_compute"]["resident_reconstructed_seconds"])
    all_events = state["events"] + events
    unknown = sum(event["cost_usd"] is None for event in all_events)
    known = math.fsum(event["cost_usd"] for event in all_events if event["cost_usd"] is not None)
    report["acquisition"].update(episodes=len(state["observations"]) + len(state["eligible"]),
        primary_episode_attempts=len(state["observations"]), isolated_recovery_attempts=len(state["eligible"]),
        logical_task_model_episodes=len(combined), http_attempts=len(all_events), unknown_http_costs=unknown,
        known_uncached_tariff_subtotal_usd=known, total_uncached_tariff_usd=None if unknown else known,
        scope="original complete paired acquisition plus isolated initialization recoveries; zero-dispatch LLM costs proven, local work and delay between runs unpriced")
    report["audit"]["isolated_recovery"] = audit
    report["claim_limits"]["additional_scope"] = "secondary recovery analysis of the same cohort; original primary result remains separately reported"
    report["report_id"] = primary.runner._digest(report)
    return report, validation, records


def markdown(report):
    view = deepcopy(report)
    # The primary execution-amendment paragraph explicitly names a primary
    # result. Its evidence remains in JSON; use a recovery-specific paragraph.
    view["audit"]["execution_amendment"] = {"present": False}
    ordinary = primary.markdown(view)
    table = ordinary[ordinary.index("| Política"):]
    recovery_info = report["infrastructure_recovery"]
    gate = ("La evaluación secundaria cumple el gate empírico para la misma política congelada."
            if report["activation"]["eligible"] else
            "La evaluación secundaria no cumple el gate; el cargador debe conservar baseline con este informe.")
    introduction = (f"Análisis secundario de recuperación de infraestructura: {recovery_info['eligible_count']} episodio(s) "
                    "fallaron por PermissionError antes de cualquier llamada lógica o despacho HTTP y recibieron un único intento aislado. "
                    "Se incluyeron todos los casos elegibles mediante esa regla, sin seleccionar por respuesta o ruta. "
                    "No es una nueva prueba independiente ni una política reajustada. El resultado primario y sus fallos permanecen intactos.")
    timing = ("En cada episodio recuperado se conserva el fallo inicial y se suman sus tiempos medidos a los del intento aislado; "
              "no se incluye la espera entre ejecuciones. El cero inicial corresponde exclusivamente a llamadas LLM cuya ausencia "
              "se verificó en llamadas lógicas y bitácoras HTTP; el trabajo local no se considera gratis. La enmienda de IO se añadió "
              "después del prefijo inicial de la adquisición, no estuvo activa desde el comienzo.")
    return introduction + "\n\n" + gate + " Esta conclusión secundaria no cambia la aprobación del resultado primario.\n\n" + table + "\n" + timing + "\n"


def write_report(report, validation, records, recovery_output, output=DEFAULT_OUTPUT):
    output, recovery_output = Path(output), Path(recovery_output)
    paths = {"json": output.with_suffix(".json"), "markdown": output.with_suffix(".md"),
             "validation": recovery_output / "validation-recovered.json",
             "records": recovery_output / "validation_records_recovered.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("A recovery report or recovered validation already exists; nothing is overwritten")
    report_raw = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    payloads = {"json": report_raw, "markdown": markdown(report),
                "validation": json.dumps(validation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                "records": json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n"}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payloads[name])
    return {name: str(path) for name, path in paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=primary.DEFAULT_RUN)
    parser.add_argument("--suite", type=Path, default=primary.DEFAULT_SUITE)
    parser.add_argument("--policy", type=Path, default=primary.DEFAULT_POLICY)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--reference", type=Path, default=primary.runner.BERT_REFERENCE_PATH)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    recovery_output = args.recovery or args.run / "pre_dispatch_recovery"
    report, validation, records = make_report(args.run, args.suite, args.policy, recovery_output, args.reference, args.selection)
    print(json.dumps(write_report(report, validation, records, recovery_output, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
