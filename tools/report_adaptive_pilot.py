"""Audit a completed frozen pilot and report paired controls, without tuning.

Reads only completed acquisition artifacts. Never calls a model, retrains a
policy, edits acquisition files or overwrites an earlier report.
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
import run_adaptive_pilot as runner
from adaptive_scoring import cached_bert_scorer
from relay_router import adaptive_policy as policy_api

DEFAULT_RUN = ROOT / "outputs/harness-adaptive/v1/run"
DEFAULT_SUITE = ROOT / "data/imported/harness-adaptive-v1/fresh_suite.json"
DEFAULT_POLICY = ROOT / "outputs/harness-adaptive/v1/policy-auto.json"
DEFAULT_OUTPUT = ROOT / "reports/openhands-adaptation-v1"
REFERENCE_SOURCES = {"tools/score_adaptive_reference.py", "src/relay_router/integrations/routellm.py"}


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_seal(value, name):
    contents = dict(value)
    seal = contents.pop(name, None)
    if seal != runner._digest(contents):
        raise ValueError(f"Invalid {name} integrity digest")


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def close(left, right):
    return left == right if left is None or right is None else math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def utc(value):
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        raise ValueError("Audit timestamps must contain a timezone")
    return date.astimezone(timezone.utc)


def metrics(rows):
    accepted = sum(row["accepted"] for row in rows)
    unknown = sum(row["cost_usd"] is None for row in rows)
    bases = sorted({row["cost_basis"] for row in rows})
    basis = bases[0] if len(bases) == 1 and bases[0] != "unknown" else None
    subtotal = math.fsum(row["cost_usd"] for row in rows if row["cost_usd"] is not None)
    complete = unknown == 0 and basis is not None
    total = subtotal if complete else None
    return {"task_count": len(rows), "group_count": len({row["group_id"] for row in rows}),
            "accepted_count": accepted, "acceptance_rate": accepted / len(rows),
            "unknown_cost_count": unknown, "known_cost_subtotal_usd": subtotal,
            "cost_complete": complete, "cost_basis": basis, "cost_bases": bases,
            "total_cost_usd": total,
            "cost_per_accepted_task_usd": total / accepted if total is not None and accepted else None}


def relative_savings(baseline, selected):
    base = baseline["cost_per_accepted_task_usd"]
    value = selected["cost_per_accepted_task_usd"]
    if base is None or base <= 0 or value is None or baseline["cost_basis"] != selected["cost_basis"]:
        return None
    return 1 - value / base


def break_even_cpu_hourly(baseline, selected, resident_seconds):
    """A conditional CPU price ceiling, not measured compute cost or net saving."""
    seconds = number(resident_seconds, "resident routing seconds")
    if (not baseline["cost_complete"] or not selected["cost_complete"]
            or baseline["cost_basis"] != selected["cost_basis"] or seconds <= 0):
        return None
    difference = baseline["total_cost_usd"] - selected["total_cost_usd"]
    return difference / seconds * 3600 if difference > 0 else None


def summarize_controls(suite, observations, frozen_routes, reference_rows):
    tasks = suite["tasks"]
    index = {(row["task_id"], row["model"]): row for row in observations}
    expected = {(task["task_id"], model) for task in tasks for model in suite["models"]}
    if len(index) != len(observations) or set(index) != expected:
        raise ValueError("Every fresh task needs both unique complete model episodes")
    strong, cheap = suite["baseline_model"], next(model for model in suite["models"] if model != suite["baseline_model"])
    routes = {row["task_id"]: row["model"] for row in frozen_routes}
    references = {row["task_id"]: row for row in reference_rows}
    task_ids = {task["task_id"] for task in tasks}
    if len(routes) != len(frozen_routes) or set(routes) != task_ids or set(references) != task_ids or len(references) != len(reference_rows):
        raise ValueError("Frozen Relay and BERT routes must cover the exact cohort")
    choices = {"fixed_strong": {task_id: strong for task_id in task_ids},
               "fixed_cheap": {task_id: cheap for task_id in task_ids}, "relay_adaptive": routes}
    for name, threshold in (("bert_0_50", .50), ("bert_0_58", .58)):
        choices[name] = {task_id: strong if row["truncated_tokens"] > 0 or row["score"] >= threshold else cheap
                         for task_id, row in references.items()}
    baseline = [index[(task["task_id"], strong)] for task in tasks]
    baseline_stats = metrics(baseline)
    results = {}
    for name, route in choices.items():
        selected = [index[(task["task_id"], route[task["task_id"]])] for task in tasks]
        stats = metrics(selected)
        losses = sum(base["accepted"] and not row["accepted"] for base, row in zip(baseline, selected))
        gains = sum(not base["accepted"] and row["accepted"] for base, row in zip(baseline, selected))
        stats.update(paired_losses_vs_strong=losses, paired_gains_vs_strong=gains,
                     paired_loss_upper_95=policy_api._loss_bound(losses, len(tasks)),
                     relative_savings_per_accepted_vs_strong=relative_savings(baseline_stats, stats),
                     selected_candidate_count=sum(model == cheap for model in route.values()),
                     selected_baseline_count=sum(model == strong for model in route.values()),
                     same_choices_as_relay=route == routes)
        # These are measured branch times, not a combined end-to-end router run.
        for field in ("transport_seconds", "wall_seconds", "quota_wait_seconds"):
            values = [row.get(field) for row in selected]
            if all(value is not None for value in values):
                values = [number(value, field) for value in values]
                stats["median_selected_branch_" + field] = statistics.median(values)
                stats["sum_selected_branch_" + field] = math.fsum(values)
            else:
                stats["median_selected_branch_" + field] = None
                stats["sum_selected_branch_" + field] = None
        results[name] = stats
    relay = results["relay_adaptive"]
    comparisons = {}
    for name in ("fixed_strong", "fixed_cheap", "bert_0_50", "bert_0_58"):
        reference = results[name]
        reference_selected = [index[(task["task_id"], choices[name][task["task_id"]])] for task in tasks]
        relay_selected = [index[(task["task_id"], routes[task["task_id"]])] for task in tasks]
        comparisons[name] = {
            "relay_relative_savings_per_accepted": relative_savings(reference, relay),
            "relay_acceptance_delta": relay["acceptance_rate"] - reference["acceptance_rate"],
            "relay_paired_losses": sum(ref["accepted"] and not row["accepted"] for ref, row in zip(reference_selected, relay_selected)),
            "relay_paired_gains": sum(not ref["accepted"] and row["accepted"] for ref, row in zip(reference_selected, relay_selected)),
            "same_choices": choices[name] == routes,
        }
    return results, comparisons


def audit_io_events(run, binding_id, earliest, maximum=5):
    """Validate only bounded local-write retries; these are never HTTP retries."""
    run = Path(run).resolve()
    path = run / "io_retry_events.jsonl"
    if not path.exists():
        return {"event_count": 0, "sha256": None, "added_http_requests": 0}
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    allowed = {"event_id", "at_utc", "amendment_id", "kind", "path", "attempt", "next_attempt", "max_attempts", "errno", "winerror", "added_HTTP_requests"}
    for index, row in enumerate(rows, 1):
        relative = row.get("path")
        if type(relative) is not str or not relative or Path(relative).is_absolute() or not (run / relative).resolve().is_relative_to(run):
            raise ValueError("An IO retry path escapes its recorded run")
        attempt = row.get("attempt")
        if (set(row) != allowed or row.get("event_id") != index or row.get("amendment_id") != binding_id
                or row.get("kind") != "local_atomic_write_retry" or row.get("added_HTTP_requests") != 0
                or type(attempt) is not int or not 1 <= attempt < maximum or row.get("next_attempt") != attempt + 1
                or row.get("max_attempts") != maximum or utc(row["at_utc"]) < utc(earliest)
                or any(row.get(key) is not None and type(row[key]) is not int for key in ("errno", "winerror"))):
            raise ValueError("IO retry metadata differs from its bounded execution amendment")
    return {"event_count": len(rows), "sha256": hashlib.sha256(content).hexdigest(), "added_http_requests": 0}


def audit_execution_amendment(run, policy, frozen, observations, events):
    """Preserve and distinguish the prefix run before the IO repair existed."""
    path = Path(run) / "execution_amendment.json"
    if not path.exists():
        if (Path(run) / "io_retry_events.jsonl").exists():
            raise ValueError("IO retry events require a recorded execution amendment")
        return {"present": False, "io_retry_events": 0}
    import continue_adaptive_pilot as continuation
    amendment, digest = read(path)
    check_seal(amendment, "amendment_id")
    expected = {"schema_version": 1, "kind": "bounded_filesystem_execution_amendment",
                "freeze_id": frozen["freeze_id"], "artifact_id": policy["artifact_id"],
                "frozen_routes_sha256": hashlib.sha256((Path(run) / "frozen_routes.json").read_bytes()).hexdigest(),
                "source_sha256": continuation.source_hashes(), "reason": continuation.REASON,
                "write_retry": continuation._write_contract()}
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise ValueError("Execution amendment source or frozen contract no longer matches")
    snapshot = amendment["original_snapshot"]
    count, attempts = snapshot["completed_episodes"], snapshot["http_attempts"]
    if (type(count) is not int or not 0 < count < len(observations)
            or type(attempts) is not int or not 0 <= attempts <= len(events)):
        raise ValueError("Invalid pre-amendment acquisition prefix")
    prefix = observations[:count]
    failures = [row for row in prefix if row.get("error_type")]
    pre_http = [row for row in failures if row.get("error_type") == "PermissionError" and row.get("http_attempts") == 0]
    checks = {"completed_episode_ids": continuation._ids(prefix), "error_episode_ids": continuation._ids(failures),
              "pre_http_PermissionError_episode_ids": continuation._ids(pre_http),
              "observations_prefix_sha256": runner._digest(prefix), "http_prefix_sha256": runner._digest(events[:attempts])}
    if not pre_http or any(snapshot.get(key) != value for key, value in checks.items()):
        raise ValueError("The original pre-amendment observations or dispatches changed")
    amended_at = utc(amendment["created_at_utc"])
    if amended_at < utc(frozen["frozen_at_utc"]):
        raise ValueError("Execution amendment cannot precede the frozen policy")
    for position, row in enumerate(observations):
        start, _ = read(Path(run) / "episode_starts" / (runner._episode_id(row["task_id"], row["model"]) + ".json"))
        if (position < count and utc(start["started_at_utc"]) > amended_at
                or position >= count and utc(start["started_at_utc"]) < amended_at):
            raise ValueError("Episode chronology disagrees with the execution amendment boundary")
    if (any(utc(event["started_at_utc"]) > amended_at for event in events[:attempts])
            or any(utc(event["started_at_utc"]) < amended_at for event in events[attempts:])):
        raise ValueError("HTTP chronology disagrees with the execution amendment boundary")
    io = audit_io_events(run, amendment["amendment_id"], amendment["created_at_utc"], continuation.MAX_WRITE_ATTEMPTS)
    return {"present": True, "amendment_id": amendment["amendment_id"], "sha256": digest,
            "created_at_utc": amendment["created_at_utc"], "source_sha256": amendment["source_sha256"],
            "original_completed_episodes_before_guard": count, "original_http_attempts_before_guard": attempts,
            "original_pre_http_permission_errors_preserved": continuation._ids(pre_http),
            "original_prefix_unchanged": True, "io_retry_events": io["event_count"],
            "io_retry_events_sha256": io["sha256"], "added_http_requests": 0,
            "scope": "guard added after the recorded initial prefix; initial failed episodes remain in the primary result"}


def audit_run(run, suite_path, policy_path, reference_path):
    """Read/recompute sealed artifacts; never invoke the mutating runner CLI."""
    run = Path(run)
    suite, suite_hash = read(suite_path)
    policy, policy_hash = read(policy_path)
    frozen, freeze_file_hash = read(run / "frozen_routes.json")
    comparison, comparison_hash = read(run / "comparison.json")
    validation, validation_hash = read(run / "validation.json")
    manifest, manifest_hash = read(run / "run_manifest.json")
    reference, reference_hash = read(reference_path)
    check_seal(frozen, "freeze_id")
    check_seal(comparison, "report_id")
    runner.validate_suite(suite)
    policy_api.validate_artifact(policy)
    policy_api.validate_report(validation, policy)
    contract = frozen["contract"]
    expected = {"suite_sha256": suite_hash, "policy_sha256": policy_hash,
                "suite_content_sha256": runner._digest(suite), "policy_content_sha256": runner._digest(policy),
                "artifact_id": policy["artifact_id"], "source_sha256": runner.source_hashes(),
                "tariffs": runner.TARIFFS, "limits": runner.FIXED_LIMITS}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen sources, policy, suite, tariffs or limits no longer match")
    if (contract.get("bert_reference", {}).get("sha256") != reference_hash
            or reference.get("suite_sha256") != suite_hash):
        raise ValueError("BERT cache no longer matches the frozen reference and suite")
    reference_sources = {name.replace("\\", "/"): digest for name, digest in reference.get("source_sha256", {}).items()}
    if (set(reference_sources) != REFERENCE_SOURCES
            or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in reference_sources.items())):
        raise ValueError("BERT reference generation sources differ from their recorded hashes")
    if (comparison.get("contract") != contract or comparison.get("freeze_id") != frozen["freeze_id"]
            or comparison.get("evaluation") != validation or manifest.get("freeze_id") != frozen["freeze_id"]):
        raise ValueError("Acquisition, comparison and validation are not bound to the same frozen routes")
    scorer = cached_bert_scorer(reference_path, suite["tasks"])
    options = {"scorer": scorer} if policy["representation"] == "routellm-bert" else {}
    references = {row["task_id"]: row for row in reference["rows"]}
    routes = {row["task_id"]: row for row in frozen["routes"]}
    if len(routes) != len(frozen["routes"]) or set(routes) != {task["task_id"] for task in suite["tasks"]}:
        raise ValueError("Frozen route coverage differs from the task cohort")
    for task in suite["tasks"]:
        text = runner.task_prompt(task)
        decision = policy_api.select_model(policy, text, **options)
        row = routes[task["task_id"]]
        expected_route = {"group_id": task["group_id"], "routing_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "model": decision["model"], "score": decision["score"],
                          "input_tokens": decision.get("input_tokens"), "truncated_tokens": decision.get("truncated_tokens"),
                          "encoder_seconds": references[task["task_id"]]["routing_seconds"] if options else 0.}
        if any(row.get(key) != value for key, value in expected_route.items()):
            raise ValueError("A frozen policy decision or its encoder timing no longer reproduces")
        number(row["routing_seconds"], "frozen routing head seconds")
    observations = runner.read_observations(suite, run)
    events = runner.reconcile_http(suite, run, observations)
    if len(observations) != 2 * len(suite["tasks"]):
        raise ValueError("A complete paired cohort is required; incomplete acquisition is not a result")
    frozen_time = utc(frozen["frozen_at_utc"])
    for row in observations:
        if (type(row.get("accepted")) is not bool or type(row.get("public_passed")) is not bool
                or type(row.get("private_passed")) is not bool
                or row["accepted"] != (row["public_passed"] and row["private_passed"])):
            raise ValueError("Acceptance differs from the public and private verifier results")
        if (type(row.get("logical_calls")) is not int or not 0 <= row["logical_calls"] <= suite["max_calls_per_episode"]
                or row.get("cost_basis") != ("tariff_estimate" if row["cost_complete"] else "unknown")):
            raise ValueError("Episode call bound or cost basis differs from the protocol")
        start, _ = read(run / "episode_starts" / (runner._episode_id(row["task_id"], row["model"]) + ".json"))
        if start.get("freeze_id") != frozen["freeze_id"] or utc(start["started_at_utc"]) < frozen_time:
            raise ValueError("Episode start was not bound to the earlier frozen route plan")
    if any(utc(event["started_at_utc"]) < frozen_time for event in events):
        raise ValueError("A model dispatch preceded policy freezing")
    if utc(reference["at_utc"]) > frozen_time:
        raise ValueError("The reference was scored after the route freeze")
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    records = [{**{key: row[key] for key in ("task_id", "group_id", "model", "accepted", "cost_usd", "cost_complete", "cost_basis")},
                "prompt_text": runner.task_prompt(tasks[row["task_id"]])} for row in observations]
    saved_records, records_hash = read(run / "validation_records.json")
    if saved_records != records:
        raise ValueError("Saved validation records differ from the audited paired observations")
    reproduced = policy_api.evaluate_policy(policy, records, **options)
    if reproduced != validation:
        raise ValueError("Sealed validation does not reproduce from all audited observations")
    amendment = audit_execution_amendment(run, policy, frozen, observations, events)
    policies, comparisons = summarize_controls(suite, observations, frozen["routes"], reference["rows"])
    for key, value in validation["metrics"]["relay"].items():
        if policies["relay_adaptive"][key] != value:
            raise ValueError("Independent Relay summary differs from sealed core metrics")
    if (policies["relay_adaptive"]["paired_loss_upper_95"] != validation["paired_loss_upper_95"]
            or policies["relay_adaptive"]["paired_losses_vs_strong"] != validation["paired_losses"]):
        raise ValueError("Paired quality or Wilson bound differs from the core validation")
    known = math.fsum(event["cost_usd"] for event in events if event["cost_usd"] is not None)
    unknown = sum(event["cost_usd"] is None for event in events)
    total = None if unknown else known
    head = math.fsum(row["routing_seconds"] for row in frozen["routes"])
    encoder = math.fsum(row["encoder_seconds"] for row in frozen["routes"])
    checks = {"fresh_tasks": len(tasks), "completed_episodes": len(observations), "http_attempts": len(events),
              "unknown_http_costs": unknown, "acquisition_known_tariff_subtotal_usd": known,
              "acquisition_total_tariff_usd": total, "routing_policy_seconds": head,
              "routing_encoder_seconds": encoder, "routing_seconds": head + encoder,
              "routing_encoder_load_seconds": scorer.provenance["original_load_seconds"] if options else 0.}
    if any(not close(comparison.get(key), value) for key, value in checks.items()):
        raise ValueError("Acquisition comparison totals or routing time do not reproduce")
    audit = {"suite_sha256": suite_hash, "policy_sha256": policy_hash, "frozen_routes_sha256": freeze_file_hash,
             "reporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "comparison_sha256": comparison_hash, "validation_sha256": validation_hash,
             "run_manifest_sha256": manifest_hash, "bert_reference_sha256": reference_hash,
             "validation_records_sha256": records_hash, "bert_reference_sources_verified": reference_sources,
             "recorded_runtime_packages": manifest.get("packages"),
             "observations_sha256": hashlib.sha256((run / "observations.jsonl").read_bytes()).hexdigest(),
             "http_attempts_sha256": hashlib.sha256((run / "http_attempts.jsonl").read_bytes()).hexdigest(),
             "source_hashes_match_freeze": True, "routes_reproduced": True, "all_dispatches_reconciled": True,
             "validation_reproduced_without_tuning": True, "freeze_id": frozen["freeze_id"],
             "frozen_at_utc": frozen["frozen_at_utc"],
             "first_dispatch_at_utc": min((event["started_at_utc"] for event in events), default=None),
             "execution_amendment": amendment,
             "integrity_scope": "local hashes and recorded chronology; not independently certified evidence"}
    return suite, policy, validation, policies, comparisons, checks, audit, scorer.provenance


def training_metadata(selection_path, policy):
    if selection_path is None or not Path(selection_path).exists():
        return {"recorded": False, "head_fit_and_selection_seconds": None, "compute_cost_usd": None}
    selection, digest = read(selection_path)
    if (selection.get("selected_artifact_id") != policy["artifact_id"]
            or selection.get("selection_scope") != "historical_calibration_only_no_fresh_validation_outcomes"
            or policy not in selection.get("candidates", [])):
        raise ValueError("Training selection metadata does not identify the frozen policy")
    for candidate in selection["candidates"]:
        policy_api.validate_artifact(candidate)
    provenance = selection.get("scorer_provenance") or {}
    fields = {}
    for key in ("original_load_seconds", "original_scoring_seconds", "original_total_compute_seconds"):
        fields["historical_encoder_" + key] = number(provenance[key], key) if key in provenance else None
    return {"recorded": True, "selection_sha256": digest, "head_fit_and_selection_seconds": number(selection["elapsed_seconds"], "training seconds"),
            **fields, "compute_cost_usd": None, "amortization_assumed": False,
            "scope": "saved training metadata; original historical encoder work is separate from cached fit/selection"}


def make_report(run=DEFAULT_RUN, suite_path=DEFAULT_SUITE, policy_path=DEFAULT_POLICY, reference_path=None, selection_path=None):
    reference_path = runner.BERT_REFERENCE_PATH if reference_path is None else reference_path
    selection_path = Path(str(policy_path) + ".selection.json") if selection_path is None else selection_path
    suite, policy, validation, policies, comparisons, acquisition, audit, reference = audit_run(run, suite_path, policy_path, reference_path)
    resident = acquisition["routing_seconds"]
    result = {
        "schema_version": 1, "kind": "audited_openhands_adaptation_pilot", "evaluation_id": suite["evaluation_id"],
        "artifact_id": policy["artifact_id"], "representation": policy["representation"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "fresh_tasks": len(suite["tasks"]),
        "policies": policies, "relay_comparisons": comparisons,
        "activation": {"eligible": validation["activation_eligible"], "reasons": validation["reasons"],
                       "scope": policy["scope"], "criteria": policy["criteria"],
                       "production_approved": False, "scope_note": "empirical pilot gate; earlier per-workflow gate unchanged"},
        "routing_compute": {"resident_reconstructed_seconds": resident,
                            "earlier_encoder_inference_seconds": acquisition["routing_encoder_seconds"],
                            "cached_head_replay_seconds": acquisition["routing_policy_seconds"],
                            "encoder_cold_load_seconds": acquisition["routing_encoder_load_seconds"],
                            "bert_control_earlier_inference_seconds": reference["original_scoring_seconds"],
                            "bert_control_cold_load_seconds": reference["original_load_seconds"],
                            "break_even_cpu_price_usd_per_hour_vs_strong": break_even_cpu_hourly(policies["fixed_strong"], policies["relay_adaptive"], resident),
                            "break_even_scope": "conditional ceiling from LLM tariff difference / reconstructed resident seconds; excludes loading, training and evidence acquisition",
                            "measurement": "earlier real encoder timing plus frozen cached-head replay; not measured end-to-end policy latency",
                            "monetary_cost_usd": None, "net_savings_verified": False},
        "training": training_metadata(selection_path, policy),
        "acquisition": {"episodes": acquisition["completed_episodes"], "http_attempts": acquisition["http_attempts"],
                        "unknown_http_costs": acquisition["unknown_http_costs"],
                        "known_uncached_tariff_subtotal_usd": acquisition["acquisition_known_tariff_subtotal_usd"],
                        "total_uncached_tariff_usd": acquisition["acquisition_total_tariff_usd"],
                        "scope": "both fresh measured branches; not an operational policy cost and excludes prior development/diagnostics/local compute",
                        "actual_paid_bill_usd": None, "amortization_assumed": False},
        "cost_semantics": "uncached Groq tariff scenario from all recorded HTTP attempts, including failures and retries; unknown totals stay null",
        "reference": reference, "audit": audit,
        "claim_limits": {"general_routellm_superiority_established": False, "not_diamond_evaluated": False,
                         "net_monetary_savings_established": False, "statistical_noninferiority_established": False,
                         "comparison": "same paired branches; local published RouteLLM-BERT checkpoint at frozen .50/.58 with baseline on truncation, not official RouteLLM SDK",
                         "population": f"{len(suite['tasks'])} fresh-for-routing historical public Python tasks; pretraining contamination possible, not customer repositories"},
    }
    result["report_id"] = runner._digest(result)
    return result


def markdown(report):
    def money(value):
        return "desconocido" if value is None else f"USD {value:.8f}"
    def percent(value):
        return "desconocido" if value is None else f"{100 * value:.2f}%"
    status = ("La adaptación cumple el gate empírico de este piloto y puede activarse dentro de su alcance declarado."
              if report["activation"]["eligible"] else
              "La adaptación no cumple el gate de este piloto; el cargador conserva el modelo baseline.")
    text = [status, f"Se verificaron {report['fresh_tasks']} tareas nuevas para la selección de Relay y {report['acquisition']['episodes']} episodios pareados. "
            "Cada política reutiliza las mismas dos ramas medidas por tarea; las rutas de Relay se congelaron antes de las llamadas.",
            "| Política | Aceptadas | Coste LLM total | Coste por aceptada | Pérdidas pareadas frente a 120B | Ahorro por aceptada frente a 120B |",
            "|---|---:|---:|---:|---:|---:|"]
    names = {"fixed_strong": "120B fijo", "fixed_cheap": "20B fijo", "relay_adaptive": "Relay adaptado",
             "bert_0_50": "BERT umbral 0.50", "bert_0_58": "BERT umbral 0.58"}
    for name, stats in report["policies"].items():
        text.append(f"| {names[name]} | {stats['accepted_count']}/{stats['task_count']} | {money(stats['total_cost_usd'])} | "
                    f"{money(stats['cost_per_accepted_task_usd'])} | {stats['paired_losses_vs_strong']} | {percent(stats['relative_savings_per_accepted_vs_strong'])} |")
    text.append("")
    relay = report["policies"]["relay_adaptive"]
    text.append(f"Relay seleccionó 20B en {relay['selected_candidate_count']} tareas y 120B en {relay['selected_baseline_count']}. "
                f"El límite superior Wilson unilateral del 95% para pérdidas pareadas observado es {percent(relay['paired_loss_upper_95'])}; "
                "es descriptivo y no demuestra no inferioridad poblacional.")
    if report["activation"]["reasons"]:
        text.append("Motivos registrados por el gate: " + ", ".join(report["activation"]["reasons"]) + ".")
    for name in ("fixed_cheap", "bert_0_50", "bert_0_58"):
        comparison = report["relay_comparisons"][name]
        text.append(f"Frente a {names[name]}, Relay presenta {comparison['relay_paired_losses']} pérdidas y {comparison['relay_paired_gains']} ganancias pareadas; "
                    f"su variación de coste por aceptada es {percent(comparison['relay_relative_savings_per_accepted'])} de ahorro. "
                    + ("Las decisiones son idénticas: esta muestra no demuestra una ventaja incremental de routing frente a ese control." if comparison["same_choices"] else
                       "La diferencia es propia de esta muestra y no acredita superioridad general."))
    compute, train = report["routing_compute"], report["training"]
    text.append(f"El routing residente reconstruido suma {compute['resident_reconstructed_seconds']:.4f} s: "
                f"{compute['earlier_encoder_inference_seconds']:.4f} s de encoder medidos anteriormente y "
                f"{compute['cached_head_replay_seconds']:.4f} s de replay de la cabeza con caché. "
                f"La carga del encoder agrega {compute['encoder_cold_load_seconds']:.4f} s. Estos tiempos no son latencia end-to-end.")
    ceiling = compute["break_even_cpu_price_usd_per_hour_vs_strong"]
    text.append((f"El precio horario de CPU de equilibrio, condicionado a ese ahorro LLM y al tiempo residente reconstruido, es USD {ceiling:.6f}/h. "
                 "Excluye carga, entrenamiento y adquisición; no es una tarifa medida ni ahorro neto."
                 if ceiling is not None else "No se puede calcular un precio horario positivo de CPU de equilibrio con costes completos y ahorro LLM positivo."))
    text.append((f"El ajuste y selección automáticos registraron {train['head_fit_and_selection_seconds']:.4f} s; el trabajo histórico del encoder se conserva aparte en el JSON. "
                 if train["recorded"] else "No hay tiempo de entrenamiento verificable en los metadatos disponibles. ")
                + "No se supone amortización ni precio de cómputo.")
    acquisition = report["acquisition"]
    text.append(f"La adquisición de las dos ramas conserva {acquisition['http_attempts']} despachos HTTP y {acquisition['unknown_http_costs']} costes desconocidos; "
                f"subtotal tarifario conocido {money(acquisition['known_uncached_tariff_subtotal_usd'])}, total {money(acquisition['total_uncached_tariff_usd'])}. "
                "Este gasto de adquisición se informa separado del coste de ejecutar cada política.")
    amendment = report["audit"].get("execution_amendment", {})
    if amendment.get("present"):
        text.append(f"Tras los primeros {amendment['original_completed_episodes_before_guard']} episodios se registró una enmienda de ejecución "
                    f"para reintentar escrituras locales bloqueadas por Windows. Se verificaron {amendment['io_retry_events']} reintentos de IO, "
                    "sin reintentos HTTP añadidos por ese mecanismo. El guard no estuvo presente desde el inicio; "
                    "los fallos anteriores y sus observaciones permanecen íntegros en este resultado primario.")
    text.extend(["Los importes incluyen todos los intentos de las ramas seleccionadas, incluso fallos y reintentos. Los costes desconocidos no se convierten en cero. "
                 "Son escenarios tarifarios sin caché, no una factura. El ahorro neto queda sin verificar porque falta monetizar cómputo y adquisición.",
                 "La referencia BERT usa el checkpoint publicado de RouteLLM, con otra supervisión y umbrales históricos congelados; conserva 120B si truncó el prompt. "
                 "No es una ejecución del SDK oficial, no cubre todas las variantes RouteLLM y no se evaluó Not Diamond.",
                 "HumanEval y QuixBugs son fuentes públicas históricas potencialmente presentes en el preentrenamiento. Este piloto no demuestra preparación productiva, "
                 "ausencia de deterioro poblacional ni resultados en repositorios de clientes. El gate anterior por workflow permanece intacto."])
    return "\n\n".join(text[:2]) + "\n\n" + "\n".join(text[2:9]) + "\n\n" + "\n\n".join(text[9:]) + "\n"


def write_report(report, output=DEFAULT_OUTPUT):
    output = Path(output)
    json_path = output.with_suffix(".json")
    md_path = output.with_suffix(".md")
    if json_path.exists() or md_path.exists():
        raise FileExistsError("Report already exists; reports are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with md_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(report))
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--reference", type=Path, default=runner.BERT_REFERENCE_PATH)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = make_report(args.run, args.suite, args.policy, args.reference, args.selection)
    print(json.dumps(write_report(result, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
