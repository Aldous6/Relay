"""Bounded, verifier-driven task cascades. No model calls or quality oracle here.

The default executes the baseline. The experimental mode is explicit and never
claims validated savings. A monetary reservation is an admission control based
on the host's declared per-call bound, NOT a provider-enforced invoice cap.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
from typing import Callable
import uuid

from .workflow import _name


class BudgetExceeded(ValueError):
    """No further dispatch may start under the declared admission limits."""


class AccountingIncomplete(RuntimeError):
    """An unpriced/pending request prevents admitting additional expenditure."""


def _finite(value, field, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class TariffBound:
    """Upper bound for one token-priced generation under the cited contract."""

    max_input_tokens: int
    max_output_tokens: int
    input_usd_per_token: float
    output_usd_per_token: float
    source: str

    def __post_init__(self):
        if any(type(count) is not int or count < 1 for count in (self.max_input_tokens, self.max_output_tokens)):
            raise ValueError("A tariff bound needs positive integer token limits")
        _finite(self.input_usd_per_token, "input_usd_per_token", positive=True)
        _finite(self.output_usd_per_token, "output_usd_per_token", positive=True)
        if not self.source.startswith("https://"):
            raise ValueError("A tariff bound needs its documented source")
        _finite(self.maximum_usd, "maximum_usd", positive=True)

    @property
    def maximum_usd(self):
        return self.max_input_tokens * self.input_usd_per_token + self.max_output_tokens * self.output_usd_per_token


class SpendBudget:
    """Account every actual transport dispatch, including failed/internal retries.

    One request at a time. An identified tool-call rejection may retain a
    documented tariff maximum and admit a different model under the same
    budget. Other unresolved requests stop admission. A measured overrun stops new
    dispatches. The deadline controls admission, not cancellation of an in-flight
    request; provider timeouts and verifier timeouts remain separately required.
    """

    def __init__(self, max_cost_usd, max_calls, max_seconds=300):
        self.max_cost_usd = _finite(max_cost_usd, "max_cost_usd", positive=True)
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.max_calls = max_calls
        self.max_seconds = _finite(max_seconds, "max_seconds", positive=True)
        self._started = time.monotonic()
        self._calls = {}
        self._unobserved_transport = []
        self._journal = None
        self._journal_failed = False
        self._lock = threading.RLock()

    def _bind_journal(self, path):
        if self._journal is not None or self._calls or self._unobserved_transport:
            raise ValueError("Use a fresh budget for each task")
        self._journal = Path(path)
        self._started = time.monotonic()

    def _emit(self, event):
        if self._journal is not None:
            try:
                with self._journal.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
            except OSError:
                self._journal_failed = True
                raise

    def _admission(self, next_model=None):
        if self._journal_failed:
            raise AccountingIncomplete("The transport journal could not be completed")
        if self._unobserved_transport:
            raise AccountingIncomplete("A dispatch escaped transport observation")
        if any(row["status"] == "pending" or row["cost_upper_bound_usd"] is None for row in self._calls.values()):
            raise AccountingIncomplete("A request has unresolved cost; further dispatch is blocked")
        if any(row["provider_error_code"] == "tool_use_failed" and row["model"] == next_model
               for row in self._calls.values()):
            raise AccountingIncomplete("A rejected tool call must end this model branch")
        if len({row["cost_bound_basis"] for row in self._calls.values()}) > 1:
            raise AccountingIncomplete("Mixed cost bases cannot support further admission")
        if any(row.get("reservation_exceeded", False) for row in self._calls.values()):
            raise BudgetExceeded("A request exceeded its declared cost reservation")
        if len(self._calls) >= self.max_calls:
            raise BudgetExceeded("Actual transport call limit reached")
        if time.monotonic() - self._started >= self.max_seconds:
            raise BudgetExceeded("Task admission deadline reached")

    def record_unobserved_transport(self, model):
        """Fail closed if an SDK invocation cannot be joined to a wire request."""
        with self._lock:
            event = {"kind": "unobserved_transport", "model": _name(model, "model")}
            self._unobserved_transport.append(event)
            self._emit(event)

    def begin_call(self, model, reservation_usd, *, tariff_bound: TariffBound | None = None):
        model = _name(model, "model")
        reserve = _finite(reservation_usd, "reservation_usd", positive=True)
        if tariff_bound is not None:
            if not isinstance(tariff_bound, TariffBound):
                raise TypeError("Use a documented TariffBound")
            reserve = max(reserve, tariff_bound.maximum_usd)
        with self._lock:
            self._admission(model)
            if tariff_bound is None and any(row["cost_usd"] is None for row in self._calls.values()):
                raise AccountingIncomplete("Recovery requires a bounded next dispatch")
            spent = math.fsum(row["cost_upper_bound_usd"] for row in self._calls.values())
            if spent + reserve > self.max_cost_usd + 1e-12:
                raise BudgetExceeded("The next reservation would exceed the task admission budget")
            call_id = uuid.uuid4().hex
            row = {"call_id": call_id, "model": model, "reservation_usd": reserve,
                   "status": "pending", "cost_usd": None, "cost_basis": "unknown",
                   "input_tokens": None, "output_tokens": None,
                   "tariff_bound": asdict(tariff_bound) if tariff_bound is not None else None,
                   "cost_upper_bound_usd": None, "cost_bound_basis": None,
                   "http_status": None, "provider_error_code": None, "provider_request_id": None}
            # Write ahead of transport. Failure to journal means no dispatch.
            self._emit({"kind": "cascade_call_started", **row})
            self._calls[call_id] = row
            return call_id

    def finish_call(self, call_id, *, cost_usd=None, cost_basis="tariff_estimate",
                    input_tokens=None, output_tokens=None, status="completed",
                    http_status=None, provider_error_code=None, provider_request_id=None):
        with self._lock:
            if call_id not in self._calls or self._calls[call_id]["status"] != "pending":
                raise ValueError("Finish a pending transport request exactly once")
            if status not in {"completed", "error", "unknown"}:
                raise ValueError("Unknown request status")
            if cost_basis not in {"tariff_estimate", "provider_reported", "unknown"}:
                raise ValueError("Unknown cost basis")
            if cost_usd is not None:
                cost_usd = _finite(cost_usd, "cost_usd")
                if cost_basis == "unknown":
                    raise ValueError("A numeric cost requires a cost basis")
            for value in (input_tokens, output_tokens):
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("Token counts must be nonnegative integers or unknown")
            pending = self._calls[call_id]
            bound = TariffBound(**pending["tariff_bound"]) if pending["tariff_bound"] else None
            if provider_error_code is not None and not (
                bound and provider_error_code == "tool_use_failed" and http_status == 400 and status == "error"
            ):
                raise ValueError("Only a verified, bounded Groq tool rejection can authorize recovery")
            upper = bound.maximum_usd if provider_error_code and cost_usd is None else cost_usd
            limit_exceeded = bound is not None and (
                (input_tokens is not None and input_tokens > bound.max_input_tokens)
                or (output_tokens is not None and output_tokens > bound.max_output_tokens)
                or (cost_usd is not None and cost_usd > bound.maximum_usd + 1e-12))
            row = {**pending, "status": status, "cost_usd": cost_usd,
                   "cost_basis": cost_basis if cost_usd is not None else "unknown",
                   "cost_upper_bound_usd": upper,
                   "cost_bound_basis": cost_basis if cost_usd is not None else "tariff_estimate" if upper is not None else None,
                   "http_status": http_status, "provider_error_code": provider_error_code,
                   "provider_request_id": provider_request_id,
                   "input_tokens": input_tokens, "output_tokens": output_tokens,
                   "reservation_exceeded": limit_exceeded or (cost_usd is not None and cost_usd > pending["reservation_usd"] + 1e-12)}
            self._calls[call_id] = row
            self._emit({"kind": "cascade_call_finished", **row})
            return deepcopy(row)

    def snapshot(self):
        with self._lock:
            calls = deepcopy(list(self._calls.values()))
            bases = {row["cost_basis"] for row in calls}
            trace_complete = not self._journal_failed and not self._unobserved_transport
            complete = (trace_complete and len(bases) <= 1
                        and all(row["status"] != "pending" and row["cost_usd"] is not None for row in calls))
            known = math.fsum(row["cost_usd"] for row in calls if row["cost_usd"] is not None)
            bound_bases = {row["cost_bound_basis"] for row in calls}
            bounded = (trace_complete and len(bound_bases) <= 1 and
                       all(row["cost_upper_bound_usd"] is not None for row in calls))
            if not complete and any(row.get("reservation_exceeded") for row in calls):
                bounded = False
            committed = math.fsum(row["reservation_usd"] if row["cost_upper_bound_usd"] is None
                                  else row["cost_upper_bound_usd"] for row in calls)
            return {"max_cost_usd": self.max_cost_usd, "max_calls": self.max_calls,
                    "max_seconds": self.max_seconds, "http_calls": len(calls), "calls": calls,
                    "http_calls_complete": not self._unobserved_transport,
                    "unobserved_transport": deepcopy(self._unobserved_transport),
                    "cost_complete": complete, "cost_usd": known if complete else None,
                    "journal_complete": not self._journal_failed,
                    "known_cost_subtotal_usd": known, "committed_usd": committed if not self._unobserved_transport else None,
                    "cost_upper_bound_usd": committed if bounded else None,
                    "cost_bound_basis": next(iter(bound_bases)) if bounded and calls else None,
                    "remaining_admission_usd": max(0., self.max_cost_usd - committed) if not self._unobserved_transport else 0.,
                    "cost_basis": next(iter(bases)) if complete and calls else "no_dispatch" if complete else "unknown",
                    "reservation_exceeded": any(row.get("reservation_exceeded", False) for row in calls),
                    "budget_scope": "host_declared_reservation_admission_not_provider_invoice_cap"}


@dataclass(frozen=True)
class AttemptContext:
    attempt_id: int
    model: str
    workspace: Path
    state_dir: Path
    budget: SpendBudget


@dataclass(frozen=True)
class AttemptOutcome:
    status: str = "completed"
    error_type: str | None = None

    def __post_init__(self):
        if self.status not in {"completed", "error"}:
            raise ValueError("Attempt status must be completed or error")
        if self.error_type is not None:
            _name(self.error_type, "error_type")


def _manifest(root, *, max_files=1000, max_bytes=20_000_000):
    if not root.is_dir() or _linked(root):
        raise ValueError("Supply a real, prepared task snapshot directory")
    files, size = {}, 0
    for path in sorted(root.rglob("*")):
        if _linked(path):
            raise ValueError("Task snapshots cannot contain links/junctions")
        if path.is_file():
            size += path.stat().st_size
            if size > max_bytes or len(files) >= max_files:
                raise ValueError("Prepare a task snapshot with at most 1000 files / 20 MB")
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("An empty task snapshot cannot be verified")
    return files


def _linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def _same_snapshot(source, manifest):
    try:
        return _manifest(source) == manifest
    except (OSError, ValueError):
        return False


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_cascade(*, snapshot_dir, output_dir, task_id, candidate_model, baseline_model,
                run_attempt: Callable[[AttemptContext], AttemptOutcome], verifier,
                budget: SpendBudget, mode="baseline", allow_experimental=False,
                cpu_hourly_usd=None, require_initial_failure=True):
    """Run at most two fresh branches and expose ONLY operational verification.

    Callbacks own the existing harness and must meter every transport through
    context.budget. The OpenHands adapter implements that contract. This function
    never receives private tests, final acceptance labels or reference outcomes.
    A passing operational verifier is NOT recorded as final accepted=True.
    No changes are applied to the input snapshot or to a user's original repo.
    By default the operational checks must reproduce the initial defect before
    allowing the cheap branch. Already-passing checks keep the baseline; broken
    or inconclusive check infrastructure stops the task before generation.
    """
    from .cascade_verifier import CommandVerifier, VerificationResult
    if not isinstance(verifier, CommandVerifier):
        raise TypeError("Supply a captured host CommandVerifier")
    if not isinstance(budget, SpendBudget) or budget.snapshot()["http_calls"]:
        raise ValueError("Supply a fresh SpendBudget")
    if not callable(run_attempt):
        raise TypeError("run_attempt must be a callable")
    if type(allow_experimental) is not bool or mode not in {"baseline", "experimental"}:
        raise ValueError("Choose baseline or explicit experimental mode")
    if mode == "experimental" and not allow_experimental:
        raise ValueError("This cascade has no independent savings validation; set allow_experimental=True explicitly")
    if type(require_initial_failure) is not bool:
        raise ValueError("require_initial_failure must be boolean")
    task_id = _name(task_id, "task_id")
    candidate_model = _name(candidate_model, "candidate_model")
    baseline_model = _name(baseline_model, "baseline_model")
    if candidate_model == baseline_model:
        raise ValueError("Candidate and baseline must differ")
    if cpu_hourly_usd is not None:
        cpu_hourly_usd = _finite(cpu_hourly_usd, "cpu_hourly_usd")
    source_path = Path(snapshot_dir).absolute()
    if any(_linked(path) for path in (source_path, *source_path.parents)):
        raise ValueError("Snapshot paths cannot traverse links/junctions")
    source, destination = source_path.resolve(), Path(output_dir).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Output and snapshot must be separate directory trees")
    source_manifest = _manifest(source)
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=False)
    budget._bind_journal(destination / "transport.jsonl")
    _write(destination / "manifest.json", {"schema_version": 1, "task_id": task_id,
           "snapshot_sha256": source_manifest, "candidate_model": candidate_model,
           "baseline_model": baseline_model, "mode": mode,
           "require_initial_failure": require_initial_failure,
           "verifier_config_sha256": verifier.config_sha256,
           "budget": budget.snapshot(), "cpu_hourly_usd": cpu_hourly_usd})
    attempts, selected, reason = [], None, "no_verified_result"
    initial_verification, route_reason = None, "baseline_mode" if mode == "baseline" else "experimental_cascade"
    models = [baseline_model] if mode == "baseline" else [candidate_model, baseline_model]
    if require_initial_failure:
        initial_workspace = destination / "initial-verification"
        shutil.copytree(source, initial_workspace)
        if _manifest(initial_workspace) != source_manifest:
            raise ValueError("Initial snapshot copy mismatch; no dispatch occurred")
        initial_verification = verifier.verify(initial_workspace).to_dict()
        _write(destination / "initial-verification.json", initial_verification)
        if initial_verification["status"] == "inconclusive":
            models, reason, route_reason = [], "initial_verification_inconclusive", "checks_unavailable"
        elif initial_verification["status"] == "passed" and mode == "experimental":
            models, route_reason = [baseline_model], "baseline_checks_do_not_reproduce_defect"
        elif mode == "experimental":
            route_reason = "initial_defect_reproduced"
    for index, model in enumerate(models, 1):
        try:
            budget._admission(model)
        except (BudgetExceeded, AccountingIncomplete) as exc:
            reason = type(exc).__name__
            break
        if not _same_snapshot(source, source_manifest):
            reason = "snapshot_changed"
            break
        branch = destination / f"attempt-{index}"
        workspace, state_dir = branch / "workspace", branch / "state"
        branch.mkdir()
        shutil.copytree(source, workspace)
        if _manifest(workspace) != source_manifest:
            reason = "snapshot_copy_mismatch"
            break
        state_dir.mkdir()
        context = AttemptContext(index, model, workspace, state_dir, budget)
        before = budget.snapshot()["http_calls"]
        attempt_started = time.monotonic()
        try:
            outcome = run_attempt(context)
            if not isinstance(outcome, AttemptOutcome):
                raise TypeError("run_attempt must return AttemptOutcome")
        except Exception as exc:
            outcome = AttemptOutcome("error", type(exc).__name__)
        # Verification exceptions are recorded as unknown, never as success.
        try:
            verification = verifier.verify(workspace)
            if not isinstance(verification, VerificationResult):
                raise TypeError("Verifier returned an invalid result")
            checked = verification.to_dict()
        except Exception as exc:
            checked = {"status": "inconclusive", "error_type": type(exc).__name__,
                       "verifier_config_sha256": verifier.config_sha256}
        after = budget.snapshot()
        row = {"attempt_id": index, "model": model, "status": outcome.status,
               "error_type": outcome.error_type, "verification": checked,
               "transport_calls": after["http_calls"] - before,
               "call_ids": [call["call_id"] for call in after["calls"][before:]],
               "wall_seconds": time.monotonic() - attempt_started,
               "workspace": f"attempt-{index}/workspace"}
        attempts.append(row)
        _write(branch / "attempt.json", row)
        if checked["status"] == "passed" and outcome.status == "completed":
            selected, reason = index, "operational_checks_passed"
            break
        reason = "operational_checks_not_passed"
    elapsed = time.monotonic() - started
    accounting = budget.snapshot()
    compute = None if cpu_hourly_usd is None else elapsed * cpu_hourly_usd / 3600
    complete_total = accounting["cost_usd"] + compute if accounting["cost_complete"] and compute is not None else None
    model_upper = accounting["cost_upper_bound_usd"]
    total_upper = model_upper + compute if model_upper is not None and compute is not None else None
    result = {"schema_version": 1, "kind": "relay_cascade_task", "task_id": task_id,
              "mode": mode, "validation_status": "not_independently_validated",
              "selected_attempt": selected, "selected_model": attempts[selected - 1]["model"] if selected else None,
              "selected_workspace": attempts[selected - 1]["workspace"] if selected else None,
              "operational_passed": selected is not None, "accepted": None,
              "reason": reason, "attempts": attempts, "escalated": len(attempts) == 2,
              "initial_verification": initial_verification,
              "require_initial_failure": require_initial_failure, "route_reason": route_reason,
              "accounting": accounting, "wall_seconds": elapsed,
              "compute_cost_scenario_usd": compute, "cpu_hourly_usd": cpu_hourly_usd,
              "total_cost_scenario_usd": complete_total,
              "total_cost_scenario_upper_bound_usd": total_upper,
              "compute_scope": "whole_task_wall_time_at_host_hourly_rate_not_cpu_measurement",
              "savings_usd": None, "original_snapshot_unchanged": _same_snapshot(source, source_manifest)}
    _write(destination / "result.json", result)
    return result
