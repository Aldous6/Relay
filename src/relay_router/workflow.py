"""Small, provider-independent task routing and metadata-only observations.

The harness owns prompts, tools, generation, retries and the acceptance verdict.
Relay makes one sticky decision per task and never calls a provider itself.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import inspect
import math
from pathlib import Path
import re
import threading
import time
import uuid


def _name(value, field):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}", value):
        raise ValueError(f"{field} must be a nonempty identifier, not free text.")
    return value


def _number(value, field, *, integer=False):
    if value is None:
        return None
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{field} must be finite and nonnegative, or None.")
    return value


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class Relay:
    """Route at task boundaries, optionally using a validated workflow policy.

    ``observe`` preserves the baseline and records what the policy suggested.
    ``apply`` uses approved workflow routes; missing evidence keeps the baseline.
    A prompt selector is a separate experimental engine, enabled explicitly
    with allow_experimental=True for apply and never labelled approved. Models
    and capabilities remain the host's responsibility through eligible_models.
    No prompts, replies, credentials or exception messages enter the log.
    """

    def __init__(self, baseline_model: str, *, policy=None, mode="observe", log_path=None,
                 prompt_selector=None, allow_experimental=False):
        self._baseline = _name(baseline_model, "baseline_model")
        if mode not in {"observe", "apply"}:
            raise ValueError("mode must be observe or apply.")
        self._mode = mode
        if type(allow_experimental) is not bool:
            raise ValueError("allow_experimental must be a boolean.")
        self._prompt_selector = None
        if prompt_selector is not None:
            from .prompt_selection import PairwisePromptSelector
            from .adaptive_runtime import AdaptivePromptSelector
            if type(prompt_selector) not in (PairwisePromptSelector, AdaptivePromptSelector):
                raise TypeError("prompt_selector must be a PairwisePromptSelector or AdaptivePromptSelector.")
            if policy is not None:
                raise ValueError("Choose either a workflow policy or a prompt selector.")
            if prompt_selector.strong_model != self._baseline:
                raise ValueError("The prompt selector's strong model must match the baseline.")
            if type(prompt_selector) is PairwisePromptSelector and mode == "apply" and not allow_experimental:
                raise ValueError("Applying a prompt selector requires allow_experimental=True; it is not an approved workflow policy.")
            self._prompt_selector = prompt_selector
        self._policy = None
        if policy is not None:
            from .workflow_policy import validate_artifact
            if isinstance(policy, (str, Path)):
                policy = json.loads(Path(policy).read_text(encoding="utf-8"))
            self._policy = deepcopy(validate_artifact(policy))
            if self._policy["baseline_model"] != baseline_model:
                raise ValueError("The policy baseline differs from this harness baseline.")
        self._log_path = Path(log_path) if log_path is not None else None
        self._lock = threading.RLock()
        self._active = set()

    def _emit(self, event):
        if self._log_path is None:
            return
        row = {"schema_version": 1, "at_utc": datetime.now(timezone.utc).isoformat(), **event}
        serialized = _encode(row) + "\n"
        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()

    def task(self, workflow: str, *, task_id=None, group_id=None, split="production",
             evaluation_id="production", max_calls=20, eligible_models=None, model_override=None,
             routing_text=None):
        """Start one task, with a caller-supplied workflow label, never inferred text.

        model_override is only for explicit paired development/validation runs.
        max_calls bounds logical dispatches recorded with begin_call, not hidden
        retries inside a client's implementation or monetary expenditure.
        A configured prompt selector requires routing_text and runs once here.
        Prompt selectors cannot be combined with evaluation model_override;
        use a separate Relay instance for explicit paired candidate runs.
        """
        workflow = _name(workflow, "workflow")
        task_id = _name(uuid.uuid4().hex if task_id is None else task_id, "task_id")
        group_id = _name(task_id if group_id is None else group_id, "group_id")
        evaluation_id = _name(evaluation_id, "evaluation_id")
        if split not in {"production", "development", "validation"}:
            raise ValueError("Unknown split.")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer.")
        if self._prompt_selector is None and routing_text is not None:
            raise ValueError("routing_text requires a configured prompt_selector.")
        if self._prompt_selector is not None and model_override is not None:
            raise ValueError("prompt_selector and model_override cannot be combined.")
        eligible = None
        if eligible_models is not None:
            if isinstance(eligible_models, str):
                raise ValueError("eligible_models must be a collection of model names.")
            eligible = {_name(model, "eligible model") for model in eligible_models}
            if not eligible:
                raise ValueError("No eligible model is available.")
        suggested = self._baseline
        reason = "baseline_no_approved_route"
        prompt_selection = None
        if self._prompt_selector is not None:
            prompt_selection = self._prompt_selector.select(routing_text)
            suggested = prompt_selection["model"]
            reason = ("empirical_adaptive_prompt_route"
                      if prompt_selection["status"] == "empirically_validated_pilot" else
                      "baseline_adaptation_gate_not_met"
                      if prompt_selection["status"] == "baseline_no_validation" else "experimental_prompt_route")
        if self._policy is not None:
            suggested = self._policy["routes"].get(workflow, self._baseline)
            if suggested != self._baseline:
                reason = "approved_workflow_route"
        selected = suggested if self._mode == "apply" else self._baseline
        _name(selected, "selected model")
        if self._mode == "observe" and suggested != self._baseline:
            reason = "observation_keeps_baseline"
        effective_mode = self._mode
        if model_override is not None:
            if split == "production":
                raise ValueError("model_override is reserved for explicit paired evaluations.")
            selected = _name(model_override, "model_override")
            effective_mode, reason = "evaluation", "explicit_evaluation_candidate"
        if eligible is not None and selected not in eligible:
            if model_override is not None or self._baseline not in eligible:
                raise ValueError("The selected model is ineligible; no generation was requested.")
            selected, reason = self._baseline, "baseline_after_capability_filter"
        decision = {"model": selected, "suggested_model": suggested, "baseline_model": self._baseline,
                    "mode": effective_mode, "reason": reason,
                    "policy_id": self._policy["policy_id"] if self._policy else None,
                    "learned": False, "scope": "empirical_workflow_policy_not_per_prompt_quality_prediction"}
        if prompt_selection is not None:
            decision.update({"prompt_selection": prompt_selection,
                             "routing_seconds": prompt_selection["routing_seconds"],
                             "learned": None, "scope": "experimental_pairwise_prompt_selection"})
            if "adaptation_scope" in prompt_selection:
                decision.update(learned=True, scope=prompt_selection["adaptation_scope"],
                                policy_id=prompt_selection["selector_id"])
        identity = (evaluation_id, split, workflow, task_id, selected)
        with self._lock:
            if identity in self._active:
                raise ValueError("This task/model observation already exists in this Relay instance.")
            session = TaskSession(self, workflow, task_id, group_id, split, evaluation_id, max_calls, decision)
            self._emit({"kind": "relay_task_started", **session._identity(), "decision": decision})
            self._active.add(identity)
        return session


class TaskSession:
    """One sticky task. Record every logical call, then the host's acceptance verdict."""

    def __init__(self, relay, workflow, task_id, group_id, split, evaluation_id, max_calls, decision):
        self._relay = relay
        self._workflow, self._task_id, self._group_id = workflow, task_id, group_id
        self._split, self._evaluation_id = split, evaluation_id
        self._max_calls, self._decision = max_calls, deepcopy(decision)
        self._session_id = uuid.uuid4().hex
        self._calls = {}
        self._finished = False
        self._started = time.perf_counter() - self._decision.get("routing_seconds", 0.)
        self._lock = threading.RLock()

    @property
    def model(self):
        return self._decision["model"]

    @property
    def decision(self):
        return deepcopy(self._decision)

    def _identity(self):
        return {"session_id": self._session_id, "workflow": self._workflow, "task_id": self._task_id,
                "group_id": self._group_id, "split": self._split, "model": self.model,
                "evaluation_id": self._evaluation_id}

    def _open(self):
        if self._finished:
            raise ValueError("The task is already finished.")

    def begin_call(self, *, model=None):
        with self._lock:
            self._open()
            if model is not None and model != self.model:
                raise ValueError("A task cannot silently change its selected model.")
            if len(self._calls) >= self._max_calls:
                raise ValueError("The task's logical call limit was reached before dispatch.")
            call_id = uuid.uuid4().hex
            self._relay._emit({"kind": "relay_call_started", **self._identity(), "call_id": call_id})
            self._calls[call_id] = {"status": "pending", "started": time.perf_counter()}
            return call_id

    def call(self, generate, /, *args, measure=None, **kwargs):
        """Call synchronous generate(model, ...) once; return its response unchanged.

        An optional measure(response) returns record_call keyword fields such
        as reported usage and cost. It must not return the response content.
        No inspection, provider discovery, retry or fallback is performed.
        """
        if not callable(generate) or (measure is not None and not callable(measure)):
            raise TypeError("generate and measure must be callables.")
        if inspect.iscoroutinefunction(generate) or inspect.iscoroutinefunction(measure):
            raise TypeError("call is synchronous; use begin_call/record_call for async clients.")
        call_id = self.begin_call()
        try:
            response = generate(self.model, *args, **kwargs)
        except BaseException:
            self.record_call(call_id, status="error")
            raise
        try:
            if inspect.isawaitable(response):
                if inspect.iscoroutine(response):
                    response.close()
                raise TypeError("call is synchronous; an awaitable response cannot be measured.")
            metadata = {} if measure is None else measure(response)
            if type(metadata) is not dict or set(metadata) - {
                "cost_usd", "cost_basis", "input_tokens", "output_tokens", "cached_input_tokens", "elapsed_seconds"
            }:
                raise ValueError("measure must return only numeric usage/cost metadata.")
            self.record_call(call_id, **metadata)
        except BaseException:
            if self._calls[call_id]["status"] == "pending":
                self.record_call(call_id, status="unknown")
            raise
        return response

    def record_call(self, call_id, *, model=None, cost_usd=None, cost_basis="unknown",
                    input_tokens=None, output_tokens=None, cached_input_tokens=None,
                    status="completed", elapsed_seconds=None):
        with self._lock:
            self._open()
            if call_id not in self._calls or self._calls[call_id]["status"] != "pending":
                raise ValueError("Record a pending call exactly once.")
            if model is not None and model != self.model:
                raise ValueError("The returned model differs from the task's selected model.")
            if status not in {"completed", "error", "unknown"}:
                raise ValueError("Unknown call status.")
            if cost_basis not in {"unknown", "provider_reported", "tariff_estimate"}:
                raise ValueError("Unknown cost basis.")
            cost_usd = _number(cost_usd, "cost_usd")
            if cost_usd is not None and cost_basis == "unknown":
                raise ValueError("A numeric cost requires an explicit cost basis.")
            counts = {name: _number(value, name, integer=True) for name, value in
                      (("input_tokens", input_tokens), ("output_tokens", output_tokens),
                       ("cached_input_tokens", cached_input_tokens))}
            if (counts["cached_input_tokens"] is not None and counts["input_tokens"] is not None
                    and counts["cached_input_tokens"] > counts["input_tokens"]):
                raise ValueError("Cached input tokens cannot exceed total input tokens.")
            elapsed = (_number(elapsed_seconds, "elapsed_seconds") if elapsed_seconds is not None
                       else max(0., time.perf_counter() - self._calls[call_id]["started"]))
            record = {"call_id": call_id, "status": status, "model": self.model,
                      "cost_usd": cost_usd, "cost_basis": cost_basis if cost_usd is not None else "unknown",
                      "usage": counts, "elapsed_seconds": elapsed}
            self._relay._emit({"kind": "relay_call_finished", **self._identity(), **record})
            self._calls[call_id] = record

    def finish(self, *, accepted, additional_cost_usd=0.0, cost_complete=False):
        """Close with the external verifier's verdict; unknown is never success.

        cost_complete=True is an explicit host declaration: all task calls,
        retries and auxiliary model costs are included. Additional costs use the
        same cost basis as the recorded calls. Missing or mixed cost bases keep
        the task cost unknown. Tokens and costs alone cannot establish savings.
        """
        with self._lock:
            self._open()
            if accepted is not None and type(accepted) is not bool:
                raise ValueError("accepted must be True, False or None.")
            if type(cost_complete) is not bool:
                raise ValueError("cost_complete must be a boolean declaration from the harness.")
            extra = _number(additional_cost_usd, "additional_cost_usd")
            if any(row["status"] == "pending" for row in self._calls.values()):
                raise ValueError("Finalize pending calls before finishing the task.")
            calls = list(self._calls.values())
            bases = {row["cost_basis"] for row in calls}
            all_known = bool(calls) and all(row["cost_usd"] is not None for row in calls)
            complete = cost_complete and all_known and extra is not None and len(bases) == 1
            known_subtotal = math.fsum(row["cost_usd"] for row in calls if row["cost_usd"] is not None)
            total = known_subtotal + extra if complete else None
            if total is not None and not math.isfinite(total):
                raise ValueError("The total task cost must be finite.")
            usage = {name: sum(row["usage"][name] for row in calls)
                     if calls and all(row["usage"][name] is not None for row in calls) else None
                     for name in ("input_tokens", "output_tokens", "cached_input_tokens")}
            result = {"schema_version": 1, "kind": "relay_task_observation", **self._identity(),
                      "accepted": accepted, "cost_usd": total,
                      "cost_basis": next(iter(bases)) if complete else "unknown",
                      "cost_complete": complete, "cost_scope": "recorded_calls_plus_declared_additional",
                      "known_recorded_cost_subtotal_usd": known_subtotal,
                      "additional_cost_usd": extra, "unknown_cost_calls": sum(row["cost_usd"] is None for row in calls),
                      "logical_calls": len(calls), "usage": usage, "calls": deepcopy(calls),
                      "elapsed_seconds": max(0., time.perf_counter() - self._started),
                      "decision": self.decision, "savings_usd": None}
            if "prompt_selection" in self._decision:
                result.update({"routing_seconds": self._decision["routing_seconds"],
                               "routing_cost_usd": None, "routing_cost_complete": False,
                               "cost_scope": "recorded_calls_plus_declared_additional_excludes_unpriced_selector_compute"})
            self._relay._emit(result)
            self._finished = True
            identity = (self._evaluation_id, self._split, self._workflow, self._task_id, self.model)
            with self._relay._lock:
                self._relay._active.discard(identity)
            return result
