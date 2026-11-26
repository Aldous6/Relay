"""Bounded, synchronous OpenHands attempts for a verifier-driven cascade.

The host supplies a fresh Agent and local tools for each clean workspace. This
adapter observes the actual LiteLLM chat dispatch boundary, including SDK repair
attempts, instead of estimating cost from a cumulative conversation metric.

The reservation is a host-declared admission estimate, not a provider billing
guarantee. Optional Groq recovery verifies and retains a maximum token cost for
a rejected tool call. Other unresolved usage stops dispatches. Only one local conversation
may use this process-global transport guard at a time; other threads are denied
while it is active. OpenHands persistence contains conversation content, whereas
Relay's budget ledger contains only numeric usage and identifiers.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from importlib.metadata import version as distribution_version
import json
import math
from pathlib import Path
import threading
from unittest.mock import patch


TESTED_OPENHANDS_VERSION = "1.62.0"
_TRANSPORT_LOCK = threading.Lock()
_BUDGET_OVERRIDES = frozenset({
    "max_tokens", "max_completion_tokens", "max_output_tokens", "timeout", "stream", "n",
    "num_retries", "max_retries", "retries", "input_cost_per_token", "output_cost_per_token",
    "model", "model_name", "api_base", "base_url", "fallbacks", "fallback_strategy", "model_list",
    "context_window_fallback_dict",
})


class OpenHandsIncompleteAttempt(RuntimeError):
    """The SDK did not finish or pause normally; final checks cannot hide it."""


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _validate_agent(agent, model):
    from openhands.sdk import Agent, LLM
    from .openhands import _validate_llm

    if distribution_version("openhands-sdk") != TESTED_OPENHANDS_VERSION:
        raise ValueError("This transport guard requires openhands-sdk==" + TESTED_OPENHANDS_VERSION)
    if not isinstance(agent, Agent) or getattr(agent, "_initialized", False):
        raise ValueError("agent_factory must return a fresh, uninitialized OpenHands Agent")
    if type(agent.llm) is not LLM:
        raise ValueError("Use a plain OpenHands LLM, without another router or provider wrapper")
    if agent.condenser is not None or agent.critic is not None or agent.mcp_config:
        raise ValueError("Cascade v1 requires local tools, condenser=None, critic=None and no MCP servers")
    if agent.tool_concurrency_limit != 1:
        raise ValueError("Cascade v1 executes tools sequentially")
    llm = agent.llm
    _validate_llm(llm, model)
    if llm.api_mode != "chat" or llm.stream:
        raise ValueError("Cascade v1 supports synchronous, non-streaming chat completions only")
    if type(llm.max_output_tokens) is not int or llm.max_output_tokens < 1:
        raise ValueError("Declare max_output_tokens explicitly")
    if type(llm.timeout) is not int or llm.timeout < 1:
        raise ValueError("Declare a positive per-request timeout")
    if _BUDGET_OVERRIDES.intersection(llm.litellm_extra_body):
        raise ValueError("litellm_extra_body cannot override cascade token, timeout, cost or routing limits")
    _positive(llm.input_cost_per_token, "input_cost_per_token")
    _positive(llm.output_cost_per_token, "output_cost_per_token")
    if llm.metrics.costs:
        raise ValueError("agent_factory must return a fresh LLM with no earlier calls")
    return llm


@contextmanager
def _audit_transport(context, llm, *, reservation_usd, max_request_bytes, recover_tool_use_failed=False):
    """Guard every synchronous SDK chat dispatch for this one attempt.

    A missing provider usage field is unknown, never zero. Errors preserve an
    unknown call before propagating; an SDK repair must re-enter begin_call.
    This controls SDK dispatches, not arbitrary network activity in host tools.
    """
    import openhands.sdk.llm.llm as sdk_llm

    if not _TRANSPORT_LOCK.acquire(blocking=False):
        raise RuntimeError("OpenHands cascade transport guards cannot overlap")
    owner = threading.get_ident()
    original = sdk_llm.litellm_completion

    def denied(*args, **kwargs):
        raise ValueError("Only synchronous chat dispatches are supported in this cascade")

    def dispatch(*args, **kwargs):
        if threading.get_ident() != owner:
            raise RuntimeError("Other threads cannot dispatch through an active cascade guard")
        if args:
            raise ValueError("SDK chat dispatches must use keyword arguments")
        actual_model = kwargs.get("model")
        provider = kwargs.get("custom_llm_provider")
        canonical_model = f"{provider}/{actual_model}" if provider else actual_model
        if llm.model not in (actual_model, canonical_model):
            raise ValueError("Dispatch model differs from the selected cascade model")
        if kwargs.get("stream") or any(kwargs.get(key, 0) != 0 for key in ("max_retries", "num_retries", "retries")):
            raise ValueError("Streaming and transport retries are outside the cascade contract")
        if type(kwargs.get("n", 1)) is not int or kwargs.get("n", 1) != 1:
            raise ValueError("Cascade accounting supports one completion per request")
        timeout = kwargs.get("timeout")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= llm.timeout:
            raise ValueError("The request must preserve its declared timeout")
        for key in ("extra_body", "litellm_extra_body"):
            body = kwargs.get(key)
            if isinstance(body, Mapping) and _BUDGET_OVERRIDES.intersection(body):
                raise ValueError("Extra request fields cannot override cascade budget limits")
        if any(kwargs.get(key) for key in ("fallbacks", "fallback_strategy", "model_list", "context_window_fallback_dict")):
            raise ValueError("Transport fallback overrides are outside the cascade contract")
        limits = [kwargs[key] for key in ("max_tokens", "max_completion_tokens") if key in kwargs]
        if not limits or any(type(value) is not int or not 0 < value <= llm.max_output_tokens for value in limits):
            raise ValueError("The request must preserve its declared output-token limit")
        request_bytes = len(json.dumps(
            {"messages": kwargs.get("messages"), "tools": kwargs.get("tools")},
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
        if request_bytes > max_request_bytes:
            raise ValueError("The request exceeds the declared context admission size")
        if recover_tool_use_failed:
            from .groq_recovery import GroqDispatch
            wire = GroqDispatch(llm, kwargs, max(limits))
            def begin_wire_call():
                return context.budget.begin_call(llm.model, reservation_usd, tariff_bound=wire.bound)
            try:
                with wire.observe(begin_wire_call):
                    response = original(**kwargs)
                if wire.http_status != 200:
                    raise OpenHandsIncompleteAttempt("The direct Groq dispatch did not return HTTP 200")
            except BaseException:
                if wire.call_id is not None:
                    context.budget.finish_call(wire.call_id, **wire.accounting("error"))
                elif not wire.local_denial:
                    context.budget.record_unobserved_transport(llm.model)
                raise
            context.budget.finish_call(wire.call_id, **wire.accounting("completed"))
            return response
        call_id = context.budget.begin_call(llm.model, reservation_usd)
        try:
            response = original(**kwargs)
        except BaseException:
            context.budget.finish_call(call_id, status="error", cost_usd=None)
            raise
        usage = _field(response, "usage")
        inp = _count(_field(usage, "prompt_tokens"))
        out = _count(_field(usage, "completion_tokens"))
        # Zero input usage on a nonempty chat is not credible accounting.
        cost = None
        if inp is not None and inp > 0 and out is not None:
            cost = inp * llm.input_cost_per_token + out * llm.output_cost_per_token
        context.budget.finish_call(call_id, status="completed", cost_usd=cost,
                                   cost_basis="tariff_estimate", input_tokens=inp, output_tokens=out)
        return response

    try:
        with patch.object(sdk_llm, "litellm_completion", dispatch), \
             patch.object(sdk_llm, "litellm_acompletion", denied), \
             patch.object(sdk_llm, "litellm_responses", denied), \
             patch.object(sdk_llm, "litellm_aresponses", denied):
            yield
    finally:
        _TRANSPORT_LOCK.release()


def make_openhands_attempt(
    agent_factory: Callable,
    prompt: str,
    *,
    reservation_usd_by_model: Mapping[str, float],
    tools_are_local: bool,
    max_iterations: int = 3,
    max_request_bytes: int = 18_000,
    recover_tool_use_failed: bool = False,
):
    """Return ``run_attempt(context)`` for ``cascade.run_cascade``.

    ``agent_factory(context)`` builds fresh LLM/Agent objects with tools bound to
    ``context.workspace``. The prompt is shared unchanged between model attempts;
    a fallback starts from the clean snapshot, not the cheap model's edits.
    Local host tools must not call other paid services or subagents: those calls
    cannot be accounted for here. ``tools_are_local=True`` is that declaration.

    The returned callback never calls the operational verifier or the final
    evaluator. Relay owns those boundaries. SDK messages/persistence remain under
    ``context.state_dir``; no exception message or prompt enters AttemptOutcome.

    ``recover_tool_use_failed=True`` pins Groq on-demand for GPT-OSS 20B/120B.
    It reserves at least one full context plus the request's maximum output.
    A verified HTTP 400 tool_use_failed can then admit a clean different-model
    branch within the remaining cap. The point cost stays unknown unless the
    provider returned valid usage on the actual response.
    """
    if not callable(agent_factory) or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Supply an Agent factory and a nonempty task prompt")
    if tools_are_local is not True:
        raise ValueError("Declare tools_are_local=True only for tools without external model/service costs")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if type(max_request_bytes) is not int or max_request_bytes < 1:
        raise ValueError("max_request_bytes must be a positive integer")
    if type(recover_tool_use_failed) is not bool:
        raise ValueError("recover_tool_use_failed must be boolean")
    if not isinstance(reservation_usd_by_model, Mapping) or not reservation_usd_by_model:
        raise ValueError("Supply conservative per-call reservations for the model catalog")
    reservations = {model: _positive(value, "reservation_usd")
                    for model, value in reservation_usd_by_model.items()}

    def run_attempt(context):
        from ..cascade import AttemptOutcome
        from openhands.sdk.conversation.impl.local_conversation import LocalConversation
        from openhands.sdk.conversation.state import ConversationExecutionStatus
        import openhands.sdk.llm.llm_profile_store as profile_store
        from .openhands_io import retry_openhands_writes

        conversation = None
        try:
            if context.model not in reservations:
                raise ValueError("The selected model lacks a declared call reservation")
            agent = agent_factory(context)
            llm = _validate_agent(agent, context.model)
            if reservations[context.model] < llm.max_output_tokens * llm.output_cost_per_token:
                raise ValueError("The call reservation must at least cover its maximum declared output cost")
            state_dir = Path(context.state_dir)
            state_dir.mkdir(parents=True, exist_ok=True)
            if any(state_dir.iterdir()):
                raise ValueError("Every cascade attempt requires an empty state directory")
            with _audit_transport(context, llm, reservation_usd=reservations[context.model],
                                  max_request_bytes=max_request_bytes, recover_tool_use_failed=recover_tool_use_failed), \
                 retry_openhands_writes(state_dir), \
                 patch.object(profile_store, "_DEFAULT_PROFILE_DIR", state_dir / "profiles"):
                conversation = LocalConversation(
                    agent=agent, workspace=context.workspace,
                    persistence_dir=state_dir / "conversation",
                    profile_store_dir=state_dir / "profiles",
                    max_iteration_per_run=max_iterations,
                    visualizer=None, plugins=[],
                )
                conversation.send_message(prompt)
                conversation.run()
                if conversation.state.execution_status not in {
                    ConversationExecutionStatus.FINISHED, ConversationExecutionStatus.PAUSED,
                }:
                    raise OpenHandsIncompleteAttempt("The SDK attempt did not finish or pause normally")
                conversation.close()
                conversation = None
            return AttemptOutcome(status="completed")
        except Exception as exc:
            return AttemptOutcome(status="error", error_type=type(exc).__name__)
        finally:
            if conversation is not None:
                conversation.close()

    return run_attempt
