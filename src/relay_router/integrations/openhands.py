"""Local OpenHands SDK 1.62.0 adapter with one Relay decision per task.

Use ``make_openhands_router(task, llms)`` as an existing Agent's ``llm``.
The harness still owns tools, execution, acceptance and auxiliary costs.
Only local, in-process conversations are supported: a live TaskSession cannot
be restored from an OpenHands serialized LLM profile or sent to a remote agent.
``max_calls`` counts logical SDK dispatches, not HTTP requests. Relay adds no
retries; SDK retry/fallback settings must be disabled. Provider-specific cache
repair inside the SDK can still make an additional transport request.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from typing import Any

try:
    from openhands.sdk.llm import LLM, LLMResponse, Message, RouterLLM
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError as exc:
    raise ImportError(
        "The OpenHands adapter requires Python >=3.12 and openhands-sdk==1.62.0. "
        "Install the OpenHands extra in a separate environment from the legacy api extra."
    ) from exc

from ..workflow import TaskSession

TESTED_OPENHANDS_VERSION = "1.62.0"
_ROUTE_OVERRIDES = frozenset({
    "model", "model_name", "api_base", "base_url", "custom_llm_provider",
    "max_retries", "num_retries", "retries", "fallbacks", "fallback_strategy",
    "context_window_fallback_dict", "model_list", "deployment_id",
})


def _validate_kwargs(kwargs: Mapping) -> None:
    if _ROUTE_OVERRIDES.intersection(kwargs):
        raise ValueError("Configure model, transport and retry settings on the OpenHands LLM, not per call")
    for name in ("extra_body", "litellm_extra_body"):
        body = kwargs.get(name)
        if isinstance(body, Mapping) and _ROUTE_OVERRIDES.intersection(body):
            raise ValueError("extra_body cannot override the Relay-selected model or retry settings")


def _value(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _count(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _charge(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _call_metrics(response: LLMResponse, llm: LLM, cost_offset: int) -> dict:
    """Read allowlisted metadata only; never serialize messages or raw responses.

    OpenHands response.metrics is cumulative and therefore deliberately unused.
    A zero in generic metrics/usage is not evidence of a free request. An explicit
    provider cost header, including zero, is the sole zero-charge exception.
    """
    raw = response.raw_response
    usage = _value(raw, "usage")
    input_tokens = _count(_value(usage, "prompt_tokens"))
    if input_tokens is None:
        input_tokens = _count(_value(usage, "input_tokens"))
    output_tokens = _count(_value(usage, "completion_tokens"))
    if output_tokens is None:
        output_tokens = _count(_value(usage, "output_tokens"))
    details = _value(usage, "prompt_tokens_details")
    if details is None:
        details = _value(usage, "input_tokens_details")
    cached = _count(_value(details, "cached_tokens"))
    if cached is not None and input_tokens is not None and cached > input_tokens:
        # Some gateways use disjoint input/cache buckets. Do not manufacture a
        # total or reject a valid generation based on incompatible metadata.
        cached = None
    cost, basis = None, "unknown"
    hidden = _value(raw, "_hidden_params")
    headers = _value(hidden, "additional_headers")
    reported = _charge(_value(headers, "llm_provider-x-litellm-response-cost"))
    if reported is not None:
        cost, basis = reported, "provider_reported"
    else:
        reported = _charge(_value(usage, "cost"))
        if reported is not None and reported > 0:
            cost, basis = reported, "provider_reported"
        else:
            # Public Metrics.costs contains only successful price calculations.
            # Refuse ambiguous zero, multiple, or missing records for this call.
            added = llm.metrics.costs[cost_offset:]
            if len(added) == 1:
                estimated = _charge(added[0].cost)
                if estimated is not None and estimated > 0:
                    cost, basis = estimated, "tariff_estimate"
    return dict(cost_usd=cost, cost_basis=basis, input_tokens=input_tokens,
                output_tokens=output_tokens, cached_input_tokens=cached)


class RelayOpenHandsRouter(RouterLLM):
    """A real RouterLLM, bound to a live TaskSession; create via the factory.

    No prompts, responses, credentials or exception messages enter the Relay
    ledger. Existing SDK logging/tracing remains controlled by the host.
    The router is sequential per task; concurrent callers need distinct tasks.
    """

    relay_session: Any = Field(exclude=True, repr=False)
    _dispatch_lock: Any = PrivateAttr(default_factory=threading.Lock)

    def __getattr__(self, name: str) -> Any:
        # RouterLLM.__getattr__ skips Pydantic's PrivateAttr implementation.
        # Preserve it, instead of delegating our private task lock to the LLM.
        return BaseModel.__getattr__(self, name)

    def select_llm(self, messages: list[Message]) -> str:
        return self.relay_session.model

    def _selected(self) -> LLM:
        if self.model != self.relay_session.model:
            raise ValueError("A Relay task's model cannot change during its conversation")
        selected = self.llms_for_routing[self.model]
        _validate_llm(selected, self.model)
        return selected

    @property
    def metrics(self):
        return self.llms_for_routing[self.model].metrics

    @property
    def telemetry(self):
        return self.llms_for_routing[self.model].telemetry

    def reset_metrics(self) -> None:
        self.llms_for_routing[self.model].reset_metrics()

    def restore_metrics(self, metrics) -> None:
        self.llms_for_routing[self.model].restore_metrics(metrics)

    def model_copy(self, *, update=None, deep=False):
        # OpenHands uses model_copy(usage_id=...) for condenser profiles. Keep
        # the same live task, with independent SDK accounting for the new role.
        if update and set(update) - set(LLM.model_fields):
            raise ValueError("Create a new Relay router to change its session or candidate set")
        selected = self._selected().model_copy(update=update, deep=deep)
        selected.reset_metrics()
        return make_openhands_router(self.relay_session, {self.model: selected})

    def _start(self):
        selected = self._selected()
        if not self._dispatch_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent calls require separate Relay task sessions")
        try:
            call_id = self.relay_session.begin_call(model=self.model)
            return selected, call_id, len(selected.metrics.costs), time.perf_counter()
        except BaseException:
            self._dispatch_lock.release()
            raise

    def _complete(self, response, selected, call_id, cost_offset, started):
        self.relay_session.record_call(
            call_id, model=self.model, status="completed",
            elapsed_seconds=time.perf_counter() - started,
            **_call_metrics(response, selected, cost_offset),
        )
        return response

    def _failed(self, call_id, started):
        self.relay_session.record_call(
            call_id, model=self.model, status="error", cost_usd=None,
            cost_basis="unknown", elapsed_seconds=time.perf_counter() - started,
        )

    def _dispatch(self, method, **kwargs):
        _validate_kwargs(kwargs)
        selected, call_id, offset, started = self._start()
        try:
            try:
                response = getattr(selected, method)(**kwargs)
            except BaseException:
                self._failed(call_id, started)
                raise
            return self._complete(response, selected, call_id, offset, started)
        finally:
            self._dispatch_lock.release()

    async def _adispatch(self, method, **kwargs):
        _validate_kwargs(kwargs)
        selected, call_id, offset, started = self._start()
        try:
            try:
                response = await getattr(selected, method)(**kwargs)
            except BaseException:
                self._failed(call_id, started)
                raise
            return self._complete(response, selected, call_id, offset, started)
        finally:
            self._dispatch_lock.release()

    def completion(self, messages, tools=None, add_security_risk_prediction=False,
                   on_token=None, call_context=None, **kwargs):
        return self._dispatch(
            "completion", messages=messages, tools=tools,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token, call_context=call_context, **kwargs,
        )

    async def acompletion(self, messages, tools=None, add_security_risk_prediction=False,
                          on_token=None, call_context=None, **kwargs):
        return await self._adispatch(
            "acompletion", messages=messages, tools=tools,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token, call_context=call_context, **kwargs,
        )

    def responses(self, messages, tools=None, include=None, store=None,
                  add_security_risk_prediction=False, on_token=None, call_context=None, **kwargs):
        return self._dispatch(
            "responses", messages=messages, tools=tools, include=include, store=store,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token, call_context=call_context, **kwargs,
        )

    async def aresponses(self, messages, tools=None, include=None, store=None,
                         add_security_risk_prediction=False, on_token=None, call_context=None, **kwargs):
        return await self._adispatch(
            "aresponses", messages=messages, tools=tools, include=include, store=store,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token, call_context=call_context, **kwargs,
        )


def _validate_llm(llm: LLM, model: str) -> None:
    if not isinstance(llm, LLM) or llm.model != model:
        raise ValueError("Candidate keys must exactly match each OpenHands LLM.model")
    if llm.num_retries != 0 or llm.fallback_strategy is not None:
        raise ValueError("Relay requires num_retries=0 and fallback_strategy=None on OpenHands LLMs")
    if getattr(llm, "_api_key_refresh_hook", None) is not None:
        raise ValueError("Disable the SDK API-key refresh retry hook for Relay accounting")
    _validate_kwargs({"extra_body": llm.litellm_extra_body})


def make_openhands_router(session: TaskSession, llms: Mapping[str, LLM]) -> RelayOpenHandsRouter:
    """Bind caller-configured models to a task, preserving its selected model.

    Use fresh LLM instances per task. A model can appear in the policy only if
    the harness already established that it supports the required tools/context.
    This factory neither provisions credentials nor starts a conversation.
    """
    if session.model not in llms:
        raise ValueError("The Relay-selected model has no configured OpenHands LLM")
    for name, llm in llms.items():
        _validate_llm(llm, name)
    selected = llms[session.model]
    config = {name: getattr(selected, name) for name in LLM.model_fields}
    # Copy the real selected configuration, not a fictitious 'relay' model,
    # so inherited OpenHands capability/context checks describe the candidate.
    return RelayOpenHandsRouter(
        **config, router_name="relay", relay_session=session,
        llms_for_routing={session.model: selected}, active_llm=selected,
    )


def attach_openhands_agent(agent, relay, *, workflow, models, routing_text=None, task_id=None, max_calls=20):
    """Attach a frozen Relay policy to an existing local Agent in one call.

    Return (task, agent_copy). The caller keeps its tools, Conversation and
    verifier, then closes task.finish with acceptance and full cost accounting.
    """
    from openhands.sdk import Agent
    if not isinstance(agent, Agent) or not isinstance(models, Mapping) or not models:
        raise ValueError("Provide an existing OpenHands Agent and configured model mapping")
    for name, llm in models.items():
        _validate_llm(llm, name)
    task = relay.task(workflow, task_id=task_id, eligible_models=list(models),
                      routing_text=routing_text, max_calls=max_calls)
    try:
        routed = agent.model_copy(update={"llm": make_openhands_router(task, models)})
    except BaseException:
        task.finish(accepted=None, cost_complete=False)
        raise
    return task, routed
