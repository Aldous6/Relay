"""Native multi-turn BFCL transport, with one completion per harness attempt.

No provider policy, action repair, context trimming, or implicit retries. The
native batch is preserved for the harness's validated ordered tool dispatch.
"""
from copy import deepcopy
import json
import os
import time

from relay_router.errors import (BackendActionError, BackendContextError, BackendHTTPError,
                                 BackendIdentityError, BackendResponseError)
from .agent_litellm import (LiteLLMEpisodeGenerator, _default_completion, _get, _name,
                            _provider_error, _usage, _RESERVE)
from .agent_pacing import PacedEpisodeGenerator, _known_total
from .bfcl_harness import BFCLRequest, BFCLReply


class BFCLLiteLLMGenerator(LiteLLMEpisodeGenerator):
    """Reuse the validated catalog/error contract without converting BFCL turns."""

    def __init__(self, models_metadata, options, *, completion=None, input_estimator=None):
        options = deepcopy(options)
        self._reasoning_effort = options.pop("reasoning_effort", None)
        if self._reasoning_effort is not None and self._reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("Unsupported declared reasoning effort.")
        if options.get("action_format", "native_tools") != "native_tools":
            raise ValueError("BFCL requires native tools.")
        options["action_format"] = "native_tools"
        super().__init__(models_metadata, options, completion=completion)
        self._input_estimator = input_estimator
        self._estimator_manifest = None
        if input_estimator is not None:
            if (not callable(getattr(input_estimator, "estimate", None))
                    or not callable(getattr(input_estimator, "manifest", None))
                    or not set(models_metadata) <= set(getattr(input_estimator, "supported_models", ()))):
                raise ValueError("The input estimator must explicitly support every selected model.")
            self._estimator_manifest = deepcopy(input_estimator.manifest())

    def request_payload(self, request):
        if not isinstance(request, BFCLRequest):
            raise ValueError("BFCL transport requires BFCLRequest.")
        request = request.snapshot()
        payload = {"messages": list(request.messages)}
        if request.tools:
            payload.update(tools=list(request.tools), tool_choice="auto", parallel_tool_calls=True)
        return payload

    def request_input_bytes(self, request):
        return len(json.dumps(self.request_payload(request), ensure_ascii=False,
                              allow_nan=False, separators=(",", ":")).encode("utf-8"))

    def request_input_estimate(self, request):
        if self._input_estimator is None:
            return {"input_budget_units": self.request_input_bytes(request), "method": "utf8_bytes_bound",
                    "server_template_verified": False, "actual_provider_tokens": False}
        if self._input_estimator.manifest() != self._estimator_manifest:
            raise ValueError("The frozen input estimator changed.")
        detail = self._input_estimator.estimate(request)
        count = detail.get("estimated_input_tokens")
        if type(count) is not int or count < 0:
            raise ValueError("A finite nonnegative integer input estimate is required.")
        return {"input_budget_units": count, "method": "gptoss_local_token_estimate",
                "details": detail, "server_template_verified": False, "actual_provider_tokens": False}

    def manifest(self):
        result = super().manifest()
        result.update(kind="litellm_bfcl_native_generator", input_format="bfcl_native_multiturn_local_v1",
                      native_batches_preserved=True, text_final_scope="current user turn",
                      parallel_tool_calls=True, action_repair=False)
        result["options"]["reasoning_effort"] = self._reasoning_effort
        result["quota_input_estimator"] = deepcopy(self._estimator_manifest)
        return result

    def __call__(self, model, request):
        if type(model) is not str or model not in self._models:
            raise ValueError("The model is outside the frozen catalog.")
        payload = self.request_payload(request)
        incoming_bytes = self.request_input_bytes(request)
        budget = incoming_bytes + self._options["max_output_tokens"] + _RESERVE
        if budget > self._models[model]["context_window"]:
            raise BackendContextError("BFCL request exceeds its conservative context budget; no truncation or generation.")
        key = os.environ.get(self._models[model]["api_key_env"])
        if not _name(key):
            raise BackendHTTPError("A valid credential is required in the declared environment variable.", status_code=401)
        completion = self._completion if self._completion is not None else _default_completion()
        arguments = {"model": model, **payload, "api_key": key,
            "temperature": self._options["temperature"], "max_tokens": self._options["max_output_tokens"],
            "timeout": self._options["timeout"], "stream": False, "num_retries": 0, "max_retries": 0,
            "fallbacks": None, "context_window_fallback_dict": {}, "caching": False, "drop_params": False}
        if "api_base" in self._models[model]:
            arguments["api_base"] = self._models[model]["api_base"]
        if self._reasoning_effort is not None:
            arguments["reasoning_effort"] = self._reasoning_effort
        started = time.perf_counter()
        try:
            response = completion(**arguments)
        except Exception as error:
            raise _provider_error(error) from None
        elapsed = max(0., time.perf_counter() - started)
        usage = _usage(response)
        returned = _get(response, "model")
        if returned not in self._models[model]["returned_model_ids"]:
            raise BackendIdentityError("The provider returned an undeclared model identity.", usage=usage)
        choices = _get(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise BackendResponseError("Exactly one completion is required.", usage=usage)
        message = _get(choices[0], "message")
        if _get(message, "role") != "assistant":
            raise BackendResponseError("The provider did not return an assistant message.", usage=usage)
        content, native = _get(message, "content"), _get(message, "tool_calls")
        if content is not None and type(content) is not str:
            raise BackendResponseError("BFCL completion content must be text or null.", usage=usage)
        if native is not None and not isinstance(native, (list, tuple)):
            raise BackendResponseError("The native tool-call batch is not a list.", usage=usage)
        finish = _get(choices[0], "finish_reason")
        calls = tuple({"id": _get(call, "id"), "type": _get(call, "type"), "function": {
            "name": _get(_get(call, "function"), "name"),
            "arguments": _get(_get(call, "function"), "arguments")}} for call in (native or ()))
        if (any(type(call["id"]) is not str or not call["id"] or call["type"] != "function"
                or type(call["function"]["name"]) is not str
                or type(call["function"]["arguments"]) is not str for call in calls)
                or len({call["id"] for call in calls}) != len(calls)):
            raise BackendResponseError("The provider violated the native function-call envelope.", usage=usage)
        # Do not parse or repair JSON here; malformed arguments are measured
        # model output, and the harness validates the entire batch before effects.
        try:
            json.dumps(calls, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise BackendResponseError("Tool-call fields violate the JSON response contract.", usage=usage) from None
        usage["provider_metadata"] = {"provider": model.split("/", 1)[0], "requested_model": model,
            "returned_model": returned, "finish_reason": finish if type(finish) is str else None,
            "request_seconds": elapsed, "input_utf8_bytes": incoming_bytes,
            "conservative_context_budget": budget, "input_truncated_by_client": False,
            "provider_context_preservation_verified": False, "action_format": "native_tools",
            "native_tool_calls": deepcopy(list(calls))}
        if finish == "length" and not calls and not (content or "").strip():
            raise BackendActionError("The output budget ended before an action was supplied.", usage=usage)
        return BFCLReply(content, calls, usage, None)


class BFCLPacedGenerator(PacedEpisodeGenerator):
    """Explicit byte-bound or token-estimate accounting for BFCLReply batches."""

    def manifest(self):
        result = super().manifest()
        pacing = result["harness_pacing"]
        pacing["request_budget_limit"] = pacing.pop("request_budget_utf8_bytes_plus_output_and_reserve")
        pacing["input_measurement"] = ("local token estimate; not provider tokenization" if
            self.generate.manifest().get("quota_input_estimator") is not None else "UTF-8 byte bound")
        pacing["scope"] = "local quota reservation with explicit input method; no coordination with other account clients or billing guarantee"
        return result

    def __call__(self, model, request):
        estimate = self.generate.request_input_estimate(request)
        budget = estimate["input_budget_units"] + self.output + _RESERVE
        if budget > self.budget or (self.tokens_per_minute is not None and budget > self.tokens_per_minute):
            raise BackendContextError("BFCL request exceeds the declared quota accounting limit; no truncation or call.")
        wait_started = self.clock()
        remaining = self._wait_seconds(model, budget, wait_started)
        waited = remaining > 0
        last_notice = wait_started
        if self.show_progress and waited:
            print(f"[cuota BFCL] Esperando {remaining:.0f} s.", flush=True)
        while remaining > 0:
            self.sleep(min(remaining, 10.))
            remaining = self._wait_seconds(model, budget, self.clock())
            if self.show_progress and remaining > 0 and self.clock() - last_notice >= 20:
                print(f"[cuota BFCL] Faltan {remaining:.0f} s.", flush=True)
                last_notice = self.clock()
        self._last_started[model] = self.clock()
        waiting = self.clock() - wait_started if waited else 0.
        reservation = None
        if self.tokens_per_minute is not None:
            reservation = [self.clock(), budget]
            self._reservations.setdefault(model, []).append(reservation)
        if self.show_progress:
            print(f"[API BFCL] Enviando solicitud a {model}.", flush=True)
        try:
            reply = self.generate(model, request)
            if reservation is not None and isinstance(reply, BFCLReply):
                total = _known_total(reply.usage)
                if total is not None:
                    reservation[1] = total
        finally:
            if reservation is not None:
                reservation[0] = self.clock()
        if not isinstance(reply, BFCLReply):
            return reply  # Harness classifies a bad callback contract; no repair.
        usage = deepcopy(reply.usage)
        usage.setdefault("provider_metadata", {})["quota_wait_seconds"] = waiting
        usage["provider_metadata"]["quota_input_estimate"] = deepcopy(estimate)
        return BFCLReply(reply.content, reply.tool_calls, usage, reply.cost_usd)
