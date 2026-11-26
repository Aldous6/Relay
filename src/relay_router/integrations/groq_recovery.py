"""Bound a single Groq on-demand text generation and observe the raw HTTP reply.

The bound prices one admitted model context at the uncached input rate plus
the entire requested output at its own rate. It is a conservative token-tariff
scenario, independent of the host's heuristic reservation and not an invoice.
See https://console.groq.com/docs/api-reference#chat-create and the model
card URL written to the ledger. No provider-side tools are covered.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from importlib.metadata import version
import json
import re
import threading
from unittest.mock import patch

from ..cascade import TariffBound


BASE_URL = "https://api.groq.com/openai/v1"
_MODEL_RATES = {
    "groq/openai/gpt-oss-20b": (.075 / 1_000_000, .30 / 1_000_000),
    "groq/openai/gpt-oss-120b": (.15 / 1_000_000, .60 / 1_000_000),
}
_CHAT_FIELDS = frozenset({
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream", "n",
    "max_completion_tokens", "max_tokens", "service_tier", "reasoning_effort",
    "temperature", "top_p", "stop", "seed", "frequency_penalty", "presence_penalty",
    "response_format", "user", "logprobs", "top_logprobs",
})


class GroqDispatch:
    def __init__(self, llm, kwargs, output_limit):
        if version("httpx") != "0.28.1" or version("litellm") != "1.105.7":
            raise ValueError("Groq recovery requires httpx==0.28.1 and litellm==1.105.7")
        rates = _MODEL_RATES.get(llm.model)
        if rates is None or rates != (llm.input_cost_per_token, llm.output_cost_per_token):
            raise ValueError("Groq recovery requires the documented GPT-OSS on-demand token tariffs")
        if kwargs.get("api_base") not in (None, BASE_URL, BASE_URL + "/"):
            raise ValueError("Groq recovery requires the direct Groq endpoint")
        if not 0 < output_limit <= 65_536:
            raise ValueError("Output limit is outside the GPT-OSS model contract")
        self.model = llm.model.removeprefix("groq/")
        self.bound = TariffBound(131_072, output_limit, *rates,
                                 "https://console.groq.com/docs/model/" + self.model)
        # Pin both the destination and tariff. Validate the final serialized
        # request too, after all OpenHands and LiteLLM transformations.
        for value in (kwargs, kwargs.get("extra_body") or {}):
            if value.get("service_tier") not in (None, "on_demand"):
                raise ValueError("Groq recovery supports on_demand only")
        kwargs["api_base"] = BASE_URL
        # OpenHands may supply OpenAI cache hints from its model registry.
        # Groq handles prefix caching itself and has no such chat parameters.
        kwargs.pop("prompt_cache_key", None)
        kwargs.pop("prompt_cache_retention", None)
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "service_tier": "on_demand"}
        self.sent = False
        self.local_denial = False
        self.call_id = None
        self.http_status = self.provider_error_code = self.provider_request_id = None
        self.input_tokens = self.output_tokens = None

    def _validate_request(self, request):
        if request.method != "POST" or str(request.url) != BASE_URL + "/chat/completions":
            raise ValueError("The final HTTP request is outside the Groq chat contract")
        body = json.loads(request.content)
        if (not isinstance(body, dict) or set(body) - _CHAT_FIELDS or body.get("model") != self.model
                or body.get("service_tier") != "on_demand" or body.get("stream", False) is not False
                or type(body.get("n", 1)) is not int or body.get("n", 1) != 1):
            raise ValueError("Unsupported HTTP body for a bounded, single Groq generation")
        limits = [body[name] for name in ("max_completion_tokens", "max_tokens") if name in body]
        if not limits or any(type(n) is not int or not 0 < n <= self.bound.max_output_tokens for n in limits):
            raise ValueError("Missing or invalid token limit on the actual HTTP request")
        if any(not isinstance(tool, dict) or tool.get("type") != "function" for tool in body.get("tools") or ()):
            raise ValueError("Bounded recovery permits only host-executed function tools")

    def _observe_response(self, response):
        self.http_status = response.status_code
        id_value = response.headers.get("x-request-id")
        try:
            payload = json.loads(response.read())
        except (ValueError, UnicodeError):
            payload = {}
        if not isinstance(payload, dict):
            return
        x_groq = payload.get("x_groq")
        if not id_value and isinstance(x_groq, dict):
            id_value = x_groq.get("id")
        if isinstance(id_value, str) and re.fullmatch(r"req_[A-Za-z0-9_-]{1,96}", id_value):
            self.provider_request_id = id_value
        if payload.get("model", self.model) != self.model or payload.get("service_tier") not in (None, "default", "on_demand"):
            raise ValueError("Groq replied with a different model or tariff tier")
        error = payload.get("error")
        if self.http_status == 400 and isinstance(error, dict) and error.get("code") == "tool_use_failed":
            self.provider_error_code = "tool_use_failed"
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            inp, out, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            if (type(inp) is int and inp > 0 and type(out) is int and out >= 0
                    and (total is None or type(total) is int and inp + out == total)):
                self.input_tokens, self.output_tokens = inp, out
            elif ((type(inp) is int and inp > self.bound.max_input_tokens)
                  or (type(out) is int and out > self.bound.max_output_tokens)
                  or (type(total) is int and total > self.bound.max_input_tokens)):
                # Internally inconsistent usage also contradicts the contract.
                # It can neither price the call nor authorize a bounded fallback.
                self.provider_error_code = None

    @contextmanager
    def observe(self, begin_call):
        # HTTPX 0.28 dispatches both redirects and auth retries here, once per
        # wire request. A lower-level TCP connection retry cannot resubmit an
        # already delivered HTTP request. The outer SDK guard serializes callers.
        import httpx
        send = httpx.Client._send_single_request
        owner = threading.get_ident()

        def send_once(client, request):
            if threading.get_ident() != owner:
                raise RuntimeError("Other threads cannot send HTTP during a bounded Groq dispatch")
            if self.sent:
                raise ValueError("A Groq dispatch cannot replay or redirect its HTTP request")
            try:
                self._validate_request(request)
                self.call_id = begin_call()
            except Exception:
                self.local_denial = True
                raise
            self.sent = True
            response = send(client, request)
            try:
                self._observe_response(response)
            except BaseException:
                response.close()
                raise
            return response

        with patch.object(httpx.Client, "_send_single_request", send_once):
            yield

    def accounting(self, status):
        cost = None
        if self.input_tokens is not None and self.output_tokens is not None:
            cost = (self.input_tokens * self.bound.input_usd_per_token
                    + self.output_tokens * self.bound.output_usd_per_token)
        return dict(status=status, cost_usd=cost, input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens, provider_error_code=self.provider_error_code,
                    http_status=self.http_status, provider_request_id=self.provider_request_id)
