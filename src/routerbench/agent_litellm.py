"""One explicit LiteLLM completion per harness attempt, without provider routing.

Construction, manifests and imports of this module perform no provider I/O.
LiteLLM is optional and loaded only when the callback is actually called.
Retries, escalation and monetary estimates belong to the harness, not this
adapter. Official contracts inspected: https://docs.litellm.ai/docs/completion/input,
https://docs.litellm.ai/docs/completion/reliable_completions and
https://docs.litellm.ai/docs/exception_mapping.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.metadata
import json
import math
import os
import re
import time
from urllib.parse import urlsplit

from relay_router.agent import AgentRequest, ModelReply
from relay_router.errors import (BackendActionError, BackendContextError, BackendHTTPError,
                                 BackendIdentityError, BackendRateLimitError,
                                 BackendResponseError, BackendTimeoutError,
                                 BackendTransportError, LiveBackendError)
from .agent_context import render_messages


_RESERVE = 512
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_AUTO_MODELS = {"openrouter/auto", "openrouter/auto-beta", "openrouter/auto-code"}


def _name(value):
    return (type(value) is str and bool(value) and value == value.strip()
            and not any(character.isspace() for character in value))


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _native_action(message):
    """Translate one function call without repairing its arguments or executing it.

    Embed raw argument JSON so parse_action still rejects duplicate keys, NaN,
    unknown tools and malformed JSON. Multiple calls never execute partially.
    """
    calls = _get(message, "tool_calls")
    if calls is None or (isinstance(calls, (list, tuple)) and not calls):
        content = _get(message, "content")
        return json.dumps({"type": "final", "answer": content}, ensure_ascii=False) if type(content) is str and content.strip() else ""
    if not isinstance(calls, (list, tuple)) or len(calls) != 1:
        return ""
    call = calls[0]
    function = _get(call, "function")
    name, arguments = _get(function, "name"), _get(function, "arguments")
    if _get(call, "type") != "function" or type(name) is not str or type(arguments) is not str:
        return ""
    return '{"type":"tool","name":' + json.dumps(name) + ',"arguments":' + arguments + '}'


def _usage(response):
    usage = _get(response, "usage")
    incoming = _count(_get(usage, "prompt_tokens"))
    outgoing = _count(_get(usage, "completion_tokens"))
    total = _count(_get(usage, "total_tokens"))
    details = _get(usage, "prompt_tokens_details")
    cached = _count(_get(details, "cached_tokens"))
    # Invalid counters remain unknown; never subtract a cache count exceeding input.
    if incoming is not None and cached is not None and cached > incoming:
        cached = None
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return {"input_tokens": incoming, "output_tokens": outgoing,
            "total_tokens": total, "cached_input_tokens": cached}


def _retry_after(error):
    headers = _get(_get(error, "response"), "headers")
    if headers is None:
        headers = _get(error, "headers")
    try:
        value = headers.get("retry-after") if headers is not None else None
        seconds = float(value)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _provider_rejection(chain):
    """Extract allowlisted error identifiers, never free text or failed output."""
    import json
    allowed = {"json_validate_failed", "tool_use_failed", "output_parse_failed", "invalid_request_error", "invalid_api_key",
               "model_not_found", "context_length_exceeded", "rate_limit_exceeded", "insufficient_quota",
               "unsupported_parameter", "invalid_parameter", "invalid_value"}
    facts = {}
    for item in chain:
        body = getattr(item, "body", None)
        if body is None:
            response = getattr(item, "response", None)
            if response is not None:
                try:
                    body = response.json()
                except Exception:
                    pass
        if type(body) is str and len(body) <= 20000:
            try:
                body = json.loads(body)
            except (ValueError, TypeError):
                continue
        if type(body) is not dict:
            continue
        body = body.get("error", body)
        if type(body) is not dict:
            continue
        for key in ("code", "type"):
            value = body.get(key)
            if type(value) is str and value in allowed:
                facts["provider_error_" + key] = value
    return facts


def _provider_error(error):
    """Map safe structural facts only; never inspect exception messages/bodies."""
    if isinstance(error, LiveBackendError):
        return error
    # OpenAILikeChatHandler can wrap local transport failures as status 500.
    # Inspect exception TYPES/codes, never messages, URLs, headers or bodies.
    chain, seen, current = [], set(), error
    while isinstance(current, BaseException) and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    chain_names = {cls.__name__ for item in chain for cls in type(item).__mro__}
    actual_http = next((item for item in chain if type(item).__name__ == "HTTPStatusError"
                        and type(_get(_get(item, "response"), "status_code")) is int), None)
    diagnostic = {
        "exception_types": [type(item).__name__ for item in chain],
        "os_error_codes": sorted({code for item in chain for code in
                                  (getattr(item, "errno", None), getattr(item, "winerror", None))
                                  if type(code) is int}),
        "status_source": "http_response" if actual_http else "sdk_reported_unverified"}
    diagnostic.update(_provider_rejection(chain))
    missing = [getattr(item, "name", None) for item in chain if isinstance(item, AttributeError)]
    if "service_tier" in missing:
        diagnostic["missing_sdk_attribute"] = "service_tier"
    if actual_http is None:
        if any(isinstance(item, (AttributeError, TypeError, ImportError)) for item in chain):
            diagnostic["status_source"] = "sdk_local_exception"
            return BackendResponseError("El SDK falló localmente al procesar la solicitud o respuesta.", diagnostic=diagnostic)
        transport = chain_names & {"APIConnectionError", "ConnectError", "NetworkError", "RemoteProtocolError",
                                   "ProxyError", "SSLError", "SSLCertVerificationError", "gaierror", "ReadError", "WriteError"}
        timeout = chain_names & {"Timeout", "APITimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout"}
        if timeout or any(isinstance(item, TimeoutError) for item in chain):
            diagnostic["status_source"] = "transport_exception"
            return BackendTimeoutError("La conexión no completó la solicitud a tiempo.", diagnostic=diagnostic)
        if transport or any(isinstance(item, (ConnectionError, OSError)) for item in chain):
            diagnostic["status_source"] = "transport_exception"
            return BackendTransportError("Falló el transporte antes de recibir una respuesta HTTP verificable.", diagnostic=diagnostic)
    status = _get(error, "status_code")
    if actual_http is not None:
        status = actual_http.response.status_code
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    names = {cls.__name__ for cls in type(error).__mro__}
    common = {"status_code": status, "retry_after_seconds": _retry_after(error), "diagnostic": diagnostic}
    if status == 400 and diagnostic.get("provider_error_code") in {"tool_use_failed", "json_validate_failed", "output_parse_failed"}:
        return BackendActionError("El proveedor rechazó la acción generada por el modelo.", **common)
    if "ContextWindowExceededError" in names:
        return BackendContextError("El proveedor rechazó el presupuesto de contexto.", **common)
    if status == 429 or "RateLimitError" in names:
        return BackendRateLimitError("El proveedor alcanzó su cuota temporal.", **common)
    if status == 408 or names & {"Timeout", "APITimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout"} or isinstance(error, TimeoutError):
        return BackendTimeoutError("El proveedor no completó la solicitud a tiempo.", **common)
    if status is not None:
        return BackendHTTPError("El proveedor rechazó la solicitud HTTP.", **common)
    if names & {"APIConnectionError", "ConnectError", "NetworkError", "RemoteProtocolError"} or isinstance(error, (ConnectionError, OSError)):
        return BackendTransportError("No se pudo completar el transporte al proveedor.", **common)
    return BackendResponseError("El proveedor o su SDK incumplió el contrato de respuesta.", **common)


def _redacted_provider_reason(error, credential):
    """Diagnostic-only bounded error.message; exclude full bodies and output."""
    import json
    chain, seen, current = [], set(), error
    while isinstance(current, BaseException) and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    for item in reversed(chain):
        candidates = [getattr(item, 'body', None)]
        response = getattr(item, 'response', None)
        if response is not None:
            try:
                candidates.append(response.json())
            except Exception:
                pass
        for body in candidates:
            if type(body) is str and len(body) <= 100000:
                try:
                    body = json.loads(body[body.find('{'):])
                except (ValueError, TypeError):
                    continue
            if type(body) is not dict:
                continue
            detail = body.get('error', body)
            message = detail.get('message') if type(detail) is dict else None
            if type(message) is not str:
                continue
            if credential:
                message = message.replace(credential, '[REDACTED]')
            message = re.sub(r'(?:gsk_|sk-)[A-Za-z0-9_-]{6,}', '[REDACTED]', message)
            message = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', message)
            return ' '.join(message.split())[:1000]
    return None


def _default_completion():
    """Load only an explicitly invoked SDK with telemetry and remote price fetch off."""
    os.environ["LITELLM_TELEMETRY"] = "False"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        module = importlib.import_module("litellm")
    except ImportError:
        raise RuntimeError("Instala la dependencia opcional de proveedores antes de ejecutar APIs.") from None
    # A shared process with configured hooks/aliases would change the experiment.
    # Refuse it rather than silently editing another client's configuration.
    # In 1.109.1, explicit num_retries=0 and fallbacks=None still fall back to
    # positive/nonempty global defaults through `or`. drop_params=False also
    # cannot override the global True. Refuse these settings before completion.
    for name in ("callbacks", "input_callback", "success_callback", "failure_callback",
                 "_async_success_callback", "_async_failure_callback", "model_alias_map",
                 "fallbacks", "model_fallbacks", "context_window_fallbacks",
                 "content_policy_fallbacks", "num_retries", "drop_params", "api_base"):
        if getattr(module, name, None):
            raise RuntimeError("LiteLLM tiene hooks, aliases, reintentos, fallbacks, endpoint o descarte de parámetros globales activos; usa un proceso aislado.")
    module.telemetry = False
    module.set_verbose = False
    module.suppress_debug_info = True
    return module.completion


class LiteLLMEpisodeGenerator:
    """Provider translator with a frozen N-model catalog and no retry loop.

    Catalog keys are exact LiteLLM IDs. Metadata declares ``returned_model_ids``,
    ``context_window`` and ``api_key_env``; optional ``api_base`` is an explicitly
    chosen HTTPS endpoint. Options are ``max_output_tokens``, ``temperature`` (0)
    and ``timeout``; optional ``action_format`` selects json_object (default)
    or native_tools. The injected completion is for offline contract tests.
    """

    def __init__(self, models_metadata, options, *, completion=None, capture_error_reason=False):
        if type(models_metadata) is not dict or not models_metadata:
            raise ValueError("Declara un catálogo LiteLLM explícito.")
        for model, metadata in models_metadata.items():
            if (not _name(model) or "/" not in model or model.lower() in _AUTO_MODELS
                    or type(metadata) is not dict
                    or not {"returned_model_ids", "context_window", "api_key_env"} <= set(metadata)
                    or set(metadata) - {"returned_model_ids", "context_window", "api_key_env", "api_base"}):
                raise ValueError("Cada modelo requiere identidad concreta y metadatos explícitos.")
            identities = metadata["returned_model_ids"]
            if (type(identities) is not list or not identities or not all(_name(item) for item in identities)
                    or len(identities) != len(set(identities))
                    or type(metadata["context_window"]) is not int or metadata["context_window"] < 1
                    or type(metadata["api_key_env"]) is not str or not _ENV_NAME.fullmatch(metadata["api_key_env"])):
                raise ValueError("Identidades, contexto o nombre de variable de credencial inválidos.")
            if "api_base" in metadata:
                value = metadata["api_base"]
                if not _name(value):
                    raise ValueError("api_base requiere una URL HTTPS explícita.")
                parsed = urlsplit(value)
                if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                        or parsed.password is not None or parsed.query or parsed.fragment):
                    raise ValueError("api_base requiere HTTPS sin credenciales, consulta ni fragmento.")
        required = {"max_output_tokens", "temperature", "timeout"}
        if (type(options) is not dict or not required <= set(options)
                or set(options) - required - {"action_format"}
                or options.get("action_format", "json_object") not in ("json_object", "native_tools")):
            raise ValueError("Declara max_output_tokens, temperature, timeout y opcionalmente action_format.")
        if (type(options["max_output_tokens"]) is not int or options["max_output_tokens"] < 1
                or type(options["temperature"]) not in (int, float) or options["temperature"] != 0
                or type(options["timeout"]) not in (int, float) or not math.isfinite(options["timeout"])
                or options["timeout"] <= 0
                or any(options["max_output_tokens"] + _RESERVE >= item["context_window"] for item in models_metadata.values())):
            raise ValueError("Opciones de generación incompatibles con el catálogo.")
        if completion is not None and not callable(completion):
            raise ValueError("completion debe ser un callable explícito.")
        self._models = deepcopy(models_metadata)
        self._options = deepcopy(options)
        self._completion = completion
        if type(capture_error_reason) is not bool:
            raise ValueError('capture_error_reason debe ser bool.')
        self._capture_error_reason = capture_error_reason

    def request_payload(self, request):
        if self._options.get("action_format", "json_object") == "native_tools":
            from .agent_native import render_native
            return render_native(request.snapshot())
        return {"messages": render_messages(request.snapshot()), "response_format": {"type": "json_object"}}

    def request_input_bytes(self, request):
        payload = self.request_payload(request)
        if self._options.get("action_format", "json_object") == "native_tools":
            # Include tool schemas and argument history, not just text content.
            return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        return sum(len(item["content"].encode("utf-8")) for item in payload["messages"])

    def manifest(self):
        try:
            version = importlib.metadata.version("litellm")
        except importlib.metadata.PackageNotFoundError:
            version = None
        native = self._options.get("action_format", "json_object") == "native_tools"
        return {"kind": "litellm_episode_generator", "models_metadata": deepcopy(self._models),
                "options": deepcopy(self._options), "litellm_version": version,
                "completion_injected": self._completion is not None,
                "diagnostic_error_reason_capture": self._capture_error_reason,
                "input_format": "native_tool_history_text_final_v2" if native else "compact_progress_explicit_json_v2",
                "format": "native_tools" if native else "json_object",
                "num_retries": 0, "max_retries": 0, "fallbacks": None,
                "context_window_fallback_dict": {}, "stream": False,
                "cache_enabled": False, "telemetry_enabled": False,
                "context_budget": {"template_reserve": _RESERVE,
                                   "rule": ("full messages/tools payload UTF-8 bytes" if native else "message UTF-8 bytes")
                                           + " + max_output_tokens + reserve <= context_window",
                                   "exact_tokenization": False},
                "input_truncation_allowed": False, "provider_context_preservation_verified": False,
                "cost_source": "harness tariffs; no inferred provider invoice"}

    def __call__(self, model, request):
        if type(model) is not str or model not in self._models:
            raise ValueError("El modelo no pertenece al catálogo LiteLLM congelado.")
        if not isinstance(request, AgentRequest):
            raise ValueError("Se requiere AgentRequest.")
        payload = self.request_payload(request)
        input_bytes = self.request_input_bytes(request)
        budget = input_bytes + self._options["max_output_tokens"] + _RESERVE
        if budget > self._models[model]["context_window"]:
            raise BackendContextError("El episodio excede el presupuesto conservador; no se recorta ni genera.")
        key = os.environ.get(self._models[model]["api_key_env"])
        if not _name(key):
            raise BackendHTTPError("Falta una credencial válida en la variable declarada.", status_code=401)
        completion = self._completion if self._completion is not None else _default_completion()
        arguments = {"model": model, **payload, "api_key": key,
                     "temperature": self._options["temperature"], "max_tokens": self._options["max_output_tokens"],
                     "timeout": self._options["timeout"], "stream": False,
                     "num_retries": 0, "max_retries": 0, "fallbacks": None,
                     "context_window_fallback_dict": {}, "caching": False,
                     "drop_params": False}
        if "api_base" in self._models[model]:
            arguments["api_base"] = self._models[model]["api_base"]
        started = time.perf_counter()
        try:
            response = completion(**arguments)
        except Exception as error:
            failure = _provider_error(error)
            if self._capture_error_reason:
                failure.operator_reason = _redacted_provider_reason(error, key)
            raise failure from None
        elapsed = max(0., time.perf_counter() - started)
        usage = _usage(response)
        returned = _get(response, "model")
        if returned not in self._models[model]["returned_model_ids"]:
            raise BackendIdentityError("El proveedor devolvió una identidad de modelo no declarada.", usage=usage)
        choices = _get(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise BackendResponseError("El proveedor no devolvió exactamente una respuesta.", usage=usage)
        native = self._options.get("action_format", "json_object") == "native_tools"
        message = _get(choices[0], "message")
        content = _native_action(message) if native else _get(message, "content")
        if type(content) is not str:
            raise BackendResponseError("La respuesta del proveedor no contiene texto.", usage=usage)
        finish = _get(choices[0], "finish_reason")
        usage["provider_metadata"] = {"provider": model.split("/", 1)[0],
                                      "requested_model": model, "returned_model": returned,
                                      "finish_reason": finish if type(finish) is str else None,
                                      "request_seconds": elapsed, "input_utf8_bytes": input_bytes,
                                      "conservative_context_budget": budget,
                                      "input_truncated_by_client": False,
                                      "provider_context_preservation_verified": False}
        if native:
            usage["provider_metadata"]["action_format"] = "native_tools"
            # Preserve the provider's action fields for audit, including invalid arguments.
            calls = _get(message, "tool_calls")
            usage["provider_metadata"]["native_tool_calls"] = [
                {key: value if type(value) is str else None for key, value in {
                    "type": _get(call, "type"), "name": _get(_get(call, "function"), "name"),
                    "arguments": _get(_get(call, "function"), "arguments")}.items()}
                for call in calls] if isinstance(calls, (list, tuple)) else None
        # An empty/malformed action is still a paid/measured generation. Preserve
        # its usage and let the harness parser record InvalidAction separately.
        return ModelReply(content, usage=usage, cost_usd=None)
