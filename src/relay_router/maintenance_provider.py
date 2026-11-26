"""Bounded Groq JSON completions for the maintenance integration.

No credentials are discovered, no dependencies/assets are downloaded, and no
request is retried. Construction performs no provider I/O. Transport uses the
fixed Groq HTTPS endpoint and refuses redirects. Account eligibility/billing
must be established by the caller; local budgets cannot inspect account usage.

Contracts checked 2026-11-15:
https://console.groq.com/docs/api-reference
https://console.groq.com/docs/structured-outputs
https://console.groq.com/docs/rate-limits
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .errors import LiveBackendError

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_ERROR_BYTES = 64 * 1024
_RESERVE = 512
_ERROR_IDENTIFIERS = frozenset({
    "json_validate_failed", "tool_use_failed", "output_parse_failed", "invalid_request_error",
    "invalid_api_key", "model_not_found", "context_length_exceeded", "rate_limit_exceeded",
    "insufficient_quota", "unsupported_parameter", "invalid_parameter", "invalid_value",
    "authentication_error", "permission_error", "server_error", "api_error",
})


class MaintenanceProviderError(LiveBackendError):
    """Sanitized failure; usage stays unknown unless a response reported it."""

    def __init__(self, message, *, category="provider_unknown", status_code=None,
                 retry_after=None, usage=None, request_seconds=None, input_estimate=None,
                 provider_error_code=None, provider_error_type=None, provider_error_message=None):
        super().__init__(message, status_code=status_code,
                         retry_after_seconds=retry_after, usage=usage)
        self.category = category
        self.retry_after = self.retry_after_seconds
        self.request_seconds = request_seconds
        self.input_estimate = deepcopy(input_estimate)
        self.provider_error_code = provider_error_code
        self.provider_error_type = provider_error_type
        self.provider_error_message = provider_error_message

    def details(self):
        return {**super().details(), "usage": deepcopy(self.usage),
                "request_seconds": self.request_seconds,
                "input_estimate": deepcopy(self.input_estimate),
                **{key: value for key in ("provider_error_code", "provider_error_type", "provider_error_message")
                   if (value := getattr(self, key)) is not None},
                "automatic_retry": False}


def _positive(value, name, *, integer=False, allow_zero=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"{name} tiene un valor inválido.")
    return value


def _native_model(model):
    if type(model) is not str:
        raise ValueError("Se requiere un identificador de modelo Groq.")
    native = model.removeprefix("groq/")
    if native not in SUPPORTED_MODELS:
        raise ValueError("Modelo fuera del catálogo GPT-OSS declarado.")
    return native


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("El JSON tiene claves duplicadas.")
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError("El JSON contiene un número no finito.")


def _json(value):
    return json.loads(value, object_pairs_hook=_unique, parse_constant=_invalid_number)


def _response_format(schema):
    """Snapshot a harness-owned schema, without interpreting model instructions.

    The workflow validates returned content against its own contract. These
    checks cover the root requirements of Groq strict mode; this is not a
    general JSON Schema validator.
    """
    if schema is None:
        return {"type": "json_object"}, None
    if (type(schema) is not dict or schema.get("type") != "object"
            or type(schema.get("properties")) is not dict
            or type(schema.get("required")) is not list
            or schema.get("additionalProperties") is not False):
        raise ValueError("El esquema estricto requiere un objeto con properties, required y additionalProperties=false.")
    properties, required = schema["properties"], schema["required"]
    if (any(type(name) is not str or type(value) is not dict for name, value in properties.items())
            or any(type(name) is not str for name in required)
            or len(required) != len(set(required)) or set(required) != set(properties)):
        raise ValueError("Todos los campos del esquema estricto deben ser requeridos una sola vez.")
    try:
        raw = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        snapshot = _json(raw)
    except (ValueError, TypeError, RecursionError):
        raise ValueError("El esquema debe contener únicamente valores JSON finitos.") from None
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return {"type": "json_schema", "json_schema": {
        "name": "relay_maintenance", "strict": True, "schema": snapshot}}, digest


def _usage(response):
    raw = response.get("usage") if type(response) is dict else None
    raw = raw if type(raw) is dict else {}
    def count(key):
        value = raw.get(key)
        return value if type(value) is int and value >= 0 else None
    incoming, outgoing, total = count("prompt_tokens"), count("completion_tokens"), count("total_tokens")
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": total}


def _http_error_metadata(error):
    """Bound parsing and retain taxonomy/counts, never reflected text or output.

    Provider messages may quote the key, prompt or failed_generation. A local
    controlled description replaces them; unfamiliar identifiers are omitted.
    """
    try:
        content = error.read(_MAX_ERROR_BYTES + 1)
        if type(content) is not bytes or len(content) > _MAX_ERROR_BYTES:
            return {}
        body = _json(content.decode("utf-8"))
    except Exception:
        return {}
    if type(body) is not dict:
        return {}
    rejection = body.get("error")
    rejection = rejection if type(rejection) is dict else {}
    metadata = {}
    for key in ("code", "type"):
        value = rejection.get(key)
        if type(value) is str and value in _ERROR_IDENTIFIERS:
            metadata["provider_error_" + key] = value
    if metadata.get("provider_error_code") == "json_validate_failed":
        metadata["provider_error_message"] = "El proveedor no pudo validar la salida JSON generada."
    elif metadata:
        metadata["provider_error_message"] = "El proveedor rechazó la solicitud; consulta el código y tipo registrados."
    # Preserve only explicit numeric usage; absence remains unknown, not zero.
    usage_source = body if type(body.get("usage")) is dict else rejection
    if type(usage_source.get("usage")) is dict:
        metadata["usage"] = _usage(usage_source)
    return metadata


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward authorization or source material to a redirected host.
        return None


def _transport(request, timeout):
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        content = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(content) > _MAX_RESPONSE_BYTES:
            raise MaintenanceProviderError("Respuesta del proveedor demasiado grande.", category="provider_contract")
        try:
            return _json(content.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise MaintenanceProviderError("Respuesta HTTP sin JSON válido.", category="provider_contract") from None


class GroqMaintenanceProvider:
    """At most ``max_requests`` sends; fixed model and zero automatic retries.

    ``transport`` is an optional test callback taking (urllib Request, timeout)
    and returning a decoded response dictionary. ``tokenizer_mode='auto'``
    reuses verified local GPT-OSS assets when available. Missing optional assets
    fall back to UTF-8 bytes plus a margin, without downloading anything.
    A byte/token estimate is a local budget, not a proven Groq server bound.
    """

    def __init__(self, api_key, *, max_output_tokens=2048, timeout=90,
                 max_requests=6, request_budget=8000, minimum_interval_seconds=60,
                 tokenizer_mode="auto", transport=None, sleep=None, clock=None,
                 on_wait=None, output_mode="strict"):
        if type(api_key) is not str or not api_key.strip() or api_key != api_key.strip() or any(c.isspace() for c in api_key):
            raise ValueError("Se requiere una credencial Groq válida; no se guarda en archivos.")
        self._api_key = api_key
        self.max_output_tokens = _positive(max_output_tokens, "max_output_tokens", integer=True)
        self.timeout = _positive(timeout, "timeout")
        self.max_requests = _positive(max_requests, "max_requests", integer=True)
        self.request_budget = _positive(request_budget, "request_budget", integer=True)
        self.minimum_interval_seconds = _positive(minimum_interval_seconds, "minimum_interval_seconds", allow_zero=True)
        if self.max_output_tokens + _RESERVE >= self.request_budget:
            raise ValueError("El presupuesto debe dejar espacio para la entrada.")
        if tokenizer_mode not in {"auto", "bytes"}:
            raise ValueError("tokenizer_mode debe ser auto o bytes.")
        if type(output_mode) is not str or output_mode not in {"strict", "prompt_json"}:
            raise ValueError("output_mode debe ser strict o prompt_json.")
        for callback in (transport, sleep, clock, on_wait):
            if callback is not None and not callable(callback):
                raise ValueError("Los callbacks deben ser funciones.")
        self._transport = transport if transport is not None else _transport
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        self._on_wait = on_wait
        self._tokenizer_mode = tokenizer_mode
        self.output_mode = output_mode
        self._tokenizer = None
        self._tokenizer_checked = False
        self._last_send = {}
        self.request_count = 0

    def manifest(self):
        return {"provider": "groq", "models": list(SUPPORTED_MODELS),
                "max_output_tokens": self.max_output_tokens, "timeout": self.timeout,
                "max_requests": self.max_requests, "request_budget": self.request_budget,
                "minimum_interval_seconds": self.minimum_interval_seconds,
                "tokenizer_mode": self._tokenizer_mode,
                "response_format": {"type": "json_object"} if self.output_mode == "strict" else None,
                "output_mode": self.output_mode, "reasoning_effort": "medium",
                "supported_response_formats": ["json_object", "json_schema_strict"],
                "schema_budget": ("additional UTF-8 bytes of the structured response_format as conservative token reserve"
                                  if self.output_mode == "strict" else
                                  "schema in caller messages; no separate provider response_format reserve"),
                "automatic_retries": 0, "input_truncation": False,
                "account_tier_verified_by_client": False,
                "monetary_cost": "unknown; account configuration belongs to caller"}

    def _estimate(self, messages):
        serialized = json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":"))
        byte_count = len(serialized.encode("utf-8"))
        if self._tokenizer_mode == "auto" and not self._tokenizer_checked:
            try:
                from routerbench.bfcl_tokens import GPTOSSTokenEstimator
                self._tokenizer = GPTOSSTokenEstimator()
            except (ImportError, FileNotFoundError):
                self._tokenizer = None
            # A corrupted manifest/asset raises rather than silently falling back.
            self._tokenizer_checked = True
        if self._tokenizer is not None:
            from routerbench.bfcl_harness import BFCLRequest, BFCLState
            state = BFCLState(0, 0, 0, (), (), 0, (), 1, 0)
            result = self._tokenizer.estimate(BFCLRequest(tuple(messages), (), state))
            return {**result, "method": "verified_local_gptoss_estimate",
                    "provider_options_reserve": _RESERVE}
        return {"estimated_input_tokens": byte_count,
                "input_utf8_bytes": byte_count,
                "method": "utf8_bytes_conservative_estimate",
                "provider_options_reserve": _RESERVE,
                "provider_server_template_verified": False,
                "is_proven_upper_bound": False, "input_truncated": False}

    def preflight(self, model, messages, *, response_schema=None):
        """Validate and estimate without credentials leaving the process."""
        _native_model(model)
        response_format, schema_sha256 = _response_format(response_schema)
        if type(messages) not in (list, tuple) or not messages:
            raise ValueError("messages debe contener mensajes de texto.")
        for message in messages:
            if (type(message) is not dict or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or type(message["content"]) is not str):
                raise ValueError("Cada mensaje requiere role y content de texto.")
        if not any(message["role"] == "user" and message["content"].strip() for message in messages):
            raise ValueError("Se requiere una solicitud de usuario no vacía.")
        estimate = self._estimate(deepcopy(list(messages)))
        # The existing tokenizer estimates messages, not provider schema
        # framing. In strict mode charge the additional format as bytes. In
        # prompt_json mode the schema is already in the caller's messages.
        schema_bytes = (len(json.dumps(response_format, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8"))
                        if response_schema is not None and self.output_mode == "strict" else 0)
        estimate["response_format_utf8_bytes"] = schema_bytes
        estimate["additional_schema_reserve_tokens"] = schema_bytes
        estimate["response_schema_sha256"] = schema_sha256
        estimate["output_mode"] = self.output_mode
        estimate["estimated_input_tokens"] += schema_bytes
        estimate["reserved_total"] = estimate["estimated_input_tokens"] + self.max_output_tokens + _RESERVE
        estimate["request_budget"] = self.request_budget
        if estimate["reserved_total"] > self.request_budget:
            raise MaintenanceProviderError(
                "La solicitud excede el presupuesto local. Reduce los archivos autorizados o su tamaño; no se recorta el contexto.",
                category="request_budget", input_estimate=estimate)
        return estimate

    def complete_structured(self, model, messages, response_schema):
        """Validate the declared schema; delivery follows the fixed output mode.

        In prompt_json mode the caller supplies the schema in its messages and
        the workflow still validates every returned field/path locally.
        """
        if response_schema is None:
            raise ValueError("complete_structured requiere un esquema explícito.")
        return self.complete(model, messages, response_schema=response_schema)

    def complete(self, model, messages, *, response_schema=None):
        native = _native_model(model)
        # Snapshot before preflight and transport so caller mutations cannot alter
        # the checked request. Include every supplied byte of message content.
        snapshot = deepcopy(list(messages)) if type(messages) in (list, tuple) else messages
        response_format, schema_sha256 = _response_format(response_schema)
        frozen_schema = response_format.get("json_schema", {}).get("schema")
        estimate = self.preflight(native, snapshot, response_schema=frozen_schema)
        if self.request_count >= self.max_requests:
            raise MaintenanceProviderError("Se alcanzó el máximo de llamadas de esta ejecución.", category="call_budget", input_estimate=estimate)
        previous = self._last_send.get(native)
        wait = max(0., self.minimum_interval_seconds - (self._clock() - previous)) if previous is not None else 0.
        if wait:
            if self._on_wait:
                self._on_wait(wait)
            remaining = wait
            while remaining > 0:
                chunk = min(remaining, 60.)
                self._sleep(chunk)
                remaining -= chunk
        body = {"model": native, "messages": snapshot,
                "temperature": 0, "max_completion_tokens": self.max_output_tokens,
                "reasoning_effort": "medium",
                "stream": False, "n": 1}
        if self.output_mode == "strict":
            body["response_format"] = response_format
        else:
            body["include_reasoning"] = False
        request = Request(ENDPOINT, data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self._api_key,
                                   "Content-Type": "application/json", "User-Agent": "Relay-Maintenance/1"}, method="POST")
        started = self._clock()
        self._last_send[native] = started
        self.request_count += 1  # Uncertain sends consume the local allowance too.
        failure = None
        try:
            response = self._transport(request, self.timeout)
        except HTTPError as error:
            status = error.code if type(error.code) is int else None
            try:
                retry_after = float(error.headers.get("retry-after"))
            except (AttributeError, TypeError, ValueError, OverflowError):
                retry_after = None
            metadata = _http_error_metadata(error)
            category = ("rate_limit" if status == 429 else "model_output_json"
                        if status == 400 and metadata.get("provider_error_code") == "json_validate_failed"
                        else "http")
            failure = MaintenanceProviderError(
                "El proveedor rechazó la solicitud; no se reintentó.",
                category=category, status_code=status, retry_after=retry_after, **metadata)
            try:
                error.close()
            except Exception:
                pass
        except MaintenanceProviderError as error:
            failure = error
        except (TimeoutError, socket.timeout):
            failure = MaintenanceProviderError("La llamada agotó su tiempo de espera; el consumo es desconocido.", category="timeout")
        except (URLError, OSError):
            failure = MaintenanceProviderError("Falló la conexión; el estado de la llamada es desconocido.", category="transport")
        except Exception:
            failure = MaintenanceProviderError("Falló el transporte del proveedor; no se reintentó.", category="transport")
        elapsed = max(0., self._clock() - started)
        if failure is not None:
            failure.request_seconds = elapsed
            failure.input_estimate = estimate
            # Raise outside the handler to avoid attaching a secret-bearing
            # exception as __context__ (even a suppressed one).
            failure.__context__ = None
            failure.__cause__ = None
            raise failure from None
        usage = _usage(response)
        def invalid(message, category="provider_contract"):
            return MaintenanceProviderError(message, category=category, usage=usage,
                                            request_seconds=elapsed, input_estimate=estimate)
        if type(response) is not dict:
            raise invalid("Respuesta inesperada del proveedor.")
        if response.get("model") != native:
            raise invalid("El proveedor devolvió otro modelo.", "model_identity")
        choices = response.get("choices")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            raise invalid("El proveedor no devolvió exactamente una respuesta.")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise invalid("El modelo agotó el límite de salida; la propuesta quedó incompleta.", "output_truncated")
        message = choice.get("message")
        if (choice.get("finish_reason") != "stop" or type(message) is not dict
                or message.get("tool_calls") or type(message.get("content")) is not str):
            raise invalid("El proveedor no devolvió un objeto JSON final.", "model_output")
        content = message["content"]
        try:
            parsed = _json(content)
            valid = type(parsed) is dict
        except (ValueError, RecursionError):
            valid = False
        if not valid:
            raise invalid("La respuesta no contiene un objeto JSON válido.", "model_output_json")
        return {"content": content, "usage": usage, "returned_model": native,
                "request_seconds": elapsed, "quota_wait_seconds": wait,
                "finish_reason": choice["finish_reason"], "input_estimate": estimate,
                "response_format": deepcopy(response_format) if self.output_mode == "strict" else None,
                "output_mode": self.output_mode, "response_schema_sha256": schema_sha256}
