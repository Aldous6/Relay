"""Explicit live text-generation adapters for the Relay demonstration.

Importing this module does not discover providers, start services, download
models or generate text. A generation function makes exactly one selected-model
request. It does not substitute another model or silently trim conversation.

API references inspected for this implementation:
https://docs.ollama.com/api/chat
https://docs.ollama.com/api/tags
https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion
"""
from __future__ import annotations

import json
import math
import os
import time
from numbers import Real
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from relay_router.errors import (LiveBackendError, BackendTransportError, BackendTimeoutError,
                                 BackendRateLimitError, BackendHTTPError, BackendResponseError)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OPENROUTER_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_TEXT_ROLES = {"system", "user", "assistant"}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        # In particular, never forward an API Authorization header to another
        # URL. A redirect is a visible failure rather than an additional call.
        return None


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo.")
    return value


def _timeout(value):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout debe ser un número positivo finito.")
    return float(value)


def _model_name(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError("Se requiere un identificador de modelo concreto válido.")
    return value


def _ollama_url(base_url, path):
    if not isinstance(base_url, str):
        raise ValueError("La URL de Ollama debe ser texto.")
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("La URL de Ollama debe ser HTTP(S), sin credenciales, consulta ni fragmento.")
    if parts.path not in {"", "/"}:
        raise ValueError("La URL base de Ollama no debe contener una ruta.")
    return base_url.rstrip("/") + path


def validate_messages(messages):
    """Copy a complete text-only conversation without role conversion or trim.

    Documents must be content within a user message, never promoted to a system
    role by this adapter. Tool calls and multimodal messages require a separate
    contract and are rejected rather than partially forwarded.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages debe ser una lista no vacía de mensajes de texto.")
    copied = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Cada mensaje requiere exclusivamente role y content.")
        if not isinstance(message["role"], str) or message["role"] not in _TEXT_ROLES or not isinstance(message["content"], str):
            raise ValueError("Solo se admiten roles system/user/assistant y contenido de texto.")
        copied.append({"role": message["role"], "content": message["content"]})
    if not any(message["role"] == "user" and message["content"].strip() for message in copied):
        raise ValueError("La conversación debe contener una solicitud del usuario no vacía.")
    return copied


def _request_json(url, payload=None, *, timeout=120, api_key=None):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=_timeout(timeout)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # Do not read a response body: gateways can reflect requests or secrets.
        kind = BackendRateLimitError if error.code == 429 else BackendHTTPError
        retry_after = error.headers.get("Retry-After") if error.headers else None
        try:
            retry_after = float(retry_after)
        except (ValueError, TypeError):
            retry_after = None
        raise kind(f"El proveedor rechazó la solicitud (HTTP {error.code}).",
                   status_code=error.code, retry_after_seconds=retry_after) from None
    except (URLError, TimeoutError, OSError) as error:
        kind = BackendTimeoutError if isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError) else BackendTransportError
        raise kind("No se pudo completar la conexión con el proveedor; no se reintentó.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BackendResponseError("El proveedor devolvió una respuesta que no es JSON válido.") from None
    if not isinstance(result, dict) or result.get("error") is not None:
        raise BackendResponseError("El proveedor devolvió un error o una respuesta inesperada.")
    return result


def _token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _finite_nonnegative(value):
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _usage(input_tokens, output_tokens, total_tokens=None, cached_tokens=None):
    incoming, outgoing = _token_count(input_tokens), _token_count(output_tokens)
    total = _token_count(total_tokens)
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return {"input_tokens": incoming, "output_tokens": outgoing,
            "total_tokens": total, "cached_input_tokens": _token_count(cached_tokens)}


def _text_content(message):
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LiveBackendError("El proveedor no devolvió una respuesta de texto visible.")
    return content


def discover_ollama(base_url=OLLAMA_BASE_URL, timeout=3, include_context=False):
    """Read already installed models from a running Ollama service; never start it."""
    result = _request_json(_ollama_url(base_url, "/api/tags"), timeout=timeout)
    models = result.get("models")
    if not isinstance(models, list):
        raise LiveBackendError("Ollama no devolvió una lista de modelos válida.")
    discovered = []
    for item in models:
        if not isinstance(item, dict):
            raise LiveBackendError("Ollama devolvió metadatos de modelo inesperados.")
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name.strip():
            raise LiveBackendError("Un modelo instalado no tiene identificador válido.")
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        context_length = _token_count(details.get("context_length"))
        if include_context and context_length is None:
            metadata = _request_json(_ollama_url(base_url, "/api/show"), {"model": name}, timeout=timeout)
            information = metadata.get("model_info") if isinstance(metadata.get("model_info"), dict) else {}
            lengths = [_token_count(value) for key, value in information.items() if key.endswith(".context_length")]
            lengths = [value for value in lengths if value is not None and value > 0]
            context_length = min(lengths) if lengths else None
        discovered.append({"name": name, "model": item.get("model", name), "size_bytes": _token_count(item.get("size")),
                           "digest": item.get("digest"), "details": details,
                           "context_length": context_length,
                           "parameter_size": details.get("parameter_size"),
                           "quantization_level": details.get("quantization_level")})
    return sorted(discovered, key=lambda item: item["name"])


def generate_ollama(model, messages, max_output_tokens=512, timeout=120,
                    base_url=OLLAMA_BASE_URL, context_window=None):
    """Make one local selected-model request; cost of local compute is unknown.

    Full text is forwarded. Context capacity must be checked by the caller;
    this function cannot certify that an external server does not trim inputs.
    ``context_window`` sets Ollama num_ctx explicitly when the caller has
    validated the corresponding model limit.
    """
    model = _model_name(model)
    options = {"num_predict": _positive_integer(max_output_tokens, "max_output_tokens")}
    if context_window is not None:
        options["num_ctx"] = _positive_integer(context_window, "context_window")
        if max_output_tokens >= context_window:
            raise ValueError("La reserva de salida debe ser menor que context_window.")
    payload = {"model": model, "messages": validate_messages(messages), "stream": False, "options": options}
    started = time.perf_counter()
    result = _request_json(_ollama_url(base_url, "/api/chat"), payload, timeout=timeout)
    content = _text_content(result.get("message"))
    provider_nanoseconds = _finite_nonnegative(result.get("total_duration"))
    return {"mode": "live", "provider": "ollama", "requested_model": model,
            "model": result.get("model") or model, "content": content,
            "usage": _usage(result.get("prompt_eval_count"), result.get("eval_count"), cached_tokens=result.get("prompt_eval_cached_count")),
            "duration_seconds": time.perf_counter()-started,
            "provider_duration_seconds": provider_nanoseconds/1e9 if provider_nanoseconds is not None else None,
            "finish_reason": result.get("done_reason"), "cost_usd": None, "cost_source": None,
            "input_truncated_by_client": False, "provider_context_preservation_verified": False}


def generate_openrouter(api_key, model, messages, max_output_tokens=512, timeout=120):
    """Call a concrete OpenRouter model once, with no automatic model routing.

    The caller should obtain the key from the environment. It is neither stored
    nor returned. This explicit function can incur provider charges when called.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip() or api_key != api_key.strip() or any(character.isspace() for character in api_key):
        raise ValueError("Configura una credencial válida en OPENROUTER_API_KEY.")
    model = _model_name(model)
    if model.lower() in {"openrouter/auto", "openrouter/auto-beta", "openrouter/auto-code"}:
        raise ValueError("Relay requiere un modelo concreto, no un autorouter externo.")
    payload = {"model": model, "messages": validate_messages(messages),
               "max_tokens": _positive_integer(max_output_tokens, "max_output_tokens"), "stream": False,
               "provider": {"allow_fallbacks": False}}
    started = time.perf_counter()
    result = _request_json(OPENROUTER_COMPLETIONS_URL, payload, timeout=timeout, api_key=api_key)
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LiveBackendError("OpenRouter no devolvió una respuesta utilizable.")
    content = _text_content(choices[0].get("message"))
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cost = _finite_nonnegative(usage.get("cost"))
    return {"mode": "live", "provider": "openrouter", "requested_model": model,
            "model": result.get("model") or model, "content": content,
            "usage": _usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"), prompt_details.get("cached_tokens")),
            "duration_seconds": time.perf_counter()-started, "provider_duration_seconds": None,
            "finish_reason": choices[0].get("finish_reason"), "cost_usd": cost,
            "cost_source": "provider_response_usage.cost" if cost is not None else None,
            "input_truncated_by_client": False, "provider_context_preservation_verified": False}
