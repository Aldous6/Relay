"""Portable Ollama metadata inventory (Python 3.11+, standard library only).

Copy this file to the machine that can reach the laboratory's Ollama server.
It never starts Ollama, downloads models or requests inference. Templates,
system messages, Modelfiles and arbitrary server response bodies are omitted.
This is setup information, not a quality ranking or a calibrated Relay policy.

Official API contracts:
https://docs.ollama.com/api/tags
https://docs.ollama.com/api/ps
https://docs.ollama.com/api-reference/show-model-details.md
https://docs.ollama.com/api-reference/get-version.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


DEFAULT_URL = "http://127.0.0.1:11434"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class InventoryError(ValueError):
    """A bounded, sanitized error safe to include in the metadata report."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(value):
    try:
        parts = urlsplit(value)
        valid = (parts.scheme in {"http", "https"} and parts.hostname
                 and parts.username is None and parts.password is None
                 and not parts.query and not parts.fragment
                 and parts.path in {"", "/"} and parts.port != 0
                 and not any(char.isspace() or ord(char) < 32 for char in value))
    except ValueError:
        valid = False
    if not valid:
        raise InventoryError("Usa la URL base HTTP(S) de Ollama, sin ruta, claves ni credenciales.")
    return value.rstrip("/")


def request_metadata(base_url, path, payload=None, *, timeout=5):
    allowed = {"/api/tags": "GET", "/api/ps": "GET", "/api/version": "GET", "/api/show": "POST"}
    method = "GET" if payload is None else "POST"
    if allowed.get(path) != method or (path == "/api/show" and (
            not isinstance(payload, dict) or set(payload) != {"model"} or not isinstance(payload["model"], str))):
        raise InventoryError("Este diagnóstico sólo permite consultas de metadatos.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise InventoryError("El timeout debe ser un número positivo finito.")
    base_url = validate_url(base_url)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(base_url + path, data=data, method=method,
                      headers={"Accept": "application/json", "Content-Type": "application/json"})
    try:
        # Query only the configured server: no environment proxy or redirects.
        with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        raise InventoryError(f"Consulta de metadatos rechazada (HTTP {error.code}); no se reintentó.") from None
    except (URLError, TimeoutError, OSError):
        raise InventoryError("No se pudo consultar Ollama. Comprueba que esté disponible y revisa --url.") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise InventoryError("La respuesta de metadatos excede el límite de 2 MiB.")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        raise InventoryError("Ollama no devolvió JSON válido.") from None
    if not isinstance(value, dict) or value.get("error") is not None:
        raise InventoryError("Ollama devolvió una respuesta de metadatos inesperada.")
    return value


def _text(value):
    return value if isinstance(value, str) and 0 < len(value) <= 512 and all(ord(c) >= 32 for c in value) else None


def _number(value):
    return value if type(value) is int and value >= 0 else None


def _model_list(value):
    items = value.get("models")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise InventoryError("Ollama no devolvió una lista de modelos válida.")
    return items


def collect_inventory(base_url=DEFAULT_URL, *, timeout=5):
    base_url = validate_url(base_url)
    query = lambda path, payload=None: request_metadata(base_url, path, payload, timeout=timeout)
    installed = _model_list(query("/api/tags"))
    warnings = []

    def optional(path, payload=None):
        try:
            return query(path, payload)
        except InventoryError as error:
            warnings.append({"endpoint": path, "model": payload["model"] if payload else None, "error": str(error)})
            return {}

    version = _text(optional("/api/version").get("version"))
    if version is None and not any(warning["endpoint"] == "/api/version" for warning in warnings):
        warnings.append({"endpoint": "/api/version", "model": None, "error": "Versión de Ollama desconocida."})
    running_response = optional("/api/ps")
    try:
        running = _model_list(running_response)
    except InventoryError:
        running = None
        if not any(warning["endpoint"] == "/api/ps" for warning in warnings):
            warnings.append({"endpoint": "/api/ps", "model": None, "error": "Lista de modelos cargados desconocida."})
    models = []
    for item in installed:
        name = _text(item.get("name")) or _text(item.get("model"))
        if name is None:
            raise InventoryError("Un modelo del inventario carece de un nombre válido.")
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        shown = optional("/api/show", {"model": name})
        info = shown.get("model_info") if isinstance(shown.get("model_info"), dict) else {}
        contexts = {key: value for key, value in info.items()
                    if _text(key) and key.endswith(".context_length") and _number(value) is not None}
        caps = shown.get("capabilities")
        capabilities = list(caps) if isinstance(caps, list) and all(_text(cap) for cap in caps) else None
        digest = _text(item.get("digest"))
        # Match the exact name and digest; aliases and changed versions are not inferred.
        loaded = None if running is None else next((row for row in running
            if name in (row.get("name"), row.get("model")) and digest is not None and row.get("digest") == digest), None)
        known_loaded = running is not None and digest is not None
        models.append({"name": name, "digest": digest, "size_bytes": _number(item.get("size")),
                       "family": _text(details.get("family")), "parameter_size": _text(details.get("parameter_size")),
                       "quantization_level": _text(details.get("quantization_level")), "capabilities": capabilities,
                       "declared_context_lengths": contexts or None,
                       "loaded_at_inspection": loaded is not None if known_loaded else None,
                       "active_context_length": _number(loaded.get("context_length")) if loaded else None,
                       "allocated_vram_bytes": _number(loaded.get("size_vram")) if loaded else None,
                       "quality_evaluated": False})
    return {"schema_version": 1, "kind": "relay_ollama_inventory",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "ollama_version": version,
            "all_metadata_requests_succeeded": not warnings, "models": sorted(models, key=lambda row: row["name"]),
            "warnings": warnings, "generation_requests": 0, "download_requests": 0,
            "selection_policy": None, "gpu_hardware": None,
            "limitations": ["Metadata does not establish model quality, speed or local-only execution.",
                            "Declared context capacity is not the active configuration or a truncation guarantee.",
                            "Allocated model VRAM is not total GPU capacity; host hardware is not inspected.",
                            "Loaded state is a sequential snapshot and can change during collection.",
                            "Prompts, templates, system messages and server address are not exported; model names are included."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inventario de Ollama para Relay, sin inferencia ni descargas.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Servidor Ollama accesible desde esta máquina.")
    parser.add_argument("--timeout", type=float, default=5, help="Segundos máximos por consulta (5).")
    parser.add_argument("--output", default="relay-ollama-inventory.json", type=Path, help="Archivo JSON nuevo.")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise InventoryError("El informe ya existe. Elige otro nombre con --output para conservarlo.")
        result = collect_inventory(args.url, timeout=args.timeout)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    except (InventoryError, OSError) as error:
        message = str(error) if isinstance(error, InventoryError) else "No se pudo crear el archivo de salida."
        print(message, file=sys.stderr)
        return 1
    print(f"Informe: {args.output}\nModelos encontrados: {len(result['models'])}. No se generaron respuestas ni se descargaron modelos.")
    if result["warnings"]:
        print("Inventario parcial: algunas consultas no respondieron. Los datos faltantes están marcados en el informe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
