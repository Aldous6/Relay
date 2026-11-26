"""Local first-turn routing for the live OpenCode catalog.

The routing policy uses advertised capabilities and a tariff comparison for a
reference input/output token mix. It has no fitted quality or latency estimates
for an arbitrary OpenCode account. The reference mix is not the agent's actual
context, a generation limit, or a charge to the user.
"""
from __future__ import annotations

from collections.abc import Iterable
import re
import unicodedata

from .integrations.opencode import OpenCodeModel


POLICY_ID = "opencode-catalog-v1"
_LARGE_PROMPT_BYTES = 4096
_EXTENDED_TASK = re.compile(
    r"\b(?:arquitectura|architecture|diagnos\w*|depur\w*|debug\w*|"
    r"refactor\w*|migra\w*|concurren\w*|intermit\w*|race condition|"
    r"root cause|causa raiz|distributed|distribuid[oa]s?|"
    r"multi[- ]?(?:file|module)|(?:varios|multiples) (?:archivos|modulos))\b"
)
_FREE_LABEL = re.compile(r"(?:^|[\s._:/-])free(?:$|[\s._:/-])", re.IGNORECASE)


def advertised_free(model: OpenCodeModel) -> bool:
    """Keep zero-priced entries backed by native metadata or a legacy free label."""
    rates = (model.input_usd_per_million, model.output_usd_per_million)
    if rates != (0, 0):
        return False
    declared_free = getattr(model, "advertised_free", None)
    if declared_free is not None:
        return declared_free is True
    slug = model.model_id.partition("/")[2]
    return bool(_FREE_LABEL.search(slug) or _FREE_LABEL.search(model.name))


def _tariff(model, input_tokens, output_tokens):
    """Return a sort key and provenance, keeping ambiguous catalog zeros apart."""
    price_in, price_out = model.input_usd_per_million, model.output_usd_per_million
    if advertised_free(model):
        return (0, 0.), "advertised_free"
    if price_in is None or price_out is None:
        return (1, 0.), "unknown"
    if price_in == price_out == 0:
        # Some catalog versions also fill missing rates with zero.
        return (1, 0.), "zero_unverified"
    reference_cost = input_tokens * price_in / 1_000_000 + output_tokens * price_out / 1_000_000
    return (0, reference_cost), "catalog_rates"


def _available_output(model, input_estimate):
    limits = []
    if model.output_tokens is not None:
        limits.append(model.output_tokens)
    if model.context_tokens is not None:
        limits.append(model.context_tokens - input_estimate)
    return min(limits) if limits else None


def select_opencode_model(prompt: str, models: Iterable[OpenCodeModel], *, selected_model: str | None = None) -> dict:
    """Choose one exact catalog ID. Callers may prefilter a user's allowed set."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Escribe un prompt para seleccionar un modelo.")
    catalog = list(models)
    if not catalog:
        raise ValueError("El catálogo seleccionado de OpenCode está vacío.")
    if len({model.model_id for model in catalog}) != len(catalog):
        raise ValueError("El catálogo de OpenCode contiene identificadores de modelo duplicados.")
    prompt_bytes = len(prompt.encode("utf-8"))
    # A planning estimate for the prompt alone. Tokenization and OpenCode's
    # eventual system message, tool schemas and file context vary by model.
    input_estimate = (prompt_bytes + 3) // 4
    folded = unicodedata.normalize("NFKD", prompt.casefold())
    words = "".join(c for c in folded if not unicodedata.combining(c))
    extended = prompt_bytes >= _LARGE_PROMPT_BYTES or bool(_EXTENDED_TASK.search(words))
    target = 2048 if extended else 512
    ranked = []
    for model in catalog:
        capability = (model.input_text, model.output_text, model.tool_call)
        if False in capability or model.status in {"disabled", "retired", "unavailable"}:
            continue
        if model.input_tokens is not None and input_estimate > model.input_tokens:
            continue
        room = _available_output(model, input_estimate)
        if room is not None and room < 1:
            continue
        price_key, tariff_status = _tariff(model, input_estimate, target)
        # Prefer proven text/tools support and enough output room. If the
        # catalog is incomplete, an otherwise usable entry remains eligible.
        room_rank = 1 if room is None else (0 if room >= target else 2)
        context = model.context_tokens or 0
        capacity_rank = (-context, -(room or 0)) if extended else (context, room or 0)
        key = (
            model.status == "deprecated",
            capability.count(None),
            room_rank,
            extended and model.reasoning is not True,
            price_key,
            model.context_tokens is None,
            capacity_rank,
            model.model_id.partition("/")[0],
            model.model_id,
        )
        ranked.append((key, model, tariff_status))
    if not ranked:
        raise ValueError("Ningún modelo seleccionado cumple los requisitos de texto, herramientas y contexto del prompt.")
    if selected_model is None:
        _, model, tariff_status = min(ranked, key=lambda row: row[0])
    else:
        choice = next((row for row in ranked if row[1].model_id == selected_model), None)
        if choice is None:
            raise ValueError("El modelo que elegiste ya no está disponible para este mensaje.")
        _, model, tariff_status = choice
    if selected_model is not None:
        reason = "Elegiste este modelo para este mensaje."
    elif extended:
        reason = "El prompt es extenso o menciona trabajo de diagnóstico, diseño o cambios amplios. Se priorizan razonamiento y espacio de respuesta."
    else:
        reason = "Para esta solicitud estándar se prioriza una tarifa anunciada baja y una ventana suficiente."
    if tariff_status == "advertised_free":
        reason += " OpenCode lo anuncia como gratuito."
    elif tariff_status == "catalog_rates":
        reason += " Las tarifas de entrada y salida proceden del catálogo."
    else:
        reason += " La tarifa de este modelo no está confirmada."
    return {
        "model": model.model_id,
        "policy_id": POLICY_ID,
        "status": "experimental",
        "profile": "extended" if extended else "standard",
        "reason": reason,
        "eligible_models": len(ranked),
        "eligible_model_ids": sorted(row[1].model_id for row in ranked),
        "catalog_models": len(catalog),
        "prompt_bytes": prompt_bytes,
        "estimated_prompt_tokens": input_estimate,
        "requested_output_tokens": target,
        "tariff_status": tariff_status,
    }
