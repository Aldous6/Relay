"""Offline GPT-OSS input estimates; never a claim about a provider's template.

The published OpenAI template is rendered only when its single-call contract is
compatible. A separate, complete native JSON count always retains every message,
tool, schema field and parallel call. Neither quantity is a proven server bound.
No loader in this module downloads assets or reads credentials.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .bfcl_harness import BFCLRequest

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = _ROOT / ".cache/bfcl-tokenizer/gpt-oss-v1"
DEFAULT_PROVENANCE = _ROOT / "configs/bfcl_tokenizer_gptoss_v1.json"
PROVENANCE_SHA256 = "0d1e7f0cdab52e89aad7559582666bee6428d0bfaec2e202094711da6f594a8d"
_BPE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"

# OpenAI's o200k_base splitting expression, used by o200k_harmony. The pinned
# primary implementation and the vocabulary's SHA256 are in the provenance file.
_PATTERN = "|".join([
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
])


def native_request_payload(request: BFCLRequest) -> dict:
    """Snapshot exactly the messages/tools/flags sent by the native BFCL adapter."""
    if not isinstance(request, BFCLRequest):
        raise ValueError("The token estimator requires BFCLRequest.")
    request = request.snapshot()
    payload = {"messages": list(request.messages)}
    if request.tools:
        payload.update(tools=list(request.tools), tool_choice="auto", parallel_tool_calls=True)
    return payload


def _raise_template_error(message):
    # Never propagate a rendered prompt or model content into diagnostic logs.
    from jinja2 import TemplateError
    raise TemplateError("The published template rejected this message contract.")


class GPTOSSTokenEstimator:
    """max(full native JSON tokens, compatible published Harmony tokens) + margin.

    Uses only the current BFCLRequest, with no case, future turn, reference or
    dataset access. The returned input estimate excludes output reservations.
    Construction validates all local assets before creating a tokenizer/template.
    """

    supported_models = ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")

    def __init__(self, *, asset_root=DEFAULT_ASSET_ROOT, provenance_path=DEFAULT_PROVENANCE,
                 margin_tokens=256, template_date="2026-11-12", reasoning_effort="medium"):
        if type(margin_tokens) is not int or margin_tokens < 0:
            raise ValueError("margin_tokens must be a nonnegative integer.")
        if type(template_date) is not str or date.fromisoformat(template_date).isoformat() != template_date:
            raise ValueError("template_date must be an ISO date.")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("Unknown GPT-OSS reasoning effort.")
        raw = Path(provenance_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != PROVENANCE_SHA256:
            raise ValueError("Tokenizer provenance SHA256 mismatch.")
        provenance = json.loads(raw)
        root = Path(asset_root).resolve()
        assets = {}
        for item in provenance["assets"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Tokenizer asset is outside the declared local cache.")
            content = path.read_bytes()
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Tokenizer asset SHA256 mismatch: {item['path']}.")
            assets[item["path"]] = content
        for suffix in ("chat_template.jinja", "tokenizer_config.json", "special_tokens_map.json"):
            if assets[f"gpt-oss-20b/{suffix}"] != assets[f"gpt-oss-120b/{suffix}"]:
                raise ValueError("This estimator requires identical GPT-OSS tokenizer/template assets.")
        bpe = assets["o200k_base.tiktoken"]
        if hashlib.sha256(bpe).hexdigest() != _BPE_SHA256:
            raise ValueError("The vocabulary differs from official o200k_harmony.")
        ranks = {base64.b64decode(token, validate=True): int(rank)
                 for token, rank in (line.split() for line in bpe.splitlines() if line)}
        # Construct directly from verified bytes: tiktoken.get_encoding would
        # implicitly download its vocabulary when the global cache is missing.
        import tiktoken
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        specials = {f"<|reserved_{index}|>": index
                    for index in (200000, 200001, 200004, 200009, 200010, 200011, *range(200013, 201088))}
        specials.update({"<|startoftext|>": 199998, "<|endoftext|>": 199999,
                         "<|return|>": 200002, "<|constrain|>": 200003,
                         "<|channel|>": 200005, "<|start|>": 200006,
                         "<|end|>": 200007, "<|message|>": 200008,
                         "<|call|>": 200012, "<|endofprompt|>": 200018})
        # o200k_harmony retains reserved aliases even where named tokens exist.
        self._encoding = tiktoken.Encoding(name="o200k_harmony", pat_str=_PATTERN,
                                           mergeable_ranks=ranks, special_tokens=specials)
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                    extensions=["jinja2.ext.loopcontrols"])
        environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
        environment.globals["raise_exception"] = _raise_template_error
        frozen_date = date.fromisoformat(template_date)
        environment.globals["strftime_now"] = lambda fmt: frozen_date.strftime(fmt)
        self._template = environment.from_string(assets["gpt-oss-20b/chat_template.jinja"].decode("utf-8"))
        self._margin = margin_tokens
        self._template_date = template_date
        self._reasoning_effort = reasoning_effort
        self._provenance = deepcopy(provenance)

    def _harmony_count(self, payload):
        from jinja2 import TemplateError
        messages = payload["messages"]
        if not messages:
            return None, "unsupported_empty_messages"
        if any(message["role"] == "system" for message in messages[1:]):
            return None, "unsupported_late_system_message"
        if any(message.get("content") is not None and type(message["content"]) is not str
               for message in messages):
            return None, "unsupported_nontext_content"
        # The published template explicitly reads tool_calls[0]. Do not silently
        # render a truncated batch and describe that as the complete request.
        if any("tool_calls" in message and (not isinstance(message["tool_calls"], (list, tuple))
               or len(message["tool_calls"]) != 1) for message in messages):
            return None, "unsupported_parallel_or_empty_tool_batch"
        # Native null assistant content contains no text; the HF template expects
        # a string for its tag check. Only this render copy is normalized.
        render_messages = deepcopy(messages)
        for message in render_messages:
            if message.get("content") is None:
                message["content"] = ""
        try:
            rendered = self._template.render(messages=render_messages, tools=payload.get("tools", []),
                add_generation_prompt=True, builtin_tools=[], reasoning_effort=self._reasoning_effort)
        except (TemplateError, TypeError, ValueError, KeyError):
            return None, "published_template_rejected_contract"
        return len(self._encoding.encode(rendered, allowed_special="all")), "rendered_published_template"

    def estimate(self, request: BFCLRequest) -> dict:
        payload = native_request_payload(request)
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # Literal special-token spellings in content are ordinary text for this
        # complete JSON count, so they cannot collapse the estimate to one token.
        json_tokens = len(self._encoding.encode_ordinary(serialized))
        harmony_tokens, status = self._harmony_count(payload)
        return {"estimated_input_tokens": max(json_tokens, harmony_tokens or 0) + self._margin,
                "serialized_payload_tokens": json_tokens, "published_harmony_tokens": harmony_tokens,
                "safety_margin_tokens": self._margin, "published_template_status": status,
                "input_utf8_bytes": len(serialized.encode("utf-8")), "input_truncated": False,
                "provider_server_template_verified": False, "is_proven_upper_bound": False}

    def estimated_input_tokens(self, request: BFCLRequest) -> int:
        return self.estimate(request)["estimated_input_tokens"]

    def manifest(self) -> dict:
        return deepcopy({"kind": "gpt_oss_offline_native_input_estimate_v1", "encoding": "o200k_harmony",
            "supported_models": list(self.supported_models),
            "supported_model_ids": list(self.supported_models),
            "method": "max(full_native_json_ordinary_tokens,compatible_published_harmony_tokens)+margin",
            "margin_tokens": self._margin, "output_tokens_included": False,
            "packages": {name: version(name) for name in ("tiktoken", "jinja2")},
            "provenance_sha256": PROVENANCE_SHA256, "provenance": self._provenance,
            "template_parameters": {"date": self._template_date, "reasoning_effort": self._reasoning_effort,
                                    "builtin_tools": [], "add_generation_prompt": True},
            "template_compatibility": "published template assumes one tool call per assistant message; otherwise JSON-only fallback",
            "template_normalization": "null content becomes empty text only in the render copy; native JSON is unchanged",
            "native_json_includes": ["all current messages", "all tool schemas", "tool_choice", "parallel_tool_calls"],
            "server_template_verification": "no server template verification",
            "provider_server_template_verified": False, "is_proven_upper_bound": False,
            "actual_provider_usage_required": True, "input_truncation_allowed": False,
            "network_access": False, "future_turns_or_references_accessed": False})
