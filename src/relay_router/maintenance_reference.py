"""Offline published RouteLLM-BERT scores for a maintenance comparison.

This scores supplied last-user-message text only. It neither sees outcomes nor
chooses an operating threshold. The published GPT-4/Mixtral preference score is
transferred to GPT-OSS; it is not a calibrated maintenance-success probability.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = "configs/routellm_reference.json"
MANIFEST_SHA256 = "3bf541749991e7a17d400040c0bba70adaf36e08a1464cd220c7c5bcf6be619b"
CATALOG = ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")
PINNED = {
    "source_repository": "https://github.com/lm-sys/RouteLLM",
    "source_revision": "0b64fdafe049e596a3f5657c219329f24af24198",
    "checkpoint": "routellm/bert_gpt4_augmented",
    "checkpoint_revision": "86237e3df400762178ea98379477b8296e66d5e4",
    "weights_file": "model.safetensors",
    "weights_sha256": "e2a92240bad1c7fb9befbf021a62fad5feb1e2a3990d53a28d160a39bb1cac7b",
    "model_dir": "artifacts/routellm-bert",
    "packages": {"torch": "2.9.0", "transformers": "4.57.1"},
    "reference_tokenization": {"max_length": 512, "truncation": True, "padding": True, "prefix": None},
    "threads": 4,
}


def _load_predictor():
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in PINNED.items()):
        raise ValueError("La referencia BERT debe conservar configuración y revisión fijadas.")
    directory = ROOT / PINNED["model_dir"]
    if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("El manifiesto BERT no coincide con el hash fijado.")
    # This loader verifies every required checkpoint/tokenizer file before
    # loading safetensors, then enforces CPU, pinned versions and local-only I/O.
    from routerbench.routellm_reference import LocalBERTReference
    return LocalBERTReference(directory, config)


def _records(records):
    if not isinstance(records, list):
        raise ValueError("records debe ser una lista de task_id y routing_text.")
    seen, validated = set(), []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"task_id", "routing_text"}:
            raise ValueError("Cada entrada admite únicamente task_id y routing_text, sin resultados.")
        task_id, text = record["task_id"], record["routing_text"]
        if not isinstance(task_id, str) or not task_id.strip() or task_id in seen:
            raise ValueError("task_id debe ser texto no vacío y único.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("routing_text debe ser texto no vacío del último mensaje de usuario.")
        seen.add(task_id)
        validated.append((task_id, text))
    return validated


def _prediction(result):
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise ValueError("BERT debe devolver exactamente una predicción por texto.")
    record = result[0]
    probability = record.get("probability")
    if (isinstance(probability, bool) or not isinstance(probability, Real)
            or not isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("La probabilidad BERT debe ser finita y estar entre 0 y 1.")
    counts = {}
    for key in ("input_tokens", "retained_tokens", "truncated_tokens"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("Los conteos BERT deben ser enteros no negativos.")
        counts[key] = int(value)
    if (counts["input_tokens"] < 1 or not 1 <= counts["retained_tokens"] <= 512
            or counts["retained_tokens"] > counts["input_tokens"]
            or counts["truncated_tokens"] != counts["input_tokens"] - counts["retained_tokens"]):
        raise ValueError("Los conteos BERT no respetan la truncación fijada de 512 tokens.")
    return {"probability": float(probability), **counts}


def score_records(records: list[dict], predictor=None) -> dict:
    """Return ``reference`` provenance and ordered ``rows`` with individual scores.

    Each input has exactly ``task_id`` and ``routing_text``; the caller supplies
    the unchanged last user message, without a system-message prefix. Validate
    the entire input before loading or scoring. A supplied predictor is for
    offline tests and is explicitly marked as unverified in the result.

    ``router_seconds`` measures each actual ``predict_batch([text])`` call,
    including token counting. ``load_seconds`` is reported separately and is
    zero when no model is loaded. No cache, threshold selection or model call
    is performed here. Empty input does not load the checkpoint.
    """
    validated = _records(records)
    injected = predictor is not None
    if injected and not callable(getattr(predictor, "predict_batch", None)):
        raise TypeError("El predictor requiere predict_batch.")
    load_seconds = 0.0
    if validated and not injected:
        started = time.perf_counter()
        predictor = _load_predictor()
        load_seconds = time.perf_counter() - started
    verified = bool(validated) and not injected
    reference = {
        "implementation": "local_wrapper_published_routellm_bert_transfer",
        "checkpoint": PINNED["checkpoint"],
        "revision": PINNED["checkpoint_revision"],
        "weights_sha256": PINNED["weights_sha256"],
        "source_repository": PINNED["source_repository"],
        "source_revision": PINNED["source_revision"],
        "catalog": list(CATALOG),
        "load_seconds": load_seconds,
        "checkpoint_verified": verified,
        "predictor_injected": injected,
        "checkpoint_manifest_sha256": MANIFEST_SHA256 if verified else None,
        "provenance": "pinned_local_checkpoint" if verified else (
            "injected_predictor_unverified" if injected else "not_loaded_empty_input"),
        "tokenization": deepcopy(PINNED["reference_tokenization"]),
        "runtime_packages": deepcopy(PINNED["packages"]) if verified else None,
        "cpu_threads": PINNED["threads"] if verified else None,
        "input": "unchanged last user message content; no system prefix",
        "scoring": "1 - sum(softmax(logits)[-2:]); higher prefers the strong action",
        "supervision": "published GPT-4/Mixtral preferences with external augmentation",
        "scope": "pretrained transfer to GPT-OSS; not MF, SW or the official RouteLLM SDK",
        "calibrated_for_execution_pair": False,
        "threshold_selected": False,
    }
    rows = []
    for task_id, text in validated:
        started = time.perf_counter()
        result = predictor.predict_batch([text])
        seconds = time.perf_counter() - started
        prediction = _prediction(result)
        rows.append({"task_id": task_id, **prediction, "router_seconds": seconds,
                     "routing_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     "input_characters": len(text)})
    return {"reference": reference, "rows": rows}
