"""Use frozen BERT preference scores for exact prompts without loading weights.

Only checkpoint metadata and exact prompt coverage are verified here. A cache
digest identifies the file snapshot; it is not an independent recomputation of
the predictions. Original checkpoint loading/scoring time remains in provenance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# This integration's module imports only stdlib; heavy imports are inside the
# LocalBERTScorer constructor, which this module never invokes.
from relay_router.integrations.routellm import CHECKPOINT, REVISION, FILES

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_OLD_ROW = {"task_id", "probability", "input_tokens", "retained_tokens", "truncated_tokens",
            "router_seconds", "routing_text_sha256", "input_characters"}
_FRESH_ROW = {"task_id", "score", "input_tokens", "truncated_tokens", "routing_seconds", "prompt_sha256"}
_FRESH_TOP = {"kind", "at_utc", "suite_sha256", "reference", "thresholds", "truncation_policy",
              "threshold_provenance", "source_sha256", "rows"}
_OLD_REFERENCE = {"implementation", "checkpoint", "revision", "weights_sha256", "source_repository",
                  "source_revision", "catalog", "load_seconds", "checkpoint_verified", "predictor_injected",
                  "checkpoint_manifest_sha256", "provenance", "tokenization", "runtime_packages", "cpu_threads",
                  "input", "scoring", "supervision", "scope", "calibrated_for_execution_pair", "threshold_selected"}
_FRESH_REFERENCE = {"implementation", "checkpoint", "revision", "weights_sha256", "input_contract",
                    "max_input_tokens", "source_supervision", "success_probability", "cpu_threads",
                    "load_seconds", "router_cost_usd"}


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate cache JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("Cache JSON must not contain nonfinite numbers")


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _number(value, name, *, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid finite nonnegative {name}")
    return float(value)


def _tokens(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt(record):
    if type(record) is not dict:
        raise ValueError("Expected task or development record objects")
    text = record.get("prompt_text")
    task_fields = {"instruction", "initial_source", "public_tests"}
    if "prompt_text" not in record or task_fields.issubset(record):
        from relay_router.harness_episode import task_prompt
        try:
            generated = task_prompt(record)
        except (KeyError, TypeError) as exc:
            raise ValueError("A record requires prompt_text or the complete public task") from exc
        if "prompt_text" in record and text != generated:
            raise ValueError("prompt_text differs from the actual task prompt")
        text = generated
    _text(text, "prompt_text")
    if "routing_text_sha256" in record and record["routing_text_sha256"] != _hash(text):
        raise ValueError("Input record routing hash differs from its exact prompt")
    return text


def _reference(value, *, historical):
    if type(value) is not dict:
        raise ValueError("Expected a checkpoint reference object")
    if set(value) - (_OLD_REFERENCE if historical else _FRESH_REFERENCE):
        raise ValueError("Unexpected data in BERT checkpoint reference metadata")
    for name, expected in (("checkpoint", CHECKPOINT), ("revision", REVISION),
                           ("weights_sha256", FILES["model.safetensors"])):
        if value.get(name) != expected:
            raise ValueError(f"Cache reference differs from pinned BERT {name}")
    load_seconds = _number(value.get("load_seconds"), "original load_seconds")
    if historical:
        if (value.get("implementation") != "local_wrapper_published_routellm_bert_transfer"
                or value.get("checkpoint_verified") is not True or value.get("predictor_injected") is not False
                or value.get("tokenization") != {"max_length": 512, "truncation": True, "padding": True, "prefix": None}
                or value.get("supervision") != "published GPT-4/Mixtral preferences with external augmentation"
                or value.get("scoring") != "1 - sum(softmax(logits)[-2:]); higher prefers the strong action"):
            raise ValueError("Unsupported or unverified historical BERT scoring contract")
    elif (value.get("implementation") != "local_published_routellm_bert_scoring"
          or value.get("max_input_tokens") != 512 or value.get("success_probability") is not False
          or value.get("source_supervision") != "GPT-4/Mixtral preferences"):
        raise ValueError("Unsupported fresh BERT scoring contract")
    return load_seconds


class _CachedBERTScorer:
    load_seconds = 0.0
    cached = True

    def __init__(self, scores, provenance):
        self._scores = deepcopy(scores)
        self._provenance = deepcopy(provenance)

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def __call__(self, text):
        digest = _hash(_text(text, "routing text"))
        if digest not in self._scores:
            raise ValueError("The exact prompt is absent from the frozen BERT score cache")
        return dict(self._scores[digest])


def cached_bert_scorer(reference_path, records_or_tasks):
    """Return a text-only callable backed by one complete, validated cache.

    Pass the task/record iterable, not a suite wrapper. Two historical model
    branches may repeat a task with its identical prompt. Cache rows themselves
    must be unique and cover exactly all supplied task IDs and text hashes.
    No acceptance, cost, model output or model identity from records is read.
    Fresh cache ``suite_sha256`` must additionally be checked by callers that
    have the suite file; this helper has only the iterable of task records.
    """
    path = Path(reference_path)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_invalid_constant)
    if type(data) is not dict:
        raise ValueError("Expected a BERT score cache object")
    historical = set(data) == {"reference", "rows"}
    if not historical:
        if (set(data) != _FRESH_TOP or data.get("kind") != "frozen_historical_BERT_controls"
                or data.get("truncation_policy") != "keep_baseline"
                or data.get("threshold_provenance") != "historical_public_v2_development_frozen_no_new_outcomes"
                or data.get("thresholds") != [.5, .58]
                or type(data.get("suite_sha256")) is not str or not _HEX.fullmatch(data["suite_sha256"])):
            raise ValueError("Unsupported fresh BERT cache schema or contract")
    load_seconds = _reference(data.get("reference"), historical=historical)
    if isinstance(records_or_tasks, (dict, str, bytes)):
        raise ValueError("Pass task/development records, not a suite wrapper")
    expected = {}
    for record in records_or_tasks:
        text = _prompt(record)
        task_id = _text(record.get("task_id"), "task_id")
        previous = expected.setdefault(task_id, text)
        if previous != text:
            raise ValueError("Repeated task ID has different exact prompts")
    if not expected:
        raise ValueError("At least one task prompt is required")
    rows = data.get("rows")
    if type(rows) is not list or len(rows) != len(expected):
        raise ValueError("BERT cache has missing or extra task rows")
    scores, seen, timings = {}, set(), []
    for row in rows:
        if type(row) is not dict or set(row) != (_OLD_ROW if historical else _FRESH_ROW):
            raise ValueError("BERT cache row fields differ from the frozen score schema")
        task_id = _text(row.get("task_id"), "cache task_id")
        if task_id in seen or task_id not in expected:
            raise ValueError("Duplicate or unexpected BERT cache task")
        seen.add(task_id)
        text = expected[task_id]
        digest = row["routing_text_sha256" if historical else "prompt_sha256"]
        if digest != _hash(text):
            raise ValueError("BERT cache hash differs from the exact task prompt")
        score = _number(row["probability" if historical else "score"], "BERT score", maximum=1)
        incoming = _tokens(row["input_tokens"], "input_tokens")
        truncated = _tokens(row["truncated_tokens"], "truncated_tokens")
        if incoming < 1 or truncated != max(0, incoming - 512):
            raise ValueError("BERT truncation does not match the pinned 512-token contract")
        if historical and (type(row["retained_tokens"]) is not int or row["retained_tokens"] != min(incoming, 512)
                           or type(row["input_characters"]) is not int or row["input_characters"] != len(text)):
            raise ValueError("Historical retained tokens or character count do not match the task")
        timings.append(_number(row["router_seconds" if historical else "routing_seconds"], "original scoring seconds"))
        result = {"score": score, "input_tokens": incoming, "truncated_tokens": truncated}
        if digest in scores and scores[digest] != result:
            raise ValueError("Identical exact prompts have inconsistent cached scores or truncation")
        scores[digest] = result
    if seen != set(expected):
        raise ValueError("BERT cache does not cover every task")
    try:
        scoring_seconds = math.fsum(timings)
        original_total = _number(load_seconds + scoring_seconds, "original total compute seconds")
    except OverflowError as exc:
        raise ValueError("Original scoring time must remain finite") from exc
    provenance = dict(data["reference"])
    # The historical wrapper used the same pinned checkpoint and published
    # aggregation. Normalize its metadata alias only after validating that
    # contract above; retain the complete original declaration for auditing.
    provenance.update(original_reference=deepcopy(data["reference"]),
                      implementation="local_published_routellm_bert_scoring",
                      max_input_tokens=512, source_supervision="GPT-4/Mixtral preferences")
    provenance.update(cached=True, cache_sha256=hashlib.sha256(raw).hexdigest(),
                      cache_path=str(path.resolve()), cache_schema="historical" if historical else "fresh",
                      cache_task_count=len(seen), cache_unique_prompt_count=len(scores),
                      cache_checkpoint_load_seconds=0.0, original_load_seconds=load_seconds,
                      original_scoring_seconds=scoring_seconds,
                      original_total_compute_seconds=original_total,
                      router_cost_usd=None, cache_verification="pinned_checkpoint_metadata_and_exact_prompt_coverage_not_recomputed_predictions")
    if not historical:
        provenance["cache_suite_sha256"] = data["suite_sha256"]
    return _CachedBERTScorer(scores, provenance)
