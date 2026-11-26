"""Own, small prompt policy with offline adaptation and fresh paired validation.

Training alone imports scikit-learn. Exported heads and artifact verification
use the standard library; a BERT representation requires a supplied local scorer.
Public historical outcomes are development, never a new test.
An empirical gate is not a statistical noninferiority guarantee. Artifact hashes
detect accidental changes; user-supplied evidence is not independently certified.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import string
import time
import unicodedata


FEATURE_NAMES = (
    "log_chars", "log_words", "log_lines", "digit_fraction",
    "punctuation_fraction", "indent_fraction", "code_keyword_count", "log_max_line",
)
FEATURE_VERSION = "prompt-eight-v1"
BERT_FEATURE_VERSION = "routellm-bert-preference-v1"
BERT_FEATURE_NAMES = ("routellm_bert_preference_score",)
TARGET_VERSION = "candidate-failure-v1"
SCORE_SEMANTICS = "uncalibrated_candidate_failure_score_not_success_probability"
SPLIT_RULE = "sha256('relay-adapt-v1|' + group_id); first floor(2*n/3) fit, rest calibration"
_KEYWORDS = frozenset("def return if elif else for while class import from try except raise with lambda yield assert async await".split())
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_LIMITATION = "Empirical public-task pilot only; no statistical noninferiority or future savings guarantee."


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _seal(value, field):
    value[field] = _digest(value)
    return value


def _unseal(value, field):
    if type(value) is not dict:
        raise ValueError("Expected a JSON object.")
    try:
        result = json.loads(_canonical(value))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Expected finite JSON data.") from exc
    signature = result.pop(field, None)
    if type(signature) is not str or not _HEX.fullmatch(signature) or _digest(result) != signature:
        raise ValueError(f"Invalid {field} integrity digest.")
    result[field] = signature
    return result


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonempty text.")
    return value


def _number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data.")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its permitted range.")
    return float(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _unique_strings(values, name, *, hashes=False):
    if type(values) is not list or any(type(v) is not str or not v for v in values):
        raise ValueError(f"{name} must be a list of nonempty strings.")
    if len(set(values)) != len(values) or hashes and any(not _HEX.fullmatch(v) for v in values):
        raise ValueError(f"Invalid unique values in {name}.")
    return set(values)


def normalized_prompt_hash(text):
    """Independent duplicate key; never use normalized text for predictors."""
    normalized = " ".join(unicodedata.normalize("NFKC", _text(text, "prompt_text")).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prompt_features(text):
    """Eight fixed variables, full original text, no learned vocabulary or IDs."""
    _text(text, "prompt_text")
    lines = text.splitlines() or [text]
    words = re.findall(r"\b\w+\b", text)
    length = len(text)
    indent = sum(len(line) - len(line.lstrip(" \t")) for line in lines)
    return [math.log1p(length), math.log1p(len(words)), math.log1p(len(lines)),
            sum(c.isdigit() for c in text) / length,
            sum(c in string.punctuation for c in text) / length,
            indent / length, float(sum(word in _KEYWORDS for word in words)),
            math.log1p(max(map(len, lines)))]


def _encoder_provenance():
    # This adapter module imports only stdlib; no weights are loaded here.
    from .integrations.routellm import CHECKPOINT, REVISION, FILES
    return {"checkpoint": CHECKPOINT, "revision": REVISION,
            "weights_sha256": FILES["model.safetensors"], "max_input_tokens": 512,
            "implementation": "local_published_routellm_bert_scoring",
            "source_supervision": "GPT-4/Mixtral preferences"}


def _representation_values(representation, text, scorer=None):
    _text(text, "prompt_text")
    if representation == "structural":
        return prompt_features(text), {"input_tokens": None, "truncated_tokens": None}
    if representation != "routellm-bert":
        raise ValueError("Unsupported prompt representation.")
    if not callable(scorer) or inspect.iscoroutinefunction(scorer):
        raise ValueError("The routellm-bert representation requires a synchronous local scorer.")
    # A supplied resident engine exposes provenance. A cached-score callback is
    # also supported, with its verified origin remaining the caller's contract.
    provenance = getattr(scorer, "provenance", None)
    if provenance is not None and (type(provenance) is not dict or any(
            provenance.get(key) != value for key, value in _encoder_provenance().items())):
        raise ValueError("The supplied encoder differs from the frozen checkpoint contract.")
    result = scorer(text)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise ValueError("The encoder must return a synchronous scoring result.")
    if type(result) is not dict:
        raise ValueError("The encoder must return score, input_tokens and truncated_tokens.")
    score = _number(result.get("score"), "encoder score", minimum=0, maximum=1)
    incoming = _integer(result.get("input_tokens"), "input_tokens", 1)
    truncated = _integer(result.get("truncated_tokens"), "truncated_tokens")
    if truncated > incoming or incoming - truncated > 512:
        raise ValueError("Encoder token counts contradict the pinned 512-token input contract.")
    return [score], {"input_tokens": incoming, "truncated_tokens": truncated}


def _pairs(records, baseline, candidate):
    rows, index, prompt_groups = [], {}, {}
    for raw in records:
        if type(raw) is not dict:
            raise ValueError("Each observation must be a dictionary.")
        row = {key: _text(raw.get(key), key) for key in ("task_id", "group_id", "prompt_text", "model", "cost_basis")}
        if row["model"] not in (baseline, candidate):
            raise ValueError("Observations must use exactly the declared candidate pair.")
        if type(raw.get("accepted")) is not bool:
            raise ValueError("accepted must be a known boolean; failures cannot be dropped.")
        row["accepted"] = raw["accepted"]
        row["cost_usd"] = None if raw.get("cost_usd") is None else _number(raw["cost_usd"], "cost_usd", minimum=0)
        if "cost_complete" in raw and (type(raw["cost_complete"]) is not bool or not raw["cost_complete"] and row["cost_usd"] is not None):
            raise ValueError("cost_complete contradicts the recorded cost.")
        key = (row["task_id"], row["model"])
        if key in index:
            raise ValueError("Duplicate task/model observation; aggregate attempts into the complete episode first.")
        index[key] = row
        rows.append(row)
        prompt_hash = normalized_prompt_hash(row["prompt_text"])
        previous = prompt_groups.setdefault(prompt_hash, row["group_id"])
        if previous != row["group_id"]:
            raise ValueError("Normalized duplicate prompts cannot belong to different groups.")
    if not rows:
        raise ValueError("At least one complete paired task is required.")
    pairs = []
    for task_id in sorted({row["task_id"] for row in rows}):
        left, right = index.get((task_id, baseline)), index.get((task_id, candidate))
        if left is None or right is None:
            raise ValueError("Every task requires both complete model episodes; missing pairs cannot be dropped.")
        if left["group_id"] != right["group_id"] or left["prompt_text"] != right["prompt_text"]:
            raise ValueError("Paired episodes must share exactly the same group and original prompt.")
        pairs.append({"task_id": task_id, "group_id": left["group_id"], "prompt_text": left["prompt_text"],
                      "prompt_sha256": normalized_prompt_hash(left["prompt_text"]), "baseline": left, "candidate": right})
    return pairs, sorted(rows, key=lambda row: (row["task_id"], row["model"]))


def _metrics(rows):
    count = len(rows)
    accepted = sum(row["accepted"] for row in rows)
    unknown = sum(row["cost_usd"] is None for row in rows)
    bases = sorted({row["cost_basis"] for row in rows})
    basis = bases[0] if len(bases) == 1 and bases[0] != "unknown" else None
    subtotal = math.fsum(row["cost_usd"] for row in rows if row["cost_usd"] is not None)
    complete = unknown == 0 and basis is not None
    total = subtotal if complete else None
    return {"task_count": count, "group_count": len({row["group_id"] for row in rows}),
            "accepted_count": accepted, "acceptance_rate": accepted / count if count else None,
            "unknown_cost_count": unknown, "known_cost_subtotal_usd": subtotal,
            "cost_complete": complete, "cost_basis": basis, "cost_bases": bases,
            "total_cost_usd": total,
            "cost_per_accepted_task_usd": total / accepted if total is not None and accepted else None}


def _savings(baseline, selected):
    left, right = baseline["cost_per_accepted_task_usd"], selected["cost_per_accepted_task_usd"]
    if left is None or left <= 0 or right is None or baseline["cost_basis"] != selected["cost_basis"]:
        return None
    return 1 - right / left


def _score_values_unchecked(artifact, values):
    predictor = artifact["predictor"]
    if predictor["kind"] == "constant_safe":
        return 1.
    try:
        linear = math.fsum(coef * ((value - mean) / scale)
                           for coef, value, mean, scale in zip(predictor["coefficients"], values,
                                                              predictor["mean"], predictor["scale"])) + predictor["intercept"]
    except (OverflowError, ValueError) as exc:
        raise ValueError("Prompt score is outside the supported numeric range.") from exc
    if not math.isfinite(linear):
        raise ValueError("Prompt score must remain finite.")
    if linear >= 0:
        return 1 / (1 + math.exp(-linear))
    exp_value = math.exp(linear)
    return exp_value / (1 + exp_value)


def _scored_unchecked(artifact, text, scorer=None):
    values, metadata = _representation_values(artifact.get("representation", "structural"), text, scorer)
    return _score_values_unchecked(artifact, values), metadata


def _selected(artifact, pairs, threshold=None, *, scorer=None, scored=None):
    cutoff = artifact["threshold"] if threshold is None else threshold
    if scored is None:
        scored = [_scored_unchecked(artifact, pair["prompt_text"], scorer) for pair in pairs]
    decisions = [score < cutoff and not metadata["truncated_tokens"] for score, metadata in scored]
    rows = [pair["candidate" if cheap else "baseline"] for pair, cheap in zip(pairs, decisions)]
    losses = sum(pair["baseline"]["accepted"] and not row["accepted"] for pair, row in zip(pairs, rows))
    gains = sum(not pair["baseline"]["accepted"] and row["accepted"] for pair, row in zip(pairs, rows))
    return rows, decisions, losses, gains


def train_policy(records, *, baseline_model, candidate_model, scope="openhands-python-v1",
                 min_validation_groups=20, min_savings=.10, scorer=None, representation="structural"):
    """Fit risk on historical fit groups, choose one cutoff on calibration only.

    The score targets ``candidate fails``, including failures shared with the
    baseline. Its sigmoid is
    not a separately calibrated success probability. A one-class fit retains
    baseline, including when all observed risks were zero. No OOD cutoffs are
    learned from this tiny sample. Structural variables use the full text;
    the optional pinned BERT representation forces baseline after truncation.
    Applicability is limited by explicit scope and fresh validation. The BERT
    feature is transferred pretrained preference, not an own-trained encoder.
    """
    _text(baseline_model, "baseline_model")
    _text(candidate_model, "candidate_model")
    _text(scope, "scope")
    if representation not in ("structural", "routellm-bert"):
        raise ValueError("representation must be structural or routellm-bert.")
    if baseline_model == candidate_model:
        raise ValueError("The candidate must differ from baseline.")
    _integer(min_validation_groups, "min_validation_groups", 1)
    min_savings = _number(min_savings, "min_savings", minimum=0, maximum=1)
    pairs, rows = _pairs(records, baseline_model, candidate_model)
    groups = sorted({pair["group_id"] for pair in pairs},
                    key=lambda group: hashlib.sha256(("relay-adapt-v1|" + group).encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError("Adaptation requires at least three independent development groups.")
    fit_groups = set(groups[:2 * len(groups) // 3])
    fit = [pair for pair in pairs if pair["group_id"] in fit_groups]
    calibration = [pair for pair in pairs if pair["group_id"] not in fit_groups]
    # Exactly one feature extraction per task; calibration sweeps never rerun
    # the encoder or inspect outcomes to construct a representation.
    representation_cache, represented = {}, {}
    for pair in pairs:
        text = pair["prompt_text"]
        if text not in representation_cache:
            representation_cache[text] = _representation_values(representation, text, scorer)
        represented[pair["task_id"]] = representation_cache[text]
    values = [represented[pair["task_id"]][0] for pair in fit]
    targets = [int(not pair["candidate"]["accepted"]) for pair in fit]
    known_bases = {row["cost_basis"] for row in rows if row["cost_usd"] is not None}
    if len(known_bases) != 1 or "unknown" in known_bases:
        raise ValueError("Development requires one comparable known cost basis; unknown costs stay unknown.")
    feature_names = list(FEATURE_NAMES if representation == "structural" else BERT_FEATURE_NAMES)
    dimension = len(feature_names)
    predictor = {"kind": "constant_safe", "mean": [0.] * dimension, "scale": [1.] * dimension,
                 "coefficients": [0.] * dimension, "intercept": 0.}
    version = None
    if len(set(targets)) == 2:
        # No optional ML imports occur when loading or running an exported policy.
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", random_state=42))
        pipeline.fit(values, targets)
        scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
        predictor = {"kind": "logistic", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                     "coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}
        version = sklearn.__version__
    artifact = {
        "schema_version": 1, "kind": "relay_adaptive_prompt_policy", "scope": scope,
        "baseline_model": baseline_model, "candidate_model": candidate_model,
        "representation": representation,
        "feature_version": FEATURE_VERSION if representation == "structural" else BERT_FEATURE_VERSION,
        "feature_names": feature_names,
        "encoder": None if representation == "structural" else _encoder_provenance(),
        "target_version": TARGET_VERSION,
        "predictor": predictor, "threshold": 0., "cost_basis": next(iter(known_bases)),
        "criteria": {"min_validation_groups": min_validation_groups, "min_savings": min_savings, "max_paired_losses": 0},
        "training": {"split_rule": SPLIT_RULE, "fit_group_ids": sorted(fit_groups),
                     "calibration_group_ids": sorted(set(groups) - fit_groups),
                     "development_group_ids": sorted(groups), "development_task_ids": [pair["task_id"] for pair in pairs],
                     "development_prompt_sha256": sorted({pair["prompt_sha256"] for pair in pairs}),
                     "input_digest": _digest(rows), "fit_task_count": len(fit), "calibration_task_count": len(calibration),
                     "fit_positive_count": sum(targets), "sklearn_version": version,
                     "encoder_truncated_task_count": sum(bool(metadata["truncated_tokens"]) for _, metadata in represented.values()),
                     "algorithm": "StandardScaler+LogisticRegression(C=.1,class_weight=balanced,random_state=42)",
                     "historical_data_role": "development_only_including_any_previously_consumed_validation"},
        "score_semantics": SCORE_SEMANTICS,
        "inference_scope": "full_original_prompt_no_truncation_no_claim_of_ood_detection" if representation == "structural"
                           else "pinned_pretrained_preference_feature_baseline_after_encoder_truncation",
        "status": "experimental", "router_cost_usd": None,
        "data_provenance": "user_supplied_paired_episodes_not_independently_verified",
        "limitation": _LIMITATION,
    }
    baseline = _metrics([pair["baseline"] for pair in calibration])
    scored = [(_score_values_unchecked(artifact, represented[pair["task_id"]][0]), represented[pair["task_id"]][1])
              for pair in calibration]
    baseline_rows, _, _, _ = _selected(artifact, calibration, scored=scored)
    chosen = (0., _metrics(baseline_rows), 0, 0, 0.)
    scores = sorted({score for score, _ in scored})
    thresholds = sorted({0., 1., *scores, *((left + right) / 2 for left, right in zip(scores, scores[1:]))})
    qualifying = []
    if predictor["kind"] != "constant_safe":
        for threshold in thresholds:
            selected, decisions, losses, _ = _selected(artifact, calibration, threshold, scored=scored)
            stats = _metrics(selected)
            saving = _savings(baseline, stats)
            if losses == 0 and saving is not None and saving + 1e-12 >= min_savings and any(decisions):
                qualifying.append((stats["cost_per_accepted_task_usd"], sum(decisions), threshold, stats, saving))
    if qualifying:
        _, count, threshold, stats, saving = min(qualifying, key=lambda value: value[:3])
        chosen = (threshold, stats, 0, count, saving)
    artifact["threshold"] = chosen[0]
    artifact["calibration"] = {"group_count": len(groups) - len(fit_groups), "baseline": baseline,
                               "relay": chosen[1], "paired_losses": chosen[2], "selected_candidate_count": chosen[3],
                               "relative_savings": _savings(baseline, chosen[1]), "thresholds_evaluated": len(thresholds),
                               "reason": "candidate_qualified_on_calibration" if qualifying else
                               "constant_target_safe_baseline" if predictor["kind"] == "constant_safe" else "no_qualifying_threshold"}
    return validate_artifact(_seal(artifact, "artifact_id"))


def validate_artifact(value):
    artifact = _unseal(value, "artifact_id")
    if artifact.get("schema_version") != 1 or artifact.get("kind") != "relay_adaptive_prompt_policy":
        raise ValueError("Unsupported adaptive policy artifact.")
    representation = artifact.get("representation", "structural")
    if representation == "structural":
        if (artifact.get("feature_version") != FEATURE_VERSION or artifact.get("feature_names") != list(FEATURE_NAMES)
                or artifact.get("encoder") is not None):
            raise ValueError("Policy structural feature contract has changed.")
    elif representation == "routellm-bert":
        if (artifact.get("feature_version") != BERT_FEATURE_VERSION or artifact.get("feature_names") != list(BERT_FEATURE_NAMES)
                or artifact.get("encoder") != _encoder_provenance()):
            raise ValueError("Policy pretrained encoder feature contract has changed.")
    else:
        raise ValueError("Unsupported prompt representation.")
    if artifact.get("target_version") != TARGET_VERSION or artifact.get("score_semantics") != SCORE_SEMANTICS:
        raise ValueError("Policy target must be the versioned absolute candidate-failure objective.")
    for key in ("baseline_model", "candidate_model", "scope", "cost_basis"):
        _text(artifact.get(key), key)
    if artifact["baseline_model"] == artifact["candidate_model"] or artifact["cost_basis"] == "unknown":
        raise ValueError("Invalid model pair or development cost basis.")
    if artifact.get("status") != "experimental" or artifact.get("router_cost_usd") is not None:
        raise ValueError("Policy artifacts alone are experimental and do not price router compute.")
    _number(artifact.get("threshold"), "threshold", minimum=0, maximum=1)
    predictor = artifact.get("predictor")
    if type(predictor) is not dict or predictor.get("kind") not in ("logistic", "constant_safe"):
        raise ValueError("Unsupported predictor.")
    for key in ("mean", "scale", "coefficients"):
        values = predictor.get(key)
        if type(values) is not list or len(values) != len(artifact["feature_names"]):
            raise ValueError("Predictor dimensions must match the fixed feature contract.")
        for number in values:
            _number(number, key)
            if key == "scale" and number <= 0:
                raise ValueError("Scaler values must be positive.")
    _number(predictor.get("intercept"), "intercept")
    if predictor["kind"] == "constant_safe" and artifact["threshold"] != 0:
        raise ValueError("A constant-target fit must retain baseline.")
    criteria = artifact.get("criteria")
    if type(criteria) is not dict or set(criteria) != {"min_validation_groups", "min_savings", "max_paired_losses"}:
        raise ValueError("Invalid validation criteria.")
    _integer(criteria["min_validation_groups"], "min_validation_groups", 1)
    _number(criteria["min_savings"], "min_savings", minimum=0, maximum=1)
    if type(criteria["max_paired_losses"]) is not int or criteria["max_paired_losses"] != 0:
        raise ValueError("This policy requires zero observed paired losses.")
    training = artifact.get("training")
    if type(training) is not dict or training.get("split_rule") != SPLIT_RULE:
        raise ValueError("Invalid development split contract.")
    fit = _unique_strings(training.get("fit_group_ids"), "fit_group_ids")
    calibration = _unique_strings(training.get("calibration_group_ids"), "calibration_group_ids")
    development = _unique_strings(training.get("development_group_ids"), "development_group_ids")
    if not fit or not calibration or fit & calibration or fit | calibration != development:
        raise ValueError("Fit and calibration must be nonempty, disjoint and cover development.")
    ordered = sorted(development, key=lambda group: hashlib.sha256(("relay-adapt-v1|" + group).encode()).hexdigest())
    if fit != set(ordered[:2 * len(ordered) // 3]):
        raise ValueError("Development split differs from the declared outcome-independent rule.")
    _unique_strings(training.get("development_task_ids"), "development_task_ids")
    _unique_strings(training.get("development_prompt_sha256"), "development_prompt_sha256", hashes=True)
    if not _HEX.fullmatch(str(training.get("input_digest", ""))):
        raise ValueError("Invalid training input digest.")
    return artifact


def score_prompt(artifact, prompt_text, *, scorer=None):
    """Return an uncalibrated candidate-failure score; exported head is stdlib."""
    policy = validate_artifact(artifact)
    _text(prompt_text, "prompt_text")
    return _scored_unchecked(policy, prompt_text, scorer)[0]


def select_model(artifact, prompt_text, *, scorer=None):
    started = time.perf_counter()
    policy = validate_artifact(artifact)
    _text(prompt_text, "prompt_text")
    score, metadata = _scored_unchecked(policy, prompt_text, scorer)
    cheap = score < policy["threshold"] and not metadata["truncated_tokens"]
    return {"model": policy["candidate_model"] if cheap else policy["baseline_model"],
            "selector_id": "relay-adaptive-" + policy["artifact_id"][:16], "score": score,
            "preference_score": score, "threshold": policy["threshold"],
            "input_tokens": metadata["input_tokens"], "truncated_tokens": metadata["truncated_tokens"],
            "routing_seconds": max(0., time.perf_counter() - started),
            "status": "experimental", "router_cost_usd": None,
            "reason": "baseline_after_encoder_truncation" if metadata["truncated_tokens"] else
                      "candidate_below_frozen_risk_threshold" if cheap else "baseline_at_or_above_frozen_risk_threshold"}


def validate_fresh_prompts(artifact, records):
    """Check freshness before generation; accepts prompt-only task records."""
    policy = validate_artifact(artifact)
    training = policy["training"]
    tasks, groups, hashes = set(training["development_task_ids"]), set(training["development_group_ids"]), set(training["development_prompt_sha256"])
    seen_hash_groups, seen_tasks = {}, {}
    for row in records:
        task_id = _text(row.get("task_id"), "task_id")
        group_id = _text(row.get("group_id"), "group_id")
        prompt = _text(row.get("prompt_text"), "prompt_text")
        prompt_hash = normalized_prompt_hash(prompt)
        if task_id in tasks or group_id in groups or prompt_hash in hashes:
            raise ValueError("Validation must use fresh tasks, groups and normalized prompts disjoint from all historical development.")
        if seen_hash_groups.setdefault(prompt_hash, group_id) != group_id:
            raise ValueError("Fresh normalized duplicate prompts must share one group.")
        if seen_tasks.setdefault(task_id, (group_id, prompt)) != (group_id, prompt):
            raise ValueError("A fresh task identifier must have one group and original prompt.")


def _gate(artifact, report):
    reasons = []
    baseline, relay = report["metrics"]["baseline"], report["metrics"]["relay"]
    if report["group_count"] < artifact["criteria"]["min_validation_groups"]:
        reasons.append("insufficient_validation_groups")
    if report["paired_losses"]:
        reasons.append("observed_paired_quality_loss")
    if baseline["accepted_count"] == 0 or relay["accepted_count"] == 0:
        reasons.append("no_accepted_tasks")
    if not baseline["cost_complete"] or not relay["cost_complete"]:
        reasons.append("unknown_or_incomparable_selected_cost")
    if baseline["cost_basis"] != artifact["cost_basis"] or relay["cost_basis"] != artifact["cost_basis"]:
        reasons.append("cost_basis_changed_or_incomparable")
    saving = _savings(baseline, relay)
    if saving is None or saving + 1e-12 < artifact["criteria"]["min_savings"]:
        reasons.append("insufficient_savings_per_accepted_task")
    if report["selected_candidate_count"] == 0 or artifact["threshold"] == 0:
        reasons.append("baseline_only_policy")
    return reasons, saving


def _loss_bound(losses, count):
    # One-sided 95% Wilson bound; descriptive public prompts, not a guarantee.
    z = 1.6448536269514722
    rate, z2 = losses / count, z * z
    return (rate + z2 / (2 * count) + z * math.sqrt(rate * (1 - rate) / count + z2 / (4 * count * count))) / (1 + z2 / count)


def evaluate_policy(artifact, validation_records, *, scorer=None):
    """Evaluate frozen paired branches without tuning or modifying the policy."""
    policy = validate_artifact(artifact)
    pairs, rows = _pairs(validation_records, policy["baseline_model"], policy["candidate_model"])
    validate_fresh_prompts(policy, pairs)
    selected, choices, losses, gains = _selected(policy, pairs, scorer=scorer)
    report = {"schema_version": 1, "kind": "relay_adaptive_validation", "artifact_id": policy["artifact_id"],
              "scope": policy["scope"], "input_digest": _digest(rows), "group_count": len({p["group_id"] for p in pairs}),
              "validation_task_ids": [p["task_id"] for p in pairs], "validation_group_ids": sorted({p["group_id"] for p in pairs}),
              "validation_prompt_sha256": sorted({p["prompt_sha256"] for p in pairs}),
              "metrics": {"baseline": _metrics([p["baseline"] for p in pairs]),
                          "candidate": _metrics([p["candidate"] for p in pairs]), "relay": _metrics(selected)},
              "paired_losses": losses, "paired_gains": gains,
              "paired_loss_upper_95": _loss_bound(losses, len(pairs)),
              "uncertainty_scope": "one_sided_95_wilson_descriptive_public_prompts_not_a_noninferiority_guarantee",
              "selected_candidate_count": sum(choices), "selected_baseline_count": len(choices) - sum(choices),
              "evaluation_method": "paired_episode_branch_replay_not_independent_extra_policy_runs",
              "cost_scope": "whole_episode_llm_cost_including_failures_and_retries_excluding_unpriced_local_compute",
              "router_cost_usd": None, "net_savings_verified": False, "limitation": _LIMITATION,
              "data_provenance": "user_supplied_paired_episodes_not_independently_verified"}
    reasons, saving = _gate(policy, report)
    report.update(activation_eligible=not reasons, reasons=reasons, relative_savings=saving)
    candidate, relay = report["metrics"]["candidate"], report["metrics"]["relay"]
    report["candidate_fixed_comparison"] = {
        "relative_savings_vs_candidate": _savings(candidate, relay),
        "acceptance_delta_vs_candidate": relay["acceptance_rate"] - candidate["acceptance_rate"],
        "incremental_advantage_demonstrated": candidate["cost_complete"] and relay["cost_complete"]
        and relay["acceptance_rate"] >= candidate["acceptance_rate"]
        and _savings(candidate, relay) is not None and _savings(candidate, relay) > 0,
    }
    return validate_report(_seal(report, "validation_report_id"), policy)


def _validate_metric(metric, tasks, groups):
    if type(metric) is not dict or metric.get("task_count") != tasks or metric.get("group_count") != groups:
        raise ValueError("Validation metric counts do not match the validation population.")
    accepted = _integer(metric.get("accepted_count"), "accepted_count")
    unknown = _integer(metric.get("unknown_cost_count"), "unknown_cost_count")
    if accepted > tasks or unknown > tasks:
        raise ValueError("Metric counts exceed the number of tasks.")
    subtotal = _number(metric.get("known_cost_subtotal_usd"), "known_cost_subtotal_usd", minimum=0)
    bases = _unique_strings(metric.get("cost_bases"), "cost_bases")
    basis = next(iter(bases)) if len(bases) == 1 and "unknown" not in bases else None
    complete = unknown == 0 and basis is not None
    if (metric.get("cost_basis") != basis or type(metric.get("cost_complete")) is not bool
            or metric["cost_complete"] != complete or metric.get("total_cost_usd") != (subtotal if complete else None)
            or metric.get("cost_per_accepted_task_usd") != (subtotal / accepted if complete and accepted else None)
            or metric.get("acceptance_rate") != accepted / tasks):
        raise ValueError("Validation costs and acceptance metrics are inconsistent.")


def validate_report(value, artifact):
    """Verify report integrity and recompute the empirical gate from metrics."""
    policy, report = validate_artifact(artifact), _unseal(value, "validation_report_id")
    if (report.get("schema_version") != 1 or report.get("kind") != "relay_adaptive_validation"
            or report.get("artifact_id") != policy["artifact_id"] or report.get("scope") != policy["scope"]):
        raise ValueError("Validation report does not match this policy and scope.")
    tasks = _unique_strings(report.get("validation_task_ids"), "validation_task_ids")
    groups = _unique_strings(report.get("validation_group_ids"), "validation_group_ids")
    hashes = _unique_strings(report.get("validation_prompt_sha256"), "validation_prompt_sha256", hashes=True)
    training = policy["training"]
    if (not tasks or not groups or not hashes or len(groups) > len(tasks) or len(hashes) > len(tasks)
            or tasks & set(training["development_task_ids"]) or groups & set(training["development_group_ids"])
            or hashes & set(training["development_prompt_sha256"])):
        raise ValueError("Report validation population must be fresh and disjoint from development.")
    if type(report.get("group_count")) is not int or report["group_count"] != len(groups):
        raise ValueError("Validation group count is inconsistent.")
    if not _HEX.fullmatch(str(report.get("input_digest", ""))):
        raise ValueError("Invalid validation input digest.")
    metrics = report.get("metrics")
    if type(metrics) is not dict or set(metrics) != {"baseline", "candidate", "relay"}:
        raise ValueError("Validation requires both fixed-model controls and Relay.")
    for metric in metrics.values():
        _validate_metric(metric, len(tasks), len(groups))
    for key in ("paired_losses", "paired_gains", "selected_candidate_count", "selected_baseline_count"):
        if _integer(report.get(key), key) > len(tasks):
            raise ValueError("Paired counts exceed the population.")
    baseline, relay = metrics["baseline"], metrics["relay"]
    if (report["selected_baseline_count"] + report["selected_candidate_count"] != len(tasks)
            or report["paired_losses"] > baseline["accepted_count"]
            or report["paired_gains"] > len(tasks) - baseline["accepted_count"]
            or report["paired_losses"] + report["paired_gains"] > report["selected_candidate_count"]
            or baseline["accepted_count"] - report["paired_losses"] + report["paired_gains"] != relay["accepted_count"]):
        raise ValueError("Paired quality counts are inconsistent.")
    reasons, saving = _gate(policy, report)
    if (report.get("reasons") != reasons or report.get("relative_savings") != saving
            or type(report.get("activation_eligible")) is not bool or report["activation_eligible"] != (not reasons)):
        raise ValueError("The report activation decision differs from its empirical gate.")
    if report.get("paired_loss_upper_95") != _loss_bound(report["paired_losses"], len(tasks)):
        raise ValueError("Paired uncertainty bound is inconsistent.")
    candidate = metrics["candidate"]
    candidate_saving = _savings(candidate, relay)
    expected_comparison = {
        "relative_savings_vs_candidate": candidate_saving,
        "acceptance_delta_vs_candidate": relay["acceptance_rate"] - candidate["acceptance_rate"],
        "incremental_advantage_demonstrated": candidate["cost_complete"] and relay["cost_complete"]
        and relay["acceptance_rate"] >= candidate["acceptance_rate"]
        and candidate_saving is not None and candidate_saving > 0,
    }
    if report.get("candidate_fixed_comparison") != expected_comparison:
        raise ValueError("The fixed-candidate comparison is inconsistent.")
    if report.get("net_savings_verified") is not False or report.get("router_cost_usd") is not None:
        raise ValueError("Unpriced router compute cannot certify net savings.")
    return report
