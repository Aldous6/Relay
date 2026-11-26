"""Small prompt-selector interface; inference is supplied by an optional engine.

The core never generates an answer, downloads a model or evaluates quality.
A preference score is not a calibrated probability of task success. Decisions
remain experimental and local selector compute has no assumed monetary price.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
import re
import time
from typing import Callable


def _identifier(value, name):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}", value):
        raise ValueError(f"{name} must be a nonempty identifier, not free text.")
    return value


def _score(value, name):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and within [0, 1].")
    return float(value)


def _tokens(value, name):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None.")
    return value


@dataclass(frozen=True, kw_only=True)
class PairwisePromptSelector:
    """Select once from two exact model names using a resident scorer.

    ``scorer(text)`` returns ``score`` and optional ``input_tokens`` and
    ``truncated_tokens``. Unknown extra fields are never copied into a decision.
    Scores at or above ``threshold`` prefer strong. Reported truncation always
    retains strong because the selector did not inspect the complete input.
    This conservative rule does not certify the strong model's capabilities.
    """

    weak_model: str
    strong_model: str
    threshold: float
    scorer: Callable = field(repr=False, compare=False)
    selector_id: str

    def __post_init__(self):
        _identifier(self.weak_model, "weak_model")
        _identifier(self.strong_model, "strong_model")
        _identifier(self.selector_id, "selector_id")
        if self.weak_model == self.strong_model:
            raise ValueError("weak_model and strong_model must differ.")
        object.__setattr__(self, "threshold", _score(self.threshold, "threshold"))
        if not callable(self.scorer) or inspect.iscoroutinefunction(self.scorer):
            raise TypeError("scorer must be a synchronous callable.")

    def select(self, routing_text: str) -> dict:
        if type(routing_text) is not str or not routing_text.strip():
            raise ValueError("routing_text must be nonempty text.")
        started = time.perf_counter()
        scored = self.scorer(routing_text)
        if inspect.isawaitable(scored):
            if inspect.iscoroutine(scored):
                scored.close()
            raise TypeError("scorer must return a synchronous result.")
        if type(scored) is not dict or "score" not in scored:
            raise ValueError("scorer must return a dictionary with score.")
        score = _score(scored["score"], "score")
        incoming = _tokens(scored.get("input_tokens"), "input_tokens")
        truncated = _tokens(scored.get("truncated_tokens"), "truncated_tokens")
        if incoming is not None and truncated is not None and truncated > incoming:
            raise ValueError("truncated_tokens cannot exceed input_tokens.")
        if truncated is not None and truncated > 0:
            model, reason = self.strong_model, "strong_after_truncation"
        elif score >= self.threshold:
            model, reason = self.strong_model, "strong_score_at_or_above_threshold"
        else:
            model, reason = self.weak_model, "weak_score_below_threshold"
        return {"model": model, "selector_id": self.selector_id, "preference_score": score,
                "threshold": self.threshold, "routing_seconds": max(0., time.perf_counter() - started),
                "input_tokens": incoming, "truncated_tokens": truncated, "reason": reason,
                "status": "experimental", "router_cost_usd": None}
