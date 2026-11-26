"""Fourteen deterministic, prompt-only features.

These are operational definitions for this experiment, not language detectors or
tokenizer counts. Whitespace is preserved for feature extraction. Only the
separate duplicate key normalizes text.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

FEATURE_NAMES = (
    "n_chars",
    "n_words",
    "n_lines",
    "avg_word_length",
    "lexical_diversity",
    "digit_ratio",
    "punctuation_ratio",
    "uppercase_ratio",
    "n_questions",
    "n_brackets",
    "n_code_markers",
    "n_math_symbols",
    "non_ascii_ratio",
    "cjk_ratio",
)

# Unicode \w includes digits and underscores. Words are case-folded only when
# measuring lexical diversity, while their original lengths are retained.
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
CODE_PATTERN = re.compile(
    r"```|=>|::|#include\b|"
    r"\b(?:def|class|import|from|function|const|let|var|return|"
    r"public|private|void|SELECT|INSERT|UPDATE|DELETE)\b"
)
MATH_SYMBOLS = frozenset("+-−*/=<>^%√∑∏∫∞≈≠≤≥±×÷∂∇∈∉∩∪")
QUESTION_SYMBOLS = frozenset("?？¿")
BRACKET_SYMBOLS = frozenset("()[]{}")
# Fixed script-block heuristic: Han, hiragana, katakana and hangul ranges.
# This measures character structure, not the prompt's language.
CJK_RANGES = (
    (0x1100, 0x11FF),
    (0x3040, 0x30FF),
    (0x3130, 0x318F),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xD7B0, 0xD7FF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9D),
    (0x20000, 0x2FA1F),
    (0x30000, 0x323AF),
)


def parse_prompt(value: str) -> str:
    """Reconstruct serialized lists of strings with newline separators.

    Plain strings, including code or malformed list literals, remain unchanged.
    ``ast.literal_eval`` interprets literals only and never executes expressions.
    """
    if not isinstance(value, str):
        raise TypeError("A prompt must be a string.")
    candidate = value.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return "\n".join(parsed)
    return value


def normalize_for_dedup(prompt: str) -> str:
    """Return a conservative duplicate key, never a feature-extraction input.

    The rule is NFKC, Unicode casefold and collapsed whitespace. It can merge
    case-sensitive code variants, so duplicate removal must be reported.
    """
    if not isinstance(prompt, str):
        raise TypeError("A prompt must be a string.")
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def _feature_row(text: str) -> dict[str, int | float]:
    if not isinstance(text, str):
        raise TypeError("Feature inputs must be parsed prompt strings.")
    words = WORD_PATTERN.findall(text)
    n_chars = len(text)
    denominator = max(n_chars, 1)
    n_words = len(words)
    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_lines": len(text.splitlines()),
        "avg_word_length": sum(map(len, words)) / max(n_words, 1),
        "lexical_diversity": len({word.casefold() for word in words}) / max(n_words, 1),
        "digit_ratio": sum(char.isdigit() for char in text) / denominator,
        "punctuation_ratio": sum(
            unicodedata.category(char).startswith("P") for char in text
        ) / denominator,
        "uppercase_ratio": sum(char.isupper() for char in text) / denominator,
        "n_questions": sum(char in QUESTION_SYMBOLS for char in text),
        "n_brackets": sum(char in BRACKET_SYMBOLS for char in text),
        "n_code_markers": len(CODE_PATTERN.findall(text)),
        "n_math_symbols": sum(char in MATH_SYMBOLS for char in text),
        "non_ascii_ratio": sum(ord(char) > 127 for char in text) / denominator,
        "cjk_ratio": sum(
            any(start <= ord(char) <= end for start, end in CJK_RANGES)
            for char in text
        ) / denominator,
    }


def extract_features(prompts: Iterable[str]) -> pd.DataFrame:
    """Build the fixed predictor allowlist from already parsed prompts.

    Ratios use total character count. Empty strings have zero-valued features.
    Lines use ``str.splitlines`` (a trailing newline creates no extra line).
    No score, cost, response, identifier or benchmark is accepted as a feature.
    """
    if isinstance(prompts, str):
        raise TypeError("Pass an iterable of prompts, not one string.")
    return pd.DataFrame.from_records(
        (_feature_row(prompt) for prompt in prompts), columns=FEATURE_NAMES
    ).astype(float)
