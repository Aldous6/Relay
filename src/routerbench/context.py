"""Tokenizer-independent windows with explicit, complete input coverage.

Tokenization, special-token reservation, roles and learned aggregation belong to
the caller. These windows do not establish cross-window semantic understanding.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class TokenWindow:
    start: int
    end: int
    token_ids: tuple[int, ...]


def window_token_ids(token_ids: Sequence[int], max_tokens: int, overlap: int = 0) -> list[TokenWindow]:
    """Return half-open token windows, covering every input position.

    ``max_tokens`` is the payload capacity after the caller reserves the
    encoder's special tokens and any prefix. Empty input produces no windows.
    Indices describe positions in the original sequence, not decoded characters.
    """
    for name, value in (("max_tokens", max_tokens), ("overlap", overlap)):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"{name} debe ser un entero.")
    if max_tokens <= 0 or not 0 <= overlap < max_tokens:
        raise ValueError("Se requiere max_tokens > 0 y 0 <= overlap < max_tokens.")
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError("token_ids debe ser una secuencia de enteros.")
    if any(not isinstance(token, Integral) or isinstance(token, bool) or token < 0 for token in token_ids):
        raise ValueError("Cada token debe ser un entero no negativo.")
    tokens = tuple(int(token) for token in token_ids)
    windows = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        windows.append(TokenWindow(start, end, tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return windows
