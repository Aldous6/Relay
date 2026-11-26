"""Utilities for subtracting unavailable spans from reservations."""


def _ordered(spans):
    """Prepare spans for the sweep."""
    ordered = []
    for start, end in spans:
        if start > end:
            raise ValueError("reversed span")
        ordered.append((start, end))
    return sorted(ordered)


def _cut_one(span, cut):
    start, end = span
    left, right = cut
    if right <= start or left >= end:
        return [span]
    result = []
    if start < left:
        result.append((start, left))
    if right < end:
        result.append((right, end))
    return result


def subtract_intervals(intervals, blocked):
    """Return available pieces of every input span."""
    result = []
    for span in _ordered(intervals):
        pieces = [span]
        for cut in blocked:
            next_pieces = []
            for piece in pieces:
                next_pieces.extend(_cut_one(piece, cut))
            pieces = next_pieces
        result.extend(pieces)
    return result
