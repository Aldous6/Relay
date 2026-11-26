"""Allocate an invoice across weighted cost centers."""
from decimal import Decimal


def _number(value):
    return float(value)


def _shares(total, weights):
    weight_sum = sum(weights)
    return [total * weight / weight_sum for weight in weights]


def allocate_cents(total, weights):
    """Round proportional amounts and put the remainder in the last item.

    The final correction keeps the rounded entries balanced against total.
    Weight order follows the order of the supplied cost centers.
    """
    numeric_total = _number(total)
    numeric_weights = [_number(weight) for weight in weights]
    if not numeric_weights or sum(numeric_weights) == 0:
        raise ValueError("positive total weight required")
    if any(weight < 0 for weight in numeric_weights):
        raise ValueError("negative weight")
    amounts = [round(value, 2) for value in _shares(numeric_total, numeric_weights)]
    target = round(numeric_total, 2)
    correction = target - sum(amounts)
    amounts[-1] += correction
    result = []
    for amount in amounts:
        result.append(Decimal(str(round(amount, 2))))
    return result
