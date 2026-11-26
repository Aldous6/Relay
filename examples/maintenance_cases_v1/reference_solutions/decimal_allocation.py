from decimal import Decimal, InvalidOperation
from fractions import Fraction


def _number(value):
    if not isinstance(value, (str, Decimal)):
        raise ValueError("decimal or string required")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError("invalid decimal") from None
    if not parsed.is_finite():
        raise ValueError("nonfinite decimal")
    return Fraction(parsed)


def allocate_cents(total, weights):
    amount = _number(total)
    parsed = [_number(weight) for weight in weights]
    if not parsed or any(weight < 0 for weight in parsed) or not sum(parsed):
        raise ValueError("invalid weights")
    cents = round(amount * 100)
    total_weight = sum(parsed)
    shares = [abs(cents) * weight / total_weight for weight in parsed]
    floors = [share.numerator // share.denominator for share in shares]
    order = sorted(range(len(parsed)), key=lambda i: (-(shares[i] - floors[i]), i))
    for i in order[:abs(cents) - sum(floors)]:
        floors[i] += 1
    sign = int(cents < 0)
    return [Decimal((sign, tuple(map(int, str(value))), -2)) for value in floors]
