import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from module import allocate_cents


class AcceptanceTests(unittest.TestCase):
    def test_half_even_negative_and_zero_weights(self):
        self.assertEqual(allocate_cents("1.005", ["1"]), [Decimal("1.00")])
        self.assertEqual(allocate_cents("1.015", ["1"]), [Decimal("1.02")])
        self.assertEqual(allocate_cents("-0.05", ["1", "1", "1"]), [Decimal("-0.02"), Decimal("-0.02"), Decimal("-0.01")])
        self.assertEqual(allocate_cents("0.03", ["0", "2", "1", "0"]),
                         [Decimal("0.00"), Decimal("0.02"), Decimal("0.01"), Decimal("0.00")])

    def test_non_tied_remainders(self):
        self.assertEqual(allocate_cents("0.07", ["1", "2", "3"]), [Decimal("0.01"), Decimal("0.02"), Decimal("0.04")])

    def test_precision_independent_exactness_and_exponents(self):
        weights = [Decimal("0.1000000000000000000000000001"), Decimal("0.1")]
        with localcontext() as context:
            context.prec = 4
            result = allocate_cents(Decimal("12345678901234567890.125"), iter(weights))
        self.assertEqual(sum(map(Fraction, result)), Fraction("12345678901234567890.12"))
        self.assertTrue(all(type(value) is Decimal and value.as_tuple().exponent == -2 for value in result))
        self.assertEqual(allocate_cents("0.01", weights), [Decimal("0.01"), Decimal("0.00")])
        self.assertEqual(weights, [Decimal("0.1000000000000000000000000001"), Decimal("0.1")])

    def test_invalid_values(self):
        for total in ["NaN", "Infinity", "bad", 1.0, 1, None]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                allocate_cents(total, ["1"])
        for weights in [["-1", "2"], ["0", "0"], ["NaN"], ["Infinity"], ["bad"], [1], [True]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                allocate_cents("1", weights)

    def test_small_integer_invariant(self):
        for cents in range(-21, 22):
            result = allocate_cents(str(Decimal(cents) / 100), ["3", "2", "5"])
            self.assertEqual(sum(result), Decimal(cents) / 100)
            shares = [Fraction(abs(cents) * w, 10) for w in (3, 2, 5)]
            actual = [abs(Fraction(value)) * 100 for value in result]
            self.assertTrue(all(share.numerator // share.denominator <= value <= -(-share.numerator // share.denominator)
                                for share, value in zip(shares, actual)))
