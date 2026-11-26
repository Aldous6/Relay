import unittest
from decimal import Decimal
from module import allocate_cents


class PublicTests(unittest.TestCase):
    def test_even_shares(self):
        self.assertEqual(allocate_cents("10.00", ["1", "1"]), [Decimal("5.00"), Decimal("5.00")])

    def test_tied_remainders(self):
        self.assertEqual(allocate_cents("0.01", ["1", "1", "1"]),
                         [Decimal("0.01"), Decimal("0.00"), Decimal("0.00")])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            allocate_cents("1.00", [])
