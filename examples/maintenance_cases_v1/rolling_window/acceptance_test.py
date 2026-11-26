import random
import unittest
from module import WindowStats


class AcceptanceTests(unittest.TestCase):
    def test_snapshot_expiry_and_time_contract(self):
        stats = WindowStats(3)
        stats.add(-2, -7)
        stats.add(-2, 4)
        self.assertEqual(stats.snapshot(0), (2, -3))
        self.assertEqual(stats.snapshot(1), (0, 0))
        with self.assertRaises(ValueError):
            stats.add(0, 10)
        self.assertEqual(stats.add(1, 2), (1, 2))

    def test_invalid_calls_atomic(self):
        for width in [True, 1.5, 0, -1, "3"]:
            with self.subTest(width=width), self.assertRaises(ValueError):
                WindowStats(width)
        stats = WindowStats(20)
        stats.add(10, 5)
        invalid = [lambda: stats.add(9, 100), lambda: stats.add(100, 2.5),
                   lambda: stats.add(True, 1), lambda: stats.snapshot(10.5),
                   lambda: stats.snapshot(9), lambda: stats.add(11, False)]
        for operation in invalid:
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(stats.snapshot(10), (1, 5))

    def test_repeated_times_and_independent_instances(self):
        left, right = WindowStats(2), WindowStats(2)
        self.assertEqual(left.add(2, 0), (1, 0))
        self.assertEqual(left.add(2, -1), (2, -1))
        self.assertEqual(right.snapshot(2), (0, 0))
        self.assertEqual(left.snapshot(4), (0, 0))

    def test_deterministic_stream(self):
        rng = random.Random(47)
        stats, events, now = WindowStats(7), [], -100
        for i in range(100):
            now += rng.randrange(4)
            if i % 3:
                value = rng.randrange(-12, 13)
                events.append((now, value))
                actual = stats.add(now, value)
            else:
                actual = stats.snapshot(now)
            live = [value for stamp, value in events if now - 7 < stamp <= now]
            self.assertEqual(actual, (len(live), sum(live)))
