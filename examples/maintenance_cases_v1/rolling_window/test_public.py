import unittest
from module import WindowStats


class PublicTests(unittest.TestCase):
    def test_accumulation(self):
        stats = WindowStats(5)
        self.assertEqual(stats.add(1, 3), (1, 3))
        self.assertEqual(stats.add(2, 4), (2, 7))

    def test_boundary(self):
        stats = WindowStats(5)
        stats.add(1, 3)
        self.assertEqual(stats.add(6, 4), (1, 4))

    def test_empty(self):
        self.assertEqual(WindowStats(2).snapshot(0), (0, 0))
