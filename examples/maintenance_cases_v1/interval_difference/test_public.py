import unittest
from module import subtract_intervals


class PublicTests(unittest.TestCase):
    def test_cut(self):
        self.assertEqual(subtract_intervals([(1, 9)], [(3, 5)]), [(1, 3), (5, 9)])

    def test_union(self):
        self.assertEqual(subtract_intervals([(3, 8), (1, 4)], []), [(1, 8)])

    def test_empty(self):
        self.assertEqual(subtract_intervals([], []), [])
