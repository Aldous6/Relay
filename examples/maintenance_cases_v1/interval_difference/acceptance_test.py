import copy
import random
import unittest
from module import subtract_intervals


class AcceptanceTests(unittest.TestCase):
    def test_touching_empty_nested(self):
        self.assertEqual(subtract_intervals([(1, 2), (2, 8), (4, 4)], [(2, 3), (3, 5)]), [(1, 2), (5, 8)])
        self.assertEqual(subtract_intervals([(-9, 9), (0, 1)], [(-20, 20)]), [])
        self.assertEqual(subtract_intervals([(1, 1)], []), [])

    def test_iterators_and_immutability(self):
        source, cuts = [[0, 5], [9, 12]], [[2, 3], [10, 11]]
        original = copy.deepcopy((source, cuts))
        self.assertEqual(subtract_intervals(iter(source), iter(cuts)), [(0, 2), (3, 5), (9, 10), (11, 12)])
        self.assertEqual((source, cuts), original)

    def test_validation_not_short_circuited(self):
        for invalid in [[(3, 1)], [(True, 2)], [(1, 2.0)]]:
            for args in [(invalid, []), ([], invalid)]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    subtract_intervals(*args)

    def test_integer_coverage_and_canonical_result(self):
        rng = random.Random(173)
        for _ in range(60):
            source = [tuple(sorted((rng.randrange(-8, 9), rng.randrange(-8, 9)))) for _ in range(6)]
            cuts = [tuple(sorted((rng.randrange(-8, 9), rng.randrange(-8, 9)))) for _ in range(4)]
            expected = {x for a, b in source for x in range(a, b)} - {x for a, b in cuts for x in range(a, b)}
            result = subtract_intervals(source, cuts)
            self.assertEqual({x for a, b in result for x in range(a, b)}, expected)
            self.assertTrue(all(a < b for a, b in result))
            self.assertTrue(all(result[i][1] < result[i + 1][0] for i in range(len(result) - 1)))
