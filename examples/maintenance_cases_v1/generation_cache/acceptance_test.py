import unittest
from module import GenerationCache


class AcceptanceTests(unittest.TestCase):
    def test_out_of_order_and_old_value_retained(self):
        cache = GenerationCache()
        cache.commit("x", cache.begin("x"), "old")
        first, second = cache.begin("x"), cache.begin("x")
        self.assertEqual(cache.get("x"), "old")
        self.assertFalse(cache.commit("x", first, "stale"))
        self.assertTrue(cache.commit("x", second, "fresh"))
        self.assertEqual(cache.get("x"), "fresh")

    def test_no_aba_after_invalidate_or_clear(self):
        cache = GenerationCache()
        old = cache.begin("x")
        cache.invalidate("x")
        new = cache.begin("x")
        self.assertNotEqual(old, new)
        self.assertFalse(cache.commit("x", old, 1))
        cache.clear()
        last = cache.begin("x")
        self.assertNotIn(last, (old, new))
        self.assertFalse(cache.commit("x", new, 2))
        self.assertTrue(cache.commit("x", last, 3))

    def test_clear_pending_only_and_cross_key_tokens(self):
        cache = GenerationCache()
        a, b = cache.begin("a"), cache.begin("b")
        self.assertIs(type(a), int)
        self.assertGreater(a, 0)
        self.assertNotEqual(a, b)
        self.assertFalse(cache.commit("a", b, 7))
        self.assertTrue(cache.commit("a", a, None))
        self.assertIsNone(cache.get("a", "missing"))
        cache.clear()
        self.assertFalse(cache.commit("b", b, 2))
        self.assertEqual(cache.get("a", "missing"), "missing")

    def test_invalidation_is_per_key_and_failures_do_not_consume(self):
        cache, other = GenerationCache(), GenerationCache()
        a, b = cache.begin("a"), cache.begin("b")
        cache.invalidate("a")
        self.assertFalse(cache.commit("a", a, 1))
        self.assertFalse(cache.commit("b", -10, 2))
        self.assertTrue(cache.commit("b", b, 3))
        self.assertIsNone(other.get("b"))
        self.assertFalse(cache.commit("never", 900, 4))
