import unittest
from module import GenerationCache


class PublicTests(unittest.TestCase):
    def test_publish(self):
        cache = GenerationCache()
        self.assertTrue(cache.commit("a", cache.begin("a"), 7))
        self.assertEqual(cache.get("a"), 7)

    def test_token_consumed(self):
        cache = GenerationCache()
        token = cache.begin("a")
        self.assertTrue(cache.commit("a", token, 1))
        self.assertFalse(cache.commit("a", token, 2))
        self.assertEqual(cache.get("a"), 1)

    def test_default(self):
        self.assertEqual(GenerationCache().get("missing", 42), 42)
