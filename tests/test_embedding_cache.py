import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from routerbench.embedding_cache import EmbeddingCache


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "nested/cache.sqlite3"
        self.cache = EmbeddingCache(self.path, "e5-revision-pooling-v1", dimension=3)

    def test_round_trip_reordering_and_exact_text_keys(self):
        self.cache.put_many([
            ("A\n B", np.array([1, 2, 3]), {"tokens": 4, "windows": 1}),
            ("a b", np.array([3, 2, 1]), {"tokens": 2}),
        ])
        cached = self.cache.get_many(["a b", "missing", "A\n B", "a b"])
        self.assertEqual(set(cached), {"a b", "A\n B"})
        self.assertNotEqual(self.cache.text_key("A\n B"), self.cache.text_key("a b"))
        np.testing.assert_array_equal(cached["A\n B"][0], [1, 2, 3])
        self.assertEqual(cached["a b"][0].dtype, np.dtype("float32"))
        self.assertEqual(cached["A\n B"][1], {"tokens": 4, "windows": 1})
        cached["a b"][0][0] = 999
        self.assertEqual(self.cache.get_many(["a b"])["a b"][0][0], 3)

    def test_namespace_invalidates_without_erasing_other_settings(self):
        self.cache.put_many([("same", np.array([1, 2, 3]), {})])
        other = EmbeddingCache(self.path, "different-windowing", dimension=3)
        self.assertEqual(other.get_many(["same"]), {})
        other.put_many([("same", np.array([4, 5, 6]), {})])
        self.assertEqual(self.cache.get_many(["same"])["same"][0][0], 1)
        self.assertEqual(other.get_many(["same"])["same"][0][0], 4)

    def test_raw_prompt_is_absent_from_storage(self):
        prompt = "CONFIDENTIAL unique prompt texto 12345"
        self.cache.put_many([(prompt, np.ones(3), {"tokens": 8})])
        with closing(sqlite3.connect(self.path)) as connection:
            schema = connection.execute("PRAGMA table_info(embeddings)").fetchall()
            stored = connection.execute("SELECT namespace, text_sha256, stats_json FROM embeddings").fetchone()
        self.assertNotIn("text", {column[1] for column in schema})
        self.assertEqual(stored[1], self.cache.text_key(prompt))
        self.assertNotIn(prompt, self.path.read_bytes().decode("utf-8", errors="ignore"))

    def test_corrupt_vector_and_stats_are_rejected(self):
        for field, replacement in [("vector", b"damaged"), ("stats_json", '{"tokens":99}')]:
            with self.subTest(field=field):
                self.cache.put_many([("text", np.ones(3), {"tokens": 1})])
                with closing(sqlite3.connect(self.path)) as connection:
                    connection.execute(f"UPDATE embeddings SET {field} = ?", (replacement,))
                    connection.commit()
                with self.assertRaises(ValueError):
                    self.cache.get_many(["text"])

    def test_invalid_values_do_not_partially_write_batch(self):
        invalid = [np.zeros(3), np.ones((1, 3)), np.array([1, np.nan, 2]), np.array([1, np.inf, 2])]
        for vector in invalid:
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    self.cache.put_many([("good", np.ones(3), {}), ("bad", vector, {})])
        self.assertEqual(self.cache.get_many(["good", "bad"]), {})
        with self.assertRaises(ValueError):
            self.cache.put_many([("bad stats", np.ones(3), {"duration": float("nan")})])

    def test_large_batch_and_empty_operations(self):
        self.assertEqual(self.cache.get_many([]), {})
        self.cache.put_many([])
        self.assertFalse(self.path.exists())
        records = [(f"prompt {index}", np.array([1, 2, 3]), {"index": index}) for index in range(901)]
        self.cache.put_many(records)
        self.assertEqual(len(self.cache.get_many([record[0] for record in records])), 901)

    def test_dimension_and_configuration_mismatches_are_explicit(self):
        with self.assertRaises(ValueError):
            EmbeddingCache(self.path, "", dimension=3)
        with self.assertRaises(ValueError):
            EmbeddingCache(self.path, "valid", dimension=True)
        self.cache.put_many([("text", np.ones(3), {})])
        different_dimension = EmbeddingCache(self.path, self.cache.namespace, dimension=4)
        with self.assertRaises(ValueError):
            different_dimension.get_many(["text"])


if __name__ == "__main__":
    unittest.main()
