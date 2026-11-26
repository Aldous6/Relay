import unittest
from module import dependency_order


class PublicTests(unittest.TestCase):
    def test_chain(self):
        self.assertEqual(dependency_order({"c": ["b"], "b": ["a"]}), ["a", "b", "c"])

    def test_ready_tie(self):
        self.assertEqual(dependency_order({"a": ["z"], "b": []}), ["b", "z", "a"])

    def test_empty(self):
        self.assertEqual(dependency_order({}), [])
