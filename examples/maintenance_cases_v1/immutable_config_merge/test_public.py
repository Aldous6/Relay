import unittest
from module import DELETE, merge_config


class PublicTests(unittest.TestCase):
    def test_nested_merge(self):
        self.assertEqual(merge_config({"db": {"host": "a", "port": 1}}, {"db": {"port": 2}}),
                         {"db": {"host": "a", "port": 2}})

    def test_deletion(self):
        self.assertEqual(merge_config({"obsolete": 1, "keep": 2}, {"obsolete": DELETE}), {"keep": 2})

    def test_list_replacement(self):
        self.assertEqual(merge_config({"tags": [1, 2]}, {"tags": [3]}), {"tags": [3]})
