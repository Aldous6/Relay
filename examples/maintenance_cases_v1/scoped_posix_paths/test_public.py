import unittest
from module import resolve_path


class PublicTests(unittest.TestCase):
    def test_relative_canonicalization(self):
        self.assertEqual(resolve_path("/srv/app", "a/./b/../c"), "/srv/app/a/c")

    def test_sibling_prefix_rejected(self):
        with self.assertRaises(ValueError):
            resolve_path("/srv/app", "/srv/application/config")

    def test_empty_path(self):
        self.assertEqual(resolve_path("/srv/app/", ""), "/srv/app")
