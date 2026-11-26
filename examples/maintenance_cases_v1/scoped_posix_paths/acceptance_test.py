import unittest
from module import resolve_path


class AcceptanceTests(unittest.TestCase):
    def test_normalized_roots_and_absolute_reentry(self):
        self.assertEqual(resolve_path("//srv/old/../app/.", "//srv/app/a//b/"), "/srv/app/a/b")
        self.assertEqual(resolve_path("/srv/app", "/srv/app/../app/x"), "/srv/app/x")
        self.assertEqual(resolve_path("/srv/app", "/srv/app/"), "/srv/app")
        self.assertEqual(resolve_path("/", "a/../"), "/")

    def test_relative_escape_rejected_at_boundary(self):
        for path in ["..", "../app/x", "a/../../app/x", "./../app"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolve_path("/srv/app", path)

    def test_above_root_nul_and_nonabsolute_root(self):
        for root, path in [("relative", "a"), ("/../../srv", "a"), ("/", "../x"),
                           ("/", "/../x"), ("/a\0b", "x"), ("/a", "x\0y"), ("/a", "/other")]:
            with self.subTest(root=root, path=path), self.assertRaises(ValueError):
                resolve_path(root, path)

    def test_literal_backslashes_and_percent(self):
        self.assertEqual(resolve_path("/app", r"..\secret"), r"/app/..\secret")
        self.assertEqual(resolve_path("/app", "%2e%2e/%2f/x"), "/app/%2e%2e/%2f/x")

    def test_canonical_result_is_idempotent(self):
        for root, path in [("/", "a//b/../c"), ("/a/b", "./x/../y//"), ("//a//", "b/c/..")]:
            result = resolve_path(root, path)
            self.assertEqual(resolve_path(root, result), result)
            self.assertFalse("//" in result)
            self.assertTrue(result == "/" or not result.endswith("/"))
