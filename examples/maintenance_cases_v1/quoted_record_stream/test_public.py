import unittest
from module import parse_records


class PublicTests(unittest.TestCase):
    def test_plain_chunks(self):
        self.assertEqual(parse_records(["a,b\n", "c,", "d"]), [["a", "b"], ["c", "d"]])

    def test_escaped_quote(self):
        self.assertEqual(parse_records(['"say ""hello""",x']), [['say "hello"', 'x']])

    def test_empty(self):
        self.assertEqual(parse_records([]), [])
