import unittest
from module import parse_records


class AcceptanceTests(unittest.TestCase):
    def test_every_boundary(self):
        source = '"a,b","line1\r\nline2",\r\n"say ""hi""", x\n'
        expected = [["a,b", "line1\r\nline2", ""], ['say "hi"', " x"]]
        for split in range(len(source) + 1):
            self.assertEqual(parse_records(iter([source[:split], "", source[split:]])), expected)
        self.assertEqual(parse_records(iter(source)), expected)

    def test_empty_fields_terminators(self):
        for source, expected in [("", []), ("\n", [[""]]), ("\r\n\n", [[""], [""]]),
                                 (",", [["", ""]]), ('""', [[""]]), ("a,\n", [["a", ""]]),
                                 ("a\n", [["a"]]), ('"a\rb"', [["a\rb"]])]:
            with self.subTest(source=source):
                self.assertEqual(parse_records([source]), expected)

    def test_strict_syntax(self):
        for source in ['a"b', '"a"x', '"a" ', '"open', "x\r", "x\ry", '"a"\r']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_records(iter(source))

    def test_custom_and_invalid_delimiter(self):
        self.assertEqual(parse_records(['"x|y"| z|'], "|"), [["x|y", " z", ""]])
        for delimiter in ["", "xx", '"', "\r", "\n", None]:
            with self.subTest(delimiter=delimiter), self.assertRaises(ValueError):
                parse_records([], delimiter)
