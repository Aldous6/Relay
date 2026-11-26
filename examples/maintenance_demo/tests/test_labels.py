import unittest

from labels import normalize_labels


class NormalizeLabelsTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(normalize_labels([]), [])

    def test_whitespace_case_and_empty_labels(self):
        self.assertEqual(normalize_labels([" Python ", "", "PYTHON", "  ", "Relay"]), ["python", "relay"])

    def test_first_occurrence_order(self):
        self.assertEqual(normalize_labels(["B", "A", "b", "C", "a"]), ["b", "a", "c"])

    def test_unicode_casefold(self):
        self.assertEqual(normalize_labels(["Straße", "STRASSE"]), ["strasse"])

    def test_input_not_modified(self):
        original = [" Relay ", "RELAY"]
        normalize_labels(original)
        self.assertEqual(original, [" Relay ", "RELAY"])


if __name__ == "__main__":
    unittest.main()
