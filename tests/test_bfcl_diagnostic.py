import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from diagnose_bfcl_request import safe_error_facts


class BFCLDiagnosticTests(unittest.TestCase):
    def test_only_selected_error_fields_are_retained_and_credentials_redacted(self):
        error = RuntimeError("not copied")
        error.body = {"error": {"code": "output_parse_failed", "type": "invalid_request_error",
            "message": "fixture-secret gsk_examplekey https://example.test/private",
            "failed_generation": "never copy this", "headers": {"Authorization": "never"}}}
        result = safe_error_facts(error, "fixture-secret")
        self.assertEqual(set(result[0]), {"code", "type", "message"})
        self.assertNotIn("fixture-secret", str(result))
        self.assertNotIn("gsk_examplekey", str(result))
        self.assertNotIn("example.test", str(result))
        self.assertNotIn("never", str(result))

    def test_exception_cycles_and_non_json_bodies_do_not_leak_exception_text(self):
        first, second = RuntimeError("secret first"), RuntimeError("secret second")
        first.__cause__, second.__cause__ = second, first
        first.body = "not JSON secret"
        second.body = '{"error":{"code":"output_parse_failed","message":"safe"}}'
        self.assertEqual(safe_error_facts(first, "fixture-secret"), [{"code": "output_parse_failed", "message": "safe"}])


if __name__ == "__main__":
    unittest.main()
