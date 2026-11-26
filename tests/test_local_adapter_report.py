"""Shadow reconstruction must preserve identity, decisions and failures."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("local_adapter_report", ROOT / "tools/report_local_adapter.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class LocalAdapterReportTests(unittest.TestCase):
    def setUp(self):
        digest = hashlib.sha256(b"question").hexdigest()
        self.rows = [{"id": "a", "prompt_sha256": digest, "reference": "17", "grade": "exact"}]
        self.routes = [{"id": "a", "prompt_sha256": digest, "routes": {"candidate": "weak", "reference": "weak"}}]
        self.shadow = {"threshold": .5, "routes": [{"id": "a", "prompt_sha256": digest,
                       "model": "strong", "probability": .8, "adapted_route": "weak"}]}
        self.shadow["payload_sha256"] = REPORT._hash(self.shadow["routes"])
        self.key = REPORT._hash(["a", digest, "strong"])
        self.responses = {self.key: {"id": "a", "model": "strong", "prompt_sha256": digest,
                                   "status": "ok", "content": " 17 "}}

    def test_published_decision_reuses_its_actual_response_not_adapted_action(self):
        result = REPORT.shadow_outcomes(self.shadow, self.rows, self.routes, self.responses)
        self.assertEqual(result[0]["model"], "strong")
        self.assertTrue(result[0]["success"])
        self.assertEqual(result[0]["response_key"], self.key)
        self.responses[self.key]["status"] = "error"
        self.assertFalse(REPORT.shadow_outcomes(self.shadow, self.rows, self.routes, self.responses)[0]["success"])

    def test_rejects_reordered_identity_or_threshold_inconsistent_route(self):
        for field, value in [("id", "other"), ("model", "weak"), ("probability", float("inf")), ("adapted_route", "strong")]:
            shadow = copy.deepcopy(self.shadow)
            shadow["routes"][0][field] = value
            if field == "probability":
                # JSON hashing already rejects non-finite values.
                with self.assertRaises(ValueError):
                    REPORT._hash(shadow["routes"])
                continue
            shadow["payload_sha256"] = REPORT._hash(shadow["routes"])
            with self.assertRaises(ValueError):
                REPORT.shadow_outcomes(shadow, self.rows, self.routes, self.responses)

    def test_rejects_tampered_payload_and_cross_case_response(self):
        shadow = copy.deepcopy(self.shadow)
        shadow["routes"][0]["probability"] = .9
        with self.assertRaises(ValueError):
            REPORT.shadow_outcomes(shadow, self.rows, self.routes, self.responses)
        self.responses[self.key]["id"] = "other"
        with self.assertRaises(ValueError):
            REPORT.shadow_outcomes(self.shadow, self.rows, self.routes, self.responses)


if __name__ == "__main__":
    unittest.main()
