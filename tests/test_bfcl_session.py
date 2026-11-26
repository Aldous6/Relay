from types import SimpleNamespace
import unittest

from relay_router.errors import BackendContextError, BackendTransportError
from routerbench.bfcl_harness import BFCLReply
from routerbench.bfcl_session import BFCLSessionGenerator


class PacedFixture:
    output, budget, tokens_per_minute = 100, 8000, 8000
    def __init__(self):
        self.generate = SimpleNamespace(request_input_estimate=lambda request: {"input_budget_units": request})
        self.calls = 0
        self.error = None
    def manifest(self): return {"kind": "fixture"}
    def __call__(self, model, request):
        if request + 612 > self.budget:
            raise BackendContextError("Minute allocation.")
        self.calls += 1
        if self.error:
            raise self.error
        return BFCLReply("done", (), {"input_tokens": 70, "output_tokens": 30, "total_tokens": 100})


class BFCLSessionTests(unittest.TestCase):
    def test_known_usage_and_per_model_reservation_stop_without_network(self):
        base = PacedFixture()
        session = BFCLSessionGenerator(base, max_tokens_per_model=750, max_provider_calls=20)
        before = session.manifest()
        session("small", 100)
        session("large", 100)
        self.assertEqual(session.tokens, {"small": 100, "large": 100})
        with self.assertRaises(BackendContextError) as caught:
            session("small", 100)
        self.assertFalse(caught.exception.usage["provider_metadata"]["provider_call_started"])
        self.assertEqual(base.calls, 2)
        self.assertEqual(session.manifest(), before)
        self.assertEqual(session.session_stop_reason()["reason"], "session_token_reservation_limit")

    def test_failed_call_retains_reservation_and_total_cap_stops(self):
        base = PacedFixture()
        base.error = BackendTransportError()
        session = BFCLSessionGenerator(base, max_tokens_per_model=900, max_provider_calls=1)
        with self.assertRaises(BackendTransportError):
            session("small", 100)
        self.assertEqual(session.tokens, {"small": 712})
        self.assertEqual(session.session_stop_reason()["reason"], "session_provider_call_limit")
        with self.assertRaises(BackendContextError):
            session("small", 1)
        self.assertEqual(base.calls, 1)

    def test_local_context_rejection_is_not_a_provider_call(self):
        base = PacedFixture()
        session = BFCLSessionGenerator(base, max_tokens_per_model=100000, max_provider_calls=20)
        with self.assertRaises(BackendContextError):
            session("small", 8000)
        self.assertEqual(session.provider_calls, 0)
        self.assertEqual(session.tokens, {})


if __name__ == "__main__":
    unittest.main()
