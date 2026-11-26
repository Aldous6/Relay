"""Bound one explicitly authorized BFCL session without provider retries.

Token reservations are local estimates, not the provider's remaining daily quota.
An interrupted provider call remains unknown and is never automatically reissued.
"""
from copy import deepcopy

from relay_router.errors import BackendContextError, LiveBackendError
from .agent_pacing import _known_total


class BFCLSessionGenerator:
    def __init__(self, generate, *, max_tokens_per_model, max_provider_calls):
        for value in (max_tokens_per_model, max_provider_calls):
            if type(value) is not int or value < 1:
                raise ValueError("Session limits must be explicit positive integers.")
        self.generate = generate
        self.max_tokens = max_tokens_per_model
        self.max_calls = max_provider_calls
        self.tokens = {}
        self.provider_calls = 0
        self._stopped = None

    def manifest(self):
        return {"kind": "bfcl_session_budget", "generate": self.generate.manifest(),
            "max_tokens_per_model": self.max_tokens, "max_provider_calls": self.max_calls,
            "token_accounting": "local input estimate plus output reservation and reserve; measured total replaces reservation when available",
            "unknown_usage": "retain full local reservation; actual provider consumption remains unknown",
            "scope": "one process/session; not remaining account quota, server token guarantee or billing limit"}

    def session_stop_reason(self):
        if self._stopped is not None:
            return deepcopy(self._stopped)
        if self.provider_calls >= self.max_calls:
            return {"reason": "session_provider_call_limit", "limit": self.max_calls}
        if any(value >= self.max_tokens for value in self.tokens.values()):
            return {"reason": "session_token_limit", "limit": self.max_tokens}
        return None

    def __call__(self, model, request):
        stop = self.session_stop_reason()
        estimate = self.generate.generate.request_input_estimate(request)
        reservation = estimate["input_budget_units"] + self.generate.output + 512
        if stop is None and self.tokens.get(model, 0) + reservation > self.max_tokens:
            stop = {"reason": "session_token_reservation_limit", "model": model,
                    "limit": self.max_tokens, "accounted_tokens": self.tokens.get(model, 0),
                    "next_reservation": reservation}
        if stop is not None:
            self._stopped = stop
            raise BackendContextError("Local session allocation reached; no provider call.",
                usage={"provider_metadata": {"provider_call_started": False, "session_stop": deepcopy(stop)}})
        # A request outside the minute/context allocation is rejected locally by
        # the paced adapter, and must not consume a provider call allocation.
        if reservation > self.generate.budget or (self.generate.tokens_per_minute is not None
                and reservation > self.generate.tokens_per_minute):
            return self.generate(model, request)
        self.provider_calls += 1
        self.tokens[model] = self.tokens.get(model, 0) + reservation
        try:
            reply = self.generate(model, request)
        except LiveBackendError as error:
            total = _known_total(error.usage)
            if total is not None:
                self.tokens[model] += total - reservation
            raise
        total = _known_total(reply.usage)
        if total is not None:
            self.tokens[model] += total - reservation
        return reply

