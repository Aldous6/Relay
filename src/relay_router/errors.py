"""Sanitized provider failures with separate infrastructure and action scopes."""
from copy import deepcopy
from math import isfinite


class LiveBackendError(RuntimeError):
    category = "provider_unknown"
    retryable = False

    def __init__(self, message="Fallo del proveedor.", *, status_code=None,
                 retry_after_seconds=None, usage=None, diagnostic=None):
        super().__init__(message)
        self.status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None
        self.retry_after_seconds = (float(retry_after_seconds)
            if type(retry_after_seconds) in (int, float) and isfinite(retry_after_seconds)
            and retry_after_seconds >= 0 else None)
        self.usage = deepcopy(usage) if type(usage) is dict else {}
        self.diagnostic = deepcopy(diagnostic) if type(diagnostic) is dict else {}

    def details(self):
        # Never persist message, URL, body, headers, credentials or traceback.
        return {"type": type(self).__name__, "category": self.category,
                "retryable": self.retryable, "status_code": self.status_code,
                "retry_after_seconds": self.retry_after_seconds,
                **({"diagnostic": deepcopy(self.diagnostic)} if self.diagnostic else {})}


class BackendTransportError(LiveBackendError):
    category = "transport"
    retryable = True


class BackendTimeoutError(BackendTransportError):
    category = "timeout"


class BackendRateLimitError(LiveBackendError):
    category = "rate_limit"
    retryable = True


class BackendHTTPError(LiveBackendError):
    category = "http"

    @property
    def retryable(self):
        return self.status_code in {408, 409, 425, 500, 502, 503, 504}


class BackendResponseError(LiveBackendError):
    category = "provider_contract"


class BackendActionError(LiveBackendError):
    """Provider rejected generated JSON/tool output, not the request transport.

    No infrastructure retry: the harness records a semantic failure and lets
    the routing policy select the model for the next bounded attempt.
    """

    category = "model_action"


class BackendIdentityError(LiveBackendError):
    category = "model_identity"


class BackendContextError(LiveBackendError):
    category = "context_capacity"
