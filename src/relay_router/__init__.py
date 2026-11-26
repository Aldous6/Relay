"""Provider-independent Relay routing for Python harnesses.

The workflow runtime uses only the standard library. Historical learned
bundles keep their existing public imports, loading scientific dependencies
only when those imports are requested.
"""
from importlib import import_module

__all__ = ["RelayRouter", "RoutingDecision", "load_bundle", "Relay", "TaskSession", "load_adapted_relay",
           "SpendBudget", "run_cascade", "CheckCommand", "CommandVerifier"]

_EXPORT_MODULES = {
    "RelayRouter": ".client",
    "RoutingDecision": ".client",
    "load_bundle": ".client",
    "Relay": ".workflow",
    "TaskSession": ".workflow",
    "load_adapted_relay": ".adaptive_runtime",
    "SpendBudget": ".cascade",
    "run_cascade": ".cascade",
    "CheckCommand": ".cascade_verifier",
    "CommandVerifier": ".cascade_verifier",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
