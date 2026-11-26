"""Explicit, offline episode contracts; model/tool execution belongs to the harness.

Requests expose only the current prompt, past observations and current state.
An evaluator's reference answer or future outcomes are not request fields.
Frozen dataclasses prevent field reassignment; nested JSON containers remain
mutable, so constructors and snapshot methods copy them at callback boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from numbers import Real
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import RelayRouter


def _json_copy(value: Any, ancestors: set[int] | None = None) -> Any:
    """Copy JSON-compatible values without coercing object keys or nonfinite numbers."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("JSON numbers must be finite.")
        return value
    if type(value) not in (dict, list, tuple):
        raise ValueError("Values must contain only JSON-compatible data.")
    ancestors = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in ancestors:
        raise ValueError("JSON data cannot contain cycles.")
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ValueError("JSON object keys must be strings.")
            return {key: _json_copy(item, ancestors) for key, item in value.items()}
        return [_json_copy(item, ancestors) for item in value]
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep.") from exc
    finally:
        ancestors.remove(identity)


def _names(values: tuple[str, ...], label: str, *, unique: bool = False) -> None:
    if not isinstance(values, tuple) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must be a tuple of nonempty strings.")
    if unique and (len(set(values)) != len(values) or any(value != value.strip() for value in values)):
        raise ValueError(f"{label} must contain unique names without surrounding whitespace.")


def _catalog(strong: str, weak: str) -> dict[str, str]:
    _names((strong, weak), "model catalog", unique=True)
    return {"strong": strong, "weak": weak}


@dataclass(frozen=True)
class AgentState:
    """State known before a call; previous_errors must be cumulative per episode."""

    step: int
    attempt: int
    previous_errors: tuple[str, ...]
    available_tools: tuple[str, ...]
    remaining_calls: int

    def __post_init__(self):
        for name in ("step", "attempt", "remaining_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer, not a boolean.")
        _names(self.previous_errors, "previous_errors")
        _names(self.available_tools, "available_tools", unique=True)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "attempt": self.attempt,
                "previous_errors": list(self.previous_errors), "available_tools": list(self.available_tools),
                "remaining_calls": self.remaining_calls}

    def snapshot(self) -> AgentState:
        return AgentState(self.step, self.attempt, self.previous_errors, self.available_tools, self.remaining_calls)


@dataclass(frozen=True)
class AgentStateV8(AgentState):
    """Explicit v8 state; transport does not enter semantic errors or attempts.

    context_utf8_bytes measures serialized prompt and history, never tokenizer
    tokens. failed_tiers is decision metadata, not a predictor feature.
    """

    infrastructure_errors: tuple[str, ...] = ()
    infrastructure_attempt: int = 0
    context_utf8_bytes: int = 0
    progress: dict = field(default_factory=dict)
    failed_tiers: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        _names(self.infrastructure_errors, "infrastructure_errors")
        for name in ("infrastructure_attempt", "context_utf8_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if (type(self.failed_tiers) is not tuple or
                any(type(tier) is not int or tier < 0 for tier in self.failed_tiers)):
            raise ValueError("failed_tiers must contain nonnegative integers.")
        if type(self.progress) is not dict:
            raise ValueError("progress must be a JSON object.")
        object.__setattr__(self, "progress", _json_copy(self.progress))

    def to_dict(self):
        return {**super().to_dict(), "infrastructure_errors": list(self.infrastructure_errors),
                "infrastructure_attempt": self.infrastructure_attempt,
                "context_utf8_bytes": self.context_utf8_bytes,
                "context_length_unit": "utf8_bytes_prompt_and_history",
                "progress": _json_copy(self.progress), "failed_tiers": list(self.failed_tiers)}

    def snapshot(self):
        return AgentStateV8(self.step, self.attempt, self.previous_errors, self.available_tools,
                            self.remaining_calls, self.infrastructure_errors, self.infrastructure_attempt,
                            self.context_utf8_bytes, self.progress, self.failed_tiers)


@dataclass(frozen=True)
class AgentRequest:
    """Full prompt and observed history, with independent nested containers."""

    prompt: str
    history: tuple[dict, ...]
    state: AgentState

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be nonempty text.")
        if not isinstance(self.state, AgentState):
            raise ValueError("state must be an AgentState.")
        if not isinstance(self.history, tuple):
            raise ValueError("history must be a tuple of role/content objects.")
        copied = []
        for entry in self.history:
            if (type(entry) is not dict or set(entry) != {"role", "content"} or
                    entry["role"] not in ("assistant", "tool", "error")):
                raise ValueError("History entries require exactly role/content and role assistant, tool or error.")
            copied.append(_json_copy(entry))
        object.__setattr__(self, "history", tuple(copied))
        object.__setattr__(self, "state", self.state.snapshot())

    def snapshot(self) -> AgentRequest:
        """Create an independent request for a policy, provider or trace owner."""
        return AgentRequest(self.prompt, self.history, self.state)

    def to_dict(self) -> dict[str, Any]:
        # Revalidate because a caller may have modified a nested JSON container.
        snapshot = self.snapshot()
        return {"prompt": snapshot.prompt, "history": list(snapshot.history), "state": snapshot.state.to_dict()}

    def serialize(self) -> str:
        """Deterministic JSON including every supplied prompt/history/state field."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ModelReply:
    """One provider reply; cost is reported expenditure, not an inferred tariff."""

    content: str
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None

    def __post_init__(self):
        # Empty replies are retained as observations; parse_action rejects them.
        if not isinstance(self.content, str):
            raise ValueError("content must be text.")
        if type(self.usage) is not dict:
            raise ValueError("usage must be a JSON object.")
        object.__setattr__(self, "usage", _json_copy(self.usage))
        if self.cost_usd is not None:
            if isinstance(self.cost_usd, bool) or not isinstance(self.cost_usd, Real):
                raise ValueError("cost_usd must be a finite nonnegative number or None.")
            try:
                cost = float(self.cost_usd)
            except (ValueError, OverflowError) as exc:
                raise ValueError("cost_usd must be a finite nonnegative number or None.") from exc
            if not isfinite(cost) or cost < 0:
                raise ValueError("cost_usd must be a finite nonnegative number or None.")
            object.__setattr__(self, "cost_usd", cost)

    def snapshot(self) -> ModelReply:
        return ModelReply(self.content, self.usage, self.cost_usd)

    def to_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {"content": snapshot.content, "usage": snapshot.usage, "cost_usd": snapshot.cost_usd}


def _request(request: AgentRequest) -> AgentRequest:
    if not isinstance(request, AgentRequest):
        raise ValueError("A policy requires an AgentRequest.")
    return request


@dataclass(frozen=True)
class AlwaysStrong:
    strong: str
    weak: str
    name: ClassVar[str] = "always_strong"

    def __post_init__(self):
        _catalog(self.strong, self.weak)

    def select(self, request: AgentRequest) -> str:
        _request(request)
        return self.strong

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "catalog": _catalog(self.strong, self.weak), "rule": "Always select the declared strong model."}


@dataclass(frozen=True)
class AlwaysWeak:
    strong: str
    weak: str
    name: ClassVar[str] = "always_weak"

    def __post_init__(self):
        _catalog(self.strong, self.weak)

    def select(self, request: AgentRequest) -> str:
        _request(request)
        return self.weak

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "catalog": _catalog(self.strong, self.weak), "rule": "Always select the declared weak model."}


@dataclass(frozen=True)
class EscalateOnError:
    """Use weak until the harness records an error; stay strong in that episode.

    The harness carries the cumulative previous_errors tuple. This policy does
    not infer errors from prompt/history text or compare replies with a reference
    answer. It retains no state between episodes.
    """

    strong: str
    weak: str
    name: ClassVar[str] = "cheap_then_escalate"

    def __post_init__(self):
        _catalog(self.strong, self.weak)

    def select(self, request: AgentRequest) -> str:
        return self.strong if _request(request).state.previous_errors else self.weak

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "catalog": _catalog(self.strong, self.weak),
                "rule": "Select weak until cumulative harness previous_errors is nonempty, then strong for the rest of that episode; never inspect answer correctness or error-like text."}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def parse_action(content: str, available_tools: tuple[str, ...]) -> dict[str, Any]:
    """Parse exactly one JSON action; never dispatch tools or execute its text."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("An action must be nonempty JSON text.")
    _names(available_tools, "available_tools", unique=True)
    try:
        action = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The reply must contain exactly one JSON action object.") from exc
    action = _json_copy(action)  # Also rejects overflow such as 1e999.
    if type(action) is not dict:
        raise ValueError("The reply must contain one action object, not a list of actions.")
    if action.get("type") == "tool":
        if (set(action) != {"type", "name", "arguments"} or type(action["name"]) is not str or
                action["name"] not in available_tools or type(action["arguments"]) is not dict):
            raise ValueError("A tool action requires exactly type/name/arguments and a declared tool with object arguments.")
    elif action.get("type") == "final":
        if set(action) != {"type", "answer"}:
            raise ValueError("A final action requires exactly type/answer.")
    else:
        raise ValueError("Action type must be tool or final.")
    return action


@dataclass(frozen=True)
class RelayPolicy:
    """Adapt the existing SDK without replacing its trained model identities."""

    router: RelayRouter
    name: ClassVar[str] = "relay"
    _strong: str = field(init=False, repr=False)
    _weak: str = field(init=False, repr=False)

    def __post_init__(self):
        from .client import RelayRouter

        if not isinstance(self.router, RelayRouter):
            raise ValueError("router must be a RelayRouter.")
        catalog = self.router.model_catalog
        checked = _catalog(catalog["strong"], catalog["weak"])
        object.__setattr__(self, "_strong", checked["strong"])
        object.__setattr__(self, "_weak", checked["weak"])
        self._check_catalog()

    def _check_catalog(self):
        expected = _catalog(self._strong, self._weak)
        if (self.router.model_catalog != expected or
                any(getattr(self.router.decider, role, None) != model for role, model in expected.items())):
            raise ValueError("RelayPolicy cannot remap or change the trained model identities.")

    @property
    def strong(self) -> str:
        return self._strong

    @property
    def weak(self) -> str:
        return self._weak

    def select(self, request: AgentRequest) -> str:
        self._check_catalog()
        decision = self.router.route(_request(request).serialize())
        if decision.selected_model not in (self._strong, self._weak):
            raise ValueError("Relay selected a model outside its original catalog.")
        return decision.selected_model

    def manifest(self) -> dict[str, Any]:
        self._check_catalog()
        import joblib

        try:
            state_fingerprint = joblib.hash(self.router.decider, hash_name="sha1")
        except Exception as exc:
            raise ValueError("The decider must have serializable state to freeze its model fingerprint.") from exc
        return _json_copy({"name": self.name, "catalog": _catalog(self._strong, self._weak),
                           "rule": "Pass complete serialized AgentRequest to the existing RelayRouter with its exact trained catalog; no model aliasing.",
                           "decider_state_fingerprint": {"algorithm": "joblib.hash/sha1", "value": state_fingerprint,
                                                         "joblib_version": joblib.__version__,
                                                         "scope": "serialized decider state, including fitted parameters; not an artifact-file SHA or authenticity proof"},
                           "router": {"policy_mode": self.router.policy_mode, "max_harm": self.router.max_harm,
                                      "min_weak_success": self.router.min_weak_success, "tradeoff": self.router.tradeoff,
                                      "cost_unit": self.router.cost_unit, "evaluation_scope": self.router.evaluation_scope}})
