"""Load a frozen adaptation and its independent, scope-bound pilot evidence.

Inference uses only the standard library. A missing or unfavorable report keeps
the baseline. Passing the gate is empirical pilot evidence, not a statistical
guarantee or permission to extrapolate to other tasks, models or harnesses.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .adaptive_policy import select_model, validate_artifact, validate_report


def _read(value):
    if isinstance(value, (str, Path)):
        from .workflow_cli import _json
        return _json(Path(value).read_text(encoding="utf-8-sig"))
    return deepcopy(value)


class AdaptivePromptSelector:
    """One versioned policy, one declared scope, one frozen validation report."""

    def __init__(self, policy, validation=None, *, scope, checkpoint_dir=None):
        self._artifact = deepcopy(validate_artifact(_read(policy)))
        if scope != self._artifact["scope"]:
            raise ValueError("The declared integration scope differs from the adaptation scope.")
        self._report = (None if validation is None else
                        deepcopy(validate_report(_read(validation), self._artifact)))
        self._scorer = None
        if self.activated and self._artifact.get("representation") == "routellm-bert":
            if checkpoint_dir is None:
                raise ValueError("This validated adaptation requires its verified local BERT checkpoint.")
            from .integrations.routellm import LocalBERTScorer
            self._scorer = LocalBERTScorer(checkpoint_dir)

    @property
    def strong_model(self):
        return self._artifact["baseline_model"]

    @property
    def weak_model(self):
        return self._artifact["candidate_model"]

    @property
    def scope(self):
        return self._artifact["scope"]

    @property
    def activated(self):
        return self._report is not None and self._report["activation_eligible"] is True

    def select(self, routing_text):
        if not self.activated:
            if type(routing_text) is not str or not routing_text.strip():
                raise ValueError("routing_text must be nonempty text.")
            # Rejected or missing evidence needs neither GPU/CPU inference nor
            # optional dependencies merely to retain the existing baseline.
            return {"model": self.strong_model, "selector_id": "relay-adaptive-" + self._artifact["artifact_id"][:16],
                    "threshold": self._artifact["threshold"], "preference_score": None, "routing_seconds": 0.,
                    "router_cost_usd": None, "adaptation_scope": self.scope,
                    "validation_report_id": self._report["validation_report_id"] if self._report else None,
                    "status": "baseline_no_validation", "reason": "baseline_adaptation_gate_not_met"}
        decision = select_model(self._artifact, routing_text, scorer=self._scorer)
        decision.update(validation_report_id=(self._report["validation_report_id"]
                                               if self._report else None),
                        adaptation_scope=self.scope,
                        status="empirically_validated_pilot" if self.activated else "baseline_no_validation")
        if not self.activated:
            decision.update(model=self.strong_model, reason="baseline_adaptation_gate_not_met")
        return decision


def load_adapted_relay(policy, validation=None, *, scope, mode="apply", log_path=None, checkpoint_dir=None):
    """Return a Relay ready to attach to an existing OpenHands agent.

The caller explicitly declares the integration scope used by its verifier.
Missing/failed validation automatically preserves the baseline. Invalid or
tampered artifacts raise before any model call. Model capability filtering
and task acceptance remain the existing harness's responsibility.
"""
    from .workflow import Relay
    selector = AdaptivePromptSelector(policy, validation, scope=scope, checkpoint_dir=checkpoint_dir)
    return Relay(selector.strong_model, prompt_selector=selector, mode=mode, log_path=log_path)
