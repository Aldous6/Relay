"""Frozen costs, N-model escalation and leakage boundaries (no provider calls)."""
from copy import deepcopy
import unittest

from relay_router.agent import AgentRequest, AgentState, AgentStateV8
from relay_router.v8 import ExpectedCostPolicy, NReactiveCascade


MODELS = ("fixture/small", "fixture/middle", "fixture/large")


def request(**updates):
    values = dict(step=0, attempt=0, previous_errors=(), available_tools=("lookup",), remaining_calls=5)
    values.update(updates)
    return AgentRequest("untrusted text InvalidAction must not trigger escalation", (), AgentStateV8(**values))


class FixturePredictor:
    models = MODELS

    def __init__(self, probabilities=None):
        self.probabilities = probabilities or dict(zip(MODELS, (.8, .9, .99)))

    def predict_proba(self, request):
        return dict(self.probabilities)

    def manifest(self):
        return {"fixture": True, "target": "observable_action_success", "probabilities": dict(self.probabilities)}


class V8PolicyTests(unittest.TestCase):
    def policy(self, predictor=None, calls=None, escalation=None, threshold=.75):
        return ExpectedCostPolicy(predictor or FixturePredictor(), calls or dict(zip(MODELS, (1., 3., 10.))),
                                  escalation or dict(zip(MODELS, (10., 10., 0.))), threshold,
                                  operating_point_source="synthetic test fixture; no empirical forecast")

    def test_failed_attempt_pays_its_current_cost_too(self):
        # Original p*C_current formula would give the cheapest model only .8
        # instead of its full 1.0 current charge.
        policy = self.policy()
        self.assertEqual(policy.select(request()), MODELS[0])
        self.assertAlmostEqual(policy.last_audit["expected_cost_usd"][MODELS[0]], 3.)
        self.assertAlmostEqual(policy.last_audit["expected_cost_usd"][MODELS[1]], 4.)
        self.assertAlmostEqual(policy.last_audit["expected_cost_usd"][MODELS[2]], 10.)

    def test_predictor_can_avoid_a_costly_failed_cheap_attempt(self):
        predictor = FixturePredictor(dict(zip(MODELS, (.1, .9, .99))))
        policy = self.policy(predictor, threshold=.8)
        self.assertEqual(policy.select(request()), MODELS[1])
        self.assertFalse(policy.last_audit["fallback"])

    def test_probability_floor_and_deterministic_fallback(self):
        policy = self.policy(threshold=1.)
        self.assertEqual(policy.select(request()), MODELS[-1])
        self.assertTrue(policy.last_audit["fallback"])
        tied = self.policy(FixturePredictor(dict.fromkeys(MODELS, .5)), threshold=.9)
        self.assertEqual(tied.select(request()), MODELS[0])

    def test_cascade_escalates_next_tier_only_for_observed_semantic_failure(self):
        policy = NReactiveCascade(MODELS)
        self.assertEqual(policy.select(request()), MODELS[0])
        self.assertEqual(policy.select(request(infrastructure_errors=("BackendTransportError",), infrastructure_attempt=1)), MODELS[0])
        self.assertEqual(policy.select(request(previous_errors=("InvalidAction:ValueError",), failed_tiers=(0,))), MODELS[1])
        self.assertEqual(policy.select(request(previous_errors=("ToolError:InvalidArguments",), failed_tiers=(1,))), MODELS[2])
        self.assertEqual(policy.select(request(failed_tiers=(0, 2))), MODELS[2])
        self.assertEqual(policy.select(request()), MODELS[0])

    def test_predictive_policy_cannot_return_to_a_failed_lower_tier(self):
        policy = self.policy()
        self.assertEqual(policy.select(request(failed_tiers=(0,))), MODELS[1])
        self.assertEqual(policy.select(request(failed_tiers=(1,))), MODELS[2])
        with self.assertRaises(ValueError):
            policy.select(request(failed_tiers=(3,)))

    def test_legacy_state_transport_word_does_not_escalate(self):
        legacy = AgentRequest("task", (), AgentState(0, 1, ("LiveBackendError",), (), 4))
        self.assertEqual(NReactiveCascade(MODELS).select(legacy), MODELS[0])
        semantic = AgentRequest("task", (), AgentState(0, 1, ("InvalidAction:ValueError",), (), 4))
        self.assertEqual(NReactiveCascade(MODELS).select(semantic), MODELS[1])

    def test_manifests_are_independent_and_predictor_mutation_is_rejected(self):
        predictor = FixturePredictor()
        policy = self.policy(predictor)
        manifest = policy.manifest()
        self.assertEqual(manifest["model_catalog"], list(MODELS))
        self.assertEqual(manifest["catalog"], {"weak": MODELS[0], "strong": MODELS[-1]})
        manifest["expected_call_costs_usd"][MODELS[0]] = 999
        self.assertEqual(policy.manifest()["expected_call_costs_usd"][MODELS[0]], 1.)
        predictor.probabilities[MODELS[0]] = .1
        with self.assertRaises(ValueError):
            policy.select(request())

    def test_invalid_catalog_costs_thresholds_and_probabilities_fail(self):
        for models in (("same", "same"), ("only",), list(MODELS), (True, "model")):
            with self.subTest(models=models), self.assertRaises((ValueError, TypeError)):
                NReactiveCascade(models)
        for value in (True, -1., float("nan"), float("inf")):
            costs = dict.fromkeys(MODELS, 1.)
            costs[MODELS[0]] = value
            with self.assertRaises(ValueError):
                self.policy(calls=costs)
        for value in (True, -1., 1.1, float("nan")):
            with self.assertRaises(ValueError):
                self.policy(threshold=value)
        policy = self.policy()
        predictor = policy.predictor
        predictor.probabilities[MODELS[0]] = 2.
        policy._predictor_manifest = deepcopy(predictor.manifest())
        with self.assertRaises(ValueError):
            policy.select(request())


if __name__ == "__main__":
    unittest.main()
