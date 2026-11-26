import unittest
from copy import deepcopy

from relay_router.agent import AgentRequest, AgentState, AgentStateV8
from relay_router.trajectory import TARGET, TrajectoryValuePolicy
from relay_router.v8 import NReactiveCascade
from routerbench.agent_learning import state_features

MODELS = ("fixture/cheap", "fixture/middle", "fixture/strong")


class Predictor:
    models = MODELS

    def __init__(self):
        self.values = {model: {"success_probability": p, "remaining_provider_cost_usd": c}
                       for model, p, c in zip(MODELS, (.6, .9, .95), (8., 2., 4.))}
        self.calls = 0

    def predict_outcomes(self, request):
        self.calls += 1
        return deepcopy(self.values)

    def manifest(self):
        return {"target": TARGET, "continuation_policy": "fixture_reference", "cost_basis": "fixture"}


def request(**updates):
    base = dict(step=0, attempt=0, previous_errors=(), available_tools=(), remaining_calls=5)
    base.update(updates)
    return AgentRequest("fixture", (), AgentStateV8(**base))


class TrajectoryPolicyTests(unittest.TestCase):
    def test_uses_total_cost_once_and_joint_prediction_once(self):
        predictor = Predictor()
        policy = TrajectoryValuePolicy(predictor, .8, "predeclared fixture threshold")
        self.assertEqual(policy.select(request()), MODELS[1])
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(policy.last_audit["expected_remaining_provider_cost_usd"][MODELS[0]], 8.)
        self.assertEqual(policy.manifest()["predictor"]["target"], TARGET)

    def test_fallback_maximizes_probability_and_floor_is_semantic_only(self):
        policy = TrajectoryValuePolicy(Predictor(), 1., "fixture")
        self.assertEqual(policy.select(request()), MODELS[2])
        policy = TrajectoryValuePolicy(Predictor(), .5, "fixture")
        self.assertEqual(policy.select(request(infrastructure_errors=("BackendTransportError",))), MODELS[1])
        self.assertEqual(policy.select(request(failed_tiers=(1,))), MODELS[2])

    def test_operating_point_mutation_is_rejected_before_prediction(self):
        for field, changed in (("min_probability", .5), ("operating_point_source", "changed after freeze")):
            predictor = Predictor()
            policy = TrajectoryValuePolicy(predictor, .8, "predeclared fixture threshold")
            setattr(policy, field, changed)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "operating point changed after policy construction"):
                    policy.select(request())
                with self.assertRaisesRegex(ValueError, "operating point changed after policy construction"):
                    policy.manifest()
                self.assertEqual(predictor.calls, 0)

    def test_returned_manifest_cannot_change_frozen_operating_point(self):
        predictor = Predictor()
        policy = TrajectoryValuePolicy(predictor, .8, "predeclared fixture threshold")
        manifest = policy.manifest()
        manifest["min_probability"] = 0.
        manifest["operating_point_source"] = "external mutation"
        self.assertEqual(policy.manifest()["min_probability"], .8)
        self.assertEqual(policy.manifest()["operating_point_source"], "predeclared fixture threshold")
        self.assertEqual(policy.select(request()), MODELS[1])

    def test_rejects_bad_forecast_and_old_target(self):
        predictor = Predictor()
        policy = TrajectoryValuePolicy(predictor, .8, "fixture")
        predictor.values[MODELS[0]]["remaining_provider_cost_usd"] = None
        with self.assertRaises(ValueError):
            policy.select(request())
        predictor.manifest = lambda: {"target": "observable_action_success"}
        with self.assertRaises(ValueError):
            TrajectoryValuePolicy(predictor, .8, "fixture")

    def test_semantic_backend_error_is_in_features_and_legacy_cascade(self):
        state = AgentState(0, 1, ("BackendActionError:tool_use_failed",), (), 4)
        req = AgentRequest("fixture", (), state)
        self.assertEqual(state_features(req)["errors_BackendActionError"], 1)
        self.assertEqual(NReactiveCascade(MODELS).select(req), MODELS[1])


if __name__ == "__main__":
    unittest.main()
