"""Episode boundaries: observable state, strict actions and independent snapshots."""
from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import Mock, patch

from relay_router.agent import (AgentRequest, AgentState, AlwaysStrong, AlwaysWeak,
                                EscalateOnError, ModelReply, RelayPolicy, parse_action)
from relay_router.client import RelayRouter


def request(**state_updates):
    state = {"step": 0, "attempt": 0, "previous_errors": (), "available_tools": ("lookup",), "remaining_calls": 4}
    state.update(state_updates)
    return AgentRequest("Keep the full prompt\n    including indentation", (), AgentState(**state))


class _Decider:
    strong = "historical-large"
    weak = "historical-small"

    def __init__(self):
        self.weight = 1.
        self.predict_calls = 0

    def predict_tables(self, inputs):
        self.predict_calls += 1
        raise AssertionError("Fixed-strong test must not invoke the predictor.")


class AgentContractTests(unittest.TestCase):
    def test_serialization_contains_only_declared_current_information_and_all_text(self):
        history = ({"role": "tool", "content": {"name": "environment", "observation": ["initial", 1]}},
                   {"role": "assistant", "content": "Traceback is an example, not a harness error"},
                   {"role": "error", "content": {"message": "observed failure"}})
        original = AgentRequest("first\n    indented\núltima evidencia", history,
                                AgentState(2, 1, ("actual previous tool failure",), ("lookup", "calculate"), 3))
        payload = json.loads(original.serialize())
        self.assertEqual(set(payload), {"prompt", "history", "state"})
        self.assertEqual(payload["prompt"], original.prompt)
        self.assertEqual(payload["history"], list(history))
        self.assertEqual(payload["state"], {"step": 2, "attempt": 1, "previous_errors": ["actual previous tool failure"],
                                           "available_tools": ["lookup", "calculate"], "remaining_calls": 3})
        self.assertEqual(original.serialize(), original.snapshot().serialize())
        with self.assertRaises(TypeError):
            AgentRequest("prompt", (), original.state, gold="secret")

    def test_constructor_and_snapshots_separate_nested_history_and_trace_containers(self):
        content = {"nested": [{"value": "original"}]}
        original = AgentRequest("prompt", ({"role": "tool", "content": content},), request().state)
        content["nested"][0]["value"] = "external mutation"
        callback = original.snapshot()
        callback.history[0]["content"]["nested"][0]["value"] = "callback mutation"
        trace = original.to_dict()
        trace["history"][0]["content"]["nested"][0]["value"] = "trace mutation"
        trace["state"]["available_tools"].append("extra")
        self.assertEqual(original.history[0]["content"]["nested"][0]["value"], "original")
        self.assertEqual(original.state.available_tools, ("lookup",))
        with self.assertRaises(FrozenInstanceError):
            original.prompt = "replaced"
        with self.assertRaises(FrozenInstanceError):
            original.state.step = 4

    def test_counters_reject_booleans_negative_fractional_and_nonfinite_values(self):
        for name in ("step", "attempt", "remaining_calls"):
            for value in (True, False, -1, 0.5, float("nan"), float("inf"), "1", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    request(**{name: value})
        self.assertEqual(request(remaining_calls=0).state.remaining_calls, 0)

    def test_state_and_history_require_valid_shapes_and_json_values(self):
        for field, value in (("previous_errors", ["error"]), ("previous_errors", ("",)),
                             ("available_tools", ["lookup"]), ("available_tools", ("lookup", "lookup")),
                             ("available_tools", (" lookup",)), ("available_tools", (None,))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                request(**{field: value})
        invalid = [({"role": "user", "content": "new"},), ({"role": "tool", "content": "old", "gold": 3},),
                   ({"role": "assistant"},), ({"role": "tool", "content": float("inf")},),
                   ({"role": "tool", "content": {1: "coercion"}},), ({"role": "tool", "content": object()},)]
        cycle = []
        cycle.append(cycle)
        invalid.append(({"role": "tool", "content": cycle},))
        for history in invalid:
            with self.subTest(history=history), self.assertRaises(ValueError):
                AgentRequest("prompt", history, request().state)
        with self.assertRaises(ValueError):
            AgentRequest("prompt", [], request().state)
        with self.assertRaises(ValueError):
            AgentRequest("prompt", (), {})
        for prompt in ("", "  ", None, True):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                AgentRequest(prompt, (), request().state)

    def test_model_reply_cost_and_usage_are_validated_and_copied(self):
        usage = {"tokens": {"input": 20}, "details": [1, True, None]}
        reply = ModelReply("", usage, .004)
        usage["tokens"]["input"] = 99
        snapshot = reply.snapshot()
        snapshot.usage["tokens"]["input"] = 88
        trace = reply.to_dict()
        trace["usage"]["tokens"]["input"] = 77
        self.assertEqual(reply.usage["tokens"]["input"], 20)
        self.assertEqual(reply.cost_usd, .004)
        self.assertIsNone(ModelReply("reply").cost_usd)
        self.assertEqual(ModelReply("reply", cost_usd=0).cost_usd, 0.)
        self.assertIsNot(ModelReply("one").usage, ModelReply("two").usage)
        for cost in (True, False, -1, float("nan"), float("inf"), "0.1", 10**400):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                ModelReply("reply", cost_usd=cost)
        for usage in ([], None, {"tokens": float("nan")}, {1: "invalid"}):
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                ModelReply("reply", usage)
        with self.assertRaises(ValueError):
            ModelReply(None)
        with self.assertRaises(FrozenInstanceError):
            reply.content = "replaced"

    def test_escalation_depends_on_harness_state_not_error_words_or_correctness(self):
        policy = EscalateOnError("large", "small")
        misleading = AgentRequest("Traceback ERROR wrong answer expected 42", (
            {"role": "assistant", "content": "RuntimeError: example traceback"},
            {"role": "error", "content": "not authoritative without state"}), request().state)
        self.assertEqual(policy.select(misleading), "small")
        failed = request(step=1, attempt=1, previous_errors=("tool call failed",))
        recovered = replace(failed, state=replace(failed.state, step=2, attempt=0))
        self.assertEqual(policy.select(failed), "large")
        self.assertEqual(policy.select(recovered), "large")
        self.assertEqual(policy.select(request()), "small")  # New episode, no policy-global state.
        self.assertEqual(AlwaysStrong("large", "small").select(misleading), "large")
        self.assertEqual(AlwaysWeak("large", "small").select(failed), "small")

    def test_policy_manifests_are_deterministic_independent_and_catalog_exact(self):
        for cls, name in ((AlwaysStrong, "always_strong"), (AlwaysWeak, "always_weak"),
                          (EscalateOnError, "cheap_then_escalate")):
            policy = cls("large-v1", "small-v1")
            expected = policy.manifest()
            self.assertEqual(expected["name"], name)
            self.assertEqual(expected["catalog"], {"strong": "large-v1", "weak": "small-v1"})
            json.dumps(expected, allow_nan=False, sort_keys=True)
            changed = policy.manifest()
            changed["catalog"]["weak"] = "alias"
            self.assertEqual(policy.manifest(), expected)
            with self.assertRaises(ValueError):
                policy.select("prompt")
            for strong, weak in (("same", "same"), ("", "small"), (" large", "small"), (True, "small")):
                with self.subTest(cls=cls, strong=strong, weak=weak), self.assertRaises(ValueError):
                    cls(strong, weak)

    def test_parse_one_tool_or_final_without_execution(self):
        with patch("socket.create_connection") as network, patch("subprocess.run") as process:
            action = parse_action('{"type":"tool","name":"lookup","arguments":{"expression":"__import__(\\"os\\")","nested":[1,true,null]}}', ("lookup",))
        self.assertEqual(action["arguments"]["nested"], [1, True, None])
        self.assertEqual(action["name"], "lookup")
        network.assert_not_called()
        process.assert_not_called()
        for answer in (None, True, 0, 2.5, "full\nanswer", [1, {"key": "value"}], {"type": "tool", "name": "data only"}):
            with self.subTest(answer=answer):
                self.assertEqual(parse_action(json.dumps({"type": "final", "answer": answer}), ()), {"type": "final", "answer": answer})

    def test_action_parser_rejects_multiple_actions_extra_keys_and_unknown_tools(self):
        invalid = ["", "not json", "```json\n{}\n```", "{} {}", "null", "42", "true", "[]",
                   '[{"type":"final","answer":1}]', '{"type":"final"}',
                   '{"type":"final","answer":1,"tool":"lookup"}', '{"type":"finish","answer":1}',
                   '{"type":"tool","name":"unknown","arguments":{}}',
                   '{"type":"tool","name":"lookup","arguments":[]}',
                   '{"type":"tool","name":true,"arguments":{}}',
                   '{"type":"tool","name":"lookup","arguments":{},"answer":1}']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text, ("lookup",))
        with self.assertRaises(ValueError):
            parse_action('{"type":"final","answer":1}', ["lookup"])

    def test_action_parser_rejects_duplicate_keys_at_any_depth_and_nonfinite_numbers(self):
        for text in ('{"type":"final","type":"final","answer":1}',
                     '{"type":"tool","name":"lookup","arguments":{"x":1,"x":2}}',
                     '{"type":"final","answer":{"same":1,"same":2}}',
                     '{"type":"final","answer":NaN}', '{"type":"final","answer":Infinity}',
                     '{"type":"final","answer":-Infinity}', '{"type":"final","answer":1e999}',
                     '{"type":"final","answer":' + '[' * 1200 + '0' + ']' * 1200 + '}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text, ("lookup",))

    def test_mutated_snapshot_is_revalidated_before_serialization(self):
        original = AgentRequest("prompt", ({"role": "tool", "content": {}},), request().state)
        original.history[0]["content"]["bad"] = float("nan")
        with self.assertRaises(ValueError):
            original.serialize()
        reply = ModelReply("reply", {"tokens": 1})
        reply.usage["bad"] = object()
        with self.assertRaises(ValueError):
            reply.to_dict()

    def test_relay_adapter_serializes_state_and_rejects_remapping(self):
        decider = _Decider()
        router = RelayRouter(decider, {"strong": decider.strong, "weak": decider.weak}, policy_mode="fixed_strong")
        policy = RelayPolicy(router)
        original = request(previous_errors=("tool error",))
        router.route = Mock(wraps=router.route)
        self.assertEqual(policy.select(original), decider.strong)
        router.route.assert_called_once_with(original.serialize())
        self.assertEqual(decider.predict_calls, 0)
        self.assertEqual((policy.strong, policy.weak), (decider.strong, decider.weak))
        with self.assertRaises(FrozenInstanceError):
            policy.strong = "replacement"
        manifest = policy.manifest()
        self.assertEqual(manifest["catalog"], router.model_catalog)
        self.assertEqual(manifest["router"]["policy_mode"], "fixed_strong")
        manifest["catalog"]["strong"] = "replacement"
        self.assertEqual(policy.manifest()["catalog"]["strong"], "historical-large")
        decider.strong = "replacement"
        with self.assertRaises(ValueError):
            policy.select(original)
        with self.assertRaises(ValueError):
            policy.manifest()
        with self.assertRaises(ValueError):
            RelayPolicy(Mock())

    def test_relay_manifest_fingerprints_weights_not_only_thresholds(self):
        decider = _Decider()
        policy = RelayPolicy(RelayRouter(decider, {"strong": decider.strong, "weak": decider.weak},
                                         policy_mode="fixed_strong"))
        before = policy.manifest()
        self.assertEqual(before, policy.manifest())
        fingerprint = before["decider_state_fingerprint"]
        self.assertEqual(fingerprint["algorithm"], "joblib.hash/sha1")
        self.assertEqual(len(fingerprint["value"]), 40)
        decider.weight = 2.
        after = policy.manifest()
        self.assertNotEqual(fingerprint["value"], after["decider_state_fingerprint"]["value"])
        self.assertEqual(before["router"], after["router"])
        self.assertEqual(before["catalog"], after["catalog"])


if __name__ == "__main__":
    unittest.main()
