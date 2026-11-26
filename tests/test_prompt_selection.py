"""Prompt routing contracts; synthetic scores do not demonstrate product savings."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router import Relay
from relay_router.prompt_selection import PairwisePromptSelector


def selector(scorer=None, **overrides):
    values = dict(weak_model="weak", strong_model="strong", threshold=.58,
                  scorer=scorer if scorer is not None else lambda text: {"score": .1},
                  selector_id="synthetic-local-v1")
    values.update(overrides)
    return PairwisePromptSelector(**values)


class PairwiseSelectorTests(unittest.TestCase):
    def test_threshold_boundary_and_truncated_input_keep_strong(self):
        for score, truncated, expected in ((.57, 0, "weak"), (.58, 0, "strong"),
                                            (.99, 0, "strong"), (.1, 2, "strong")):
            with self.subTest(score=score, truncated=truncated):
                route = selector(lambda text: {"score": score, "input_tokens": 5,
                                                "truncated_tokens": truncated}).select("request")
                self.assertEqual(route["model"], expected)
                self.assertEqual(route["status"], "experimental")
                self.assertIsNone(route["router_cost_usd"])
                if truncated:
                    self.assertEqual(route["reason"], "strong_after_truncation")

    def test_invalid_configuration_and_text_fail_before_scorer(self):
        scorer = Mock(return_value={"score": .1})
        for kwargs in ({"weak_model": "strong"}, {"selector_id": "free text"},
                       {"strong_model": ""}, {"threshold": True}, {"threshold": -.1},
                       {"threshold": float("nan")}, {"threshold": float("inf")}, {"threshold": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                selector(scorer, **kwargs)
        route = selector(scorer)
        for text in (None, 1, {}, "", "  \n "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                route.select(text)
        scorer.assert_not_called()
        with self.assertRaises(TypeError):
            selector(scorer="not a function")

    def test_invalid_scores_and_token_metadata_never_select(self):
        malformed = [None, [], {}, {"score": True}, {"score": float("nan")},
                     {"score": float("inf")}, {"score": -1}, {"score": 10 ** 400}, {"score": 1.1}, {"score": ".1"},
                     {"score": .1, "input_tokens": -1}, {"score": .1, "input_tokens": True},
                     {"score": .1, "truncated_tokens": 1.5},
                     {"score": .1, "truncated_tokens": -1},
                     {"score": .1, "input_tokens": 3, "truncated_tokens": 4}]
        for returned in malformed:
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                selector(lambda text: returned).select("request")
        known = selector().select("request")
        self.assertIsNone(known["input_tokens"])
        self.assertIsNone(known["truncated_tokens"])

    def test_async_scorer_is_rejected_and_awaitable_closed(self):
        async def asynchronous(text):
            return {"score": .1}
        with self.assertRaises(TypeError):
            selector(asynchronous)
        returned = asynchronous("request")
        with self.assertRaises(TypeError):
            selector(lambda text: returned).select("request")
        self.assertIsNone(returned.cr_frame)


class PromptWorkflowTests(unittest.TestCase):
    def test_same_workflow_selects_two_models_from_two_prompts(self):
        scorer = Mock(side_effect=[{"score": .1}, {"score": .9}])
        relay = Relay("strong", prompt_selector=selector(scorer), mode="apply", allow_experimental=True)
        cheap = relay.task("coding", routing_text="first request")
        strong = relay.task("coding", routing_text="second request")
        self.assertEqual((cheap.model, strong.model), ("weak", "strong"))
        self.assertEqual([call.args for call in scorer.call_args_list], [("first request",), ("second request",)])
        self.assertEqual(cheap.decision["scope"], "experimental_pairwise_prompt_selection")
        self.assertIsNone(cheap.decision["policy_id"])
        self.assertNotIn("approved", cheap.decision["reason"])

    def test_selection_once_session_sticky_payload_unchanged_and_private(self):
        secret = "PRIVATE_PROMPT\n    indentation remains"
        scorer = Mock(return_value={"score": .1, "input_tokens": 8, "truncated_tokens": 0,
                                    "private": secret, "routing_text": secret})
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "relay.jsonl"
            relay = Relay("strong", prompt_selector=selector(scorer), mode="apply",
                          allow_experimental=True, log_path=log)
            task = relay.task("coding", routing_text=secret)
            message = {"messages": [{"role": "user", "content": secret}]}
            response = {"content": secret}
            generate = Mock(return_value=response)
            for _ in range(2):
                self.assertIs(task.call(generate, message), response)
            scorer.assert_called_once_with(secret)
            for call in generate.call_args_list:
                self.assertEqual(call.args[0], "weak")
                self.assertIs(call.args[1], message)
            with self.assertRaises(ValueError):
                task.begin_call(model="strong")
            changed = task.decision
            changed["prompt_selection"]["model"] = "tampered"
            self.assertEqual(task.decision["prompt_selection"]["model"], "weak")
            result = task.finish(accepted=None, cost_complete=True)
            self.assertIsNone(result["cost_usd"])
            self.assertFalse(result["cost_complete"])
            self.assertNotIn(secret, log.read_text(encoding="utf-8"))
            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["decision"]["prompt_selection"]["preference_score"], .1)

    def test_observe_keeps_baseline_and_apply_requires_explicit_experimental_flag(self):
        observed = Relay("strong", prompt_selector=selector()).task("coding", routing_text="request")
        self.assertEqual(observed.model, "strong")
        self.assertEqual(observed.decision["suggested_model"], "weak")
        self.assertEqual(observed.decision["reason"], "observation_keeps_baseline")
        with self.assertRaises(ValueError):
            Relay("strong", prompt_selector=selector(), mode="apply")
        for value in (1, None, "yes"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Relay("strong", allow_experimental=value)

    def test_ineligible_prompt_choice_falls_back_only_to_eligible_baseline(self):
        relay = Relay("strong", prompt_selector=selector(), mode="apply", allow_experimental=True)
        fallback = relay.task("coding", routing_text="request", eligible_models=["strong"])
        self.assertEqual(fallback.model, "strong")
        self.assertEqual(fallback.decision["suggested_model"], "weak")
        self.assertEqual(fallback.decision["reason"], "baseline_after_capability_filter")
        for eligible in ([], ["unrelated"]):
            with self.subTest(eligible=eligible), self.assertRaises(ValueError):
                relay.task("coding", routing_text="request", eligible_models=eligible)
        with self.assertRaises(ValueError):
            Relay("strong", prompt_selector=selector()).task("coding", routing_text="request", eligible_models=["weak"])

    def test_invalid_combinations_and_missing_text_fail_without_scoring(self):
        scorer = Mock(return_value={"score": .1})
        configured = selector(scorer)
        with self.assertRaises(ValueError):
            Relay("strong", policy={}, prompt_selector=configured)
        with self.assertRaises(ValueError):
            Relay("another-strong", prompt_selector=configured)
        with self.assertRaises(TypeError):
            Relay("strong", prompt_selector={"model": "weak"})
        relay = Relay("strong", prompt_selector=configured)
        for text in (None, 1, "", "  "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                relay.task("coding", routing_text=text)
        with self.assertRaises(ValueError):
            relay.task("coding", routing_text="request", split="development", model_override="weak")
        with self.assertRaises(ValueError):
            Relay("strong").task("coding", routing_text="request")
        scorer.assert_not_called()

    def test_routing_time_counted_and_unpriced_compute_kept_separate_from_call_costs(self):
        with patch("relay_router.prompt_selection.time") as clock:
            clock.perf_counter.side_effect = [10., 12.]
            task = Relay("strong", prompt_selector=selector(), mode="apply", allow_experimental=True).task(
                "coding", routing_text="request")
        task.call(lambda model: "response", measure=lambda value: {"cost_usd": .02, "cost_basis": "tariff_estimate"})
        result = task.finish(accepted=True, cost_complete=True)
        self.assertEqual(result["routing_seconds"], 2.)
        self.assertGreaterEqual(result["elapsed_seconds"], 2.)
        self.assertEqual(result["cost_usd"], .02)
        self.assertTrue(result["cost_complete"])
        self.assertIsNone(result["routing_cost_usd"])
        self.assertFalse(result["routing_cost_complete"])
        self.assertEqual(result["cost_scope"], "recorded_calls_plus_declared_additional_excludes_unpriced_selector_compute")
        self.assertIsNone(result["savings_usd"])

    def test_scorer_error_creates_no_session_or_ledger(self):
        scorer = Mock(side_effect=RuntimeError("private backend detail"))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "relay.jsonl"
            relay = Relay("strong", prompt_selector=selector(scorer), log_path=log)
            with self.assertRaises(RuntimeError):
                relay.task("coding", routing_text="request")
            self.assertFalse(log.exists())
            self.assertEqual(relay._active, set())


if __name__ == "__main__":
    unittest.main()
