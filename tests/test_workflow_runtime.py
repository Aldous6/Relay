import json
from pathlib import Path
import tempfile
import unittest

from relay_router import Relay


class WorkflowRuntimeTests(unittest.TestCase):
    def test_validated_policy_observe_apply_capabilities_and_rollback(self):
        # Synthetic outcomes exercise deployment, not measured model savings.
        from relay_router.workflow_policy import propose_policy, validate_policy
        def observations(split):
            return [dict(workflow="repair", task_id=f"{split}-{i}", group_id=f"{split}-{i}",
                         split=split, model=model, accepted=True, cost_usd=cost,
                         cost_basis="tariff_estimate", evaluation_id="synthetic")
                    for i in range(2) for model, cost in (("strong", 1.), ("weak", .5))]
        proposal = propose_policy(observations("development"), baseline_model="strong", min_groups=2)
        with self.assertRaises(ValueError):
            Relay("strong", policy=proposal, mode="apply")
        policy = validate_policy(proposal, observations("validation"))
        observed = Relay("strong", policy=policy).task("repair")
        self.assertEqual(observed.model, "strong")
        self.assertEqual(observed.decision["suggested_model"], "weak")
        relay = Relay("strong", policy=policy, mode="apply")
        policy["routes"]["repair"] = "tampered"
        applied = relay.task("repair", eligible_models=["strong", "weak"])
        self.assertEqual(applied.model, "weak")
        self.assertEqual(relay.task("repair", eligible_models=["strong"]).model, "strong")
        self.assertEqual(relay.task("new-workflow").model, "strong")
        self.assertEqual(Relay("strong", mode="observe").task("repair").model, "strong")
        self.assertEqual(applied.model, "weak")

    def test_task_decision_sticky_and_host_input_response_unchanged(self):
        relay = Relay("provider/strong-v1")
        task = relay.task("maintenance")
        prompt = {"messages": [{"role": "user", "content": "line\n    indentation"}], "tools": [{"name": "test"}]}
        answer = {"content": "unchanged", "private": "not logged"}
        received = []
        def generate(model, request):
            received.append((model, request))
            return answer
        result = task.call(generate, prompt, measure=lambda _: {"cost_usd": .02, "cost_basis": "provider_reported"})
        self.assertIs(result, answer)
        self.assertIs(received[0][1], prompt)
        decision = task.decision
        decision["model"] = "tampered"
        self.assertEqual(task.model, "provider/strong-v1")
        summary = task.finish(accepted=True, cost_complete=True)
        self.assertEqual(summary["cost_usd"], .02)
        self.assertIsNone(summary["savings_usd"])

    def test_default_cost_requires_explicit_whole_task_declaration(self):
        task = Relay("strong").task("repair")
        call = task.begin_call()
        task.record_call(call, cost_usd=.1, cost_basis="tariff_estimate")
        result = task.finish(accepted=True)
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(result["known_recorded_cost_subtotal_usd"], .1)
        self.assertFalse(result["cost_complete"])

    def test_unknown_cost_survives_failure_and_auxiliary_cost(self):
        task = Relay("strong").task("repair")
        call = task.begin_call()
        task.record_call(call, status="error")
        result = task.finish(accepted=False, additional_cost_usd=.3, cost_complete=True)
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(result["unknown_cost_calls"], 1)

    def test_auxiliary_cost_and_tokens_are_counted_without_cache_double_count(self):
        task = Relay("strong").task("repair")
        for _ in range(2):
            call = task.begin_call()
            task.record_call(call, cost_usd=.02, cost_basis="tariff_estimate", input_tokens=10,
                             output_tokens=4, cached_input_tokens=6)
        result = task.finish(accepted=True, additional_cost_usd=.01, cost_complete=True)
        self.assertAlmostEqual(result["cost_usd"], .05)
        self.assertEqual(result["usage"], {"input_tokens": 20, "output_tokens": 8, "cached_input_tokens": 12})

    def test_limit_enforced_before_callback_and_no_error_retry(self):
        task = Relay("strong").task("repair", max_calls=1)
        seen = []
        def fail(model):
            seen.append(model)
            raise TimeoutError("SECRET provider message")
        with self.assertRaises(TimeoutError):
            task.call(fail)
        with self.assertRaises(ValueError):
            task.call(fail)
        self.assertEqual(seen, ["strong"])
        self.assertIsNone(task.finish(accepted=None, cost_complete=True)["cost_usd"])

    def test_metadata_log_does_not_capture_payloads_exception_text_or_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            task = Relay("strong", log_path=path).task("repair")
            secret = "private-customer-code-and-token"
            task.call(lambda model, prompt: {"content": secret}, secret)
            task.finish(accepted=None)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(secret, text)
            events = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([row["kind"] for row in events], ["relay_task_started", "relay_call_started",
                             "relay_call_finished", "relay_task_observation"])

    def test_no_silent_model_switch_or_ineligible_baseline(self):
        relay = Relay("strong")
        with self.assertRaises(ValueError):
            relay.task("repair", eligible_models=["weak"])
        task = relay.task("repair")
        with self.assertRaises(ValueError):
            task.begin_call(model="weak")
        call = task.begin_call()
        with self.assertRaises(ValueError):
            task.record_call(call, model="weak")
        task.record_call(call, status="unknown")
        task.finish(accepted=None)

    def test_pending_duplicate_and_closed_state_checks(self):
        relay = Relay("strong")
        task = relay.task("repair", task_id="t1")
        with self.assertRaises(ValueError):
            relay.task("repair", task_id="t1")
        call = task.begin_call()
        with self.assertRaises(ValueError):
            task.finish(accepted=False)
        task.record_call(call)
        with self.assertRaises(ValueError):
            task.record_call(call)
        task.finish(accepted=None)
        with self.assertRaises(ValueError):
            task.begin_call()
        with self.assertRaises(ValueError):
            task.finish(accepted=True)

    def test_invalid_metadata_and_mixed_cost_bases_never_produce_complete_cost(self):
        task = Relay("strong").task("repair")
        call = task.begin_call()
        for bad in (True, -1, float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                task.record_call(call, cost_usd=bad, cost_basis="provider_reported")
        with self.assertRaises(ValueError):
            task.record_call(call, input_tokens=2, cached_input_tokens=3)
        task.record_call(call, cost_usd=.1, cost_basis="provider_reported")
        call = task.begin_call()
        task.record_call(call, cost_usd=.2, cost_basis="tariff_estimate")
        result = task.finish(accepted=True, cost_complete=True)
        self.assertIsNone(result["cost_usd"])

    def test_explicit_evaluation_override_is_tagged_and_not_production(self):
        relay = Relay("strong", mode="apply")
        with self.assertRaises(ValueError):
            relay.task("repair", model_override="weak")
        task = relay.task("repair", task_id="paired", group_id="repo-a", split="development",
                          evaluation_id="frozen-harness-1", model_override="weak")
        self.assertEqual(task.model, "weak")
        self.assertEqual(task.decision["mode"], "evaluation")
        other = relay.task("repair", task_id="paired", group_id="repo-a", split="development",
                           evaluation_id="frozen-harness-1", model_override="strong")
        self.assertEqual(other.model, "strong")

    def test_observation_failure_is_not_a_model_failure_or_an_extra_call(self):
        task = Relay("strong").task("repair")
        seen = []
        def generate(model):
            seen.append(model)
            return {"content": "do not store"}
        with self.assertRaises(ValueError):
            task.call(generate, measure=lambda value: value)
        result = task.finish(accepted=None)
        self.assertEqual(seen, ["strong"])
        self.assertEqual(result["calls"][0]["status"], "unknown")

    def test_invalid_ids_and_finished_tasks_do_not_accumulate(self):
        relay = Relay("strong")
        for bad in (False, 0, ""):
            with self.assertRaises(ValueError):
                relay.task("repair", task_id=bad)
            with self.assertRaises(ValueError):
                relay.task("repair", group_id=bad)
        task = relay.task("repair", task_id="one")
        with self.assertRaises(ValueError):
            relay.task("repair", task_id="one")
        task.finish(accepted=None)
        self.assertEqual(relay._active, set())

    def test_async_callback_is_rejected_before_dispatch(self):
        task = Relay("strong").task("repair")
        async def generate(model):
            raise AssertionError("must not be called")
        with self.assertRaises(TypeError):
            task.call(generate)
        self.assertEqual(task.finish(accepted=None)["logical_calls"], 0)


if __name__ == "__main__":
    unittest.main()
