"""Offline native multi-turn contracts; synthetic fixtures are not BFCL scores."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from relay_router.errors import (BackendTransportError, BackendActionError, BackendHTTPError,
                                 BackendRateLimitError, BackendContextError)
from routerbench.agent_evaluation import _unseal, _seal, _write, object_hash
from routerbench.bfcl_environment import BFCLResourceLimit
from routerbench.bfcl_harness import (
    BFCLReply, BFCLRequest, BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade,
    prepare_bfcl_episodes, run_bfcl_episodes, validate_native_batch, request_input_bytes, build_initial_request, UPSTREAM_COMMIT,
)


def native(name, arguments, identifier="call-1"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class FixtureRuntime:
    instances = []

    def __init__(self, snapshot_path, entry):
        self.total, self.executed, self.evaluations = 0, [], 0
        self.entry = deepcopy(entry)
        type(self).instances.append(self)

    def native_tools(self):
        return [{"type": "function", "function": {"name": "inc", "description": "Increment a virtual counter.",
            "parameters": {"type": "object", "properties": {"amount": {"type": "integer"}}, "required": ["amount"]}}}]

    def validate_batch(self, calls):
        for call in calls:
            if (call["name"] != "inc" or set(call["arguments"]) != {"amount"}
                    or type(call["arguments"]["amount"]) is not int):
                raise ValueError("Invalid increment.")

    def encode_call(self, name, arguments):
        self.validate_batch([{"name": name, "arguments": arguments}])
        return f"inc(amount={arguments['amount']!r})"

    def execute_batch(self, calls):
        self.validate_batch(calls)
        results = []
        for call in calls:
            self.total += call["arguments"]["amount"]
            self.executed.append(deepcopy(call))
            results.append({"ok": True, "observation": json.dumps({"total": self.total}), "error_type": None})
        return results

    def evaluate(self, decoded_turn_calls, reference):
        self.evaluations += 1
        return {"valid": self.total == reference["expected_total"] and len(decoded_turn_calls) == 2,
                "private_detail": "GRADER_PRIVATE_DETAIL"}


class BFCLHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "run"
        FixtureRuntime.instances = []
        self.models = ("fixture/cheap", "fixture/strong")
        self.cases = [{"id": "multi_turn_base_fixture", "source_group": "base-fixture",
            "entry": {"id": "multi_turn_base_fixture", "question": [[{"role": "user", "content": "First public turn."}],
                    [{"role": "user", "content": "SECOND_FUTURE_SECRET turn."}]], "initial_config": {}, "involved_classes": ["Fixture"]},
            "reference": {"expected_total": 3, "private_answer": "HIDDEN_REFERENCE"}}]
        self.config = {"schema_version": 1, "run_id": "bfcl-contract-fixture", "execution_mode": "simulated",
            "snapshot_path": "fixture://virtual-only", "upstream_commit": UPSTREAM_COMMIT, "model_catalog": list(self.models),
            "max_calls_per_turn": 4, "max_total_generation_calls": 32,
            "harness": {"max_transport_retries": 1, "retry_delay_seconds": 0},
            "generation_manifest": {"kind": "offline fixture", "real_provider": False}, "pricing": {}, "compute_rates": {}}

    def fixture_generate(self, model, request):
        current = next(i for i in range(len(request.messages)-1, -1, -1) if request.messages[i]["role"] == "user")
        if any(m["role"] == "tool" for m in request.messages[current+1:]):
            return BFCLReply(content="Turn finished.", usage={"input_tokens": 100, "output_tokens": 10})
        return BFCLReply(tool_calls=(native("inc", {"amount": request.state.turn_index+1}),),
                         usage={"input_tokens": 100, "output_tokens": 10})

    def run_case(self, generate=None, policies=None):
        return run_bfcl_episodes(self.cases, policies or [BFCLAlwaysStrong(self.models)],
            generate or self.fixture_generate, self.output, self.config, runtime_factory=FixtureRuntime)

    def add_second_case(self):
        second = deepcopy(self.cases[0])
        second["id"] = second["entry"]["id"] = "multi_turn_base_fixture_2"
        second["source_group"] = "base-fixture-2"
        self.cases.append(second)

    def test_shared_failures_stop_session_after_first_censored_episode(self):
        self.add_second_case()
        self.config["session_stop_on_infrastructure"] = True
        for error, calls in ((BackendHTTPError(status_code=401), 1),
                             (BackendRateLimitError(status_code=429), 2),
                             (BackendTransportError(), 2)):
            with self.subTest(error=type(error).__name__):
                def generate(model, request):
                    raise error
                path = self.output / type(error).__name__
                result = run_bfcl_episodes(self.cases, [BFCLReactiveCascade(self.models)], generate,
                    path, self.config, runtime_factory=FixtureRuntime)
                self.assertEqual(result["generation_calls"], calls)
                self.assertEqual(len(result["episodes"]), 1)
                self.assertIsNone(result["episodes"][0]["success"])
                self.assertEqual(result["episodes"][0]["censored_reason"], "infrastructure_failure")
                self.assertEqual(result["completed_pairs"], [[self.cases[0]["id"], "cheap_then_escalate"]])
                self.assertEqual(result["unattempted_pairs"], [[self.cases[1]["id"], "cheap_then_escalate"]])
                self.assertTrue(result["session_stopped"])
                self.assertFalse(result["execution_complete"])
                self.assertEqual(result["session_stop_reason"]["category"], error.category)
                self.assertTrue(all(e["model"] == self.models[0] for e in result["episodes"][0]["events"]))
                self.assertTrue(all(e["request"]["state"]["failed_tiers"] == [] for e in result["episodes"][0]["events"]))

    def test_context_capacity_is_episode_local_and_default_keeps_session_running(self):
        self.add_second_case()
        for enabled, error in ((True, BackendContextError()), (False, BackendHTTPError(status_code=401))):
            with self.subTest(enabled=enabled):
                self.config["session_stop_on_infrastructure"] = enabled
                def generate(model, request):
                    raise error
                result = run_bfcl_episodes(self.cases, [BFCLAlwaysStrong(self.models)], generate,
                    self.output / str(enabled), self.config, runtime_factory=FixtureRuntime)
                self.assertEqual(len(result["episodes"]), 2)
                self.assertEqual(result["generation_calls"], 2)
                self.assertTrue(result["execution_complete"])
                self.assertFalse(result["session_stopped"])
                self.assertEqual(result["unattempted_pairs"], [])
                self.assertIsNone(result["session_stop_reason"])

    def test_transport_exhausting_turn_cap_stops_session(self):
        self.add_second_case()
        self.config.update(session_stop_on_infrastructure=True, max_calls_per_turn=1)
        def generate(model, request):
            raise BackendTransportError()
        result = self.run_case(generate)
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(result["episodes"][0]["censored_reason"], "infrastructure_budget_exhausted")
        self.assertEqual(result["session_stop_reason"]["category"], "transport")
        self.assertEqual(len(result["unattempted_pairs"]), 1)

    def test_generator_budget_hook_runs_before_new_episode_and_partial_summary_is_immutable(self):
        self.add_second_case()
        fixture_generate = self.fixture_generate
        class BudgetGenerator:
            calls = 0
            hook_calls = 0
            def __call__(self, model, request):
                self.calls += 1
                return fixture_generate(model, request)
            def session_stop_reason(self):
                self.hook_calls += 1
                return {"reason": "local_session_budget", "generation_calls": self.calls} if self.calls >= 4 else None
        generate = BudgetGenerator()
        result = self.run_case(generate)
        self.assertEqual(generate.calls, 4)
        self.assertEqual(generate.hook_calls, 2)
        self.assertTrue(result["episodes"][0]["success"])
        self.assertEqual(result["session_stop_reason"]["reason"], "local_session_budget")
        self.assertEqual(len(result["unattempted_pairs"]), 1)
        self.assertEqual(result["planned_pairs"], result["completed_pairs"]+result["unattempted_pairs"])
        records = _unseal(self.output / "journal.json")["records"]
        self.assertTrue(all(r["episode_id"] == self.cases[0]["id"] for r in records))
        self.assertEqual(self.run_case(generate), result)
        self.assertEqual(generate.calls, 4)
        self.assertEqual(generate.hook_calls, 2)

    def test_generator_stop_before_first_attempt_never_creates_pending_call(self):
        class ExhaustedGenerator:
            def __call__(self, model, request):
                raise AssertionError("Budget must stop before generation.")
            def session_stop_reason(self):
                return {"reason": "local_session_budget"}
        result = self.run_case(ExhaustedGenerator())
        self.assertEqual(result["generation_calls"], 0)
        self.assertEqual(result["journal_records"], 0)
        self.assertEqual(result["episodes"], [])
        self.assertEqual(result["unattempted_pairs"], result["planned_pairs"])

    def test_session_stop_contract_is_frozen_and_bool_strict(self):
        self.config["session_stop_on_infrastructure"] = 1
        with self.assertRaisesRegex(ValueError, "booleano"):
            self.run_case()
        self.config["session_stop_on_infrastructure"] = True
        frozen = prepare_bfcl_episodes(self.cases, [BFCLAlwaysStrong(self.models)], self.output, self.config,
            generate=self.fixture_generate, runtime_factory=FixtureRuntime)
        rule = frozen["session_control"]
        self.assertTrue(rule["stop_on_shared_infrastructure"])
        self.assertIn("rate_limit", rule["shared_failure_categories"])
        self.assertEqual(rule["excluded_failure_category"], "context_capacity")
        self.assertFalse(rule["generator_stop_hook"])

    def test_final_closes_turn_and_future_messages_and_reference_stay_private(self):
        initial = build_initial_request(self.cases[0], self.config, runtime_factory=FixtureRuntime)
        requests = []
        def generate(model, request):
            requests.append(request.snapshot())
            text = request.serialize()
            self.assertNotIn("HIDDEN_REFERENCE", text)
            self.assertNotIn("GRADER_PRIVATE_DETAIL", text)
            if request.state.turn_index == 0:
                self.assertNotIn("SECOND_FUTURE_SECRET", text)
            self.assertEqual(request.state.context_utf8_bytes, request_input_bytes(request))
            return self.fixture_generate(model, request)
        result = self.run_case(generate)
        episode = result["episodes"][0]
        self.assertTrue(episode["success"])
        self.assertEqual(result["generation_calls"], 4)
        self.assertEqual([r.state.turn_index for r in requests], [0, 0, 1, 1])
        self.assertEqual(requests[0].to_dict(), initial.to_dict())
        self.assertNotIn("remaining_episode_calls", initial.state.to_dict())
        self.assertEqual(episode["decoded_turn_calls"], [[["inc(amount=1)"]], [["inc(amount=2)"]]])
        self.assertEqual(sum(r.evaluations for r in FixtureRuntime.instances), 1)

    def test_batch_is_preserved_in_order_with_native_results(self):
        observed = []
        def generate(model, request):
            observed.append(request)
            if request.state.step == 0:
                return BFCLReply(tool_calls=(native("inc", {"amount": 1}, "a"), native("inc", {"amount": 2}, "b")))
            return BFCLReply(content="done")
        result = self.run_case(generate)
        first = result["episodes"][0]["events"][0]
        self.assertEqual([c["id"] for c in first["native_tool_calls"]], ["a", "b"])
        self.assertEqual([m.get("tool_call_id") for m in observed[1].messages if m["role"] == "tool"], ["a", "b"])
        self.assertEqual(result["episodes"][0]["decoded_turn_calls"][0], [["inc(amount=1)", "inc(amount=2)"]])
        self.assertEqual(result["generation_calls"], 3)

    def test_whole_invalid_batch_has_no_effect_and_advances_turn_then_escalates(self):
        seen = []
        def generate(model, request):
            seen.append((model, request))
            if request.state.turn_index == 0:
                return BFCLReply(tool_calls=(native("inc", {"amount": 1000}, "a"), native("missing", {}, "b")),
                                 usage={"input_tokens": 17777}, cost_usd=.12345)
            return BFCLReply(content="done")
        result = self.run_case(generate, [BFCLReactiveCascade(self.models)])
        self.assertEqual([m for m, _ in seen], [self.models[0], self.models[-1]])
        self.assertEqual(seen[1][1].state.failed_tiers, (0,))
        self.assertEqual(seen[1][1].state.attempt, 0)
        self.assertNotIn("17777", seen[1][1].serialize())
        self.assertNotIn("0.12345", seen[1][1].serialize())
        self.assertEqual(sum(len(r.executed) for r in FixtureRuntime.instances), 0)
        self.assertEqual(result["episodes"][0]["events"][0]["error_scope"], "semantic")
        self.assertFalse(result["episodes"][0]["success"])

    def test_transport_is_durable_pinned_and_does_not_advance_turn_or_semantic_flags(self):
        observed = []
        def generate(model, request):
            rows = _unseal(self.output / "journal.json")["records"]
            self.assertEqual(rows[-1]["kind"], "call_started")
            self.assertEqual(rows[-1]["data"]["request"], request.to_dict())
            observed.append((model, request))
            if len(observed) == 1:
                raise BackendTransportError()
            return self.fixture_generate(model, request)
        result = self.run_case(generate, [BFCLReactiveCascade(self.models)])
        first, second = observed[:2]
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1].messages, second[1].messages)
        self.assertEqual(second[1].state.previous_errors, ())
        self.assertEqual(second[1].state.failed_tiers, ())
        self.assertEqual(second[1].state.infrastructure_attempt, 1)
        self.assertEqual(second[1].state.remaining_calls, first[1].state.remaining_calls-1)
        episode = result["episodes"][0]
        self.assertTrue(episode["success"])
        self.assertIsNone(episode["events"][0]["costs"]["llm_usd"])

    def test_provider_semantic_failure_closes_turn_and_is_not_transport(self):
        seen = []
        def generate(model, request):
            seen.append((model, request))
            if request.state.turn_index == 0:
                raise BackendActionError(status_code=400)
            return BFCLReply(content="done")
        result = self.run_case(generate, [BFCLReactiveCascade(self.models)])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][0], self.models[-1])
        self.assertEqual(seen[1][1].state.infrastructure_errors, ())
        self.assertTrue(result["episodes"][0]["events"][0]["turn_completed"])

    def test_public_runtime_tool_error_escalates_within_turn_without_private_feedback(self):
        class PublicToolErrorRuntime(FixtureRuntime):
            def __init__(self, snapshot_path, entry):
                super().__init__(snapshot_path, entry)
                self.failed_once = False
            def execute_batch(self, calls):
                self.validate_batch(calls)
                if not self.failed_once:
                    self.failed_once = True
                    return [{"ok": False, "observation": '{"error": "public fixture failure"}',
                             "error_type": "UpstreamToolError"} for _ in calls]
                return super().execute_batch(calls)
        observed = []
        def generate(model, request):
            observed.append((model, request.snapshot()))
            if request.state.turn_index == 0 and request.messages[-1]["role"] == "tool":
                if "error" in json.loads(request.messages[-1]["content"]):
                    return BFCLReply(tool_calls=(native("inc", {"amount": 1}, "retry"),),
                                     usage={"input_tokens": 100, "output_tokens": 10})
            return self.fixture_generate(model, request)
        result = run_bfcl_episodes(self.cases, [BFCLReactiveCascade(self.models)], generate,
            self.output, self.config, runtime_factory=PublicToolErrorRuntime)
        first, second = result["episodes"][0]["events"][:2]
        self.assertEqual(first["error_type"], "ToolError:UpstreamToolError")
        self.assertEqual(first["error_scope"], "semantic")
        self.assertFalse(first["turn_completed"])
        self.assertEqual(first["tool_results"][0]["observation"], '{"error": "public fixture failure"}')
        self.assertEqual([first["model"], second["model"]], list(self.models))
        self.assertEqual(second["turn_index"], first["turn_index"])
        self.assertEqual(second["request"]["state"]["attempt"], 1)
        self.assertEqual(second["request"]["state"]["failed_tiers"], [0])
        self.assertEqual(second["request"]["state"]["infrastructure_errors"], [])
        self.assertTrue(result["episodes"][0]["success"])
        self.assertEqual(sum(r.evaluations for r in FixtureRuntime.instances), 1)
        self.assertTrue(all("GRADER_PRIVATE_DETAIL" not in request.serialize() for _, request in observed))

    def test_interrupted_generation_is_never_reissued_on_resume(self):
        seen = []
        def generate(model, request):
            seen.append(model)
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_case(generate)
        result = self.run_case(generate)
        self.assertEqual(seen, [self.models[-1]])
        episode = result["episodes"][0]
        self.assertEqual(episode["termination"], "interrupted_not_retried")
        self.assertIsNone(episode["success"])
        self.assertFalse(episode["operational_success"])
        self.assertIsNone(episode["events"][0]["costs"]["total_usd"])
        self.assertEqual(sum(r.evaluations for r in FixtureRuntime.instances), 0)
        self.assertEqual(self.run_case(generate), result)

    def test_failed_private_grading_does_not_trigger_another_attempt(self):
        calls = []
        def wrong(model, request):
            calls.append((model, request))
            return BFCLReply(content="unsupported claim of success")
        result = self.run_case(wrong, [BFCLReactiveCascade(self.models)])
        self.assertFalse(result["episodes"][0]["success"])
        self.assertEqual(len(calls), 2)
        self.assertEqual([m for m, _ in calls], [self.models[0]]*2)
        self.assertEqual(sum(r.evaluations for r in FixtureRuntime.instances), 1)

    def test_checker_failure_preserves_completed_usage_and_never_retries(self):
        class BrokenChecker(FixtureRuntime):
            def evaluate(self, decoded_turn_calls, reference):
                raise ValueError("private evaluator details must not be persisted")
        result = run_bfcl_episodes(self.cases, [BFCLAlwaysStrong(self.models)], self.fixture_generate,
            self.output, self.config, runtime_factory=BrokenChecker)
        episode = result["episodes"][0]
        self.assertEqual(result["generation_calls"], 4)
        self.assertEqual(episode["termination"], "evaluation_error")
        self.assertIsNone(episode["success"])
        self.assertEqual(episode["censored_reason"], "evaluation_error")
        self.assertEqual(episode["events"][-1]["usage"]["input_tokens"], 100)
        self.assertNotIn("private evaluator details", json.dumps(result))

    def test_resource_guard_validation_execution_and_evaluation_are_censored(self):
        for phase in ("validation", "execution", "evaluation"):
            with self.subTest(phase=phase):
                class ResourceRuntime(FixtureRuntime):
                    def validate_batch(self, calls):
                        if phase == "validation":
                            raise BFCLResourceLimit("fixture_precision", 4096)
                        return super().validate_batch(calls)
                    def execute_batch(self, calls):
                        if phase == "execution":
                            raise BFCLResourceLimit("fixture_output", 65536, phase="output")
                        return super().execute_batch(calls)
                    def evaluate(self, decoded, reference):
                        if phase == "evaluation":
                            raise BFCLResourceLimit("fixture_gold", 4096)
                        return super().evaluate(decoded, reference)
                path = self.output / phase
                result = run_bfcl_episodes(self.cases, [BFCLReactiveCascade(self.models)], self.fixture_generate,
                    path, self.config, runtime_factory=ResourceRuntime)
                episode = result["episodes"][0]
                expected = "evaluation_resource_failure" if phase == "evaluation" else "tool_resource_failure"
                self.assertEqual(episode["termination"], expected)
                self.assertEqual(episode["censored_reason"], expected)
                self.assertIsNone(episode["success"])
                self.assertFalse(episode["operational_success"])
                self.assertTrue(all(e["model"] == self.models[0] for e in episode["events"]))
                self.assertTrue(all(e["request"]["state"]["failed_tiers"] == [] for e in episode["events"]))
                self.assertEqual(episode["events"][-1]["usage"]["input_tokens"], 100)
                self.assertEqual(result["generation_calls"], 4 if phase == "evaluation" else 1)
                self.assertEqual(run_bfcl_episodes(self.cases, [BFCLReactiveCascade(self.models)], self.fixture_generate,
                    path, self.config, runtime_factory=ResourceRuntime), result)

    def test_resume_rejects_a_completed_call_whose_request_changed(self):
        self.run_case()
        path = self.output / "journal.json"
        saved = _unseal(path)
        completed = next(r for r in saved["records"] if r["kind"] == "call_completed")
        completed["data"]["request"]["messages"][0]["content"] = "changed after generation"
        previous = saved["freeze_sha256"]
        for record in saved["records"]:
            record["previous_sha256"] = previous
            record["record_sha256"] = object_hash({k: v for k, v in record.items() if k != "record_sha256"})
            previous = record["record_sha256"]
        _write(path, _seal(saved))
        with self.assertRaisesRegex(ValueError, "decision persistida"):
            self.run_case()

    def test_budget_is_per_turn_counts_retries_and_does_not_reveal_future_user(self):
        self.config["max_calls_per_turn"] = 1
        def transport(model, request):
            self.assertNotIn("SECOND_FUTURE_SECRET", request.serialize())
            raise BackendTransportError()
        result = self.run_case(transport)
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(result["episodes"][0]["termination"], "turn_call_budget_exhausted")
        self.assertEqual(sum(r.evaluations for r in FixtureRuntime.instances), 0)

    def test_prepare_and_global_budget_validation_never_generate(self):
        def forbidden(model, request):
            raise AssertionError("Generation during prepare")
        frozen = prepare_bfcl_episodes(self.cases, [BFCLAlwaysStrong(self.models)], self.output, self.config,
            generate=forbidden, runtime_factory=FixtureRuntime)
        self.assertEqual(frozen["maximum_generation_calls"], 8)
        self.assertEqual(frozen["budget_contract"]["official_executed_batch_cap"], 21)
        self.assertFalse((self.output / "journal.json").exists())
        self.config["max_total_generation_calls"] = 7
        with self.assertRaisesRegex(ValueError, "limite global"):
            self.run_case(forbidden)

    def test_requests_and_replies_are_snapshots_and_reject_nonfinite_values(self):
        captured = []
        def generate(model, request):
            captured.append(request.to_dict())
            request.messages[0]["content"] = "mutated callback copy"
            request.tools[0]["function"]["description"] = "mutated schema copy"
            return BFCLReply(content="done")
        result = self.run_case(generate)
        first = result["episodes"][0]["events"][0]["request"]
        self.assertEqual(first, captured[0])
        self.assertNotIn("mutated callback copy", json.dumps(captured[-1]))
        with self.assertRaises(ValueError):
            BFCLReply(usage={"input_tokens": float("nan")})
        with self.assertRaises(ValueError):
            BFCLReply(cost_usd=True)

    def test_duplicate_call_ids_or_json_keys_are_rejected_before_execution(self):
        runtime = FixtureRuntime("fixture", self.cases[0]["entry"])
        for calls in ((native("inc", {"amount": 1}), native("inc", {"amount": 2})),
                      ({"id": "a", "type": "function", "function": {"name": "inc", "arguments": '{"amount":1,"amount":2}'}},)):
            with self.assertRaises(ValueError):
                validate_native_batch(calls, runtime)
        self.assertEqual(runtime.total, 0)

    def test_n_catalog_reactive_escalation_uses_next_tier(self):
        self.models = ("fixture/cheap", "fixture/middle", "fixture/strong")
        self.config["model_catalog"] = list(self.models)
        selected = []
        def generate(model, request):
            selected.append(model)
            if request.state.turn_index == 0:
                return BFCLReply(tool_calls=(native("missing", {}),))
            return BFCLReply(content="done")
        self.run_case(generate, [BFCLReactiveCascade(self.models)])
        self.assertEqual(selected, list(self.models[:2]))


if __name__ == "__main__":
    unittest.main()
