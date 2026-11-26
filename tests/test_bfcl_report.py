"""BFCL reporting uses offline fixtures; it never invokes providers or graders."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from relay_router.errors import BackendContextError, BackendHTTPError
from routerbench.agent_evaluation import _seal, _unseal, _write
from routerbench.bfcl_harness import (
    BFCLReply, BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade,
    prepare_bfcl_episodes, run_bfcl_episodes, UPSTREAM_COMMIT,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bfcl_report_tool", ROOT / "tools/report_bfcl.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ReportRuntime:
    grading_calls = 0

    def __init__(self, snapshot, entry):
        self.total = 0

    def native_tools(self):
        return [{"type": "function", "function": {"name": "inc", "description": "Fixture counter", "parameters": {
            "type": "object", "properties": {"amount": {"type": "integer"}}, "required": ["amount"]}}}]

    def validate_batch(self, calls):
        if any(c["name"] != "inc" or set(c["arguments"]) != {"amount"} for c in calls):
            raise ValueError("Fixture arguments")

    def encode_call(self, name, arguments):
        return f"inc(amount={arguments['amount']})"

    def execute_batch(self, calls):
        self.validate_batch(calls)
        results = []
        for call in calls:
            if call["arguments"]["amount"] <= 0:
                results.append({"ok": False, "observation": '{"error":"fixture"}', "error_type": "UpstreamToolError"})
            else:
                self.total += call["arguments"]["amount"]
                results.append({"ok": True, "observation": json.dumps({"total": self.total}), "error_type": None})
        return results

    def evaluate(self, decoded, reference):
        type(self).grading_calls += 1
        return {"valid": self.total == reference["expected"]}


def generate(model, request):
    if request.messages[-1]["role"] == "tool" and "total" in json.loads(request.messages[-1]["content"]):
        return BFCLReply(content="done", usage={"input_tokens": 100, "output_tokens": 10})
    return BFCLReply(tool_calls=({"id": "call", "type": "function", "function": {"name": "inc",
        "arguments": json.dumps({"amount": 1 if model.endswith("strong") else 0})}},),
        usage={"input_tokens": 100, "output_tokens": 10})


class BFCLReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "run"
        self.models = ["fixture/weak", "fixture/strong"]
        self.policies = [p(self.models) for p in (BFCLAlwaysStrong, BFCLAlwaysWeak, BFCLReactiveCascade)]
        self.cases = [{"id": "multi_turn_base_fixture", "source_group": "fixture-group", "entry": {
            "id": "multi_turn_base_fixture", "question": [[{"role": "user", "content": "Increment virtual counter."}]],
            "initial_config": {}, "involved_classes": ["Fixture"]}, "reference": {"expected": 1}}]
        self.config = {"schema_version": 1, "run_id": "report-fixture", "execution_mode": "simulated",
            "snapshot_path": "fixture://virtual", "upstream_commit": UPSTREAM_COMMIT, "model_catalog": self.models,
            "max_calls_per_turn": 3, "max_total_generation_calls": 9,
            "harness": {"max_transport_retries": 0, "retry_delay_seconds": 0},
            "generation_manifest": {"kind": "offline fixture"}, "pricing": {}, "compute_rates": {}}
        self.tariffs = self.root / "tariffs.json"
        _write(self.tariffs, {"schema_version": 1, "currency": "USD", "unit": "one million tokens", "models": {
            model: {"input_usd_per_million": i+1., "output_usd_per_million": 2*(i+1.), "source": "fixture only"}
            for i, model in enumerate(self.models)}})

    def run_fixture(self, callback=generate):
        return run_bfcl_episodes(self.cases, self.policies, callback, self.output, self.config, runtime_factory=ReportRuntime)

    def test_complete_audit_keeps_same_conversation_pairs_and_known_failures(self):
        result = self.run_fixture()
        grading_before = ReportRuntime.grading_calls
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["accounting"]["calls"], result["generation_calls"])
        self.assertEqual(report["accounting"]["calls"], 8)
        self.assertEqual(report["accounting"]["input_tokens"], 800)
        self.assertEqual(report["accounting"]["output_tokens"], 80)
        self.assertAlmostEqual(report["accounting"]["tariff_no_cache_usd"], .00144)
        self.assertIsNone(report["accounting"]["reported_provider_usd"])
        self.assertIsNone(report["accounting"]["wall_clock_total_seconds"])
        self.assertEqual(report["policies"]["always_strong"]["successes"], 1)
        self.assertEqual(report["policies"]["always_weak"]["known_failures"], 1)
        self.assertEqual(report["policies"]["cheap_then_escalate"]["successes"], 1)
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["closed_pairs"], 1)
        self.assertEqual(pair["first_only_success"], 1)
        self.assertIsNone(pair["confidence_interval"])
        self.assertEqual(pair["accounting_first"]["calls"], 2)
        self.assertEqual(pair["accounting_second"]["calls"], 3)
        self.assertAlmostEqual(pair["tariff_no_cache_cost_ratio_second_to_first"], .75)
        self.assertAlmostEqual(pair["tariff_no_cache_savings_fraction_second_vs_first"], .25)
        variation = report["same_request_response_variation"]
        self.assertGreaterEqual(variation["groups_with_cross_policy_responses"], 1)
        self.assertEqual(variation["groups_with_response_variation"], 0)
        self.assertEqual(ReportRuntime.grading_calls, grading_before)
        self.assertEqual(before, {p.relative_to(self.output): p.read_bytes() for p in self.output.iterdir() if p.is_file()})

    def test_prepared_population_is_not_reported_as_failed(self):
        prepare_bfcl_episodes(self.cases, self.policies, self.output, self.config, generate=generate, runtime_factory=ReportRuntime)
        report = REPORT.audit_bfcl(self.output)
        self.assertEqual(report["status"], "prepared")
        self.assertEqual(report["accounting"]["calls"], 0)
        for row in report["policies"].values():
            self.assertEqual(row["not_started_conversations"], 1)
            self.assertEqual(row["known_failures"], 0)
            self.assertIsNone(row["operational_success_rate"])
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["resource_pair_count"], 0)
        self.assertIsNone(pair["tariff_no_cache_cost_ratio_second_to_first"])

    def test_pending_attempt_has_unknown_usage_and_no_outcome_or_reissue(self):
        invocations = []
        def interrupted(model, request):
            invocations.append(model)
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(interrupted)
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        self.assertEqual(invocations, [self.models[-1]])
        self.assertEqual(report["status"], "in_progress")
        self.assertEqual(report["accounting"]["calls"], 1)
        self.assertIsNone(report["accounting"]["input_tokens"])
        self.assertIsNone(report["accounting"]["tariff_no_cache_usd"])
        self.assertEqual(report["episodes"][0]["status"], "pending")
        self.assertIsNone(report["episodes"][0]["success"])
        self.assertEqual(report["policies"]["always_strong"]["known_failures"], 0)

    def test_local_rejections_are_distinct_from_provider_calls(self):
        def local(model, request):
            raise BackendContextError(usage={"provider_metadata": {"provider_call_started": False}})
        self.run_fixture(local)
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        self.assertEqual(report["accounting"]["calls"], 3)
        self.assertEqual(report["accounting"]["no_provider_call_attempts"], 3)
        self.assertEqual(report["accounting"]["provider_calls"], 0)
        self.assertEqual(report["accounting"]["tariff_no_cache_usd"], 0.)
        self.assertTrue(all(row["censored_conversations"] == 1 for row in report["policies"].values()))
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["resource_pair_count"], 1)
        self.assertEqual(pair["uncensored_pairs"], 0)
        self.assertEqual(pair["tariff_no_cache_cost_difference_second_minus_first_usd"], 0.)
        self.assertIsNone(pair["tariff_no_cache_cost_ratio_second_to_first"])
        self.assertIsNone(pair["tariff_no_cache_savings_fraction_second_vs_first"])

    def test_resource_pairs_exclude_unmatched_closed_or_pending_conversations(self):
        second = deepcopy(self.cases[0])
        second["id"] = second["entry"]["id"] = "multi_turn_base_second"
        second["source_group"] = "second-group"
        second["entry"]["question"][0][0]["content"] = "SECOND fixture"
        self.cases.append(second)
        self.config["max_total_generation_calls"] = 18

        def interrupt_second_weak(model, request):
            if model == self.models[0] and request.messages[0]["content"] == "SECOND fixture":
                raise KeyboardInterrupt()
            return generate(model, request)

        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(interrupt_second_weak)
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        self.assertEqual(report["policies"]["always_strong"]["accounting"]["calls"], 4)
        self.assertIsNone(report["policies"]["always_weak"]["accounting"]["tariff_no_cache_usd"])
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["resource_pair_count"], 1)
        self.assertEqual(pair["resource_episode_ids"], [self.cases[0]["id"]])
        self.assertEqual(pair["accounting_first"]["calls"], 2)
        self.assertEqual(pair["accounting_second"]["calls"], 3)
        self.assertAlmostEqual(pair["accounting_first"]["tariff_no_cache_usd"], .00048)
        self.assertAlmostEqual(pair["accounting_second"]["tariff_no_cache_usd"], .00036)
        self.assertAlmostEqual(pair["tariff_no_cache_cost_ratio_second_to_first"], .75)
        self.assertAlmostEqual(pair["tariff_no_cache_savings_fraction_second_vs_first"], .25)
        self.assertEqual(pair["first_only_success"], 1)

    def test_censored_pairs_keep_measured_failed_attempt_costs(self):
        def weak_infrastructure(model, request):
            if model == self.models[0]:
                raise BackendHTTPError(status_code=400, usage={"input_tokens": 17, "output_tokens": 3})
            return generate(model, request)

        self.run_fixture(weak_infrastructure)
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["resource_pair_count"], 1)
        self.assertEqual(pair["censored_pairs"], 1)
        self.assertEqual(pair["uncensored_pairs"], 0)
        self.assertIsNone(pair["known_outcome_success_delta"])
        self.assertEqual(pair["accounting_second"]["calls"], 1)
        self.assertAlmostEqual(pair["accounting_second"]["tariff_no_cache_usd"], .000023)
        self.assertAlmostEqual(pair["tariff_no_cache_cost_ratio_second_to_first"], .000023/.00048)
        self.assertEqual(report["policies"]["always_weak"]["success_rate_among_closed"], 0.)
        self.assertIsNone(report["policies"]["always_weak"]["success_rate_with_known_outcome"])

    def test_missing_usage_prevents_paired_cost_ratio(self):
        def weak_unknown(model, request):
            if model == self.models[0]:
                raise BackendHTTPError(status_code=400)
            return generate(model, request)

        self.run_fixture(weak_unknown)
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        pair = report["paired_comparisons"]["always_strong__always_weak"]
        self.assertEqual(pair["resource_pair_count"], 1)
        self.assertIsNone(pair["accounting_second"]["tariff_no_cache_usd"])
        self.assertIsNone(pair["tariff_no_cache_cost_difference_second_minus_first_usd"])
        self.assertIsNone(pair["tariff_no_cache_cost_ratio_second_to_first"])
        self.assertIsNone(pair["tariff_no_cache_savings_fraction_second_vs_first"])

    def test_routing_p95_excludes_pinned_infrastructure_attempts(self):
        first = {"model": self.models[0], "routing_source": "policy", "router_seconds": .1,
            "generation_seconds": 1., "verification_seconds": 2., "tool_seconds": 3., "retry_wait_seconds": 4.,
            "usage": {"input_tokens": 5, "output_tokens": 2}}
        pinned = {**first, "routing_source": "pinned_infrastructure_retry", "router_seconds": 99.}
        account = REPORT._account([first, pinned], {self.models[0]: (1., 2.)})
        self.assertEqual(account["routing_p95_seconds"], .1)
        self.assertEqual(account["routing_decisions"], 1)
        self.assertEqual(account["pinned_infrastructure_attempts"], 1)
        self.assertAlmostEqual(account["measured_phase_seconds"], 119.1)
        self.assertIsNone(account["wall_clock_total_seconds"])

    def test_response_signature_ignores_call_ids_usage_timings_and_metadata(self):
        first = {"model": self.models[0], "request_sha256": "a"*64, "policy": "always_weak",
            "episode_id": "fixture", "call_index": 0, "response": {"content": "private fixture response",
                "tool_calls": [{"id": "random-first", "type": "function", "function": {
                    "name": "inc", "arguments": '{"amount": 1}'}}], "usage": {"input_tokens": 7}},
            "usage": {"input_tokens": 7}, "generation_seconds": 1., "tool_results": [
                {"ok": True, "observation": "private tool observation", "error_type": None}]}
        second = deepcopy(first)
        second.update(policy="cheap_then_escalate", generation_seconds=3.,
            usage={"input_tokens": 9, "provider_metadata": {"id": "different"}})
        second["response"]["tool_calls"][0]["id"] = "random-second"
        second["response"]["usage"] = {"input_tokens": 9}
        result = REPORT._response_variation([first, second])
        self.assertEqual(result["cross_policy_groups"], 1)
        self.assertEqual(result["groups_with_response_variation"], 0)
        self.assertEqual(result["groups_with_error_variation"], 0)
        serialized = json.dumps(result)
        for private in ("private fixture response", "private tool observation", "random-first", "random-second"):
            self.assertNotIn(private, serialized)

    def test_response_variation_compares_exact_text_raw_argument_strings_and_order(self):
        first = {"model": self.models[0], "request_sha256": "a"*64, "policy": "always_weak",
            "episode_id": "fixture", "call_index": 0, "response": {"content": "first",
                "tool_calls": [{"id": "a", "function": {"name": "inc", "arguments": '{"amount": 1}'}},
                               {"id": "b", "function": {"name": "inc", "arguments": '{"amount": 2}'}}]}}
        second = deepcopy(first)
        second["policy"] = "cheap_then_escalate"
        second["response"]["content"] = "second"
        second["response"]["tool_calls"][0]["function"]["arguments"] = '{"amount":1}'
        result = REPORT._response_variation([first, second])
        self.assertEqual(result["groups_with_response_variation"], 1)
        self.assertEqual(result["groups_with_text_variation"], 1)
        self.assertEqual(result["groups_with_native_call_variation"], 1)
        self.assertEqual(result["groups_with_error_variation"], 0)
        second["response"] = deepcopy(first["response"])
        second["response"]["tool_calls"].reverse()
        result = REPORT._response_variation([first, second])
        self.assertEqual(result["groups_with_text_variation"], 0)
        self.assertEqual(result["groups_with_native_call_variation"], 1)

    def test_response_variation_requires_same_model_request_and_two_policies(self):
        first = {"model": self.models[0], "request_sha256": "a"*64, "policy": "always_weak",
            "episode_id": "fixture", "call_index": 0, "response": {"content": "a", "tool_calls": []}}
        same_policy = {**first, "call_index": 1, "response": {"content": "b", "tool_calls": []}}
        other_request = {**first, "policy": "cheap_then_escalate", "request_sha256": "b"*64}
        other_model = {**first, "policy": "always_strong", "model": self.models[-1]}
        result = REPORT._response_variation([first, same_policy, other_request, other_model])
        self.assertEqual(result["cross_policy_groups"], 0)
        self.assertEqual(result["groups_with_response_variation"], 0)

    def test_missing_response_is_not_fabricated_as_text_variation(self):
        first = {"model": self.models[0], "request_sha256": "a"*64, "policy": "always_weak",
            "episode_id": "fixture", "call_index": 0, "response": {"content": "a", "tool_calls": []}}
        second = {**first, "policy": "cheap_then_escalate", "response": None,
            "error_type": "BackendHTTPError", "error_scope": "infrastructure",
            "backend_failure": {"category": "http", "status_code": 500}}
        result = REPORT._response_variation([first, second])
        self.assertEqual(result["cross_policy_groups"], 1)
        self.assertEqual(result["groups_with_cross_policy_responses"], 0)
        self.assertEqual(result["groups_with_response_variation"], 0)
        self.assertEqual(result["groups_with_error_variation"], 1)
        self.assertEqual(result["groups"][0]["missing_responses"], 1)

    def test_tampered_summary_or_source_snapshot_is_rejected(self):
        self.run_fixture()
        path = self.output / "summary.json"
        summary = _unseal(path)
        summary["episodes"][0]["success"] = False
        _write(path, _seal(summary))
        with self.assertRaisesRegex(ValueError, "resumen no reproduce"):
            REPORT.audit_bfcl(self.output)
        archive = self.output / "source_bundle.zip"
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Snapshot"):
            REPORT.audit_bfcl(self.output)

    def test_markdown_labels_scenario_and_missing_wall_time(self):
        self.run_fixture()
        text = REPORT.render_report(REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs))
        self.assertIn("no son facturacion ni ahorro real", text)
        self.assertIn("no registro wall-clock completo", text)
        self.assertIn("no demuestran no inferioridad", text)
        self.assertIn("| Politica |", text)
        self.assertIn("Pares cerrados comunes", text)
        self.assertIn("Intentos primera / segunda", text)
        self.assertIn("incluyendo fallos y censuras", text)
        self.assertIn("mismo modelo y request_sha256", text)
        self.assertIn("no aisla un efecto causal", text)

    def test_partial_summary_preserves_unattempted_denominator(self):
        self.config["session_stop_on_infrastructure"] = True
        def authentication(model, request):
            raise BackendHTTPError(status_code=401)
        result = self.run_fixture(authentication)
        self.assertFalse(result["execution_complete"])
        report = REPORT.audit_bfcl(self.output, tariffs_path=self.tariffs)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["accounting"]["calls"], 1)
        self.assertEqual(report["policies"]["always_weak"]["not_started_conversations"], 1)
        self.assertEqual(report["policies"]["always_weak"]["known_failures"], 0)
        self.assertIsNone(report["policies"]["always_weak"]["operational_success_rate"])


if __name__ == "__main__":
    unittest.main()
