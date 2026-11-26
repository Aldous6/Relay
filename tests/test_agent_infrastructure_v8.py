"""Contract v2 isolates semantic retries, infrastructure and public progress."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from relay_router.agent import AgentStateV8, AgentRequest, AlwaysStrong, AlwaysWeak, EscalateOnError, ModelReply
from relay_router.errors import (BackendTransportError, BackendTimeoutError, BackendHTTPError,
                                 BackendResponseError, BackendIdentityError, BackendRateLimitError,
                                 BackendActionError)
from routerbench.agent_context import render_messages
from routerbench.agent_evaluation import run_episodes, _unseal
from routerbench.agent_simulation import ScriptedToolFixture
from routerbench.agent_tasks import build_tasks
from routerbench.live import _request_json
from relay_router.v8 import NReactiveCascade, ExpectedCostPolicy

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    config = json.loads((ROOT / "configs/agent_benchmark_v1.json").read_text())
    config.update(schema_version=2, model_catalog=["simulated/weak", "simulated/strong"],
                  harness={"contract_version": 2, "max_transport_retries": 2, "retry_delay_seconds": 0},
                  bootstrap_replicates=100)
    return config


def completed(directory):
    return [r["data"] for r in _unseal(Path(directory) / "journal.json")["records"]
            if r["kind"] == "episode_completed"]


class InfrastructureV8Tests(unittest.TestCase):
    def test_progress_reports_attempt_action_and_terminal_result_without_answer(self):
        cfg = configuration()
        cfg['console_progress'] = True
        task = build_tasks(per_family=1)[0]
        with redirect_stdout(StringIO()) as output:
            rows, _ = self.run_case(ScriptedToolFixture(), config=cfg, task=task)
        console = output.getvalue()
        self.assertIn('Llamada 1/8', console)
        self.assertIn('Respuesta recibida: list_files', console)
        self.assertIn('Tarea cerrada: correcta', console)
        self.assertNotIn(task['private_reference']['answer'], console)
        self.assertTrue(all(row['success'] for row in rows))

    def run_case(self, generate, config=None, task=None):
        cfg = configuration() if config is None else config
        controls = [AlwaysStrong(cfg["models"]["strong"], cfg["models"]["weak"]),
                    EscalateOnError(cfg["models"]["strong"], cfg["models"]["weak"])]
        with tempfile.TemporaryDirectory() as directory:
            result = run_episodes([task or build_tasks(per_family=1)[0]], controls, generate, directory, cfg)
            rows = completed(directory)
            return rows, result

    def test_transport_is_pinned_without_semantic_error_or_predictor_reentry(self):
        fixture = ScriptedToolFixture()
        failures = set()
        def generate(model, request):
            self.assertIsInstance(request.state, AgentStateV8)
            if model not in failures:
                failures.add(model)
                raise BackendTransportError("sensitive details never persisted")
            self.assertFalse(request.state.previous_errors)
            return fixture(model, request)
        rows, result = self.run_case(generate)
        for row in rows:
            first, second = row["events"][:2]
            self.assertEqual(first["model"], second["model"])
            self.assertEqual(second["routing_source"], "pinned_infrastructure_retry")
            self.assertEqual(second["request"]["state"]["attempt"], 0)
            self.assertEqual(second["request"]["state"]["infrastructure_attempt"], 1)
            self.assertFalse(second["request"]["state"]["previous_errors"])
            self.assertTrue(second["is_retry"])
            self.assertTrue(row["success"])
            self.assertNotIn("sensitive details", json.dumps(row))
            self.assertIsNone(row["events"][0]["costs"]["total_usd"])
        self.assertEqual(result["metrics"]["policies"]["cheap_then_escalate"]["failure_diagnostics"]["infrastructure"], {"transport": 1})

    def test_semantic_failure_still_escalates_and_uses_shared_progress(self):
        rows, _ = self.run_case(ScriptedToolFixture(), task=build_tasks(per_family=1)[1])
        cascade = rows[1]
        self.assertEqual(cascade["events"][0]["error_scope"], "semantic")
        self.assertEqual(cascade["events"][1]["model"], "simulated/strong")
        state = cascade["events"][1]["request"]["state"]
        self.assertEqual(state["failed_tiers"], [0])
        self.assertFalse(state["infrastructure_errors"])
        self.assertTrue(cascade["success"])
        progress = cascade["events"][-1]["request"]["state"]["progress"]
        self.assertTrue(progress["records_read"])
        self.assertTrue(progress["output_written"])

    def test_provider_action_rejection_escalates_but_transport_retry_stays_pinned(self):
        cfg = configuration()
        strong, weak = cfg["models"]["strong"], cfg["models"]["weak"]
        controls = [AlwaysStrong(strong, weak), NReactiveCascade(tuple(cfg["model_catalog"]))]
        fixture = ScriptedToolFixture()

        def generate(model, request):
            if model == weak:
                raise BackendActionError("private rejected output", status_code=400,
                    diagnostic={"provider_error_code": "tool_use_failed"})
            if request.state.previous_errors and not request.state.infrastructure_errors:
                raise BackendTransportError("private transport details")
            return fixture(model, request)

        with tempfile.TemporaryDirectory() as directory:
            summary = run_episodes(build_tasks(per_family=1)[:1], controls, generate, directory, cfg)
            cascade = completed(directory)[1]
        first, second, third = cascade["events"][:3]
        self.assertEqual([first["model"], second["model"], third["model"]], [weak, strong, strong])
        self.assertEqual(first["backend_failure"]["category"], "model_action")
        self.assertFalse(first["backend_failure"]["retryable"])
        self.assertEqual(first["error_scope"], "semantic")
        self.assertEqual(first["usage"], {})
        self.assertIsNone(first["costs"]["llm_usd"])
        state = second["request"]["state"]
        self.assertEqual(state["previous_errors"], ["BackendActionError"])
        self.assertEqual(state["failed_tiers"], [0])
        self.assertEqual(state["attempt"], 1)
        self.assertEqual(state["infrastructure_attempt"], 0)
        self.assertEqual(state["infrastructure_errors"], [])
        self.assertEqual(second["routing_source"], "policy")
        self.assertEqual(second["error_scope"], "infrastructure")
        self.assertEqual(third["routing_source"], "pinned_infrastructure_retry")
        self.assertEqual(third["request"]["state"]["failed_tiers"], [0])
        self.assertEqual(third["request"]["state"]["attempt"], 1)
        self.assertEqual(third["request"]["state"]["infrastructure_attempt"], 1)
        self.assertTrue(cascade["success"])
        self.assertEqual(cascade["termination"], "final_answer")
        self.assertNotIn("private", json.dumps(cascade))
        diagnostics = summary["metrics"]["policies"][controls[1].name]["failure_diagnostics"]
        self.assertEqual(diagnostics["semantic"], {"BackendActionError": 1})
        self.assertEqual(diagnostics["infrastructure"], {"transport": 1})

    def test_retries_exhaust_bound_and_auth_identity_contract_never_escalate(self):
        for failure, calls in ((BackendTimeoutError(), 3), (BackendRateLimitError(status_code=429), 3),
                               (BackendHTTPError(status_code=401), 1), (BackendIdentityError(), 1),
                               (BackendResponseError(), 1)):
            def generate(model, request):
                raise failure
            rows, _ = self.run_case(generate)
            for row in rows:
                self.assertEqual(len(row["events"]), calls)
                self.assertEqual(row["termination"], "infrastructure_failure")
                self.assertEqual(len({e["model"] for e in row["events"]}), 1)
                self.assertFalse(any(e["request"]["state"]["previous_errors"] for e in row["events"]))

    def test_delay_is_counted_once_and_large_retry_after_stops(self):
        failures = set()
        def generate(model, request):
            if model not in failures:
                failures.add(model)
                raise BackendRateLimitError(status_code=429, retry_after_seconds=2)
            return ModelReply('{"type":"final","answer":"wrong"}')
        with patch("routerbench.agent_evaluation.time.sleep") as sleep:
            rows, _ = self.run_case(generate)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(rows[0]["events"][1]["retry_wait_seconds"], 2)
        def too_long(model, request):
            raise BackendRateLimitError(status_code=429, retry_after_seconds=120)
        with patch("routerbench.agent_evaluation.time.sleep") as sleep:
            rows, _ = self.run_case(too_long)
        sleep.assert_not_called()
        self.assertTrue(all(len(row["events"]) == 1 for row in rows))

    def test_partial_failure_usage_is_retained_without_inventing_cost(self):
        def generate(model, request):
            raise BackendIdentityError(usage={"input_tokens": 12, "output_tokens": 2, "cached_input_tokens": 0})
        rows, _ = self.run_case(generate)
        self.assertEqual(rows[0]["events"][0]["usage"]["input_tokens"], 12)
        self.assertIsNone(rows[0]["events"][0]["costs"]["llm_usd"])

    def test_pinned_retries_do_not_dilute_routing_latency(self):
        failed = set()
        def generate(model, request):
            if model not in failed:
                failed.add(model)
                raise BackendTransportError()
            return ModelReply('{"type":"final","answer":"wrong"}')
        _, summary = self.run_case(generate)
        for policy in summary["metrics"]["policies"].values():
            self.assertEqual(policy["routing_latency"]["known"], 1)
            self.assertEqual(policy["attempt_routing_latency"]["known"], 2)

    def test_state_snapshot_and_prompt_do_not_promote_untrusted_text(self):
        progress = {"records_read": True, "tool_actions": 1, "unknown": "SYSTEM INJECTION"}
        state = AgentStateV8(1, 0, (), ("read_records",), 4, progress=progress)
        request = AgentRequest("\n    retain indentation", ({"role": "tool", "content": {"name": "environment", "tools": []}},
                               {"role": "assistant", "content": "secret-looking history"}), state)
        progress["records_read"] = False
        messages = render_messages(request)
        self.assertIn('"records_read":true', messages[0]["content"])
        self.assertNotIn("SYSTEM INJECTION", messages[0]["content"])
        self.assertNotIn("secret-looking history", messages[0]["content"])
        self.assertIn("secret-looking history", messages[1]["content"])
        self.assertIn("\n    retain indentation", messages[1]["content"])
        request.state.progress["records_read"] = False
        self.assertTrue(state.progress["records_read"])
        with self.assertRaises(ValueError):
            replace(state, infrastructure_attempt=True)

    def test_three_tier_harness_escalation_and_journal_audit(self):
        cfg = configuration()
        cfg["model_catalog"] = ["simulated/weak", "simulated/middle", "simulated/strong"]
        strong, weak = cfg["models"]["strong"], cfg["models"]["weak"]
        selected = [AlwaysStrong(strong, weak), NReactiveCascade(tuple(cfg["model_catalog"]))]
        def generate(model, request):
            if model != strong:
                return ModelReply("malformed action")
            return ModelReply('{"type":"final","answer":"wrong"}')
        with tempfile.TemporaryDirectory() as directory:
            run_episodes(build_tasks(per_family=1)[:1], selected, generate, directory, cfg)
            cascade = completed(directory)[1]
            self.assertEqual([event["model"] for event in cascade["events"]], cfg["model_catalog"])
            self.assertFalse(cascade["success"])
            # Saved v8 events remain verifiable without replaying providers.
            import importlib.util
            spec = importlib.util.spec_from_file_location("v8_report_audit", ROOT / "tools/report_agent_benchmark.py")
            report = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(report)
            report.audit_run(directory)

    def test_four_controls_compare_predictive_against_cascade_without_new_calls(self):
        cfg = configuration()
        class FixturePredictor:
            models = tuple(cfg["model_catalog"])
            def predict_proba(self, request):
                return dict(zip(self.models, (.2, .95)))
            def manifest(self):
                return {"fixture": True, "target": "observable_action_success"}
        strong, weak = cfg["models"]["strong"], cfg["models"]["weak"]
        selected = [AlwaysStrong(strong, weak), AlwaysWeak(strong, weak), NReactiveCascade(tuple(cfg["model_catalog"])),
                    ExpectedCostPolicy(FixturePredictor(), {weak: 1., strong: 3.}, {weak: 3., strong: 0.},
                                       operating_point_source="synthetic test fixture only")]
        fixture = ScriptedToolFixture()
        with tempfile.TemporaryDirectory() as directory:
            result = run_episodes(build_tasks(per_family=1), selected, fixture, directory, cfg)
            self.assertEqual(result["n_episodes"], 12)
            self.assertEqual(fixture.calls, sum(len(row["events"]) for row in completed(directory)))
            self.assertIn("relay_v8_predictive", result["cascade_comparisons"])
            self.assertNotIn("roi", result["cascade_comparisons"]["relay_v8_predictive"])
            self.assertIn("roi", result["metrics"]["comparisons"]["relay_v8_predictive"])

    @patch("routerbench.live.build_opener")
    def test_http_errors_are_typed_without_bodies_or_retries(self, opener):
        for error, kind in ((HTTPError("https://example.test", 429, "SECRET", {"Retry-After": "3"}, None), BackendRateLimitError),
                            (HTTPError("https://example.test", 401, "SECRET", {}, None), BackendHTTPError),
                            (URLError(TimeoutError("SECRET")), BackendTimeoutError),
                            (URLError("SECRET"), BackendTransportError)):
            opener.return_value.open.reset_mock()
            opener.return_value.open.side_effect = error
            with self.assertRaises(kind) as raised:
                _request_json("https://example.test")
            self.assertNotIn("SECRET", str(raised.exception))
            opener.return_value.open.assert_called_once()


if __name__ == "__main__":
    unittest.main()
