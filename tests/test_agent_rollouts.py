"""Offline collector contracts; fixture outcomes are not quality evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from relay_router.agent import AlwaysStrong, AlwaysWeak, ModelReply
from relay_router.errors import BackendTransportError, BackendActionError
from routerbench.agent_evaluation import run_episodes, object_hash, _unseal
from routerbench.agent_rollouts import (
    initial_anchors, select_anchors, select_followup_anchors, collect_paired_rollouts,
    replay_prefix, request_from_dict, remaining_provider_cost, CONTINUATION, TARGET,
)
from routerbench.agent_simulation import ScriptedToolFixture
from routerbench.agent_tasks import build_tasks, create_environment

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    config = json.loads((ROOT / "configs/agent_benchmark_v1.json").read_text())
    config.update(schema_version=2, model_catalog=["simulated/weak", "simulated/strong"],
                  harness={"contract_version": 2, "max_transport_retries": 1, "retry_delay_seconds": 0},
                  bootstrap_replicates=100)
    return config


def tariffs(models):
    return {"as_of": "fixture-only", "models": {m: {
        "input_usd_per_million": i+1., "output_usd_per_million": 2*(i+1.),
        "source": "fictitious tariff, not provider invoice"} for i, m in enumerate(models)}}


def events(output, result, candidate):
    branch = next(b for b in result["branches"] if b["candidate"] == candidate)
    records = _unseal(Path(output) / "branches" / branch["branch_id"] / "journal.json")["records"]
    return [r["data"] for r in records if r["kind"] == "call_completed"]


class PairedRolloutTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()
        self.models = self.cfg["model_catalog"]
        self.tasks = [build_tasks(per_family=1)[1]]
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def collect(self, callback=None, anchors=None, output=None, execute=True, **kwargs):
        return collect_paired_rollouts(self.tasks, anchors or initial_anchors(self.tasks, self.cfg),
            self.models, callback or ScriptedToolFixture(), self.cfg, output or self.root / "collection",
            max_generation_calls=kwargs.pop("max_generation_calls", 128),
            tariffs=kwargs.pop("tariffs", tariffs(self.models)), execute=execute, **kwargs)

    def source(self):
        controls = [AlwaysStrong(self.models[-1], self.models[0]), AlwaysWeak(self.models[-1], self.models[0])]
        run_episodes(self.tasks, controls, ScriptedToolFixture(), self.root / "source", self.cfg)
        return self.root / "source"

    def test_prepare_is_pure_and_same_initial_request_for_every_candidate(self):
        fixture = ScriptedToolFixture()
        anchors = initial_anchors(self.tasks, self.cfg)
        self.assertNotIn("private_reference", json.dumps(anchors["anchors"][0]["request"]))
        plan = self.collect(fixture, anchors, execute=False)
        self.assertEqual(fixture.calls, 0)
        self.assertEqual(plan["maximum_generation_calls"], 12)
        self.assertEqual(plan["branches"], 2)
        result = self.collect(fixture, anchors)
        first = [events(self.root / "collection", result, m)[0] for m in self.models]
        self.assertEqual(first[0]["request"], first[1]["request"])
        self.assertEqual(first[0]["request_sha256"], anchors["anchors"][0]["request_sha256"])
        self.assertTrue(result["training_ready"])
        self.assertTrue(all(r["success"] and r["target"] == TARGET for r in result["rows"]))
        self.assertTrue(all(r["assignment"] == "synthetic_fixture" for r in result["rows"]))

    def test_semantic_error_escalates_but_reference_continuation_is_common(self):
        result = self.collect()
        weak = events(self.root / "collection", result, self.models[0])
        strong = events(self.root / "collection", result, self.models[-1])
        self.assertEqual(weak[0]["error_scope"], "semantic")
        self.assertEqual(weak[1]["model"], self.models[-1])
        # Strong probe succeeds; fixed continuation goes back to reference weak.
        self.assertEqual(strong[1]["model"], self.models[0])
        self.assertEqual(result["continuation_policy"], CONTINUATION)
        cost = next(r["remaining_provider_cost_usd"] for r in result["rows"] if r["model"] == self.models[0])
        self.assertAlmostEqual(cost, .00014 + 3*.00028)

    def test_transport_pins_candidate_and_missing_usage_stays_unknown(self):
        fixture = ScriptedToolFixture()
        failures = set()
        def generate(model, request):
            if request.state.step == 0 and not request.state.previous_errors and model not in failures:
                failures.add(model)
                raise BackendTransportError()
            return fixture(model, request)
        result = self.collect(generate)
        for model in self.models:
            calls = events(self.root / "collection", result, model)
            self.assertEqual([c["model"] for c in calls[:2]], [model, model])
            self.assertEqual(calls[1]["routing_source"], "pinned_infrastructure_retry")
            self.assertFalse(calls[1]["request"]["state"]["previous_errors"])
        self.assertTrue(all(row["remaining_provider_cost_usd"] is None for row in result["rows"]))

    def test_provider_semantic_failure_is_not_censored(self):
        fixture = ScriptedToolFixture()
        def generate(model, request):
            if model == self.models[0] and not request.state.previous_errors and request.state.step == 0:
                raise BackendActionError(status_code=400)
            return fixture(model, request)
        result = self.collect(generate)
        self.assertFalse(result["censored"])
        self.assertIsNone(result["rows"][0]["remaining_provider_cost_usd"])
        calls = events(self.root / "collection", result, self.models[0])
        self.assertEqual(calls[1]["model"], self.models[-1])

    def test_terminal_transport_remains_operational_failure_and_censored(self):
        def generate(model, request):
            raise BackendTransportError()
        result = self.collect(generate)
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["generation_calls"], 4)
        self.assertTrue(all(b["operational_success"] is False and b["censored_reason"] == "infrastructure_failure"
                            for b in result["branches"]))

    def test_transport_that_consumes_last_call_is_censored(self):
        self.cfg["max_calls"] = 1
        def generate(model, request):
            raise BackendTransportError()
        result = self.collect(generate)
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["rows"], [])
        self.assertTrue(all(b["censored_reason"] == "infrastructure_budget_exhausted" for b in result["branches"]))

    def test_unknown_actual_model_cannot_use_requested_model_tariff(self):
        episode = {"events": [{"model": self.models[0], "usage": {"input_tokens": 10, "output_tokens": 10},
                               "backend_failure": {"category": "model_identity"}}]}
        self.assertIsNone(remaining_provider_cost(episode, tariffs(self.models), self.models))

    def test_incorrect_final_and_exhausted_budget_are_real_negative_labels(self):
        def wrong(model, request):
            return ModelReply('{"type":"final","answer":"wrong"}', usage={"input_tokens": 7, "output_tokens": 9})
        result = self.collect(wrong)
        self.assertTrue(result["training_ready"])
        self.assertTrue(all(r["success"] is False for r in result["rows"]))
        self.assertEqual(result["generation_calls"], 2)
        def invalid(model, request):
            return ModelReply("bad", usage={"input_tokens": 7, "output_tokens": 9})
        exhausted = self.collect(invalid, output=self.root / "exhausted")
        self.assertTrue(all(b["termination"] == "call_budget_exhausted" for b in exhausted["branches"]))
        self.assertEqual(exhausted["generation_calls"], 12)
        self.assertTrue(exhausted["training_ready"])

    def test_fixed_source_positions_restore_mutations_and_never_charge_prefix(self):
        source = self.source()
        source_bytes = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
        anchors = select_anchors(source, decision_indices=(2, 20))
        self.assertEqual(len(anchors["anchors"]), 1)
        self.assertEqual(len(anchors["unavailable"]), 1)
        original = anchors["anchors"][0]["request"]
        self.assertTrue(original["state"]["progress"]["records_read"])
        self.assertEqual(original["state"]["attempt"], 0)
        self.assertEqual(original["state"]["failed_tiers"], [0])
        result = self.collect(anchors=anchors)
        for model in self.models:
            calls = events(self.root / "collection", result, model)
            self.assertEqual(calls[0]["request"], original)
            self.assertEqual(calls[0]["action"]["name"], "write_json")
            self.assertEqual(len(calls), 2)
        self.assertTrue(all(r["success"] for r in result["rows"]))
        self.assertEqual(source_bytes, {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()})

    def test_followup_anchors_reuse_fixed_probe_positions(self):
        initial = self.collect()
        anchors = select_followup_anchors(self.root / "collection", decision_indices=(2, 99))
        self.assertEqual(len(anchors["anchors"]), 1)
        self.assertEqual(len(anchors["unavailable"]), 1)
        self.assertEqual(anchors["anchors"][0]["source_group"], self.tasks[0]["source_group"])
        followup = self.collect(anchors=anchors, output=self.root / "followup")
        self.assertEqual(len(initial["rows"]), len(followup["rows"]))
        self.assertTrue(all(r["success"] for r in followup["rows"]))

    def test_replay_never_grades_and_rejects_different_observations(self):
        anchors = select_anchors(self.source(), decision_indices=(2,))
        request = request_from_dict(anchors["anchors"][0]["request"])
        def environment_factory(task):
            environment = create_environment(task)
            def fail(answer):
                raise AssertionError("private final grader consulted while replaying")
            environment.finish = fail
            return environment
        environment, prefix = replay_prefix(self.tasks[0], request, environment_factory)
        self.assertTrue(environment.observation()["progress"]["records_read"])
        self.assertEqual(len(prefix), 1)
        damaged = request.to_dict()
        damaged["history"][-1]["content"]["observation"]["records"] = []
        with self.assertRaisesRegex(ValueError, "observacion"):
            replay_prefix(self.tasks[0], request_from_dict(damaged), environment_factory)

    def test_incomplete_and_modified_source_are_rejected(self):
        source = self.source()
        anchors = select_anchors(source)
        modified = deepcopy(anchors)
        modified["anchors"].pop()
        fixture = ScriptedToolFixture()
        with self.assertRaises(ValueError):
            self.collect(fixture, modified)
        self.assertEqual(fixture.calls, 0)
        (source / "summary.json").unlink()
        with self.assertRaises(FileNotFoundError):
            select_anchors(source)

    def test_preflight_rejects_budget_or_invalid_tariff_before_callback(self):
        fixture = ScriptedToolFixture()
        with self.assertRaisesRegex(ValueError, "presupuesto global"):
            self.collect(fixture, max_generation_calls=11)
        invalid = tariffs(self.models)
        invalid["models"][self.models[0]]["input_usd_per_million"] = float("nan")
        with self.assertRaisesRegex(ValueError, "tarifas"):
            self.collect(fixture, tariffs=invalid)
        self.assertEqual(fixture.calls, 0)

    def test_interrupted_attempt_is_never_reissued_and_cannot_be_negative_label(self):
        fixture = ScriptedToolFixture()
        attempts = []
        def generate(model, request):
            attempts.append((model, request.serialize()))
            if len(attempts) == 1:
                raise KeyboardInterrupt()
            return fixture(model, request)
        with self.assertRaises(KeyboardInterrupt):
            self.collect(generate)
        result = self.collect(generate)
        self.assertEqual(sum(model == self.models[0] and req == attempts[0][1] for model, req in attempts), 1)
        self.assertEqual(len(result["censored"]), 1)
        self.assertEqual(len(result["rows"]), 1)
        self.assertFalse(result["training_ready"])
        total = len(attempts)
        again = self.collect(generate)
        self.assertEqual(again, result)
        self.assertEqual(len(attempts), total)

    def test_n_model_catalog_uses_next_tier(self):
        self.models.insert(1, "simulated/middle")
        result = self.collect()
        calls = events(self.root / "collection", result, self.models[0])
        self.assertEqual(calls[1]["model"], "simulated/middle")
        self.assertEqual(len(result["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
