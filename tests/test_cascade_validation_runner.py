"""Frozen-cohort and transport-audit contracts; no SDK/provider generation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import run_cascade_validation as runner


class CascadeValidationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        shutil.copytree(runner.DEFAULT_SUITE.parent, self.inputs)
        self.suite_path = self.inputs / "fresh_suite.json"
        self.evaluators_path = self.inputs / "private_evaluators.json"
        self.suite = runner.read(self.suite_path)
        self.output = self.root / "run"
        self.output.mkdir()

    def replace_json(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def prepare(self):
        # Source fingerprints are tested independently of concurrent work on
        # the not-yet-frozen reporter. No generation import or provider access.
        with patch.object(runner, "source_hashes", return_value={"fixture.py": "a" * 64}):
            return runner.prepare_run(self.suite_path, self.evaluators_path, self.output)

    def fixture(self, *, known=True):
        task = self.suite["tasks"][0]
        model = self.suite["models"][0]
        episode = runner._episode_id(task["task_id"], model)
        rates = runner.TARIFFS[model]
        cost = (100 * rates["input_per_million_usd"] + 20 * rates["output_per_million_usd"]) / 1e6 if known else None
        event = {"attempt_id": 1, "episode_id": episode, "model": model,
                 "started_at_utc": "2026-11-18T19:00:00+00:00", "status": "completed" if known else "error",
                 "input_tokens": 100 if known else None, "output_tokens": 20 if known else None,
                 "total_tokens": 120 if known else None, "cost_usd": cost,
                 "quota_wait_seconds": 0., "transport_seconds": .1}
        start = {**event, "status": "started", "input_tokens": None, "output_tokens": None,
                 "total_tokens": None, "cost_usd": None}
        observation = {"task_id": task["task_id"], "group_id": task["group_id"], "workflow": task["workflow"],
                       "model": model, "episode_id": episode, "stage": "cascade",
                       "accepted": known, "public_passed": known, "private_passed": known,
                       "episode_status": "completed" if known else "error",
                       "error_type": None if known else "RuntimeError", "cost_usd": cost,
                       "cost_complete": known, "cost_basis": "tariff_estimate" if known else "unknown",
                       "known_cost_subtotal_usd": cost or 0., "http_attempts": 1,
                       "wall_seconds": .2, "quota_wait_seconds": 0., "final_evaluation_seconds": .1,
                       "runtime_decision_sha256_before_final_evaluation": "b" * 64,
                       "final_evaluation_at_utc": "2026-11-18T19:00:01+00:00", "source_sha256": "c" * 64}
        return observation, event, start

    def store(self, observation, event, start):
        self.replace_json(self.output / "attempt_starts/1.json", start)
        (self.output / "http_attempts.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        (self.output / "observations.jsonl").write_text(json.dumps(observation) + "\n", encoding="utf-8")
        self.replace_json(self.output / "tasks" / observation["task_id"] /
                          (observation["model"].rsplit("/", 1)[-1] + "-observation.json"), observation)

    def test_frozen_run_reopens_only_with_identical_contract(self):
        suite, evaluators, frozen = self.prepare()
        again = self.prepare()[2]
        self.assertEqual(frozen, again)
        self.assertEqual(set(evaluators["tasks"]), {task["task_id"] for task in suite["tasks"]})
        self.assertNotIn("private_tests", json.dumps(frozen["contract"]["initial_public_checks"]))
        with patch.object(runner, "source_hashes", return_value={"fixture.py": "d" * 64}):
            with self.assertRaises(ValueError):
                runner.prepare_run(self.suite_path, self.evaluators_path, self.output)

    def test_frozen_seal_and_content_mutations_are_rejected(self):
        suite, _, frozen = self.prepare()
        tampered = deepcopy(frozen)
        tampered["contract"]["policy_config"]["max_task_cost_usd"] = 100
        with self.assertRaises(ValueError):
            runner.verify_frozen(suite, tampered)
        suite["tasks"][0]["initial_source"] += "\n# changed\n"
        with self.assertRaises(ValueError):
            runner.verify_frozen(suite, frozen)

    def test_private_evaluator_cannot_be_replaced_by_self_consistent_new_hash(self):
        evaluators = runner.read(self.evaluators_path)
        item = next(iter(evaluators["tasks"].values()))
        item["private_tests"] = "assert True  # substituted evaluator\n"
        item["sha256"] = hashlib.sha256(item["private_tests"].encode()).hexdigest()
        self.replace_json(self.evaluators_path, evaluators)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_mutating_agent_task_without_its_prompt_digest_is_rejected(self):
        changed = deepcopy(self.suite)
        changed["tasks"][0]["instruction"] += " changed task"
        with self.assertRaises(ValueError):
            runner.validate_suite(changed)

    def test_private_labels_and_duplicate_groups_cannot_enter_suite(self):
        for key in ("private_tests", "gold_source", "accepted"):
            changed = deepcopy(self.suite)
            changed["tasks"][0][key] = "private-sentinel"
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_suite(changed)
        changed = deepcopy(self.suite)
        changed["tasks"][1]["group_id"] = changed["tasks"][0]["group_id"]
        with self.assertRaises(ValueError):
            runner.validate_suite(changed)

    def test_known_and_unknown_usage_remain_distinct(self):
        for known in (True, False):
            observation, event, start = self.fixture(known=known)
            self.store(observation, event, start)
            self.assertEqual(runner.read_observations(self.suite, self.output), [observation])
            actual = runner.reconcile_http(self.suite, self.output, [observation])
            self.assertEqual(actual, [event])
            self.assertIs(actual[0]["cost_usd"] is None, not known)

    def test_actual_request_without_final_record_is_not_resumable(self):
        observation, event, start = self.fixture()
        self.store(observation, event, start)
        (self.output / "http_attempts.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            runner.reconcile_http(self.suite, self.output, [observation])

    def test_start_identity_must_equal_its_final_transport_record(self):
        for key, bad in (("episode_id", "another-episode"), ("model", self.suite["models"][1]),
                         ("started_at_utc", "2020-01-01T00:00:00+00:00")):
            observation, event, start = self.fixture()
            start[key] = bad
            self.store(observation, event, start)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.reconcile_http(self.suite, self.output, [observation])

    def test_nonfinite_cost_and_invalid_token_types_are_rejected(self):
        for bad in (float("nan"), float("inf"), -1., True):
            observation, event, start = self.fixture()
            event["cost_usd"] = observation["cost_usd"] = bad
            self.store(observation, event, start)
            with self.subTest(cost=bad), self.assertRaises(ValueError):
                runner.reconcile_http(self.suite, self.output, [observation])
        observation, event, start = self.fixture()
        event.update(input_tokens=True, output_tokens=20, total_tokens=21)
        rates = runner.TARIFFS[event["model"]]
        event["cost_usd"] = observation["cost_usd"] = (rates["input_per_million_usd"] + 20 * rates["output_per_million_usd"]) / 1e6
        self.store(observation, event, start)
        with self.assertRaises(ValueError):
            runner.reconcile_http(self.suite, self.output, [observation])

    def test_declared_http_and_token_caps_are_verified_on_read(self):
        observation, event, start = self.fixture()
        self.store(observation, event, start)
        for key, cap in (("max_total_http_attempts", 0), ("max_observed_tokens_per_model", 119)):
            changed = deepcopy(self.suite)
            changed[key] = cap
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.reconcile_http(changed, self.output, [observation])

    def test_duplicate_and_changed_observation_identity_is_rejected(self):
        observation, event, start = self.fixture()
        self.store(observation, event, start)
        with (self.output / "observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observation) + "\n")
        with self.assertRaises(ValueError):
            runner.read_observations(self.suite, self.output)
        observation["episode_id"] = "different-episode"
        self.store(observation, event, start)
        with self.assertRaises(ValueError):
            runner.read_observations(self.suite, self.output)

    def test_success_requires_public_private_and_completed_outcome(self):
        for field, bad in (("private_passed", False), ("public_passed", False), ("episode_status", "error")):
            observation, event, start = self.fixture()
            observation[field] = bad
            self.store(observation, event, start)
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.read_observations(self.suite, self.output)

    def test_observation_jsonl_must_match_its_per_branch_artifact(self):
        observation, event, start = self.fixture()
        self.store(observation, event, start)
        per_branch = self.output / "tasks" / observation["task_id"] / (observation["model"].rsplit("/", 1)[-1] + "-observation.json")
        changed = deepcopy(observation)
        changed["cost_usd"] *= 10
        self.replace_json(per_branch, changed)
        with self.assertRaises(ValueError):
            runner.read_observations(self.suite, self.output)

    def sdk_fixture(self, *, cheap_passes):
        # Real SDK, tools, subprocess public/final checks, runtime, transport
        # accounting and reporter inputs. Only provider replies are simulated.
        from litellm import ModelResponse
        import openhands.sdk.llm.llm as sdk_llm
        from openhands.sdk.tool.builtins.finish import FinishTool
        task = {"task_id": "HumanEval-999", "group_id": "offline-fixture-group", "workflow": "function-implementation",
                "instruction": "Implement add(a, b) returning their sum.",
                "initial_source": "def add(a, b):\n    return 0\n", "public_tests": "assert add(2, 3) == 5\n"}
        private_source = "assert add(9, 9) == 18  # FINAL_ONLY_SENTINEL\n"
        suite = deepcopy(self.suite)
        suite["tasks"] = [task]
        evaluators = {"tasks": {task["task_id"]: {"private_tests": private_source,
                                               "sha256": hashlib.sha256(private_source.encode()).hexdigest()}}}
        source_hashes = {"offline-fixture.py": "e" * 64}
        frozen = {"schema_version": 1, "created_at_utc": "2026-11-18T19:00:00+00:00",
                  "contract": {"suite_content_sha256": runner.digest(suite), "source_sha256": source_hashes,
                               "private_evaluators_content_sha256": runner.digest(evaluators)}}
        frozen["freeze_id"] = runner.digest(frozen)
        cheap, strong = suite["models"]
        seen = []

        def completion(**kwargs):
            serial = json.dumps(kwargs, default=str)
            self.assertNotIn("FINAL_ONLY_SENTINEL", serial)
            self.assertNotIn(private_source, serial)
            is_cheap = str(kwargs["model"]).endswith("gpt-oss-20b")
            seen.append(cheap if is_cheap else strong)
            if is_cheap and not cheap_passes:
                name = FinishTool.create()[0].name
                arguments = {"message": "Fixture stopped without fixing the defect"}
            else:
                tool = next(tool for tool in kwargs["tools"] if "source" in tool["function"].get("parameters", {}).get("properties", {}))
                name = tool["function"]["name"]
                arguments = {"source": "def add(a, b):\n    return a + b\n"}
            return ModelResponse(id="offline-validation", model=kwargs["model"],
                choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": "offline-tool", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})

        with patch.object(runner, "source_hashes", return_value=source_hashes), \
             patch.object(socket.socket, "connect", side_effect=AssertionError("No network allowed in integration fixture")), \
             patch.object(sdk_llm, "litellm_completion", side_effect=completion) as transport:
            status = runner.execute(suite, evaluators, frozen, self.output, key="offline-test-key")
            self.assertEqual(status, 0)
            self.assertEqual(transport.call_count, 2)
            self.assertEqual(seen, [cheap, strong])
        observations = runner.read_observations(suite, self.output)
        events = runner.reconcile_http(suite, self.output, observations)
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(runner.rows(self.output / "task_completions.jsonl")), 1)
        runtime = runner.read(self.output / "tasks/HumanEval-999/cascade/result.json")
        self.assertEqual(runtime["selected_model"], cheap if cheap_passes else strong)
        self.assertIs(runtime["escalated"], not cheap_passes)
        self.assertIsNone(runtime["accepted"])
        self.assertTrue(runtime["original_snapshot_unchanged"])
        by_model = {row["model"]: row for row in observations}
        self.assertIs(by_model[cheap]["accepted"], cheap_passes)
        self.assertTrue(by_model[strong]["accepted"])
        self.assertEqual(by_model[strong]["stage"], "control" if cheap_passes else "cascade")
        for observation in observations:
            self.assertEqual(observation["runtime_decision_sha256_before_final_evaluation"],
                             runner.file_hash(self.output / "tasks/HumanEval-999/cascade/result.json"))
        metadatalog = (self.output / "http_attempts.jsonl").read_text() + (self.output / "observations.jsonl").read_text()
        self.assertNotIn("offline-test-key", metadatalog)
        self.assertNotIn("FINAL_ONLY_SENTINEL", metadatalog)
        expected = sum(row["cost_usd"] for row in observations if row["stage"] == "cascade")
        self.assertAlmostEqual(runtime["accounting"]["cost_usd"], expected)

    def test_real_sdk_cascade_fallback_and_final_evaluator_separation(self):
        self.sdk_fixture(cheap_passes=False)

    def test_real_sdk_cheap_success_then_missing_baseline_control(self):
        self.sdk_fixture(cheap_passes=True)


if __name__ == "__main__":
    unittest.main()
