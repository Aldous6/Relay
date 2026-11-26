"""Trajectory isolation, accounting boundaries and exactly-once recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from relay_router.agent import AlwaysStrong, AlwaysWeak, EscalateOnError, ModelReply
from routerbench import agent_evaluation as evaluation
from routerbench.agent_tasks import build_tasks
from routerbench.agent_simulation import ScriptedToolFixture

ROOT = Path(__file__).resolve().parents[1]


def config():
    value = json.loads((ROOT / "configs/agent_benchmark_v1.json").read_text())
    value["bootstrap_replicates"] = 100
    return value


def policies():
    return [cls("simulated/strong", "simulated/weak") for cls in (AlwaysStrong, AlwaysWeak, EscalateOnError)]


def episodes(directory):
    return [r["data"] for r in evaluation._unseal(Path(directory) / "journal.json")["records"]
            if r["kind"] == "episode_completed"]


class EpisodeEvaluationTests(unittest.TestCase):
    def test_windows_transient_commit_retries_file_only(self):
        original = Path.replace
        attempts = []
        def transient(path, destination):
            attempts.append(str(destination))
            if len(attempts) < 3:
                raise PermissionError("temporary reader lock")
            return original(path, destination)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            with patch.object(Path, "replace", transient), patch.object(evaluation, "os", SimpleNamespace(name="nt", fsync=evaluation.os.fsync)), patch.object(evaluation.time, "sleep") as sleep:
                evaluation._write(destination, {"committed": True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(destination.read_text()), {"committed": True})

    def test_persistent_commit_failure_is_bounded_and_retains_previous_record(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            evaluation._write(destination, {"committed": "previous"})
            with patch.object(Path, "replace", side_effect=PermissionError("locked")) as replace, patch.object(evaluation, "os", SimpleNamespace(name="nt", fsync=evaluation.os.fsync)), patch.object(evaluation.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    evaluation._write(destination, {"committed": "new"})
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(sleep.call_count, 7)
            self.assertEqual(json.loads(destination.read_text()), {"committed": "previous"})

    def test_multistep_smoke_and_repeat_make_no_new_generation(self):
        tasks, generate = build_tasks(per_family=1), ScriptedToolFixture()
        with tempfile.TemporaryDirectory() as directory:
            evaluation.prepare_episodes(tasks, policies(), directory, config(), generate=generate)
            self.assertEqual(generate.calls, 0)
            result = evaluation.run_episodes(tasks, policies(), generate, directory, config())
            self.assertEqual(result["n_episodes"], 9)
            data = episodes(directory)
            self.assertTrue(all(row["success"] and len(row["events"]) >= 3 for row in data))
            self.assertTrue(all(row["events"][0]["request"]["state"]["step"] == 0 for row in data))
            for row in data:
                initial = row["events"][0]["request"]["history"][0]["content"]["observation"]
                self.assertEqual(initial["successful_tool_calls"], 0)
            for value in result["metrics"]["comparisons"].values():
                self.assertEqual(value["roi"]["adoption_status"], "not_assessed_simulation")
            count = generate.calls
            self.assertEqual(result, evaluation.run_episodes(tasks, policies(), generate, directory, config()))
            self.assertEqual(generate.calls, count)

    def test_cascade_uses_past_error_but_weak_control_stays_weak(self):
        task = build_tasks(per_family=1)[1]
        with tempfile.TemporaryDirectory() as directory:
            evaluation.run_episodes([task], policies(), ScriptedToolFixture(), directory, config())
            rows = {e["policy"]: e for e in episodes(directory)}
            events = rows["cheap_then_escalate"]["events"]
            self.assertEqual(events[0]["model"], "simulated/weak")
            self.assertTrue(events[0]["error_type"].startswith("InvalidAction"))
            self.assertEqual(events[1]["model"], "simulated/strong")
            self.assertTrue(events[1]["is_retry"])
            self.assertTrue(events[1]["request"]["state"]["previous_errors"])
            self.assertTrue(all(e["model"] == "simulated/weak" for e in rows["always_weak"]["events"]))

    def test_hidden_answer_never_exposed_and_wrong_final_never_retried(self):
        task = build_tasks(per_family=1)[0]
        calls = []
        def wrong(model, request):
            calls.append((model, request.serialize()))
            self.assertNotIn("private_reference", request.serialize())
            self.assertNotIn('"task_id"', request.serialize())
            self.assertNotIn('"family"', request.serialize())
            self.assertNotIn(task["private_reference"]["answer"], request.serialize())
            return ModelReply('{"type":"final","answer":"valid JSON but wrong"}')
        with tempfile.TemporaryDirectory() as directory:
            evaluation.run_episodes([task], policies(), wrong, directory, config())
            self.assertEqual(len(calls), 3)
            for episode in episodes(directory):
                self.assertFalse(episode["success"])
                self.assertEqual(episode["termination"], "final_answer")
                self.assertEqual(len(episode["events"]), 1)
                self.assertIsNone(episode["events"][0]["error_type"])

    def test_callback_mutations_do_not_change_later_requests(self):
        delegate = ScriptedToolFixture()
        def mutation(model, request):
            self.assertNotIn("injected", request.serialize())
            reply = delegate(model, request)
            request.history[0]["content"]["injected"] = True
            return reply
        with tempfile.TemporaryDirectory() as directory:
            evaluation.run_episodes(build_tasks(per_family=1)[:1], policies(), mutation, directory, config())
            self.assertTrue(all(e["success"] for e in episodes(directory)))

    def test_pending_generation_is_not_reissued_on_resume(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            def interrupted(model, request):
                saved = evaluation._unseal(Path(directory) / "journal.json")
                self.assertEqual(saved["records"][-1]["kind"], "call_started")
                calls.append(model)
                raise KeyboardInterrupt()
            args = (build_tasks(per_family=1)[:1], policies()[:1], interrupted, directory, config())
            with self.assertRaises(KeyboardInterrupt):
                evaluation.run_episodes(*args)
            result = evaluation.run_episodes(*args)
            self.assertEqual(len(calls), 1)
            episode = episodes(directory)[0]
            self.assertEqual(episode["termination"], "interrupted_not_retried")
            self.assertFalse(episode["success"])
            self.assertIsNone(episode["events"][0]["costs"]["total_usd"])
            self.assertIsNone(episode["end_to_end_seconds"])
            self.assertEqual(result["metrics"]["policies"]["always_strong"]["coverage"]["unknown_cost_episodes"], 1)

    def test_durable_final_success_is_preserved_after_interrupted_close(self):
        generate = ScriptedToolFixture()
        original = evaluation._Journal.append
        def interrupted_close(journal, kind, key, data):
            if kind == "episode_completed":
                raise KeyboardInterrupt()
            original(journal, kind, key, data)
        with tempfile.TemporaryDirectory() as directory:
            args = (build_tasks(per_family=1)[:1], policies()[:1], generate, directory, config())
            with patch.object(evaluation._Journal, "append", interrupted_close):
                with self.assertRaises(KeyboardInterrupt):
                    evaluation.run_episodes(*args)
            count = generate.calls
            evaluation.run_episodes(*args)
            self.assertEqual(generate.calls, count)
            self.assertTrue(episodes(directory)[0]["success"])
            self.assertIsNone(episodes(directory)[0]["end_to_end_seconds"])

    def test_bad_config_rejected_before_callback_or_output(self):
        bad = []
        value = config(); value["pricing"]["simulated/weak"]["input_per_million"] = -1; bad.append(value)
        value = config(); value["compute_rates"]["router_usd_per_second"] = True; bad.append(value)
        value = config(); value["roi"]["max_success_rate_loss"] = 2; bad.append(value)
        value = config(); value["bootstrap_replicates"] = 99; bad.append(value)
        value = config(); value["seed"] = True; bad.append(value)
        value = config(); value["independent_validation"] = True; bad.append(value)
        value = config(); value["policy_order"] = "adapt_to_results"; bad.append(value)
        for changed in bad:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                generate = ScriptedToolFixture()
                output = Path(directory) / "run"
                with self.assertRaises(ValueError):
                    evaluation.run_episodes(build_tasks(per_family=1)[:1], policies(), generate, output, changed)
                self.assertEqual(generate.calls, 0)
                self.assertFalse(output.exists())

    def test_cyclic_order_is_prespecified_and_audit_does_not_enter_requests(self):
        class AuditedStrong(AlwaysStrong):
            def select(self, request):
                self.last_audit = {"private_router_audit": request.state.remaining_calls}
                return super().select(request)
        chosen = policies()
        chosen[0] = AuditedStrong("simulated/strong", "simulated/weak")
        def wrong(model, request):
            self.assertNotIn("private_router_audit", request.serialize())
            return ModelReply('{"type":"final","answer":"wrong"}')
        cfg = config(); cfg["policy_order"] = "cyclic_by_task"
        with tempfile.TemporaryDirectory() as directory:
            evaluation.run_episodes(build_tasks(per_family=1), chosen, wrong, directory, cfg)
            rows = episodes(directory)
            expected = [p.name for i in range(3) for p in chosen[i:]+chosen[:i]]
            self.assertEqual([e["policy"] for e in rows], expected)
            event = rows[0]["events"][0]
            self.assertIn("routing_audit", event)
            chosen[0].last_audit["private_router_audit"] = -1
            self.assertGreater(event["routing_audit"]["private_router_audit"], 0)
            with self.assertRaises(ValueError):
                evaluation.run_episodes(build_tasks(per_family=1), chosen[::-1], wrong, directory, cfg)

    def test_invalid_reply_or_usage_records_unknown_consumption_without_poisoning_resume(self):
        for response in ("not ModelReply", ModelReply("{}", {"input_tokens": -1}), ModelReply("{}", {"input_tokens": True})):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                calls = []
                def invalid(model, request):
                    calls.append(model)
                    return response
                cfg = config(); cfg["max_calls"] = 2
                args = (build_tasks(per_family=1)[:1], policies(), invalid, directory, cfg)
                result = evaluation.run_episodes(*args)
                self.assertEqual(len(calls), 6)
                self.assertFalse(any(e["success"] for e in episodes(directory)))
                self.assertTrue(all(e["events"][0]["reported_llm_cost_usd"] is None for e in episodes(directory)))
                self.assertEqual(result, evaluation.run_episodes(*args))
                self.assertEqual(len(calls), 6)

    def test_tampered_journal_and_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            generate = ScriptedToolFixture()
            tasks = build_tasks(per_family=1)[:1]
            evaluation.run_episodes(tasks, policies()[:1], generate, directory, config())
            count = generate.calls
            changed = config(); changed["max_calls"] += 1
            with self.assertRaises(ValueError):
                evaluation.run_episodes(tasks, policies()[:1], generate, directory, changed)
            path = Path(directory) / "journal.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["payload"]["records"][0]["data"]["model"] = "other/model"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluation.run_episodes(tasks, policies()[:1], generate, directory, config())
            self.assertEqual(generate.calls, count)

    def test_concurrent_writer_cannot_start_a_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            generate = ScriptedToolFixture()
            with evaluation._single_writer(directory):
                with self.assertRaises(RuntimeError):
                    evaluation.run_episodes(build_tasks(per_family=1)[:1], policies(), generate, directory, config())
            self.assertEqual(generate.calls, 0)

    def test_observed_test_failure_can_escalate_but_final_gold_is_unused(self):
        task = build_tasks(per_family=1)[2]
        calls = []
        def wrong_patch_then_tests(model, request):
            calls.append(model)
            public = [e["content"] for e in request.history if e["role"] == "tool" and isinstance(e["content"], dict)]
            seen = [e.get("name") for e in public]
            if "inspect_function" not in seen:
                action = {"type": "tool", "name": "inspect_function", "arguments": {}}
            elif "patch_function" not in seen:
                action = {"type": "tool", "name": "patch_function", "arguments": {"operator": "multiply", "offset": 10, "lower": -10, "upper": 10}}
            elif "run_tests" not in seen:
                action = {"type": "tool", "name": "run_tests", "arguments": {}}
            else:
                action = {"type": "final", "answer": "done"}
            return ModelReply(json.dumps(action))
        with tempfile.TemporaryDirectory() as directory:
            evaluation.run_episodes([task], [policies()[0], policies()[2]], wrong_patch_then_tests, directory, config())
            cascade = episodes(directory)[1]
            self.assertEqual([e["model"] for e in cascade["events"]], ["simulated/weak"]*3+["simulated/strong"])
            self.assertEqual(cascade["events"][2]["error_type"], "VerificationFailed:TestFailure")
            self.assertFalse(cascade["success"])


if __name__ == "__main__":
    unittest.main()
