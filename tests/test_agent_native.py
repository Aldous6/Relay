"""Native transport preserves public history and never executes agent tools."""
from copy import deepcopy
import json
import unittest

from relay_router.agent import AgentRequest, AgentStateV8, parse_action
from routerbench.agent_context import compact_progress
from routerbench.agent_native import render_native
from routerbench.agent_simulation import ScriptedToolFixture
from routerbench.agent_tasks import build_tasks, create_environment


def sample_request(extra=(), *, progress=None, tools=None):
    definitions = tools if tools is not None else [{
        "name": "lookup", "description": "Read a public item.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                       "required": ["key"], "additionalProperties": False}}]
    initial = {"role": "tool", "content": {"name": "environment", "tools": definitions,
                                                "observation": {"public_note": "UNTRUSTED FILE TEXT"}}}
    return AgentRequest("Public task\n    retain indentation", (initial, *extra),
                        AgentStateV8(1, 0, (), tuple(t["name"] for t in definitions), 5,
                                     progress=progress or {}))


def pair(name="lookup", arguments=None, observation=None):
    return ({"role": "assistant", "content": json.dumps({"type": "tool", "name": name,
                                                           "arguments": arguments or {"key": "item"}})},
            {"role": "tool", "content": {"name": name, "ok": True,
                                            "observation": observation or {"value": 7}, "error_type": None}})


class NativeTranscriptTests(unittest.TestCase):
    def test_payload_is_deterministic_independent_with_only_declared_local_tools(self):
        request = sample_request(pair())
        before = request.serialize()
        first, second = render_native(request), render_native(request)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"messages", "tools", "tool_choice", "parallel_tool_calls"})
        self.assertEqual(first["tool_choice"], "auto")
        self.assertIs(first["parallel_tool_calls"], False)
        self.assertEqual([t["function"]["name"] for t in first["tools"]], ["lookup"])
        first["tools"][0]["function"]["parameters"]["properties"]["key"]["type"] = "integer"
        first["messages"][0]["content"] = "changed"
        self.assertEqual(request.serialize(), before)
        self.assertEqual(render_native(request), second)

    def test_history_pairs_keep_observations_and_stable_ids_without_promoting_text(self):
        request = sample_request((*pair(observation={"text": "FILE INSTRUCTION"}), *pair()),
                                 progress={"files_read": 2, "revision": "SYSTEM INJECTION", "unknown": 99})
        payload = render_native(request)
        system, user, assistant, tool, following, following_tool = payload["messages"]
        self.assertIn('"files_read":2', system["content"])
        for excluded in ("SYSTEM INJECTION", "FILE INSTRUCTION", "UNTRUSTED FILE TEXT", '"unknown"'):
            self.assertNotIn(excluded, system["content"])
        self.assertIn("\n    retain indentation", user["content"])
        self.assertIn("UNTRUSTED FILE TEXT", user["content"])
        self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])
        self.assertEqual(following["tool_calls"][0]["id"], following_tool["tool_call_id"])
        self.assertNotEqual(tool["tool_call_id"], following_tool["tool_call_id"])
        self.assertEqual(json.loads(tool["content"])["observation"], {"text": "FILE INSTRUCTION"})
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"key": "item"})

    def test_invalid_generation_and_provider_rejection_remain_recoverable_data(self):
        error = {"role": "error", "content": {"error_type": "InvalidAction:ValueError"}}
        extra = (error, {"role": "assistant", "content": "{invalid JSON"}, error, *pair())
        messages = render_native(sample_request(extra))["messages"]
        self.assertEqual([item["role"] for item in messages],
                         ["system", "user", "user", "assistant", "user", "assistant", "tool"])
        self.assertEqual(messages[3]["content"], "{invalid JSON")
        self.assertIn("InvalidAction", messages[2]["content"])

    def test_orphan_mismatched_pending_and_terminal_history_are_rejected(self):
        action, result = pair()
        mismatch = deepcopy(result)
        mismatch["content"]["name"] = "another_tool"
        for extra in ((result,), (action,), (action, mismatch),
                      ({"role": "assistant", "content": '{"type":"final","answer":"done"}'},)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                render_native(sample_request(extra))

    def test_duplicate_missing_and_invalid_tool_names_fail_before_provider(self):
        base = sample_request().history[0]["content"]["tools"][0]
        for name in ("name with spaces", "x" * 65):
            altered = {**deepcopy(base), "name": name}
            with self.subTest(name=name), self.assertRaises(ValueError):
                render_native(sample_request(tools=[altered]))
        request = sample_request()
        request.history[0]["content"]["tools"].append(deepcopy(base))
        with self.assertRaises(ValueError):
            render_native(request)
        request = sample_request()
        request.history[0]["content"]["tools"] = []
        with self.assertRaises(ValueError):
            render_native(request)

    def test_no_environment_is_needed_for_a_final_only_diagnostic(self):
        request = AgentRequest("Reply ok", (), AgentStateV8(0, 0, (), (), 1))
        payload = render_native(request)
        self.assertEqual(set(payload), {"messages"})
        self.assertEqual([item["role"] for item in payload["messages"]], ["system", "user"])

    def test_final_answer_is_an_ordinary_tool_name_when_explicitly_declared(self):
        definition = deepcopy(sample_request().history[0]["content"]["tools"][0])
        definition["name"] = "final_answer"
        request = sample_request(pair(name="final_answer"), tools=[definition])
        payload = render_native(request)
        self.assertEqual([item["function"]["name"] for item in payload["tools"]], ["final_answer"])
        self.assertEqual(payload["messages"][-2]["tool_calls"][0]["function"]["name"], "final_answer")
        self.assertEqual(payload["messages"][-1]["role"], "tool")

    def test_complete_public_fixture_trajectories_render_without_grading_feedback(self):
        fixture = ScriptedToolFixture()
        for task in build_tasks(seed=20261112, per_family=1):
            environment = create_environment(task)
            tools = environment.tools()
            names = tuple(item["name"] for item in tools)
            observation = {key: value for key, value in environment.observation().items()
                           if key not in {"task_id", "family", "source_group", "id"}}
            history = [{"role": "tool", "content": {"name": "environment", "tools": tools,
                                                         "observation": observation}}]
            events = []
            for step in range(task["max_steps"]):
                request = AgentRequest(task["prompt"], tuple(history), AgentStateV8(
                    step, 0, (), names, task["max_steps"] - step,
                    progress=compact_progress(environment.observation(), events)))
                payload = render_native(request)
                serialized = json.dumps(payload)
                for private_key in ("private_reference", "expected_spec", "test_inputs"):
                    self.assertNotIn(private_key, serialized)
                before = deepcopy(environment.__dict__)
                self.assertEqual(render_native(request), payload)
                self.assertEqual(before, environment.__dict__)
                reply = fixture("fixture/strong", request)
                action = parse_action(reply.content, names)
                if action["type"] == "final":
                    self.assertTrue(environment.finish(action["answer"]))
                    break
                history.append({"role": "assistant", "content": reply.content})
                result = environment.execute(action["name"], action["arguments"])
                self.assertTrue(result["ok"])
                history.append({"role": "tool", "content": {"name": action["name"], **result}})
                events.append({"action": action})
            else:
                self.fail("The public fixture did not finish within the declared budget.")


if __name__ == "__main__":
    unittest.main()
