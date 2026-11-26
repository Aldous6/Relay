"""Pinned source fixtures only: no benchmark answers, providers or host tools."""
from copy import deepcopy
import decimal
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from routerbench.bfcl_data import DEFAULT_SNAPSHOT, read_snapshot
from routerbench.bfcl_environment import BFCLRuntime, BFCLCompatibilityError, BFCLResourceLimit, decode_call, _native_schema


def fixture(*classes):
    return {"id": "multi_turn_base_fixture", "question": [[{"role": "user", "content": "Synthetic integration only"}]],
            "initial_config": {}, "involved_classes": list(classes or ("TicketAPI",)), "path": [],
            "excluded_function": []}


class BFCLEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (Path(DEFAULT_SNAPSHOT) / "manifest.json").exists():
            raise unittest.SkipTest("Explicit pinned BFCL snapshot is not imported.")
        cls.metadata = read_snapshot(DEFAULT_SNAPSHOT)

    def runtime(self, *classes, entry=None):
        return BFCLRuntime(DEFAULT_SNAPSHOT, entry or fixture(*classes))

    def test_original_docs_and_explicit_native_aliases_preserve_defaults_and_order(self):
        entry = fixture("TicketAPI")
        entry["excluded_function"] = ["create_ticket"]
        runtime = self.runtime(entry=entry)
        path = Path(self.metadata["package_root"]) / "bfcl_eval/data/multi_turn_func_doc/ticket_api.json"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(runtime.tools(), original)
        self.assertIn("create_ticket", [tool["name"] for tool in runtime.tools()])
        native = runtime.native_tools()
        self.assertEqual([tool["function"]["name"] for tool in native], [tool["name"] for tool in original])
        for output, source in zip(native, original):
            self.assertEqual(output["function"]["description"], source["description"])
            self.assertEqual(output["function"]["parameters"]["required"], source["parameters"]["required"])
            self.assertEqual(output["function"]["parameters"]["type"], "object")
        sample = {"type": "dict", "default": {"type": "dict"},
                  "properties": {"literal": {"type": "float", "default": 1.5}}}
        converted = _native_schema(sample)
        self.assertEqual(converted["default"], sample["default"])
        self.assertEqual(converted["properties"]["literal"]["type"], "number")
        runtime.tools()[0]["parameters"]["required"].clear()
        self.assertEqual(runtime.tools(), original)

    def test_ast_accepts_only_bare_calls_and_data_literals(self):
        call = decode_call("f(1, -2.5, x={'a': [True, None]}, y=('text', 3))", {"f"})
        self.assertEqual(call, {"name": "f", "args": [1, -2.5],
                                "arguments": {"x": {"a": [True, None]}, "y": ("text", 3)}})
        invalid = ["__import__('os').system('x')", "f(g())", "f(x=1+2)", "f(x=[n for n in []])",
                   "f(*[1])", "f(**{'x':1})", "f(x={'a':1, 'a':2})", "f(x=1,x=2)",
                   "obj.f()", "f.__call__()", "f(x=(lambda:1)())", "f(x=foo)", "f(x=float('nan'))",
                   "f(); open('host')", "f(x={1,2})", "other()"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_call(text, {"f"})

    def test_calls_preserve_official_outputs_and_python_error_semantics_without_eval(self):
        runtime = self.runtime()
        with patch("builtins.eval", side_effect=AssertionError("No generated expression executes")), \
                patch("subprocess.run", side_effect=AssertionError("No host process")):
            login = runtime.execute("ticket_login", {"username": "fixture", "password": "local"})
            created = runtime.execute("create_ticket", {"title": "Example"})
            invalid = runtime.execute("create_ticket", {"absent_argument": 1})
        self.assertEqual(login, {"ok": True, "observation": '{"success": true}', "error_type": None})
        self.assertEqual(json.loads(created["observation"]), {"id": 1, "title": "Example", "description": "",
                          "status": "Open", "priority": 1, "created_by": "fixture"})
        self.assertFalse(invalid["ok"])
        self.assertTrue(invalid["observation"].startswith("Error during execution: TicketAPI.create_ticket()"))
        self.assertEqual(invalid["error_type"], "TypeError")
        # Public structured errors feed routing telemetry; their text is unchanged.
        error = self.runtime().execute("create_ticket", {"title": "Without login"})
        self.assertFalse(error["ok"])
        self.assertEqual(error["error_type"], "UpstreamToolError")
        self.assertIn("error", json.loads(error["observation"]))

    def test_vehicle_public_error_preserves_exact_official_observation_state_and_grade(self):
        runtime = self.runtime("VehicleControlAPI")
        original = runtime._classes["VehicleControlAPI"]()
        original._load_scenario({}, long_context=False)
        expected = json.dumps(original.fillFuelTank(fuelAmount=-1))
        vehicle = runtime._instances["VehicleControlAPI"]
        before = deepcopy({key: value for key, value in vars(vehicle).items() if not key.startswith("_")})
        result = runtime.execute("fillFuelTank", {"fuelAmount": -1})
        self.assertEqual(result, {"ok": False, "observation": expected, "error_type": "UpstreamToolError"})
        self.assertEqual(expected, '{"error": "Fuel amount cannot be negative."}')
        self.assertEqual(before, {key: value for key, value in vars(vehicle).items() if not key.startswith("_")})
        # The official checker consumes observations, independently of our routing metadata.
        calls = [['fillFuelTank(fuelAmount=-1)']]
        self.assertEqual(runtime.evaluate([calls], calls), {"valid": True})

    def test_only_nonempty_structured_error_changes_routing_metadata(self):
        runtime = self.runtime()
        for payload in ("error: ordinary text", {"message": "error mentioned as data"}, {"error": ""},
                        {"error": None}, {"error": False}, {"authentication_status": False}):
            def fixture_method(instance):
                return deepcopy(payload)
            with self.subTest(payload=payload), patch.object(runtime._classes["TicketAPI"],
                    "ticket_get_login_status", new=fixture_method):
                result = runtime.execute("ticket_get_login_status", {})
            self.assertTrue(result["ok"])
            self.assertIsNone(result["error_type"])
            self.assertEqual(result["observation"], payload if type(payload) is str else json.dumps(payload))

    def test_batch_safety_validation_is_atomic_but_runtime_failures_keep_order(self):
        runtime = self.runtime()
        calls = [{"name": "ticket_login", "arguments": {"username": "a", "password": "b"}},
                 {"name": "exec", "arguments": {"code": "unsafe"}}]
        with self.assertRaises(ValueError):
            runtime.execute_batch(calls)
        self.assertEqual(runtime.execute("ticket_get_login_status", {})["observation"], '{"login_status": false}')
        calls = [{"name": "ticket_login", "arguments": {"username": "a", "password": "b"}},
                 {"name": "create_ticket", "arguments": {"unknown": "runtime type error"}},
                 {"name": "create_ticket", "arguments": {"title": "survives"}}]
        results = runtime.execute_batch(calls)
        self.assertEqual([row["ok"] for row in results], [True, False, True])
        self.assertEqual(json.loads(results[-1]["observation"])["id"], 1)

    def test_json_arguments_and_keyword_syntax_fail_before_mutation(self):
        runtime = self.runtime()
        for args in ({"title": float("nan")}, {1: "value"}, {"title": object()}, {"title": (1, 2)},
                     {"bad-key": 1}, {"class": 1}):
            with self.subTest(arguments=str(args)):
                self.assertFalse(runtime.execute("create_ticket", args)["ok"])
        self.assertEqual(runtime.execute("ticket_get_login_status", {})["observation"], '{"login_status": false}')

    def test_function_looking_text_inside_literals_remains_data(self):
        runtime = self.runtime()
        runtime.execute("ticket_login", {"username": "a", "password": "b"})
        text = "ticket_login(username='ignored', password='ignored'); __import__('os')"
        result = runtime.execute("create_ticket", {"title": text})
        self.assertEqual(json.loads(result["observation"])["title"], text)
        self.assertEqual(runtime.decode_call(runtime.encode_call("create_ticket", {"title": text}))["arguments"],
                         {"title": text})

    def test_checker_accepts_valid_state_and_rejects_missing_response_and_wrong_state(self):
        runtime = self.runtime()
        calls = [[['ticket_login(username="fixture", password="local")', 'create_ticket(title="Example")']],
                 [['resolve_ticket(ticket_id=1, resolution="fixed")']]]
        reference = [calls[0][0], calls[1][0]]
        self.assertEqual(runtime.evaluate(calls, reference), {"valid": True})
        changed = deepcopy(calls)
        changed[1] = [['close_ticket(ticket_id=1)']]
        failed = runtime.evaluate(changed, reference)
        self.assertFalse(failed["valid"])
        self.assertEqual(failed["error_type"], "multi_turn:instance_state_mismatch")
        readonly = self.runtime()
        self.assertEqual(readonly.evaluate([[[]]], [['ticket_get_login_status()']])["error_type"],
                         "multi_turn:empty_turn_model_response")
        # Unchanged state alone is insufficient: the official response check runs.
        failed = readonly.evaluate([[['get_user_tickets()']]], [['ticket_get_login_status()']])
        self.assertEqual(failed["error_type"], "multi_turn:execution_response_mismatch")
        self.assertEqual(runtime.execute("ticket_get_login_status", {})["observation"], '{"login_status": false}')

    def test_private_reference_is_terminal_only_and_incompatible_reference_blocks(self):
        runtime = self.runtime()
        initial = runtime.tools(), runtime.native_tools()
        record = {"id": "multi_turn_base_fixture", "ground_truth": [['ticket_get_login_status()']]}
        self.assertTrue(runtime.evaluate([[['ticket_get_login_status()']]], record)["valid"])
        self.assertEqual((runtime.tools(), runtime.native_tools()), initial)
        self.assertFalse(runtime.evaluate([], record)["valid"])
        self.assertFalse(runtime.evaluate([[['open("host")']]], record)["valid"])
        for private in ({"id": "different", "ground_truth": [[]]}, [['ticket_login(username=get_secret())']]):
            with self.assertRaises(BFCLCompatibilityError):
                runtime.evaluate([[[]]], private)

    def test_replay_and_separate_instances_preserve_observations(self):
        calls = [{"name": "message_login", "arguments": {"user_id": "USR001"}},
                 {"name": "send_message", "arguments": {"receiver_id": "USR002", "message": "fixture"}},
                 {"name": "get_message_stats", "arguments": {}}]
        first, second = self.runtime("MessageAPI"), self.runtime("MessageAPI")
        observed = first.execute_batch(calls)
        self.assertEqual(second.execute_batch(calls), observed)
        result = first.execute("get_message_stats", {})
        result["observation"] = "external mutation"
        self.assertEqual(second.execute("get_message_stats", {})["observation"], observed[-1]["observation"])

    def test_mutable_defaults_do_not_cross_runtime_or_model_reference_identity(self):
        entry = fixture("TwitterAPI")
        entry["initial_config"] = {"TwitterAPI": {"authenticated": True}}
        model, independent = self.runtime(entry=entry), self.runtime(entry=entry)
        model.execute("post_tweet", {"content": "fixture"})
        model.execute("mention", {"tweet_id": 0, "mentioned_usernames": ["alice"]})
        fresh = independent.execute("post_tweet", {"content": "fixture"})
        self.assertEqual(json.loads(fresh["observation"])["mentions"], [])
        # Gold must not inherit the model's omitted mutable default, even though
        # the unchanged checker requires both instances to have the same type.
        calls = [[['post_tweet(content="fixture")', 'mention(tweet_id=0, mentioned_usernames=["alice"])']]]
        private = [['post_tweet(content="fixture")']]
        result = model.evaluate(calls, private)
        self.assertFalse(result["valid"])
        self.assertEqual(result["error_type"], "multi_turn:instance_state_mismatch")
        self.assertEqual(result["details"]["ground_truth_instance_state"]["tweets"][0]["mentions"], [])
        self.assertTrue(model.evaluate([[private[0]]], private)["valid"])

    def test_math_real_dependency_preserves_result_format_and_restores_global_precision(self):
        import mpmath
        runtime = self.runtime("MathAPI")
        decimal_precision, mp_precision = decimal.getcontext().prec, mpmath.mp.dps
        output = runtime.execute("square_root", {"number": 2, "precision": 12})
        self.assertIn("Decimal('1.41421356237')", output["observation"])
        logarithm = runtime.execute("logarithm", {"value": 100, "base": 10, "precision": 20})
        self.assertIn("mpf('2.0')", logarithm["observation"])
        self.assertEqual(decimal.getcontext().prec, decimal_precision)
        self.assertEqual(mpmath.mp.dps, mp_precision)
        self.assertTrue(runtime.evaluate([[['add(a=2, b=3)']]], [['add(a=2, b=3)']])["valid"])

    def test_all_eight_classes_load_without_sdk_or_persistent_namespace_changes(self):
        previous = types.ModuleType("bfcl_eval")
        with patch.dict(sys.modules, {"bfcl_eval": previous}):
            runtime = self.runtime("GorillaFileSystem", "MathAPI", "MessageAPI", "TwitterAPI", "TicketAPI",
                                   "TradingBot", "TravelAPI", "VehicleControlAPI")
            self.assertIs(sys.modules["bfcl_eval"], previous)
            self.assertGreater(len(runtime.tools()), 50)
        self.assertNotIn("bfcl_eval.model_handler", sys.modules)

    def test_official_checker_and_classes_are_used_without_snapshot_mutation(self):
        before = read_snapshot(DEFAULT_SNAPSHOT)["manifest_sha256"]
        runtime = self.runtime()
        self.assertTrue(runtime._checker.multi_turn_checker.__code__.co_filename.endswith("multi_turn_checker.py"))
        self.assertIn("upstream", runtime._checker.multi_turn_checker.__code__.co_filename)
        self.assertIn("upstream", runtime._classes["TicketAPI"].create_ticket.__code__.co_filename)
        reference = [['ticket_get_login_status()']]
        with patch("builtins.eval", side_effect=AssertionError("Original eval executor must never run")):
            self.assertTrue(runtime.evaluate([[reference[0]]], reference)["valid"])
        self.assertEqual(read_snapshot(DEFAULT_SNAPSHOT)["manifest_sha256"], before)
        self.assertIn("local", runtime.provenance()["scope"])

    def test_math_resource_guard_prevalidates_entire_batch_without_any_effect(self):
        runtime = self.runtime("MathAPI", "TicketAPI")
        expensive = [("square_root", {"number": 2, "precision": 4097}),
                     ("logarithm", {"value": 2, "base": 10, "precision": 10**9}),
                     ("power", {"base": 2, "exponent": 1_000_000}),
                     ("power", {"base": 2., "exponent": 1_000_001.}),
                     ("power", {"base": 2., "exponent": -1_000_001.})]
        for name, arguments in expensive:
            calls = [{"name": "ticket_login", "arguments": {"username": "a", "password": "b"}},
                     {"name": name, "arguments": arguments}]
            with self.subTest(name=name, arguments=arguments):
                with patch.object(runtime, "_invoke", side_effect=AssertionError("No method may run")):
                    with self.assertRaises(BFCLResourceLimit) as caught:
                        runtime.execute_batch(calls)
                self.assertEqual(caught.exception.details()["phase"], "preflight")
                self.assertIsNone(caught.exception.details()["capability_label"])
                self.assertEqual(runtime.execute("ticket_get_login_status", {})["observation"], '{"login_status": false}')
        with self.assertRaises(BFCLResourceLimit):
            runtime.execute("power", {"base": 2, "exponent": 10**9})
        self.assertEqual(json.loads(runtime.execute("power", {"base": 2, "exponent": 8})["observation"]), {"result": 256})

    def test_resource_limits_apply_to_positional_reference_and_model_calls(self):
        runtime = self.runtime("MathAPI")
        reference = [['add(a=1, b=2)']]
        with self.assertRaises(BFCLResourceLimit):
            runtime.evaluate([[['power(2, 1000000000)']]], reference)
        with self.assertRaises(BFCLResourceLimit):
            runtime.evaluate([[['add(a=1, b=2)']]], [['square_root(2, 1000000000)']])

    def test_output_bound_is_technical_censoring_after_effect_not_normal_tool_error(self):
        runtime = self.runtime()
        runtime.execute("ticket_login", {"username": "a", "password": "b"})
        title = "x" * 40000
        runtime.execute("create_ticket", {"title": title})
        runtime.execute("create_ticket", {"title": title})
        with self.assertRaises(BFCLResourceLimit) as caught:
            runtime.execute("get_user_tickets", {})
        self.assertEqual(caught.exception.details()["phase"], "output")
        self.assertEqual(caught.exception.details()["rule"], "output_utf8_bytes_max")
        self.assertEqual(len(runtime._instances["TicketAPI"].ticket_queue), 2)
        self.assertEqual(runtime.provenance()["resource_policy"]["math_precision_max"], 4096)


if __name__ == "__main__":
    unittest.main()
