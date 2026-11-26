"""Synthetic environment safety, actual grading, isolation and split contracts."""
from copy import deepcopy
import json
import re
import unittest
from unittest.mock import patch

from routerbench.agent_tasks import build_tasks, create_environment, validate_task_splits


def tasks_by_family():
    return {task["family"]: task for task in build_tasks(per_family=1)}


def public_repair_spec(prompt):
    operator = re.search(r"operator=(add|subtract|multiply)", prompt).group(1)
    return {"operator": operator,
            **{name: int(re.search(rf"{name}=(-?\d+)", prompt).group(1)) for name in ("offset", "lower", "upper")}}


def solve_from_public_tools(environment):
    """A scripted integration solver, deliberately never given task/private data."""
    visible = environment.observation()
    prompt, family = visible["prompt"], visible["family"]
    if family == "file_lookup":
        target = re.search(r"Record: ([\w-]+)", prompt).group(1)
        paths = environment.execute("list_files", {})["observation"]["paths"]
        for path in paths:
            content = environment.execute("read_file", {"path": path})["observation"]["content"]
            if f"Record: {target}\n" in content:
                return re.search(r"Access code: (\S+)", content).group(1)
        raise AssertionError("The requested virtual record is absent")
    if family == "aggregate_records":
        source = re.search(r"input/[\w-]+\.json", prompt).group(0)
        destination = re.search(r"output/[\w-]+\.json", prompt).group(0)
        category = re.search(r"category=(\w+)", prompt).group(1)
        rows = environment.execute("read_records", {"path": source})["observation"]["records"]
        amounts = [row["amount"] for row in rows if row["category"] == category]
        result = environment.execute("write_json", {"path": destination,
                                     "document": {"category": category, "count": len(amounts), "total": sum(amounts)}})
        if not result["ok"]:
            raise AssertionError(result)
        return "done"
    environment.execute("inspect_function", {})
    result = environment.execute("patch_function", public_repair_spec(prompt))
    if not result["ok"]:
        raise AssertionError(result)
    result = environment.execute("run_tests", {})
    if not result["observation"]["all_passed"]:
        raise AssertionError(result)
    return "done"


class AgentTaskTests(unittest.TestCase):
    def test_generation_is_deterministic_serializable_and_covers_three_families(self):
        tasks = build_tasks()
        self.assertEqual(tasks, build_tasks())
        self.assertNotEqual(tasks, build_tasks(seed=43))
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task["id"] for task in tasks}), 12)
        self.assertEqual(json.loads(json.dumps(tasks)), tasks)
        self.assertEqual({family: sum(task["family"] == family for task in tasks)
                          for family in ("file_lookup", "aggregate_records", "repair_function")},
                         {"file_lookup": 4, "aggregate_records": 4, "repair_function": 4})
        for kwargs in ({"seed": True}, {"per_family": False}, {"per_family": 0}, {"per_family": 1.5}):
            with self.assertRaises(ValueError):
                build_tasks(**kwargs)

    def test_every_episode_is_solvable_using_only_public_prompt_and_tool_results(self):
        for task in build_tasks(per_family=6):
            with self.subTest(task=task["id"]):
                env = create_environment(task)
                with patch("builtins.open", side_effect=AssertionError("No host file access")), \
                        patch("builtins.eval", side_effect=AssertionError("No evaluation")):
                    answer = solve_from_public_tools(env)
                    self.assertTrue(env.finish(answer))
                self.assertLess(env.observation()["successful_tool_calls"], task["max_steps"])

    def test_private_material_is_not_in_initial_observation_or_tools(self):
        for task in build_tasks():
            env = create_environment(task)
            serialized = json.dumps({"observation": env.observation(), "tools": env.tools()})
            for key in ("private_reference", "expected_spec", "test_inputs", "initial_state"):
                self.assertNotIn(key, serialized)
            if task["family"] == "file_lookup":
                self.assertNotIn(task["private_reference"]["answer"], serialized)
            self.assertEqual(env.observation()["successful_tool_calls"], 0)

    def test_lookup_requires_listing_and_reading_requested_file_not_a_claimed_answer(self):
        task = tasks_by_family()["file_lookup"]
        env = create_environment(task)
        answer, target = task["private_reference"]["answer"], task["private_reference"]["path"]
        self.assertFalse(env.finish(answer))
        env.execute("list_files", {})
        other = next(path for path in task["initial_state"]["files"] if path != target)
        env.execute("read_file", {"path": other})
        self.assertFalse(env.finish(answer))
        env.execute("read_file", {"path": target})
        self.assertFalse(env.finish("PASS"))
        self.assertTrue(env.finish(answer.lower()))

    def test_aggregation_checks_written_json_and_requires_input_read(self):
        task = tasks_by_family()["aggregate_records"]
        env = create_environment(task)
        state, expected = task["initial_state"], task["private_reference"]["answer"]
        env.execute("write_json", {"path": state["output_path"], "document": expected})
        self.assertFalse(env.finish("done"))
        env.execute("read_records", {"path": state["records_path"]})
        self.assertTrue(env.finish("done"))
        env.execute("write_json", {"path": state["output_path"], "document": {**expected, "total": expected["total"] + 1}})
        self.assertFalse(env.finish("done"))
        self.assertFalse(env.finish(json.dumps(expected)))

    def test_repair_checks_current_revision_and_actual_arithmetic(self):
        task = tasks_by_family()["repair_function"]
        env = create_environment(task)
        spec = public_repair_spec(env.observation()["prompt"])
        initial = env.execute("run_tests", {})
        self.assertFalse(initial["observation"]["all_passed"])
        self.assertEqual(env.verification_error("run_tests", initial), "TestFailure")
        self.assertFalse(env.finish("all tests passed"))
        env.execute("inspect_function", {})
        env.execute("patch_function", spec)
        self.assertFalse(env.finish("done"))
        result = env.execute("run_tests", {})
        self.assertTrue(result["observation"]["all_passed"])
        self.assertIsNone(env.verification_error("run_tests", result))
        self.assertTrue(env.finish("done"))
        env.execute("patch_function", {**spec, "offset": spec["offset"] + 1})
        self.assertIsNone(env.observation()["progress"]["tested_revision"])
        self.assertFalse(env.finish("done"))
        failed = env.execute("run_tests", {})
        self.assertLess(failed["observation"]["passed"], failed["observation"]["total"])
        self.assertEqual(env.verification_error("run_tests", failed), "TestFailure")
        self.assertIsNone(env.verification_error("finish", failed))
        self.assertIsNone(env.verification_error("run_tests", {"ok": False, "observation": None}))
        self.assertFalse(env.finish("done"))

    def test_invalid_actions_have_no_side_effects_and_code_is_not_a_tool(self):
        task = tasks_by_family()["repair_function"]
        env = create_environment(task)
        spec = public_repair_spec(env.observation()["prompt"])
        attempts = [("exec", {"code": "open('secret')"}), ([], {}),
                    ("patch_function", None), ("patch_function", list(spec)),
                    ("patch_function", {**spec, "extra": "ignored?"}),
                    ("patch_function", {**spec, "offset": True}),
                    ("patch_function", {**spec, "operator": "__import__('os').system('x')"}),
                    ("patch_function", {**spec, "lower": 20, "upper": -20}),
                    ("run_tests", {"passed": True})]
        for name, args in attempts:
            before = deepcopy(env.__dict__)
            result = env.execute(name, args)
            self.assertEqual(set(result), {"ok", "observation", "error_type"})
            self.assertFalse(result["ok"])
            self.assertIsNone(result["observation"])
            self.assertIsInstance(result["error_type"], str)
            self.assertEqual(env.__dict__, before)
        lookup = create_environment(tasks_by_family()["file_lookup"])
        before = deepcopy(lookup.__dict__)
        self.assertFalse(lookup.execute("read_file", {"path": "../../host.txt"})["ok"])
        self.assertEqual(lookup.__dict__, before)

    def test_resets_inputs_tool_results_and_public_objects_do_not_share_mutable_state(self):
        original = tasks_by_family()["aggregate_records"]
        task = deepcopy(original)
        first, second = create_environment(task), create_environment(task)
        task["initial_state"]["records"].clear()
        rows = first.execute("read_records", {"path": original["initial_state"]["records_path"]})["observation"]["records"]
        rows.clear()
        first.observation()["progress"]["output"] = {"fake": True}
        first.tools()[0]["parameters"]["properties"].clear()
        self.assertEqual(second.observation()["successful_tool_calls"], 0)
        self.assertTrue(first.finish(solve_from_public_tools(first)))
        self.assertFalse(second.finish("done"))
        self.assertTrue(second.finish(solve_from_public_tools(second)))
        self.assertEqual(original, tasks_by_family()["aggregate_records"])

    def test_tool_budget_and_invalid_output_shapes_are_checked_before_mutation(self):
        task = tasks_by_family()["aggregate_records"]
        env = create_environment(task)
        path, expected = task["initial_state"]["output_path"], task["private_reference"]["answer"]
        for document in ({**expected, "count": True}, {**expected, "extra": 1}, {**expected, "total": 1.2}):
            before = deepcopy(env.__dict__)
            self.assertFalse(env.execute("write_json", {"path": path, "document": document})["ok"])
            self.assertEqual(env.__dict__, before)
        task["max_steps"] = 1
        env = create_environment(task)
        env.execute("read_records", {"path": task["initial_state"]["records_path"]})
        before = deepcopy(env.__dict__)
        self.assertEqual(env.execute("write_json", {"path": path, "document": expected})["error_type"], "StepLimitExceeded")
        self.assertEqual(env.__dict__, before)

    def test_splits_keep_sources_and_whole_episodes_disjoint_and_report_families(self):
        tasks = build_tasks(per_family=2)
        splits = {"fit": tasks[:2], "calibration": tasks[2:4], "evaluation": tasks[4:]}
        report = validate_task_splits(splits, require_unseen_families=True)
        self.assertTrue(report["whole_episodes"])
        self.assertFalse(report["independence_certified"])
        self.assertEqual(report["partitions"]["evaluation"]["families"], ["repair_function"])
        same_family = {"fit": [tasks[0]], "evaluation": [tasks[1]]}
        self.assertFalse(validate_task_splits(same_family)["require_unseen_families"])
        with self.assertRaisesRegex(ValueError, "familias"):
            validate_task_splits(same_family, require_unseen_families=True)
        within = deepcopy(splits)
        within["fit"][1]["source_group"] = within["fit"][0]["source_group"]
        self.assertEqual(validate_task_splits(within)["partitions"]["fit"]["source_groups"], 1)
        for field in ("id", "source_group", "prompt"):
            changed = deepcopy(splits)
            value = changed["fit"][0][field]
            changed["evaluation"][0][field] = f"  {value.upper()}  " if field == "prompt" else value
            with self.assertRaises(ValueError):
                validate_task_splits(changed)
        changed = deepcopy(splits)
        changed["fit"][0]["step_index"] = 1
        with self.assertRaisesRegex(ValueError, "episodio"):
            validate_task_splits(changed)

    def test_invalid_tasks_and_split_modes_fail_explicitly(self):
        task = tasks_by_family()["repair_function"]
        for changed in ({**task, "max_steps": True}, {**task, "family": "execute_python"},
                        {**task, "initial_state": {"code": "print('unsafe')"}}):
            with self.assertRaises(ValueError):
                create_environment(changed)
        for splits, strict in (({}, False), ({"fit": []}, False), ({"fit": [task]}, True)):
            with self.assertRaises(ValueError):
                validate_task_splits(splits, require_unseen_families=strict)


if __name__ == "__main__":
    unittest.main()
