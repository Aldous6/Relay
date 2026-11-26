"""Development challenges: public solvability, private grades and bounded tools."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from relay_router.agent import AgentRequest, AgentStateV8, AlwaysStrong, ModelReply
from routerbench import agent_evaluation as evaluation
from routerbench.agent_challenges import CHALLENGE_FAMILIES, build_challenge_tasks, validate_challenge_task
from routerbench.agent_context import compact_progress
from routerbench.agent_native import render_native
from routerbench.agent_tasks import FAMILIES, build_tasks, create_environment, validate_task_splits


ROOT = Path(__file__).resolve().parents[1]


def public_plan(state):
    """Independent scripted solver using public dependencies only."""
    graph = {row["id"]: set(row["requires"]) for row in state["nodes"]}
    needed = set(state["requested"])
    while True:
        expanded = needed | set().union(*(graph[name] for name in needed))
        if expanded == needed:
            break
        needed = expanded
    order = []
    while len(order) < len(needed):
        ready = sorted(name for name in needed - set(order) if graph[name] <= set(order))
        if not ready:
            raise AssertionError("Invalid public graph")
        order.extend(ready)
    return order


def public_allocations(state):
    """Exhaustive fixture solver; no hidden optimum or task object is received."""
    capacities = {machine["id"]: machine["capacity"] for machine in state["machines"]}
    answers = []
    for choices in product(*(sorted(job["costs"]) for job in state["jobs"])):
        loads = {machine: 0 for machine in capacities}
        total = 0
        assignments = []
        for job, machine in zip(state["jobs"], choices):
            assignments.append({"job": job["id"], "machine": machine})
            loads[machine] += job["load"]
            total += job["costs"][machine]
        if all(loads[machine] <= capacity for machine, capacity in capacities.items()):
            answers.append((total, assignments))
    return sorted(answers, key=lambda answer: answer[0])


def public_actions(environment):
    """Yield actions and consume public tool results, never the private task."""
    observation = environment.observation()
    family = observation["family"]
    if family == "table_join":
        customers = yield "read_table", {"name": "customers"}
        orders = yield "read_table", {"name": "orders"}
        region = re.search(r"region=(north|south)", observation["prompt"]).group(1)
        selected = {row["id"] for row in customers["rows"] if row["active"] and row["region"] == region}
        rows = [row for row in orders["rows"] if row["customer_id"] in selected and row["status"] == "paid"]
        yield "submit_summary", {"order_count": len(rows), "total": sum(row["amount"] for row in rows)}
    elif family == "dependency_plan":
        graph = yield "read_graph", {}
        yield "submit_plan", {"order": public_plan(graph)}
        yield "check_plan", {}
    else:
        allocation = yield "read_allocation", {}
        yield "submit_assignment", {"assignments": public_allocations(allocation)[0][1]}
        yield "check_assignment", {}


def solve_public(environment, before_action=None):
    actions = public_actions(environment)
    result, events = None, []
    while True:
        if before_action:
            before_action(events)
        try:
            name, arguments = actions.send(result)
        except StopIteration:
            return "done"
        action = {"type": "tool", "name": name, "arguments": arguments}
        outcome = environment.execute(name, arguments)
        if not outcome["ok"]:
            raise AssertionError(outcome)
        events.append({"action": action, "result": {"name": name, **outcome}})
        result = outcome["observation"]


class ChallengeTests(unittest.TestCase):
    def family_tasks(self):
        return {task["family"]: task for task in build_challenge_tasks(per_family=1)}

    def test_deterministic_json_cases_leave_old_families_unchanged(self):
        tasks = build_challenge_tasks()
        self.assertEqual(tasks, build_challenge_tasks())
        self.assertNotEqual(tasks, build_challenge_tasks(seed=43))
        self.assertEqual(json.loads(json.dumps(tasks)), tasks)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task["id"] for task in tasks}), 12)
        self.assertEqual({family: sum(task["family"] == family for task in tasks)
                          for family in CHALLENGE_FAMILIES}, dict.fromkeys(CHALLENGE_FAMILIES, 4))
        self.assertEqual(FAMILIES, ("file_lookup", "aggregate_records", "repair_function"))
        self.assertEqual({task["family"] for task in build_tasks()}, set(FAMILIES))
        for arguments in ({"seed": True}, {"per_family": False}, {"per_family": 0}, {"per_family": 1.5}):
            with self.assertRaises(ValueError):
                build_challenge_tasks(**arguments)

    def test_cases_are_solvable_from_public_data_without_host_execution(self):
        for seed in (42, 711, 20261112):
            for task in build_challenge_tasks(seed=seed, per_family=4):
                with self.subTest(task=task["id"]):
                    environment = create_environment(task)
                    with patch("builtins.open", side_effect=AssertionError("No host files")), \
                            patch("builtins.eval", side_effect=AssertionError("No generated code")), \
                            patch("subprocess.run", side_effect=AssertionError("No commands")):
                        answer = solve_public(environment)
                        self.assertTrue(environment.finish(answer))
                    self.assertEqual(environment.observation()["successful_tool_calls"], 3)

    def test_complete_native_trajectories_are_small_and_exclude_private_grades(self):
        for task in build_challenge_tasks(seed=20261112, per_family=4):
            environment = create_environment(task)
            initial = {key: value for key, value in environment.observation().items()
                       if key not in {"task_id", "family", "source_group", "id"}}
            tools = environment.tools()
            def inspect(events):
                history = [{"role": "tool", "content": {"name": "environment", "tools": tools,
                                                          "observation": initial}}]
                for event in events:
                    history.extend([{"role": "assistant", "content": json.dumps(event["action"])},
                                    {"role": "tool", "content": event["result"]}])
                context = len(json.dumps({"prompt": task["prompt"], "history": history}, ensure_ascii=False,
                                         sort_keys=True, separators=(",", ":")).encode("utf-8"))
                request = AgentRequest(task["prompt"], tuple(history), AgentStateV8(
                    len(events), 0, (), tuple(tool["name"] for tool in tools), task["max_steps"] - len(events),
                    context_utf8_bytes=context, progress=compact_progress(environment.observation(), events)))
                payload = render_native(request)
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for private in ("private_reference", "optimal_cost", "required_nodes", "initial_state"):
                    self.assertNotIn(private, encoded)
                # Same conservative byte reservation as the free-tier pilot.
                self.assertLessEqual(len(encoded.encode("utf-8")) + 2048 + 512, 8000,
                                     (task["id"], len(events)))
            answer = solve_public(environment, inspect)
            self.assertTrue(environment.finish(answer))

    def test_join_requires_reading_both_tables_and_actual_saved_summary(self):
        task = self.family_tasks()["table_join"]
        environment = create_environment(task)
        expected = task["private_reference"]["answer"]
        environment.execute("submit_summary", expected)
        self.assertFalse(environment.finish("done"))
        environment.execute("read_table", {"name": "customers"})
        self.assertFalse(environment.finish("done"))
        environment.execute("read_table", {"name": "orders"})
        self.assertTrue(environment.finish("done"))
        environment.execute("submit_summary", {**expected, "total": expected["total"] + 1})
        self.assertFalse(environment.finish("done"))
        self.assertFalse(environment.finish(json.dumps(expected)))

    def test_public_dependency_failures_and_new_revisions_require_new_check(self):
        environment = create_environment(self.family_tasks()["dependency_plan"])
        graph = environment.execute("read_graph", {})["observation"]
        correct = public_plan(graph)
        environment.execute("submit_plan", {"order": list(reversed(correct))})
        failed = environment.execute("check_plan", {})
        self.assertFalse(failed["observation"]["all_passed"])
        self.assertEqual(environment.verification_error("check_plan", failed), "ConstraintViolation")
        self.assertFalse(environment.finish("done"))
        environment.execute("submit_plan", {"order": correct})
        self.assertIsNone(environment.observation()["progress"]["tested_revision"])
        self.assertFalse(environment.finish("done"))
        passed = environment.execute("check_plan", {})
        self.assertIsNone(environment.verification_error("check_plan", passed))
        self.assertTrue(environment.finish("done"))
        self.assertIsNone(environment.verification_error("finish", failed))

    def test_suboptimal_publicly_feasible_allocation_fails_only_terminal_private_grade(self):
        task = next(task for task in build_challenge_tasks(per_family=4)
                    if task["family"] == "constrained_allocation"
                    and len({cost for cost, _ in public_allocations(task["initial_state"])}) > 1)
        environment = create_environment(task)
        public = environment.execute("read_allocation", {})["observation"]
        solutions = public_allocations(public)
        environment.execute("submit_assignment", {"assignments": solutions[-1][1]})
        result = environment.execute("check_assignment", {})
        self.assertTrue(result["observation"]["all_passed"])
        self.assertEqual(set(result["observation"]), {"all_passed", "violations", "total_cost", "revision"})
        self.assertIsNone(environment.verification_error("check_assignment", result))
        before = deepcopy(environment.__dict__)
        self.assertFalse(environment.finish("done"))
        self.assertEqual(environment.__dict__, before)
        environment.execute("submit_assignment", {"assignments": solutions[0][1]})
        self.assertFalse(environment.finish("done"))
        environment.execute("check_assignment", {})
        self.assertTrue(environment.finish("done"))

    def test_malformed_calls_missing_submission_and_budget_errors_are_atomic(self):
        for task in self.family_tasks().values():
            environment = create_environment(task)
            name = environment.tools()[0]["name"]
            attempts = [("exec", {"code": "open('secret')"}), ([], {}), (name, None), (name, {"extra": 1})]
            if task["family"] == "table_join":
                attempts += [("read_table", {"name": "../../host.txt"}),
                             ("submit_summary", {"order_count": True, "total": 1}),
                             ("submit_summary", {"order_count": 2, "total": 0.5})]
            elif task["family"] == "dependency_plan":
                attempts += [("submit_plan", {"order": [1]}), ("submit_plan", {"order": ["x"] * 11}),
                             ("check_plan", {})]
            else:
                attempts += [("submit_assignment", {"assignments": [{"job": "j0"}]}),
                             ("submit_assignment", {"assignments": [{"job": "j0", "machine": True}]}),
                             ("check_assignment", {})]
            for name, arguments in attempts:
                before = deepcopy(environment.__dict__)
                result = environment.execute(name, arguments)
                self.assertEqual(set(result), {"ok", "observation", "error_type"})
                self.assertFalse(result["ok"])
                self.assertIsNone(result["observation"])
                self.assertIsInstance(result["error_type"], str)
                self.assertEqual(before, environment.__dict__)
            name = environment.tools()[0]["name"]
            arguments = {"name": "customers"} if name == "read_table" else {}
            for _ in range(task["max_steps"]):
                self.assertTrue(environment.execute(name, arguments)["ok"])
            before = deepcopy(environment.__dict__)
            self.assertEqual(environment.execute(name, arguments)["error_type"], "StepLimitExceeded")
            self.assertEqual(environment.__dict__, before)

    def test_external_mutations_and_repeated_environments_are_isolated(self):
        task = self.family_tasks()["constrained_allocation"]
        first, second = create_environment(task), create_environment(task)
        task["initial_state"]["jobs"].clear()
        first.execute("read_allocation", {})["observation"]["jobs"].clear()
        first.tools()[0]["parameters"]["properties"]["injected"] = {}
        first.observation()["progress"]["output"] = {"injected": True}
        self.assertEqual(second.observation()["successful_tool_calls"], 0)
        self.assertFalse(second.finish("done"))
        self.assertTrue(first.finish(solve_public(first)))
        self.assertTrue(second.finish(solve_public(second)))

    def test_invalid_tasks_cannot_create_inconsistent_or_unbounded_environments(self):
        tasks = self.family_tasks()
        malformed = []
        for task in tasks.values():
            malformed.extend(({**task, "max_steps": True}, {**task, "extra": 1},
                              {**task, "private_reference": {}}, {**task, "initial_state": {}}))
        cycle = deepcopy(tasks["dependency_plan"])
        cycle["initial_state"]["nodes"][0]["requires"] = [cycle["initial_state"]["nodes"][0]["id"]]
        malformed.append(cycle)
        missing = deepcopy(tasks["dependency_plan"])
        missing["initial_state"]["requested"] = ["absent"]
        malformed.append(missing)
        duplicate = deepcopy(tasks["table_join"])
        duplicate["initial_state"]["customers"][1]["id"] = duplicate["initial_state"]["customers"][0]["id"]
        malformed.append(duplicate)
        infeasible = deepcopy(tasks["constrained_allocation"])
        for job in infeasible["initial_state"]["jobs"]:
            job["load"], job["costs"] = 5, {"m0": 1}
        for machine in infeasible["initial_state"]["machines"]:
            machine["capacity"] = 1
        malformed.append(infeasible)
        for task in malformed:
            with self.subTest(family=task["family"]), self.assertRaises(ValueError):
                validate_challenge_task(task)

    def test_split_validator_accepts_new_families_without_claiming_independence(self):
        tasks = build_challenge_tasks(per_family=2)
        partitions = {"fit": tasks[:2], "calibration": tasks[2:4], "evaluation": tasks[4:]}
        result = validate_task_splits(partitions, require_unseen_families=True)
        self.assertFalse(result["independence_certified"])
        self.assertEqual(result["partitions"]["evaluation"]["families"], ["constrained_allocation"])
        altered = deepcopy(partitions)
        altered["evaluation"][0]["source_group"] = altered["fit"][0]["source_group"]
        with self.assertRaises(ValueError):
            validate_task_splits(altered)

    def test_harness_never_retries_private_terminal_failure_for_new_families(self):
        config = json.loads((ROOT / "configs/agent_benchmark_v1.json").read_text())
        config["bootstrap_replicates"] = 100
        calls = []
        def wrong(model, request):
            calls.append(request.serialize())
            for private in ("private_reference", "optimal_cost", "required_nodes", '"task_id"', '"family"'):
                self.assertNotIn(private, request.serialize())
            return ModelReply('{"type":"final","answer":"done"}')
        tasks = build_challenge_tasks(per_family=1)
        with tempfile.TemporaryDirectory() as directory:
            evaluation.run_episodes(tasks, [AlwaysStrong("simulated/strong", "simulated/weak")],
                                    wrong, directory, config)
            rows = [record["data"] for record in evaluation._unseal(Path(directory) / "journal.json")["records"]
                    if record["kind"] == "episode_completed"]
        self.assertEqual(len(calls), len(tasks))
        for row in rows:
            self.assertFalse(row["success"])
            self.assertEqual(row["termination"], "final_answer")
            self.assertEqual(len(row["events"]), 1)
            self.assertIsNone(row["events"][0]["error_type"])


if __name__ == "__main__":
    unittest.main()
