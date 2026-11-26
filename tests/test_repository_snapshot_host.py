import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import repository_snapshot_host as host


class SnapshotHostTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.task = {"task_id": "example", "source": "example/project", "revision": "a" * 40}
        self.folder = self.root / "example"
        self.snapshot = self.folder / "snapshot"
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "code.py").write_bytes(b"x = 1\r\n")
        (self.snapshot / host.PUBLIC).write_text("raise SystemExit(10)\n", encoding="utf-8")
        (self.folder / "private_check.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.config = {**self.task, "editable_paths": ["code.py"], "image": "example/runner@sha256:" + "b" * 64}
        self.write_config()
        environment = patch.dict(host.os.environ, RELAY_REPOSITORY_CASES=str(self.root))
        environment.start()
        self.addCleanup(environment.stop)

    def write_config(self):
        (self.folder / "case.json").write_text(json.dumps(self.config), encoding="utf-8")

    def test_freeze_has_no_private_test_in_agent_workspace(self):
        with patch.object(host, "_docker", return_value=["docker"]), patch.object(host.subprocess, "run") as run:
            run.return_value.stdout = b"linux\n"
            snapshot, verifier, pins = host.prepare(self.task)
        self.assertEqual(snapshot, self.snapshot)
        self.assertEqual([name for name, _ in verifier.protected_files], [host.PUBLIC])
        self.assertTrue(all(not path.is_relative_to(snapshot) for path in pins.values()))
        self.assertNotIn(str(self.folder / "private_check.py"), verifier.checks[0].argv)
        self.assertTrue(host.preflight({"tasks": [self.task]})["complete"])
        self.config["image"] = "example/runner:latest"
        self.write_config()
        self.assertFalse(host.preflight({"tasks": [self.task]})["complete"])

    def test_edits_require_existing_allowlisted_path_and_exact_match(self):
        action = SimpleNamespace(operation="replace", path="code.py", find="x = 1", replacement="x = 2")
        host._file_operation(self.snapshot, ["code.py"], action)
        self.assertEqual((self.snapshot / "code.py").read_bytes(), b"x = 2\r\n")
        for path in ("../private_check.py", host.PUBLIC, "code.py"):
            action.path = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                host._file_operation(self.snapshot, ["code.py"], action)

    def test_private_check_mounts_only_snapshot_and_one_checker_and_classifies_exit(self):
        with patch.object(host, "_docker", return_value=["docker"]), patch.object(host.subprocess, "run") as run:
            for code, outcome in ((0, "passed"), (10, "failed"), (1, "inconclusive"), (125, "inconclusive")):
                run.return_value.returncode = code
                self.assertEqual(host.private_verdict(self.task, self.snapshot), outcome)
            args = run.call_args.args[0]
            self.assertIn("--pull=never", args)
            self.assertIn("--network=none", args)
            self.assertIn("--read-only", args)
            self.assertEqual(args.count("--mount"), 2)
            self.assertNotIn("--privileged", args)
            self.assertNotIn("--env", args)

    def test_preflight_reports_all_missing_cases_without_starting_containers(self):
        with patch.object(host.subprocess, "run", side_effect=AssertionError):
            cohort = {"tasks": [self.task, {**self.task, "task_id": "missing"}]}
            result = host.preflight(cohort)
            self.assertFalse(result["complete"])
            self.assertEqual([row["prepared"] for row in result["cases"]], [True, False])

    @unittest.skipUnless(importlib.util.find_spec("openhands"), "OpenHands SDK is optional")
    def test_real_sdk_builds_fresh_metered_agents_and_resolves_file_tool_offline(self):
        settings = {"GROQ_API_KEY": "offline-fixture", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                    "OPENHANDS_SUPPRESS_BANNER": "1"}
        with patch.dict(host.os.environ, settings), patch("socket.socket.connect", side_effect=AssertionError):
            from openhands.sdk.tool.registry import resolve_tool
            from relay_router.integrations.openhands_cascade import _validate_agent
            for size in (20, 120):
                context = SimpleNamespace(workspace=self.snapshot, model=f"groq/openai/gpt-oss-{size}b")
                agent = host.build_agent(self.task, context)
                llm = _validate_agent(agent, context.model)
                self.assertEqual((llm.max_output_tokens, llm.temperature, llm.reasoning_effort), (1024, 0, "low"))
                tool = resolve_tool(agent.tools[0], None)[0]
                self.assertEqual(tool.name, "repository_files")
                action = tool.action_type(operation="read", path="code.py")
                self.assertFalse(tool.executor(action).is_error)


if __name__ == "__main__":
    unittest.main()
