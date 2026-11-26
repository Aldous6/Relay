"""Bounded local OpenHands episodes for public, small Python tasks.

The AST checks reduce the permitted tool surface; they are not an OS sandbox.
Only fixed host test scripts run, never shell commands supplied by a model.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ALLOWED_IMPORTS = {"math", "cmath", "collections", "heapq", "itertools", "functools", "operator",
                   "bisect", "fractions", "decimal", "re", "string", "typing", "copy", "random"}
DENIED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit"}


def check_source(source):
    if type(source) is not str or not source.strip() or len(source.encode()) > 32768:
        raise ValueError("Provide at most 32 KiB of Python source")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("Import outside the declared standard-library scope")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in ALLOWED_IMPORTS):
            raise ValueError("Import outside the declared standard-library scope")
        if isinstance(node, ast.ImportFrom) and any(alias.name.startswith("_") for alias in node.names):
            raise ValueError("Private imports are outside task scope")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private/introspection attributes are outside task scope")
        if isinstance(node, ast.Name) and (node.id in DENIED_NAMES or node.id.startswith("__")):
            raise ValueError("Dynamic execution, IO and introspection are outside task scope")
        if isinstance(node, ast.ClassDef):
            raise ValueError("These public function tasks do not require classes")
    return tree


def run_check(source, test_code, *, directory, timeout=8):
    """Run trusted host tests with the candidate in a fresh isolated Python process."""
    start = time.perf_counter()
    try:
        check_source(source)
    except (ValueError, SyntaxError) as exc:
        return {"passed": False, "category": type(exc).__name__, "seconds": time.perf_counter() - start,
                "feedback": str(exc)[:300]}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "solution.py").write_text(source, encoding="utf-8")
    script = "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent))\nfrom solution import *\n" + test_code
    path = directory / "host_check.py"
    path.write_text(script, encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT"}}
    try:
        result = subprocess.run([sys.executable, "-I", "-B", str(path)], cwd=directory,
                                env=env, capture_output=True, timeout=timeout)
        feedback = (result.stdout + result.stderr).decode("utf-8", errors="replace")[-2000:]
        return {"passed": result.returncode == 0, "category": "passed" if result.returncode == 0 else "test_failure",
                "seconds": time.perf_counter() - start, "feedback": feedback}
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "test_timeout", "seconds": time.perf_counter() - start,
                "feedback": "Host tests exceeded their fixed timeout."}


_TOOL_CLASS = None


def make_edit_tool(workspace, public_tests, *, stop_on_public_pass=False):
    global _TOOL_CLASS
    from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor, Tool, register_tool
    from pydantic import Field
    if _TOOL_CLASS is None:
        class EditAction(Action):
            source: str = Field(description="Complete replacement Python source for solution.py. Preserve required function names.")
        class EditObservation(Observation):
            pass
        class EditExecutor(ToolExecutor):
            def __init__(self, root, tests, stop):
                self.root, self.tests, self.stop = Path(root), tests, stop
            def __call__(self, action, conversation=None):
                result = run_check(action.source, self.tests, directory=self.root)
                # Only syntax-valid, scope-valid source replaces the work file.
                try:
                    check_source(action.source)
                    (self.root / "solution.py").write_text(action.source, encoding="utf-8")
                except (ValueError, SyntaxError):
                    pass
                if result["passed"] and self.stop and conversation is not None:
                    conversation.pause()
                return EditObservation.from_text(json.dumps(result), is_error=not result["passed"])
        class EditAndTestTool(ToolDefinition[EditAction, EditObservation]):
            @classmethod
            def create(cls, conv_state=None, **params):
                return [cls(action_type=EditAction, observation_type=EditObservation,
                            description="Replace solution.py and immediately run the fixed public tests. No other files can be edited.",
                            executor=EditExecutor(params["workspace"], params["public_tests"], params.get("stop_on_public_pass", False)))]
        _TOOL_CLASS = EditAndTestTool
        register_tool("RelayEditAndTest", _TOOL_CLASS)
    return Tool(name="RelayEditAndTest", params={"workspace": str(workspace), "public_tests": public_tests,
                                               "stop_on_public_pass": stop_on_public_pass})


SYSTEM_PROMPT = (
    "You are a Python coding agent working on one small function task. "
    "Use edit_and_test to write the complete solution.py source and run public tests. "
    "Fix failures if necessary; then call finish. Preserve specified function names. "
    "Only the listed standard-library imports are permitted. Do not access files, network or process APIs. "
    "The host separately evaluates held-out tests, so implement the specification, not just examples."
)


def task_prompt(task):
    return (task["instruction"] + "\nAllowed imports: " + ", ".join(sorted(ALLOWED_IMPORTS))
            + "\nCurrent solution.py:\n```python\n" + task["initial_source"]
            + "\n```\nPublic checks:\n```python\n" + task["public_tests"] + "\n```")


def run_episode(task, llm, session, directory, *, max_calls=3, raise_on_error=False, stop_on_public_pass=False):
    from openhands.sdk import Agent
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    import openhands.sdk.llm.llm_profile_store as profile_store
    from .integrations.openhands import make_openhands_router
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    workspace = directory / "work"
    workspace.mkdir()
    source_path = workspace / "solution.py"
    source_path.write_text(task["initial_source"], encoding="utf-8")
    tool = make_edit_tool(workspace, task["public_tests"], stop_on_public_pass=stop_on_public_pass)
    router = make_openhands_router(session, {llm.model: llm})
    instruction = SYSTEM_PROMPT + (" The host stops automatically when public checks pass; no closing response is needed." if stop_on_public_pass else "")
    agent = Agent(llm=router, tools=[tool], condenser=None, include_default_tools=["FinishTool"],
                  mcp_config={}, system_prompt=instruction)
    started = time.perf_counter()
    conversation = LocalConversation(agent=agent, workspace=workspace, persistence_dir=directory / "state",
                                     profile_store_dir=directory / "profiles", max_iteration_per_run=max_calls,
                                     visualizer=None, plugins=[])
    error_type, state = None, None
    try:
        # SDK 1.62's automatic vision-profile probe ignores the Conversation's
        # profile_store_dir. Point that probe at the same empty local store;
        # never read unrelated user credentials or auto-attach another model.
        with patch.object(profile_store, "_DEFAULT_PROFILE_DIR", directory / "profiles"):
            conversation.send_message(task_prompt(task))
            conversation.run()
        state = str(conversation.state.execution_status)
    except Exception as exc:
        error_type = type(exc).__name__  # Never log provider exception text/credentials.
        if raise_on_error:
            raise
    finally:
        conversation.close()
    final_source = source_path.read_text(encoding="utf-8")
    public = run_check(final_source, task["public_tests"], directory=directory / "public_verification")
    private = run_check(final_source, task["private_tests"], directory=directory / "private_verification")
    observation = session.finish(accepted=bool(private["passed"] and public["passed"]), cost_complete=False)
    observation.update(episode_status=state, error_type=error_type, stop_on_public_pass=stop_on_public_pass,
                       public_passed=public["passed"], private_passed=private["passed"],
                       private_check_category=private["category"],
                       wall_seconds=time.perf_counter() - started,
                       verification_seconds=public["seconds"] + private["seconds"])
    (directory / "episode.json").write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    return observation
