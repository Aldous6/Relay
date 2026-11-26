"""Complete repository-campaign integration with a real SDK and an offline HTTP transport."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_repository_cascade as runner

HOST = """
import os
import sys
from pathlib import Path
from relay_router.cascade_verifier import CheckCommand, CommandVerifier
from repository_snapshot_host import build_agent

def prepare(task):
    folder = Path(os.environ["RELAY_REPOSITORY_CASES"]) / task["task_id"]
    snapshot = folder / "snapshot"
    verifier = CommandVerifier.capture(snapshot, checks=[CheckCommand("public", (sys.executable, "-I", "-B", "public.py"))],
                                       protected_files=["public.py"])
    return snapshot, verifier, {"private": folder / "private.txt", "config": folder / "case.json"}

def private_verdict(task, workspace):
    expected = Path(os.environ["RELAY_REPOSITORY_CASES"]) / task["task_id"] / "private.txt"
    return "passed" if (workspace / "value.txt").read_bytes() == expected.read_bytes() else "failed"
"""


def fixture(work):
    import httpx
    issue = {"task_id": "fixture", "group_id": "fixture-project", "source": "https://example.org/fixture",
             "revision": "fixture-version", "prompt": "Change the status from broken to fixed."}
    cohort = work / "cohort.json"
    runner.write(cohort, {"cohort_id": "synthetic-wiring-only", "tasks": [issue]})
    score_path = work / "scores.json"
    routes = {"fixture": {"arm": "fixed_strong", "router_seconds": .01, "amortized_load_seconds": .01}}
    runner.write(score_path, {"routes": routes, "threshold": .5, "truncated_route": "fixed_strong"})
    host = work / "host.py"
    host.write_text(HOST)
    folder = work / "cases" / "fixture"
    snapshot = folder / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "value.txt").write_text("broken")
    (snapshot / "public.py").write_text("import pathlib, sys\nsys.exit(pathlib.Path('value.txt').read_text() != 'fixed')")
    (folder / "private.txt").write_text("fixed")
    runner.write(folder / "case.json", {**issue, "image": "example.org/fixture@sha256:" + "1" * 64,
                                        "editable_paths": ["value.txt"]})
    physical = []
    actions = []

    def offline_http(transport, request):
        body = json.loads(request.content)
        model = body["model"]
        physical.append(model)
        if model == runner.CHEAP.removeprefix("groq/"):
            return httpx.Response(400, json={"error": {"code": "tool_use_failed", "type": "invalid_request_error",
                                                       "message": "fixture tool use error"}})
        if model != runner.STRONG.removeprefix("groq/"):
            raise AssertionError("Unexpected model in offline fixture")
        stage = sum(len(m.get("tool_calls", [])) for m in body["messages"])
        names = [t["function"]["name"] for t in body["tools"]]
        steps = [("list", {"operation": "list"}), ("read", {"operation": "read", "path": "value.txt"}),
                 ("replace", {"operation": "replace", "path": "value.txt", "find": "broken", "replacement": "fixed"}),
                 ("finish", {"message": "Repaired the fixture."})]
        action, arguments = steps[stage]
        tool = next(n for n in names if ("finish" in n) == (action == "finish"))
        actions.append(action)
        message = {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": f"fixture-{stage}",
                   "function": {"name": tool, "arguments": json.dumps(arguments)}}]}
        return httpx.Response(200, json={"id": "fixture-response", "object": "chat.completion", "created": 1,
            "model": model, "service_tier": "on_demand", "usage": {"prompt_tokens": 100, "completion_tokens": 24, "total_tokens": 124},
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]})

    sock_connect = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return sock_connect(sock, address)
        raise AssertionError("Network forbidden in repository fixture")

    output = work / "run"
    with patch.dict(os.environ, {"RELAY_REPOSITORY_CASES": str(folder.parent), "GROQ_API_KEY": "fixture-only",
                                 "LITELLM_LOCAL_MODEL_COST_MAP": "True", "OPENHANDS_SUPPRESS_BANNER": "1"}), \
            patch.object(runner, "frozen_routes", return_value=routes), \
            patch.object(socket.socket, "connect", local_only), \
            patch.object(httpx.HTTPTransport, "handle_request", offline_http):
        runner.freeze(cohort, score_path, host, output, .50, "fixture-only-rate")
        final = runner.acquire(cohort, score_path, host, output)
        assert final == runner.acquire(cohort, score_path, host, output)
    arm_rows = {arm: rows[0] for arm, rows in final["rows"].items()}
    assert arm_rows["relay"]["accepted"] and arm_rows["relay"]["escalated"], arm_rows["relay"]
    assert not arm_rows["fixed_cheap"]["accepted"] and arm_rows["fixed_strong"]["accepted"]
    assert arm_rows["relay"]["cost_usd"] is None and arm_rows["relay"]["cost_upper_bound_usd"] is not None
    assert actions == ["list", "read", "replace", "finish"] * 2 and len(physical) == 10, (actions, physical)
    assert not final["pilot_criteria_met"] and not final["all_operating_costs_point_identified"]
    return {"kind": "offline_sdk_and_repository_workflow_test", "provider_requests": 0,
            "physical_http_responses_simulated": len(physical), "cheap_rejections_without_usage": 2,
            "actions_on_each_strong_branch": actions[:4], "bert_route_fixture": True,
            "cohort": "single synthetic repository, separate from all benchmark cases",
            "private_checker": "isolated fixture file comparison", "result": "passed",
            "acceptances": {k: row["accepted"] for k, row in arm_rows.items()},
            "relay_point_cost_usd": arm_rows["relay"]["cost_usd"], "relay_upper_bound_present": True,
            "resumed_no_additional_dispatches": True, "economic_evidence": False,
            "code_sha256": {p: runner.file_hash(ROOT / p) for p in (*runner.SOURCES, "tools/repository_snapshot_host.py",
                                                                    "tests/test_repository_wire_fixture.py")}}


@unittest.skipUnless(importlib.util.find_spec("openhands") and importlib.util.find_spec("httpx"), "OpenHands optional environment")
class RepositoryWireFixtureTests(unittest.TestCase):
    def test_http_rejection_then_multistep_workspace_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(Path(directory))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--evidence":
        with tempfile.TemporaryDirectory() as directory:
            runner.write(Path(sys.argv[2]), fixture(Path(directory)))
    else:
        unittest.main()
