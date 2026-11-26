"""Executable offline wiring example: real OpenHands, simulated model replies.

Run with Python >=3.12 and openhands-sdk==1.62.0 installed. No API key, provider
request or acceptance benchmark is used. The fixture makes the cheap attempt
fail a protected project check and the strong attempt fix it from a clean copy.

For real use, keep make_openhands_attempt/run_cascade and supply your existing
Agent factory, task prompt, model tariffs and conservative call reservations.
Remove only the offline transport fixture and socket guard in your own harness.
This example deliberately provides no implicit live mode or provider defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "relay_router" / "cascade.py").is_file():
    sys.path.insert(0, str(ROOT / "src"))
    NAMESPACE = "relay_router"
else:
    NAMESPACE = "relay_harness"


def run_offline_demo(destination, *, provider_rejection=False):
    """Create artifacts for a simulated transport test, never savings evidence."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline example forbids network")):
        from litellm import ModelResponse
        from openhands.sdk import Agent, LLM
        from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor, Tool, register_tool
        cascade = import_module(NAMESPACE + ".cascade")
        verifier_module = import_module(NAMESPACE + ".cascade_verifier")
        integration = import_module(NAMESPACE + ".integrations.openhands_cascade")

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        snapshot = destination / "snapshot"
        snapshot.mkdir()
        initial = "def add(a, b):\n    return 0\n"
        (snapshot / "solution.py").write_text(initial, encoding="utf-8")
        (snapshot / "check_solution.py").write_text(
            "from solution import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
            encoding="utf-8",
        )

        class WriteAction(Action):
            source: str

        class WriteObservation(Observation):
            pass

        class WriteExecutor(ToolExecutor):
            def __init__(self, workspace):
                self.workspace = Path(workspace)

            def __call__(self, action, conversation=None):
                (self.workspace / "solution.py").write_text(action.source, encoding="utf-8")
                if conversation is not None:
                    conversation.pause()
                return WriteObservation.from_text("Saved fixture solution.py")

        class CascadeDemoWriteTool(ToolDefinition[WriteAction, WriteObservation]):
            @classmethod
            def create(cls, conv_state=None, **params):
                return [cls(action_type=WriteAction, observation_type=WriteObservation,
                            description="Write the fixture solution.py only.",
                            executor=WriteExecutor(params["workspace"]))]

        register_tool("RelayCascadeOfflineWrite", CascadeDemoWriteTool)
        tool_name = CascadeDemoWriteTool.create(workspace=snapshot)[0].name
        candidate = "groq/openai/gpt-oss-20b"
        baseline = "groq/openai/gpt-oss-120b"
        initial_sources = []

        def factory(context):
            initial_sources.append((context.workspace / "solution.py").read_text(encoding="utf-8"))
            cheap = context.model == candidate
            llm = LLM(model=context.model, api_key="offline-fixture-key", num_retries=0,
                      api_mode="chat", max_output_tokens=128, timeout=30, caching_prompt=False,
                      input_cost_per_token=(.075 if cheap else .15) / 1_000_000,
                      output_cost_per_token=(.30 if cheap else .60) / 1_000_000)
            return Agent(llm=llm, condenser=None, critic=None, mcp_config={},
                         tools=[Tool(name="RelayCascadeOfflineWrite", params={"workspace": str(context.workspace)})],
                         include_default_tools=["FinishTool"],
                         system_prompt="Use the supplied tool to implement add(a, b).")

        def transport(**kwargs):
            cheap = str(kwargs["model"]).endswith("gpt-oss-20b")
            source = "def add(a, b):\n    return a - b\n" if cheap else "def add(a, b):\n    return a + b\n"
            return ModelResponse(
                id="simulated-cascade-response", model=kwargs["model"],
                choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "simulated-write", "type": "function", "function": {
                        "name": tool_name, "arguments": json.dumps({"source": source}),
                    }}],
                }}],
                usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            )

        verifier = verifier_module.CommandVerifier.capture(
            snapshot,
            checks=[verifier_module.CheckCommand("project-tests", (sys.executable, "-B", "check_solution.py"))],
            protected_files=["check_solution.py"], verifier_id="offline-demo-checks-v1",
        )
        attempt = integration.make_openhands_attempt(
            factory, "Implement add(a, b), preserving the supplied tests.",
            reservation_usd_by_model={candidate: .001, baseline: .002},
            tools_are_local=True, max_iterations=2, recover_tool_use_failed=provider_rejection,
        )
        if provider_rejection:
            import httpx

            def offline_http(client_transport, request):
                payload = json.loads(request.content)
                model = payload["model"]
                if model == candidate.removeprefix("groq/"):
                    return httpx.Response(400, json={"error": {"code": "tool_use_failed",
                        "type": "invalid_request_error", "message": "Invalid simulated tool call"}},
                        headers={"x-request-id": "req_simulated_rejection"})
                if model != baseline.removeprefix("groq/"):
                    raise AssertionError("Unexpected model in the offline fixture")
                reply = transport(model=model).model_dump(exclude_none=True)
                reply["service_tier"] = "on_demand"
                return httpx.Response(200, json=reply)

            mock_transport = patch.object(httpx.HTTPTransport, "handle_request", offline_http)
        else:
            mock_transport = patch("openhands.sdk.llm.llm.litellm_completion", side_effect=transport)
        with mock_transport:
            report = cascade.run_cascade(
                snapshot_dir=snapshot, output_dir=destination / "run", task_id="offline-fixture",
                candidate_model=candidate, baseline_model=baseline, run_attempt=attempt,
                verifier=verifier, budget=cascade.SpendBudget(max_cost_usd=.05, max_calls=4, max_seconds=120),
                mode="experimental", allow_experimental=True,
            )
        if initial_sources != [initial, initial]:
            raise AssertionError("The fallback must start from the original snapshot")
        if provider_rejection:
            ledger = report["accounting"]
            if not (report["operational_passed"] and report["selected_attempt"] == 2
                    and ledger["http_calls"] == 2 and ledger["cost_usd"] is None
                    and ledger["known_cost_subtotal_usd"] > 0
                    and ledger["cost_upper_bound_usd"] > ledger["known_cost_subtotal_usd"]
                    and ledger["calls"][0].get("provider_error_code") == "tool_use_failed"):
                raise AssertionError("The offline HTTP rejection must recover with its expense still bounded")
        (destination / "fixture_metadata.json").write_text(json.dumps({
            "evidence_kind": "offline_integration_fixture_not_model_quality_or_savings",
            "model_responses_simulated": True, "provider_requests": 0,
            "fresh_workspace_confirmed": True,
            "simulated_provider_rejection": provider_rejection,
        }, indent=2) + "\n", encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verifica la cascada OpenHands local con respuestas simuladas, sin red.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cascade-demo" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--provider-rejection", action="store_true",
                        help="Simula un HTTP 400 de Groq sin tokens en el transporte HTTP.")
    args = parser.parse_args(argv)
    run_offline_demo(args.output, provider_rejection=args.provider_rejection)
    print("Integración comprobada: falla simulada en 20B, escalamiento a 120B desde una copia limpia.")
    print("Respuestas simuladas; este ejemplo no demuestra calidad ni ahorro con modelos reales.")
    print(f"Artefactos: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
