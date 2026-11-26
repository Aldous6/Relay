"""Attach Relay to an existing, local OpenHands Agent.

Python >=3.12, openhands-sdk==1.62.0. Run ``python examples/relay_openhands.py``
for an offline wiring check. It constructs the actual SDK Agent and RouterLLM;
it does not execute an agent, request credentials or contact a model.

In your existing harness:

    task, routed_agent = attach_relay(
        agent, relay, workflow="python-repair", models=configured_llms,
    )
    # Use routed_agent in your existing local Conversation and run it normally.
    # Obtain accepted from your existing tests/reviewer, not an LLM's text.
    task.finish(accepted=accepted, additional_cost_usd=auxiliary_cost,
                cost_complete=all_model_costs_accounted_for)

The host supplies configured LLMs (fresh instances per task, num_retries=0,
fallback_strategy=None), tools, credentials and acceptance checks. Relay keeps
one chosen model for the task and does not alter conversation messages/tools.
Model compatibility is the host's responsibility. Condensers and external
subagents must be included in auxiliary_cost unless they use this task's router
and are already recorded; do not add the entire conversation cost a second time.
Leave cost_complete=False when any costs/retries are unaccounted for.
"""
from __future__ import annotations

from pathlib import Path
from importlib import import_module
import os
import sys

# Source checkout convenience; installed packages can import relay_harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _relay_module(suffix=""):
    # A checkout must not silently execute an older installed wheel. A standalone
    # copy of this example uses the lightweight installed package instead.
    source = Path(__file__).resolve().parents[1] / "src" / "relay_router" / "workflow.py"
    namespace = "relay_router" if source.exists() else "relay_harness"
    return import_module(namespace + suffix)


def attach_relay(agent, relay, *, workflow, models, task_id=None, routing_text=None):
    """Return a task and a copy of the existing Agent with only its LLM replaced."""
    return _relay_module(".integrations.openhands").attach_openhands_agent(
        agent, relay, workflow=workflow, models=models, task_id=task_id, routing_text=routing_text)


def main():
    # Pin the local cost map for this offline check, avoiding import-time fetches.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    from unittest.mock import patch
    import socket

    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline check forbids network")):
        from openhands.sdk import Agent, LLM
        Relay = _relay_module().Relay

        llm = LLM(model="openai/gpt-4.1-mini", num_retries=0, api_mode="chat")
        agent = Agent(llm=llm, tools=[], condenser=None)
        relay = Relay(baseline_model=llm.model, mode="observe")
        task, routed_agent = attach_relay(agent, relay, workflow="python-repair", models={llm.model: llm})
        assert routed_agent.llm.relay_session is task
        assert routed_agent.tools == agent.tools
        task.finish(accepted=None, cost_complete=False)
    print("Integración local OpenHands + Relay construida. No se ejecutaron llamadas LLM.")
    print("Usa attach_relay en tu harness; accepted y el costo auxiliar los declara tu verificador.")


if __name__ == "__main__":
    main()
