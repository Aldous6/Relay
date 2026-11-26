"""Software checks only: synthetic transports do not establish product savings."""
import asyncio
import hashlib
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
sys.path.insert(0, str(ROOT / "src"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
from relay_router.harness_comparison import compare, freeze_plan
from relay_router.harness_episode import check_source, run_check, run_episode
from relay_router import Relay


class ComparisonTests(unittest.TestCase):
    def test_exact_model_tariffs_and_all_http_attempt_accounting(self):
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location("relay_live_rates_test", ROOT / "tools/relay_harness_live.py")
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
        models = ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"]
        with tempfile.TemporaryDirectory() as tmp:
            audit = live.AuditedTransport(lambda **kw: SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100,completion_tokens=50)),
                dict(models=models,max_output_tokens=1024,max_total_http_attempts=3,max_observed_tokens_per_model=10000),Path(tmp))
            for model in models:
                audit(model=model.removeprefix("groq/"), custom_llm_provider="groq", messages=[], api_key="not-recorded")
            self.assertAlmostEqual(audit.events[0]["cost_usd"], .0000225)
            self.assertAlmostEqual(audit.events[1]["cost_usd"], .000045)
            self.assertNotIn("not-recorded", (Path(tmp)/"http_attempts.jsonl").read_text())

    def data(self):
        tasks = [dict(task_id=split+str(i), group_id=split+str(i), workflow="repair", split=split)
                 for split in ("development", "validation") for i in range(2)]
        suite = dict(models=["small", "large"], baseline_model="large", evaluation_id="synthetic", tasks=tasks)
        rows = [dict(**task, model=model, accepted=True, cost_usd=cost, cost_complete=True)
                for task in tasks for model, cost in (("small", .5), ("large", 1.))]
        return suite, rows

    def test_validation_failure_does_not_reselect_frozen_route(self):
        suite, rows = self.data()
        plan = freeze_plan(suite, [row for row in rows if row["split"] == "development"])
        for row in rows:
            if row["split"] == "validation" and row["model"] == "small":
                row["accepted"] = False
        result = compare(suite, plan, rows)["policies"]
        self.assertEqual(result["relay_exploratory"]["accepted"], 0)
        self.assertEqual(result["fixed_cheap"]["accepted"], 0)
        self.assertIsNone(result["relay_exploratory"]["llm_tariff_cost_per_accepted_task_usd"])
        self.assertEqual(result["relay_exploratory"]["quality_delta_vs_strong"], -1)

    def test_failures_are_costed_and_missing_rows_block_total(self):
        suite, rows = self.data()
        plan = freeze_plan(suite, [row for row in rows if row["split"] == "development"])
        rows[-2]["accepted"] = False
        result = compare(suite, plan, rows)["policies"]["fixed_cheap"]
        self.assertEqual(result["llm_tariff_cost_per_accepted_task_usd"], 1.)
        self.assertGreater(result["wilson_95_one_sided_harm_upper"], 0)
        result = compare(suite, plan, rows[:-2])["policies"]["fixed_cheap"]
        self.assertEqual(result["missing"], 1)
        self.assertIsNone(result["llm_tariff_total_usd"])
        with self.assertRaises(ValueError):
            freeze_plan(suite, rows)
        with self.assertRaises(ValueError):
            compare(suite, plan, rows + rows[:1])

    def test_source_guard_and_host_tests(self):
        for source in ("import os", "open('secret')", "(1).__class__", "from math import __dict__"):
            with self.assertRaises(ValueError):
                check_source(source)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(run_check("def add(a,b): return a+b", "assert add(2,3)==5", directory=tmp)["passed"])
            self.assertFalse(run_check("def add(a,b): return a-b", "assert add(2,3)==5", directory=tmp)["passed"])


@unittest.skipUnless(importlib.util.find_spec("openhands"), "Optional OpenHands environment required")
class FullEpisodeTests(unittest.TestCase):
    def test_complete_agent_edits_and_verifies_with_hidden_tests_outside_context(self):
        from openhands.sdk.llm import LLM
        from litellm import ModelResponse
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        model = "groq/openai/gpt-oss-20b"
        llm = LLM(model=model, api_key="synthetic-secret", num_retries=0, api_mode="chat", caching_prompt=False,
                  capability_overrides={"supports_prompt_cache_key": False, "supports_prompt_cache_retention": False})
        task = dict(task_id="synthetic", workflow="repair", initial_source="def add(a,b): return a-b\n",
                    instruction="Repair add", public_tests="assert add(2,3)==5\n",
                    private_tests="# PRIVATE_SENTINEL\nassert add(-1,2)==1\n")
        def response(name, args):
            return ModelResponse(model=model, choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "tool_calls": [{"id": name, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}}]}}],
                usage={"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35})
        with tempfile.TemporaryDirectory() as tmp:
            session = Relay(model).task("repair", max_calls=3)
            spec = importlib.util.spec_from_file_location("relay_live_test", ROOT / "tools/relay_harness_live.py")
            live = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(live)
            from unittest.mock import Mock
            actual_transport = Mock(side_effect=[response("edit_and_test", {"source": "def add(a,b): return a+b\n"}),
                                                 response("finish", {"message": "Done"})])
            audited = live.AuditedTransport(actual_transport, dict(models=[model], max_output_tokens=1024,
                max_total_http_attempts=3, max_observed_tokens_per_model=10000), Path(tmp))
            audited.episode = "synthetic"
            with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), patch(
                "openhands.sdk.llm.llm.litellm_completion", audited):
                result = run_episode(task, llm, session, Path(tmp) / "episode", raise_on_error=True, stop_on_public_pass=True)
            self.assertTrue(result["accepted"], json.dumps({key: result[key] for key in ("error_type", "episode_status", "logical_calls", "public_passed", "private_passed")}))
            self.assertEqual(result["logical_calls"], 1)
            self.assertEqual(actual_transport.call_count, 1)
            self.assertEqual(len(audited.events), 1)
            self.assertTrue(all(event["cost_usd"] is not None for event in audited.events))
            for call in actual_transport.call_args_list:
                self.assertNotIn("PRIVATE_SENTINEL", json.dumps(call.kwargs["messages"]))
                self.assertNotIn("prompt_cache_key", call.kwargs)
            for path in Path(tmp).rglob("*.json"):
                self.assertNotIn("synthetic-secret", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
