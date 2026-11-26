"""Contracts from the OpenCode v2 model.list and session export schemas."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.integrations import opencode


class OpenCodeV2Tests(unittest.TestCase):
    def test_catalog_requires_enabled_chat_and_zero_in_every_tier(self):
        directory = Path.cwd()
        zero = {"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}
        base = {"providerID": "opencode", "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                "enabled": True, "cost": [zero], "name": "Chat model"}
        rows = [
            {**base, "id": "big-pickle"},
            {**base, "id": "promo-free", "cost": [zero, {**zero, "tier": {"size": 10, "type": "context"},
                                                         "cache": {"read": 0, "write": 0.1}}]},
            {**base, "id": "unknown-free", "cost": []},
            {**base, "providerID": "custom", "id": "aliased"},
            {**base, "id": "disabled-free", "enabled": False},
            {**base, "id": "image", "type": "image"},
        ]
        def catalog(items):
            return json.dumps({"location": {"directory": str(directory)}, "data": items})
        models = opencode._parse_v2_catalog(catalog(rows), directory)
        self.assertEqual([m.model_id for m in models], ["opencode/big-pickle", "opencode/promo-free",
                                                        "opencode/unknown-free", "custom/aliased"])
        self.assertEqual([m.advertised_free for m in models], [True, False, False, None])
        self.assertIsNone(models[2].input_usd_per_million)
        for raw in (catalog(rows[:1] * 2), catalog(rows[4:])):
            with self.assertRaises(opencode.OpenCodeError):
                opencode._parse_v2_catalog(raw, directory)
        with self.assertRaises(opencode.OpenCodeError):
            opencode._parse_v2_catalog(catalog(rows), directory.parent)
        with patch.object(opencode, "_executable", return_value="opencode.exe"), \
                patch.object(opencode, "_is_v2", return_value=True), patch.object(subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout=catalog(rows).encode())
            self.assertEqual(opencode.discover_models(directory), models)
            self.assertEqual(run.call_args.args[0][:5], ["opencode.exe", "api", "model.list", "--standalone", "--header"])

    def test_v1_cache_or_context_surcharge_can_veto_free_label(self):
        for extra in ({"cache_read": .2}, {"cache": {"read": 0, "write": .2}},
                      {"tiers": [{"input": 0, "output": 1, "cache": {"read": 0, "write": 0}}]},
                      {"experimentalOver200K": {"input": 1, "output": 0}}):
            raw = {"cost": {"input": 0, "output": 0, **extra}}
            self.assertIs(opencode.parse_catalog("opencode/promo-free\n" + json.dumps(raw))[0].advertised_free, False)

    @staticmethod
    def _record(directory):
        return {"info": {"id": "ses_test", "title": "Relay-test", "location": {"directory": str(directory)}},
                "messages": [
                    {"id": "msg_config", "type": "agent-switched"},
                    {"id": "msg_system", "type": "synthetic"},
                    {"id": "msg_user", "type": "user", "text": "Primero"},
                    {"id": "msg_result", "type": "assistant", "model": {"providerID": "opencode", "id": "a"},
                     "content": [{"type": "text", "text": "Primera respuesta"}], "cost": 0,
                     "tokens": {"input": 20, "output": 3, "cache": {"read": 2, "write": 0}}, "finish": "stop",
                     "time": {"completed": 123}},
                    {"id": "msg_idle", "type": "idle", "outcome": "succeeded"}]}

    def test_native_first_and_followup_exactly_one_turn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value=sys.executable), \
                patch.object(opencode, "_is_v2", return_value=True), patch.object(opencode, "uuid4") as uid:
            uid.return_value.hex = "test"
            directory = Path(tmp).resolve()
            record = self._record(directory)
            expected = ("import os,sys\nassert os.getcwd() == os.environ['PWD']\n"
                        "assert '--standalone' in sys.argv and '--dir' not in sys.argv and '--auto' not in sys.argv\n")
            owned = {"id": "ses_test", "directory": str(directory), "title": "Relay-test"}
            with patch.object(opencode, "_local_json", side_effect=[[owned], record]):
                (directory / "run").write_text(expected + "assert sys.stdin.read()=='Primero'\n"
                    "print('{\"sessionID\":\"ses_test\",\"type\":\"step_start\",\"part\":{\"id\":\"s\"}}')", encoding="utf8")
                out = opencode.run_prompt("Primero", "opencode/a", directory)
            self.assertEqual((out["status"], out["text"], out["session_id"]), ("completed", "Primera respuesta", "ses_test"))
            self.assertEqual((out["usage"]["reported_cost_usd"], out["usage"]["cost_usd"]), (0, None))
            after = deepcopy(record)
            after["messages"].extend([
                {"id": "msg_sw", "type": "model-switched"},
                {"id": "msg_u2", "type": "user", "text": "Segundo"},
                {**record["messages"][3], "id": "msg_a2", "model": {"providerID": "opencode", "id": "b"},
                 "content": [{"type": "tool", "name": "read", "state": {"status": "completed", "content": "SECRET"}},
                             {"type": "text", "text": "Segunda respuesta"}], "tokens": {"input": 18, "output": 5}},
                {"id": "msg_i2", "type": "idle", "outcome": "succeeded"}])
            (directory / "run").write_text(expected + "assert '--session' in sys.argv and sys.stdin.read()=='Segundo'",
                                           encoding="utf8")
            for duplicated in (False, True):
                if duplicated:
                    after["messages"].append({"id": "msg_external", "type": "user", "text": "Otro"})
                with patch.object(opencode, "_local_json", side_effect=[record, after]):
                    out = opencode.run_prompt("Segundo", "opencode/b", directory, session_id="ses_test")
                self.assertNotIn("SECRET", str(out))
                if duplicated:
                    self.assertEqual((out["status"], out["text"]), ("error", ""))
                else:
                    self.assertEqual((out["status"], out["text"]), ("completed", "Segunda respuesta"))
                    self.assertEqual(out["usage"]["input_tokens"], 18)

    def test_typed_quota_error_and_unconfirmed_charge(self):
        directory = Path.cwd()
        record = self._record(directory)
        error = {"type": "provider.free-limit", "status": 429, "message": "SECRET"}
        assistant = record["messages"][3]
        for k in ("finish", "time", "cost", "tokens"):
            assistant.pop(k, None)
        assistant.update(content=[], error=error)
        record["messages"][-1].update(outcome="failed", error=error)
        with patch.object(opencode, "_local_json", return_value=record):
            out = opencode._v2_answer("opencode", directory, "ses_test", None, (), "opencode/a", "Primero")
        self.assertEqual((out["status"], out["http_status"], out["http_error_scope"]), ("error", 429, "selected_model"))
        self.assertIsNone(out["usage"]["reported_cost_usd"])
        self.assertNotIn("SECRET", str(out))

    def test_foreign_session_and_other_model_never_supply_a_reply(self):
        directory = Path.cwd()
        row = {"id": "ses_test", "directory": str(directory), "title": "Relay-test"}
        with patch.object(opencode, "_local_json", return_value=[row]) as read:
            self.assertIsNone(opencode._v2_answer("opencode", directory, "ses_foreign", "Relay-test", (),
                                                  "opencode/a", "Primero"))
            read.assert_called_once()
        record = self._record(directory)
        with patch.object(opencode, "_local_json", side_effect=[[row], record]):
            self.assertIsNone(opencode._v2_answer("opencode", directory, "ses_test", "Relay-test", (),
                                                  "opencode/b", "Primero"))
