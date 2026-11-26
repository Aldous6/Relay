import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.integrations import opencode


class OpenCodeTransportTests(unittest.TestCase):
    def setUp(self):
        # The fixture scripts implement OpenCode v1, not Python's own --version.
        version = patch.object(opencode, "_is_v2", return_value=False)
        version.start()
        self.addCleanup(version.stop)

    def test_metadata_whitelist_and_optional_values(self):
        sample = {"name": "Uno", "limit": {"context": 1000, "output": 200},
                  "capabilities": {"toolcall": True, "reasoning": False, "input": {"text": True}},
                  "cost": {"input": 0, "output": 1.5}, "headers": {"Authorization": "SECRET"},
                  "options": {"apiKey": "SECRET"}, "status": "active"}
        catalog = "provider/a\r\n" + json.dumps(sample, indent=2) + "\r\nlocal/org/custom:latest\r\n"
        m1, m2 = opencode.parse_catalog(catalog)
        self.assertEqual((m1.input_usd_per_million, m1.output_usd_per_million), (0, 1.5))
        self.assertEqual((m1.tool_call, m1.reasoning, m1.input_text, m1.output_text), (True, False, True, None))
        self.assertEqual((m1.context_tokens, m1.output_tokens), (1000, 200))
        self.assertNotIn("SECRET", repr(m1))
        self.assertIsNone(m2.input_usd_per_million)
        self.assertEqual(m2.model_id, "local/org/custom:latest")
        with self.assertRaises(opencode.OpenCodeError):
            opencode.parse_catalog("provider/m\n{")
        with self.assertRaises(opencode.OpenCodeError):
            opencode.parse_catalog("provider/m\nprovider/m\n")

    @staticmethod
    def event(kind, part_id, **part):
        return {"type": kind, "sessionID": "ses_test", "part": {"id": part_id, "sessionID": "ses_test", **part}}

    def completed(self):
        return [self.event("step_start", "p1"),
                self.event("tool_use", "p2", tool="read", state={"status": "completed", "output": "SECRET"}),
                self.event("text", "p3", text="Respuesta de prueba."),
                self.event("step_finish", "p4", reason="stop", cost=.03,
                           tokens={"input": 80, "output": 11, "reasoning": 3, "cache": {"read": 0, "write": 20}})]

    def read(self, items):
        raw = "\n".join(json.dumps(item) for item in items).encode()
        return opencode._read_events(raw.splitlines())

    def test_native_launch_keeps_prompt_verbatim_off_the_command_line(self):
        prompt = 'Resuelve:\r\n    $HOME & ping "x" | --command\nこんにちは' * 4000
        events = self.completed()
        stdout = b"\n".join(json.dumps(e).encode() for e in events)
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value=sys.executable):
            # Python is our native mock CLI. Its "run" script is loaded from
            # cwd. Both streams exceed the pipe buffer before the mock reads stdin.
            expected_args = ["--model", "provider/m", "--dir", str(Path(tmp).resolve()), "--format", "json"]
            fixture = Path(tmp) / "run"
            fixture.write_text(
                "import re, sys\n"
                f"assert sys.argv[1:-2] == {expected_args!r}\n"
                "assert sys.argv[-2] == '--title' and re.fullmatch(r'Relay-[a-f0-9]{32}', sys.argv[-1])\n"
                "sys.stderr.buffer.write(b'x'*65534 + b'auto-rejecting' + b'x'*90000)\n"
                "sys.stdout.buffer.write(b' '*100000 + b'\\n'); sys.stdout.buffer.flush()\n"
                "data = sys.stdin.buffer.read()\n"
                f"assert data == {prompt.encode('utf-8')!r}\n"
                f"sys.stdout.buffer.write({stdout!r})\n", encoding="utf-8")
            out = opencode.run_prompt(prompt, "provider/m", Path(tmp))
            fixture.write_text(f"import sys\nsys.stdout.buffer.write({stdout!r})\nsys.exit(3)", encoding="utf-8")
            failed = opencode.run_prompt("hola", "provider/m", Path(tmp))
        self.assertEqual((failed["status"], failed["returncode"]), ("error", 3))
        self.assertFalse(failed["usage"]["all_steps_reported"])
        self.assertTrue(out["permission_denied"])
        self.assertEqual(out["result_source"], "cli_events")
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["session_id"], "ses_test")
        self.assertEqual(out["text"], "Respuesta de prueba.")
        self.assertTrue(out["usage"]["all_steps_reported"])
        self.assertEqual(out["usage"]["reported_cost_usd"], .03)
        self.assertEqual(out["usage"]["cache_write_tokens"], 20)
        self.assertIsNone(out["usage"]["cost_usd"])
        self.assertNotIn("SECRET", str(out))

    def session_export(self, tmp, title):
        sid = "ses_test"
        user = {"id": "msg_u", "role": "user", "sessionID": sid, "model": {"providerID": "provider", "modelID": "m"}}
        user_part = {"sessionID": sid, "messageID": "msg_u", "type": "text", "text": "hola"}
        reply = {"id": "msg_a", "role": "assistant", "sessionID": sid, "parentID": "msg_u",
                 "providerID": "provider", "modelID": "m", "time": {"completed": 123}, "finish": "stop"}
        parts = [{**e["part"], "messageID": "msg_a", "type": k} for e, k in
                 zip(self.completed(), ["step-start", "tool", "text", "step-finish"])]
        return {"info": {"id": sid, "directory": str(tmp), "title": title},
                "messages": [{"info": user, "parts": [user_part]}, {"info": reply, "parts": parts}]}

    def test_empty_stream_recovers_named_session_with_read_only_commands(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value=sys.executable), \
                patch.object(opencode, "uuid4") as random_id:
            title = "Relay-123"
            random_id.return_value.hex = "123"
            directory = Path(tmp).resolve()
            record = self.session_export(directory, title)
            # The mock process produces NO output for run. A second run would
            # fail on the exclusive creation of this file.
            (directory / "run").write_text("open('dispatched', 'x').close()", encoding="utf-8")
            (directory / "session").write_text("import sys\nassert sys.argv[1] == 'list'\n"
                                              f"print({json.dumps([record['info']])!r})", encoding="utf-8")
            (directory / "export").write_text("import sys\nassert sys.argv[1:] == ['ses_test']\n"
                                             f"print({json.dumps(record)!r})", encoding="utf-8")
            out = opencode.run_prompt("hola", "provider/m", directory)
        self.assertEqual((out["status"], out["result_source"]), ("completed", "session_export"))
        self.assertEqual(out["text"], "Respuesta de prueba.")
        self.assertEqual(out["tool_calls"], [{"name": "read", "status": "completed"}])
        self.assertNotIn("SECRET", str(out))
        self.assertIsNone(out["usage"]["cost_usd"])
        self.assertEqual(out["usage"]["reported_cost_usd"], .03)

    def test_recovery_does_not_read_unrelated_sessions(self):
        directory = Path.cwd()
        safe = self.session_export(directory, "owned")["info"]
        for rows in ([], [{**safe, "title": "unrelated"}], [{**safe, "directory": str(directory.parent)}],
                     [safe, safe], [{**safe, "id": "--untrusted"}]):
            with self.subTest(rows=rows), patch.object(opencode, "_local_json", return_value=rows) as cli:
                self.assertIsNone(opencode._recover_session("cli", directory, "owned", None, "provider/m", "hola"))
                cli.assert_called_once()
        with patch.object(opencode, "_local_json", return_value=[safe]) as cli:
            self.assertIsNone(opencode._recover_session("cli", directory, "owned", "ses_other", "provider/m", "hola"))
            cli.assert_called_once()

    @staticmethod
    def follow_up_turn(suffix, prompt, text, model="mejor"):
        user, reply, sid = f"msg_u_{suffix}", f"msg_a_{suffix}", "ses_test"
        return [
            {"info": {"id": user, "role": "user", "sessionID": sid,
                      "model": {"providerID": "provider", "modelID": model}},
             "parts": [{"sessionID": sid, "messageID": user, "type": "text", "text": prompt}]},
            {"info": {"id": reply, "role": "assistant", "sessionID": sid, "parentID": user,
                      "providerID": "provider", "modelID": model, "time": {"completed": 200}, "finish": "stop"},
             "parts": [{"id": f"part_{suffix}_{n}", "sessionID": sid, "messageID": reply, **data}
                       for n, data in enumerate([{"type": "step-start"}, {"type": "text", "text": text},
                                                 {"type": "step-finish", "reason": "stop", "cost": .002,
                                                  "tokens": {"input": 91, "output": 3}}])]},
        ]

    def test_resume_exact_session_selects_the_new_message_and_only_its_usage(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value=sys.executable):
            directory = Path(tmp).resolve()
            before = self.session_export(directory, "owned")
            before["messages"].extend(self.follow_up_turn("previous", "hola", "Respuesta anterior"))
            after = deepcopy(before)
            after["messages"].extend(self.follow_up_turn("another", "petición en otra ventana", "Otro turno"))
            after["messages"].extend(self.follow_up_turn("current", "hola", "Respuesta actual"))
            (directory / "run").write_text(
                "import sys\nassert sys.argv[-2:] == ['--session', 'ses_test']\n"
                "assert sys.argv[1:3] == ['--model', 'provider/mejor'] and '--title' not in sys.argv\n"
                "assert sys.stdin.buffer.read() == b'hola'\nopen('sent-once', 'x').close()", encoding="utf-8")
            with patch.object(opencode, "_local_json", side_effect=[before, after]) as reads:
                out = opencode.run_prompt("hola", "provider/mejor", directory, session_id="ses_test")
            self.assertEqual([call.args[2] for call in reads.call_args_list], [["export", "ses_test"]] * 2)
        self.assertEqual((out["status"], out["text"], out["session_id"]), ("completed", "Respuesta actual", "ses_test"))
        self.assertEqual(out["usage"]["reported_cost_usd"], .002)
        self.assertEqual(out["usage"]["input_tokens"], 91)
        self.assertIsNone(out["usage"]["cost_usd"])

    def test_resume_ambiguous_or_missing_new_message_never_replays_the_old_turn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value=sys.executable):
            directory = Path(tmp).resolve()
            before = self.session_export(directory, "owned")
            duplicate = deepcopy(before)
            for suffix in ("one", "two"):
                duplicate["messages"].extend(self.follow_up_turn(suffix, "hola", "Respuesta ambigua", model="m"))
            # Even a plausible completed stdout stream cannot attribute these messages.
            (directory / "run").write_text("print(" + repr("\n".join(json.dumps(e) for e in self.completed())) + ")",
                                         encoding="utf-8")
            for after in (before, duplicate, None):
                with self.subTest(after=after), patch.object(opencode, "_local_json", side_effect=[before, after]):
                    out = opencode.run_prompt("hola", "provider/m", directory, session_id="ses_test")
                    self.assertEqual((out["status"], out["text"], out["session_id"]), ("error", "", "ses_test"))
                    self.assertIsNone(out["usage"]["reported_cost_usd"])

    def test_resume_validates_session_and_project_before_any_dispatch(self):
        directory = Path.cwd()
        other_project = self.session_export(directory.parent, "another")
        with patch.object(opencode, "_executable", return_value=sys.executable), \
                patch.object(opencode, "_local_json", return_value=other_project) as read, \
                patch.object(subprocess, "Popen") as dispatch:
            with self.assertRaises(opencode.OpenCodeError):
                opencode.run_prompt("hola", "provider/m", directory, session_id="--continue")
            read.assert_not_called()
            with self.assertRaisesRegex(opencode.OpenCodeError, "carpeta"):
                opencode.run_prompt("hola", "provider/m", directory, session_id="ses_test")
            dispatch.assert_not_called()

    def test_resume_reports_only_the_failed_follow_up(self):
        directory = Path.cwd()
        record = self.session_export(directory, "owned")
        old_ids = {m["info"]["id"] for m in record["messages"]}
        user, reply = self.follow_up_turn("second", "otra consulta", "")
        reply["parts"] = []
        reply["info"]["error"] = {"name": "APIError", "data": {"statusCode": 403}}
        record["messages"].extend([user, reply])
        with patch.object(opencode, "_local_json", return_value=record):
            out = opencode._continued_answer("cli", directory, "ses_test", old_ids, "provider/mejor", "otra consulta")
        self.assertIn("403", out["error"])
        self.assertEqual((out["http_status"], out["http_error_scope"]), (403, "selected_model"))
        self.assertEqual(out["text"], "")
        self.assertIsNone(out["usage"]["reported_cost_usd"])
        self.assertIsNone(out["usage"]["input_tokens"])

    def test_recovered_messages_match_prompt_model_and_terminal_state(self):
        record = self.session_export(Path.cwd(), "owned")
        for which, key, value in [(0, "sessionID", "ses_other"), (0, "model", {}), (1, "parentID", "msg_other"),
                                  (1, "providerID", "other")]:
            with self.subTest(which=which, key=key):
                broken = deepcopy(record)
                broken["messages"][which]["info"][key] = value
                self.assertIsNone(opencode._export_answer(broken, "ses_test", "provider/m", "hola"))
        self.assertIsNone(opencode._export_answer(record, "ses_test", "provider/m", "other prompt"))
        for key, value in [("time", {}), ("finish", "length")]:
            partial = deepcopy(record)
            partial["messages"][1]["info"][key] = value
            answer = opencode._export_answer(partial, "ses_test", "provider/m", "hola")
            self.assertEqual(answer["status"], "error")
            self.assertEqual(answer["usage"]["reported_cost_usd"], .03)
            self.assertFalse(answer["usage"]["all_steps_reported"])
            self.assertIsNone(answer["usage"]["cost_usd"])
        failed = deepcopy(record)
        failed["messages"][1]["parts"] = []
        failed["messages"][1]["info"]["error"] = {"name": "APIError", "data": {"statusCode": 403, "message": "SECRET"}}
        out = opencode._export_answer(failed, "ses_test", "provider/m", "hola")
        self.assertEqual(out["status"], "error")
        self.assertIn("403", out["error"])
        self.assertEqual((out["http_status"], out["http_error_scope"]), (403, "selected_model"))
        self.assertIsNone(out["usage"]["reported_cost_usd"])
        self.assertNotIn("SECRET", str(out))

    def test_repeated_events_deduplicate_and_preserve_partial_usage(self):
        events = self.completed()
        repeated = [*events, events[-1], self.event("step_start", "p5"),
                    self.event("step_finish", "p6", reason="stop", tokens={"input": 50})]
        out = self.read(repeated)
        self.assertEqual(out["usage"]["steps_finished"], 2)
        self.assertEqual(out["usage"]["input_tokens"], 130)
        self.assertEqual(out["usage"]["output_tokens"], 11)
        self.assertEqual(out["usage"]["reported_cost_usd"], .03)
        self.assertIsNone(out["usage"]["cost_usd"])
        self.assertFalse(out["usage"]["all_steps_reported"])
        self.assertIsNone(self.read([])["usage"]["reported_cost_usd"])

    def test_reported_zero_is_not_a_task_charge(self):
        events = self.completed()
        events[-1]["part"]["cost"] = 0
        events[-1]["part"]["tokens"] = {"input": 0, "output": 0}
        out = self.read(events)
        self.assertEqual(out["usage"]["reported_cost_usd"], 0)
        self.assertIsNone(out["usage"]["cost_usd"])

    def test_nonzero_exit_session_error_and_partial_stream_not_mistaken_for_success(self):
        events = self.completed()
        error = {"type": "error", "sessionID": "ses_test",
                 "error": {"name": "APIError", "data": {"statusCode": 403, "message": "SECRET"}}}
        failed = self.read([*events, error])
        self.assertEqual(failed["status"], "error")
        self.assertIn("403", failed["error"])
        self.assertEqual((failed["http_status"], failed["http_error_scope"]), (403, "session"))
        self.assertNotIn("SECRET", str(failed))
        self.assertFalse(failed["usage"]["all_steps_reported"])
        self.assertEqual(self.read(events[:-1])["status"], "error")
        self.assertEqual(self.read([{**events[0], "sessionID": "other"}, *events[1:]])["status"], "error")
        self.assertEqual(opencode._read_events([b"not a json event"])["status"], "error")

    def test_discovery_uses_workspace_and_no_prompt_or_refresh(self):
        result = subprocess.CompletedProcess([], 0, stdout=b"p/m\n{}\n", stderr=b"SECRET")
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value="opencode.exe"), \
                patch.object(subprocess, "run", return_value=result) as process:
            self.assertEqual(opencode.discover_models(Path(tmp))[0].model_id, "p/m")
            self.assertEqual(process.call_args.args[0], ["opencode.exe", "models", "--verbose"])
            self.assertEqual(process.call_args.kwargs["stdin"], subprocess.DEVNULL)
            result.returncode = 1
            with self.assertRaises(opencode.OpenCodeError) as exc:
                opencode.discover_models(Path(tmp))
            self.assertNotIn("SECRET", str(exc.exception))

    def test_http_status_requires_a_typed_error_and_never_reads_tool_output(self):
        event = {"type": "error", "sessionID": "ses_test"}
        for error in ({"name": "APIError", "data": {"message": "HTTP 401"}},
                      {"name": "APIError", "data": {"statusCode": "401"}},
                      {"name": "UnknownError", "data": {"statusCode": 401}}):
            with self.subTest(error=error):
                result = self.read([{**event, "error": error}])
                self.assertIsNone(result["http_status"])
                self.assertIsNone(result["http_error_scope"])
        event = self.event("tool_use", "tool", tool="webfetch",
                           state={"status": "error", "error": "APIError (HTTP 401)", "statusCode": 401})
        result = self.read([event, *self.completed()])
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["http_status"])
        self.assertIsNone(result["http_error_scope"])

    def test_live_401_is_only_attributed_to_model_if_export_matches(self):
        error = {"name": "APIError", "data": {"statusCode": 401, "message": "SECRET"}}
        event = {"type": "error", "sessionID": "ses_test", "error": error}
        with tempfile.TemporaryDirectory() as tmp, patch.object(opencode, "_executable", return_value=sys.executable):
            directory = Path(tmp).resolve()
            (directory / "run").write_text(f"print({json.dumps(event)!r})", encoding="utf-8")
            record = self.session_export(directory, "owned")
            assistant = record["messages"][1]
            assistant["parts"] = []
            assistant["info"]["error"] = error
            matching = opencode._export_answer(record, "ses_test", "provider/m", "hola")
            different = self.read([{**event, "error": {"name": "APIError", "data": {"statusCode": 403}}}])
            for recovered, scope in ((matching, "selected_model"), (different, "session"),
                                     (self.read([]), "session"), (None, "session")):
                with self.subTest(scope=scope, recovered=recovered), \
                        patch.object(opencode, "_recover_session", return_value=recovered) as lookup:
                    out = opencode.run_prompt("hola", "provider/m", directory)
                    lookup.assert_called_once()
                    self.assertEqual((out["status"], out["http_status"], out["http_error_scope"]),
                                     ("error", 401, scope))
                    self.assertIsNone(out["usage"]["cost_usd"])
                    self.assertIsNone(out["usage"]["reported_cost_usd"])
                    self.assertNotIn("SECRET", str(out))
                    self.assertIsNone(out["auth_failure_reason"])
                    self.assertEqual(out["error"], "APIError (HTTP 401). revisa la autenticación del proveedor.")
            assistant["info"]["error"] = {"name": "APIError", "data": {"statusCode": 401,
                                          "responseBody": json.dumps({"error": {"message": "key has expired"}})}}
            expired = opencode._export_answer(record, "ses_test", "provider/m", "hola")
            with patch.object(opencode, "_recover_session", return_value=expired):
                diagnosed = opencode.run_prompt("hola", "provider/m", directory)
            self.assertEqual(diagnosed["auth_failure_reason"], "credential_expired")
            self.assertIn("credencial venció", diagnosed["error"])
            assistant["parts"] = self.session_export(directory, "owned")["messages"][1]["parts"]
            with_usage = opencode._export_answer(record, "ses_test", "provider/m", "hola")
            with patch.object(opencode, "_recover_session", return_value=with_usage):
                out = opencode.run_prompt("hola", "provider/m", directory)
            self.assertEqual(out["usage"]["reported_cost_usd"], .03)
            self.assertIsNone(out["usage"]["cost_usd"])
            stream = "\n".join(json.dumps(row) for row in [*self.completed(), event])
            (directory / "run").write_text(f"print({stream!r})", encoding="utf-8")
            with patch.object(opencode, "_recover_session", return_value=matching):
                out = opencode.run_prompt("hola", "provider/m", directory)
            self.assertEqual(out["usage"]["reported_cost_usd"], .03)
            self.assertIsNone(out["usage"]["cost_usd"])

    def test_expiry_requires_a_direct_provider_statement_and_401(self):
        cases = [
            (401, {"message": "Your API key has expired. More details: SECRET"}, True),
            (401, {"responseBody": '{"error":{"message":"key has expired"},"key":"SECRET"}'}, True),
            (401, {"responseBody": {"message": "Access token has expired"}}, True),
            (401, {"responseBody": "key has expired"}, False),
            (401, {"responseBody": '{"unrelated":{"message":"key has expired"}}'}, False),
            (401, {"message": "Check whether the key has expired.", "apiKey": "SECRET"}, False),
            (401, {"message": "The key has not expired"}, False),
            (401, {"message": "Authentication failed"}, False),
            (403, {"message": "key has expired"}, False),
        ]
        for status, details, expired in cases:
            error = {"name": "APIError", "data": {"statusCode": status, **details}}
            with self.subTest(details=details):
                result = self.read([{"type": "error", "sessionID": "ses_test", "error": error}])
                self.assertEqual(result["auth_failure_reason"], "credential_expired" if expired else None)
                self.assertNotIn("SECRET", str(result))


if __name__ == "__main__":
    unittest.main()
