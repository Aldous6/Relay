"""Optional integration check with a native OpenCode v1 CLI and a loopback fixture."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.integrations import opencode
from relay_router.opencode_selection import select_opencode_model


@unittest.skipUnless(os.environ.get("RELAY_OPENCODE_LOOPBACK_TEST") == "1", "opt-in native OpenCode fixture")
class OpenCodeLoopbackTest(unittest.TestCase):
    def test_run_prompt_with_an_isolated_project_and_provider(self):
        configured = os.environ.get("RELAY_OPENCODE_V1_TEST_BIN")
        if configured:
            exe = Path(configured)
            self.assertTrue(exe.is_absolute() and exe.is_file(), "Set RELAY_OPENCODE_V1_TEST_BIN to an absolute native CLI path")
        else:
            # The official Windows npm installs keep distinct binaries even
            # when the opencode command now points at a side-by-side v2.
            appdata = Path(os.environ.get("APPDATA", ""))
            npm_v1 = appdata / "npm/node_modules/opencode-ai/bin/opencode.exe"
            if os.name == "nt" and appdata.is_absolute() and npm_v1.is_file():
                exe = npm_v1
            else:
                try:
                    exe = Path(opencode._executable())
                except opencode.OpenCodeError:
                    self.skipTest("Native OpenCode v1 is not installed for this loopback fixture")
        exe = exe.resolve()
        default_model = "relay-fixture/default"
        expected_model = "relay-fixture/local"
        prompt = "Saludo de prueba\n  segunda línea con `$HOME` & |"
        follow_up = "Diagnostica el fallo que menciona el mensaje anterior."
        denied_prompt = "Prueba local de acceso denegado"
        auth_prompt = "Prueba local de credencial rechazada"
        refusals = {denied_prompt: (403, "permission_error"), auth_prompt: (401, "authentication_error")}
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                last_prompt = next(msg["content"] for msg in reversed(body["messages"]) if msg["role"] == "user")
                if last_prompt in refusals:
                    status, kind = refusals[last_prompt]
                    message = "key has expired" if status == 401 else "Access denied"
                    data = json.dumps({"error": {"message": message, "type": kind}}).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if body["model"] == "default":
                    response_text, tokens_in, tokens_out = "Respuesta al segundo turno", 42, 4
                else:
                    response_text, tokens_in, tokens_out = "Relay OK", 30, 2
                chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 1, "model": body["model"]}
                chunks = [
                    {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": response_text}}]},
                    {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                               "total_tokens": tokens_in + tokens_out}},
                ]
                data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data.encode())))
                self.end_headers()
                self.wfile.write(data.encode())

        with tempfile.TemporaryDirectory(prefix="relay-opencode-") as tmp:
            root = Path(tmp)
            work = root / "project"
            work.mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            with server:
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    fixture_models = {
                        name: {
                            "name": name,
                            "tool_call": True,
                            "reasoning": name == "default",
                            "cost": {"input": price, "output": 2 * price},
                            "limit": {"context": 65536, "output": 512},
                        }
                        for name, price in (("local", 1), ("default", 5))
                    }
                    config = {
                        "enabled_providers": ["relay-fixture"],
                        "model": default_model,
                        "small_model": default_model,
                        "share": "disabled",
                        "permission": "deny",
                        "autoupdate": False,
                        "provider": {
                            "relay-fixture": {
                                "npm": "@ai-sdk/openai-compatible",
                                "name": "Relay fixture",
                                "options": {
                                    "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                                    "apiKey": "fixture",
                                },
                                "models": fixture_models,
                            },
                        },
                    }
                    env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if k in os.environ}
                    system_path = str(Path(env["SYSTEMROOT"]) / "System32") if os.name == "nt" else os.defpath
                    env["PATH"] = os.pathsep.join([str(exe.parent), system_path])
                    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME",
                                "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "OPENCODE_TEST_MANAGED_CONFIG_DIR"):
                        path = root / key
                        path.mkdir()
                        env[key] = str(path)
                    env.update({k: str(root) for k in ("HOME", "USERPROFILE", "OPENCODE_TEST_HOME")})
                    env.update({key: "1" for key in (
                        "OPENCODE_PURE", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_DEFAULT_PLUGINS",
                        "OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE",
                        "OPENCODE_DISABLE_EXTERNAL_SKILLS", "OPENCODE_DISABLE_LSP_DOWNLOAD",
                        "OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_DISABLE_SHARE")})
                    env.update(HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                               NO_PROXY="localhost,127.0.0.1")
                    with patch.dict(os.environ, env, clear=True), patch.object(opencode, "_executable", return_value=str(exe)):
                        if opencode._is_v2(str(exe)):
                            if configured:
                                self.fail("RELAY_OPENCODE_V1_TEST_BIN points at OpenCode v2; this fixture requires v1")
                            self.skipTest("Only OpenCode v2 was found; set RELAY_OPENCODE_V1_TEST_BIN to a v1 CLI")
                        os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
                        inventory = opencode.discover_models(work)
                        self.assertCountEqual([m.model_id for m in inventory], [expected_model, default_model])
                        selection = select_opencode_model(prompt, inventory)
                        self.assertEqual(selection["model"], expected_model)
                        result = opencode.run_prompt(prompt, selection["model"], work)
                        self.assertEqual(len(requests), 1)
                        self.assertTrue(result["session_id"])
                        next_selection = select_opencode_model(follow_up, inventory)
                        self.assertEqual(next_selection["model"], default_model)
                        second = opencode.run_prompt(follow_up, next_selection["model"], work,
                                                     session_id=result["session_id"])
                        self.assertEqual(len(requests), 2)
                        denied_selection = select_opencode_model(denied_prompt, inventory)
                        denied = opencode.run_prompt(denied_prompt, denied_selection["model"], work)
                        self.assertEqual(len(requests), 3)
                        auth_selection = select_opencode_model(auth_prompt, inventory)
                        unauthorized = opencode.run_prompt(auth_prompt, auth_selection["model"], work)
                finally:
                    server.shutdown()
                    worker.join()
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["text"], "Relay OK")
            self.assertTrue(result["session_id"])
            self.assertEqual(result["usage"]["input_tokens"], 30)
            self.assertEqual(result["usage"]["output_tokens"], 2)
            self.assertIsNone(result["usage"]["cost_usd"])
            self.assertAlmostEqual(result["usage"]["reported_cost_usd"], .000034)
            self.assertEqual(second["status"], "completed", second)
            self.assertEqual(second["text"], "Respuesta al segundo turno")
            self.assertEqual(second["session_id"], result["session_id"])
            self.assertEqual(second["usage"]["input_tokens"], 42)
            self.assertEqual(second["usage"]["output_tokens"], 4)
            self.assertAlmostEqual(second["usage"]["reported_cost_usd"], .000250)
            self.assertIsNone(second["usage"]["cost_usd"])
            for rejected, status in ((denied, 403), (unauthorized, 401)):
                self.assertEqual(rejected["status"], "error")
                self.assertEqual((rejected["http_status"], rejected["http_error_scope"]), (status, "selected_model"))
                self.assertIn(f"HTTP {status}", rejected["error"])
                for field in ("cost_usd", "reported_cost_usd", "input_tokens", "output_tokens"):
                    self.assertIsNone(rejected["usage"][field])
            self.assertIsNone(denied["auth_failure_reason"])
            self.assertEqual(unauthorized["auth_failure_reason"], "credential_expired")
            self.assertIn("credencial venció", unauthorized["error"])
            self.assertEqual(len(requests), 4)
            histories = [
                ("local", [("user", prompt)]),
                ("default", [("user", prompt), ("assistant", "Relay OK"), ("user", follow_up)]),
                ("local", [("user", denied_prompt)]),
                ("local", [("user", auth_prompt)]),
            ]
            for (path, request), (model, messages) in zip(requests, histories, strict=True):
                self.assertEqual(path, "/v1/chat/completions")
                self.assertEqual(request["model"], model)
                self.assertEqual([(msg["role"], msg["content"]) for msg in request["messages"]
                                  if msg["role"] != "system"], messages)
