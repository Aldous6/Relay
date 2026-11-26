"""Exercise the local HTTP handler entirely in memory, without binding a port."""
from contextlib import redirect_stdout
from http.client import HTTPResponse
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.opencode_history import ChatConflict, ChatHistory
from relay_router import web_server
from relay_router.web_server import Handler, LocalApp, REPORTS


class MemoryConnection:
    def __init__(self, incoming):
        self.incoming = BytesIO(incoming)
        self.outgoing = bytearray()

    def makefile(self, mode, *args):
        assert mode == "rb"
        return self.incoming

    def sendall(self, chunk):
        self.outgoing.extend(chunk)


class WebServerTests(unittest.TestCase):
    port = 8501
    host = "127.0.0.1:8501"

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.out = self.root / "web/out"
        self.out.mkdir(parents=True)
        (self.out / "index.html").write_bytes(b"<html>Relay</html>")
        self.app = LocalApp(self.root)
        self.app.chat = Mock()
        self.app.documents = Mock()

    def http(self, method, path, headers=None, body=b""):
        fields = {"Host": self.host, **(headers or {})}
        request = [f"{method} {path} HTTP/1.1", *(f"{k}: {v}" for k, v in fields.items() if v is not None)]
        connection = MemoryConnection(("\r\n".join(request) + "\r\n\r\n").encode("ascii") + body)
        Handler(connection, ("127.0.0.1", 12345), SimpleNamespace(server_port=self.port), app=self.app)
        response = HTTPResponse(MemoryConnection(connection.outgoing), method=method)
        response.begin()
        return response, response.read()

    def post(self, action, payload, **overrides):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Origin": "http://" + self.host, "X-Relay-Client": "relay-web",
                   "Content-Type": "application/json; charset=utf-8", "Content-Length": len(body), **overrides}
        response, body = self.http("POST", "/api/" + action, headers, body)
        return response.status, json.loads(body)

    def test_entry_assets_and_static_cache_headers(self):
        asset = self.out / "_next/static/chunks/chat.js"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"/*compiled*/")
        for path, mime, content in (("/", "text/html", b"<html>Relay</html>"),
                                    ("/_next/static/chunks/chat.js?v=1", "text/javascript", b"/*compiled*/")):
            with self.subTest(path=path):
                response, data = self.http("GET", path)
                self.assertEqual(response.status, 200)
                self.assertEqual(data, content)
                self.assertTrue(response.getheader("Content-Type").startswith(mime))
                self.assertEqual(int(response.getheader("Content-Length")), len(content))
                self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
                self.assertIn("immutable" if path.startswith("/_next") else "no-store", response.getheader("Cache-Control"))
        response, data = self.http("HEAD", "/")
        self.assertEqual((response.status, data), (200, b""))
        self.assertEqual(int(response.getheader("Content-Length")), len(b"<html>Relay</html>"))
        self.app.chat.request.assert_not_called()

    def test_documents_route_supports_both_next_export_layouts(self):
        route = self.out / "documentos"
        route.mkdir()
        (route / "index.html").write_bytes(b"directory export")
        self.assertEqual(self.http("GET", "/documentos/")[1], b"directory export")
        (self.out / "documentos.html").write_bytes(b"html export")
        self.assertEqual(self.http("GET", "/documentos")[1], b"html export")
        self.app.documents.request.assert_not_called()

    def test_public_reports_and_status_are_limited_to_the_two_named_files(self):
        self.app.documents.request.side_effect = lambda action, data: {"available": False}
        status, state = self.post("documentos-state", {})
        self.assertEqual((status, state["has_report"], state["has_slides"]), (200, False, False))
        for url, filename in REPORTS.items():
            target = self.root / filename
            target.parent.mkdir(exist_ok=True, parents=True)
            content = b"%PDF fixture " + url.encode("ascii")
            target.write_bytes(content)
            response, downloaded = self.http("GET", url)
            self.assertEqual((response.status, response.getheader("Content-Type"), downloaded),
                             (200, "application/pdf", content))
            self.assertEqual(self.http("GET", "/" + filename)[0].status, 404)
        status, state = self.post("documentos-state", {"profile": "frozen_v2"})
        self.assertEqual((status, state["has_report"], state["has_slides"]), (200, True, True))
        self.app.documents.request.assert_called_with("documentos-state", {"profile": "frozen_v2"})
        (self.root / "reports/exposicion/notas.pdf").touch()
        (self.out / "implementacion.py").touch()
        for unavailable in ("/estudio/notas.pdf", "/implementacion.py", "/api/state"):
            self.assertEqual(self.http("GET", unavailable)[0].status, 404)

    def test_same_origin_json_post_reaches_the_correct_service(self):
        payload = {"prompt": "¿Cómo?", "chat": "nuevo", "previous": None}
        self.app.chat.request.return_value = {"turn_id": 3, "result": {"text": "Sí"}}
        self.assertEqual(self.post("send", payload), (200, self.app.chat.request.return_value))
        self.app.chat.request.assert_called_once_with("send", payload)
        self.app.documents.request.assert_not_called()
        self.app.documents.request.return_value = {"answer": "42"}
        self.assertEqual(self.post("documentos-send", {"prompt": "Dato", "document": "42"}), (200, {"answer": "42"}))
        self.app.documents.request.assert_called_once_with("documentos-send", {"prompt": "Dato", "document": "42"})

    def test_host_origin_and_client_header_must_match_before_dispatch(self):
        self.assertEqual(self.http("GET", "/", {"Host": "example.test"})[0].status, 403)
        for key, wrong in (("Host", "example.test"), ("Origin", "http://example.test"),
                           ("X-Relay-Client", "other")):
            for value in (None, wrong):
                with self.subTest(key=key, value=value):
                    self.assertEqual(self.post("send", {}, **{key: value})[0], 403)
        self.app.chat.request.assert_not_called()
        self.app.documents.request.assert_not_called()

    def test_delete_chat_http_route_accepts_only_local_requests_and_known_chats(self):
        self.enterContext(patch.dict(os.environ, {"RELAY_CHAT_DB": str(self.root / "test-chat.sqlite3")}))
        self.app = LocalApp(self.root, frontend_port=3000)
        history = ChatHistory(self.root)
        turn = history.begin("elegida", None, "Conversación", {"model": "test/local", "reason": "Prueba"})
        history.finish("elegida", turn, {"status": "completed"})
        payload = {"workspace": str(self.root), "chat": "elegida"}
        self.assertEqual(self.post("delete-chat", payload, Origin="http://ejemplo.test")[0], 403)
        self.assertEqual(len(history.turns("elegida")), 1)
        origin = "http://127.0.0.1:3000"
        self.assertEqual(self.post("delete-chat", payload, Origin=origin), (200, {"deleted": True}))
        self.assertFalse(history.turns("elegida"))
        self.assertEqual(self.post("delete-chat", {**payload, "chat": "desconocida"}, Origin=origin)[0], 404)

    def test_dev_api_accepts_only_the_explicit_frontend_port_and_backend_host(self):
        self.app = LocalApp(self.root, frontend_port=3000)
        self.app.chat = Mock()
        self.app.chat.request.return_value = {"chat": "activo"}
        for origin in ("http://127.0.0.1:3000", "http://localhost:3000"):
            self.assertEqual(self.post("state", {}, Origin=origin)[0], 200)
        self.app.chat.reset_mock()
        for origin in (None, "http://localhost:3001", "http://sitio.test:3000", "http://" + self.host):
            self.assertEqual(self.post("state", {}, Origin=origin)[0], 403)
        self.assertEqual(self.post("state", {}, Origin="http://localhost:3000", Host="localhost:3000")[0], 403)
        self.assertEqual(self.post("state", {}, Origin="http://localhost:3000", **{"X-Relay-Client": None})[0], 403)
        self.app.chat.request.assert_not_called()

    def test_dev_api_serves_only_the_named_reports_even_if_an_export_exists(self):
        self.app = LocalApp(self.root, frontend_port=3000)
        self.assertEqual(self.http("GET", "/")[0].status, 404)
        self.assertEqual(self.http("GET", "/index.html")[0].status, 404)
        self.assertEqual(self.http("GET", "/__relay/health")[0].status, 404)
        for url, filename in REPORTS.items():
            file = self.root / filename
            file.parent.mkdir(exist_ok=True, parents=True)
            file.write_bytes(b"%PDF")
            response, content = self.http("GET", url)
            self.assertEqual((response.status, content), (200, b"%PDF"))
            self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))

    def test_dev_cli_reports_assigned_loopback_port_without_build_or_reuse(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(web_server, "ROOT", Path(folder)), \
                patch.object(web_server, "RelayHTTPServer") as constructor, \
                patch.object(web_server, "_same_instance") as probe, redirect_stdout(StringIO()) as output:
            constructor.return_value.server_port = 49152
            self.assertEqual(web_server.main(["--dev-api", "--frontend-port", "3070"]), 0)
        self.assertEqual(constructor.call_args.args[0], ("127.0.0.1", 0))
        self.assertEqual(json.loads(output.getvalue()), {"relay_api_port": 49152})
        constructor.return_value.serve_forever.assert_called_once_with()
        probe.assert_not_called()

    def test_conflicts_and_invalid_requests_are_machine_readable(self):
        self.app.chat.request.side_effect = ChatConflict("El turno ya cambió.")
        self.assertEqual(self.post("send", {"previous": 7}), (409, {"error": "El turno ya cambió."}))
        self.app.chat.request.reset_mock()
        self.assertEqual(self.post("send", {}, **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post("missing", {})[0], 404)
        self.assertEqual(self.post("send", [1, 2])[0], 400)
        self.assertEqual(self.post("send", {}, **{"Content-Length": 32 * 1024 * 1024 + 1})[0], 413)
        self.app.chat.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
