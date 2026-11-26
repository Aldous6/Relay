"""Serve the compiled Relay workspace and its local JSON API on loopback."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from urllib.parse import unquote, urlsplit
import webbrowser

from .integrations.opencode import OpenCodeError
from .opencode_history import ChatConflict, ChatNotFound
from .web_chat import ChatService
from .web_documents import DocumentService, DocumentsUnavailable


ROOT = Path(__file__).resolve().parents[2]
CHAT_ACTIONS = {"state", "preferences", "preview", "send", "reenable", "delete-chat"}
DOCUMENT_ACTIONS = {"documentos-state", "documentos-send"}
REPORTS = {
    "/estudio/informe.pdf": "reports/exposicion/Relay_guia_de_estudio.pdf",
    "/estudio/presentacion.pdf": "reports/exposicion/Relay_presentacion_final.pdf",
}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".png": "image/png", ".ico": "image/x-icon",
                ".woff2": "font/woff2", ".woff": "font/woff", ".svg": "image/svg+xml",
                ".json": "application/json", ".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8"}


class LocalApp:
    def __init__(self, root: Path, *, frontend_port: int | None = None):
        self.root = root.resolve()
        self.assets = self.root / "web/out"
        self.dev_origins = (() if frontend_port is None else
                            (f"http://127.0.0.1:{frontend_port}", f"http://localhost:{frontend_port}"))
        example = self.root / "ejemplo"
        self.chat = ChatService(example if example.is_dir() else self.root)
        self.documents = DocumentService(self.root)
        self.has_documents = (self.root / "src/routerbench/demo.py").is_file()
        self.identity = None
        if self.dev_origins:
            return
        if not (self.assets / "index.html").is_file():
            raise FileNotFoundError("Falta la interfaz compilada. Extrae el ZIP completo o compila el directorio web.")
        # A second launch can reuse only the same checkout, build and Python service.
        digest = hashlib.sha256(os.fsencode(os.path.normcase(str(self.root))))
        digest.update((self.assets / "index.html").read_bytes())
        for name in ("web_server.py", "web_chat.py", "web_documents.py", "opencode_history.py",
                     "opencode_selection.py", "integrations/opencode.py"):
            digest.update((Path(__file__).parent / name).read_bytes())
        self.identity = digest.hexdigest()

    def request(self, action, data):
        if action in DOCUMENT_ACTIONS:
            response = self.documents.request(action, data)
            if action == "documentos-state":
                response["has_report"] = (self.root / REPORTS["/estudio/informe.pdf"]).is_file()
                response["has_slides"] = (self.root / REPORTS["/estudio/presentacion.pdf"]).is_file()
            return response
        response = self.chat.request(action, data)
        if action in {"state", "preferences", "reenable"}:
            response["has_documents"] = self.has_documents
        return response

    def public_file(self, request_path):
        path = unquote(request_path, errors="strict")
        if path in REPORTS:
            file = (self.root / REPORTS[path]).resolve()
            return file if file.parent == self.root / "reports/exposicion" and file.is_file() else None
        if self.dev_origins:
            return None  # Next.js serves pages and assets in development.
        parts = path.strip("/").split("/")
        if not path.startswith("/") or any(p in {".", ".."} or "\\" in p or ":" in p or "\x00" in p for p in parts):
            return None
        relative = path.strip("/") or "index.html"
        if relative == "documentos":
            relative = "documentos.html" if (self.assets / "documentos.html").is_file() else "documentos/index.html"
        file = (self.assets / relative).resolve()
        if self.assets not in file.parents or file.suffix not in STATIC_TYPES or not file.is_file():
            return None
        return file


class RelayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    server_version = "Relay"
    sys_version = ""

    def __init__(self, *args, app: LocalApp):
        self.app = app
        super().__init__(*args)

    def log_message(self, format, *args):
        # Requests may contain project or conversation identifiers.
        pass

    def _host(self):
        return f"127.0.0.1:{self.server.server_port}"

    def _headers(self, status, content_type, length, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()

    def _json(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.headers.get("Host") != self._host():
            return self._json(403, {"error": "Abre la dirección local que indica Relay."})
        path = urlsplit(self.path).path
        if path == "/__relay/health" and self.app.identity is not None:
            return self._json(200, {"instance": self.app.identity})
        try:
            file = self.app.public_file(path)
        except (ValueError, OSError, RuntimeError):
            file = None
        if file is None:
            return self._json(404, {"error": "Este recurso no existe."})
        with file.open("rb") as source:
            cache = "public, max-age=31536000, immutable" if path.startswith("/_next/static/") else "no-store"
            self._headers(200, STATIC_TYPES[file.suffix], os.fstat(source.fileno()).st_size, cache)
            if self.command != "HEAD":
                shutil.copyfileobj(source, self.wfile)

    def do_POST(self):
        allowed_origins = self.app.dev_origins or (f"http://{self._host()}",)
        if (self.headers.get("Host") != self._host() or self.headers.get("Origin") not in allowed_origins
                or self.headers.get("X-Relay-Client") != "relay-web"):
            return self._json(403, {"error": "La operación debe iniciarse desde la ventana local de Relay."})
        action = self.path.removeprefix("/api/")
        if not self.path.startswith("/api/") or action not in CHAT_ACTIONS | DOCUMENT_ACTIONS:
            return self._json(404, {"error": "Operación desconocida."})
        if self.headers.get_content_type() != "application/json":
            return self._json(415, {"error": "La solicitud debe ser JSON."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("La solicitud está vacía.")
            if length > 32 * 1024 * 1024:
                return self._json(413, {"error": "La solicitud completa supera 32 MiB. Reduce el documento o inicia otra conversación."})
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Se esperaba un objeto JSON.")
            response = self.app.request(action, data)
        except (UnicodeError, json.JSONDecodeError):
            return self._json(400, {"error": "El mensaje debe ser JSON válido en UTF-8."})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except ChatNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except ChatConflict as exc:
            return self._json(409, {"error": str(exc)})
        except (OpenCodeError, DocumentsUnavailable) as exc:
            return self._json(503, {"error": str(exc)})
        except (OSError, sqlite3.Error):
            return self._json(503, {"error": "Relay no pudo acceder al recurso local. Revisa la carpeta y vuelve a intentarlo."})
        except Exception:
            return self._json(500, {"error": "Relay no pudo completar la operación."})
        self._json(200, response)


def _same_instance(port, identity):
    connection = HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", "/__relay/health")
        response = connection.getresponse()
        return response.status == 200 and json.loads(response.read(256)) == {"instance": identity}
    except (OSError, ValueError, EOFError, HTTPException):
        return False
    finally:
        connection.close()


def _port(value):
    port = int(value)
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("El puerto debe estar entre 1024 y 65535.")
    return port


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--port", type=_port, default=8501, metavar="PUERTO")
    mode.add_argument("--dev-api", action="store_true", help="Sirve la API para el frontend local de desarrollo.")
    parser.add_argument("--frontend-port", type=_port, metavar="PUERTO")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--documents", action="store_true")
    args = parser.parse_args(argv)
    if args.dev_api != (args.frontend_port is not None):
        parser.error("--dev-api y --frontend-port se usan juntos.")
    if args.dev_api and (args.open or args.documents):
        parser.error("En desarrollo, Next.js abre las páginas.")
    try:
        app = LocalApp(ROOT, frontend_port=args.frontend_port)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    page = "/documentos/" if args.documents else "/"
    ports = (0,) if args.dev_api else range(args.port, min(65536, args.port + 16))
    for port in ports:
        try:
            server = RelayHTTPServer(("127.0.0.1", port), partial(Handler, app=app))
        except OSError as exc:
            if args.dev_api:
                print(f"No se pudo iniciar la API local de Relay. {exc}", file=sys.stderr)
                return 1
            if not _same_instance(port, app.identity):
                continue
            url = f"http://127.0.0.1:{port}{page}"
            print(f"Relay ya está abierto en {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            return 0
        with server:
            if args.dev_api:
                # The dev launcher waits for this single readiness record before starting Next.js.
                print(json.dumps({"relay_api_port": server.server_port}), flush=True)
            else:
                url = f"http://127.0.0.1:{port}{page}"
                print(f"Relay está en {url}\nDeja esta ventana abierta. Pulsa Ctrl+C para salir.", flush=True)
                if args.open:
                    webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return 0
    print("Los puertos de Relay están ocupados. Elige otro con --port.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
