"""Package the compiled local Windows chat and verify it with Python's standard library."""
from __future__ import annotations

import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "Relay-prueba/"
CHAT_FILES = (
    "Abrir Relay.cmd", "src/relay_router/__init__.py", "src/relay_router/errors.py",
    "src/relay_router/web_server.py", "src/relay_router/web_chat.py", "src/relay_router/web_documents.py",
    "src/relay_router/opencode_history.py", "src/relay_router/opencode_selection.py",
    "src/relay_router/integrations/__init__.py", "src/relay_router/integrations/opencode.py",
)
PUBLIC_FILES = {"index.html", "404.html", "404/index.html", "500.html", "documentos.html",
                "documentos/index.html", "routing-switch.png", "favicon.ico"}
STATIC_SUFFIXES = {".js", ".css", ".woff", ".woff2", ".ttf", ".png", ".ico", ".svg"}
LICENSE_PACKAGES = ("next", "react", "react-dom", "scheduler")


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        field = {"script": "src", "link": "href", "img": "src"}.get(tag)
        if not field or (tag == "link" and not set((attrs.get("rel") or "").split()).intersection(
            {"preload", "stylesheet", "icon", "modulepreload"}
        )):
            return
        url = urlsplit(attrs.get(field) or "")
        if not url.scheme and not url.netloc and url.path.startswith("/"):
            self.local.add(url.path.lstrip("/"))


def inputs(modules):
    sources = {name: ROOT / name for name in CHAT_FILES}
    sources["LEEME.txt"] = ROOT / "docs/prueba_web_windows.txt"
    for file in ("ventas.csv", "total.py"):
        sources["ejemplo/" + file] = ROOT / "examples/prueba_web" / file
    assets = ROOT / "web/out"
    for file in sorted(assets.rglob("*")):
        if not file.is_file():
            continue
        name = file.relative_to(assets).as_posix()
        notice = "license" in file.name.casefold() or "notice" in file.name.casefold()
        if name in PUBLIC_FILES or (name.startswith("_next/static/") and file.suffix in STATIC_SUFFIXES) or (
            notice and file.suffix.casefold() == ".txt"
        ):
            sources["web/out/" + name] = file
    index = assets / "index.html"
    if not index.is_file() or "web/out/routing-switch.png" not in sources:
        raise ValueError("Falta la exportación completa en web/out.")
    for page in (p for k, p in sources.items() if k.startswith("web/out/") and p.suffix == ".html"):
        refs = AssetReferences()
        refs.feed(page.read_text(encoding="utf-8"))
        if not refs.local or any("web/out/" + name not in sources for name in refs.local):
            raise ValueError(f"Hay recursos compilados sin empaquetar en {page.name}.")
    for package in (*LICENSE_PACKAGES, "styled-jsx", "@swc/helpers", "tslib"):
        directory = modules / package
        if package not in LICENSE_PACKAGES and not directory.is_dir():
            continue
        identifier = json.loads((directory / "package.json").read_text(encoding="utf-8"))["version"]
        notices = sorted(p for p in directory.iterdir() if p.is_file() and p.name.upper().startswith(
            ("LICENSE", "LICENCE", "NOTICE", "COPYING", "COPYRIGHT", "THIRDPARTYNOTICE", "THIRD-PARTY-NOTICE")
        ))
        if not notices:
            raise ValueError(f"Falta la licencia de {package} {identifier}.")
        for notice in notices:
            sources[f"licencias/{package}-{identifier}/{notice.name}"] = notice
    return sources


SMOKE = """
from pathlib import Path
import os, sys, tempfile
from unittest.mock import patch
root = Path.cwd()
sys.path.insert(0, str(root / 'src'))
from relay_router import web_chat, web_server
from relay_router.integrations.opencode import OpenCodeModel
assert Path(web_server.__file__).resolve().is_relative_to(root)
model = OpenCodeModel('test/small', 'Prueba', 0.0, 0.0, context_tokens=8192, output_tokens=1024, tool_call=True)
reply = {'status': 'completed', 'text': '98', 'session_id': 'ses_demo'}
with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'RELAY_CHAT_DB': str(Path(tmp)/'test.sqlite')}), \
     patch.object(web_chat, 'discover_models', return_value=[model]), patch.object(web_chat, 'run_prompt', return_value=reply) as generate:
    app = web_server.LocalApp(root)
    state = app.request('state', {})
    assert not state['has_documents'] and state['turns'] == []
    assert Path(state['workspace']).resolve() == (root/'ejemplo').resolve()
    assert app.public_file('/').is_file()
    address = {'workspace': state['workspace'], 'chat': state['chat']}
    prepared = app.request('preview', {**address, 'prompt': 'Calcula el ingreso'})
    assert prepared['decision']['model'] == model.model_id
    generate.assert_not_called()
    app.request('send', {**address, 'prompt': 'Calcula el ingreso', 'previous': None, 'expected_model': model.model_id})
    assert app.request('state', address)['turns'][-1]['result']['text'] == '98'
    generate.assert_called_once()
assert 'streamlit' not in sys.modules
"""


def package(destination, modules):
    sources = inputs(modules)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="relay-", suffix=".tmp", delete=False) as tmp:
        candidate = Path(tmp.name)
    try:
        digests = {}
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(sources.items()):
                data = path.read_bytes()
                info = zipfile.ZipInfo(PREFIX + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                archive.writestr(info, data)
                digests[PREFIX + name] = sha256(data).digest()
        with zipfile.ZipFile(candidate) as archive, tempfile.TemporaryDirectory(prefix="relay-web-qa-") as temp:
            if len(archive.namelist()) != len(digests) or set(archive.namelist()) != set(digests) or archive.testzip():
                raise ValueError("El ZIP no coincide con los archivos seleccionados.")
            archive.extractall(temp)
            for name, digest in digests.items():
                if sha256((Path(temp) / name).read_bytes()).digest() != digest:
                    raise ValueError(f"El archivo extraído no coincide con su fuente: {name}")
            subprocess.run([sys.executable, "-I", "-B", "-S", "-c", SMOKE],
                           cwd=Path(temp) / PREFIX.rstrip("/"), check=True)
        os.replace(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)
    print(f"{destination}: {len(sources)} archivos, {destination.stat().st_size} bytes")
    print("SHA-256", sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modules = ROOT / "web/node_modules"
    if not (modules / "next/package.json").is_file():
        modules = ROOT / "node_modules"
    parser.add_argument("--node-modules", type=Path, default=modules)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/relay-prueba/Relay-chat-Windows.zip")
    options = parser.parse_args()
    package(options.output.resolve(), options.node_modules.resolve())
