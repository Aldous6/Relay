"""Prepare a pinned, local CPU RouteLLM BERT reference without provider APIs.

Writes only .venv-routellm and the configured project artifact directory. Requires
network approval for package/model downloads; does not train or run inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected_sha256=None):
    destination = Path(destination)
    if destination.is_file():
        if expected_sha256 is not None and checksum(destination) != expected_sha256:
            raise ValueError(f"El archivo existente no coincide: {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "Relay-offline-reference-setup"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        append = offset > 0 and response.status == 206
        with temporary.open("ab" if append else "wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    if expected_sha256 is not None and checksum(temporary) != expected_sha256:
        raise ValueError(f"La descarga no coincide con el SHA-256 fijado: {destination.name}")
    temporary.replace(destination)


def main():
    config = json.loads((ROOT / "configs/routellm_reference.json").read_text(encoding="utf-8"))
    environment = ROOT / ".venv-routellm"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    f"torch=={config['packages']['torch']}", "--index-url", config["torch_index_url"]], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    f"transformers=={config['packages']['transformers']}"], check=True)
    subprocess.run([str(python), "-c", "import torch,transformers; print('Reference CPU backend:',torch.__version__,transformers.__version__)"], check=True)
    directory = (ROOT / config["model_dir"]).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("La carpeta del modelo debe estar dentro del proyecto.")
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in (config["weights_file"], "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "LICENSE"):
        target = directory / name
        url = f"https://huggingface.co/{config['checkpoint']}/resolve/{config['checkpoint_revision']}/{name}"
        print(f"Preparing {name}", flush=True)
        download(url, target, config["weights_sha256"] if name == config["weights_file"] else None)
        files[name] = {"sha256": checksum(target), "bytes": target.stat().st_size, "source": url}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": config["checkpoint"],
                "revision": config["checkpoint_revision"], "files": files,
                "paid_api_calls": False, "inference_executed": False,
                "license": config["license"], "configuration": config}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Local reference ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
