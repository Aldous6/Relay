"""Install a local optional CPU backend and download a pinned public encoder.

No APIs, credentials, telemetry configuration or global Python installation.
Only the project virtual environment, model files and provenance are written.
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
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    config = json.loads((ROOT / "configs/semantic_encoder.json").read_text(encoding="utf-8"))
    environment = ROOT / ".venv-relay"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
    packages = [f"{name}=={version}" for name, version in config["packages"].items()]
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", *packages], check=True)
    subprocess.run([str(python), "-c", "import onnxruntime,tokenizers; print('CPU backend:',onnxruntime.__version__,tokenizers.__version__)"], check=True)
    directory = ROOT / config["model_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    names = [config["model_file"], "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]
    files = {}
    for name in names:
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{config['model_id']}/resolve/{config['revision']}/{name}"
        if not destination.exists():
            print(f"Downloading {name}", flush=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = Request(url, headers={"User-Agent": "Relay-research-local-setup"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if name == config["model_file"] and checksum(temporary) != config["model_sha256"]:
                raise ValueError("El ONNX descargado no coincide con el hash oficial.")
            temporary.replace(destination)
        digest = checksum(destination)
        if name == config["model_file"] and (digest != config["model_sha256"] or destination.stat().st_size != config["model_bytes"]):
            raise ValueError("El ONNX local no coincide con el archivo oficial fijado.")
        files[name] = {"sha256": digest, "bytes": destination.stat().st_size, "source": url}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model_id": config["model_id"],
                "revision": config["revision"], "files": files, "specification": config,
                "paid_api_calls": False, "model_inputs_uploaded": False}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Local encoder ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
