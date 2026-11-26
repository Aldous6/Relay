"""Build the dependency-free Relay runtime wheel from its reviewed source files.

Uses the standard wheel ZIP/RECORD format without downloading build tools.
The scientific package remains unchanged; only the explicit allowlist ships.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


VERSION = "0.1.0.dev5"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("workflow.py", "workflow_policy.py", "workflow_cli.py", "prompt_selection.py",
           "adaptive_policy.py", "adaptive_runtime.py", "adaptive_training.py",
           "cascade.py", "cascade_verifier.py", "cascade_evaluation.py", "cascade_unittest.py",
           "integrations/__init__.py", "integrations/openhands.py", "integrations/openhands_io.py",
           "integrations/openhands_cascade.py", "integrations/groq_recovery.py",
           "integrations/routellm.py")


def build(output: Path) -> Path:
    package = {}
    provenance = {}
    for relative in SOURCES:
        source = ROOT / "src" / "relay_router" / relative
        content = source.read_bytes()
        package[f"relay_harness/{relative}"] = content
        provenance[f"src/relay_router/{relative}"] = hashlib.sha256(content).hexdigest()
    package["relay_harness/__init__.py"] = (
        '"""Relay: task routing for existing Python harnesses."""\n'
        'from .workflow import Relay, TaskSession\n'
        'from .adaptive_runtime import load_adapted_relay\n'
        'from .cascade import SpendBudget, run_cascade\n'
        'from .cascade_verifier import CheckCommand, CommandVerifier\n'
        f'__version__ = "{VERSION}"\n'
        '__all__ = ["Relay", "TaskSession", "load_adapted_relay", "SpendBudget", "run_cascade", "CheckCommand", "CommandVerifier"]\n'
    ).encode()
    package["relay_harness/__main__.py"] = (
        'from .workflow_cli import main\nraise SystemExit(main())\n'
    ).encode()
    package["relay_harness/source_manifest.json"] = json.dumps(
        {"version": VERSION, "source_sha256": provenance,
         "scope": "local_runtime_preview_no_demonstrated_savings",
         "scientific_data_included": False}, indent=2, sort_keys=True,
    ).encode() + b"\n"
    info = f"relay_harness-{VERSION}.dist-info"
    package[f"{info}/METADATA"] = (
        "Metadata-Version: 2.1\nName: relay-harness\n"
        f"Version: {VERSION}\n"
        "Summary: Task-level routing and outcome accounting for Python harnesses\n"
        "Requires-Python: >=3.11\n"
        "Provides-Extra: openhands\n"
        "Provides-Extra: routellm-bert\n"
        "Provides-Extra: adapt\n"
        'Requires-Dist: openhands-sdk==1.62.0; extra == "openhands"\n'
        'Requires-Dist: httpx==0.28.1; extra == "openhands"\n'
        'Requires-Dist: litellm==1.105.7; extra == "openhands"\n'
        'Requires-Dist: torch==2.9.0; extra == "routellm-bert"\n'
        'Requires-Dist: transformers==4.57.1; extra == "routellm-bert"\n'
        'Requires-Dist: scikit-learn==1.8.0; extra == "adapt"\n\n'
        "Local preview. OpenHands integration requires Python >=3.12.\n"
    ).encode()
    package[f"{info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: relay-runtime-builder\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode()
    package[f"{info}/entry_points.txt"] = (
        "[console_scripts]\nrelay-harness = relay_harness.workflow_cli:main\n"
    ).encode()
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, content in sorted(package.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(content)))
    writer.writerow((f"{info}/RECORD", "", ""))
    package[f"{info}/RECORD"] = records.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"relay_harness-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(package.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 11, 17, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, content)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    wheel.with_suffix(".whl.sha256").write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")
    return wheel


def main(argv=None):
    parser = argparse.ArgumentParser(description="Construye el paquete local Relay para harnesses, sin descargas.")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "relay-harness")
    args = parser.parse_args(argv)
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
