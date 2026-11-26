"""Install and exercise a released wheel offline with the host's OpenHands SDK."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0.dev5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(wheel, output):
    example = ROOT / "examples/relay_openhands_cascade.py"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        manifest = json.loads(archive.read("relay_harness/source_manifest.json"))
    if manifest["version"] != VERSION or sha(wheel) != wheel.with_suffix(".whl.sha256").read_text().split()[0]:
        raise ValueError("Wheel checksum or version mismatch")
    with TemporaryDirectory(prefix="relay-customer-install-") as tmp:
        clean = Path(tmp)
        site = clean / "installed"
        subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check",
                        "install", "--no-index", "--no-deps", "--target", str(site), str(wheel.resolve())],
                       cwd=clean, check=True)
        for source, expected in manifest["source_sha256"].items():
            installed = site / "relay_harness" / Path(source).relative_to("src/relay_router")
            if sha(installed) != expected or sha(ROOT / source) != expected:
                raise ValueError(f"Installed-source mismatch in {source}")
        shutil.copyfile(example, clean / "demo.py")
        # -I ignores the checkout, PYTHONPATH and user site. Use the fresh install
        # with the host interpreter's existing OpenHands dependencies.
        code = (
            "import pathlib, runpy, sys\n"
            "site, demo, output = sys.argv[1:]\n"
            "sys.path.insert(0, site)\n"
            "import relay_harness\n"
            "assert pathlib.Path(relay_harness.__file__).resolve().is_relative_to(site)\n"
            "example = runpy.run_path(demo)\n"
            "assert example['NAMESPACE'] == 'relay_harness'\n"
            "example['run_offline_demo'](output, provider_rejection=True)\n"
        )
        subprocess.run([sys.executable, "-I", "-B", "-c", code, str(site), str(clean / "demo.py"),
                        str(output / "demo")], cwd=clean, check=True)
    run = output / "demo/run"
    result = json.loads((run / "result.json").read_bytes())
    accounts = result["accounting"]
    rows = [json.loads(line) for line in (run / "transport.jsonl").read_text().splitlines()]
    if (not result["original_snapshot_unchanged"] or result["selected_attempt"] != 2
            or result["accepted"] is not None or result["savings_usd"] is not None
            or [r["kind"] for r in rows] != ["cascade_call_started", "cascade_call_finished"] * 2
            or rows[1]["http_status"] != 400 or rows[1]["provider_request_id"] != "req_simulated_rejection"
            or rows[1]["cost_usd"] is not None or not rows[1]["cost_upper_bound_usd"] > 0
            or rows[3]["http_status"] != 200 or not rows[3]["cost_usd"] > 0):
        raise ValueError("The simulated HTTP regression did not exercise the released recovery")
    report = {"version": VERSION, "wheel_sha256": sha(wheel), "sources_verified": len(manifest["source_sha256"]),
              "example_sha256": sha(example), "verifier_sha256": sha(Path(__file__)),
              "dependencies": {p: version(p) for p in ("openhands-sdk", "litellm", "httpx")},
              "scenario": "Groq tool_use_failed over simulated HTTP with real OpenHands and LiteLLM",
              "provider_requests": 0, "real_user_install_test": False, "installation": "fresh pip target, existing SDK",
              "selected_model": result["selected_model"], "simulated_http_requests": accounts["http_calls"],
              "cost_usd": accounts["cost_usd"], "known_cost_subtotal_usd": accounts["known_cost_subtotal_usd"],
              "cost_upper_bound_usd": accounts["cost_upper_bound_usd"], "accepted": None, "savings": None,
              "result_sha256": sha(run / "result.json"), "transport_sha256": sha(run / "transport.jsonl")}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=ROOT / "dist/relay-harness" / f"relay_harness-{VERSION}-py3-none-any.whl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.wheel, args.output), indent=2))
