"""One separately logged provider diagnostic of an already closed BFCL request.

Never resumes or changes a benchmark episode. No retries; the credential remains
in memory. Error diagnostics exclude headers, URLs and failed_generation bodies.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from report_bfcl import audit_bfcl
from run_bfcl import build_generator
from routerbench.agent_evaluation import object_hash
from routerbench.agent_litellm import _default_completion
from routerbench.bfcl_harness import BFCLRequest, BFCLState
from routerbench.bfcl_litellm import BFCLPacedGenerator


def safe_error_facts(error, key):
    queue, seen, facts = [error], set(), []
    while queue and len(seen) < 12:
        item = queue.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        body = getattr(item, "body", None)
        if body is None:
            response = getattr(item, "response", None)
            if response is not None:
                try:
                    body = response.json()
                except Exception:
                    pass
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                body = None
        if isinstance(body, dict):
            data = body.get("error", body)
            if isinstance(data, dict):
                row = {}
                for name in ("code", "type", "param", "message"):
                    value = data.get(name)
                    if isinstance(value, str):
                        value = value.replace(key, "[REDACTED]")
                        value = re.sub(r"(?:gsk_|sk-)[A-Za-z0-9_-]+", "[REDACTED]", value)
                        value = re.sub(r"https?://\S+", "[URL OMITTED]", value)
                        row[name] = value[:2000]
                if row and row not in facts:
                    facts.append(row)
        for name in ("__cause__", "__context__", "original_exception"):
            child = getattr(item, name, None)
            if isinstance(child, BaseException):
                queue.append(child)
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new diagnostic file.")
    audited = audit_bfcl(args.run)
    if not audited["summary_current"]:
        raise ValueError("Only a closed, audited benchmark can be diagnosed separately.")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))["payload"]
    event = summary["episodes"][-1]["events"][-1]
    value = deepcopy(event["request"])
    for name in ("previous_errors", "infrastructure_errors", "failed_tiers"):
        value["state"][name] = tuple(value["state"][name])
    request = BFCLRequest(tuple(value["messages"]), tuple(value["tools"]), BFCLState(**value["state"]))
    config = manifest["configuration"]
    record = {"scope": "one separately authorized connection/contract diagnostic; not a benchmark retry or result",
        "benchmark_freeze_sha256": manifest["freeze_sha256"], "request_sha256": event["request_sha256"],
        "requested_model": event["model"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "maximum_provider_calls": 1, "provider_calls": 0, "outcome": "prepared", "usage": None}
    previous = os.environ.get("GROQ_API_KEY")
    try:
        key = previous or getpass.getpass("Groq API key (oculta; solo en memoria): ")
        if not key or any(c.isspace() for c in key):
            raise ValueError("Invalid credential input.")
        os.environ["GROQ_API_KEY"] = key
        completion = _default_completion()
        def observed(**kwargs):
            record.update(provider_calls=1, outcome="call_started")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            try:
                return completion(**kwargs)
            except Exception as error:
                record["provider_error_facts"] = safe_error_facts(error, key)
                raise
        base = build_generator(config)
        base._completion = observed  # Explicit diagnostic hook, never a benchmark generator.
        generate = BFCLPacedGenerator(base, **config["pacing"],
            max_output_tokens=config["generation_options"]["max_output_tokens"], show_progress=True)
        try:
            reply = generate(event["model"], request)
        except Exception as error:
            record.update(outcome="error", error_type=type(error).__name__,
                error_details=error.details() if hasattr(error, "details") else None,
                usage=getattr(error, "usage", None))
        else:
            record.update(outcome="response_received", usage=reply.usage,
                native_tool_call_count=len(reply.tool_calls), response_sha256=object_hash(reply.to_dict()))
        record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    finally:
        if previous is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous


if __name__ == "__main__":
    main()
