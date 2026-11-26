"""Authorized Windows runner: credential arrives over a one-use memory pipe.

The pipe name is not a file or a credential. No secret is written to disk,
environment variables, command arguments, or child test processes.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-pipe", required=True)
    parser.add_argument("--check-access", action="store_true")
    parser.add_argument("--diagnose-request", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--diagnostic-mode", choices=("strict", "prompt_json"), default="strict")
    parser.add_argument("--resume-known-run", type=Path)
    parser.add_argument("--resume-output", type=Path)
    parser.add_argument("--resume-prior-calls", type=int)
    args, remainder = parser.parse_known_args()
    if not re.fullmatch(r"\\\\\.\\pipe\\relay-groq-[a-f0-9]{32}", args.credential_pipe):
        raise ValueError("Expected a one-use Relay Windows memory pipe.")
    with open(args.credential_pipe, "rb", buffering=0) as stream:
        key = stream.readline(256).decode("ascii").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("Missing credential from memory pipe.")
    try:
        if args.resume_known_run:
            if not args.resume_output or args.resume_prior_calls is None or remainder:
                raise ValueError("Known continuation requires destination and prior call count only.")
            import relay_resume_maintenance_pair
            from relay_router import maintenance_cli
            previous = maintenance_cli._key
            maintenance_cli._key = lambda: key
            try:
                directory = relay_resume_maintenance_pair.recover(
                    args.resume_known_run, args.resume_output, prior_calls=args.resume_prior_calls)
                audit = json.loads((directory / "recovery-audit.json").read_text(encoding="utf-8"))
                print(json.dumps({name: audit[name] for name in
                      ("status", "new_calls_started", "total_calls_after", "original_run_unchanged")}))
                return 0 if audit["status"] == "completed" else 2
            finally:
                maintenance_cli._key = previous
        if args.diagnose_request:
            from relay_router.maintenance_provider import GroqMaintenanceProvider, _transport, MaintenanceProviderError
            from datetime import datetime, timezone
            import hashlib
            import io
            original = args.diagnose_request.read_bytes()
            request_data = json.loads(original)
            evidence = {"kind": "explicit_single_request_diagnostic", "request_file_sha256": hashlib.sha256(original).hexdigest(),
                        "started_at_utc": datetime.now(timezone.utc).isoformat(), "automatic_retry": False,
                        "not_part_of_comparison": True, "model": request_data["model"],
                        "output_mode": args.diagnostic_mode}
            def diagnostic_transport(request, timeout):
                try:
                    response = _transport(request, timeout)
                    choices = response.get("choices", []) if type(response) is dict else []
                    if len(choices) == 1 and type(choices[0]) is dict:
                        content = choices[0].get("message", {}).get("content")
                        if type(content) is str:
                            evidence["raw_model_content"] = content.replace(key, "[redacted]")[:65536]
                            evidence["raw_model_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
                    return response
                except urllib.error.HTTPError as exc:
                    body = exc.read(65536)
                    try:
                        error = json.loads(body).get("error", {})
                    except (ValueError, AttributeError):
                        error = {}
                    evidence["http_error"] = {"status": exc.code,
                        "code": error.get("code") if re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", str(error.get("code"))) else None,
                        "type": error.get("type") if re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", str(error.get("type"))) else None,
                        "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
                        "failed_generation_present": "failed_generation" in error}
                    raise urllib.error.HTTPError(exc.url, exc.code, exc.reason, exc.headers, io.BytesIO(body)) from None
            provider = GroqMaintenanceProvider(key, max_requests=1, max_output_tokens=4096,
                                               request_budget=8000, transport=diagnostic_transport,
                                               output_mode=args.diagnostic_mode)
            try:
                evidence["response"] = provider.complete_structured(request_data["model"], request_data["messages"], request_data["response_schema"])
                evidence["status"] = "completed"
            except MaintenanceProviderError as exc:
                evidence.update(status="error", category=exc.category, usage=exc.usage)
            evidence["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            if args.diagnostic_output.exists():
                raise ValueError("Diagnostic output already exists.")
            args.diagnostic_output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({key: evidence[key] for key in ("status", "http_error", "category", "usage") if key in evidence}))
            return 0
        if args.check_access:
            request = urllib.request.Request("https://api.groq.com/openai/v1/models",
                headers={"Authorization": "Bearer " + key, "User-Agent": "Relay-Maintenance/1"})
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = json.load(response)
                models = {row["id"] for row in data.get("data", [])}
                print(json.dumps({"authenticated": True, "required_models_available":
                    {name: name in models for name in ("openai/gpt-oss-20b", "openai/gpt-oss-120b")}}))
                return 0
            except urllib.error.HTTPError as exc:
                detail = exc.read(2048).decode("utf-8", errors="replace").replace(key, "[redacted]")
                detail = re.sub(r"gsk_[A-Za-z0-9]+", "[redacted]", detail)
                print(json.dumps({"authenticated": False, "http_status": exc.code, "detail": detail}))
                return 2
            except Exception as exc:
                print(json.dumps({"authenticated": False, "error_type": type(exc).__name__}))
                return 2
        from relay_router import maintenance_competition
        previous = maintenance_competition._key
        maintenance_competition._key = lambda: key
        try:
            return maintenance_competition.main(remainder)
        finally:
            maintenance_competition._key = previous
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
