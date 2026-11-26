"""Read-only credential check through a one-use memory pipe; no generation."""
import argparse
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-pipe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"\\\\\.\\pipe\\relay-groq-[a-f0-9]{32}", args.credential_pipe):
        raise ValueError("Expected a one-use local credential pipe")
    if args.output.exists():
        raise FileExistsError("Choose a fresh probe output")
    key = None
    result = {"provider": "Groq", "endpoint": "models", "generation_requests": 0}
    try:
        with open(args.credential_pipe, "rb", buffering=0) as stream:
            key = stream.readline(256).decode("ascii").strip()
        request = urllib.request.Request("https://api.groq.com/openai/v1/models",
                                         headers={"Authorization": "Bearer " + key, "User-Agent": "Relay-validation"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = json.load(response)
            available = {item["id"] for item in content["data"]}
            result.update(http_status=response.status,
                          required_models_available=all(model in available for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b")))
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code, error_type="HTTPError")
    except Exception as exc:
        result.update(http_status=None, error_type=type(exc).__name__)
    finally:
        key = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    return 0 if result.get("required_models_available") else 2


if __name__ == "__main__":
    raise SystemExit(main())
