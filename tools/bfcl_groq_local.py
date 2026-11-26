"""Prepare all 24 BFCL development groups; --execute uses a hidden in-memory key.

No model request is made unless --execute is supplied. The existing Free account
declaration is reused. Credentials never enter configuration, files or arguments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_bfcl

DEFAULT_CONFIG = ROOT / "configs/bfcl_groq_free.example.json"
MODELS = ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")
MAXIMUM_GENERATION_CALLS = 77 * 21 * 3
SESSION_LIMITS = {"max_tokens_per_model": 160000, "max_provider_calls": 250}


def prepare_config(directory, config_path=DEFAULT_CONFIG):
    """Create an immutable local configuration without touching credentials."""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Use a new BFCL comparison output directory.")
    config = run_bfcl._config(Path(config_path))
    if tuple(config["model_catalog"]) != MODELS or any(
            row.get("api_key_env") != "GROQ_API_KEY" for row in config["models_metadata"].values()):
        raise ValueError("This helper requires the declared Groq GPT-OSS 20B/120B catalog.")
    spend = config.get("spending", {})
    if spend.get("account_is_free_tier") is not True or spend.get("max_provider_spend_usd") != 0:
        raise ValueError("The existing config must declare the confirmed Free account and zero provider budget.")
    config.update(run_id="bfcl-development-comparison", max_total_generation_calls=MAXIMUM_GENERATION_CALLS,
        session_stop_on_infrastructure=True, session_limits=dict(SESSION_LIMITS),
        local_helper_source={"path": str(Path(__file__).resolve()),
                             "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    run_bfcl._check_no_inline_secrets(config)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "config.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    directory = (args.output or ROOT / "outputs/bfcl-comparison" /
                 datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    path = prepare_config(directory, args.config)
    command = ["--config", str(path), "--output", str(directory / "runplan")]
    if not args.execute:
        return run_bfcl.main(command)
    previous = os.environ.get("GROQ_API_KEY")
    try:
        key = previous
        if not key:
            if not sys.stdin.isatty():
                raise RuntimeError("Use a terminal for hidden key entry, or the authorized GROQ_API_KEY environment variable.")
            key = getpass.getpass("Groq API key (oculta; solo en memoria): ")
        if not key or any(character.isspace() for character in key):
            raise ValueError("The API key must be nonempty and contain no whitespace.")
        os.environ["GROQ_API_KEY"] = key
        del key
        return run_bfcl.main([*command, "--execute"])
    finally:
        if previous is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous


if __name__ == "__main__":
    main()
