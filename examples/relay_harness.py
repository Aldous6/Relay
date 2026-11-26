"""Offline integration example: python examples/relay_harness.py BUNDLE PROMPT.

Use a trusted bundle whose model identities match your harness. The example
only returns a mock response; replace generate() with your own provider adapter
after evaluating the bundle on representative, independent tasks.
"""
from __future__ import annotations

import argparse
import json

from relay_router import load_bundle


def generate(model: str, prompt: str) -> dict:
    """The harness owns execution, credentials, capabilities, retries and cost."""
    return {"mode": "offline_example", "model": model, "prompt": prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Trusted manifest JSON or bundle directory")
    parser.add_argument("prompt", help="Complete serialized input to route and generate")
    args = parser.parse_args()
    router = load_bundle(args.bundle)
    print(json.dumps(router.run(args.prompt, generate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
