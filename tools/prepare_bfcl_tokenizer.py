"""Verify/prepare local pinned GPT-OSS tokenizer assets; --download opts into public asset retrieval.

No model API, credential, account, weights, dataset or benchmark execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.bfcl_tokens import (DEFAULT_ASSET_ROOT, DEFAULT_PROVENANCE,
                                    GPTOSSTokenEstimator, PROVENANCE_SHA256)


def prepare_assets(asset_root=DEFAULT_ASSET_ROOT, *, download=False, opener=None):
    """Validate provenance first, then optionally acquire only its pinned assets."""
    raw = DEFAULT_PROVENANCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROVENANCE_SHA256:
        raise ValueError("Tokenizer provenance SHA256 mismatch.")
    provenance = json.loads(raw)
    root = Path(asset_root).resolve()
    rows = []
    for item in provenance["assets"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Tokenizer asset is outside the declared local cache.")
        before = path.read_bytes() if path.exists() else None
        valid = before is not None and len(before) == item["bytes"] and hashlib.sha256(before).hexdigest() == item["sha256"]
        action = "verified" if valid else "missing" if before is None else "invalid"
        if download and not valid:
            # This is the only network branch. Pinned provenance restricts URLs;
            # enforce size and digest before committing any fetched file.
            open_url = opener if opener is not None else urlopen
            with open_url(item["url"], timeout=30) as response:
                content = response.read(item["bytes"] + 1)
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded tokenizer asset SHA256 mismatch: {item['path']}.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".verified-download")
            # Exclusive creation avoids following a pre-existing temporary link.
            with temporary.open("xb") as stream:
                stream.write(content)
            temporary.replace(path)
            action, valid = "downloaded_verified", True
        rows.append({"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
                     "status": action, "verified": valid})
    ready = all(row["verified"] for row in rows)
    result = {"mode": "download" if download else "prepare_verify", "asset_root": str(root),
              "provenance_sha256": PROVENANCE_SHA256, "files": rows, "ready": ready,
              "provider_calls": 0, "model_weights_downloaded": False,
              "scope": "Pinned public tokenizer/template assets only; no server template verification."}
    if ready:
        result["estimator"] = GPTOSSTokenEstimator(asset_root=root).manifest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_assets(args.asset_root, download=args.download)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return result


if __name__ == "__main__":
    main()
