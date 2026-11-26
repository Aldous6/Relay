"""Prepare by default; --download explicitly acquires pinned public BFCL files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.bfcl_data import DEFAULT_SNAPSHOT, download_snapshot, prepare_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if args.download:
        snapshot = download_snapshot(args.output)
        result = {"mode": "downloaded", "snapshot": snapshot["root"],
                  "manifest_sha256": snapshot["manifest_sha256"],
                  "files": len(snapshot["manifest"]["files"]),
                  "development_groups": len(snapshot["selection"]["development"]),
                  "reserved_groups": len(snapshot["selection"]["evaluation"]),
                  "partition_groups": {part: len(snapshot["selection"][part])
                                       for part in ("fit", "calibration", "development", "evaluation")},
                  "provider_calls": 0, "reserved_payloads_decoded": False}
    else:
        result = prepare_snapshot(args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return result


if __name__ == "__main__":
    main()
