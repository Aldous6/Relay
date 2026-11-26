"""Resume a frozen pilot with a recorded, bounded Windows file-write repair.

The original runner and its frozen sources remain unchanged. Completed failed
episodes remain in the primary cohort. Only the same local atomic write may be
retried; this wrapper never retries an LLM request or replaces an observation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_adaptive_pilot as original

MAX_WRITE_ATTEMPTS = 5
BASE_DELAY_SECONDS = .05
AMENDMENT_SOURCES = (
    "tools/continue_adaptive_pilot.py", "src/relay_router/integrations/openhands_io.py",
)
REASON = "observed_Windows_PermissionError_during_local_atomic_state_write_before_any_HTTP"


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in AMENDMENT_SOURCES}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(observations):
    return [original._episode_id(row["task_id"], row["model"]) for row in observations]


def _write_contract():
    return {
        "mechanism": "retry_identical_OpenHands_atomic_local_write_on_PermissionError",
        "max_write_attempts": MAX_WRITE_ATTEMPTS, "base_delay_seconds": BASE_DELAY_SECONDS,
        "path_scope": "resolved_paths_inside_this_run_output_only",
        "model_scope": "all_remaining_candidate_branches_equally",
        "model_or_HTTP_retries_added": 0, "completed_episodes_rerun": 0,
        "policy_routes_thresholds_suite_and_verifiers_changed": False,
        "original_failed_episodes_preserved": True,
    }


def prepare_amendment(suite_path, policy_path, output):
    """Check the original freeze and persist the execution amendment first."""
    output = Path(output).resolve()
    if not (output / "frozen_routes.json").is_file() or not (output / "run_manifest.json").is_file():
        raise ValueError("This continuation requires an existing frozen acquisition")
    suite, policy, frozen = original.prepare_run(suite_path, policy_path, output)
    observations = original.read_observations(suite, output)
    events = original.reconcile_http(suite, output, observations)
    if not observations:
        raise ValueError("A continuation must preserve completed primary observations")
    contract = {
        "schema_version": 1, "kind": "bounded_filesystem_execution_amendment",
        "freeze_id": frozen["freeze_id"], "artifact_id": policy["artifact_id"],
        "frozen_routes_sha256": hashlib.sha256((output / "frozen_routes.json").read_bytes()).hexdigest(),
        "source_sha256": source_hashes(), "reason": REASON, "write_retry": _write_contract(),
    }
    path = output / "execution_amendment.json"
    if path.exists():
        amendment = _read(path)
        payload = dict(amendment)
        seal = payload.pop("amendment_id", None)
        if seal != original._digest(payload):
            raise ValueError("Execution amendment integrity mismatch")
        if any(amendment.get(key) != value for key, value in contract.items()):
            raise ValueError("Cannot resume with changed amendment sources, freeze or retry settings")
        snapshot = amendment["original_snapshot"]
        episodes, attempts = snapshot["completed_episodes"], snapshot["http_attempts"]
        if (type(episodes) is not int or not 0 < episodes <= len(observations)
                or type(attempts) is not int or not 0 <= attempts <= len(events)
                or snapshot["completed_episode_ids"] != _ids(observations[:episodes])
                or snapshot["observations_prefix_sha256"] != original._digest(observations[:episodes])
                or snapshot["http_prefix_sha256"] != original._digest(events[:attempts])):
            raise ValueError("Original acquisition observations or dispatches changed after amendment")
    else:
        failed = [row for row in observations if row.get("error_type")]
        pre_http = [row for row in failed if row.get("error_type") == "PermissionError" and row.get("http_attempts") == 0]
        if not pre_http:
            raise ValueError("This amendment requires a recorded pre-HTTP PermissionError")
        if len(observations) >= len(suite["tasks"]) * len(suite["models"]):
            raise ValueError("The primary acquisition is already complete; do not amend its execution retroactively")
        amendment = {
            **contract, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_snapshot": {
                "completed_episodes": len(observations), "http_attempts": len(events),
                "completed_episode_ids": _ids(observations), "error_episode_ids": _ids(failed),
                "pre_http_PermissionError_episode_ids": _ids(pre_http),
                "observations_prefix_sha256": original._digest(observations),
                "http_prefix_sha256": original._digest(events),
            },
            "accounting": "local IO retries are logged separately; all actual LLM dispatches retain original accounting",
            "scope": "execution amendment after observed host IO failure; not a new routing policy or a replacement primary result",
        }
        amendment["amendment_id"] = original._digest(amendment)
        original.write(path, amendment)
    return amendment


def retry_logger(output, amendment):
    """Append only allowed local IO metadata, never text, errors or credentials."""
    output = Path(output).resolve()
    path = output / "io_retry_events.jsonl"
    count = 0
    if path.exists():
        previous = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for index, row in enumerate(previous, 1):
            if row.get("event_id") != index or row.get("amendment_id") != amendment["amendment_id"]:
                raise ValueError("Existing IO retry log is not bound to this amendment")
        count = len(previous)

    def append(event):
        nonlocal count
        if type(event) is not dict:
            raise ValueError("IO retry metadata must be a dictionary")
        relative = event.get("path")
        if type(relative) is not str or not relative or len(relative) > 4096:
            raise ValueError("IO retry path must be a relative local path")
        candidate = Path(relative)
        if candidate.is_absolute() or not (output / candidate).resolve().is_relative_to(output):
            raise ValueError("IO retry path is outside the amended run")
        attempt, next_attempt, maximum = event.get("attempt"), event.get("next_attempt"), event.get("max_attempts")
        if (type(attempt) is not int or not 1 <= attempt < MAX_WRITE_ATTEMPTS
                or type(next_attempt) is not int or next_attempt != attempt + 1
                or type(maximum) is not int or maximum != MAX_WRITE_ATTEMPTS):
            raise ValueError("IO retry counters differ from the fixed amendment")
        for name in ("errno", "winerror"):
            if event.get(name) is not None and type(event[name]) is not int:
                raise ValueError("IO error metadata must be a numeric code or None")
        row = {
            "event_id": count + 1, "at_utc": datetime.now(timezone.utc).isoformat(),
            "amendment_id": amendment["amendment_id"], "kind": "local_atomic_write_retry",
            "path": candidate.as_posix(), "attempt": attempt, "next_attempt": next_attempt,
            "max_attempts": maximum, "errno": event.get("errno"), "winerror": event.get("winerror"),
            "added_HTTP_requests": 0,
        }
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        count += 1

    return append


def _retry_context(output, callback):
    from relay_router.integrations.openhands_io import retry_openhands_writes
    return retry_openhands_writes(output, max_attempts=MAX_WRITE_ATTEMPTS,
                                  base_delay=BASE_DELAY_SECONDS, on_retry=callback)


def main(argv=None):
    forwarded = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-pipe")
    parser.add_argument("--max-new-episodes", type=int, default=48)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(forwarded)
    if not 0 <= args.max_new_episodes <= 48:
        parser.error("--max-new-episodes must be within [0,48]")
    amendment = prepare_amendment(args.suite, args.policy, args.output)
    callback = retry_logger(args.output, amendment)
    if args.freeze_only or args.max_new_episodes == 0 or args.report_only:
        # These original modes perform no SDK writes or model dispatches.
        return original.main(forwarded)
    with _retry_context(args.output.resolve(), callback):
        return original.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
