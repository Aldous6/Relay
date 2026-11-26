"""Project pinned private SWE-bench-Live grading inputs to local data files."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_repository_cohort import COHORT, REVISION, ROOT, SOURCE, SOURCE_SHA256

DEFAULT_COHORT = ROOT / "data/imported/relay-repository-v1/cohort.json"
DEFAULT_PRIVATE = ROOT / "data/imported/relay-repository-acceptance-v1"
COLUMNS = ("repo", "instance_id", "base_commit", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
           "test_cmds", "rebuild_cmds", "log_parser", "docker_image")
DATA = "official_cases.jsonl.gz"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def public_tasks(cohort_path):
    evidence = json.loads((cohort_path.parent / "manifest.json").read_text(encoding="utf-8"))
    raw = cohort_path.read_bytes()
    cohort = json.loads(raw)
    tasks = cohort["tasks"]
    if (evidence["cohort_sha256"] != sha(raw) or cohort["cohort_id"] != COHORT or evidence["cohort_id"] != COHORT
            or evidence["source_sha256"] != SOURCE_SHA256 or evidence["dataset_revision"] != REVISION
            or len(tasks) != evidence["task_count"] or len({t["task_id"] for t in tasks}) != len(tasks)):
        raise ValueError("Cohort differs from its pinned preparation manifest")
    return tasks, sha(raw)


def describe(row, task):
    if (set(row) != set(COLUMNS) or row["instance_id"] != task["task_id"] or row["base_commit"] != task["revision"]
            or row["repo"] != task["group_id"] or task["source"] != "https://github.com/" + row["repo"]):
        raise ValueError("Official grading record differs from its frozen repository case")
    if any(not isinstance(row[k], str) or not row[k].strip() for k in ("test_patch", "docker_image", "log_parser")):
        raise ValueError("Missing official patch, parser or image reference")
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS", "test_cmds", "rebuild_cmds"):
        value = row[k]
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            raise ValueError(f"Malformed grading field {k}")
    if not row["FAIL_TO_PASS"] or not row["test_cmds"]:
        raise ValueError("Missing official fail-to-pass cases or commands")
    return {"task_id": row["instance_id"], "repository": row["repo"], "base_commit": row["base_commit"],
            "row_sha256": sha(encoded(row)), "test_patch_sha256": sha(row["test_patch"].encode("utf-8")),
            "fail_to_pass_count": len(row["FAIL_TO_PASS"]), "pass_to_pass_count": len(row["PASS_TO_PASS"]),
            "docker_image": row["docker_image"]}


def check_destination(path):
    blocked = [ROOT / "data/raw"]
    if os.environ.get("RELAY_REPOSITORY_CASES"):
        blocked.append(Path(os.environ["RELAY_REPOSITORY_CASES"]))
    if any(path.resolve().is_relative_to(root.resolve()) for root in blocked):
        raise ValueError("Write private acceptance records outside raw data and operating case bundles")


def prepare(source, cohort_path, target):
    import pyarrow
    import pyarrow.parquet as pq

    tasks, cohort_sha = public_tasks(cohort_path)
    check_destination(target)
    if target.exists():
        raise FileExistsError("Choose a new private dossier directory or verify the existing one")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SOURCE_SHA256:
            raise ValueError("Unexpected Parquet bytes")
        stream.seek(0)
        rows = pq.read_table(stream, columns=list(COLUMNS),
                             filters=[("instance_id", "in", [t["task_id"] for t in tasks])]).to_pylist()
    index = {r["instance_id"]: r for r in rows}
    if len(rows) != len(index) or set(index) != {t["task_id"] for t in tasks}:
        raise ValueError("Private records do not cover the exact cohort")
    ordered = [index[t["task_id"]] for t in tasks]
    entries = [describe(row, task) for row, task in zip(ordered, tasks)]
    raw = b"".join(encoded(row) + b"\n" for row in ordered)
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=packed, mode="wb", mtime=0) as stream:
        stream.write(raw)
    manifest = {"kind": "official_swe_bench_live_private_inputs", "cohort_id": COHORT, "cohort_sha256": cohort_sha,
                "source": SOURCE, "source_sha256": SOURCE_SHA256, "dataset_revision": REVISION,
                "columns_read": list(COLUMNS), "pyarrow": pyarrow.__version__, "cases": entries,
                "jsonl_sha256": sha(raw), "gzip_sha256": sha(packed.getvalue()), "private_grading_executed": False,
                "materializer_sha256": sha(Path(__file__).read_bytes())}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        staging = Path(temp) / "complete"
        staging.mkdir()
        (staging / DATA).write_bytes(packed.getvalue())
        (staging / "manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
        verify(cohort_path, staging)
        staging.rename(target)
    return manifest


def verify(cohort_path, private, export=None):
    tasks, cohort_sha = public_tasks(cohort_path)
    manifest = json.loads((private / "manifest.json").read_bytes())
    compressed = (private / DATA).read_bytes()
    if (manifest["gzip_sha256"] != sha(compressed) or manifest["cohort_sha256"] != cohort_sha
            or manifest["source_sha256"] != SOURCE_SHA256 or manifest["dataset_revision"] != REVISION
            or manifest["columns_read"] != list(COLUMNS) or manifest["cohort_id"] != COHORT):
        raise ValueError("Private archive, source or cohort provenance mismatch")
    raw = gzip.decompress(compressed)
    rows = [json.loads(line) for line in raw.splitlines()]
    if sha(raw) != manifest["jsonl_sha256"] or len(rows) != len(tasks):
        raise ValueError("Private records changed or are incomplete")
    if [describe(row, task) for row, task in zip(rows, tasks)] != manifest["cases"]:
        raise ValueError("Private case hashes or task order changed")
    if export:
        check_destination(export)
        if export.suffix != ".jsonl":
            raise ValueError("Use a new .jsonl destination for the official harness")
        with export.open("xb") as stream:
            stream.write(raw)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "export"))
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/swe-bench-live-lite-a6375fbe.parquet")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    if (args.action == "export") != (args.jsonl is not None):
        parser.error("Only export requires --jsonl")
    result = (prepare(args.source, args.cohort, args.private) if args.action == "prepare"
              else verify(args.cohort, args.private, args.jsonl))
    print(json.dumps({"official_cases": len(result["cases"]), "jsonl_sha256": result["jsonl_sha256"],
                      "private_grading_executed": False}))
