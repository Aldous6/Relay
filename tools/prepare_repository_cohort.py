"""Materialize the fixed Relay maintenance cohort from a hash-pinned local Parquet."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COHORT = "swe-bench-live-lite-repository-maintenance-v1"
REVISION = "babb27e45e072219195db7a744bbf0d3c1a36e56"
SOURCE = f"https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live/resolve/{REVISION}/data/lite-00000-of-00001.parquet"
SOURCE_SHA256 = "a6375fbe7a55cb9630b6c3ceee980d504b9dac9351bd35f5c62cb965ac396ebc"
# Scope selected from repository names only, before reading individual tickets.
# One ticket per project prevents a prolific repository from dominating the pilot.
REPOSITORIES = (
    "Delgan/loguru", "Kozea/WeasyPrint", "PyPSA/PyPSA", "arviz-devs/arviz",
    "beetbox/beets", "fonttools/fonttools", "geopandas/geopandas", "ipython/ipython",
    "jazzband/tablib", "joke2k/faker", "matplotlib/matplotlib", "mikedh/trimesh",
    "projectmesa/mesa", "pvlib/pvlib-python", "pydata/xarray", "python-babel/babel",
    "python-control/python-control", "shapely/shapely", "sphinx-doc/sphinx", "wireservice/csvkit",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def prepare(source, destination):
    import pyarrow
    import pyarrow.parquet as pq

    if sha(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("Source bytes differ from the pinned SWE-bench-Live Lite revision")
    columns = ["repo", "instance_id", "base_commit", "problem_statement"]
    grouped = defaultdict(list)
    for row in pq.read_table(source, columns=columns).to_pylist():
        if row["repo"] in REPOSITORIES:
            grouped[row["repo"]].append(row)
    tasks, inventory = [], {}
    for repo in REPOSITORIES:
        ordered = sorted(grouped[repo], key=lambda r: sha((COHORT + "/" + r["instance_id"]).encode()))
        if not ordered:
            raise ValueError(f"No upstream task in the pinned partition for {repo}")
        inventory[repo] = [r["instance_id"] for r in ordered]
        first = ordered[0]
        if not first["problem_statement"].strip() or not re.fullmatch("[0-9a-f]{40}", first["base_commit"]):
            raise ValueError("Incomplete public ticket or missing repository revision")
        tasks.append({"task_id": first["instance_id"], "group_id": repo, "source": "https://github.com/" + repo,
                      "revision": first["base_commit"], "prompt": first["problem_statement"]})
    prompts = {sha(" ".join(t["prompt"].casefold().split()).encode()) for t in tasks}
    if len(prompts) != len(tasks) or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Repeated ticket or prompt in candidate cohort")
    cohort = serialized({"cohort_id": COHORT, "tasks": tasks})
    manifest = {"cohort_id": COHORT, "dataset_revision": REVISION, "source": SOURCE,
                "source_sha256": SOURCE_SHA256, "columns_read": columns, "partition": "lite",
                "selection": "first SHA256(cohort_id + '/' + instance_id) per prespecified repository",
                "ordered_candidate_ids": inventory, "cohort_sha256": sha(cohort),
                "preparation_source_sha256": sha(Path(__file__).read_bytes()), "pyarrow": pyarrow.__version__,
                "task_count": len(tasks), "model_outcomes_used": False}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "cohort.json").write_bytes(cohort)
    (destination / "manifest.json").write_bytes(serialized(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/swe-bench-live-lite-a6375fbe.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/imported/relay-repository-v1")
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(f"{manifest['task_count']} tasks, SHA-256 {manifest['cohort_sha256']}")
