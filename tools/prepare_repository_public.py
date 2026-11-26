"""Prepare pinned, source-only SWE-Live views using public issue text and Git trees.

Never imports project files or reads patches, evaluation fields, or model outputs.
All rankings use filenames before any source or test contents are downloaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from math import log
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/imported/relay-repository-v1/cohort.json"
IMAGES = COHORT.with_name("image-manifest.json")
OUTPUT = ROOT / "data/imported/relay-repository-public-v1"
CACHE = ROOT / ".cache/relay-public-git-v1"
CHECKER = ROOT / "tools/repository_public_checks.py"
MAX_BLOB = 300_000
MAX_SOURCE, MAX_TESTS = 6, 2
SKIP = set("docs doc examples example benchmarks benchmark vendor vendored third_party external build dist fixtures data scripts tools".split())
STOP = set("and are with for from the has this that there have how but into python using should when does not what which your none true false issue bug test tests error traceback example function src init lib packages".split())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def get(url, limit):
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Relay-public-source-preparation-v1"})
            with urlopen(req, timeout=25) as response:
                raw = response.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("Public metadata or Git blob exceeded its byte limit")
            return raw
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 2 or isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            time.sleep(1 + attempt)


def checkout_tree(repo, rev):
    # The Git Trees endpoint accepts the immutable commit as ref and returns
    # the root tree SHA. It has no patch/PR/issue contents.
    url = f"https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1"
    cache = CACHE / ("tree-" + sha(url.encode()) + ".json")
    if cache.exists():
        raw = cache.read_bytes()
    else:
        raw = get(url, 9_000_000)
        cache.write_bytes(raw)
    tree = json.loads(raw)
    if tree.get("truncated") is not False or not re.fullmatch("[0-9a-f]{40}", tree["sha"]):
        raise ValueError(f"Incomplete Git tree for {repo}@{rev}")
    return tree, url


def safe_file(entry):
    path = entry["path"]
    return (entry["type"] == "blob" and entry["mode"] in ("100644", "100755")
            and type(entry["size"]) is int and 0 < entry["size"] <= MAX_BLOB
            and re.fullmatch(r"[A-Za-z0-9_.\-/]+", path) and len(path) < 170 and
            all(part not in ("", ".", "..") and not part.startswith(".") and not part.endswith(".")
                and not PureWindowsPath(part).is_reserved() for part in path.split("/")))


def tokens(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {token.removesuffix("s") if token.endswith("s") and len(token) > 4 else token
            for token in re.findall(r"[a-z][a-z0-9]+", text.lower()) if token not in STOP}


def role(path):
    p = PurePosixPath(path)
    lower = p.stem.lower()
    if lower in ("setup", "conftest", "__main__") or any(part.lower() in SKIP for part in p.parts[:-1]):
        return None
    if lower.startswith("test_") or lower.endswith(("_test", "_tests", "test", "tests")):
        return "test" if lower != "__init__" else None
    if any(part.lower() in ("test", "tests", "testing") for part in p.parts[:-1]):
        return None
    return "source" if len(p.parts) > 1 else None


def select(tree, issue):
    files = [e for e in tree["tree"] if safe_file(e)]
    candidates = [(e, role(e["path"])) for e in files if e["path"].endswith(".py")]
    candidates = [(e, kind) for e, kind in candidates if kind]
    names = {e["path"]: tokens(e["path"][:-3]) for e, _ in candidates}
    df = Counter(token for group in names.values() for token in group)
    headline = issue.splitlines()[0]
    title, query = tokens(headline), tokens(issue)
    # Splits and compounds both occur in Python names, e.g. snake_case and
    # a filename that concatenates the consecutive words in an issue title.
    title_words = re.findall("[a-z0-9]+", headline.lower())
    title_pairs = {a + b for a, b in zip(title_words, title_words[1:])}
    issue_lower = issue.lower()

    def score(entry):
        path = entry["path"]
        stem = PurePosixPath(path).stem.lower()
        words = tokens(PurePosixPath(path).stem)
        value = 0.
        for word in sorted(names[path] & query):
            value += (1 + log((1 + len(names)) / (1 + df[word]))) * (4 if word in words else 1) * (3 if word in title else 1)
        for word in sorted(words - title):
            if word in title_pairs:
                value += 36
            elif any(len(key) >= 4 and len(word) >= 4 and (word.startswith(key) or key.startswith(word))
                     for key in title):
                value += 12
        if len(stem) >= 4 and stem.strip("_") and re.search(r"(?<!\w)" + re.escape(stem) + r"(?!\w)", issue_lower):
            value += 18
        if path.lower() in issue_lower:
            value += 90
        if stem == "__init__":
            value -= 8
        return round(value, 8)

    chosen = []
    for kind, count in (("source", MAX_SOURCE), ("test", MAX_TESTS)):
        available = [(score(e), e) for e, k in candidates if k == kind]
        available.sort(key=lambda pair: (-pair[0], pair[1]["path"].count("/"), pair[1]["path"].casefold(), pair[1]["path"]))
        chosen.extend((kind, e, value) for value, e in available[:count])
    license_paths = [e for e in files if len(PurePosixPath(e["path"]).parts) <= 2 and
                     (re.fullmatch(r"(license|licence|copying|notice)([.-].*)?", PurePosixPath(e["path"]).name, re.I)
                      and ("/" not in e["path"] or e["path"].split("/")[0].lower() in ("licenses", "license")))]
    chosen.extend(("license", e, None) for e in sorted(license_paths, key=lambda e: e["path"]))
    if not any(kind == "source" for kind, _, _ in chosen) or not any(kind == "test" for kind, _, _ in chosen):
        raise ValueError("A public tree lacks eligible source files or tests")
    if not license_paths:
        raise ValueError("No upstream license found for local redistribution")
    return chosen


def fetch_blob(repo, revision, entry):
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{quote(entry['path'])}"
    git_id = entry["sha"]
    if not re.fullmatch("[0-9a-f]{40}", git_id):
        raise ValueError("Invalid upstream Git blob ID")
    cache = CACHE / ("blob-" + git_id)
    raw = cache.read_bytes() if cache.exists() else get(url, entry["size"])
    git_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if git_hash != git_id or len(raw) != entry["size"]:
        raise ValueError(f"Git blob verification failed for {repo}/{entry['path']}")
    if not cache.exists():
        cache.write_bytes(raw)
    return raw, url


def prepare_task(task, image, folder):
    repo = task["group_id"]
    rev = task["revision"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch("[0-9a-f]{40}", rev):
        raise ValueError("Invalid pinned repository or revision")
    if task["source"] != "https://github.com/" + repo or image["repository_revision"] != rev or image["status"] != "registry_manifest_verified":
        raise ValueError("Image or source differs from the pinned cohort")
    tree, tree_url = checkout_tree(repo, rev)
    selected = select(tree, task["prompt"])
    folder.mkdir()
    snapshot = folder / "snapshot"
    snapshot.mkdir()
    items, editable, tests = [], [], []
    for kind, entry, score in selected:
        raw, url = fetch_blob(repo, rev, entry)
        path = entry["path"]
        if kind in ("source", "test"):
            raw.decode("utf-8")
            (editable if kind == "source" else tests).append(path)
        local = ("upstream-licenses/" if kind == "license" else "snapshot/") + path
        dest = folder / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        items.append({"kind": kind, "path": path, "local_path": local, "git_blob": entry["sha"],
                      "sha256": sha(raw), "size": len(raw), "url": url, "score": score})
    (snapshot / "relay_public_check.py").write_bytes(CHECKER.read_bytes())
    (snapshot / "relay_public_tests.json").write_bytes(encoded(tests))
    case = {k:task[k] for k in ("task_id", "source", "revision")}
    case.update(official_instance_id=task["task_id"], editable_paths=editable, image=image["image"], base_path="/testbed")
    (folder / "case.json").write_bytes(encoded(case))
    provenance = {"task_id": task["task_id"], "tree_url": tree_url, "root_tree_sha": tree["sha"],
                  "prompt_sha256": sha(task["prompt"].encode()), "files": items,
                  "case_sha256": sha(encoded(case)), "public_test_paths": tests,
                  "public_checker_sha256": sha(CHECKER.read_bytes()), "checks_executed": False}
    (folder / "provenance.json").write_bytes(encoded(provenance))
    return {"task_id": task["task_id"], "provenance_sha256": sha(encoded(provenance)),
            "source_files": len(editable), "test_files": len(tests), "license_files": len(items) - len(editable) - len(tests),
            "upstream_bytes": sum(i["size"] for i in items)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    dest = parser.parse_args().output.resolve()
    if dest.exists():
        raise FileExistsError("Choose a new output directory")
    cache_source = COHORT.read_bytes()
    cohort = json.loads(cache_source)
    images = json.loads(IMAGES.read_bytes())
    if images["cohort_sha256"] != sha(cache_source):
        raise ValueError("Image manifest does not match cohort")
    image_map = {r["task_id"]: r for r in images["tasks"]}
    tasks = cohort["tasks"]
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)) or set(ids) != image_map.keys() or any(not re.fullmatch(r"[a-z0-9_-]+", k) for k in ids):
        raise ValueError("Image and task inventories differ or contain invalid IDs")
    CACHE.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A sibling created normally inherits the destination parent's ACL.
    root = dest.parent / (".relay-public-staging-" + uuid4().hex)
    root.mkdir()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda t: prepare_task(t, image_map[t["task_id"]], root / t["task_id"]), tasks))
        manifest = {"cohort_id": cohort["cohort_id"], "cohort_sha256": sha(cache_source),
                    "image_manifest_sha256": sha(IMAGES.read_bytes()), "preparer_sha256": sha(Path(__file__).read_bytes()),
                    "public_checker_sha256": sha(CHECKER.read_bytes()), "tasks": rows,
                    "ranking": "Public issue title and words against base-tree filenames, path document-frequency weighting, title prefix/compound matches, exact stem/path bonus; ties by path depth and spelling. No file content scored.",
                    "limits": {"max_library_files": MAX_SOURCE, "max_test_files": MAX_TESTS, "max_file_bytes": MAX_BLOB},
                    "access": ["public cohort fields", "pinned image metadata", "recursive tree metadata", "selected base-commit blobs and license texts"],
                    "model_or_private_outcomes_used": False, "containers_started": 0, "checks_executed": False,
                    "github_tree_documentation": "https://docs.github.com/en/rest/git/trees#get-a-tree"}
        (root / "manifest.json").write_bytes(encoded(manifest))
        try:
            root.rename(dest)
        except PermissionError:
            # Windows scanners can retain handles to a newly assembled tree.
            # Publish a fresh directory and commit the manifest last.
            dest.mkdir()
            try:
                for task_id in ids:
                    shutil.copytree(root / task_id, dest / task_id)
                (dest / "manifest.json").write_bytes(encoded(manifest))
            except BaseException:
                if dest.resolve() == root.parent / dest.name:
                    shutil.rmtree(dest)
                raise
    finally:
        if root.exists() and root.resolve().parent == dest.parent and root.name.startswith(".relay-public-staging-"):
            shutil.rmtree(root)
    print(f"{len(rows)} repository snapshots prepared at {dest}")


if __name__ == "__main__":
    main()
