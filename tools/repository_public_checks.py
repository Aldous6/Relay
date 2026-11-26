"""Fixed pytest driver for an isolated repository overlay. Exit 0/10/2."""
from __future__ import annotations

from collections import defaultdict
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _public_modules(values, root):
    if not isinstance(values, list) or not values:
        raise ValueError("No declared test modules")
    for rel in values:
        if (not isinstance(rel, str) or not rel or rel.startswith("-") or Path(rel).is_absolute()
                or ".." in Path(rel).parts or not (root / rel).is_file()
                or not (root / rel).resolve().is_relative_to(root)):
            raise ValueError("A test module could not be resolved within the overlay")
    return sorted(set(values))


def _worker(selection_path, evidence_path):
    import pytest

    root = Path.cwd().resolve()
    selection = json.loads(selection_path.read_bytes())
    wanted = set(selection["ids"])
    phases = defaultdict(dict)
    collected = []
    class Evidence:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            if wanted:
                keep = [i for i in items if i.nodeid in wanted]
                config.hook.pytest_deselected(items=[i for i in items if i.nodeid not in wanted])
                items[:] = keep

        def pytest_collection_finish(self, session):
            collected.extend(item.nodeid for item in session.items)

        def pytest_runtest_logreport(self, report):
            case = phases[report.nodeid]
            if report.when in case or hasattr(report, "wasxfail"):
                case["ambiguous"] = True
            case[report.when] = report.outcome

    # Isolated Python omits cwd from sys.path. Put this overlay ahead of editable installs.
    sys.path[:0] = [str(path) for path in (root / "src", root / "lib", root) if path.is_dir()]
    os.environ.pop("PYTEST_ADDOPTS", None)
    status = pytest.main(["-q", "-o", "addopts=", "--rootdir", str(root), "-p", "no:cacheprovider",
                          "--", *selection["modules"]], plugins=[Evidence()])
    loaded = set()
    for mod in tuple(sys.modules.values()):
        origin = getattr(mod, "__file__", None)
        if isinstance(origin, str):
            resolved = Path(origin).resolve()
            if resolved.is_relative_to(root):
                loaded.add(resolved.relative_to(root).as_posix())
    evidence = {"pytest_status": int(status), "collected": collected, "phases": dict(phases),
                "overlay_modules": sorted(loaded)}
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")


def _verdict(evidence, required, sources, changed):
    items = evidence["collected"]
    if (not items or len(set(items)) != len(items) or required and set(items) != set(required)
            or set(evidence["phases"]) != set(items) or evidence["pytest_status"] not in (0, 1)):
        return 2
    loaded = set(evidence["overlay_modules"])
    if not loaded.intersection(sources) or not set(changed).issubset(loaded):
        return 2
    failed = False
    executed = 0
    for phases in evidence["phases"].values():
        if not required and "skipped" in phases.values() and "failed" not in phases.values():
            continue
        if (set(phases) != {"setup", "call", "teardown"} or phases["setup"] != "passed"
                or phases["teardown"] != "passed" or phases["call"] not in ("passed", "failed")):
            return 2
        executed += 1
        failed |= phases["call"] == "failed"
    if not executed or evidence["pytest_status"] != int(failed):
        return 2
    return 10 if failed else 0


def check(private_input=None):
    root = Path.cwd().resolve()
    sources = json.loads(os.environ["RELAY_GRADER_SOURCE_FILES"])
    changed = json.loads(os.environ["RELAY_GRADER_CHANGED_FILES"])
    ids = []
    if private_input:
        row = json.loads(private_input.read_bytes())
        if row["log_parser"].strip().lower() != "pytest" or not row["FAIL_TO_PASS"]:
            return 2
        ids = [*row["FAIL_TO_PASS"], *row["PASS_TO_PASS"]]
        if (len(ids) != len(set(ids)) or any(not isinstance(i, str) or "::" not in i for i in ids)):
            return 2
        # Apply only the upstream test patch, in the throwaway overlay and only after selection.
        source_hashes = {p: hashlib.sha256((root / p).read_bytes()).digest() for p in sources}
        applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=row["test_patch"].encode("utf-8"),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if applied.returncode != 0 or any(hashlib.sha256((root / p).read_bytes()).digest() != value
                                          for p, value in source_hashes.items()):
            return 2
        modules = [i.partition("::")[0] for i in ids]
    else:
        modules = json.loads((root / "relay_public_tests.json").read_bytes())
    modules = _public_modules(modules, root)
    with tempfile.TemporaryDirectory() as temp:
        selection, evidence = (Path(temp) / name for name in ("selection.json", "evidence.json"))
        selection.write_text(json.dumps({"modules": modules, "ids": ids}), encoding="utf-8")
        run = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker",
                              str(selection), str(evidence)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, check=False)
        if run.returncode != 0 or not evidence.is_file():
            return 2
        return _verdict(json.loads(evidence.read_bytes()), ids, sources, changed)


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--worker"]:
            _worker(Path(sys.argv[2]), Path(sys.argv[3]))
        else:
            private = None
            if len(sys.argv) != 1:
                if (sys.argv[1:] != ["/relay/private.json"] or Path.cwd().as_posix() != "/tmp/work"
                        or not Path("/relay/input").is_dir()):
                    raise SystemExit(2)
                private = Path(sys.argv[1])
            raise SystemExit(check(private))
    except (ValueError, IndexError, KeyError, OSError, ImportError, TypeError):
        raise SystemExit(2)
