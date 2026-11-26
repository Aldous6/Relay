"""Offline authored-case consistency checks; no model or provider access.

Copies only one candidate module and its declared tests to each temporary
directory. Reference modules never coexist with the flawed module there.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parents[1] / "configs/maintenance_cases_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_check(folder, module, test_names):
    # The environment does not copy credentials into test children.
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if key in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    with TemporaryDirectory(prefix="relay-authored-check-") as temporary:
        root = Path(temporary)
        (root / "module.py").write_bytes(module.read_bytes())
        for name in test_names:
            (root / f"{name}.py").write_bytes((folder / f"{name}.py").read_bytes())
        process = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", *test_names],
                                 cwd=root, env=env, capture_output=True, text=True, timeout=30)
    output = process.stdout + process.stderr
    match = re.search(r"Ran (\d+) tests? in", output)
    return {"exit_code": process.returncode, "tests": int(match.group(1)) if match else None,
            "output": output}


def verify_task(task):
    folder = ROOT / task["task_id"]
    reference = ROOT / "reference_solutions" / f"{task['task_id']}.py"
    baseline = run_check(folder, folder / "module.py", ["test_public"])
    corrected = run_check(folder, reference, ["test_public", "acceptance_test"])
    visible_bytes = len(task["task"].encode()) + sum((folder / name).stat().st_size
                   for name in ("module.py", "test_public.py"))
    result = {"task_id": task["task_id"], "baseline_public": baseline,
              "reference_public_and_acceptance": corrected,
              "visible_input_bytes": visible_bytes,
              "source_lines": len((folder / "module.py").read_text().splitlines()),
              "sha256": {name: sha(folder / name) for name in
                         ("module.py", "test_public.py", "acceptance_test.py", "config.json")},
              "reference_sha256": sha(reference)}
    result["passed"] = (baseline["exit_code"] != 0 and baseline["tests"] == 3
                        and "FAIL:" in baseline["output"] and "ERROR:" not in baseline["output"]
                        and corrected["exit_code"] == 0 and corrected["tests"] >= 7
                        and visible_bytes < 5000 and 30 <= result["source_lines"] <= 100)
    return result


def main():
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(verify_task, suite["tasks"]))
    report = {"schema_version": 1, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "suite_sha256": sha(SUITE), "provenance_sha256": sha(ROOT / "provenance.json"),
              "python_version": sys.version, "network_used": False,
              "passed": all(row["passed"] for row in rows), "tasks": rows}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['task_id']}: {'PASS' if row['passed'] else 'FAIL'}; "
              f"baseline={row['baseline_public']['exit_code']}; "
              f"reference={row['reference_public_and_acceptance']['exit_code']}; "
              f"tests={row['reference_public_and_acceptance']['tests']}; "
              f"visible_bytes={row['visible_input_bytes']}; lines={row['source_lines']}")
        if row["reference_public_and_acceptance"]["exit_code"]:
            print(row["reference_public_and_acceptance"]["output"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
