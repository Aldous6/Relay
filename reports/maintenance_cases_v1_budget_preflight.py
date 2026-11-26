"""Offline preflight for the frozen authored development cohort.

No credentials are read. Only GroqMaintenanceProvider.preflight is called;
transport is explicitly forbidden. The model inspection plan is a placeholder.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.maintenance import SYSTEM, _diagnostics, _json, _response_schema
from relay_router.maintenance_competition import load_suite
from relay_router.maintenance_provider import GroqMaintenanceProvider, MaintenanceProviderError
from relay_router.maintenance_workspace import Workspace


def forbid_transport(*args, **kwargs):
    raise AssertionError("Network transport forbidden in this offline preflight")


def main():
    suite_path = ROOT / "configs/maintenance_cases_v1.json"
    suite = load_suite(suite_path)
    provider = GroqMaintenanceProvider("offline-placeholder", max_output_tokens=4096,
                                      request_budget=8000, transport=forbid_transport)
    rows = []
    for task in suite["tasks"]:
        config = task["config"]
        source_files, test_files = config["source_files"], config["test_files"]
        with TemporaryDirectory(prefix="relay-budget-preflight-") as temporary:
            work = Workspace(Path(task["repository"]), Path(temporary) / "run", source_files, test_files)
            work.prepare()
            baseline = work.verify(Path(sys.executable), config["limits"]["test_timeout_seconds"])
            if baseline["status"] != "test_failed" or baseline["tests_run"] != 3:
                raise AssertionError("Unexpected baseline")
            diagnostics = _diagnostics(baseline, work.worktree)
            def files(paths):
                return [{"path": name, "content": work.read_file(name)} for name in paths]
            plan = {"summary": "Corregir la implementación siguiendo el contrato de la tarea y preservar la interfaz pública.",
                    "files": list(source_files)}
            payloads = {
                "inspection": {"task": task["task"], "editable_files": files(source_files),
                    "editable_paths": list(source_files), "read_only_tests": files(test_files),
                    "baseline": diagnostics, "response_schema": _response_schema(source_files, inspection=True),
                    "instruction": "Inspect the task and prioritize editable files needed. Return JSON only. "
                        "Copy file paths exactly from editable_paths; tests are read-only references."},
                "editing": {"task": task["task"], "plan": plan, "editable_paths": list(source_files),
                    "editable_files": files(source_files), "read_only_tests": files(test_files),
                    "verification": diagnostics, "response_schema": _response_schema(source_files),
                    "instruction": "Return JSON. Make minimal exact replacements. old_text must match once. "
                        "Do not modify tests. Fix the requested behavior, not just the visible examples."},
            }
            estimates = {}
            for activity, payload in payloads.items():
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _json(payload)}]
                try:
                    estimate = provider.preflight(config["models"][activity], messages,
                                                  response_schema=payload["response_schema"])
                    accepted = True
                except MaintenanceProviderError as exc:
                    estimate, accepted = exc.input_estimate, False
                estimates[activity] = {"accepted": accepted, **estimate}
            rows.append({"task_id": task["task_id"], "estimates": estimates,
                         "input_sum_inspection_plus_editing": sum(e["estimated_input_tokens"] for e in estimates.values()),
                         "maximum_reserved_total": max(e["reserved_total"] for e in estimates.values()),
                         "minimum_remaining_budget": min(8000 - e["reserved_total"] for e in estimates.values())})
    result = {"schema_version": 1, "at_utc": datetime.now(timezone.utc).isoformat(),
              "suite_sha256": suite["suite_sha256"], "maintenance_source_sha256": hashlib.sha256(
                  (ROOT / "src/relay_router/maintenance.py").read_bytes()).hexdigest(),
              "python_executable": sys.executable, "output_reservation": 4096, "request_budget": 8000,
              "actual_api_calls": provider.request_count, "network_used": False,
              "inspection_plan": plan, "all_accepted": all(e["accepted"] for row in rows for e in row["estimates"].values()),
              "maximum_reserved_total": max(row["maximum_reserved_total"] for row in rows),
              "limitations": ["Editing uses a short fixed inspection plan, not a future model output.",
                              "Repair inputs depend on future patches and diagnostics and are not bounded here.",
                              "Provider token accounting is not independently verified; this is a local estimate.",
                              "Output reservation does not guarantee generation will finish without truncation."],
              "tasks": rows}
    destination = ROOT / "reports/maintenance-cases-v1-budget-preflight.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        inspection, editing = (row["estimates"][key] for key in ("inspection", "editing"))
        print(f"{row['task_id']}: inspection={inspection['estimated_input_tokens']} "
              f"editing={editing['estimated_input_tokens']} sum={row['input_sum_inspection_plus_editing']} "
              f"max_reserved={row['maximum_reserved_total']} remaining={row['minimum_remaining_budget']} "
              f"method={editing['method']}")
    print(f"all_accepted={result['all_accepted']} actual_api_calls={provider.request_count}")
    print(destination)
    return 0 if result["all_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
