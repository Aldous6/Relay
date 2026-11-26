"""Paired maintenance trajectories from one shared, observed initial state.

This measures the first editing decision only. Inspection is shared; either
editing model is followed by at most one 120B repair. It is acquisition code,
not a trained router, and passing visible tests is not independent correctness.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .maintenance import _json, _save, run_maintenance
from .maintenance_workspace import Workspace


WEAK = "groq/openai/gpt-oss-20b"
STRONG = "groq/openai/gpt-oss-120b"
MODELS = {"weak": WEAK, "strong": STRONG}


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _usage_total(calls):
    usages = [call.get("usage") for call in calls]
    complete = bool(usages) and all(type(usage) is dict and all(
        type(usage.get(key)) is int and usage[key] >= 0
        for key in ("input_tokens", "output_tokens")) for usage in usages)
    return {"usage_complete": complete, "total_tokens": sum(
        usage["input_tokens"] + usage["output_tokens"] for usage in usages) if complete else None}


class _BranchRouter:
    def __init__(self, branch, inspection_mode="model"):
        self.branch = branch
        self.inspection_mode = inspection_mode

    def route(self, activity, *, attempt=0):
        if activity not in {"inspection", "editing", "repair"}:
            raise ValueError("Actividad desconocida en la adquisición emparejada.")
        model = WEAK if activity == "inspection" else (
            MODELS[self.branch] if activity == "editing" else STRONG)
        return {"selected_model": model, "activity": activity, "attempt": attempt,
                "policy_mode": "paired_first_edit_reference_continuation",
                "reason": "Acción experimental fijada; reparación común con 120B.",
                "router_seconds": 0., "evaluation_scope": "paired_acquisition_not_learned"}

    def manifest(self):
        return {"policy": "paired_first_edit_reference_continuation", "branch": self.branch,
                "inspection_model": WEAK if self.inspection_mode == "model" else None, "editing_model": MODELS[self.branch],
                "inspection_mode": self.inspection_mode,
                "repair_model": STRONG, "max_repairs": 1, "learned": False}


class _SharedInspectionProvider:
    """Reuse only the exact inspection response, accounting for it in each arm."""
    kind = "paired_shared_inspection"

    def __init__(self, provider, directory, collection):
        self.provider, self.directory, self.collection = provider, directory, collection
        self.output_mode = getattr(provider, "output_mode", "strict")
        self.branch = None
        self.branch_index = 0
        self.inspection_request = None
        self.inspection_reply = None
        self.editing_request = None
        self.anchor_baseline_identity = None
        self.mismatch = False

    def manifest(self):
        return {"kind": self.kind, "provider": self.provider.manifest() if callable(
            getattr(self.provider, "manifest", None)) else {"kind": type(self.provider).__name__},
                "reuse_scope": "identical inspection only; charged logically to both trajectories"}

    def begin(self, branch):
        self.branch, self.branch_index = branch, 0

    def save(self):
        attempts = self.collection["acquisition"]["calls"]
        self.collection["acquisition"].update(
            calls_started=len(attempts),
            calls_completed=sum(call["state"] == "completed" for call in attempts),
            **_usage_total(attempts))
        _save(self.directory / "collection.json", self.collection)

    def complete(self, model, messages):
        return self._dispatch(model, messages, None)

    def complete_structured(self, model, messages, response_schema):
        return self._dispatch(model, messages, response_schema)

    def _baseline_identity(self):
        report = json.loads((self.directory / self.branch / "report.json").read_text(encoding="utf-8"))
        baseline = report["verifications"][0]
        return {key: baseline.get(key) for key in (
            "status", "passed", "tests_run", "returncode", "timed_out", "result")}

    def _mismatch(self, reason):
        self.mismatch = True
        self.collection["pairing_error"] = reason
        self.save()
        raise ValueError("Los estados iniciales no coinciden; se detuvo el par antes de enviar otra petición.")

    def _dispatch(self, model, messages, schema):
        direct = self.collection["continuation"].get("inspection_mode") == "direct"
        index = self.branch_index + int(direct)
        activity = ("inspection", "editing", "repair")[index] if index < 3 else None
        if activity is None:
            raise ValueError("La trayectoria excede inspección, edición y una reparación.")
        expected_model = WEAK if index == 0 else (MODELS[self.branch] if index == 1 else STRONG)
        if model != expected_model:
            raise ValueError("La trayectoria cambió el catálogo o la continuación fijada.")
        snapshot = {"messages": deepcopy(messages), "response_schema": deepcopy(schema)}
        digest = _hash(snapshot)
        if self.branch_index == 0:
            identity = self._baseline_identity()
            if self.anchor_baseline_identity is None:
                self.anchor_baseline_identity = identity
            elif identity != self.anchor_baseline_identity:
                self._mismatch("baseline_result_changed")
            if direct and self.collection["initial_diagnostics"] is None:
                self.collection["initial_diagnostics"] = deepcopy(json.loads(messages[-1]["content"])["verification"])
        if index == 0:
            if self.inspection_request is not None:
                if snapshot != self.inspection_request or self.inspection_reply is None:
                    self._mismatch("inspection_request_changed_or_incomplete")
                self.branch_index += 1
                self.collection["acquisition"]["inspection_reuses"] += 1
                self.collection["acquisition"]["reused_inspections"].append(
                    {"branch": self.branch, "request_sha256": digest,
                     "physical_call_index": 0, "logical_usage": deepcopy(self.inspection_reply.get("usage"))})
                self.save()
                return deepcopy(self.inspection_reply)
            self.inspection_request = snapshot
            payload = json.loads(messages[-1]["content"])
            self.collection["initial_diagnostics"] = deepcopy(payload["baseline"])
        elif index == 1:
            if self.editing_request is None:
                self.editing_request = snapshot
                self.collection["routing_text"] = messages[-1]["content"]
                self.collection["routing_text_sha256"] = hashlib.sha256(
                    messages[-1]["content"].encode("utf-8")).hexdigest()
                self.collection["editing_request_sha256"] = digest
            elif snapshot != self.editing_request:
                self._mismatch("first_edit_request_changed")
        self.branch_index += 1
        calls = self.collection["acquisition"]["calls"]
        record = {"index": len(calls), "branch": self.branch, "activity": activity,
                  "model": model, "state": "pending", "usage": None,
                  "request_sha256": digest}
        request_path = self.directory / "requests" / f"call-{record['index']:03d}.json"
        request_path.parent.mkdir(exist_ok=True)
        _save(request_path, {"model": model, **snapshot})
        record["request_path"] = str(request_path.relative_to(self.directory))
        calls.append(record)
        self.save()  # An interrupted/uncertain attempt is never silently resent.
        try:
            if schema is not None and callable(getattr(self.provider, "complete_structured", None)):
                reply = self.provider.complete_structured(model, messages, schema)
            else:
                reply = self.provider.complete(model, messages)
        except BaseException as exc:
            record.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                          error_type=type(exc).__name__, category=getattr(exc, "category", None),
                          usage=deepcopy(getattr(exc, "usage", None)))
            self.save()
            raise
        record.update(state="completed", **{key: deepcopy(reply.get(key)) for key in (
            "usage", "request_seconds", "quota_wait_seconds", "returned_model", "finish_reason")})
        if index == 0:
            self.inspection_reply = deepcopy(reply)
        self.save()
        return reply


def _known_output_failure(call):
    """A completed generation that exhausted/output-violated the fixed contract.

Transport failures remain unknown. Reported usage is required; there is no
automatic resend and a failed first edit is an outcome of this fixed harness.
"""
    usage = call.get("usage")
    if (call.get("state") == "error" and call.get("category") == "model_output_json"
            and call.get("provider_error_code") == "json_validate_failed"):
        return True  # Known rejected generation; its unreported cost stays unknown.
    return (call.get("state") == "error" and call.get("category") in
            {"output_truncated", "model_output", "model_output_json"} and type(usage) is dict
            and all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("input_tokens", "output_tokens")))


def _semantic_outcome(report):
    """Separate observed model failures from missing operational outcomes."""
    if report["status"] == "ready_for_review":
        return True
    if report["status"] == "tests_failed":
        return False
    error = report.get("error", {})
    if (report["status"] == "blocked" and error.get("stage") in {"editing", "repair"}
            and report["calls"] and _known_output_failure(report["calls"][-1])):
        return False
    if (report["status"] == "blocked" and error.get("stage") in {"editing", "repair"}
            and error.get("type") in {"ValueError", "JSONDecodeError"}
            and report["calls"] and report["calls"][-1]["state"] == "completed"
            and all(call["state"] == "completed" or _known_output_failure(call)
                    for call in report["calls"])):
        # The known model output could not be parsed or applied. It is a failed
        # action, while a missing test verdict or filesystem failure is unknown.
        return False
    return None


def _branch_summary(report, directory, branch, reused):
    path = directory / branch / "report.json"
    outcome = _semantic_outcome(report)
    return {"status": report["status"], "success": outcome, "outcome_known": outcome is not None,
            "report_path": str(path.relative_to(directory)),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "total_tokens": report.get("total_tokens"), "usage_complete": report.get("usage_complete", False),
            "inspection_reused": reused,
            "calls": [{"model": call["decision"]["selected_model"],
                       **{key: deepcopy(call.get(key)) for key in (
                           "activity", "state", "usage", "request_seconds", "quota_wait_seconds")}}
                      for call in report["calls"]]}


def collect_pair(*, root, source_files, test_files, task, run_dir, provider,
                 python_executable=None, timeout_seconds=30, progress=None,
                 branch_order=("weak", "strong"), inspection_mode="model"):
    """Collect two actual trajectories; no original changes or automatic resume.

Each logical trajectory includes the shared inspection usage. ``acquisition``
contains physical provider dispatches separately, so acquisition cost is never
confused with the cost of deploying one selected trajectory. The test result
is visible-test success, not an independent quality label.
"""
    if type(branch_order) not in (tuple, list) or tuple(branch_order) not in (
            ("weak", "strong"), ("strong", "weak")):
        raise ValueError("branch_order debe contener weak y strong una vez cada uno.")
    if inspection_mode not in {"model", "direct"}:
        raise ValueError("inspection_mode debe ser model o direct.")
    if not callable(getattr(provider, "complete", None)):
        raise TypeError("Se requiere un proveedor de mantenimiento explícito.")
    if type(task) is not str or not task.strip() or len(task) > 16000:
        raise ValueError("Se requiere una petición de mantenimiento no vacía.")
    root, directory = Path(root).resolve(), Path(run_dir).resolve()
    if directory.exists():
        raise ValueError("La colección requiere una carpeta nueva; no reanuda llamadas inciertas.")
    workspace = Workspace(root, directory / "validation-unused", list(source_files), list(test_files))
    inputs = workspace.validate_inputs()
    fingerprints = {row["path"]: row["sha256"] for row in inputs["files"]}
    directory.mkdir(parents=True, exist_ok=False)
    collection = {"schema_version": 1, "kind": "relay_maintenance_pair", "status": "running",
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "task": task,
                  "repository": str(root), "source_files": list(source_files), "test_files": list(test_files),
                  "input_files_sha256": fingerprints, "branch_order": list(branch_order),
                  "target": "visible_test_success_under_reference_continuation",
                  "continuation": {"inspection_model": WEAK, "repair_model": STRONG, "max_repairs": 1},
                  "scope": "first editing decision from shared observed inspection and baseline",
                  "baseline_policy": "reuse first observed public diagnostics; verify both actual baselines separately",
                  "verification_scope": "declared visible tests; not independent correctness validation",
                  "independent_quality_validation": False, "original_modified": False,
                  "routing_text": None, "routing_text_sha256": None, "initial_diagnostics": None,
                  "branches": {}, "acquisition": {"calls": [], "calls_started": 0, "calls_completed": 0,
                      "inspection_reuses": 0, "reused_inspections": [],
                      "usage_complete": False, "total_tokens": None}, "automatic_resume": False}
    bridge = _SharedInspectionProvider(provider, directory, collection)
    if inspection_mode == "direct":
        collection["continuation"].update(inspection_model=None, inspection_mode="direct")
        collection["scope"] = "first editing decision from shared baseline; deterministic plan; no inspection model call"
    bridge.save()
    try:
        for ordinal, branch in enumerate(branch_order):
            current = workspace.validate_inputs()
            if {row["path"]: row["sha256"] for row in current["files"]} != fingerprints:
                collection["pairing_error"] = "original_input_changed"
                collection["original_modified"] = True
                break
            collection["active_branch"] = branch
            collection["branches"][branch] = {"status": "started", "success": None, "outcome_known": False}
            bridge.begin(branch)
            bridge.save()
            if progress:
                progress(f"Trayectoria {branch}: primera edición {MODELS[branch]}; reparación común 120B.")
            report = run_maintenance(root=root, source_files=source_files, test_files=test_files,
                task=task, run_dir=directory / branch, provider=bridge, router=_BranchRouter(branch, inspection_mode),
                python_executable=python_executable, timeout_seconds=timeout_seconds, max_repairs=1,
                progress=progress, initial_diagnostics=deepcopy(collection["initial_diagnostics"]) if ordinal else None,
                inspection_mode=inspection_mode)
            collection["branches"][branch] = _branch_summary(report, directory, branch, bool(ordinal))
            bridge.save()
            # Quota/transport uncertainty is not evidence of model quality, and
            # no second inspection should be attempted after an invalid first.
            if (bridge.mismatch or _semantic_outcome(report) is None or
                    not any(call["activity"] == "editing" for call in report["calls"]) or
                    any(call["state"] != "completed" and not _known_output_failure(call) for call in report["calls"])):
                collection["stop_reason"] = "pairing_or_operational_failure"
                break
        current = workspace.validate_inputs()
        collection["original_modified"] = collection["original_modified"] or (
            {row["path"]: row["sha256"] for row in current["files"]} != fingerprints)
        complete = (len(collection["branches"]) == 2 and not collection.get("stop_reason")
                    and not collection.get("pairing_error") and not collection["original_modified"])
        collection["status"] = "completed" if complete else "incomplete"
        collection["paired_state_verified"] = bool(complete and bridge.editing_request is not None)
    except KeyboardInterrupt:
        collection.update(status="interrupted", stop_reason="interrupted_no_automatic_resume")
    except Exception as exc:
        collection.update(status="incomplete", stop_reason="collection_error",
                          error={"type": type(exc).__name__})
    finally:
        collection["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        bridge.save()
    return collection
