"""Episode-level evaluation with independent environments and durable decisions.

This module imports no model provider and executes no generated code. Callers
supply a generation callback and an explicit manifest; the bundled CLI only
uses a scripted simulation. Hidden references never enter either callback.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import time
import zipfile

from relay_router.agent import AgentRequest, AgentState, AgentStateV8, ModelReply, parse_action
from relay_router.errors import BackendActionError, LiveBackendError
from .agent_context import compact_progress
from .agent_tasks import create_environment, validate_task_splits
from .agent_metrics import annotate_costs, summarize, DEFAULT_ROI


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers/antivirus may briefly deny replacement. Retry only the
    # already-fsynced file commit, never a generation or environment action.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            time.sleep(.01 * 2**attempt)


def _seal(value):
    return {"payload": value, "payload_sha256": object_hash(value)}


def _unseal(path):
    saved = _read(path)
    if (not isinstance(saved, dict) or set(saved) != {"payload", "payload_sha256"}
            or object_hash(saved["payload"]) != saved["payload_sha256"]):
        raise ValueError(f"Checkpoint alterado: {path}")
    return saved["payload"]


def _fingerprint(callback):
    target = callback if inspect.isfunction(callback) else type(callback)
    try:
        source = inspect.getsourcefile(target)
    except TypeError:
        source = None
    return {"module": target.__module__, "name": target.__qualname__,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
            if source and Path(source).is_file() else None}


def _sources():
    root = Path(__file__).resolve().parent.parent
    paths = [p for package in ("routerbench", "relay_router") for p in (root / package).rglob("*.py")]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def _source_bundle(config):
    root = Path(__file__).resolve().parent.parent
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _sources():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (root / name).read_bytes())
        archive.writestr(zipfile.ZipInfo("configuration.json", date_time=(1980, 1, 1, 0, 0, 0)),
                         json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return buffer.getvalue()


@contextmanager
def _single_writer(output):
    """An OS lock is released on process death; stale file content is harmless."""
    with (Path(output) / ".writer.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Otra ejecución mantiene esta salida bloqueada.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate(config, policies):
    if config.get("schema_version") not in (1, 2) or type(config.get("schema_version")) is not int:
        raise ValueError("Se requiere schema_version=1 o 2.")
    if config.get("execution_mode") not in {"simulated", "external"}:
        raise ValueError("execution_mode debe declarar simulated o external.")
    if config.get("evaluation_split") not in {"development", "calibration", "evaluation"}:
        raise ValueError("Declarar la partición de episodios.")
    if config.get("independent_validation") is not False:
        raise ValueError("Esta infraestructura inicial no autoriza confirmación independiente.")
    if type(config.get("max_calls")) is not int or not 1 <= config["max_calls"] <= 128:
        raise ValueError("max_calls debe ser un entero entre 1 y 128.")
    if not isinstance(config.get("run_id"), str) or not config["run_id"].strip():
        raise ValueError("run_id es obligatorio.")
    if not isinstance(config.get("generation_manifest"), dict) or not config["generation_manifest"]:
        raise ValueError("Declarar el generador y sus versiones/parámetros.")
    annotate_costs([], config.get("pricing"), config.get("compute_rates"))
    roi = config.get("roi")
    if not isinstance(roi, dict) or set(roi) != set(DEFAULT_ROI):
        raise ValueError("Declarar los tres criterios ROI.")
    for name, value in roi.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Los criterios ROI deben ser números finitos no negativos.")
        if (name != "max_p95_router_seconds" and value > 1) or (name == "max_p95_router_seconds" and value <= 0):
            raise ValueError("Fracciones ROI fuera de rango o latencia no positiva.")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] <= 2**32-1:
        raise ValueError("seed debe ser un entero uint32.")
    if type(config.get("bootstrap_replicates")) is not int or config["bootstrap_replicates"] < 100:
        raise ValueError("bootstrap_replicates debe ser un entero de al menos 100.")
    pair = config.get("models")
    if (not isinstance(pair, dict) or set(pair) != {"strong", "weak"}
            or any(not isinstance(v, str) or not v.strip() for v in pair.values())
            or pair["strong"] == pair["weak"]):
        raise ValueError("Declarar dos identidades de modelos distintas.")
    if config["schema_version"] == 2:
        catalog = config.get("model_catalog")
        if (type(catalog) is not list or len(catalog) < 2 or
                any(type(model) is not str or not model.strip() or model != model.strip() for model in catalog)
                or len(set(catalog)) != len(catalog) or catalog[0] != pair["weak"] or catalog[-1] != pair["strong"]):
            raise ValueError("model_catalog requiere N identidades únicas ordenadas de barato a fuerte.")
        harness = config.get("harness")
        if (type(harness) is not dict or set(harness) != {"contract_version", "max_transport_retries", "retry_delay_seconds"}
                or type(harness["contract_version"]) is not int or harness["contract_version"] != 2
                or type(harness["max_transport_retries"]) is not int or not 0 <= harness["max_transport_retries"] <= 5
                or type(harness["retry_delay_seconds"]) not in (int, float)
                or not math.isfinite(harness["retry_delay_seconds"]) or not 0 <= harness["retry_delay_seconds"] <= 30):
            raise ValueError("Declarar contrato v2 y recuperación de infraestructura acotada.")
    if not policies or len({p.name for p in policies}) != len(policies):
        raise ValueError("Las políticas requieren nombres únicos.")
    if config.get("policy_order", "as_list") not in {"as_list", "cyclic_by_task"}:
        raise ValueError("policy_order debe ser as_list o cyclic_by_task.")
    if "always_strong" not in {p.name for p in policies}:
        raise ValueError("Se requiere el control always_strong.")
    for policy in policies:
        if getattr(policy, "strong", None) != pair["strong"] or getattr(policy, "weak", None) != pair["weak"]:
            raise ValueError("La política no coincide con las identidades congeladas.")
        if not callable(getattr(policy, "select", None)) or not callable(getattr(policy, "manifest", None)):
            raise ValueError("La política debe implementar select y manifest.")
        if config["schema_version"] == 2:
            declared = policy.manifest().get("model_catalog", config["model_catalog"])
            if list(declared) != config["model_catalog"]:
                raise ValueError("La política declara otro catálogo de candidatos.")
    object_hash(config)  # Reject non-JSON/non-finite settings before any call.


def prepare_episodes(tasks, policies, output_dir, config, *, generate=None,
                     environment_factory=create_environment):
    """Freeze inputs without generation. Existing directories must match exactly."""
    tasks, config, policies = deepcopy(tasks), deepcopy(config), list(policies)
    _validate(config, policies)
    split_audit = validate_task_splits({config["evaluation_split"]: tasks})
    if any(type(t["max_steps"]) is not int or not 1 <= t["max_steps"] <= 128 for t in tasks):
        raise ValueError("El límite de cada tarea debe estar entre 1 y 128.")
    # Exercise task/schema construction before any paid/external callback.
    for task in tasks:
        environment_factory(deepcopy(task))
    source_bundle = _source_bundle(config)
    body = {"schema_version": 1, "configuration": config, "tasks": tasks,
            "tasks_sha256": object_hash(tasks), "partition_audit": split_audit,
            "policy_sequence": [p.name for p in policies],
            "policies": {p.name: p.manifest() for p in policies},
            "policy_sources": {p.name: _fingerprint(p) for p in policies},
            "generation_source": _fingerprint(generate) if generate is not None else None,
            "environment_source": _fingerprint(environment_factory),
            "source_files_sha256": _sources(),
            "source_bundle_sha256": hashlib.sha256(source_bundle).hexdigest(),
            "runtime_versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                                for name in ("numpy", "pandas", "scikit-learn", "joblib")}}}
    frozen = {**body, "freeze_sha256": object_hash(body)}
    output = Path(output_dir)
    manifest = output / "manifest.json"
    if manifest.exists():
        if _read(manifest) != frozen:
            raise ValueError("El código, tareas, generador o políticas cambió desde la congelación.")
    else:
        if output.exists() and any(p.name != ".writer.lock" for p in output.iterdir()):
            raise FileExistsError("La salida debe ser nueva o conservar su manifiesto original.")
        output.mkdir(parents=True, exist_ok=True)
        _write(manifest, frozen)
    archive = output / "source_bundle.zip"
    if archive.exists():
        if hashlib.sha256(archive.read_bytes()).hexdigest() != frozen["source_bundle_sha256"]:
            raise ValueError("El archivo de fuentes congeladas cambió.")
    else:
        archive.write_bytes(source_bundle)
    return frozen


class _Journal:
    def __init__(self, output, freeze, allowed):
        self.path = Path(output) / "journal.json"
        self.freeze = freeze
        self.records = []
        if self.path.exists():
            saved = _unseal(self.path)
            if saved["freeze_sha256"] != freeze:
                raise ValueError("Journal de otra ejecución.")
            self.records = saved["records"]
        previous = freeze
        active, completed = {}, set()
        counts = {}
        for index, record in enumerate(self.records):
            body = {k: v for k, v in record.items() if k != "record_sha256"}
            if (record.get("sequence") != index or record.get("previous_sha256") != previous
                    or object_hash(body) != record.get("record_sha256")):
                raise ValueError("Cadena de eventos alterada.")
            previous = record["record_sha256"]
            key = (record["episode_id"], record["policy"])
            if key not in allowed or key in completed:
                raise ValueError("Evento ajeno a la evaluación o posterior a su cierre.")
            kind = record["kind"]
            if kind == "call_started":
                if key in active or record["data"]["call_index"] != counts.get(key, 0):
                    raise ValueError("Orden de llamadas inconsistente.")
                active[key] = record["data"]
            elif kind == "call_completed":
                if key not in active or record["data"]["call_index"] != active[key]["call_index"]:
                    raise ValueError("Resultado sin llamada iniciada.")
                if record["data"]["model"] != active[key]["model"]:
                    raise ValueError("Modelo cambiado después de decidir.")
                active.pop(key)
                counts[key] = counts.get(key, 0)+1
            elif kind == "episode_completed":
                if key in active and record["data"]["termination"] != "interrupted_not_retried":
                    raise ValueError("Un intento incierto no puede ocultarse.")
                if (record["data"]["episode_id"], record["data"]["policy"]) != key:
                    raise ValueError("Identidad de episodio inconsistente.")
                active.pop(key, None)
                completed.add(key)
            else:
                raise ValueError("Evento desconocido.")

    def append(self, kind, key, data):
        body = {"sequence": len(self.records), "previous_sha256":
                self.records[-1]["record_sha256"] if self.records else self.freeze,
                "episode_id": key[0], "policy": key[1], "kind": kind, "data": deepcopy(data)}
        self.records.append({**body, "record_sha256": object_hash(body)})
        _write(self.path, _seal({"freeze_sha256": self.freeze, "records": self.records}))

    def for_key(self, key):
        return [r for r in self.records if (r["episode_id"], r["policy"]) == key]


def _elapsed(start):
    return max(0., time.perf_counter()-start)


def _provenance(config):
    return {key: config[key] for key in ("execution_mode", "evaluation_split", "independent_validation")}


def _episode(task, policy, events, config, success, termination, elapsed):
    return {"episode_id": task["id"], "source_group": task["source_group"], "family": task["family"],
            "policy": policy.name, "success": bool(success), "termination": termination,
            "events": annotate_costs(events, config["pricing"], config["compute_rates"]),
            "end_to_end_seconds": elapsed, "provenance": _provenance(config)}


def _recover(task, policy, records, config):
    """Conservative recovery: never replay uncertain calls or tool effects."""
    if records[-1]["kind"] == "episode_completed":
        return records[-1]["data"]
    events = [r["data"] for r in records if r["kind"] == "call_completed"]
    if records[-1]["kind"] == "call_started":
        pending = deepcopy(records[-1]["data"])
        pending.update(generation_seconds=None, verification_seconds=None, tool_seconds=None,
                       usage={}, reported_llm_cost_usd=None, error_type="InterruptedGeneration",
                       action=None, final_success=None, response=None)
        events.append(pending)
    elif events and events[-1].get("final_success") is not None:
        # The final evaluator already completed durably. Preserve that known
        # outcome; total wall time remains unknown after an interrupted commit.
        return _episode(task, policy, events, config, events[-1]["final_success"], "final_answer", None)
    return _episode(task, policy, events, config, False, "interrupted_not_retried", None)


def _execute(task, policy, generate, config, journal, environment_factory, *, initial_request=None):
    key = (task["id"], policy.name)
    started = time.perf_counter()
    prefix_events = []
    if initial_request is None:
        environment = environment_factory(deepcopy(task))
    else:
        if config["schema_version"] != 2 or not isinstance(initial_request, AgentRequest):
            raise ValueError("Una rama requiere contrato v2 y AgentRequest público.")
        from .agent_rollouts import replay_prefix
        environment, prefix_events = replay_prefix(task, initial_request, environment_factory)
    tools = environment.tools()
    names = tuple(tool["name"] for tool in tools)
    observation = environment.observation()
    # Provenance is for grouping/audit, not routing or generation features.
    observation = {k: v for k, v in observation.items() if k not in {"task_id", "family", "source_group", "id"}}
    history = [{"role": "tool", "content": {"name": "environment",
                "observation": observation, "tools": tools}}]
    errors, events = [], []
    v8 = config["schema_version"] == 2
    catalog = config["model_catalog"] if v8 else list(config["models"].values())
    infrastructure_errors, failed_tiers = [], []
    infrastructure_attempt, pinned_model, retry_wait = 0, None, 0.
    step, attempt = 0, 0
    limit = min(config["max_calls"], task["max_steps"])
    if initial_request is not None:
        initial_request = initial_request.snapshot()
        initial = initial_request.state
        if not isinstance(initial, AgentStateV8) or not 1 <= initial.remaining_calls <= limit:
            raise ValueError("La rama no puede ampliar el presupuesto del estado de origen.")
        if initial.infrastructure_attempt:
            raise ValueError("Un retry de infraestructura fijado no es un ancla de decisión nueva.")
        history = deepcopy(list(initial_request.history))
        errors = list(initial.previous_errors)
        infrastructure_errors, failed_tiers = list(initial.infrastructure_errors), list(initial.failed_tiers)
        step, attempt, limit = initial.step, initial.attempt, initial.remaining_calls
    success, termination = False, "call_budget_exhausted"
    for call_index in range(limit):
        state = AgentState(step=step, attempt=attempt, previous_errors=tuple(errors),
                           available_tools=names, remaining_calls=limit-call_index)
        if v8:
            context_size = len(json.dumps({"prompt": task["prompt"], "history": history},
                               ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            state = AgentStateV8(step, attempt, tuple(errors), names, limit-call_index,
                                 tuple(infrastructure_errors), infrastructure_attempt, context_size,
                                 compact_progress(environment.observation(), prefix_events + events), tuple(failed_tiers))
        if initial_request is not None and call_index == 0:
            reconstructed = AgentRequest(task["prompt"], tuple(history), state)
            if reconstructed.serialize() != initial_request.serialize():
                raise ValueError("La reconstrucción cambió el estado público del ancla.")
        def request():
            return AgentRequest(prompt=task["prompt"], history=tuple(deepcopy(history)), state=state)
        route_start = time.perf_counter()
        model = pinned_model if v8 and pinned_model is not None else policy.select(request())
        router_seconds = _elapsed(route_start)
        if model not in catalog:
            raise ValueError("La política seleccionó un modelo fuera del catálogo congelado.")
        public_request = json.loads(request().serialize())
        event = {"call_index": call_index, "step": step, "attempt": attempt,
                 "is_retry": attempt > 0 or infrastructure_attempt > 0, "model": model, "router_seconds": router_seconds,
                 "request": public_request, "request_sha256": object_hash(public_request)}
        if v8:
            event.update(infrastructure_attempt=infrastructure_attempt, retry_wait_seconds=retry_wait,
                         routing_source="pinned_infrastructure_retry" if pinned_model else "policy")
        if pinned_model is None and getattr(policy, "last_audit", None) is not None:
            event["routing_audit"] = deepcopy(policy.last_audit)
            object_hash(event["routing_audit"])
        journal.append("call_started", key, event)  # durable BEFORE external call
        if config.get("console_progress") is True:
            print(f"[{task['family']}] Llamada {call_index+1}/{limit}; paso {step+1}.", flush=True)
        if retry_wait:
            time.sleep(retry_wait)
        generation_start = time.perf_counter()
        reply, generation_error, backend_details = None, None, None
        backend_action_failure = False
        failure_usage = {}
        try:
            reply = generate(model, request())
            if not isinstance(reply, ModelReply):
                raise TypeError("El generador debe devolver ModelReply.")
            reply = reply.snapshot()
            annotate_costs([{**event, "generation_seconds": 0., "verification_seconds": 0.,
                             "tool_seconds": 0., "usage": reply.usage,
                             "reported_llm_cost_usd": reply.cost_usd}], {}, {})
        except Exception as exc:
            generation_error = type(exc).__name__
            if v8:
                backend_action_failure = isinstance(exc, BackendActionError)
                backend_details = exc.details() if isinstance(exc, LiveBackendError) else {
                    "type": type(exc).__name__, "category": "generation_contract", "retryable": False,
                    "status_code": None, "retry_after_seconds": None}
                if isinstance(exc, LiveBackendError) and exc.usage:
                    try:
                        annotate_costs([{**event, "generation_seconds": 0., "verification_seconds": 0.,
                            "tool_seconds": 0., "usage": exc.usage, "reported_llm_cost_usd": None}], {}, {})
                        failure_usage = deepcopy(exc.usage)
                    except (ValueError, TypeError):
                        pass  # Malformed measurements remain unknown.
            reply = None  # Invalid provider contracts never poison the journal.
        generation_seconds = _elapsed(generation_start)
        event.update(generation_seconds=generation_seconds, verification_seconds=0., tool_seconds=0.,
                     usage=deepcopy(reply.usage) if reply is not None else failure_usage,
                     reported_llm_cost_usd=reply.cost_usd if reply is not None else None,
                     response=reply.content if reply is not None else None, action=None,
                     final_success=None, error_type=generation_error)
        if generation_error is None:
            history.append({"role": "assistant", "content": reply.content})
            tick = time.perf_counter()
            try:
                action = parse_action(reply.content, names)
                event["action"] = action
            except (ValueError, TypeError) as exc:
                event["error_type"] = "InvalidAction:" + type(exc).__name__
            event["verification_seconds"] += _elapsed(tick)
            if event["action"] is not None:
                if action["type"] == "final":
                    tick = time.perf_counter()
                    success = environment.finish(action["answer"])
                    if type(success) is not bool:
                        raise TypeError("El evaluador final debe devolver bool.")
                    event["verification_seconds"] += _elapsed(tick)
                    event["final_success"] = success
                    termination = "final_answer"
                else:
                    tick = time.perf_counter()
                    result = environment.execute(action["name"], deepcopy(action["arguments"]))
                    event["tool_seconds"] = _elapsed(tick)
                    if (not isinstance(result, dict) or set(result) != {"ok", "observation", "error_type"}
                            or type(result["ok"]) is not bool):
                        raise TypeError("Contrato inválido de herramienta.")
                    history.append({"role": "tool", "content": {"name": action["name"], **deepcopy(result)}})
                    event["tool_result"] = deepcopy(result)
                    if not result["ok"]:
                        event["error_type"] = "ToolError:" + str(result["error_type"])
                    elif callable(getattr(environment, "verification_error", None)):
                        tick = time.perf_counter()
                        validation_error = environment.verification_error(action["name"], deepcopy(result))
                        event["verification_seconds"] += _elapsed(tick)
                        if validation_error is not None:
                            event["error_type"] = "VerificationFailed:" + str(validation_error)
        if v8 and backend_details is not None:
            event["backend_failure"] = backend_details
        if v8 and backend_details is not None and not backend_action_failure:
            event["error_scope"] = "infrastructure"
            infrastructure_errors.append(event["error_type"])
            infrastructure_attempt += 1
            pinned_model = model
            requested_wait = backend_details["retry_after_seconds"]
            retry_wait = max(config["harness"]["retry_delay_seconds"], requested_wait or 0.)
            exhausted = infrastructure_attempt > config["harness"]["max_transport_retries"]
            if not backend_details["retryable"] or exhausted or retry_wait > 30:
                termination = "infrastructure_failure"
        elif event["error_type"] is not None:
            errors.append(event["error_type"])
            history.append({"role": "error", "content": {"error_type": event["error_type"]}})
            attempt += 1
            if v8:
                event["error_scope"] = "semantic"
                failed_tiers.append(catalog.index(model))
                infrastructure_attempt, pinned_model, retry_wait = 0, None, 0.
        else:
            step += 1
            attempt = 0
            if v8:
                event["error_scope"] = None
                infrastructure_attempt, pinned_model, retry_wait = 0, None, 0.
        events.append(event)
        journal.append("call_completed", key, event)
        if config.get("console_progress") is True:
            if event["error_type"]:
                failure = event.get("backend_failure", {})
                detail = failure.get("diagnostic", {})
                print(f"[{task['family']}] Error: {event['error_type']}; HTTP={failure.get('status_code')}; "
                      f"codigo={detail.get('provider_error_code', 'no disponible')}.", flush=True)
            else:
                action = event.get("action") or {}
                print(f"[{task['family']}] Respuesta recibida: {action.get('name', action.get('type', 'sin accion'))}.", flush=True)
        if event["final_success"] is not None:
            break  # The hidden final evaluator is NEVER an escalation signal.
        if termination == "infrastructure_failure":
            break
    if config.get("console_progress") is True:
        print(f"[{task['family']}] Tarea cerrada: {'correcta' if success else 'fallida'} ({termination}).", flush=True)
    return _episode(task, policy, events, config, success, termination, _elapsed(started))


def _run_episodes_locked(tasks, policies, generate, output_dir, config, *, environment_factory=create_environment):
    """Execute or resume explicit callbacks; never start providers on its own."""
    if not callable(generate):
        raise TypeError("Se requiere un callback de generación explícito.")
    tasks, config, policies = deepcopy(tasks), deepcopy(config), list(policies)
    frozen = prepare_episodes(tasks, policies, output_dir, config, generate=generate,
                              environment_factory=environment_factory)
    output = Path(output_dir)
    allowed = {(task["id"], p.name) for task in tasks for p in policies}
    journal = _Journal(output, frozen["freeze_sha256"], allowed)
    summary_path = output / "summary.json"
    if summary_path.exists():
        result = _unseal(summary_path)
        if (result["freeze_sha256"] != frozen["freeze_sha256"]
                or result["journal_sha256"] != object_hash(journal.records)
                or {tuple(k) for k in result["completed_pairs"]} != allowed
                or len([r for r in journal.records if r["kind"] == "episode_completed"]) != len(allowed)):
            raise ValueError("Resumen y journal no coinciden.")
        return result
    episodes = []
    for task_index, task in enumerate(tasks):
        offset = task_index % len(policies) if config.get("policy_order") == "cyclic_by_task" else 0
        for policy in policies[offset:] + policies[:offset]:
            key = (task["id"], policy.name)
            existing = journal.for_key(key)
            if existing:
                episode = _recover(task, policy, existing, config)
            else:
                episode = _execute(task, policy, generate, config, journal, environment_factory)
            if not existing or existing[-1]["kind"] != "episode_completed":
                journal.append("episode_completed", key, episode)
            episodes.append(episode)
    if (frozen["source_files_sha256"] != _sources() or _read(output / "manifest.json") != frozen
            or frozen["policies"] != {p.name: p.manifest() for p in policies}
            or frozen["generation_source"] != _fingerprint(generate)
            or frozen["environment_source"] != _fingerprint(environment_factory)):
        raise ValueError("Fuentes o manifiesto cambiaron durante la ejecución.")
    metrics = summarize(episodes, config["roi"], bootstrap_replicates=config["bootstrap_replicates"],
                        seed=config["seed"])
    result = {"schema_version": 1, "freeze_sha256": frozen["freeze_sha256"],
              "journal_sha256": object_hash(journal.records),
              "completed_pairs": [[e["episode_id"], e["policy"]] for e in episodes],
              "configuration": config, "metrics": metrics,
              "n_episodes": len(episodes), "n_tasks": len(tasks),
              "execution_mode": config["execution_mode"], "independent_validation": False,
              "quality_claim": "infrastructure fixture only" if config["execution_mode"] == "simulated"
              else "exploratory episode evaluation; no independent adoption claim"}
    if config["schema_version"] == 2 and "cheap_then_escalate" in {p.name for p in policies}:
        # Reuse the exact recorded episodes: comparison does not generate more
        # trajectories or turn the heuristic into a second adoption baseline.
        cascade = summarize(episodes, config["roi"], baseline_policy="cheap_then_escalate",
                            bootstrap_replicates=config["bootstrap_replicates"], seed=config["seed"])
        result["cascade_comparisons"] = {name: {key: value for key, value in comparison.items() if key != "roi"}
                                          for name, comparison in cascade["comparisons"].items()}
        result["cascade_comparison_scope"] = "incremental predictive benefit; adoption gates remain relative to always_strong"
    _write(summary_path, _seal(result))
    return result


def run_episodes(tasks, policies, generate, output_dir, config, *, environment_factory=create_environment):
    """One writer, explicit generation callback and independent episode states."""
    if not callable(generate):
        raise TypeError("Se requiere un callback de generación explícito.")
    tasks, config, policies = deepcopy(tasks), deepcopy(config), list(policies)
    _validate(config, policies)
    validate_task_splits({config["evaluation_split"]: tasks})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with _single_writer(output):
        return _run_episodes_locked(tasks, policies, generate, output, config,
                                    environment_factory=environment_factory)
