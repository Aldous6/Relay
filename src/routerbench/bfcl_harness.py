"""Separate, durable native-function harness for pinned BFCL conversations.

Final text ends a turn, not a conversation. Future user turns and reference
answers never enter a request. All native batches are validated before any
effect; their ordered grouping is retained for the official local checker.
This is a local adaptation, not the upstream leaderboard inference handler.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time

from relay_router.agent import _json_copy
from relay_router.errors import LiveBackendError, BackendActionError
from relay_router.v8 import model_order
from .agent_evaluation import (
    _Journal, _fingerprint, _read, _seal, _single_writer, _source_bundle,
    _sources, _unseal, _write, object_hash,
)
from .agent_metrics import annotate_costs

UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
PROTOCOL = "bfcl_native_multiturn_local_v1"
OFFICIAL_MAXIMUM_STEP_LIMIT = 20
OFFICIAL_EXECUTED_BATCH_CAP = 21  # Upstream stops after increment when count > 20.
SESSION_SHARED_FAILURE_CATEGORIES = (
    "transport", "timeout", "rate_limit", "http", "provider_contract",
    "model_identity", "generation_contract", "provider_unknown",
)


def _nonnegative(value, name, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} debe ser finito y no negativo.")


def _names(values, name):
    if type(values) is not tuple or any(type(v) is not str or not v for v in values):
        raise ValueError(f"{name} requiere tuple de nombres.")


@dataclass(frozen=True)
class BFCLState:
    turn_index: int
    step: int
    attempt: int
    previous_errors: tuple[str, ...]
    infrastructure_errors: tuple[str, ...]
    infrastructure_attempt: int
    failed_tiers: tuple[int, ...]
    remaining_calls: int
    context_utf8_bytes: int

    def __post_init__(self):
        for key in ("turn_index", "step", "attempt", "infrastructure_attempt", "remaining_calls", "context_utf8_bytes"):
            _nonnegative(getattr(self, key), key, integer=True)
        _names(self.previous_errors, "previous_errors")
        _names(self.infrastructure_errors, "infrastructure_errors")
        if type(self.failed_tiers) is not tuple:
            raise ValueError("failed_tiers requiere tuple.")
        for tier in self.failed_tiers:
            _nonnegative(tier, "failed_tiers", integer=True)

    def to_dict(self):
        return {key: list(value) if type(value) is tuple else value for key, value in vars(self).items()}


@dataclass(frozen=True)
class BFCLRequest:
    messages: tuple[dict, ...]
    tools: tuple[dict, ...]
    state: BFCLState

    def __post_init__(self):
        if type(self.messages) is not tuple or type(self.tools) is not tuple or not isinstance(self.state, BFCLState):
            raise ValueError("BFCLRequest requiere messages/tools tuple y BFCLState.")
        for message in self.messages:
            if type(message) is not dict or message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Rol de mensaje no soportado.")
            allowed = ({"role", "content", "tool_calls"} if message["role"] == "assistant" else
                       {"role", "content", "tool_call_id"} if message["role"] == "tool" else {"role", "content"})
            if set(message)-allowed or not {"role", "content"} <= set(message):
                raise ValueError("Campos de mensaje no soportados.")
        for tool in self.tools:
            if (type(tool) is not dict or set(tool) != {"type", "function"} or tool["type"] != "function"
                    or type(tool["function"]) is not dict or not tool["function"].get("name")):
                raise ValueError("Se requieren herramientas nativas OpenAI compatibles.")
        object.__setattr__(self, "messages", tuple(_json_copy(m) for m in self.messages))
        object.__setattr__(self, "tools", tuple(_json_copy(t) for t in self.tools))

    def snapshot(self):
        return BFCLRequest(self.messages, self.tools, self.state)

    def to_dict(self):
        snapshot = self.snapshot()
        return {"messages": list(snapshot.messages), "tools": list(snapshot.tools), "state": snapshot.state.to_dict()}

    def serialize(self):
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class BFCLReply:
    content: str | None = None
    tool_calls: tuple[dict, ...] = ()
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None

    def __post_init__(self):
        if self.content is not None and type(self.content) is not str:
            raise ValueError("content requiere texto o None.")
        if type(self.tool_calls) is not tuple or any(type(t) is not dict for t in self.tool_calls):
            raise ValueError("tool_calls requiere tuple de objetos nativos.")
        if type(self.usage) is not dict:
            raise ValueError("usage requiere objeto JSON.")
        if self.cost_usd is not None:
            _nonnegative(self.cost_usd, "cost_usd")
        object.__setattr__(self, "tool_calls", tuple(_json_copy(t) for t in self.tool_calls))
        object.__setattr__(self, "usage", _json_copy(self.usage))

    def snapshot(self):
        return BFCLReply(self.content, self.tool_calls, self.usage, self.cost_usd)

    def to_dict(self):
        snapshot = self.snapshot()
        return {"content": snapshot.content, "tool_calls": list(snapshot.tool_calls),
                "usage": snapshot.usage, "cost_usd": snapshot.cost_usd}


def request_input_bytes(request):
    """Observable serialized messages and tools, not tokenizer tokens."""
    return len(json.dumps({"messages": request.messages, "tools": request.tools}, sort_keys=True,
                          ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _public_request(messages, tools, turn_index, step, attempt, errors, infra_errors, infra_attempt, failed, remaining_calls):
    context_size = len(json.dumps({"messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8"))
    state = BFCLState(turn_index, step, attempt, tuple(errors), tuple(infra_errors), infra_attempt,
                      tuple(failed), remaining_calls, context_size)
    return BFCLRequest(tuple(messages), tuple(tools), state)


def build_initial_request(case, config, *, runtime_factory=None):
    """Exact first request without generation, future turns or private grading."""
    runtime = _runtime_factory(runtime_factory)(config["snapshot_path"], deepcopy(case["entry"]))
    return _public_request(deepcopy(case["entry"]["question"][0]), deepcopy(runtime.native_tools()),
                           0, 0, 0, (), (), 0, (), config["max_calls_per_turn"])


class BFCLAlwaysStrong:
    name = "always_strong"

    def __init__(self, models):
        self.models = model_order(tuple(models))

    def select(self, request):
        return self.models[-1]

    def manifest(self):
        return {"name": self.name, "models": list(self.models), "rule": "fixed strongest candidate"}


class BFCLAlwaysWeak(BFCLAlwaysStrong):
    name = "always_weak"

    def select(self, request):
        return self.models[0]

    def manifest(self):
        return {"name": self.name, "models": list(self.models), "rule": "fixed cheapest candidate"}


class BFCLReactiveCascade(BFCLAlwaysStrong):
    name = "cheap_then_escalate"

    def select(self, request):
        failed = request.state.failed_tiers
        if any(tier >= len(self.models) for tier in failed):
            raise ValueError("Tier fuera del catalogo.")
        return self.models[min(max(failed)+1, len(self.models)-1) if failed else 0]

    def manifest(self):
        return {"name": self.name, "models": list(self.models),
                "rule": "next tier after observed semantic failure; infrastructure never escalates"}


def _runtime_factory(runtime_factory):
    if runtime_factory is not None:
        return runtime_factory
    from .bfcl_environment import BFCLRuntime
    return BFCLRuntime


def _validate(cases, policies, config):
    if (config.get("schema_version") != 1 or config.get("upstream_commit") != UPSTREAM_COMMIT
            or config.get("execution_mode") not in {"external", "simulated"}
            or not isinstance(config.get("run_id"), str) or not config["run_id"]
            or not isinstance(config.get("snapshot_path"), str) or not config["snapshot_path"]
            or type(config.get("generation_manifest")) is not dict or not config["generation_manifest"]):
        raise ValueError("Declarar configuracion BFCL local, snapshot fijado y generador.")
    models = model_order(tuple(config.get("model_catalog", ())))
    if type(config.get("session_stop_on_infrastructure", False)) is not bool:
        raise ValueError("session_stop_on_infrastructure requiere booleano.")
    limit = config.get("max_calls_per_turn")
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("max_calls_per_turn requiere limite finito entre 1 y 128.")
    harness = config.get("harness", {})
    retries, delay = harness.get("max_transport_retries"), harness.get("retry_delay_seconds")
    if type(retries) is not int or not 0 <= retries <= 5 or type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 30:
        raise ValueError("Declarar retries de transporte acotados.")
    if not policies or len({p.name for p in policies}) != len(policies) or any(tuple(p.models) != models for p in policies):
        raise ValueError("Politicas unicas y catalogo identico requeridos.")
    if type(cases) is not list or not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Se requieren conversaciones BFCL completas y unicas.")
    bound = 0
    for case in cases:
        if (not isinstance(case.get("source_group"), str) or not case["source_group"]
                or case["entry"]["id"] != case["id"] or case.get("reference") is None):
            raise ValueError("Cada caso necesita identidad, grupo y referencia privada separada.")
        turns = case["entry"].get("question")
        if (type(turns) is not list or not turns or any(type(turn) is not list or not turn for turn in turns)
                or any(type(m) is not dict or m.get("role") not in {"user", "system"} for turn in turns for m in turn)):
            raise ValueError("question requiere turnos de mensajes user/system, sin respuestas futuras.")
        _json_copy(case)
        bound += len(turns)*limit*len(policies)
    maximum = config.get("max_total_generation_calls")
    if type(maximum) is not int or maximum < bound:
        raise ValueError(f"El limite global debe cubrir la cota predeclarada de {bound} intentos.")
    annotate_costs([], config.get("pricing", {}), config.get("compute_rates", {}))
    return bound


def prepare_bfcl_episodes(cases, policies, output_dir, config, *, generate=None, runtime_factory=None):
    """Freeze provenance and tasks before generation; invokes no checker."""
    cases, policies, config = deepcopy(cases), list(policies), deepcopy(config)
    bound = _validate(cases, policies, config)
    hook = getattr(generate, "session_stop_reason", None)
    if hook is not None and not callable(hook):
        raise ValueError("session_stop_reason debe ser callable o estar ausente.")
    runtime_factory = _runtime_factory(runtime_factory)
    runtime_provenance = []
    for case in cases:
        runtime = runtime_factory(config["snapshot_path"], deepcopy(case["entry"]))
        # Construction and schemas only. No execute or private evaluation.
        _json_copy(runtime.native_tools())
        runtime_provenance.append(_json_copy(runtime.provenance()) if callable(getattr(runtime, "provenance", None)) else None)
    bundle = _source_bundle(config)
    body = {"schema_version": 1, "protocol": PROTOCOL, "configuration": config, "cases": cases,
            "cases_sha256": object_hash(cases), "policies": {p.name: p.manifest() for p in policies},
            "policy_sequence": [p.name for p in policies], "policy_sources": {p.name: _fingerprint(p) for p in policies},
            "generation_source": _fingerprint(generate) if generate else None, "runtime_source": _fingerprint(runtime_factory),
            "runtime_provenance": runtime_provenance,
            "source_files_sha256": _sources(), "source_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "maximum_generation_calls": bound, "upstream_commit": UPSTREAM_COMMIT,
            "budget_contract": {"official_maximum_step_limit": 20, "official_executed_batch_cap": 21,
                "local_max_calls_per_turn": config["max_calls_per_turn"], "infrastructure_attempts_consume_cap": True},
            "session_control": {"stop_on_shared_infrastructure": config.get("session_stop_on_infrastructure", False),
                "shared_failure_categories": list(SESSION_SHARED_FAILURE_CATEGORIES),
                "excluded_failure_category": "context_capacity",
                "failure_rule": "terminal backend failure or exhausted call cap with infrastructure as last event",
                "generator_stop_hook": hook is not None,
                "hook_contract": "session_stop_reason() returns None or a nonempty JSON object before a new episode",
                "closure_rule": "completed episodes retained; unattempted pairs are not failures; summary is immutable"},
            "scope": "local native multi-turn adaptation; no leaderboard or independent-validation claim"}
    frozen = {**body, "freeze_sha256": object_hash(body)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if _read(path) != frozen:
            raise ValueError("La corrida congelada cambio; usar carpeta nueva.")
    else:
        if any(p.name != ".writer.lock" for p in output.iterdir()):
            raise FileExistsError("La salida debe ser nueva o conservar su manifiesto.")
        _write(path, frozen)
    archive = output / "source_bundle.zip"
    if archive.exists() and archive.read_bytes() != bundle:
        raise ValueError("Snapshot de fuentes alterado.")
    if not archive.exists():
        archive.write_bytes(bundle)
    return frozen


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Argumentos JSON duplicados.")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Argumento JSON no finito.")


def validate_native_batch(tool_calls, runtime):
    """Validate the whole batch before returning any executable invocation."""
    ids, decoded = set(), []
    for call in tool_calls:
        if (set(call) != {"id", "type", "function"} or call["type"] != "function"
                or type(call["id"]) is not str or not call["id"] or call["id"] in ids
                or type(call["function"]) is not dict or set(call["function"]) != {"name", "arguments"}
                or type(call["function"]["name"]) is not str or type(call["function"]["arguments"]) is not str):
            raise ValueError("Batch nativo invalido o IDs duplicados.")
        ids.add(call["id"])
        arguments = json.loads(call["function"]["arguments"], object_pairs_hook=_unique, parse_constant=_nonfinite)
        if type(arguments) is not dict:
            raise ValueError("arguments debe ser un objeto JSON.")
        decoded.append({"name": call["function"]["name"], "arguments": arguments})
    runtime.validate_batch(deepcopy(decoded))
    return decoded


def _episode(case, policy, events, config, success, termination, decoded_turns, evaluation=None):
    censored = termination if termination in {"infrastructure_failure", "interrupted_not_retried", "evaluation_error",
                                             "tool_resource_failure", "evaluation_resource_failure"} else None
    if termination.endswith("call_budget_exhausted") and any(e.get("error_scope") == "infrastructure" for e in events):
        success, censored = None, "infrastructure_budget_exhausted"
    return {"episode_id": case["id"], "source_group": case["source_group"], "policy": policy.name,
            "success": success, "operational_success": success is True, "termination": termination,
            "censored_reason": censored,
            "evaluation": evaluation, "decoded_turn_calls": deepcopy(decoded_turns),
            "events": annotate_costs(events, config.get("pricing", {}), config.get("compute_rates", {})),
            "provenance": {"protocol": PROTOCOL, "upstream_commit": UPSTREAM_COMMIT,
                           "execution_mode": config["execution_mode"], "independent_validation": False}}


def _recover(case, policy, records, config):
    if records[-1]["kind"] == "episode_completed":
        return records[-1]["data"]
    events = [r["data"] for r in records if r["kind"] == "call_completed"]
    if records[-1]["kind"] == "call_started":
        pending = deepcopy(records[-1]["data"])
        pending.update(generation_seconds=None, verification_seconds=None, tool_seconds=None, usage={},
                       reported_llm_cost_usd=None, response=None, error_type="InterruptedGeneration")
        events.append(pending)
    if events and events[-1].get("final_evaluation") is not None:
        last = events[-1]
        return _episode(case, policy, events, config, last["final_evaluation"]["valid"],
                        "conversation_complete", last["decoded_turn_calls"], last["final_evaluation"])
    if events and events[-1].get("terminal_reason") is not None:
        last = events[-1]
        success = False if last["terminal_reason"].endswith("call_budget_exhausted") else None
        return _episode(case, policy, events, config, success, last["terminal_reason"], last["decoded_turn_calls"])
    return _episode(case, policy, events, config, None, "interrupted_not_retried", [], None)


def _audit_journal(cases, policies, journal, config):
    """Replay accounting only, never provider calls, tool effects or grading."""
    if sum(r["kind"] == "call_started" for r in journal.records) > config["max_total_generation_calls"]:
        raise ValueError("El journal excede el presupuesto global.")
    for case in cases:
        for policy in policies:
            records = journal.for_key((case["id"], policy.name))
            starts = [r["data"] for r in records if r["kind"] == "call_started"]
            calls = [r["data"] for r in records if r["kind"] == "call_completed"]
            for event in starts:
                if event["model"] not in config["model_catalog"] or object_hash(event["request"]) != event["request_sha256"]:
                    raise ValueError("Modelo o solicitud persistida alterados.")
                value = event["request"]
                if set(value) != {"messages", "tools", "state"}:
                    raise ValueError("Campos de solicitud persistida incompatibles.")
                state = deepcopy(value["state"])
                for key in ("previous_errors", "infrastructure_errors", "failed_tiers"):
                    state[key] = tuple(state[key])
                request = BFCLRequest(tuple(value["messages"]), tuple(value["tools"]), BFCLState(**state))
                index = event["turn_index"]
                if (request.to_dict() != value or request_input_bytes(request) != request.state.context_utf8_bytes
                        or not 0 <= index < len(case["entry"]["question"])
                        or request.state.turn_index != index or request.state.step != event["step"]
                        or request.state.attempt != event["attempt"]
                        or not 0 <= event["within_turn_index"] < config["max_calls_per_turn"]
                        or request.state.remaining_calls != config["max_calls_per_turn"]-event["within_turn_index"]):
                    raise ValueError("Estado o presupuesto persistido incompatible.")
                visible_turns = [m for turn in case["entry"]["question"][:index+1] for m in turn]
                if [m for m in request.messages if m["role"] in {"user", "system"}] != visible_turns:
                    raise ValueError("Los mensajes revelan o modifican turnos no disponibles.")
            for started, completed in zip(starts, calls):
                if any(completed.get(key) != value for key, value in started.items()):
                    raise ValueError("La llamada completada cambio su decision persistida.")
            annotate_costs(calls, config.get("pricing", {}), config.get("compute_rates", {}))
            if records and records[-1]["kind"] == "episode_completed":
                if len(records) == 1 or _recover(case, policy, records[:-1], config) != records[-1]["data"]:
                    raise ValueError("El cierre no reproduce las llamadas y el desenlace persistidos.")


def _execute(case, policy, generate, config, journal, runtime_factory):
    from .bfcl_environment import BFCLResourceLimit
    runtime = runtime_factory(config["snapshot_path"], deepcopy(case["entry"]))
    messages, events, decoded_turns = [], [], []
    errors, infra_errors, failed = [], [], []
    step, attempt, infra_attempt, pinned, retry_wait = 0, 0, 0, None, 0.
    key = (case["id"], policy.name)
    models = config["model_catalog"]
    for turn_index, turn in enumerate(case["entry"]["question"]):
        attempt = 0  # A new user turn is not a retry of the preceding turn.
        messages.extend(deepcopy(turn))
        decoded_turns.append([])
        for within_turn in range(config["max_calls_per_turn"]):
            tools = tuple(deepcopy(runtime.native_tools()))
            used = sum(r["kind"] == "call_started" for r in journal.records)
            if used >= config["max_total_generation_calls"]:
                return _episode(case, policy, events, config, False, "global_call_budget_exhausted", decoded_turns)
            request = _public_request(messages, tools, turn_index, step, attempt, errors, infra_errors,
                                       infra_attempt, failed, config["max_calls_per_turn"]-within_turn)
            started = time.perf_counter()
            model = pinned if pinned is not None else policy.select(request.snapshot())
            router_seconds = time.perf_counter()-started
            if model not in models:
                raise ValueError("Modelo fuera del catalogo congelado.")
            event = {"call_index": len(events), "turn_index": turn_index, "within_turn_index": within_turn,
                "step": step, "attempt": attempt, "infrastructure_attempt": infra_attempt,
                "is_retry": bool(attempt or infra_attempt), "model": model, "router_seconds": router_seconds,
                "request": request.to_dict(), "request_sha256": object_hash(request.to_dict()),
                "routing_source": "pinned_infrastructure_retry" if pinned else "policy", "retry_wait_seconds": retry_wait}
            journal.append("call_started", key, event)
            if config.get("console_progress") is True:
                print(f"[BFCL] {policy.name}; turno {turn_index+1}; intento {within_turn+1}/{config['max_calls_per_turn']}.", flush=True)
            if retry_wait:
                time.sleep(retry_wait)
            started = time.perf_counter()
            reply, error, failure, failure_usage = None, None, None, {}
            try:
                reply = generate(model, request.snapshot())
                if not isinstance(reply, BFCLReply):
                    raise TypeError("El generador debe devolver BFCLReply.")
                reply = reply.snapshot()
                annotate_costs([{**event, "generation_seconds": 0., "verification_seconds": 0., "tool_seconds": 0.,
                    "usage": reply.usage, "reported_llm_cost_usd": reply.cost_usd}], {}, {})
            except Exception as exc:
                reply, error = None, type(exc).__name__
                failure = exc.details() if isinstance(exc, LiveBackendError) else {
                    "type": type(exc).__name__, "category": "generation_contract", "retryable": False,
                    "status_code": None, "retry_after_seconds": None}
                if isinstance(exc, LiveBackendError):
                    try:
                        annotate_costs([{**event, "generation_seconds": 0., "verification_seconds": 0., "tool_seconds": 0.,
                            "usage": exc.usage, "reported_llm_cost_usd": None}], {}, {})
                        failure_usage = deepcopy(exc.usage)
                    except (TypeError, ValueError):
                        pass
            event.update(generation_seconds=time.perf_counter()-started, verification_seconds=0., tool_seconds=0.,
                response=reply.to_dict() if reply else None, usage=reply.usage if reply else failure_usage,
                reported_llm_cost_usd=reply.cost_usd if reply else None, error_type=error,
                turn_completed=False, native_tool_calls=list(reply.tool_calls) if reply else [], tool_results=[])
            turn_closed, terminal = False, None
            if failure is not None:
                event["backend_failure"] = failure
                if failure["category"] == "model_action":
                    event["error_scope"] = "semantic"
                    turn_closed = True  # Upstream decoding failure advances to the next user turn.
                    messages.append({"role": "assistant", "content": ""})
                else:
                    event["error_scope"] = "infrastructure"
                    infra_errors.append(error)
                    infra_attempt += 1
                    pinned = model
                    retry_wait = max(config["harness"]["retry_delay_seconds"], failure["retry_after_seconds"] or 0.)
                    if (not failure["retryable"] or infra_attempt > config["harness"]["max_transport_retries"] or retry_wait > 30):
                        terminal = "infrastructure_failure"
            elif not reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.content or ""})
                turn_closed = True
            else:
                tick = time.perf_counter()
                try:
                    calls = validate_native_batch(reply.tool_calls, runtime)
                    encoded = [runtime.encode_call(c["name"], c["arguments"]) for c in calls]
                except BFCLResourceLimit as exc:
                    event.update(error_type=type(exc).__name__, error_scope="infrastructure",
                                 local_resource_failure={"phase": "batch_validation", "type": type(exc).__name__,
                                                         "details": _json_copy(exc.details())})
                    terminal = "tool_resource_failure"
                except (TypeError, ValueError) as exc:
                    event.update(error_type="InvalidAction:"+type(exc).__name__, error_scope="semantic")
                    # Keep invalid native output in the journal. Do not inject
                    # dangling tool-call IDs into a later provider request.
                    messages.append({"role": "assistant", "content": json.dumps(
                        {"content": reply.content, "tool_calls": list(reply.tool_calls)}, ensure_ascii=False)})
                    turn_closed = True
                else:
                    messages.append({"role": "assistant", "content": reply.content, "tool_calls": deepcopy(list(reply.tool_calls))})
                    tick_tools = time.perf_counter()
                    try:
                        results = runtime.execute_batch(deepcopy(calls))
                    except BFCLResourceLimit as exc:
                        event.update(error_type=type(exc).__name__, error_scope="infrastructure",
                                     local_resource_failure={"phase": "tool_execution", "type": type(exc).__name__,
                                                             "details": _json_copy(exc.details())})
                        terminal = "tool_resource_failure"
                    else:
                        if type(results) is not list or len(results) != len(calls):
                            raise TypeError("execute_batch debe devolver un resultado por llamada.")
                        for native, result in zip(reply.tool_calls, results):
                            if (type(result) is not dict or set(result) != {"ok", "observation", "error_type"}
                                    or type(result["ok"]) is not bool or type(result["observation"]) is not str):
                                raise TypeError("Resultado BFCL de herramienta incompatible.")
                            messages.append({"role": "tool", "tool_call_id": native["id"], "content": result["observation"]})
                            if not result["ok"]:
                                event.update(error_type="ToolError:"+str(result["error_type"]), error_scope="semantic")
                        event["tool_results"] = deepcopy(results)
                        decoded_turns[-1].append(encoded)
                    event["tool_seconds"] = time.perf_counter()-tick_tools
                event["verification_seconds"] = max(0., time.perf_counter()-tick-event["tool_seconds"])
            if event.get("error_scope") == "semantic":
                errors.append(event["error_type"])
                failed.append(models.index(model))
                attempt += 1
                infra_attempt, pinned, retry_wait = 0, None, 0.
            elif event.get("error_scope") != "infrastructure":
                step += 1
                attempt, infra_attempt, pinned, retry_wait = 0, 0, None, 0.
            event["turn_completed"] = turn_closed
            if turn_closed and turn_index == len(case["entry"]["question"])-1:
                tick = time.perf_counter()
                try:
                    evaluation = runtime.evaluate(deepcopy(decoded_turns), deepcopy(case["reference"]))
                    if type(evaluation) is not dict or type(evaluation.get("valid")) is not bool:
                        raise TypeError("El checker BFCL debe devolver valid booleano.")
                    _json_copy(evaluation)
                except BFCLResourceLimit as exc:
                    event.update(evaluation_error_type=type(exc).__name__,
                                 local_resource_failure={"phase": "private_evaluation", "type": type(exc).__name__,
                                                         "details": _json_copy(exc.details())})
                    terminal = "evaluation_resource_failure"
                except Exception as exc:
                    event.update(evaluation_error_type=type(exc).__name__, terminal_reason="evaluation_error",
                                 decoded_turn_calls=deepcopy(decoded_turns))
                    terminal = "evaluation_error"
                else:
                    event.update(final_evaluation=evaluation, decoded_turn_calls=deepcopy(decoded_turns))
                event["verification_seconds"] += time.perf_counter()-tick
            if terminal is not None:
                event.update(terminal_reason=terminal, decoded_turn_calls=deepcopy(decoded_turns))
            elif not turn_closed and within_turn == config["max_calls_per_turn"]-1:
                event.update(terminal_reason="turn_call_budget_exhausted", decoded_turn_calls=deepcopy(decoded_turns))
            events.append(event)
            journal.append("call_completed", key, event)
            if config.get("console_progress") is True:
                print(f"[BFCL] {event['error_type'] or ('turno cerrado' if turn_closed else 'herramientas ejecutadas')}.", flush=True)
            if event.get("final_evaluation") is not None:
                return _episode(case, policy, events, config, event["final_evaluation"]["valid"],
                    "conversation_complete", decoded_turns, event["final_evaluation"])
            if terminal:
                return _episode(case, policy, events, config, None, terminal, decoded_turns)
            if turn_closed:
                break
        else:
            return _episode(case, policy, events, config, False, "turn_call_budget_exhausted", decoded_turns)
    raise RuntimeError("Conversacion sin cierre registrado.")


def _shared_session_failure(episode):
    termination = episode["termination"]
    if termination != "infrastructure_failure" and not termination.endswith("call_budget_exhausted"):
        return None
    last = episode["events"][-1] if episode["events"] else {}
    failure = last.get("backend_failure", {})
    if (last.get("error_scope") != "infrastructure"
            or failure.get("category") not in SESSION_SHARED_FAILURE_CATEGORIES):
        return None
    return {"reason": "shared_infrastructure_failure", "episode_id": episode["episode_id"],
            "policy": episode["policy"], "category": failure["category"],
            "status_code": failure.get("status_code"), "error_type": last.get("error_type")}


def _generator_session_stop(generate):
    hook = getattr(generate, "session_stop_reason", None)
    if hook is None:
        return None
    value = hook()
    if value is not None and (type(value) is not dict or not value):
        raise ValueError("session_stop_reason debe devolver None o un objeto JSON no vacio.")
    return _json_copy(value) if value is not None else None


def _session_summary(frozen, journal, planned, config, stop_reason):
    """Closed records alone define outcomes; a stopped session invents no failures."""
    episodes = [r["data"] for r in journal.records if r["kind"] == "episode_completed"]
    completed = [[e["episode_id"], e["policy"]] for e in episodes]
    if completed != planned[:len(completed)]:
        raise ValueError("Los cierres no siguen el orden de episodios congelado.")
    if any(journal.for_key(tuple(pair)) for pair in planned[len(completed):]):
        raise ValueError("El resumen no puede ocultar intentos sin cierre.")
    unattempted = planned[len(completed):]
    if unattempted and (type(stop_reason) is not dict or not stop_reason):
        raise ValueError("Una sesion parcial necesita causa de detencion explicita.")
    return {"schema_version": 1, "protocol": PROTOCOL, "freeze_sha256": frozen["freeze_sha256"],
        "journal_sha256": object_hash(journal.records), "journal_records": len(journal.records), "episodes": episodes,
        "planned_pairs": planned, "completed_pairs": completed, "unattempted_pairs": unattempted,
        "session_stopped": bool(unattempted), "session_stop_reason": stop_reason if unattempted else None,
        "execution_complete": not unattempted,
        "generation_calls": sum(r["kind"] == "call_started" for r in journal.records),
        "maximum_generation_calls": frozen["maximum_generation_calls"], "independent_validation": False,
        "execution_mode": config["execution_mode"], "scope": frozen["scope"]}


def run_bfcl_episodes(cases, policies, generate, output_dir, config, *, runtime_factory=None):
    """Explicit generation only; every attempt is durable before its callback."""
    if not callable(generate):
        raise ValueError("Se requiere callback de generacion explicito.")
    cases, policies, config = deepcopy(cases), list(policies), deepcopy(config)
    runtime_factory = _runtime_factory(runtime_factory)
    _validate(cases, policies, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with _single_writer(output):
        frozen = prepare_bfcl_episodes(cases, policies, output, config, generate=generate, runtime_factory=runtime_factory)
        allowed = {(case["id"], policy.name) for case in cases for policy in policies}
        journal = _Journal(output, frozen["freeze_sha256"], allowed)
        _audit_journal(cases, policies, journal, config)
        pairs = [(case, policy) for case in cases for policy in policies]
        planned = [[case["id"], policy.name] for case, policy in pairs]
        summary = output / "summary.json"
        if summary.exists():
            previous = _unseal(summary)
            expected = _session_summary(frozen, journal, planned, config, previous.get("session_stop_reason"))
            if previous != expected:
                raise ValueError("El resumen no reproduce los episodios persistidos.")
            return previous  # A closed partial session is immutable, including its unattempted pairs.
        stop_reason = None
        for case, policy in pairs:
            key = (case["id"], policy.name)
            existing = journal.for_key(key)
            if not existing:
                stop_reason = _generator_session_stop(generate)
                if stop_reason is not None:
                    break
            episode = (_recover(case, policy, existing, config) if existing else
                _execute(case, policy, generate, config, journal, runtime_factory))
            if not existing or existing[-1]["kind"] != "episode_completed":
                journal.append("episode_completed", key, episode)
            if config.get("session_stop_on_infrastructure", False):
                stop_reason = _shared_session_failure(episode)
                if stop_reason is not None:
                    break
        _audit_journal(cases, policies, journal, config)
        if (_sources() != frozen["source_files_sha256"] or _fingerprint(generate) != frozen["generation_source"]
                or _fingerprint(runtime_factory) != frozen["runtime_source"]
                or _read(output / "manifest.json") != frozen):
            raise ValueError("Fuentes o contratos cambiaron durante la corrida.")
        result = _session_summary(frozen, journal, planned, config, stop_reason)
        _write(summary, _seal(result))
        return result
