"""Run the user's installed OpenCode CLI with an explicit primary model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote
from uuid import uuid4


class OpenCodeError(RuntimeError):
    """Configuration or local process error, before usable CLI output."""


@dataclass(frozen=True)
class OpenCodeModel:
    model_id: str
    name: str
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    context_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_call: bool | None = None
    reasoning: bool | None = None
    input_text: bool | None = None
    output_text: bool | None = None
    status: str | None = None
    advertised_free: bool | None = None


def _executable() -> str:
    # A project-local opencode.exe (or an implicit "." in PATH) must never
    # intercept a prompt. npm's Windows shim forwards %* via cmd.exe, so use
    # the adjacent official native executable directly.
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    if os.name == "nt" and local.is_absolute():
        managed = local / "Relay/opencode-v2/node_modules/@opencode/cli-windows-x64-baseline/bin/opencode.exe"
        if managed.is_file():
            return str(managed.resolve())
    for entry in os.get_exec_path():
        directory = Path(entry)
        if not directory.is_absolute() or directory.resolve() == Path.cwd().resolve():
            continue
        if os.name == "nt":
            candidates = [directory / "opencode.exe"]
            if (directory / "opencode.cmd").is_file():
                candidates.append(directory / "node_modules/@opencode/cli/bin/opencode.exe")
                candidates.append(directory / "node_modules/opencode-ai/bin/opencode.exe")
        else:
            candidates = [directory / "opencode"]
        for path in candidates:
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    raise OpenCodeError("No se encontró el ejecutable de OpenCode en PATH.")


@lru_cache(maxsize=8)
def _version(binary, modified):
    try:
        probe = subprocess.run([binary, "--version"], cwd=Path(binary).parent,
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpenCodeError("No se pudo identificar la versión de OpenCode.") from exc
    match = re.fullmatch(rb"(?:opencode v?)?([12])\.\d+\.\d+(?:[-+][\w.-]+)?", probe.stdout.strip())
    if not match or probe.returncode:
        raise OpenCodeError("La versión de OpenCode no es compatible con este adaptador.")
    return int(match[1])


def _is_v2(binary):
    return _version(binary, Path(binary).stat().st_mtime_ns) == 2


def _workdir(workspace) -> Path:
    path = Path(workspace).expanduser().resolve()
    if not path.is_dir():
        raise OpenCodeError("El directorio del proyecto no existe.")
    return path


def _valid_id(value):
    if not isinstance(value, str) or not value.isprintable() or any(c.isspace() for c in value):
        return False
    provider, slash, model = value.partition("/")
    return bool(provider and slash and model and not value.startswith("-"))


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _price(value):
    if type(value) in (int, float) and value >= 0 and math.isfinite(value):
        return value
    return None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _flag(value):
    return value if type(value) is bool else None


def _legacy_additional_charge(cost):
    # v1 can publish multiple context thresholds as well as a legacy 200K tier.
    if any(_price(cost.get(k)) != 0 for k in ("cache_read", "cache_write") if k in cost):
        return True
    cache = cost.get("cache", {"read": 0, "write": 0})
    if not isinstance(cache, dict) or set(cache) != {"read", "write"} or any(_price(x) != 0 for x in cache.values()):
        return True
    tiers = cost.get("tiers", [])
    if not isinstance(tiers, list):
        return True
    if "experimentalOver200K" in cost:
        tiers = [*tiers, cost["experimentalOver200K"]]
    return any(not _zero_token_tier(tier) for tier in tiers)


def _zero_token_tier(tier):
    if not isinstance(tier, dict) or not set(tier) <= {"input", "output", "cache", "tier"}:
        return False
    cache = _mapping(tier.get("cache"))
    return (set(cache) == {"read", "write"}
            and all(_price(rate) == 0 for rate in (tier.get("input"), tier.get("output"), *cache.values())))


def parse_catalog(output: str) -> list[OpenCodeModel]:
    """models --verbose emits an ID line followed by a pretty-printed JSON object."""
    decoder = json.JSONDecoder()
    models = {}
    cursor = 0
    while cursor < len(output):
        if output[cursor].isspace():
            cursor += 1
            continue
        end = output.find("\n", cursor)
        if end == -1:
            end = len(output)
        model_id = output[cursor:end].strip()
        if not _valid_id(model_id) or model_id in models:
            raise OpenCodeError("El catálogo de OpenCode contiene un identificador inesperado.")
        cursor = end
        while cursor < len(output) and output[cursor].isspace():
            cursor += 1
        meta = {}
        if cursor < len(output) and output[cursor] == "{":
            try:
                meta, cursor = decoder.raw_decode(output, cursor)
            except ValueError as exc:
                raise OpenCodeError("OpenCode devolvió metadatos de catálogo incompletos.") from exc
        cost, limit, cap = (_mapping(meta.get(key)) for key in ("cost", "limit", "capabilities"))
        name = meta.get("name")
        models[model_id] = OpenCodeModel(
            model_id=model_id, name=name if isinstance(name, str) and name.strip() else model_id,
            input_usd_per_million=_price(cost.get("input")), output_usd_per_million=_price(cost.get("output")),
            context_tokens=_count(limit.get("context")), input_tokens=_count(limit.get("input")),
            output_tokens=_count(limit.get("output")), tool_call=_flag(cap.get("toolcall")),
            reasoning=_flag(cap.get("reasoning")), input_text=_flag(_mapping(cap.get("input")).get("text")),
            output_text=_flag(_mapping(cap.get("output")).get("text")),
            status=meta.get("status") if isinstance(meta.get("status"), str) else None,
            advertised_free=False if _legacy_additional_charge(cost) else None,
        )
    if not models:
        raise OpenCodeError("OpenCode no devolvió modelos. Revisa sus proveedores configurados.")
    return list(models.values())


def _parse_v2_catalog(output, directory):
    record = _mapping(json.loads(output))
    location = _mapping(record.get("location")).get("directory")
    if not isinstance(location, str) or Path(location) != directory or not isinstance(record.get("data"), list):
        raise OpenCodeError("El catálogo de OpenCode no corresponde a la carpeta solicitada.")
    models = {}
    for item in record["data"]:
        row = _mapping(item)
        if row.get("enabled") is not True or row.get("type") not in (None, "chat"):
            continue
        provider, model = row.get("providerID"), row.get("id")
        if not isinstance(provider, str) or "/" in provider or not isinstance(model, str):
            raise OpenCodeError("Identificador inválido en el catálogo de OpenCode.")
        key = f"{provider}/{model}"
        if not _valid_id(key) or key in models:
            raise OpenCodeError("Identificador inválido o repetido en el catálogo de OpenCode.")
        rates = row.get("cost")
        tiers = [_mapping(tier) for tier in rates] if isinstance(rates, list) else []
        base = [tier for tier in tiers if "tier" not in tier]
        cost = base[0] if len(base) == 1 else {}
        # Evaluate every context tier, including cache charges. Missing prices
        # and future unrecognised billing fields cannot establish free access.
        zero_rates = bool(cost) and all(_zero_token_tier(tier) for tier in tiers)
        cap, limit = _mapping(row.get("capabilities")), _mapping(row.get("limit"))
        inputs, outputs = cap.get("input"), cap.get("output")
        models[key] = OpenCodeModel(
            model_id=key, name=row.get("name") if isinstance(row.get("name"), str) else key,
            input_usd_per_million=_price(cost.get("input")), output_usd_per_million=_price(cost.get("output")),
            context_tokens=_count(limit.get("context")), input_tokens=_count(limit.get("input")),
            output_tokens=_count(limit.get("output")), tool_call=_flag(cap.get("tools")),
            reasoning=True if _mapping(row.get("compatibility")).get("reasoningField") else None,
            input_text="text" in inputs if isinstance(inputs, list) else None,
            output_text="text" in outputs if isinstance(outputs, list) else None,
            status=row.get("status") if isinstance(row.get("status"), str) else None,
            advertised_free=(provider == "opencode" or None) if zero_rates else False)
    if not models:
        raise OpenCodeError("OpenCode no devolvió modelos de chat habilitados.")
    return list(models.values())


def discover_models(workspace: Path) -> list[OpenCodeModel]:
    directory = _workdir(workspace)
    binary = _executable()
    v2 = _is_v2(binary)
    args = (["api", "model.list", "--standalone", "--header", f"x-opencode-directory:{quote(str(directory), safe='')}"]
            if v2 else ["models", "--verbose"])
    try:
        result = subprocess.run([binary, *args], cwd=directory,
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpenCodeError("No se pudo consultar el catálogo local de OpenCode.") from exc
    if result.returncode != 0:
        raise OpenCodeError(f"OpenCode no pudo listar los modelos (salida {result.returncode}). "
                            "Revisa el acceso a su configuración desde este entorno.")
    try:
        if v2:
            return _parse_v2_catalog(result.stdout, directory)
        return parse_catalog(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenCodeError("OpenCode devolvió un catálogo ilegible.") from exc


def _sum_reported(values):
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _usage(finishes, started, complete_trace):
    tokens = [_mapping(part.get("tokens")) for part in finishes]
    costs = [_price(part.get("cost")) for part in finishes]
    # OpenCode can default missing upstream usage to zero; task/subagent and
    # hidden retry charges are not enumerated by the primary CLI event stream.
    # Keep raw reported subtotals separate from an exact task charge.
    usage = {
        "cost_usd": None,
        "reported_cost_usd": _sum_reported(costs),
        "scope": "opencode_reported_primary_session_steps",
        "steps_started": started,
        "steps_finished": len(finishes),
        "all_steps_reported": bool(complete_trace and started == len(finishes) and finishes
                                   and all(value is not None for value in costs)
                                   and all(_count(t.get("input")) is not None and _count(t.get("output")) is not None
                                           for t in tokens)),
    }
    for name in ("input", "output", "reasoning"):
        usage[f"{name}_tokens"] = _sum_reported(_count(row.get(name)) for row in tokens)
    for name in ("read", "write"):
        usage[f"cache_{name}_tokens"] = _sum_reported(_count(_mapping(row.get("cache")).get(name)) for row in tokens)
    return usage


_EXPIRED_MESSAGE = re.compile(
    r"^(?:(?:your|the|this) )?(?:(?:api|access|authentication) )?(?:key|token|credential) "
    r"(?:has expired|is expired)(?:[.!:]|$)", re.IGNORECASE)


def _explicitly_expired(data):
    # Only accept the provider's direct, affirmative message. Do not publish
    # response bodies or inspect tool output, headers or unrelated JSON values.
    messages = [data.get("message")]
    body = data.get("responseBody")
    if isinstance(body, str) and len(body) <= 65536:
        try:
            body = json.loads(body)
        except (ValueError, RecursionError):
            body = None
    envelope = _mapping(body)
    messages.extend([envelope.get("message"), _mapping(envelope.get("error")).get("message")])
    return any(isinstance(value, str) and _EXPIRED_MESSAGE.match(value.strip()) for value in messages)


def _error_details(event):
    info = _mapping(event.get("error"))
    error_type = info.get("type")
    if isinstance(error_type, str) and re.fullmatch(r"provider\.[a-z0-9][a-z0-9.-]{0,55}", error_type):
        status = info.get("status")
        if error_type == "provider.free-limit":
            return ("OpenCode indica que se alcanzó el límite de uso gratuito.",
                    status if type(status) is int and status == 429 else None, None)
        # v2 session errors have their own typed status and body.
        if type(status) is int and 100 <= status <= 599:
            info = {"name": "APIError", "data": {"statusCode": status, "message": info.get("message"),
                                                "responseBody": _mapping(info.get("response")).get("body")}}
        else:
            return error_type, None, None
    name = info.get("name")
    label = name if isinstance(name, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,64}", name) else "error de sesión"
    data = _mapping(info.get("data"))
    status = data.get("statusCode")
    if type(status) is int and 100 <= status <= 599:
        if name != "APIError":
            return f"{label} (HTTP {status})", None, None
        if status == 401 and _explicitly_expired(data):
            return f"{label} (HTTP 401). El proveedor indica que la credencial venció.", 401, "credential_expired"
        hints = {401: "revisa la autenticación del proveedor", 402: "revisa los créditos o el plan",
                 403: "revisa el acceso de tu cuenta al modelo", 404: "el proveedor no encontró el recurso",
                 429: "se alcanzó una cuota o un límite de frecuencia"}
        detail = hints.get(status)
        return f"{label} (HTTP {status})" + (f". {detail}." if detail else ""), status, None
    return label, None, None


_UNCONFIRMED = "La sesión no confirmó una finalización normal."


def _read_events(lines):
    session_id = None
    parts = {}
    problem = None
    http_status = None
    auth_reason = None
    complete_trace = True
    retried = False
    for number, line in enumerate(lines):
        if isinstance(line, dict):
            event = line
        else:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                complete_trace = False
                continue
        if not isinstance(event, dict):
            complete_trace = False
            continue
        sid = event.get("sessionID")
        if not isinstance(sid, str) or not sid or (session_id and sid != session_id):
            complete_trace = False
            continue
        session_id = sid
        kind = event.get("type")
        if kind == "error":
            problem, http_status, auth_reason = _error_details(event)
        elif kind == "retry":
            retried = True
        elif kind in ("step_start", "step_finish", "text", "tool_use"):
            part = _mapping(event.get("part"))
            if not part or part.get("sessionID", sid) != sid:
                complete_trace = False
                continue
            part_id = part.get("id")
            if not isinstance(part_id, str) or not part_id:
                complete_trace = False
                part_id = number
            # OpenCode forwards part updates, including possible repeated snapshots.
            if kind == "text":
                parts[kind, part_id] = {"text": part.get("text")}
            elif kind == "tool_use":
                # File contents, tool inputs and outputs can be large/private.
                name = part.get("tool")
                status = _mapping(part.get("state")).get("status")
                parts[kind, part_id] = {
                    "name": name if isinstance(name, str) else None,
                    "status": status if status in ("pending", "running", "completed", "error") else None,
                }
            elif kind == "step_finish":
                parts[kind, part_id] = {key: part.get(key) for key in ("reason", "cost", "tokens")}
            else:
                parts[kind, part_id] = None
    finishes = [p for (k, _), p in parts.items() if k == "step_finish"]
    started = sum(k == "step_start" for k, _ in parts)
    text_output = "\n\n".join(p["text"] for (k, _), p in parts.items()
                               if k == "text" and isinstance(p.get("text"), str))
    tools = [p for (k, _), p in parts.items() if k == "tool_use"]
    if not problem and (not complete_trace or not finishes or finishes[-1].get("reason") != "stop"):
        problem = _UNCONFIRMED
    return {"status": "error" if problem else "completed", "session_id": session_id,
            "text": text_output, "error": problem, "http_status": http_status,
            "http_error_scope": "session" if http_status is not None else None, "auth_failure_reason": auth_reason,
            "usage": _usage(finishes, started, complete_trace and not problem and not retried), "tool_calls": tools}


def _local_json(binary, directory, arguments):
    try:
        response = subprocess.run([binary, *arguments], cwd=directory, stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=30, check=False,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if response.returncode == 0:
            return json.loads(response.stdout)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        pass
    return None


def _same_session(row, title, directory):
    return (isinstance(row, dict) and row.get("title") == title
            and isinstance(row.get("directory"), str) and Path(row["directory"]) == directory)


def _valid_session_id(value):
    return isinstance(value, str) and re.fullmatch(r"ses_[a-zA-Z0-9_-]+", value) is not None


def _v2_snapshot(binary, directory, session_id, title=None):
    export = _mapping(_local_json(binary, directory, ["session", "export", session_id, "--standalone"]))
    info, messages = _mapping(export.get("info")), export.get("messages")
    location = _mapping(info.get("location")).get("directory")
    if (info.get("id") != session_id or not isinstance(location, str) or Path(location) != directory
            or (title is not None and info.get("title") != title) or not isinstance(messages, list)):
        return None
    indexed = {}
    for message in messages:
        mid = _mapping(message).get("id")
        if not isinstance(mid, str) or not mid.startswith("msg_") or mid in indexed:
            return None
        indexed[mid] = message
    return indexed


def _v2_answer(binary, directory, sid, title, before, model_id, prompt):
    if title is not None:
        sessions = _local_json(binary, directory, ["session", "list", "--max-count", "100", "--format", "json",
                                                  "--standalone"])
        matches = [row for row in sessions if _same_session(row, title, directory)] if isinstance(sessions, list) else []
        if len(matches) != 1 or (sid and matches[0].get("id") != sid):
            return None
        sid = matches[0].get("id")
    if not _valid_session_id(sid):
        return None
    snapshot = _v2_snapshot(binary, directory, sid, title)
    # v2 exports ordered messages without parent IDs. Accept one complete turn
    # appended to the original prefix. Concurrent writers must never mix replies.
    if snapshot is None or tuple(snapshot)[:len(before)] != before:
        return None
    new = list(snapshot.values())[len(before):]
    turns = [i for i, msg in enumerate(new) if msg.get("type") == "user"]
    if len(turns) != 1 or new[turns[0]].get("text") != prompt:
        return None
    user_idx = turns[0]
    if any(msg.get("type") not in ("model-switched", "agent-switched", "system", "synthetic")
           for msg in new[:user_idx]):
        return None
    after = new[user_idx + 1:]
    if not after or after[-1].get("type") != "idle":
        return None
    steps = []
    extra_usage = False
    for msg in after[:-1]:
        kind = msg.get("type")
        if kind == "compaction":
            extra_usage = True
        elif kind == "assistant":
            selected = _mapping(msg.get("model"))
            if f"{selected.get('providerID')}/{selected.get('id')}" != model_id:
                return None
            steps.append(msg)
        elif kind not in ("system", "synthetic", "skill"):
            return None
    events = []
    for step in steps:
        mid = step["id"]
        events.append({"type": "step_start", "sessionID": sid, "part": {"id": mid}})
        content = step.get("content")
        if not isinstance(content, list):
            return None
        for i, part in enumerate(content):
            part = _mapping(part)
            pid = f"{mid}/{i}"
            if part.get("type") == "text":
                events.append({"type": "text", "sessionID": sid, "part": {"id": pid, "text": part.get("text")}})
            elif part.get("type") == "tool":
                events.append({"type": "tool_use", "sessionID": sid, "part": {
                    "id": pid, "tool": part.get("name"), "state": {"status": _mapping(part.get("state")).get("status")}}})
        if step.get("finish") and _mapping(step.get("time")).get("completed") is not None:
            events.append({"type": "step_finish", "sessionID": sid, "part": {
                "id": mid, "reason": step["finish"], "cost": step.get("cost"), "tokens": step.get("tokens")}})
        if step.get("retry"):
            extra_usage = True
        if step.get("error"):
            events.append({"type": "error", "sessionID": sid, "error": step["error"]})
    step_error = any(event["type"] == "error" for event in events)
    if not step_error and after[-1].get("error"):
        events.append({"type": "error", "sessionID": sid, "error": after[-1]["error"]})
    answer = _read_events(events)
    if step_error and answer["http_status"] is not None:
        answer["http_error_scope"] = "selected_model"
    if after[-1].get("outcome") != "succeeded" or not steps or steps[-1].get("finish") != "stop":
        answer.update(status="error", error=answer["error"] or _UNCONFIRMED)
    answer["session_id"] = sid
    if extra_usage or answer["status"] == "error":
        answer["usage"]["all_steps_reported"] = False
    return answer


def _session_snapshot(binary, directory, session_id):
    record = _mapping(_local_json(binary, directory, ["export", session_id]))
    info = _mapping(record.get("info"))
    if (info.get("id") != session_id or not isinstance(info.get("directory"), str)
            or Path(info["directory"]) != directory or not isinstance(record.get("messages"), list)):
        return None
    messages = {}
    for message in record["messages"]:
        item = _mapping(_mapping(message).get("info"))
        mid = item.get("id")
        if not isinstance(mid, str) or not mid or mid in messages or item.get("sessionID") != session_id:
            return None
        messages[mid] = message
    return messages


def _matches_prompt(message, model_id, prompt):
    info = _mapping(message.get("info"))
    selected = _mapping(info.get("model"))
    provider, model = model_id.split("/", 1)
    parts = message.get("parts")
    if (info.get("role") != "user" or selected.get("providerID") != provider
            or selected.get("modelID") != model or not isinstance(parts, list)):
        return False
    text_parts = [p.get("text") for p in parts
                  if isinstance(p, dict) and p.get("type") == "text" and not p.get("synthetic")]
    return text_parts == [prompt]


def _export_answer(record, session_id, model_id, prompt):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    infos = [_mapping(_mapping(msg).get("info")) for msg in messages]
    user, *replies = infos
    provider, model = model_id.split("/", 1)
    if (any(info.get("sessionID") != session_id for info in infos)
            or not user.get("id") or not _matches_prompt(messages[0], model_id, prompt)
            or any(info.get("role") != "assistant" or info.get("parentID") != user["id"]
                   or info.get("providerID") != provider or info.get("modelID") != model for info in replies)):
        return None
    for info, msg in zip(infos, messages):
        if (not isinstance(msg.get("parts"), list)
                or any(not isinstance(p, dict) or p.get("sessionID") != session_id
                       or p.get("messageID") != info.get("id") for p in msg["parts"])):
            return None
    kinds = {"step-start": "step_start", "step-finish": "step_finish", "tool": "tool_use", "text": "text"}

    def events():
        for reply in messages[1:]:
            for part in reply["parts"]:
                kind = kinds.get(part.get("type"))
                if kind and not part.get("synthetic") and not part.get("ignored"):
                    yield {"type": kind, "sessionID": session_id, "part": part}
            if reply["info"].get("error"):
                yield {"type": "error", "sessionID": session_id, "error": reply["info"]["error"]}

    answer = _read_events(events())
    if answer["http_status"] is not None:
        # The assistant message's parent, provider and model were verified above.
        answer["http_error_scope"] = "selected_model"
    last = replies[-1]
    if answer["status"] == "completed" and (
        last.get("finish") != "stop" or _count(_mapping(last.get("time")).get("completed")) is None
    ):
        answer.update(status="error", error=_UNCONFIRMED)
        answer["usage"]["all_steps_reported"] = False
    return answer if answer["session_id"] == session_id else None


def _recover_session(binary, directory, title, expected_session, model_id, prompt):
    # Some releases exit before JSON events flush. Resolve ONLY the session we
    # named for this dispatch. Reading it must never trigger another generation.
    sessions = _local_json(binary, directory, ["session", "list", "--max-count", "100", "--format", "json"])
    if not isinstance(sessions, list):
        return None
    matches = [row for row in sessions if _same_session(row, title, directory)]
    if len(matches) != 1:
        return None
    sid = matches[0].get("id")
    if not _valid_session_id(sid) or (expected_session and sid != expected_session):
        return None
    record = _mapping(_local_json(binary, directory, ["export", sid]))
    info = _mapping(record.get("info"))
    if info.get("id") != sid or not _same_session(info, title, directory):
        return None
    return _export_answer(record, sid, model_id, prompt)


def _continued_answer(binary, directory, session_id, previous_ids, model_id, prompt):
    # JSON events from an existing session do not identify the initiating user
    # message. Attribute responses by parent ID in the persisted conversation.
    messages = _session_snapshot(binary, directory, session_id)
    if messages is None:
        return None
    new = [message for mid, message in messages.items() if mid not in previous_ids]
    users = [message for message in new if _matches_prompt(message, model_id, prompt)]
    if len(users) != 1:
        return None
    user = users[0]
    replies = [message for message in new if message["info"].get("parentID") == user["info"]["id"]]
    return _export_answer({"messages": [user, *replies]}, session_id, model_id, prompt)


def _send_prompt(stdin, payload):
    try:
        with stdin:
            stdin.write(payload)
    except BrokenPipeError:
        # The CLI can exit before it reads the prompt on a configuration error.
        pass


def _permission_denied(stderr):
    marker = b"auto-rejecting"
    tail = b""
    denied = False
    while chunk := stderr.read(65536):
        if not denied:
            data = tail + chunk
            denied = marker in data
            tail = data[-len(marker):]
    return denied


def run_prompt(prompt: str, model_id: str, workspace: Path, *, session_id: str | None = None) -> dict:
    """Send one message, optionally continuing the exact session in its original project."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise OpenCodeError("Escribe un prompt antes de ejecutar OpenCode.")
    if not _valid_id(model_id):
        raise OpenCodeError("El modelo debe tener el formato proveedor/modelo.")
    directory = _workdir(workspace)
    executable = _executable()
    v2 = _is_v2(executable)
    started = time.monotonic()
    command = [executable, "run", "--model", model_id]
    command.extend(["--standalone"] if v2 else ["--dir", str(directory)])
    command.extend(["--format", "json"])
    title = None
    previous_ids = ()
    if session_id is None:
        title = f"Relay-{uuid4().hex}"
        command.extend(["--title", title])
    else:
        if not _valid_session_id(session_id):
            raise OpenCodeError("El identificador de la conversación de OpenCode es inválido.")
        snapshot = (_v2_snapshot if v2 else _session_snapshot)(executable, directory, session_id)
        if snapshot is None:
            raise OpenCodeError("No se pudo abrir esta conversación de OpenCode en la carpeta indicada.")
        previous_ids = tuple(snapshot) if v2 else set(snapshot)
        del snapshot
        command.extend(["--session", session_id])
    try:
        with subprocess.Popen(command, cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              env={**os.environ, "PWD": str(directory)} if v2 else None,
                              stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as proc:
            # Drain both pipes while writing the prompt: tool output can exceed
            # the OS pipe buffer. Retain only the reply and whitelisted metadata.
            with ThreadPoolExecutor(max_workers=2) as workers:
                writer = workers.submit(_send_prompt, proc.stdin, prompt.encode("utf-8"))
                warning = workers.submit(_permission_denied, proc.stderr)
                try:
                    answer = _read_events(proc.stdout)
                    returncode = proc.wait()
                    writer.result()
                    denied = warning.result()
                except BaseException:
                    if proc.poll() is None:
                        proc.kill()
                    raise
    except OSError as exc:
        raise OpenCodeError("No se pudo comunicar con el proceso de OpenCode.") from exc
    answer["result_source"] = "cli_events"
    if v2:
        recovered = None
        if not session_id or answer["session_id"] in (None, session_id):
            recovered = _v2_answer(executable, directory, session_id or answer["session_id"], title,
                                   previous_ids, model_id, prompt)
        if recovered is not None and (recovered["error"] != _UNCONFIRMED or answer["error"] == _UNCONFIRMED):
            if not answer["usage"]["all_steps_reported"] and answer["usage"]["steps_finished"]:
                recovered["usage"]["all_steps_reported"] = False
            answer = recovered
            answer["result_source"] = "session_export"
        else:
            # The preserved projection must identify this exact model and user
            # turn before treating shared-session output as an answer.
            problem = answer["error"]
            answer = _read_events(())
            answer.update(session_id=session_id, result_source="cli_events", error=problem or _UNCONFIRMED)
    elif session_id is not None:
        recovered = _continued_answer(executable, directory, session_id, previous_ids, model_id, prompt)
        if recovered is None:
            # Unattributed session events could describe another client's turn.
            answer = _read_events(())
            answer.update(session_id=session_id, result_source="cli_events")
        else:
            answer = recovered
            answer["result_source"] = "session_export"
    elif answer["error"] == _UNCONFIRMED or answer["http_status"] in (401, 403):
        recovered = _recover_session(executable, directory, title, answer["session_id"], model_id, prompt)
        # An incomplete export can still be the only source of a confirmed
        # subtotal and session ID when the CLI event stream was empty.
        if recovered is not None:
            if answer["http_status"] in (401, 403):
                # A session.error event has no provider/model identity of its own.
                if recovered["http_status"] == answer["http_status"]:
                    answer["http_error_scope"] = recovered["http_error_scope"]
                    if recovered["auth_failure_reason"] is not None:
                        answer["auth_failure_reason"] = recovered["auth_failure_reason"]
                        answer["error"] = recovered["error"]
                    if not answer["usage"]["steps_finished"] and recovered["usage"]["steps_finished"]:
                        answer["usage"] = recovered["usage"]
            elif recovered["error"] != _UNCONFIRMED or answer["session_id"] is None:
                answer = recovered
                answer["result_source"] = "session_export"
    if returncode:
        answer["status"] = "error"
        answer["usage"]["all_steps_reported"] = False
        detail = f"OpenCode terminó con el código {returncode}."
        answer["error"] = f"{detail} {answer['error']}" if answer["error"] else detail
    answer.update(model_id=model_id, returncode=returncode, duration_seconds=time.monotonic() - started,
                  permission_denied=denied)
    return answer
