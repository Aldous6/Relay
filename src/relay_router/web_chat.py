"""Application service for Relay's local web chat. No UI or HTTP dependencies."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from threading import Lock
from time import monotonic
from uuid import uuid4

from .integrations.opencode import OpenCodeError, discover_models, run_prompt
from .opencode_history import DEFAULT_PREFERENCES, ChatConflict, ChatHistory, last_project, provider_rejected_auth
from .opencode_selection import advertised_free, select_opencode_model


def _catalog(workspace):
    try:
        return discover_models(workspace)
    except (OSError, ValueError) as exc:
        raise OpenCodeError(f"No se pudo consultar el catálogo de OpenCode. {exc}") from exc


def _route(workspace, settings, blocked, data):
    allowed = set(settings["allowed"])
    candidates = [
        model for model in _catalog(workspace)
        if settings["provider"] in (None, model.model_id.partition("/")[0])
        and (not settings["free_only"] or advertised_free(model))
        and (not settings["restrict"] or model.model_id in allowed)
    ]
    enabled = [model for model in candidates if model.model_id.partition("/")[0] not in blocked]
    preview_only = bool(settings["confirm"] and candidates and not enabled)
    decision = select_opencode_model(data.get("prompt"), candidates if preview_only else enabled,
                                     selected_model=data.get("model"))
    return decision, preview_only


def _continuation(turns, previous):
    last = turns[-1] if turns else None
    if (last["id"] if last else None) != previous or (last and last["result"] is None):
        raise ChatConflict("La conversación cambió o tiene un envío pendiente. Actualiza el historial.")
    if not last:
        return None
    auth_failed = provider_rejected_auth(last["result"], last["model"])
    for turn in reversed(turns):
        result = turn["result"] or {}
        if result.get("session_id") and (not auth_failed or result.get("status") == "completed"):
            return result["session_id"]
    if not auth_failed:
        raise ChatConflict("OpenCode no devolvió una sesión que se pueda continuar. Abre un chat nuevo.")
    return None


class ChatService:
    """Each request names its project and chat, so windows can work independently."""

    def __init__(self, default_workspace: Path):
        self.default_workspace = Path(default_workspace).resolve()
        self._unsaved = {}
        self._write_lock = Lock()
        self._catalog_view = None

    def request(self, action: str, data: dict) -> dict:
        if action not in ("state", "preferences", "reenable", "preview", "send", "delete-chat") or not isinstance(data, dict):
            raise ValueError("Solicitud de chat desconocida.")
        location = data.get("workspace")
        if location is None and action != "delete-chat":
            workspace = last_project(self.default_workspace)
        elif isinstance(location, str) and Path(location).is_absolute():
            workspace = Path(location)
        else:
            raise ValueError("Indica la ruta completa de una carpeta del proyecto.")
        try:
            workspace = workspace.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("La carpeta del proyecto no existe o no se puede abrir.") from exc
        if not workspace.is_dir():
            raise ValueError("Selecciona una carpeta de proyecto existente.")
        chat = data.get("chat")
        if chat is None and action not in ("send", "delete-chat"):
            chat = uuid4().hex
        if not isinstance(chat, str) or not chat.strip() or not chat.isprintable():
            raise ValueError("Se necesita un identificador de conversación válido.")
        history = ChatHistory(workspace)
        self._flush(history)
        if action == "delete-chat":
            history.delete_chat(chat)
            return {"deleted": True}
        if history.is_deleted(chat):
            if action in ("preview", "send"):
                raise ChatConflict("Esta conversación se eliminó. Abre un chat nuevo.")
            chat = uuid4().hex
        if action == "preferences":
            settings = data.get("preferences")
            if not isinstance(settings, dict) or set(settings) != set(DEFAULT_PREFERENCES):
                raise ValueError("Se necesitan todos los ajustes del proyecto.")
            history.save_preferences(settings)
        if action == "reenable":
            provider, through = data.get("provider"), data.get("through_turn")
            if not isinstance(provider, str) or type(through) is not int or through <= 0:
                raise ValueError("Indica el proveedor pausado y el turno que lo pausó.")
            if history.blocked_providers().get(provider) != through:
                raise ChatConflict("La pausa de este proveedor cambió. Actualiza el proyecto.")
            history.reenable_provider(provider, through)

        settings = history.get_preferences() or dict(DEFAULT_PREFERENCES, allowed=[])
        blocked = history.blocked_providers()
        if action in ("state", "preferences", "reenable"):
            return self._state(workspace, chat, history, settings, blocked, data.get("refresh") is True)
        if action == "preview":
            decision, preview_only = _route(workspace, settings, blocked, data)
            return {"decision": decision, "preview_only": preview_only}
        return self._send(workspace, chat, history, settings, blocked, data)

    def _view_models(self, workspace, refresh):
        # One immutable snapshot for navigation. Routing always queries OpenCode.
        cached = self._catalog_view
        if not refresh and cached and cached[0] == workspace and monotonic() < cached[1]:
            return cached[2], cached[3]
        error = None
        try:
            catalog = _catalog(workspace)
        except OpenCodeError as exc:
            catalog, error = [], str(exc)
        self._catalog_view = (workspace, monotonic() + 20, catalog, error)
        return catalog, error

    def _state(self, workspace, chat, history, settings, blocked, refresh):
        catalog, error = self._view_models(workspace, refresh)
        turns = history.turns(chat)
        for turn in turns:
            turn["auth_rejected"] = provider_rejected_auth(turn["result"], turn["model"])
        return {
            "workspace": str(workspace), "chat": chat, "preferences": settings,
            "models": [{"id": m.model_id, "name": m.name, "free": advertised_free(m)} for m in catalog],
            "blocked": blocked, "catalog_error": error, "turns": turns,
            "conversations": [{"id": cid, "title": title} for cid, title in history.conversations()],
        }

    def _flush(self, history):
        # If SQLite was temporarily busy after generation, retry only the write.
        # The pending turn in SQLite still prevents another model invocation.
        with self._write_lock:
            for key, result in list(self._unsaved.items()):
                workspace, chat, turn_id = key
                if workspace == history.workspace:
                    history.finish(chat, turn_id, result)
                    del self._unsaved[key]

    def _send(self, workspace, chat, history, settings, blocked, data):
        previous = data.get("previous")
        if previous is not None and (type(previous) is not int or previous <= 0):
            raise ValueError("El identificador del turno anterior es inválido.")
        session = _continuation(history.turns(chat), previous)
        decision, preview_only = _route(workspace, settings, blocked, data)
        if preview_only:
            raise ChatConflict("Los proveedores seleccionados están en pausa. Reconecta uno antes de enviar.")
        model = decision["model"]
        expected = data.get("expected_model")
        if (expected is not None or settings["confirm"]) and expected != model:
            raise ChatConflict("Revisa el modelo propuesto y confirma antes de enviar.")
        prompt = data["prompt"]
        turn_id = history.begin(chat, previous, prompt, decision, expected_preferences=settings)
        try:
            continuation = {"session_id": session} if session else {}
            result = run_prompt(prompt, model, workspace, **continuation)
        except (OpenCodeError, ValueError, OSError) as exc:
            result = {"status": "error", "text": "", "error": str(exc)}
        try:
            history.finish(chat, turn_id, result)
        except sqlite3.Error as exc:
            with self._write_lock:
                self._unsaved[(history.workspace, chat, turn_id)] = result
            raise ChatConflict("Relay recibió el resultado. Actualiza el historial para reintentar el guardado.") from exc
        return {"turn_id": turn_id, "result": result}
