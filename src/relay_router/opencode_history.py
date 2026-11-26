"""Small, per-user transcript store for the local Relay chat."""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3


class ChatConflict(Exception):
    """Another tab changed or is already writing this conversation."""


class ChatNotFound(LookupError):
    """The requested chat was never saved in this workspace."""


_LEGACY_AUTH_ERROR = "APIError (HTTP 401). revisa la autenticación del proveedor."
DEFAULT_PREFERENCES = {"provider": None, "restrict": False, "allowed": [], "confirm": False, "free_only": False}


def provider_rejected_auth(result, model):
    if not result or result.get("status") != "error":
        return False
    if "http_status" in result:
        return result["http_status"] == 401 and result.get("http_error_scope") == "selected_model"
    # Prior releases stored this exact adapter-generated diagnostic in the
    # verified session export. Assistant or tool text is never evidence here.
    return (result.get("error") == _LEGACY_AUTH_ERROR and result.get("result_source") == "session_export"
            and result.get("model_id") == model)


def history_path() -> Path:
    if location := os.environ.get("RELAY_CHAT_DB"):
        return Path(location).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return base / "Relay" / "chat.sqlite3"


def last_project(default: Path) -> Path:
    path = history_path()
    if path.is_file():
        try:
            with closing(sqlite3.connect(path, timeout=2)) as db:
                used_projects = "SELECT workspace, id FROM turns"
                # On the first launch after upgrading, ChatHistory may not have
                # created deleted_chats yet.
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='deleted_chats'").fetchone():
                    used_projects += " UNION ALL SELECT workspace, last_turn AS id FROM deleted_chats"
                row = db.execute(used_projects + " ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                candidate = Path(row[0]).resolve(strict=True)
                if candidate.is_dir():
                    return candidate
        except (OSError, RuntimeError, sqlite3.Error):
            pass
    return default


class ChatHistory:
    def __init__(self, workspace: Path):
        self.workspace = os.path.normcase(str(workspace))
        self.path = history_path()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, chat TEXT NOT NULL,
                prompt TEXT NOT NULL, model TEXT NOT NULL, reason TEXT NOT NULL, result TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS turns_by_chat ON turns(workspace, chat, id)")
            db.execute("""CREATE TABLE IF NOT EXISTS auth_resets (
                workspace TEXT NOT NULL, provider TEXT NOT NULL, through_turn INTEGER NOT NULL,
                PRIMARY KEY(workspace, provider)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS deleted_chats (
                last_turn INTEGER PRIMARY KEY, workspace TEXT NOT NULL, chat TEXT NOT NULL,
                UNIQUE(workspace, chat)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS retained_auth_pauses (
                workspace TEXT NOT NULL, provider TEXT NOT NULL, through_turn INTEGER NOT NULL,
                PRIMARY KEY(workspace, provider)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS workspace_preferences (
                workspace TEXT PRIMARY KEY, settings TEXT NOT NULL CHECK(json_valid(settings))
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def get_preferences(self) -> dict | None:
        with closing(self._connect()) as db:
            return self._get_preferences(db)

    def _get_preferences(self, db):
        row = db.execute("SELECT settings FROM workspace_preferences WHERE workspace = ?",
                         (self.workspace,)).fetchone()
        return {**DEFAULT_PREFERENCES, **json.loads(row["settings"])} if row else None

    def _write_preferences(self, db, preferences):
        settings = {key: preferences[key] for key in DEFAULT_PREFERENCES}
        if settings["provider"] is not None and not isinstance(settings["provider"], str):
            raise ValueError("El proveedor debe ser un identificador de OpenCode.")
        if not all(isinstance(settings[field], bool) for field in ("restrict", "confirm", "free_only")):
            raise ValueError("Los filtros de modelos y la confirmación deben ser booleanos.")
        if not isinstance(settings["allowed"], list) or not all(isinstance(model, str) for model in settings["allowed"]):
            raise ValueError("Los modelos permitidos deben ser una lista de identificadores de OpenCode.")
        db.execute("""INSERT INTO workspace_preferences (workspace, settings) VALUES (?, ?)
                   ON CONFLICT(workspace) DO UPDATE SET settings = excluded.settings
                   WHERE workspace_preferences.settings != excluded.settings""",
                   (self.workspace, json.dumps(settings, ensure_ascii=False)))

    def save_preferences(self, preferences: dict) -> None:
        with closing(self._connect()) as db, db:
            if "free_only" not in preferences:
                db.execute("BEGIN IMMEDIATE")
                previous = self._get_preferences(db) or DEFAULT_PREFERENCES
                preferences = {**preferences, "free_only": previous["free_only"]}
            self._write_preferences(db, preferences)

    def update_preference(self, field: str, value) -> None:
        if field not in DEFAULT_PREFERENCES:
            raise ValueError("Preferencia desconocida.")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            settings = self._get_preferences(db) or DEFAULT_PREFERENCES
            self._write_preferences(db, {**settings, field: value})

    def conversations(self):
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT first.chat, first.prompt FROM turns AS first
                JOIN (SELECT MIN(id) AS first_id, MAX(id) AS last_id FROM turns
                      WHERE workspace = ? GROUP BY chat) AS thread ON first.id = thread.first_id
                ORDER BY thread.last_id DESC""", (self.workspace,))
            return [(row["chat"], " ".join(row["prompt"].split())[:72]) for row in rows]

    def turns(self, chat):
        with closing(self._connect()) as db:
            rows = db.execute("SELECT id, prompt, model, reason, result FROM turns "
                              "WHERE workspace = ? AND chat = ? ORDER BY id", (self.workspace, chat))
            return [{**row, "result": json.loads(row["result"]) if row["result"] else None} for row in rows]

    def _is_deleted(self, db, chat):
        return db.execute("SELECT 1 FROM deleted_chats WHERE workspace = ? AND chat = ?",
                          (self.workspace, chat)).fetchone() is not None

    def is_deleted(self, chat):
        with closing(self._connect()) as db:
            return self._is_deleted(db, chat)

    def delete_chat(self, chat):
        address = (self.workspace, chat)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if self._is_deleted(db, chat):
                return
            last_id, pending = db.execute(
                "SELECT MAX(id), COUNT(*) - COUNT(result) FROM turns WHERE workspace = ? AND chat = ?",
                address).fetchone()
            if last_id is None:
                raise ChatNotFound("Esta conversación no existe en el historial del proyecto.")
            if pending:
                raise ChatConflict("Hay un envío pendiente. Espera a que termine antes de eliminar la conversación.")

            # Removing a transcript must neither clear a project-wide 401 pause
            # nor revive a pause superseded by a successful turn in this chat.
            for provider, turn_id in self._blocked(db).items():
                db.execute("""INSERT INTO retained_auth_pauses (workspace, provider, through_turn)
                    SELECT workspace, ?, id FROM turns WHERE id = ? AND workspace = ? AND chat = ?
                    ON CONFLICT(workspace, provider) DO UPDATE SET through_turn = excluded.through_turn""",
                           (provider, turn_id, *address))
            db.execute("""INSERT INTO auth_resets (workspace, provider, through_turn)
                SELECT workspace, substr(model, 1, instr(model, '/') - 1) AS provider, MAX(id)
                FROM turns WHERE workspace = ? AND chat = ? AND json_extract(result, '$.status') = 'completed'
                GROUP BY workspace, provider
                ON CONFLICT(workspace, provider) DO UPDATE SET through_turn =
                    MAX(auth_resets.through_turn, excluded.through_turn)""", address)
            db.execute("INSERT INTO deleted_chats (last_turn, workspace, chat) VALUES (?, ?, ?)", (last_id, *address))
            db.execute("DELETE FROM turns WHERE workspace = ? AND chat = ?", address)

    def _blocked(self, db):
        resets = dict(db.execute("SELECT provider, through_turn FROM auth_resets WHERE workspace = ?", (self.workspace,)))
        retained = db.execute("SELECT provider, through_turn FROM retained_auth_pauses WHERE workspace = ?",
                              (self.workspace,))
        blocked = {provider: through for provider, through in retained if through > resets.get(provider, 0)}
        # Project only the diagnostic fields, never transcripts or tool output.
        rows = db.execute("""SELECT id, model, json_type(result, '$.http_status') AS has_status,
                   json_extract(result, '$.http_status') AS http_status,
                   json_extract(result, '$.http_error_scope') AS http_error_scope,
                   json_extract(result, '$.result_source') AS result_source,
                   json_extract(result, '$.model_id') AS model_id, json_extract(result, '$.error') AS error
               FROM turns WHERE workspace = ? AND json_extract(result, '$.status') = 'error'
               AND (json_extract(result, '$.http_status') = 401 OR json_extract(result, '$.error') = ?)""",
                          (self.workspace, _LEGACY_AUTH_ERROR))
        for row in rows:
            model = row["model"]
            provider = model.partition("/")[0]
            if row["id"] <= max(resets.get(provider, 0), blocked.get(provider, 0)):
                continue
            diagnostic = dict(row)
            diagnostic["status"] = "error"
            if row["has_status"] is None:
                del diagnostic["http_status"]
            if provider_rejected_auth(diagnostic, model):
                blocked[provider] = row["id"]
        if blocked:
            # A later successful request in an older client or an already
            # running tab is newer evidence for that provider in this project.
            successes = db.execute("""SELECT substr(model, 1, instr(model, '/') - 1) AS provider, max(id)
                FROM turns WHERE workspace = ? AND id > ? AND json_extract(result, '$.status') = 'completed'
                GROUP BY provider""", (self.workspace, min(blocked.values())))
            for provider, accepted_turn in successes:
                if provider in blocked and accepted_turn > blocked[provider]:
                    del blocked[provider]
        return blocked

    def blocked_providers(self):
        with closing(self._connect()) as db:
            return self._blocked(db)

    def reenable_provider(self, provider, through_turn):
        with closing(self._connect()) as db, db:
            db.execute("""INSERT INTO auth_resets (workspace, provider, through_turn) VALUES (?, ?, ?)
                       ON CONFLICT(workspace, provider) DO UPDATE SET through_turn =
                       MAX(auth_resets.through_turn, excluded.through_turn)""", (self.workspace, provider, through_turn))

    def begin(self, chat, previous_id, prompt, decision, *, expected_preferences=None):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if self._is_deleted(db, chat):
                raise ChatConflict("Esta conversación se eliminó. Abre un chat nuevo.")
            settings = self._get_preferences(db)
            if expected_preferences is not None and (settings or DEFAULT_PREFERENCES) != expected_preferences:
                raise ChatConflict("Los ajustes del proyecto cambiaron. Revisa la selección antes de enviar.")
            if settings and settings["free_only"]:
                provider = decision["model"].partition("/")[0]
                if (decision.get("tariff_status") != "advertised_free"
                        or settings["provider"] not in (None, provider)
                        or (settings["restrict"] and decision["model"] not in settings["allowed"])):
                    raise ChatConflict("La selección cambió. Revisa los modelos gratuitos disponibles para este proyecto.")
            if decision["model"].partition("/")[0] in self._blocked(db):
                raise ChatConflict("El proveedor acaba de rechazar la autenticación. Actualiza el chat para elegir otra ruta.")
            last = db.execute("SELECT id, result FROM turns WHERE workspace = ? AND chat = ? "
                              "ORDER BY id DESC LIMIT 1", (self.workspace, chat)).fetchone()
            if (last and (last["result"] is None or last["id"] != previous_id)) or (not last and previous_id is not None):
                raise ChatConflict("Otra ventana ha actualizado este chat. Ábrelo de nuevo desde el historial.")
            # Legacy turns use INTEGER PRIMARY KEY. After deleting the latest
            # transcript, SQLite could otherwise reuse a stale turn or pause ID.
            inserted = db.execute("""INSERT INTO turns (id, workspace, chat, prompt, model, reason)
                SELECT 1 + MAX(IFNULL((SELECT MAX(id) FROM turns), 0),
                               IFNULL((SELECT MAX(last_turn) FROM deleted_chats), 0)), ?, ?, ?, ?, ?""",
                                  (self.workspace, chat, prompt, decision["model"], decision["reason"]))
            return inserted.lastrowid

    def finish(self, chat, turn_id, result):
        with closing(self._connect()) as db, db:
            db.execute("UPDATE turns SET result = ? WHERE workspace = ? AND chat = ? AND id = ? AND result IS NULL",
                       (json.dumps(result, ensure_ascii=False), self.workspace, chat, turn_id))
