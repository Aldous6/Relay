"""Durability and ownership of local chat turns across browser sessions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.opencode_history import DEFAULT_PREFERENCES, ChatConflict, ChatHistory, last_project


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"RELAY_CHAT_DB": str(self.root / "history.sqlite3")}))
        self.project = self.root / "Trabajo"
        self.project.mkdir()
        self.choice = {"model": "local/model", "reason": "Prueba"}

    def test_project_preferences_persist_independently_of_transcripts_and_authentication(self):
        history = ChatHistory(self.project)
        other = ChatHistory(self.root)
        turn = history.begin("a", None, "Primer intento", self.choice)
        history.finish("a", turn, {"status": "error", "http_status": 401, "http_error_scope": "selected_model"})
        self.assertIsNone(history.get_preferences())
        limited = {**DEFAULT_PREFERENCES, "provider": "local", "restrict": True,
                   "allowed": ["local/a", "local/b"], "confirm": True}
        unrestricted = DEFAULT_PREFERENCES
        history.save_preferences(limited)
        history.save_preferences(limited)
        self.assertEqual(ChatHistory(self.project).get_preferences(), limited)
        self.assertIsNone(other.get_preferences())
        other.save_preferences(unrestricted)
        self.assertEqual(history.get_preferences(), limited)
        history.save_preferences({**limited, "allowed": []})
        self.assertEqual(history.get_preferences(), {**limited, "allowed": []})
        self.assertEqual(other.get_preferences(), unrestricted)
        self.assertEqual(history.blocked_providers(), {"local": turn})
        self.assertEqual(history.conversations(), [("a", "Primer intento")])
        self.assertEqual(other.conversations(), [])

    def test_legacy_preferences_and_concurrent_free_only_change(self):
        history = ChatHistory(self.project)
        history.save_preferences({**DEFAULT_PREFERENCES, "provider": "local"})
        # Existing rows from before free-only mode consist of four fields.
        with closing(sqlite3.connect(history.path)) as db, db:
            db.execute("UPDATE workspace_preferences SET settings=json_remove(settings, '$.free_only')")
        self.assertFalse(history.get_preferences()["free_only"])
        history.update_preference("free_only", True)
        history.save_preferences({"provider": "local", "restrict": False, "allowed": [], "confirm": False})
        self.assertTrue(history.get_preferences()["free_only"])
        for decision in (self.choice, {**self.choice, "tariff_status": "zero_unverified"},
                         {**self.choice, "model": "other/free", "tariff_status": "advertised_free"}):
            with self.assertRaises(ChatConflict):
                history.begin("stale", None, "Hola", decision)
        self.assertEqual(history.conversations(), [])
        history.begin("valid", None, "Hola", {**self.choice, "tariff_status": "advertised_free"})

    def test_simultaneous_widget_updates_preserve_other_fields(self):
        history = ChatHistory(self.project)
        second_tab = ChatHistory(self.project)
        barrier = Barrier(2)

        def change(tab, field, value):
            barrier.wait(timeout=5)
            tab.update_preference(field, value)

        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(change, history, "allowed", ["local/a"])
            second = workers.submit(change, second_tab, "confirm", True)
            first.result(timeout=10)
            second.result(timeout=10)
        expected = {**DEFAULT_PREFERENCES, "allowed": ["local/a"], "confirm": True}
        self.assertEqual(history.get_preferences(), expected)
        with self.assertRaises(ValueError):
            second_tab.update_preference("confirm", "false")
        self.assertEqual(history.get_preferences(), expected)

    def test_pending_turn_and_stale_tab_cannot_repeat_or_overwrite_a_dispatch(self):
        first, other = ChatHistory(self.project), ChatHistory(self.project)
        turn = first.begin("a", None, "Hola", self.choice)
        self.assertIsNone(other.turns("a")[0]["result"])
        with self.assertRaises(ChatConflict):
            other.begin("a", turn, "Siguiente", self.choice)
        first.finish("a", turn, {"text": "Original"})
        other.finish("a", turn, {"text": "Escritura antigua"})
        self.assertEqual(other.turns("a")[0]["result"]["text"], "Original")
        with self.assertRaises(ChatConflict):
            other.begin("a", None, "Vista desactualizada", self.choice)
        other.begin("a", turn, "Segundo mensaje", self.choice)
        self.assertEqual(len(first.turns("a")), 2)

    def test_reopen_picks_last_used_existing_project_and_separates_transcripts(self):
        self.assertEqual(last_project(self.root), self.root)
        first = ChatHistory(self.project)
        first.begin("a", None, "Primer chat", self.choice)
        self.assertEqual(last_project(self.root), self.project.resolve())
        second_project = self.root / "Proyecto B"
        second_project.mkdir()
        second = ChatHistory(second_project)
        self.assertEqual(second.conversations(), [])
        second.begin("a", None, "Otro chat", self.choice)
        self.assertEqual(last_project(self.root), second_project.resolve())
        self.assertEqual(first.conversations(), [("a", "Primer chat")])
        second_project.rmdir()
        self.assertEqual(last_project(self.root), self.root)

    def test_newer_provider_evidence_wins_and_stale_reenable_cannot_clear_a_new_401(self):
        history = ChatHistory(self.project)
        first = history.begin("a", None, "Primero", self.choice)
        concurrent = history.begin("b", None, "Otro turno", self.choice)
        rejected = {"status": "error", "http_status": 401, "http_error_scope": "selected_model"}
        history.finish("a", first, rejected)
        self.assertEqual(history.blocked_providers(), {"local": first})
        with self.assertRaises(ChatConflict):
            history.begin("c", None, "Acción de otra ventana", self.choice)
        self.assertFalse(history.turns("c"))
        history.finish("b", concurrent, {"status": "completed", "model_id": self.choice["model"]})
        self.assertEqual(history.blocked_providers(), {})
        recent = history.begin("c", None, "Hubo una rotación de clave", self.choice)
        history.finish("c", recent, rejected)
        history.reenable_provider("local", first)
        self.assertEqual(history.blocked_providers(), {"local": recent})
        self.assertEqual(ChatHistory(self.root).blocked_providers(), {})
        history.reenable_provider("local", recent)
        self.assertEqual(history.blocked_providers(), {})
