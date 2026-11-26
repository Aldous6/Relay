"""Local web service contracts; no web server or model provider is started."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router import web_chat
from relay_router.integrations.opencode import OpenCodeError, OpenCodeModel
from relay_router.opencode_history import ChatConflict, ChatHistory, DEFAULT_PREFERENCES


class WebChatTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, {"RELAY_CHAT_DB": str(self.workspace / "chat.sqlite3")}))
        self.service = web_chat.ChatService(self.workspace)
        self.fast = OpenCodeModel("plan/fast-free", "Fast Free", 0, 0, advertised_free=True)
        self.strong = OpenCodeModel("other/advanced", "Advanced", 5, 10, reasoning=True)
        self.catalog = self.enterContext(patch.object(web_chat, "discover_models", return_value=[self.fast, self.strong]))
        self.generate = self.enterContext(patch.object(web_chat, "run_prompt"))
        self.generate.return_value = {"status": "completed", "text": "Respuesta", "session_id": "ses_local"}
        self.args = {"workspace": str(self.workspace), "chat": "chat-a"}

    def request(self, action, **changes):
        return self.service.request(action, {**self.args, **changes})

    def test_state_opens_existing_history_when_cli_is_unavailable(self):
        self.request("send", prompt="Primero")
        self.catalog.side_effect = OpenCodeError("Falta OpenCode.")
        view = self.request("state")
        self.assertEqual(view["catalog_error"], "Falta OpenCode.")
        self.assertEqual(view["models"], [])
        self.assertEqual(view["conversations"], [{"id": "chat-a", "title": "Primero"}])
        self.assertEqual(view["turns"][0]["result"]["text"], "Respuesta")
        settings = {**DEFAULT_PREFERENCES, "provider": "plan"}
        saved = self.request("preferences", preferences=settings)
        self.assertEqual(saved["preferences"], settings)
        self.assertEqual(saved["catalog_error"], "Falta OpenCode.")
        self.generate.assert_called_once()

    def test_preview_rechecks_filters_price_and_confirmed_model(self):
        self.request("preferences", preferences={**DEFAULT_PREFERENCES, "confirm": True, "free_only": True})
        preview = self.request("preview", prompt="Hola")
        self.assertEqual(preview["decision"]["eligible_model_ids"], [self.fast.model_id])
        self.assertFalse(preview["preview_only"])
        with self.assertRaises(ValueError):
            self.request("preview", prompt="Hola", model=self.strong.model_id)
        with self.assertRaises(ChatConflict):
            self.request("send", prompt="Hola")
        replacement = OpenCodeModel("plan/next-free", "Next Free", 0, 0, advertised_free=True)
        self.catalog.return_value = [replacement, self.strong]
        with self.assertRaises(ChatConflict):
            self.request("send", prompt="Hola", expected_model=self.fast.model_id)
        self.generate.assert_not_called()
        self.request("send", prompt="Hola", expected_model=replacement.model_id)
        self.generate.assert_called_once_with("Hola", replacement.model_id, self.workspace)

    def test_display_cache_never_supplies_a_routing_decision(self):
        self.assertEqual(len(self.request("state")["models"]), 2)
        self.catalog.return_value = [self.strong]
        self.assertEqual(len(self.request("state", chat="chat-b")["models"]), 2)
        self.catalog.assert_called_once()
        self.assertEqual(self.request("preview", prompt="Hola")["decision"]["model"], self.strong.model_id)
        self.assertEqual(self.request("state", refresh=True)["models"][0]["id"], self.strong.model_id)

    def test_followups_keep_session_and_duplicate_send_is_rejected(self):
        text = "Primera pregunta\n    conserva la sangría."
        first = self.request("send", prompt=text)
        second = self.request("send", prompt="Otra", previous=first["turn_id"], model=self.strong.model_id)
        with self.assertRaises(ChatConflict):
            self.request("send", prompt="Otra", previous=first["turn_id"])
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.generate.call_args_list[0].args[0], text)
        self.generate.assert_called_with("Otra", self.strong.model_id, self.workspace, session_id="ses_local")
        self.assertEqual(self.request("state")["turns"][-1]["id"], second["turn_id"])

    def test_delete_chat_is_durable_and_scoped_by_workspace_and_chat_id(self):
        first = self.request("send", prompt="Mismo título")
        last = self.request("send", prompt="Seguimiento", previous=first["turn_id"])
        self.request("send", chat="chat-b", prompt="Mismo título")
        other = self.workspace / "Otro"
        other.mkdir()
        self.request("send", workspace=str(other), prompt="De otro proyecto")
        self.catalog.reset_mock()
        self.generate.reset_mock()

        self.assertEqual(self.request("delete-chat"), {"deleted": True})
        self.assertEqual(self.request("delete-chat"), {"deleted": True})
        with self.assertRaises(LookupError):
            self.request("delete-chat", workspace=str(other), chat="chat-b")
        self.catalog.assert_not_called()
        self.service = web_chat.ChatService(self.workspace)
        reopened = self.request("state")
        self.assertNotEqual(reopened["chat"], "chat-a")
        self.assertEqual(reopened["turns"], [])
        view = self.request("state", chat="chat-b")
        self.assertEqual(view["conversations"], [{"id": "chat-b", "title": "Mismo título"}])
        self.assertEqual([turn["prompt"] for turn in view["turns"]], ["Mismo título"])
        self.assertFalse(ChatHistory(self.workspace).turns("chat-a"))
        self.assertEqual(self.request("state", workspace=str(other))["turns"][0]["prompt"], "De otro proyecto")
        for previous in (last["turn_id"], None):
            with self.subTest(previous=previous), self.assertRaises(ChatConflict):
                self.request("send", prompt="Desde una pestaña anterior", previous=previous)
        self.generate.assert_not_called()

    def test_delete_chat_requires_an_explicit_address_and_rejects_pending_work(self):
        choice = {"model": self.fast.model_id, "reason": "Prueba"}
        history = ChatHistory(self.workspace)
        turn = history.begin("chat-a", None, "Pendiente", choice)
        with self.assertRaises(ChatConflict):
            self.request("delete-chat")
        history.finish("chat-a", turn, self.generate.return_value)

        for payload in ({}, {"chat": "chat-a"}, {"workspace": str(self.workspace)}, {**self.args, "chat": ""}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.service.request("delete-chat", payload)
        self.assertEqual(history.turns("chat-a")[0]["result"], self.generate.return_value)

    def test_deleting_rejection_preserves_provider_pause_and_later_reenable(self):
        self.generate.return_value = {"status": "error", "http_status": 401, "http_error_scope": "selected_model"}
        rejected = self.request("send", prompt="Primero", model=self.fast.model_id)["turn_id"]
        self.request("delete-chat")
        self.assertEqual(self.request("state", chat="chat-b")["blocked"], {"plan": rejected})
        self.assertEqual(self.request("reenable", chat="chat-b", provider="plan", through_turn=rejected)["blocked"], {})
        retry = self.request("send", chat="chat-b", prompt="Otro intento", model=self.fast.model_id)["turn_id"]
        self.assertGreater(retry, rejected)
        self.assertEqual(self.request("state", chat="chat-b")["blocked"], {"plan": retry})

    def test_deleting_a_later_success_does_not_restore_an_old_auth_pause(self):
        history = ChatHistory(self.workspace)
        choice = {"model": self.fast.model_id, "reason": "Prueba"}
        earlier = history.begin("other", None, "Turno con rechazo", choice)
        later = history.begin("chat-a", None, "Turno simultáneo", choice)
        history.finish("other", earlier, {"status": "error", "http_status": 401, "http_error_scope": "selected_model"})
        history.finish("chat-a", later, {"status": "completed"})
        self.request("delete-chat")
        self.assertEqual(self.request("state", chat="other")["blocked"], {})

    def test_auth_failure_preview_and_retry_with_original_context(self):
        failure = {"status": "error", "http_status": 401, "http_error_scope": "selected_model", "session_id": "ses_rejected"}
        success = self.generate.return_value
        self.generate.side_effect = [success, failure, success]
        first = self.request("send", prompt="Contexto", model=self.strong.model_id)
        attempt = self.request("send", prompt="Ahora resume", previous=first["turn_id"])
        paused = self.request("preferences", preferences={**DEFAULT_PREFERENCES, "provider": "plan", "confirm": True})
        self.assertEqual(paused["blocked"], {"plan": attempt["turn_id"]})
        self.assertTrue(paused["turns"][-1]["auth_rejected"])
        self.assertTrue(self.request("preview", prompt="Ahora resume")["preview_only"])
        with self.assertRaises(ChatConflict):
            self.request("send", prompt="Ahora resume", previous=attempt["turn_id"], expected_model=self.fast.model_id)
        with self.assertRaises(ChatConflict):
            self.request("reenable", provider="plan", through_turn=attempt["turn_id"] + 1)
        self.request("preferences", preferences=DEFAULT_PREFERENCES)
        self.request("send", prompt="Ahora resume", previous=attempt["turn_id"])
        self.generate.assert_called_with("Ahora resume", self.strong.model_id, self.workspace, session_id="ses_local")
        self.assertEqual(self.request("reenable", provider="plan", through_turn=attempt["turn_id"])["blocked"], {})

    def test_selection_is_rechecked_if_another_window_changes_preferences(self):
        def refresh(_):
            ChatHistory(self.workspace).update_preference("confirm", True)
            return [self.fast]
        self.catalog.side_effect = refresh
        with self.assertRaises(ChatConflict):
            self.request("send", prompt="Enviar")
        self.generate.assert_not_called()
        self.assertEqual(ChatHistory(self.workspace).turns("chat-a"), [])

    def test_transient_storage_failure_recovers_without_resending(self):
        with patch.object(ChatHistory, "finish", side_effect=sqlite3.OperationalError("busy")):
            with self.assertRaisesRegex(ChatConflict, "guardado"):
                self.request("send", prompt="Solo una llamada")
        self.assertEqual(self.request("state")["turns"][0]["result"], self.generate.return_value)
        with self.assertRaises(ChatConflict):
            self.request("send", prompt="Solo una llamada")
        self.generate.assert_called_once()

    def test_projects_and_invalid_requests_stay_isolated(self):
        self.request("send", prompt="Carpeta A")
        other = self.workspace / "otro proyecto"
        other.mkdir()
        view = self.request("state", workspace=str(other))
        self.assertEqual(view["turns"], [])
        self.assertEqual(view["conversations"], [])
        self.assertTrue(self.service.request("state", {})["chat"])
        with self.assertRaises(ValueError):
            self.request("state", workspace="relativa")
        with self.assertRaises(ValueError):
            self.request("preferences", preferences={"confirm": True})
        with self.assertRaises(ValueError):
            self.request("send", prompt="Falta conversación", chat=None)

    def test_missing_session_requires_new_chat(self):
        self.generate.return_value = {"status": "completed", "text": "Sin sesión"}
        turn = self.request("send", prompt="Primero")
        with self.assertRaisesRegex(ChatConflict, "chat nuevo"):
            self.request("send", prompt="Siguiente", previous=turn["turn_id"])
        self.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
