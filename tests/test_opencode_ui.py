"""Exercise the chat UI without a web server, user history or provider traffic."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router import opencode_ui
from relay_router.integrations.opencode import OpenCodeError, OpenCodeModel
from relay_router.opencode_history import ChatHistory


def reply(model, session, text="", *, error=None):
    return {
        "status": "error" if error else "completed", "model_id": model, "session_id": session,
        "text": text, "error": error, "permission_denied": False, "tool_calls": [],
        "duration_seconds": .3, "usage": {"input_tokens": None, "output_tokens": None,
                                        "reported_cost_usd": None, "cost_usd": None},
    }


class OpenCodeAppTests(unittest.TestCase):
    def setUp(self):
        temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.workspace = temp / "project"
        self.workspace.mkdir()
        self.enterContext(patch.dict(os.environ, {"RELAY_CHAT_DB": str(temp / "history.sqlite3")}))
        self.cheap = "team/small"
        self.reasoner = "plan/reasoner"
        catalog = [
            OpenCodeModel(self.cheap, "Small", .1, .2, context_tokens=32000,
                          output_tokens=4096, tool_call=True, reasoning=False),
            OpenCodeModel(self.reasoner, "Reasoner", .5, 1., context_tokens=64000,
                          output_tokens=8192, tool_call=True, reasoning=True),
        ]
        self.discover = self.enterContext(patch.object(opencode_ui, "discover_models", return_value=catalog))
        self.dispatch = self.enterContext(patch.object(opencode_ui, "run_prompt"))
        self.app = self.new_app()

    def new_app(self):
        # AppTest owns argv. Enter exactly as Start-Relay.ps1 -OpenCode would.
        app_file = str(ROOT / "app/relay_app.py")
        harness = ("import runpy, sys\nfrom unittest.mock import patch\n"
                   f"with patch.object(sys, 'argv', [{app_file!r}, '--opencode']):\n"
                   f"    runpy.run_path({app_file!r}, run_name='__main__')\n")
        app = AppTest.from_string(harness, default_timeout=15)
        app.session_state["opencode_directory"] = str(self.workspace)
        return app

    def send(self, text):
        self.app.chat_input(key="opencode_chat_input").set_value(text).run()
        self.assertFalse(self.app.exception)

    def open_chat(self, title_prefix):
        previous = next(b for b in self.app.sidebar.button
                        if (b.key or "").startswith("opencode_history_") and title_prefix in b.label)
        previous.click().run()
        self.assertFalse(self.app.exception)

    def content(self):
        return "\n".join(item.value for item in self.app.markdown)

    def retry_buttons(self):
        return [b for b in self.app.button if (b.key or "").startswith("opencode_retry_auth_") and not b.disabled]

    def test_direct_chat_routes_each_turn_and_preserves_session_on_rerun(self):
        first = "Diagnostica este fallo entre archivos.\n    Conserva la sangría."
        next_turn = "Resume la solución."
        self.dispatch.side_effect = [
            reply(self.reasoner, "ses_first", "El importador omite la última fila."),
            reply(self.cheap, "ses_first", "Procesa también la última fila."),
        ]
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.chat_input(key="opencode_chat_input").disabled)
        self.dispatch.assert_not_called()
        self.send(first)
        self.dispatch.assert_called_once_with(first, self.reasoner, self.workspace)
        self.assertIn("El importador omite la última fila.", self.content())
        self.app.run()
        self.dispatch.assert_called_once()
        self.send(next_turn)
        self.assertEqual(self.dispatch.call_args_list, [
            call(first, self.reasoner, self.workspace),
            call(next_turn, self.cheap, self.workspace, session_id="ses_first"),
        ])
        self.assertIn("El importador omite la última fila.", self.content())
        self.assertIn("Procesa también la última fila.", self.content())
        self.discover.assert_called_once_with(self.workspace)

    def test_new_chat_is_separate_and_prior_history_can_be_reopened(self):
        self.dispatch.side_effect = [
            reply(self.cheap, "ses_alpha", "Respuesta exclusiva de Alfa"),
            reply(self.cheap, "ses_beta", "Respuesta exclusiva de Beta"),
            reply(self.cheap, "ses_alpha", "Segunda respuesta de Alfa"),
        ]
        self.app.run()
        self.send("Proyecto Alfa")
        self.app.sidebar.button(key="opencode_new_chat").click().run()
        self.assertNotIn("Respuesta exclusiva de Alfa", self.content())
        self.send("Proyecto Beta")
        self.assertEqual(self.dispatch.call_args, call("Proyecto Beta", self.cheap, self.workspace))
        self.app = self.new_app().run()
        self.open_chat("Proyecto Alfa")
        self.assertIn("Respuesta exclusiva de Alfa", self.content())
        self.assertNotIn("Respuesta exclusiva de Beta", self.content())
        self.assertEqual(self.dispatch.call_count, 2)
        self.send("Sigue con ese proyecto")
        self.assertEqual(self.dispatch.call_args,
                         call("Sigue con ese proyecto", self.cheap, self.workspace, session_id="ses_alpha"))

    def test_switching_projects_isolates_history_and_cancels_the_pending_draft(self):
        other = self.workspace.parent / "other"
        other.mkdir()
        self.dispatch.return_value = reply(self.cheap, "ses_local", "Respuesta privada del proyecto")
        self.app.run()
        self.send("Trabajo local")
        self.app.checkbox(key="opencode_confirm_model").check().run()
        self.send("Siguiente mensaje para el primer proyecto")
        self.assertFalse(self.app.button(key="opencode_confirm_send").disabled)
        self.app.text_input(key="opencode_directory").set_value(str(other)).run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("Respuesta privada del proyecto", self.content())
        self.assertFalse(any((b.key or "").startswith("opencode_history_") for b in self.app.sidebar.button))
        self.assertFalse(any(b.key == "opencode_confirm_send" for b in self.app.button))
        self.assertEqual(self.discover.call_args, call(other))
        self.app.text_input(key="opencode_directory").set_value(str(self.workspace)).run()
        self.open_chat("Trabajo local")
        self.assertIn("Respuesta privada del proyecto", self.content())
        self.dispatch.assert_called_once()

    def test_provider_rejection_survives_reruns_without_retry(self):
        self.dispatch.return_value = reply(self.cheap, "ses_rejected", error="HTTP 403 acceso denegado")
        self.app.run()
        self.send("Resume este README")
        self.assertTrue(any("403" in item.value for item in self.app.error))
        self.app.run()
        self.assertTrue(any("403" in item.value for item in self.app.error))
        self.dispatch.assert_called_once_with("Resume este README", self.cheap, self.workspace)

    def test_saved_legacy_401_can_retry_exact_prompt_once_and_keeps_both_cost_records(self):
        prompt = "Diseña la landing de Relay.\n    Conserva estos requisitos."
        prior = reply(self.cheap, "ses_denied",
                      error="APIError (HTTP 401). revisa la autenticación del proveedor.")
        prior["result_source"] = "session_export"
        history = ChatHistory(self.workspace)
        prior_id = history.begin("legacy", None, prompt, {"model": self.cheap, "reason": "Selección anterior"})
        history.finish("legacy", prior_id, prior)
        success = reply(self.reasoner, "ses_recovered", "Propuesta recuperada")
        success["usage"]["reported_cost_usd"] = .025
        self.dispatch.return_value = success
        self.app.run()
        self.open_chat("Diseña la landing")
        self.app.checkbox(key="opencode_confirm_model").check().run()
        self.app.run()
        button, = self.retry_buttons()
        self.assertIn(self.reasoner, button.label)
        self.dispatch.assert_not_called()
        button.click().run()
        self.assertFalse(self.app.exception)
        self.app.run()
        self.dispatch.assert_called_once_with(prompt, self.reasoner, self.workspace)
        self.assertTrue(any("401" in item.value for item in self.app.error))
        self.assertIn("Propuesta recuperada", self.content())
        rejected, recovered = history.turns("legacy")
        self.assertEqual((rejected["id"], rejected["result"]), (prior_id, prior))
        self.assertEqual((recovered["prompt"], recovered["model"]), (prompt, self.reasoner))
        self.assertEqual(recovered["result"]["usage"]["reported_cost_usd"], .025)
        self.assertIsNone(rejected["result"]["usage"]["reported_cost_usd"])
        self.assertIsNone(rejected["result"]["usage"]["cost_usd"])

    def test_structured_401_respects_provider_filter_and_blocks_siblings_after_reopen(self):
        alternate = OpenCodeModel("team/even-cheaper", "Sibling", .09, .12, context_tokens=32000,
                                  output_tokens=4096, tool_call=True, reasoning=False)
        self.discover.return_value = [*self.discover.return_value, alternate]
        denial = reply(alternate.model_id, None, error="Credenciales rechazadas")
        denial.update(http_status=401, http_error_scope="selected_model")
        self.dispatch.side_effect = [denial, reply(self.reasoner, "ses_other", "Con otro proveedor"),
                                     reply(alternate.model_id, "ses_reenabled", "Acceso restablecido")]
        self.app.run()
        self.app.selectbox(key="opencode_provider").select("team").run()
        self.send("Primera consulta")
        self.dispatch.assert_called_once_with("Primera consulta", alternate.model_id, self.workspace)
        self.assertFalse(self.retry_buttons())
        self.app.button(key="opencode_refresh").click().run()
        self.dispatch.assert_called_once()
        self.app = self.new_app().run()
        self.app.sidebar.button(key="opencode_new_chat").click().run()
        self.dispatch.assert_called_once()
        self.assertEqual(self.app.selectbox(key="opencode_provider").value, "team")
        self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
        self.app.selectbox(key="opencode_provider").select("plan").run()
        self.send("Otra consulta corta")
        self.assertEqual(self.dispatch.call_args, call("Otra consulta corta", self.reasoner, self.workspace))
        self.assertEqual(self.dispatch.call_count, 2)
        self.app.button(key="opencode_reenable_team").click().run()
        self.assertFalse(self.app.exception)
        self.app.sidebar.button(key="opencode_new_chat").click().run()
        self.assertEqual(self.dispatch.call_count, 2)
        self.app.selectbox(key="opencode_provider").select("team").run()
        self.send("Verifica el acceso")
        self.assertEqual(self.dispatch.call_args, call("Verifica el acceso", alternate.model_id, self.workspace))

    def test_auth_retry_preserves_a_previously_completed_conversation(self):
        previous = reply(self.reasoner, "ses_work", "Contexto de la tarea")
        denial = reply(self.cheap, "ses_work", error="Credenciales rechazadas")
        denial.update(http_status=401, http_error_scope="selected_model")
        self.dispatch.side_effect = [previous, denial, reply(self.reasoner, "ses_work", "Respuesta recuperada")]
        self.app.run()
        self.send("Diagnostica el fallo de concurrencia")
        prompt = "Resúmelo en dos frases."
        self.send(prompt)
        # Use the simple model for the second turn so its 401 leaves the previous
        # successful provider eligible when the user asks to retry.
        self.assertEqual(self.dispatch.call_args, call(prompt, self.cheap, self.workspace, session_id="ses_work"))
        retry, = self.retry_buttons()
        retry.click().run()
        self.assertFalse(self.app.exception)
        self.app.run()
        self.assertEqual(self.dispatch.call_args, call(prompt, self.reasoner, self.workspace, session_id="ses_work"))
        self.assertEqual(self.dispatch.call_count, 3)
        self.assertIn("Contexto de la tarea", self.content())

    def test_401_without_selected_model_attribution_does_not_block_the_provider(self):
        denial = reply(self.cheap, "ses_partial",
                       error="APIError (HTTP 401). revisa la autenticación del proveedor.")
        denial.update(http_status=401, http_error_scope="session")
        self.dispatch.side_effect = [denial, reply(self.cheap, "ses_other", "Respuesta válida")]
        self.app.run()
        self.send("Primera consulta")
        self.assertFalse(self.retry_buttons())
        self.app.sidebar.button(key="opencode_new_chat").click().run()
        self.send("Otra consulta")
        self.assertEqual(self.dispatch.call_args_list, [
            call("Primera consulta", self.cheap, self.workspace), call("Otra consulta", self.cheap, self.workspace),
        ])

    def test_optional_preview_only_sends_after_confirmation(self):
        self.dispatch.return_value = reply(self.reasoner, "ses_preview", "Revisión terminada")
        self.app.run()
        self.app.checkbox(key="opencode_confirm_model").check().run()
        self.send("Resumen que voy a cancelar")
        self.assertFalse(self.app.button(key="opencode_confirm_send").disabled)
        self.dispatch.assert_not_called()
        self.app.button(key="opencode_confirm_cancel").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.chat_input(key="opencode_chat_input").proto.value, "Resumen que voy a cancelar")
        self.dispatch.assert_not_called()
        prompt = "Diagnostica el fallo de concurrencia."
        self.send(prompt)
        self.app.run()
        self.dispatch.assert_not_called()
        self.app.button(key="opencode_confirm_send").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("Revisión terminada", self.content())
        self.app.run()
        self.dispatch.assert_called_once_with(prompt, self.reasoner, self.workspace)

    def test_starter_prepares_an_editable_draft_without_sending(self):
        self.app.run()
        self.app.button(key="opencode_starter_2").click().run()
        self.assertIn("diagnosticar", self.app.chat_input(key="opencode_chat_input").proto.value)
        self.dispatch.assert_not_called()
        self.dispatch.return_value = reply(self.cheap, "ses_edited", "Respuesta")
        self.send("Mi propia pregunta")
        self.dispatch.assert_called_once_with("Mi propia pregunta", self.cheap, self.workspace)

    def test_history_search_filters_titles_without_switching_the_active_chat(self):
        history = ChatHistory(self.workspace)
        for index in range(6):
            chat = f"chat-{index}"
            prompt = f"Consulta número {index}"
            turn = history.begin(chat, None, prompt, {"model": self.cheap, "reason": "Selección"})
            history.finish(chat, turn, reply(self.cheap, f"ses_{index}", f"Respuesta {index}"))
        self.app.run()
        self.app.sidebar.button(key="opencode_history_chat-0").click().run()
        self.app.text_input(key="opencode_history_search").set_value("NÚMERO 5").run()
        visible = [b.key for b in self.app.sidebar.button if (b.key or "").startswith("opencode_history_")]
        self.assertEqual(visible, ["opencode_history_chat-5"])
        self.assertIn("Respuesta 0", self.content())
        self.app.sidebar.button(key=visible[0]).click().run()
        self.assertIn("Respuesta 5", self.content())
        self.dispatch.assert_not_called()

    def test_filters_and_failed_catalog_refresh_control_sending(self):
        self.app.run()
        self.app.selectbox(key="opencode_provider").select("plan").run()
        self.app.checkbox(key="opencode_restrict").check().run()
        self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
        self.app.multiselect(key="opencode_allowed").select(self.reasoner).run()
        self.assertFalse(self.app.chat_input(key="opencode_chat_input").disabled)
        self.dispatch.return_value = reply(self.reasoner, "ses_limited", "Respuesta filtrada")
        self.send("Explica el código.")
        self.dispatch.assert_called_once_with("Explica el código.", self.reasoner, self.workspace)
        self.discover.side_effect = OpenCodeError("Catálogo inaccesible")
        self.app.button(key="opencode_refresh").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Catálogo inaccesible" in msg.value for msg in self.app.error))
        self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
        self.assertEqual(self.dispatch.call_count, 1)

    def test_free_catalog_includes_new_models_and_rechecks_before_confirming(self):
        history = ChatHistory(self.workspace)
        history.save_preferences({"provider": "opencode", "restrict": False, "allowed": ["opencode/old-free"],
                                  "confirm": True, "free_only": True})
        first = OpenCodeModel("opencode/a-free", "A Free", 0, 0, context_tokens=4096)
        second = replace(first, model_id="opencode/b-free", context_tokens=8192)
        new = replace(first, model_id="opencode/new-free", context_tokens=2048)
        paid = replace(first, model_id="opencode/paid-free", output_usd_per_million=.5)
        unknown = replace(first, model_id="opencode/unknown", name="Unknown")
        other = replace(first, model_id="opencode-go/other-free")
        self.discover.return_value = [first, second, paid, unknown, other]
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.checkbox(key="opencode_free_only").value)
        self.assertFalse(self.app.checkbox(key="opencode_restrict").value)
        self.assertIn("Modelos gratuitos del catálogo (2)", [e.label for e in self.app.expander])
        self.discover.return_value.append(new)
        prompt = "Un titular breve."
        self.send(prompt)
        self.assertIn(new.model_id, self.content())
        self.assertIn("Modelos gratuitos del catálogo (3)", [e.label for e in self.app.expander])
        self.discover.return_value = [first, second, paid, unknown, other,
                                      replace(new, output_usd_per_million=.1)]
        self.app.button(key="opencode_confirm_send").click().run()
        self.assertFalse(self.app.exception)
        self.dispatch.assert_not_called()
        self.assertIn(first.model_id, self.content())
        self.dispatch.return_value = reply(first.model_id, "ses_free", "Respuesta")
        self.app.button(key="opencode_confirm_send").click().run()
        self.assertFalse(self.app.exception)
        self.dispatch.assert_called_once_with(prompt, first.model_id, self.workspace)
        self.discover.return_value = [paid, unknown, other]
        self.send("¿Sigue gratuito?")
        self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
        self.assertEqual(self.dispatch.call_count, 1)

    def test_manual_free_choice_is_per_message_and_cannot_send_after_a_price_change(self):
        history = ChatHistory(self.workspace)
        history.save_preferences({"provider": "opencode", "restrict": False, "allowed": [],
                                  "confirm": True, "free_only": True})
        first = OpenCodeModel("opencode/a-free", "A Free", 0, 0, context_tokens=4096)
        manual = replace(first, model_id="opencode/b-free", context_tokens=8192)
        ineligible = replace(first, model_id="opencode/no-tools-free", tool_call=False)
        self.discover.return_value = [first, manual, ineligible]
        self.app.run()
        self.send("Un titular.")
        choices = self.app.selectbox(key="opencode_manual_model")
        self.assertEqual(choices.options, [f"Automático · {first.model_id}", first.model_id, manual.model_id])
        choices.select(manual.model_id).run()
        self.discover.return_value = [first, replace(manual, input_usd_per_million=1)]
        self.app.button(key="opencode_confirm_send").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.button(key="opencode_confirm_send").disabled)
        self.dispatch.assert_not_called()
        self.discover.return_value = [first, manual]
        self.app.button(key="opencode_refresh").click().run()
        self.dispatch.return_value = reply(manual.model_id, "ses_pick", "Titular")
        self.app.button(key="opencode_confirm_send").click().run()
        self.assertFalse(self.app.exception)
        self.dispatch.assert_called_once_with("Un titular.", manual.model_id, self.workspace)
        turn, = history.turns(history.conversations()[0][0])
        self.assertIn("Elegiste", turn["reason"])
        self.send("Otro titular.")
        self.assertIsNone(self.app.selectbox(key="opencode_manual_model").value)
        self.assertIn(first.model_id, self.content())
        self.assertFalse(history.get_preferences()["restrict"])

    def test_loads_saved_filters_and_keeps_edits_across_sessions_and_projects(self):
        sibling = OpenCodeModel("team/large", "Large")
        self.discover.return_value = [*self.discover.return_value, sibling]
        prefs = {"provider": "team", "restrict": True, "allowed": [self.cheap], "confirm": True}
        ChatHistory(self.workspace).save_preferences(prefs)
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.selectbox(key="opencode_provider").value, "team")
        self.assertTrue(self.app.checkbox(key="opencode_restrict").value)
        self.assertTrue(self.app.checkbox(key="opencode_confirm_model").value)
        self.assertEqual(self.app.multiselect(key="opencode_allowed").value, [self.cheap])
        self.app.multiselect(key="opencode_allowed").select(sibling.model_id).run()
        self.app.checkbox(key="opencode_confirm_model").uncheck().run()

        self.app = self.new_app().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.checkbox(key="opencode_confirm_model").value)
        self.assertEqual(self.app.multiselect(key="opencode_allowed").value, [self.cheap, sibling.model_id])
        other = self.workspace.parent / "other"
        other.mkdir()
        self.app.text_input(key="opencode_directory").set_value(str(other)).run()
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.app.selectbox(key="opencode_provider").value)
        self.assertFalse(self.app.checkbox(key="opencode_restrict").value)
        self.app.checkbox(key="opencode_confirm_model").check().run()
        self.app.text_input(key="opencode_directory").set_value(str(self.workspace)).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.multiselect(key="opencode_allowed").value, [self.cheap, sibling.model_id])
        self.assertFalse(self.app.checkbox(key="opencode_confirm_model").value)
        self.dispatch.assert_not_called()

    def test_paused_provider_can_preview_two_models_without_unblocking_or_dispatching(self):
        large = OpenCodeModel("team/large", "Large", .5, 1., context_tokens=64000, output_tokens=8192,
                              tool_call=True, reasoning=True)
        self.discover.return_value = [*self.discover.return_value, large]
        history = ChatHistory(self.workspace)
        denial = reply(self.cheap, "ses_denied", error="Credencial vencida")
        denial.update(http_status=401, http_error_scope="selected_model")
        first = history.begin("original", None, "Intento anterior", {"model": self.cheap, "reason": "Inicial"})
        history.finish("original", first, denial)
        history.save_preferences({"provider": "team", "restrict": True,
                                  "allowed": [self.cheap, large.model_id], "confirm": True})
        self.dispatch.side_effect = AssertionError("A local preview must never contact the provider")
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.chat_input(key="opencode_chat_input").disabled)
        for prompt, model in [("Propón un título breve.", self.cheap), ("Diseña la arquitectura.", large.model_id)]:
            self.send(prompt)
            self.assertIn(model, self.content())
            self.assertTrue(self.app.button(key="opencode_confirm_send").disabled)
            self.app.button(key="opencode_confirm_cancel").click().run()
            self.assertFalse(self.app.exception)
        self.app.checkbox(key="opencode_confirm_model").uncheck().run()
        self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
        # If the filter permits another provider, its usable model takes precedence.
        history.save_preferences({"provider": None, "restrict": True,
                                  "allowed": [self.cheap, self.reasoner], "confirm": True})
        self.app = self.new_app().run()
        self.send("Propón un título breve.")
        self.assertIn(self.reasoner, self.content())
        self.assertFalse(self.app.button(key="opencode_confirm_send").disabled)
        self.assertEqual(history.blocked_providers(), {"team": first})
        self.assertEqual(history.conversations(), [("original", "Intento anterior")])
        self.dispatch.assert_not_called()

    def test_unavailable_saved_candidates_never_expand_to_other_models(self):
        history = ChatHistory(self.workspace)
        for provider, allowed, restrict in [
            ("unavailable", [], False),
            ("team", ["team/missing-model"], True),
        ]:
            with self.subTest(provider=provider):
                history.save_preferences({"provider": provider, "restrict": restrict,
                                          "allowed": allowed, "confirm": True})
                self.app = self.new_app().run()
                self.assertFalse(self.app.exception)
                self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
                self.dispatch.assert_not_called()

    def test_missing_project_never_queries_catalog(self):
        self.app.session_state["opencode_directory"] = str(self.workspace / "does-not-exist")
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.chat_input(key="opencode_chat_input").disabled)
        self.discover.assert_not_called()
        self.dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
