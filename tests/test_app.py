"""Headless app checks; these never start a web server or generate real text."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/relay_app.py"


class FakeRelayService:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = []

    def status(self):
        return {"available_models": ["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3:latest"] if self.ready else [],
                "live_ready": self.ready}

    def respond(self, prompt, history, document=""):
        self.calls.append({"prompt": prompt, "history": history, "document": document})
        return {"answer": "Respuesta de prueba local.", "live_model": "qwen2.5:1.5b",
                "historical_model": "example-historical-model", "historical_tier": 1,
                "router_seconds": 0.02, "generation_seconds": 0.3,
                "usage": {"input_tokens": 12, "output_tokens": None}, "cost_usd": None,
                "mapping_status": "experimental", "decision": "threshold_met"}


def app_with(service):
    app = AppTest.from_file(str(APP), default_timeout=15)
    app.session_state["_relay_service"] = service
    return app.run()


class AppTests(unittest.TestCase):
    def test_v3_availability_requires_summary_and_trained_artifact(self):
        spec = importlib.util.spec_from_file_location("relay_app_availability_test", APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            module.ROOT = Path(temporary)
            directory = module.ROOT / "reports/relay-v3-development-v1"
            directory.mkdir(parents=True)
            self.assertFalse(module._v3_available())
            summary = {"selected_for_future_evaluation": {"artifact": None}}
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertFalse(module._v3_available())
            summary["selected_for_future_evaluation"]["artifact"] = "trained.joblib"
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertFalse(module._v3_available())
            (directory / "trained.joblib").write_bytes(b"fixture")
            self.assertTrue(module._v3_available())

    def test_v3_toggle_is_explicit_uses_separate_cache_and_never_generates_automatically(self):
        source = ("import importlib.util\n"
                  f"spec = importlib.util.spec_from_file_location('relay_app_toggle_test', {str(APP)!r})\n"
                  "module = importlib.util.module_from_spec(spec)\n"
                  "spec.loader.exec_module(module)\n"
                  "module._v3_available = lambda: True\n"
                  "module.main()\n")
        default, experimental = FakeRelayService(), FakeRelayService()
        app = AppTest.from_string(source, default_timeout=15)
        app.session_state["_relay_service"] = default
        app.session_state["_relay_service_v3"] = experimental
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key="_relay_use_v3").value)
        app.checkbox(key="_relay_use_v3").check().run()
        self.assertFalse(app.exception)
        self.assertEqual(default.calls, [])
        self.assertEqual(experimental.calls, [])
        app.text_area(key="relay_prompt").input("Pregunta experimental")
        next(button for button in app.button if button.label == "Enviar a Relay").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(experimental.calls), 1)
        self.assertEqual(default.calls, [])
        app.checkbox(key="_relay_use_v3").uncheck().run()
        self.assertEqual(len(experimental.calls), 1)
        self.assertEqual(default.calls, [])
        next(button for button in app.button if button.label == "Enviar a Relay").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(default.calls), 1)
        self.assertEqual(len(experimental.calls), 1)
        self.assertEqual(len(default.calls[0]["history"]), 2)

    def test_document_is_retained_once_per_conversation_and_new_documents_are_allowed(self):
        spec = importlib.util.spec_from_file_location("relay_app_document_test", APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text, digest = module._new_document("Documento original", [])
        self.assertEqual(text, "Documento original")
        history = [{"role": "user", "document_sha256": digest, "model_content": text}]
        self.assertEqual(module._new_document("Documento original", history), ("", digest))
        self.assertEqual(module._new_document("Otro documento", history)[0], "Otro documento")
        self.assertEqual(module._new_document("Documento original", [])[0], "Documento original")

    def test_initial_page_has_no_errors_and_never_generates(self):
        service = FakeRelayService()
        app = app_with(service)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Espacio de trabajo local")
        self.assertEqual([tab.label for tab in app.tabs], ["Conversar", "Resultados", "Metodología"])
        self.assertEqual(service.calls, [])
        self.assertTrue(any("límites de transferencia" in item.value for item in app.caption))
        self.assertTrue(any(item.label == "Adaptación a Qwen · evaluación nueva" for item in app.expander))

    def test_explicit_submit_preserves_history_and_missing_usage(self):
        service = FakeRelayService()
        app = app_with(service)
        app.text_area(key="relay_prompt").input("Compara las hipótesis.")
        next(button for button in app.button if button.label == "Enviar a Relay").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(service.calls[0]["prompt"], "Compara las hipótesis.")
        self.assertEqual(service.calls[0]["history"], [])
        self.assertEqual(len(app.session_state["_relay_history"]), 2)
        self.assertEqual([message.name for message in app.chat_message], ["user", "assistant"])
        self.assertTrue(any(metric.label == "Tokens de salida" and metric.value == "Sin dato" for metric in app.metric))
        app.text_area(key="relay_prompt").input("Ahora explica la segunda.")
        next(button for button in app.button if button.label == "Enviar a Relay").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(service.calls), 2)
        self.assertEqual(service.calls[1]["history"][0]["content"], "Compara las hipótesis.")
        self.assertEqual(service.calls[1]["history"][1]["content"], "Respuesta de prueba local.")

    def test_missing_backend_disables_send_without_error(self):
        service = FakeRelayService(ready=False)
        app = app_with(service)
        self.assertFalse(app.exception)
        self.assertTrue(next(button for button in app.button if button.label == "Enviar a Relay").disabled)
        self.assertEqual(service.calls, [])

    def test_empty_prompt_is_not_sent_and_reset_discards_history(self):
        service = FakeRelayService()
        app = app_with(service)
        next(button for button in app.button if button.label == "Enviar a Relay").click().run()
        self.assertEqual(service.calls, [])
        self.assertTrue(app.warning)
        app.session_state["_relay_history"] = [{"role": "user", "content": "old"}]
        next(button for button in app.button if button.label == "Reiniciar conversación").click().run()
        self.assertEqual(app.session_state["_relay_history"], [])
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
