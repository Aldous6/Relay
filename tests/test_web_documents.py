"""Public document API, using synthetic attachments and no running Ollama."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router import web_documents
from relay_router.web_documents import DocumentService, DocumentsUnavailable


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.web = DocumentService(self.root)
        self.backend = Mock()
        self.backend.status.return_value = {"available_models": ["local:small"], "live_ready": True}
        self.backend.respond.return_value = {"answer": "Son 42.", "live_model": "local:small",
            "usage": {"input_tokens": 4}, "router_seconds": 0.2, "router_mode": "frozen_v2",
            "unrelated_debug_data": "omit"}

    def test_lightweight_install_can_open_documents_without_optional_packages(self):
        with patch.dict(sys.modules, {"routerbench.demo": None}):
            state = self.web.request("documentos-state", {})
            self.assertFalse(state["available"])
            self.assertFalse(state["ready"])
            self.assertEqual(state["models"], [])
            self.assertFalse(state["v3_available"])
            with self.assertRaisesRegex(DocumentsUnavailable, "instalación completa"):
                self.web.request("documentos-send", {"prompt": "Correo ficticio"})
            self.assertIn("instalación completa", state["notice"])

    def test_state_loads_only_a_valid_profile_and_never_generates(self):
        with patch.object(web_documents, "_load_service", return_value=self.backend) as load:
            with self.assertRaises(ValueError):
                self.web.request("documentos-state", {"profile": "../alternativo"})
            v3 = self.web.request("documentos-state", {"profile": "experimental_v3"})
            self.assertFalse(v3["available"])
            load.assert_not_called()
            state = self.web.request("documentos-state", {"model": "fake", "workspace": str(self.root.parent)})
            self.assertEqual(state["models"], ["local:small"])
            self.assertTrue(state["ready"])
            self.backend.status.return_value = {"available_models": [], "live_ready": False}
            self.assertFalse(self.web.request("documentos-state", {})["ready"])
            load.assert_called_once_with(self.root, "frozen_v2")
            self.backend.respond.assert_not_called()

    def test_v3_discovery_requires_an_existing_model_in_its_run(self):
        run = self.root / web_documents._RESEARCH
        run.mkdir(parents=True)
        summary = run / "summary.json"
        summary.write_text(json.dumps({"selected_for_future_evaluation": {"artifact": "../foreign.joblib"}}))
        with patch.object(web_documents, "_load_service", return_value=self.backend) as load:
            self.assertFalse(self.web.request("documentos-state", {"profile": "experimental_v3"})["v3_available"])
            summary.write_text(json.dumps({"selected_for_future_evaluation": {"artifact": "selection.joblib"}}))
            self.assertFalse(self.web.request("documentos-state", {"profile": "experimental_v3"})["available"])
            (run / "selection.joblib").touch()
            self.assertTrue(self.web.request("documentos-state", {"profile": "experimental_v3"})["v3_available"])
            load.assert_called_once_with(self.root, "experimental_v3")

    def test_two_turns_keep_the_document_once_and_preserve_newline_and_indent(self):
        with patch.object(web_documents, "_load_service", return_value=self.backend):
            document = "A:\n    X = 42\n"
            first = self.web.request("documentos-send", {"prompt": "¿El valor?", "document": "\ufeff" + document,
                "system": "discard", "model": "remote/override"})
            self.backend.respond.assert_called_once_with("¿El valor?", history=[], document=document)
            self.assertIn(document, first["user_message"]["model_content"])
            self.assertEqual(first["detail"]["usage"], {"input_tokens": 4, "output_tokens": None})
            self.assertNotIn("unrelated_debug_data", first["detail"])
            self.assertTrue(first["detail"]["limitation"])
            history = [first["user_message"], first["assistant_message"]]
            second = self.web.request("documentos-send", {"prompt": "¿Otra vez?", "history": history, "document": document})
            self.backend.respond.assert_called_with("¿Otra vez?", history=history, document="")
            self.assertEqual(second["user_message"]["document_sha256"], first["user_message"]["document_sha256"])
            self.assertEqual(second["user_message"]["model_content"], "¿Otra vez?")
            self.web.request("documentos-send", {"prompt": "Este otro", "history": history, "document": "Otro"})
            self.backend.respond.assert_called_with("Este otro", history=history, document="Otro")

    def test_invalid_input_and_empty_output_never_add_a_turn(self):
        with patch.object(web_documents, "_load_service", return_value=self.backend):
            for invalid in ({"prompt": "  "}, {"document": 33}, {"history": [None]},
                            {"history": [{"role": "system", "content": "bad"}]},
                            {"history": [{"role": "user", "content": "ok", "model_content": []}]}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    self.web.request("documentos-send", {"prompt": "Pregunta", **invalid})
            self.backend.respond.assert_not_called()
            self.backend.respond.return_value = {"answer": ""}
            with self.assertRaisesRegex(DocumentsUnavailable, "texto"):
                self.web.request("documentos-send", {"prompt": "Pregunta"})

    def test_scientific_catalog_cannot_redirect_documents_to_a_remote_provider(self):
        (self.root / "configs").mkdir()
        (self.root / "configs/live_catalog.json").write_text("{}")
        (self.root / "profile.joblib").touch()
        self.backend.primary = {"run_dir": ".", "artifact": "profile.joblib"}
        self.backend.catalog = {"base_url": "https://192.0.2.1"}
        factory = Mock(return_value=self.backend)
        with patch.dict(sys.modules, {"routerbench.demo": SimpleNamespace(RelayDemoService=factory)}):
            state = self.web.request("documentos-state", {"base_url": "http://127.0.0.1:11434"})
            self.assertFalse(state["available"])
            self.backend.status.assert_not_called()
            self.backend.respond.assert_not_called()


if __name__ == "__main__":
    unittest.main()
