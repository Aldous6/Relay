"""Portable inventory contract checks; no Ollama server or inference involved."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("relay_ollama_inventory_tested", ROOT / "tools/relay_ollama_inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in inventory tests"))
        guard.start()
        self.addCleanup(guard.stop)

    def replies(self):
        return {
            "/api/tags": {"models": [{"name": "lab-model:latest", "digest": "a" * 64, "size": 100,
                "details": {"family": "lab", "parameter_size": "3B", "quantization_level": "Q4_K_M"},
                "private": "PRIVATE_SENTINEL"}]},
            "/api/version": {"version": "0.test"},
            "/api/ps": {"models": [{"name": "lab-model:latest", "digest": "a" * 64,
                "context_length": 4096, "size_vram": 80}]},
            "/api/show": {"capabilities": ["completion", "tools"],
                "model_info": {"lab.context_length": 32768, "lab.block_count": 20},
                "template": "PRIVATE_SENTINEL", "system": "PRIVATE_SENTINEL",
                "modelfile": "PRIVATE_SENTINEL", "parameters": "PRIVATE_SENTINEL"},
        }

    def collect(self, replies):
        def response(base, path, payload=None, **kwargs):
            value = replies[path]
            if isinstance(value, Exception):
                raise value
            return value
        with patch.object(inventory, "request_metadata", side_effect=response) as request:
            result = inventory.collect_inventory("http://127.0.0.1:11434")
        return result, request

    def test_inventory_exports_only_metadata_without_generation_or_download(self):
        report, request = self.collect(self.replies())
        row = report["models"][0]
        self.assertEqual(report["ollama_version"], "0.test")
        self.assertEqual(row["capabilities"], ["completion", "tools"])
        self.assertEqual(row["declared_context_lengths"], {"lab.context_length": 32768})
        self.assertEqual(row["active_context_length"], 4096)
        self.assertEqual(row["allocated_vram_bytes"], 80)
        self.assertTrue(row["loaded_at_inspection"])
        self.assertIsNone(report["gpu_hardware"])
        self.assertIsNone(report["selection_policy"])
        self.assertFalse(row["quality_evaluated"])
        self.assertEqual(report["generation_requests"], 0)
        self.assertEqual(report["download_requests"], 0)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(report))
        self.assertNotIn("127.0.0.1", json.dumps(report))
        self.assertEqual([call.args[1] for call in request.call_args_list],
                         ["/api/tags", "/api/version", "/api/ps", "/api/show"])
        self.assertEqual(request.call_args_list[-1].args[2], {"model": "lab-model:latest"})

    def test_loaded_requires_exact_name_and_digest_not_alias_or_changed_version(self):
        cases = [
            ({"name": "lab-model:latest", "digest": "a" * 64}, True),
            ({"model": "lab-model:latest", "digest": "a" * 64}, True),
            ({"name": "another-alias", "digest": "a" * 64}, False),
            ({"name": "lab-model:latest", "digest": "b" * 64}, False),
        ]
        for loaded, expected in cases:
            with self.subTest(loaded=loaded):
                replies = self.replies()
                replies["/api/ps"] = {"models": [loaded]}
                result, _ = self.collect(replies)
                self.assertIs(result["models"][0]["loaded_at_inspection"], expected)
        replies = self.replies()
        replies["/api/tags"]["models"][0].pop("digest")
        result, _ = self.collect(replies)
        self.assertIsNone(result["models"][0]["loaded_at_inspection"])

    def test_failed_or_invalid_running_query_is_unknown_and_partial(self):
        for value in (inventory.InventoryError("Consulta fallida."), {}, {"models": "invalid"}):
            with self.subTest(value=value):
                replies = self.replies()
                replies["/api/ps"] = value
                result, _ = self.collect(replies)
                self.assertFalse(result["all_metadata_requests_succeeded"])
                self.assertIsNone(result["models"][0]["loaded_at_inspection"])
                self.assertIsNone(result["models"][0]["allocated_vram_bytes"])
                self.assertTrue(any(warning["endpoint"] == "/api/ps" for warning in result["warnings"]))

    def test_unknown_optional_metadata_is_not_invented(self):
        replies = self.replies()
        replies["/api/version"] = inventory.InventoryError("Versión no disponible.")
        replies["/api/show"] = inventory.InventoryError("Metadatos no disponibles.")
        replies["/api/tags"]["models"][0]["size"] = True
        result, _ = self.collect(replies)
        self.assertIsNone(result["ollama_version"])
        self.assertIsNone(result["models"][0]["size_bytes"])
        self.assertIsNone(result["models"][0]["capabilities"])
        self.assertIsNone(result["models"][0]["declared_context_lengths"])
        self.assertFalse(result["all_metadata_requests_succeeded"])

    def test_invalid_installed_models_fail_before_optional_queries(self):
        for value in ({}, {"models": "invalid"}, {"models": ["invalid"]}):
            with self.subTest(value=value), patch.object(inventory, "request_metadata", return_value=value) as request:
                with self.assertRaises(inventory.InventoryError):
                    inventory.collect_inventory()
                self.assertEqual(request.call_count, 1)

    def test_url_and_timeout_validation_precede_network(self):
        urls = ("file:///tmp", "http://user:secret@localhost", "http://localhost/api",
                "http://localhost?key=secret", "http://localhost/#secret", "http://localhost:0",
                "http://localhost:65536", "http://[invalid", "http://local host", "http://localhost\n")
        with patch.object(inventory, "build_opener") as opener:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(inventory.InventoryError):
                    inventory.request_metadata(url, "/api/tags")
            for timeout in (0, -1, True, float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(inventory.InventoryError):
                    inventory.request_metadata(inventory.DEFAULT_URL, "/api/tags", timeout=timeout)
            opener.assert_not_called()
        self.assertEqual(inventory.validate_url("https://ollama.lab:11434/"), "https://ollama.lab:11434")
        self.assertEqual(inventory.validate_url("http://[::1]:11434"), "http://[::1]:11434")

    def test_request_allowlist_rejects_inference_download_and_arbitrary_show_fields(self):
        requests = [("/api/chat", {"model": "lab"}), ("/api/generate", {"model": "lab"}),
                    ("/api/pull", {"model": "lab"}), ("/api/create", {"model": "lab"}),
                    ("/api/tags", {}), ("/api/show", None),
                    ("/api/show", {"model": "lab", "prompt": "secret"})]
        with patch.object(inventory, "build_opener") as opener:
            for path, payload in requests:
                with self.subTest(path=path, payload=payload), self.assertRaises(inventory.InventoryError):
                    inventory.request_metadata(inventory.DEFAULT_URL, path, payload)
            opener.assert_not_called()

    def test_transport_disables_proxy_and_redirect_and_bounds_read(self):
        response = Mock()
        response.read.return_value = b'{"capabilities":["completion"]}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = context
        with patch.object(inventory, "build_opener", return_value=opener) as build:
            result = inventory.request_metadata(inventory.DEFAULT_URL, "/api/show", {"model": "lab"})
        self.assertEqual(result, {"capabilities": ["completion"]})
        handlers = build.call_args.args
        self.assertIsInstance(handlers[0], ProxyHandler)
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], inventory._NoRedirect)
        self.assertIsNone(handlers[1].redirect_request(None, None, 302, "redirect", {}, "https://other.invalid"))
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"model": "lab"})
        self.assertIsNone(request.get_header("Authorization"))
        response.read.assert_called_once_with(inventory.MAX_RESPONSE_BYTES + 1)

    def test_invalid_json_shapes_oversize_and_provider_errors_are_sanitized(self):
        samples = (b'not-json PRIVATE_SENTINEL', b'[]', b'{"error":"PRIVATE_SENTINEL"}',
                   b'\xff', b'x' * (inventory.MAX_RESPONSE_BYTES + 1))
        for raw in samples:
            with self.subTest(length=len(raw)):
                opener = Mock()
                opener.open.return_value = io.BytesIO(raw)
                with patch.object(inventory, "build_opener", return_value=opener):
                    with self.assertRaises(inventory.InventoryError) as error:
                        inventory.request_metadata(inventory.DEFAULT_URL, "/api/tags")
                self.assertNotIn("PRIVATE_SENTINEL", str(error.exception))
        body = Mock()
        errors = [HTTPError("http://private.invalid", 403, "PRIVATE_SENTINEL", {}, body),
                  URLError("PRIVATE_SENTINEL"), TimeoutError("PRIVATE_SENTINEL")]
        for failure in errors:
            with self.subTest(failure=type(failure).__name__):
                opener = Mock()
                opener.open.side_effect = failure
                with patch.object(inventory, "build_opener", return_value=opener):
                    with self.assertRaises(inventory.InventoryError) as error:
                        inventory.request_metadata(inventory.DEFAULT_URL, "/api/tags")
                self.assertNotIn("PRIVATE_SENTINEL", str(error.exception))
                self.assertNotIn("private.invalid", str(error.exception))
                self.assertEqual(opener.open.call_count, 1)
        body.read.assert_not_called()

    def test_cli_preserves_existing_report_without_querying(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.json"
            path.write_text("existing report", encoding="utf-8")
            with patch.object(inventory, "collect_inventory") as collect, contextlib.redirect_stderr(io.StringIO()):
                code = inventory.main(["--output", str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), "existing report")
            collect.assert_not_called()

    def test_cli_writes_partial_report_and_explains_missing_data(self):
        replies = self.replies()
        replies["/api/ps"] = inventory.InventoryError("Consulta no disponible.")
        report, _ = self.collect(replies)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.json"
            output = io.StringIO()
            with patch.object(inventory, "collect_inventory", return_value=report), contextlib.redirect_stdout(output):
                code = inventory.main(["--output", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            self.assertIn("Inventario parcial", output.getvalue())

    def test_cli_failure_does_not_create_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.json"
            with patch.object(inventory, "collect_inventory", side_effect=inventory.InventoryError("Ollama no disponible.")), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(inventory.main(["--output", str(path)]), 1)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
