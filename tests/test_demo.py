import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from routerbench.demo import RelayDemoService, assemble_messages, conservative_context_bound
from routerbench.data import sha256_file, write_json


class DemoTests(unittest.TestCase):
    def test_history_document_and_final_question_are_preserved(self):
        document = "evidence\n" * 1000 + "FINAL EVIDENCE"
        history = [{"role": "user", "content": "Previous question"}, {"role": "assistant", "content": "Previous answer"}]
        messages = assemble_messages("Compare them", history, document)
        self.assertEqual(messages[1:3], history)
        self.assertIn(document, messages[-1]["content"])
        self.assertTrue(messages[-1]["content"].endswith("Compare them"))
        self.assertGreater(conservative_context_bound(messages), len(document.encode("utf-8")))
        preserved = assemble_messages("Follow up", [{"role": "user", "content": "Short display", "model_content": document}])
        self.assertEqual(preserved[1]["content"], document)

    def test_one_selected_local_model_executes_without_historical_confidence_claim(self):
        service = RelayDemoService.__new__(RelayDemoService)
        service.catalog = {"base_url": "http://127.0.0.1:11434", "default_max_output_tokens": 256,
                           "mapping_status": "experimental", "candidate_models": [{"model": "small", "tier": 0}, {"model": "large", "tier": 2}]}
        service.primary = {"minimum_predicted_success": 0.75}
        service.tier_map = {"old_small": 0, "old_large": 2}
        router = Mock()
        router.predict_tables.return_value = (pd.DataFrame({"old_small": [0.8], "old_large": [0.9]}), pd.DataFrame({"old_small": [1.0], "old_large": [10.0]}))
        service._load_router = Mock(return_value=router)
        with patch("routerbench.demo.discover_ollama", return_value=[{"model": "small", "context_length": 8192}, {"model": "large", "context_length": 8192}]), patch("routerbench.demo.generate_ollama", return_value={"content": "answer", "duration_seconds": 1.0, "usage": {"input_tokens": 4, "output_tokens": 2}, "cost_usd": None, "provider": "ollama", "finish_reason": "stop"}) as generate:
            result = service.respond("Hello", [])
        self.assertEqual(result["live_model"], "small")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.args[0], "small")
        self.assertIsNone(result["cost_usd"])
        self.assertNotIn("predicted_success", result)


STRONG = "gpt-4-1106-preview"
WEAK = "mistralai/mixtral-8x7b-chat"


def research_fixture(root, objective="pairwise"):
    for name in ("configs", "data/processed", "reports/frozen", "reports/research"):
        (root / name).mkdir(parents=True)
    write_json(root / "configs/live_catalog.json", {"base_url": "http://127.0.0.1:11434", "default_max_output_tokens": 256,
               "mapping_status": "experimental", "candidate_models": [{"model": "small", "tier": 0},
               {"model": "middle", "tier": 1}, {"model": "large", "tier": 2}]})
    write_json(root / "data/processed/manifest.json", {"tier_map": {STRONG: 2, WEAK: 1, "old_small": 0}})
    frozen = root / "reports/frozen"
    write_json(frozen / "summary.json", {"kind": "frozen fixture"})
    (frozen / "model.joblib").write_bytes(b"frozen fixture")
    write_json(root / "configs/final_protocol.json", {"policies": [{"role": "primary_confirmatory_comparison",
               "run_dir": "reports/frozen", "artifact": "model.joblib", "artifact_sha256": sha256_file(frozen / "model.joblib"),
               "source_summary_sha256": sha256_file(frozen / "summary.json"), "variant": "lexical_hybrid", "minimum_predicted_success": .75}]})
    directory = root / "reports/research"
    (directory / "model.joblib").write_bytes(b"research fixture")
    digest = sha256_file(directory / "model.joblib")
    specification = {"id": "candidate", "objective": objective, "representation": "hybrid", "alpha": .001}
    parameter = .75 if objective == "absolute" else 10
    configuration = {"common_pair": {"strong": STRONG, "weak": WEAK}, "candidates": [specification],
                     "absolute_probability_grid": [.75], "pairwise_tradeoff_grid": [10],
                     "outer_test_evaluation_allowed": False, "automatic_live_promotion_allowed": False}
    point = {"candidate_id": "candidate", "parameter": parameter, "metrics": {"success_rate": .8}}
    summary = {"outer_test_accessed_for_v3": False, "confirmation_of_superiority": False, "automatic_live_promotion": False,
               "configuration": configuration,
               "configuration_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "quality_floor": .79,
               "selected_for_future_evaluation": {"candidate_id": "candidate", "point": point, "artifact": "model.joblib",
                                                  "artifact_sha256": digest, "requires_independent_evaluation": True},
               "candidates": {"candidate": {"specification": specification, "selected_point": point, "curve": [point],
                                             "artifact": "model.joblib", "artifact_sha256": digest}},
               "files_sha256": {"model.joblib": digest}}
    write_json(directory / "summary.json", summary)
    return directory, summary


class ExperimentalDemoTests(unittest.TestCase):
    def test_default_stays_frozen_and_explicit_v3_checks_artifact_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, _ = research_fixture(root)
            default = RelayDemoService(root)
            self.assertEqual(default.router_mode, "frozen_v2")
            self.assertIsNone(default._research_directory)
            experimental = RelayDemoService(root, "reports/research")
            self.assertEqual(experimental.router_mode, "experimental_v3")
            (directory / "model.joblib").write_bytes(b"changed")
            with patch("routerbench.demo.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "artefacto seleccionado"):
                    experimental._load_router()
                load.assert_not_called()

    def test_v3_selection_and_summary_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, summary = research_fixture(root)
            summary["selected_for_future_evaluation"]["point"] = {"candidate_id": "candidate", "parameter": 20}
            write_json(directory / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "punto operativo"):
                RelayDemoService(root, "reports/research")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, summary = research_fixture(root)
            experimental = RelayDemoService(root, "reports/research")
            summary["new_field"] = "changed"
            write_json(directory / "summary.json", summary)
            with patch("routerbench.demo.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "artefacto seleccionado"):
                    experimental._load_router()
                load.assert_not_called()

    def test_both_v3_objectives_route_only_with_the_pair_and_preserve_full_history(self):
        for objective in ("absolute", "pairwise"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                research_fixture(root, objective)
                service = RelayDemoService(root, "reports/research")
                router = Mock()
                if objective == "absolute":
                    first = pd.DataFrame({STRONG: [.9], WEAK: [.8], "old_small": [1.]})
                else:
                    first = pd.DataFrame({"strong_only": [.2], "weak_only": [.1], "equal": [.7], "expected_quality_gain": [.1]})
                router.predict_tables.return_value = first, pd.DataFrame({STRONG: [.01], WEAK: [.001]})
                history = [{"role": "user", "content": "short display", "model_content": "DOCUMENT EVIDENCE"}]
                with patch("routerbench.demo.joblib.load", return_value=router), patch("routerbench.demo.discover_ollama", return_value=[
                        {"model": "small", "context_length": 8192}, {"model": "middle", "context_length": 8192}, {"model": "large", "context_length": 8192}]), \
                        patch("routerbench.demo.generate_ollama") as generate:
                    result = service.route_only("Follow-up question", history)
                # Absolute chooses weak at .75; pairwise gain .1 exceeds .09.
                self.assertEqual(result["historical_model"], WEAK if objective == "absolute" else STRONG)
                self.assertEqual(result["router_mode"], "experimental_v3")
                self.assertIn("evaluación independiente", result["warning"])
                self.assertNotIn("predicted_success", result)
                supplied = router.predict_tables.call_args.args[0].iloc[0]["prompt_text"]
                self.assertIn("DOCUMENT EVIDENCE", supplied)
                self.assertIn("Follow-up question", supplied)
                generate.assert_not_called()

    def test_v3_capability_fallback_filters_status_and_never_uses_tier_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research_fixture(root)
            service = RelayDemoService(root, "reports/research")
            router = Mock()
            with patch("routerbench.demo.joblib.load", return_value=router), patch("routerbench.demo.discover_ollama", return_value=[
                    {"model": "small", "context_length": 8192}, {"model": "middle", "context_length": 8192}]):
                status = service.status()
                result = service.route_only("Hello", [])
            self.assertEqual(status["available_models"], ["middle"])
            self.assertEqual(status["router_mode"], "experimental_v3")
            self.assertEqual(status["primary_variant"], "candidate")
            self.assertEqual(result["historical_model"], WEAK)
            self.assertEqual(result["decision"], "capability_fallback")
            router.predict_tables.assert_not_called()
            with patch("routerbench.demo.discover_ollama", return_value=[{"model": "small", "context_length": 8192}]):
                self.assertFalse(service.status()["live_ready"])
                with self.assertRaisesRegex(ValueError, "par experimental"):
                    service.route_only("Hello", [])
            with patch("routerbench.demo.discover_ollama", return_value=[{"model": "large", "context_length": 512}]):
                with self.assertRaisesRegex(ValueError, "no fue recortado"):
                    service.route_only("Hello", [])


if __name__ == "__main__":
    unittest.main()
