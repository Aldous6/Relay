"""Regression tests for policy choices over a changing OpenCode catalog."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.integrations.opencode import OpenCodeModel
from relay_router.opencode_selection import advertised_free, select_opencode_model


class OpenCodeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.small = OpenCodeModel(
            model_id="example/small-free", name="Small Free",
            input_usd_per_million=0, output_usd_per_million=0, context_tokens=8192,
            input_tokens=None, output_tokens=4096, tool_call=True, reasoning=True,
            input_text=True, output_text=True, status="active",
        )
        self.large = replace(self.small, model_id="example/large-free", name="Large Free",
                             context_tokens=65536, output_tokens=8192)

    def test_prompt_changes_choice_inside_same_catalog(self):
        for catalog in ([self.small, self.large], [self.large, self.small]):
            short = select_opencode_model("Resume este README", catalog)
            broad = select_opencode_model("Diagnóstico de un fallo intermitente en varios módulos", catalog)
            self.assertEqual((short["model"], short["profile"]), (self.small.model_id, "standard"))
            self.assertEqual((broad["model"], broad["profile"]), (self.large.model_id, "extended"))
            self.assertEqual(broad["eligible_models"], 2)
            self.assertNotIn("README", str(short))

    def test_unknown_price_not_treated_as_free(self):
        missing = replace(self.small, model_id="private/unknown", name="Unknown",
                          input_usd_per_million=None, output_usd_per_million=None)
        zero = replace(missing, model_id="private/zero", input_usd_per_million=0,
                       output_usd_per_million=0)
        paid = replace(missing, model_id="private/paid", input_usd_per_million=1,
                       output_usd_per_million=3)
        self.assertEqual(select_opencode_model("Hola", [missing])["tariff_status"], "unknown")
        self.assertEqual(select_opencode_model("Hola", [zero])["tariff_status"], "zero_unverified")
        self.assertEqual(select_opencode_model("Hola", [missing, zero, paid])["model"], paid.model_id)
        self.assertEqual(select_opencode_model("Hola", [paid, self.small])["tariff_status"], "advertised_free")

    def test_comparison_uses_both_rates_for_the_reference_token_mix(self):
        high_output = replace(self.small, model_id="other/inexpensive-input", input_usd_per_million=.1,
                              output_usd_per_million=4)
        low_output = replace(self.small, model_id="other/inexpensive-output", input_usd_per_million=1,
                             output_usd_per_million=.1)
        self.assertEqual(select_opencode_model("Hola", [high_output, low_output])["model"], low_output.model_id)

    def test_native_free_metadata_admits_unlabelled_models_and_overrides_legacy_labels(self):
        unlabelled = replace(self.small, model_id="opencode/big-pickle", name="Big Pickle", advertised_free=True)
        self.assertTrue(advertised_free(unlabelled))
        self.assertEqual(select_opencode_model("Hola", [unlabelled])["tariff_status"], "advertised_free")
        self.assertFalse(advertised_free(replace(self.small, advertised_free=False)))
        self.assertFalse(advertised_free(replace(unlabelled, input_usd_per_million=.1)))
        self.assertFalse(advertised_free(replace(unlabelled, output_usd_per_million=None)))

    def test_excludes_incompatible_models_and_unavailable_context(self):
        ineligible = [
            replace(self.small, model_id="other/speech", input_text=False),
            replace(self.small, model_id="other/no-text", output_text=False),
            replace(self.small, model_id="other/no-tools", tool_call=False),
            replace(self.small, model_id="other/disabled", status="disabled"),
            replace(self.small, model_id="other/short-input", input_tokens=1),
            replace(self.small, model_id="other/short-context", context_tokens=3),
            replace(self.small, model_id="other/no-output", output_tokens=0),
        ]
        prompt = "Cuenta las líneas del archivo."
        with self.assertRaisesRegex(ValueError, "Ningún modelo"):
            select_opencode_model(prompt, ineligible)
        selected = select_opencode_model(prompt, [*ineligible, self.large])
        self.assertEqual(selected["model"], self.large.model_id)
        self.assertEqual(selected["eligible_models"], 1)

    def test_missing_capability_is_fallback_and_output_target_is_soft(self):
        unknown = replace(self.small, model_id="private/unknown", tool_call=None, context_tokens=None,
                          output_tokens=None, input_usd_per_million=None, output_usd_per_million=None)
        compact = replace(self.small, model_id="private/small", output_tokens=128)
        self.assertEqual(select_opencode_model("Debug a race condition", [unknown])["model"], unknown.model_id)
        self.assertEqual(select_opencode_model("Debug a race condition", [compact])["model"], compact.model_id)
        self.assertEqual(select_opencode_model("Debug a race condition", [compact, unknown])["model"], compact.model_id)

    def test_reasoning_and_long_prompt_are_explicit_preferences(self):
        basic = replace(self.large, model_id="example/basic-free", reasoning=False)
        decision = select_opencode_model("á" * 2100, [basic, self.small])
        self.assertEqual(decision["profile"], "extended")
        self.assertEqual(decision["model"], self.small.model_id)
        self.assertEqual(decision["prompt_bytes"], 4200)
        self.assertEqual(decision["estimated_prompt_tokens"], 1050)

    def test_invalid_inputs_and_deprecated_model(self):
        old = replace(self.large, model_id="old/deprecated", status="deprecated")
        self.assertEqual(select_opencode_model("Migrate a library", [self.small, old])["model"], self.small.model_id)
        self.assertEqual(select_opencode_model("Hello", [old])["model"], old.model_id)
        for prompt, catalog in [(" \n", [old]), ("Hola", []), ("Hola", [old, old])]:
            with self.assertRaises(ValueError):
                select_opencode_model(prompt, catalog)


if __name__ == "__main__":
    unittest.main()
