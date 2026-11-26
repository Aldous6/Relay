"""The bounded adaptation recipe selects only on historical calibration."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.adaptive_training import train_adaptation
from relay_router.adaptive_policy import evaluate_policy
from relay_router.adaptive_runtime import load_adapted_relay


def records(prefix, count):
    rows = []
    for i in range(count):
        text = f"task {'easy' if i % 2 == 0 else 'hard'} {prefix} {i:04d}"
        for model in ("strong", "cheap"):
            rows.append(dict(task_id=f"{prefix}-{i}", group_id=f"{prefix}-{i}", prompt_text=text,
                             model=model, accepted=model == "strong" or i % 2 == 0,
                             cost_usd=1. if model == "strong" else .1, cost_basis="tariff_estimate"))
    return rows


def scorer(text):
    return {"score": .1 if "easy" in text else .9, "input_tokens": 10, "truncated_tokens": 0}


class AdaptiveTrainingTests(unittest.TestCase):
    def test_two_fixed_representations_use_same_groups_and_preserve_rejected_candidate(self):
        policy, selection = train_adaptation(records("dev", 30), baseline_model="strong", candidate_model="cheap", scorer=scorer)
        self.assertEqual(policy["representation"], "routellm-bert")
        self.assertEqual(len(selection["candidates"]), 2)
        rejected, semantic = selection["candidates"]
        self.assertEqual(rejected["threshold"], 0.)
        self.assertEqual(rejected["training"]["fit_group_ids"], semantic["training"]["fit_group_ids"])
        self.assertEqual(rejected["training"]["calibration_group_ids"], semantic["training"]["calibration_group_ids"])
        self.assertEqual(selection["selected_artifact_id"], policy["artifact_id"])

    def test_real_runtime_loads_resident_engine_once_only_after_fresh_gate(self):
        policy, _ = train_adaptation(records("dev", 30), baseline_model="strong", candidate_model="cheap", scorer=scorer)
        with patch("relay_router.integrations.routellm.LocalBERTScorer", return_value=scorer) as factory:
            relay = load_adapted_relay(policy, scope="openhands-python-v1")
            self.assertEqual(relay.task("coding", routing_text="task easy private").model, "strong")
            factory.assert_not_called()
            report = evaluate_policy(policy, records("fresh", 24), scorer=scorer)
            self.assertTrue(report["activation_eligible"])
            with self.assertRaises(ValueError):
                load_adapted_relay(policy, report, scope="openhands-python-v1")
            relay = load_adapted_relay(policy, report, scope="openhands-python-v1", checkpoint_dir="verified-fixture")
            self.assertEqual(relay.task("coding", routing_text="task easy private").model, "cheap")
            self.assertEqual(relay.task("coding", routing_text="task hard private").model, "strong")
            factory.assert_called_once_with("verified-fixture")

    def test_no_engine_and_no_qualifier_retains_baseline(self):
        policy, selection = train_adaptation(records("dev", 30), baseline_model="strong", candidate_model="cheap")
        self.assertEqual(len(selection["candidates"]), 1)
        self.assertEqual(policy["threshold"], 0.)
        self.assertEqual(selection["reason"], "no_candidate_qualified_keep_baseline")


if __name__ == "__main__":
    unittest.main()
