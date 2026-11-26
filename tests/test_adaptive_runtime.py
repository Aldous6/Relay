"""Scope and evidence gates through the public runtime and offline CLI."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.adaptive_policy import train_policy, evaluate_policy
from relay_router.adaptive_runtime import AdaptivePromptSelector, load_adapted_relay
from relay_router.workflow_cli import main


def paired(prefix, count):
    result = []
    for i in range(count):
        text = (f"Small request {prefix} {i}" if i % 2 == 0 else
                f"Complex request {prefix} {i}\n" + "    for item in sequence:\n        if item: return item\n" * 20)
        for model in ("strong", "cheap"):
            result.append({"task_id": f"{prefix}-{i}", "group_id": f"{prefix}-{i}", "prompt_text": text,
                           "model": model, "accepted": model == "strong" or i % 2 == 0,
                           "cost_usd": 1. if model == "strong" else .1, "cost_basis": "tariff_estimate"})
    return result


class AdaptiveRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development = paired("development", 30)
        cls.validation = paired("fresh", 24)
        cls.policy = train_policy(cls.development, baseline_model="strong", candidate_model="cheap")
        cls.report = evaluate_policy(cls.policy, cls.validation)
        assert cls.report["activation_eligible"], cls.report["reasons"]

    def test_missing_or_failed_evidence_keeps_baseline(self):
        failed = evaluate_policy(self.policy, paired("too-small", 4))
        for report in (None, failed):
            relay = load_adapted_relay(self.policy, report, scope="openhands-python-v1")
            task = relay.task("python", routing_text="Small request new 2")
            self.assertEqual(task.model, "strong")
            self.assertEqual(task.decision["reason"], "baseline_adaptation_gate_not_met")

    def test_validated_prompt_route_is_sticky_private_and_unpriced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            relay = load_adapted_relay(self.policy, self.report, scope="openhands-python-v1", log_path=path)
            text = "Small request PRIVATE_PAYLOAD 2"
            task = relay.task("python", routing_text=text)
            self.assertEqual(task.model, "cheap")
            self.assertEqual(task.decision["prompt_selection"]["status"], "empirically_validated_pilot")
            received = []
            for _ in range(2):
                task.call(lambda model: received.append(model), measure=lambda _: {"cost_usd": .1, "cost_basis": "tariff_estimate"})
            outcome = task.finish(accepted=True, cost_complete=True)
            self.assertEqual(received, ["cheap", "cheap"])
            self.assertIsNone(outcome["routing_cost_usd"])
            self.assertNotIn(text, path.read_text())
            self.assertNotIn("PRIVATE_PAYLOAD", path.read_text())

    def test_scope_mismatch_and_tampering_reject_before_a_call(self):
        with self.assertRaises(ValueError):
            load_adapted_relay(self.policy, self.report, scope="unrelated")
        altered = deepcopy(self.report)
        altered["activation_eligible"] = False
        with self.assertRaises(ValueError):
            load_adapted_relay(self.policy, altered, scope="openhands-python-v1")
        changed = deepcopy(self.policy)
        changed["candidate_model"] = "third-model"
        with self.assertRaises(ValueError):
            AdaptivePromptSelector(changed, self.report, scope="openhands-python-v1")

    def test_observe_and_capability_filter_keep_baseline(self):
        for mode, eligible in (("observe", ["strong", "cheap"]), ("apply", ["strong"])):
            relay = load_adapted_relay(self.policy, self.report, scope="openhands-python-v1", mode=mode)
            task = relay.task("python", routing_text="Small request new 2", eligible_models=eligible)
            self.assertEqual(task.model, "strong")
        with self.assertRaises(ValueError):
            load_adapted_relay(self.policy, self.report, scope="openhands-python-v1").task("python")

    def test_local_cli_adapts_validates_loads_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            dev, val, policy, report = [root / name for name in ("dev.json", "val.json", "policy.json", "report.json")]
            dev.write_text(json.dumps({"records": self.development}), encoding="utf-8")
            val.write_text(json.dumps(self.validation), encoding="utf-8")
            command = ["adapt", str(dev), "--baseline", "strong", "--candidate", "cheap", "--output", str(policy)]
            self.assertEqual(main(command), 0)
            original = policy.read_bytes()
            self.assertEqual(main(command), 2)
            self.assertEqual(policy.read_bytes(), original)
            self.assertEqual(main(["validate-adaptation", str(policy), str(val), "--output", str(report)]), 0)
            relay = load_adapted_relay(policy, report, scope="openhands-python-v1")
            self.assertEqual(relay.task("python", routing_text="Small request brand-new 2").model, "cheap")


if __name__ == "__main__":
    unittest.main()
