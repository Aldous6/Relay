"""Synthetic structural fixtures only; these are not competitive evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from relay_router.maintenance_learning import (
    GRID, REFERENCE_GRID, MODELS, _request, _training_rows, load_policy, train_policy,
)
from routerbench.agent_trajectory_learning import public_request_text


def records():
    return [{"task_id": f"{part}-{index}", "group_id": f"{part}-group-{index}",
             "partition": part, "routing_text": f"{part} unique task {index}\n    return label",
             "reference_probability": .2 if index % 2 else .8,
             "branches": {"weak": {"success": bool(index % 2), "cost_usd": .001 + index * .00001},
                          "strong": {"success": bool(index % 2), "cost_usd": .004 + index * .00001}}}
            for part in ("fit", "calibration", "development") for index in range(8)]


def acquisition_contract():
    return {"suite_sha256": "a" * 64, "tasks_sha256": "f" * 64, "tariffs_sha256": "b" * 64,
            "sources_sha256": {"fixture.py": "c" * 64}, "scope": "development_and_reserved_holdout",
            "models": {"weak": MODELS[0], "strong": MODELS[1]}, "continuation": "fixture_declared_continuation"}


class MaintenanceLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.result = train_policy(records(), cls.root / "frozen")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_roundtrip_frozen_operating_points_use_development_and_equal_search_budget(self):
        policy = load_policy(self.result["policy_path"], self.result["policy_sha256"])
        manifest = policy.manifest()
        self.assertTrue(manifest["frozen"])
        self.assertTrue(manifest["learned"])
        self.assertFalse(manifest["holdout_used"])
        self.assertEqual(manifest["threshold_grid"], list(GRID))
        self.assertEqual(manifest["reference_threshold_grid"], list(REFERENCE_GRID))
        self.assertEqual(len(manifest["selection"]["candidates"]), 12)
        self.assertEqual(len(manifest["reference_selection"]["candidates"]), 12)
        self.assertEqual(manifest["data_provenance"], "unverified_injected_records")
        self.assertEqual(manifest["reference_selection"]["task_count"], 8)
        decision = policy.choose("new maintenance request\n    preserve indentation")
        self.assertIn(decision["choice"], ("weak", "strong"))
        self.assertGreaterEqual(decision["router_seconds"], 0.)
        self.assertEqual(set(decision["forecasts"]), set(MODELS))
        restored = load_policy(self.result["policy_path"], self.result["policy_sha256"])
        self.assertEqual(decision["forecasts"], restored.choose("new maintenance request\n    preserve indentation")["forecasts"])
        manifest["min_probability"] = .99
        self.assertNotEqual(policy.manifest()["min_probability"], .99)

    def test_outcomes_ids_and_reference_predictions_never_enter_request_features(self):
        original = records()[0]
        changed = deepcopy(original)
        changed.update(task_id="secret-task", group_id="secret-group", reference_probability=.99)
        changed["branches"]["weak"].update(success=not original["branches"]["weak"]["success"], cost_usd=9000.)
        first = _training_rows([original])[0]
        second = _training_rows([changed])[0]
        self.assertEqual(first.request.serialize(), second.request.serialize())
        self.assertNotEqual(first.success, second.success)
        self.assertNotEqual(first.remaining_provider_cost_usd, second.remaining_provider_cost_usd)
        self.assertIn("\n    return label", public_request_text(first.request))
        self.assertNotIn("secret", public_request_text(second.request))
        self.assertEqual(_request("different text").state, first.request.state)

    def test_fit_vocabulary_excludes_calibration_and_development_terms(self):
        policy = load_policy(self.result["policy_path"], self.result["policy_sha256"])
        union = dict(policy._policy.predictor._features.named_steps["union"].transformer_list)
        vocabulary = union["text"].named_steps["tfidf"].vocabulary_
        self.assertIn("fit", vocabulary)
        self.assertNotIn("calibration", vocabulary)
        self.assertNotIn("development", vocabulary)

    def test_rejects_holdout_group_overlap_and_normalized_duplicates_before_fit(self):
        variants = []
        holdout = records(); holdout[-1]["partition"] = "holdout"; variants.append(holdout)
        groups = records(); groups[8]["group_id"] = groups[0]["group_id"]; variants.append(groups)
        duplicate = records(); duplicate[8]["routing_text"] = "  " + duplicate[0]["routing_text"].upper().replace("\n", " "); variants.append(duplicate)
        task = records(); task[8]["task_id"] = task[0]["task_id"]; variants.append(task)
        unknown_cost = records(); unknown_cost[0]["branches"]["weak"]["cost_usd"] = None; variants.append(unknown_cost)
        for index, values in enumerate(variants):
            with self.subTest(index=index), patch("relay_router.maintenance_learning.TrajectorySuccessPredictor.fit") as fit:
                with self.assertRaises(ValueError):
                    train_policy(values, self.root / f"invalid-{index}")
                fit.assert_not_called()

    def test_missing_classes_are_not_replaced_with_uniform_probabilities(self):
        for partition in ("fit", "calibration"):
            values = records()
            for row in values:
                if row["partition"] == partition:
                    row["branches"]["weak"]["success"] = True
            with self.assertRaisesRegex(ValueError, "Both classes"):
                train_policy(values, self.root / f"constant-{partition}")

    def test_dedup_ignores_generated_plan_and_transient_test_diagnostics(self):
        values = records()
        original = {"task": "fix labels", "editable_files": [{"path": "src/a.py", "content": "return labels"}],
                    "read_only_tests": [], "plan": {"summary": "first plan"},
                    "verification": {"stderr": "C:/temporary/one"}}
        changed = {**original, "plan": {"summary": "different generated plan"},
                   "verification": {"stderr": "C:/temporary/two"}}
        values[0]["routing_text"] = json.dumps(original)
        values[8]["routing_text"] = json.dumps(changed)
        with patch("relay_router.maintenance_learning.TrajectorySuccessPredictor.fit") as fit:
            with self.assertRaisesRegex(ValueError, "duplicados"):
                train_policy(values, self.root / "generated-duplicate")
            fit.assert_not_called()

    def test_policy_and_artifact_tampering_fail_before_joblib(self):
        for target in ("policy.json", "predictor.joblib", "predictor.joblib.manifest.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                clone = Path(temporary) / "clone"
                shutil.copytree(self.root / "frozen", clone)
                path = clone / target
                path.write_bytes(path.read_bytes() + b" ")
                with patch("routerbench.agent_trajectory_learning.joblib.load") as unsafe_load:
                    with self.assertRaises(ValueError):
                        load_policy(clone / "policy.json", self.result["policy_sha256"])
                    unsafe_load.assert_not_called()

    def test_threshold_change_requires_new_external_policy_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            clone = Path(temporary) / "clone"
            shutil.copytree(self.root / "frozen", clone)
            path = clone / "policy.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["reference_threshold"] = 1. - value["reference_threshold"]
            path.write_text(json.dumps(value), encoding="utf-8")
            with patch("routerbench.agent_trajectory_learning.joblib.load") as unsafe_load:
                with self.assertRaises(ValueError):
                    load_policy(path, self.result["policy_sha256"])
                unsafe_load.assert_not_called()

    def test_holdout_audit_rejects_seen_development_and_accepts_unseen_groups(self):
        policy = load_policy(self.result["policy_path"], self.result["policy_sha256"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            policy.audit_records([records()[-1]])
        audit = policy.audit_records([{"group_id": "new-group", "routing_text": "unseen new task"}])
        self.assertEqual(audit["evaluation_split"], "holdout")
        self.assertFalse(audit["fit_calibration_overlap"])

    def test_acquisition_contract_rejects_mixed_missing_and_diagnostic_before_fit(self):
        for variant in ("mixed", "missing", "diagnostic"):
            values = records()
            for row in values:
                row["acquisition_contract"] = acquisition_contract()
            if variant == "mixed":
                values[-1]["acquisition_contract"]["tariffs_sha256"] = "d" * 64
            elif variant == "missing":
                values[-1].pop("acquisition_contract")
            else:
                for row in values:
                    row["acquisition_contract"]["scope"] = "diagnostic"
            with self.subTest(variant=variant), patch("relay_router.maintenance_learning.TrajectorySuccessPredictor.fit") as fit:
                with self.assertRaises(ValueError):
                    train_policy(values, self.root / f"bad-contract-{variant}")
                fit.assert_not_called()

    def test_bound_acquisition_contract_is_required_for_holdout_audit(self):
        values = records()
        for row in values:
            row["acquisition_contract"] = acquisition_contract()
            row["reference_probability"] = 1.
        result = train_policy(values, self.root / "contract-policy")
        policy = load_policy(result["policy_path"], result["policy_sha256"])
        self.assertEqual(policy.manifest()["acquisition_contract"], acquisition_contract())
        self.assertEqual(policy.manifest()["reference_threshold"], 1.000001)
        new = {"group_id": "unseen-group", "routing_text": "unseen code request",
               "acquisition_contract": acquisition_contract()}
        self.assertEqual(policy.audit_records([new])["tasks"], 1)
        altered = deepcopy(new)
        altered["acquisition_contract"]["suite_sha256"] = "e" * 64
        for row in (altered, {"group_id": new["group_id"], "routing_text": new["routing_text"]}):
            with self.assertRaisesRegex(ValueError, "acquisition_contract"):
                policy.audit_records([row])


if __name__ == "__main__":
    unittest.main()
