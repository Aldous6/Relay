"""Synthetic contract fixtures do not establish product savings or real quality."""
from copy import deepcopy
import builtins
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router.adaptive_policy import (
    evaluate_policy, normalized_prompt_hash, prompt_features, score_prompt,
    select_model, train_policy, validate_artifact, validate_fresh_prompts, validate_report,
)


def records(prefix="historical", count=24):
    rows = []
    for index in range(count):
        risky = index % 2 == 1
        prompt = (f"{prefix} request {index:03}: add two integers." if not risky else
                  f"{prefix} request {index:03}: repair nested branching.\n" +
                  "def branch(x):\n    if x:\n        for y in x:\n            return y\n" * 15)
        for model, cost in (("strong", 1.), ("cheap", .2)):
            rows.append(dict(task_id=f"{prefix}-{index:03}", group_id=f"{prefix}-group-{index:03}",
                             prompt_text=prompt, model=model, accepted=model == "strong" or not risky,
                             cost_usd=cost, cost_basis="synthetic_fixture_usd"))
    return rows


def resign(value, field):
    value = deepcopy(value)
    value.pop(field, None)
    value[field] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def fixture_encoder(text):
    """Synthetic preference values test plumbing, not checkpoint performance."""
    return {"score": .8 if "branching" in text else .15, "input_tokens": 120, "truncated_tokens": 0}


class AdaptivePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = records()
        cls.policy = train_policy(cls.dev, baseline_model="strong", candidate_model="cheap")

    def test_exported_linear_inference_matches_fitted_sklearn_without_serialization(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        fit_groups = set(self.policy["training"]["fit_group_ids"])
        fit = [row for row in self.dev if row["model"] == "cheap" and row["group_id"] in fit_groups]
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", random_state=42))
        pipeline.fit([prompt_features(row["prompt_text"]) for row in fit], [int(not row["accepted"]) for row in fit])
        exported = json.loads(json.dumps(self.policy))
        for row in records("fresh", count=6)[::2]:
            expected = pipeline.predict_proba([prompt_features(row["prompt_text"])])[0, 1]
            self.assertAlmostEqual(score_prompt(exported, row["prompt_text"]), expected, places=14)
        self.assertEqual(exported["status"], "experimental")
        self.assertNotIn("prompt_text", json.dumps(exported))

    def test_inference_and_validation_do_not_import_optional_ml(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"sklearn", "numpy", "scipy", "torch", "transformers"}:
                raise AssertionError(f"Unexpected optional inference dependency: {name}")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            decision = select_model(self.policy, records("fresh")[0]["prompt_text"])
            report = evaluate_policy(self.policy, records("fresh"))
        self.assertEqual(decision["model"], "cheap")
        self.assertTrue(report["activation_eligible"])
        self.assertIsNone(decision["router_cost_usd"])

    def test_split_scaler_and_coefficients_never_learn_calibration_text_or_outcomes(self):
        changed = deepcopy(self.dev)
        calibration = set(self.policy["training"]["calibration_group_ids"])
        for row in changed:
            if row["group_id"] in calibration:
                row["prompt_text"] += "\n" + "extra " * 10000
                row["accepted"] = False
                row["cost_usd"] = None
        adapted = train_policy(changed, baseline_model="strong", candidate_model="cheap")
        self.assertEqual(adapted["predictor"], self.policy["predictor"])
        self.assertEqual(adapted["training"]["fit_group_ids"], self.policy["training"]["fit_group_ids"])
        self.assertEqual(adapted["threshold"], 0.)
        self.assertEqual(self.policy, train_policy(list(reversed(self.dev)), baseline_model="strong", candidate_model="cheap"))

    def test_constant_target_is_safe_even_when_every_observed_cheap_answer_passed(self):
        for outcome in (True, False):
            rows = records()
            for row in rows:
                if row["model"] == "cheap":
                    row["accepted"] = outcome
            artifact = train_policy(rows, baseline_model="strong", candidate_model="cheap")
            self.assertEqual(artifact["predictor"]["kind"], "constant_safe")
            self.assertEqual(artifact["threshold"], 0)
            self.assertEqual(select_model(artifact, "an arbitrary request")["model"], "strong")

    def test_unknown_fit_costs_do_not_remove_risk_training_observations(self):
        rows = deepcopy(self.dev)
        fit_groups = set(self.policy["training"]["fit_group_ids"])
        for row in rows:
            if row["group_id"] in fit_groups:
                row["cost_usd"] = None
        artifact = train_policy(rows, baseline_model="strong", candidate_model="cheap")
        self.assertEqual(artifact["predictor"], self.policy["predictor"])
        self.assertEqual(artifact["threshold"], self.policy["threshold"])
        self.assertEqual(artifact["training"]["fit_task_count"], self.policy["training"]["fit_task_count"])

    def test_both_models_failing_is_a_positive_candidate_failure_target(self):
        rows = deepcopy(self.dev)
        fit_groups = set(self.policy["training"]["fit_group_ids"])
        changed_task = next(row["task_id"] for row in rows
                            if row["group_id"] in fit_groups and row["model"] == "cheap" and row["accepted"])
        for row in rows:
            if row["task_id"] == changed_task:
                row["accepted"] = False
        artifact = train_policy(rows, baseline_model="strong", candidate_model="cheap")
        fit_candidate_rows = [row for row in rows if row["group_id"] in fit_groups and row["model"] == "cheap"]
        self.assertEqual(artifact["training"]["fit_positive_count"], sum(not row["accepted"] for row in fit_candidate_rows))
        self.assertEqual(artifact["training"]["fit_positive_count"], self.policy["training"]["fit_positive_count"] + 1)
        self.assertEqual(artifact["training"]["fit_task_count"], len(fit_candidate_rows))
        self.assertEqual(artifact["target_version"], "candidate-failure-v1")
        self.assertEqual(artifact["score_semantics"], "uncalibrated_candidate_failure_score_not_success_probability")
        for row in rows:
            if row["task_id"] == changed_task and row["model"] == "strong":
                row["accepted"] = True
        baseline_succeeds = train_policy(rows, baseline_model="strong", candidate_model="cheap")
        self.assertEqual(artifact["predictor"], baseline_succeeds["predictor"])

    def test_old_relative_target_artifact_cannot_load_as_the_new_objective(self):
        for field, value in (("target_version", None), ("target_version", "relative-failure-v1"),
                             ("score_semantics", "uncalibrated_relative_failure_score_not_success_probability")):
            artifact = deepcopy(self.policy)
            if value is None:
                artifact.pop(field)
            else:
                artifact[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_artifact(resign(artifact, "artifact_id"))

    def test_fresh_reserve_gate_and_all_fixed_controls(self):
        report = evaluate_policy(self.policy, records("fresh"))
        self.assertEqual(report["group_count"], 24)
        self.assertEqual(report["metrics"]["baseline"]["accepted_count"], 24)
        self.assertEqual(report["metrics"]["candidate"]["accepted_count"], 12)
        self.assertEqual(report["metrics"]["relay"]["accepted_count"], 24)
        self.assertAlmostEqual(report["metrics"]["relay"]["cost_per_accepted_task_usd"], .6)
        self.assertAlmostEqual(report["relative_savings"], .4)
        self.assertEqual(report["paired_losses"], 0)
        self.assertTrue(report["activation_eligible"])
        self.assertFalse(report["net_savings_verified"])
        self.assertGreater(report["paired_loss_upper_95"], 0)
        self.assertEqual(report, validate_report(json.loads(json.dumps(report)), self.policy))
        self.assertEqual(self.policy["status"], "experimental")

    def test_losses_cannot_be_hidden_by_equal_number_of_gains(self):
        fresh = records("fresh")
        fresh[1]["accepted"] = False  # cheap selected: one lost baseline success
        fresh[4]["accepted"] = False  # cheap selected: one gained success
        report = evaluate_policy(self.policy, fresh)
        self.assertEqual(report["paired_losses"], 1)
        self.assertEqual(report["paired_gains"], 1)
        self.assertEqual(report["metrics"]["baseline"]["accepted_count"], report["metrics"]["relay"]["accepted_count"])
        self.assertFalse(report["activation_eligible"])
        self.assertIn("observed_paired_quality_loss", report["reasons"])

    def test_unknown_unselected_candidate_cost_does_not_erase_realized_relay_cost(self):
        fresh = records("fresh")
        fresh[3]["cost_usd"] = None  # hard task, baseline selected
        report = evaluate_policy(self.policy, fresh)
        self.assertTrue(report["activation_eligible"])
        self.assertIsNone(report["metrics"]["candidate"]["total_cost_usd"])
        self.assertEqual(report["metrics"]["relay"]["unknown_cost_count"], 0)
        self.assertIsNotNone(report["metrics"]["relay"]["total_cost_usd"])
        self.assertIsNone(report["candidate_fixed_comparison"]["relative_savings_vs_candidate"])
        fresh[1]["cost_usd"] = None  # cheap selected, now actual policy cost unknown
        report = evaluate_policy(self.policy, fresh)
        self.assertFalse(report["activation_eligible"])
        self.assertIsNone(report["metrics"]["relay"]["cost_per_accepted_task_usd"])
        self.assertIsNone(report["relative_savings"])

    def test_failed_episode_cost_remains_in_numerator_and_zero_accepted_is_undefined(self):
        fresh = records("fresh")
        fresh[2]["accepted"] = False
        fresh[2]["cost_usd"] = 2.  # whole episode cost, including its failed attempts
        report = evaluate_policy(self.policy, fresh)
        self.assertEqual(report["metrics"]["relay"]["accepted_count"], 23)
        self.assertAlmostEqual(report["metrics"]["relay"]["total_cost_usd"], 15.4)
        self.assertAlmostEqual(report["metrics"]["relay"]["cost_per_accepted_task_usd"], 15.4 / 23)
        for row in fresh:
            row["accepted"] = False
        report = evaluate_policy(self.policy, fresh)
        self.assertIsNone(report["metrics"]["relay"]["cost_per_accepted_task_usd"])
        self.assertFalse(report["activation_eligible"])

    def test_all_old_development_ids_groups_and_normalized_prompts_are_forbidden(self):
        for field in ("task_id", "group_id", "prompt_text"):
            fresh = records("fresh")
            for row in fresh[:2]:
                row[field] = self.dev[0][field]
                if field == "prompt_text":
                    row[field] = "  " + row[field].upper().replace(" ", "   ") + "\n"
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_policy(self.policy, fresh)
            with self.subTest(preflight=field), self.assertRaises(ValueError):
                validate_fresh_prompts(self.policy, fresh[:1])

    def test_independent_groups_not_repeated_tasks_control_support(self):
        fresh = records("fresh")
        for row in fresh:
            row["group_id"] = "single-new-family"
        report = evaluate_policy(self.policy, fresh)
        self.assertEqual(report["group_count"], 1)
        self.assertFalse(report["activation_eligible"])
        self.assertIn("insufficient_validation_groups", report["reasons"])

    def test_missing_pairs_duplicate_groups_and_malformed_values_are_rejected(self):
        malformed = []
        malformed.append(records("fresh")[:-1])
        duplicate = records("fresh")
        for row in duplicate[2:4]:
            row["prompt_text"] = duplicate[0]["prompt_text"].upper()
        malformed.append(duplicate)
        for field, value in (("accepted", None), ("cost_usd", float("nan")), ("cost_usd", True), ("cost_usd", -1)):
            rows = records("fresh")
            rows[0][field] = value
            malformed.append(rows)
        mismatched = records("fresh")
        mismatched[1]["prompt_text"] += " "
        malformed.append(mismatched)
        for index, rows in enumerate(malformed):
            with self.subTest(case=index), self.assertRaises(ValueError):
                evaluate_policy(self.policy, rows)
        with self.assertRaises(ValueError):
            validate_fresh_prompts(self.policy, duplicate)

    def test_mixed_or_changed_cost_bases_never_qualify(self):
        for all_rows in (False, True):
            fresh = records("fresh")
            for row in fresh if all_rows else fresh[:1]:
                row["cost_basis"] = "other_tariff"
            report = evaluate_policy(self.policy, fresh)
            self.assertFalse(report["activation_eligible"])
            self.assertIn("cost_basis_changed_or_incomparable", report["reasons"])

    def test_artifact_and_report_integrity_and_gate_are_checked(self):
        modified = deepcopy(self.policy)
        modified["threshold"] = 1.
        with self.assertRaises(ValueError):
            validate_artifact(modified)
        modified = deepcopy(self.policy)
        modified["predictor"]["scale"][0] = 0
        with self.assertRaises(ValueError):
            validate_artifact(resign(modified, "artifact_id"))
        report = evaluate_policy(self.policy, records("fresh", count=4))
        self.assertFalse(report["activation_eligible"])
        for field, value in (("activation_eligible", True), ("reasons", []), ("relative_savings", .99)):
            modified = deepcopy(report)
            modified[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(resign(modified, "validation_report_id"), self.policy)
        changed_policy = deepcopy(self.policy)
        changed_policy["scope"] = "another-harness"
        with self.assertRaises(ValueError):
            validate_report(report, resign(changed_policy, "artifact_id"))

    def test_features_preserve_indentation_and_duplicate_normalization_is_separate(self):
        left, right = "def f():\n    return 1", "def f(): return 1"
        self.assertEqual(normalized_prompt_hash(left), normalized_prompt_hash(right))
        self.assertNotEqual(prompt_features(left), prompt_features(right))
        self.assertEqual(len(prompt_features(left)), 8)
        self.assertAlmostEqual(prompt_features(left)[0], math.log1p(len(left)))

    def test_bert_head_export_matches_sklearn_and_extracts_once_per_original_text(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        calls = []
        def scorer(text):
            calls.append(text)
            return fixture_encoder(text)
        artifact = train_policy(self.dev, baseline_model="strong", candidate_model="cheap",
                                representation="routellm-bert", scorer=scorer)
        self.assertEqual(len(calls), len({row["prompt_text"] for row in self.dev}))
        self.assertEqual(artifact["representation"], "routellm-bert")
        self.assertEqual(len(artifact["predictor"]["coefficients"]), 1)
        self.assertEqual(artifact["feature_names"], ["routellm_bert_preference_score"])
        self.assertEqual(artifact["training"]["fit_group_ids"], self.policy["training"]["fit_group_ids"])
        fit_groups = set(artifact["training"]["fit_group_ids"])
        fit = [row for row in self.dev if row["model"] == "cheap" and row["group_id"] in fit_groups]
        expected = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", random_state=42))
        expected.fit([[fixture_encoder(row["prompt_text"])["score"]] for row in fit], [int(not row["accepted"]) for row in fit])
        for row in records("fresh", count=4)[::2]:
            probability = expected.predict_proba([[fixture_encoder(row["prompt_text"])["score"]]])[0, 1]
            self.assertAlmostEqual(score_prompt(artifact, row["prompt_text"], scorer=fixture_encoder), probability, places=14)
        report = evaluate_policy(artifact, records("fresh"), scorer=fixture_encoder)
        self.assertTrue(report["activation_eligible"])
        self.assertEqual(validate_report(report, artifact), report)  # no encoder needed to verify aggregate evidence
        duplicated = deepcopy(self.dev)
        duplicated.extend(dict(row, task_id="another-episode-same-group") for row in self.dev[:2])
        calls.clear()
        train_policy(duplicated, baseline_model="strong", candidate_model="cheap", representation="routellm-bert", scorer=scorer)
        self.assertEqual(len(calls), len({row["prompt_text"] for row in duplicated}))

    def test_bert_requires_engine_and_exact_encoder_provenance(self):
        artifact = train_policy(self.dev, baseline_model="strong", candidate_model="cheap",
                                representation="routellm-bert", scorer=fixture_encoder)
        for operation in (lambda: score_prompt(artifact, "request"), lambda: select_model(artifact, "request"),
                          lambda: evaluate_policy(artifact, records("fresh")),
                          lambda: train_policy(self.dev, baseline_model="strong", candidate_model="cheap", representation="routellm-bert")):
            with self.assertRaises(ValueError):
                operation()
        for field in ("checkpoint", "revision", "weights_sha256"):
            changed = deepcopy(artifact)
            changed["encoder"][field] = "unverified-encoder"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_artifact(resign(changed, "artifact_id"))
        class WrongEncoder:
            provenance = {"checkpoint": "another-encoder"}
            def __call__(self, text):
                return fixture_encoder(text)
        with self.assertRaises(ValueError):
            select_model(artifact, "request", scorer=WrongEncoder())

    def test_bert_truncation_forces_baseline_in_runtime_calibration_and_evaluation(self):
        artifact = train_policy(self.dev, baseline_model="strong", candidate_model="cheap",
                                representation="routellm-bert", scorer=fixture_encoder)
        prompt = records("fresh")[0]["prompt_text"]
        normal = select_model(artifact, prompt, scorer=fixture_encoder)
        self.assertEqual(normal["model"], "cheap")
        def truncated(text):
            return dict(fixture_encoder(text), input_tokens=600, truncated_tokens=88)
        route = select_model(artifact, prompt, scorer=truncated)
        self.assertEqual(route["score"], normal["score"])
        self.assertEqual(route["model"], "strong")
        self.assertEqual(route["reason"], "baseline_after_encoder_truncation")
        self.assertEqual(route["input_tokens"], 600)
        self.assertEqual(route["truncated_tokens"], 88)
        report = evaluate_policy(artifact, records("fresh"), scorer=truncated)
        self.assertEqual(report["selected_candidate_count"], 0)
        self.assertEqual(report["relative_savings"], 0)
        self.assertFalse(report["activation_eligible"])
        calibration = set(artifact["training"]["calibration_group_ids"])
        calibration_texts = {row["prompt_text"] for row in self.dev if row["group_id"] in calibration}
        guarded = train_policy(self.dev, baseline_model="strong", candidate_model="cheap", representation="routellm-bert",
                               scorer=lambda text: truncated(text) if text in calibration_texts else fixture_encoder(text))
        self.assertEqual(guarded["predictor"], artifact["predictor"])
        self.assertEqual(guarded["threshold"], 0)
        self.assertEqual(guarded["training"]["encoder_truncated_task_count"], len(calibration_texts))

    def test_encoder_score_and_token_contract_rejects_invalid_or_missing_values(self):
        artifact = train_policy(self.dev, baseline_model="strong", candidate_model="cheap",
                                representation="routellm-bert", scorer=fixture_encoder)
        for result in ({"score": .1}, {"score": -1, "input_tokens": 3, "truncated_tokens": 0},
                       {"score": .1, "input_tokens": 600, "truncated_tokens": 0},
                       {"score": .1, "input_tokens": 3, "truncated_tokens": 4}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                select_model(artifact, "request", scorer=lambda text: result)


if __name__ == "__main__":
    unittest.main()
