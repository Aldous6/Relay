"""Offline fixtures exercise policy gates; their savings are not product evidence."""
from copy import deepcopy
import hashlib
import json
import unittest

from relay_router.workflow_policy import propose_policy, validate_artifact, validate_policy


def records(split, *, workflow="triage", count=3, candidate="fixture-small", candidate_cost=.4):
    return [dict(schema_version=1, kind="relay_task_observation", workflow=workflow,
                 task_id=f"{split}-{index}", group_id=f"{split}-group-{index}", split=split,
                 model=model, accepted=True, cost_usd=cost, cost_basis="tariff_estimate",
                 evaluation_id="synthetic-contract-v1")
            for index in range(count) for model, cost in (("fixture-baseline", 1.), (candidate, candidate_cost))]


def proposal(rows=None, **kwargs):
    return propose_policy(records("development") if rows is None else rows,
                          baseline_model="fixture-baseline", min_groups=3, **kwargs)


def resign(value):
    value = deepcopy(value)
    value.pop("policy_id")
    value["policy_id"] = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


class WorkflowPolicyTests(unittest.TestCase):
    def test_proposal_requires_separate_validation_before_deployment(self):
        candidate = proposal()
        self.assertFalse(candidate["ready"])
        self.assertEqual(candidate["workflows"]["triage"]["proposed_model"], "fixture-small")
        with self.assertRaises(ValueError):
            validate_artifact(candidate)
        policy = validate_policy(candidate, records("validation"))
        self.assertTrue(policy["ready"])
        self.assertEqual(policy["routes"], {"triage": "fixture-small"})
        self.assertEqual(policy["workflows"]["triage"]["status"], "approved_empirical")
        self.assertEqual(policy, validate_artifact(json.loads(json.dumps(policy))))
        self.assertIn("not_statistical_guarantee", policy["scope"])
        self.assertEqual(policy["data_provenance"], "user_supplied_observations_not_independently_verified")

    def test_unknown_cost_or_acceptance_blocks_substitution_without_imputing_zero(self):
        for field in ("accepted", "cost_usd"):
            with self.subTest(field=field):
                observed = records("validation")
                observed[1][field] = None
                policy = validate_policy(proposal(), observed)
                self.assertFalse(policy["ready"])
                self.assertEqual(policy["routes"]["triage"], "fixture-baseline")
                stats = policy["workflows"]["triage"]["validation"]
                self.assertEqual(stats["reason"], "unknown_outcome_or_cost")
                self.assertIsNone(stats["relative_cost_savings"])

    def test_single_model_logs_cannot_create_an_alternative_or_estimate_savings(self):
        dev = [row for row in records("development") if row["model"] == "fixture-baseline"]
        candidate = proposal(dev)
        self.assertEqual(candidate["candidate_models"], ["fixture-baseline"])
        self.assertEqual(candidate["workflows"]["triage"]["development_comparisons"], [])
        policy = validate_policy(candidate, [row for row in records("validation") if row["model"] == "fixture-baseline"])
        self.assertFalse(policy["ready"])
        self.assertEqual(policy["routes"]["triage"], "fixture-baseline")

    def test_missing_pairs_are_not_silently_dropped(self):
        observed = records("validation")[:-1]
        policy = validate_policy(proposal(), observed)
        self.assertFalse(policy["ready"])
        stats = policy["workflows"]["triage"]["validation"]
        self.assertEqual(stats["missing_pair_count"], 1)
        self.assertEqual(stats["paired_task_count"], 2)
        self.assertIsNone(stats["relative_cost_savings"])

    def test_quality_loss_in_validation_reverts_to_baseline(self):
        observed = records("validation")
        observed[1]["accepted"] = False
        policy = validate_policy(proposal(), observed)
        stats = policy["workflows"]["triage"]["validation"]
        self.assertEqual(stats["reason"], "quality_below_baseline")
        self.assertAlmostEqual(stats["quality_delta"], -1/3)
        self.assertEqual(policy["routes"]["triage"], "fixture-baseline")

    def test_validation_does_not_select_a_different_candidate(self):
        dev = records("development")
        other = [dict(row, model="fixture-other", cost_usd=.6) for row in dev if row["model"] == "fixture-small"]
        candidate = proposal(dev + other)
        observed = records("validation")
        observed[1]["accepted"] = False
        other = [dict(row, model="fixture-other", cost_usd=.2, accepted=True)
                 for row in observed if row["model"] == "fixture-small"]
        policy = validate_policy(candidate, observed + other)
        self.assertEqual(policy["routes"]["triage"], "fixture-baseline")
        self.assertEqual(policy["workflows"]["triage"]["proposed_model"], "fixture-small")

    def test_groups_and_task_ids_cannot_leak_between_selection_and_validation(self):
        for field in ("group_id", "task_id"):
            observed = records("validation")
            for row in observed[:2]:
                row[field] = records("development")[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_policy(proposal(), observed)

    def test_group_repetitions_do_not_inflate_support(self):
        dev = records("development", count=30)
        for row in dev:
            row["group_id"] = "one-duplicate-family"
        candidate = propose_policy(dev, baseline_model="fixture-baseline")
        stats = candidate["workflows"]["triage"]["development_comparisons"][0]
        self.assertEqual(stats["group_count"], 1)
        self.assertEqual(stats["reason"], "insufficient_groups")
        self.assertEqual(candidate["workflows"]["triage"]["proposed_model"], "fixture-baseline")

    def test_equal_group_weighting_does_not_hide_rare_group_failures(self):
        dev = records("development", count=20)
        for row in dev:
            last = row["task_id"].endswith("-19")
            row["group_id"] = "rare" if last else "frequent"
            if last and row["model"] == "fixture-small":
                row["accepted"] = False
        candidate = propose_policy(dev, baseline_model="fixture-baseline", min_groups=2)
        stats = candidate["workflows"]["triage"]["development_comparisons"][0]
        self.assertEqual(stats["candidate_success_rate"], .5)
        self.assertEqual(stats["quality_delta"], -.5)
        self.assertFalse(stats["qualified"])

    def test_cost_basis_must_be_comparable_across_models_and_splits(self):
        for change in ("one", "all"):
            observed = records("validation")
            for row in observed if change == "all" else observed[:1]:
                row["cost_basis"] = "provider_reported"
            policy = validate_policy(proposal(), observed)
            self.assertFalse(policy["ready"])
            reason = policy["workflows"]["triage"]["validation"]["reason"]
            self.assertEqual(reason, "cost_basis_changed_since_development" if change == "all" else "incomparable_cost_bases")

    def test_missing_workflow_validation_uses_baseline_without_borrowing_other_tasks(self):
        candidate = proposal(records("development") + records("development", workflow="editing"))
        policy = validate_policy(candidate, records("validation"))
        self.assertTrue(policy["ready"])
        self.assertEqual(policy["routes"], {"editing": "fixture-baseline", "triage": "fixture-small"})
        self.assertEqual(policy["workflows"]["editing"]["status"], "baseline_missing_validation")

    def test_no_accepted_tasks_never_promotes_a_cheaper_failure(self):
        dev = records("development")
        for row in dev:
            row["accepted"] = False
        candidate = proposal(dev)
        self.assertEqual(candidate["workflows"]["triage"]["development_comparisons"][0]["reason"], "no_accepted_tasks")

    def test_costs_include_all_declared_task_spend_instead_of_initial_call_only(self):
        observed = records("validation", candidate_cost=1.2)
        policy = validate_policy(proposal(), observed)
        self.assertEqual(policy["routes"]["triage"], "fixture-baseline")
        self.assertAlmostEqual(policy["workflows"]["triage"]["validation"]["relative_cost_savings"], -.2)

    def test_unknowns_invalid_numbers_duplicates_and_changed_contracts(self):
        bad_updates = [{"cost_usd": float("nan")}, {"cost_usd": True}, {"cost_usd": -1},
                       {"cost_usd": 10**1000}, {"accepted": 1}, {"split": "production"},
                       {"cost_basis": "unknown"}, {"cost_basis": []}, {"schema_version": True},
                       {"evaluation_id": "changed"}, {"cost_complete": False}, {"cost_complete": "yes"}]
        for update in bad_updates:
            rows = records("development")
            rows[0].update(update)
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                proposal(rows)
        with self.assertRaises(ValueError):
            proposal(records("development") + records("development")[:1])
        with self.assertRaises(ValueError):
            propose_policy(records("development"), baseline_model=None)
        observed = records("validation")
        for row in observed:
            row["evaluation_id"] = "changed"
        with self.assertRaises(ValueError):
            validate_policy(proposal(), observed)

    def test_metadata_and_prompts_do_not_enter_artifacts_or_change_selection(self):
        rows = records("development")
        for row in rows:
            row.update(prompt="PRIVATE_CONTENT", expected_answer="DO_NOT_USE", decision={"future_score": 1})
        self.assertEqual(proposal(), proposal(rows))
        self.assertNotIn("PRIVATE_CONTENT", json.dumps(proposal(rows)))

    def test_policy_hash_and_decision_consistency_are_checked_and_copy_is_independent(self):
        policy = validate_policy(proposal(), records("validation"))
        copied = validate_artifact(policy)
        copied["routes"]["triage"] = "fixture-baseline"
        self.assertEqual(policy["routes"]["triage"], "fixture-small")
        with self.assertRaises(ValueError):
            validate_artifact(copied)
        with self.assertRaises(ValueError):
            validate_artifact(resign(copied))
        candidate = proposal()
        candidate["workflows"]["triage"]["proposed_model"] = "fixture-baseline"
        with self.assertRaises(ValueError):
            validate_policy(candidate, records("validation"))


if __name__ == "__main__":
    unittest.main()
