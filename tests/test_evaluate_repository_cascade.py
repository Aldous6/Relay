"""Paired repository acquisition and accounting regressions, with no model/network."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_repository_cascade as runner
from repository_reference import PINNED
from relay_router.cascade import AttemptOutcome
from relay_router.cascade_verifier import CheckCommand, CommandVerifier


class RepositoryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.path = self.work / "cohort.json"
        self.case = {"task_id": "heldout-repo", "group_id": "owner/project", "source": "https://example.org/project",
                     "revision": "abc", "prompt": "Correct multiplication across supported types.\n    Keep the API."}
        self.cohort = {"cohort_id": "offline-test", "tasks": [self.case]}
        runner.write(self.path, self.cohort)
        self.score_path = self.work / "scores.json"
        self.scores = {"reference": {"checkpoint_verified": True, "predictor_injected": False,
                       "revision": PINNED["checkpoint_revision"], "weights_sha256": PINNED["weights_sha256"],
                       "tokenization": PINNED["reference_tokenization"], "catalog": list(runner.CATALOG),
                       "load_seconds": 5.},
                       "cohort_sha256": runner.file_hash(self.path),
                       "score_source_sha256": {name: runner.file_hash(ROOT / name) for name in (
                          "tools/repository_reference.py", "src/relay_router/maintenance_reference.py",
                          "src/routerbench/routellm_reference.py", "configs/routellm_reference.json")},
                       "rows": [{"task_id": self.case["task_id"], "routing_text_sha256": hashlib.sha256(
                           self.case["prompt"].encode()).hexdigest(), "probability": .1,
                           "truncated_tokens": 0, "router_seconds": 1.}]}
        self.scores["routes"] = runner.frozen_routes(self.cohort, self.path, self.scores)
        self.scores.update(threshold=.50, truncated_route="fixed_strong")
        runner.write(self.score_path, self.scores)

    def test_reference_is_prompt_only_and_sealed_before_generation(self):
        routes = runner.frozen_routes(self.cohort, self.path, self.scores)
        self.assertEqual(routes[self.case["task_id"]]["arm"], "fixed_cheap")
        altered = copy.deepcopy(self.scores)
        altered["rows"][0]["truncated_tokens"] = 1
        self.assertEqual(runner.frozen_routes(self.cohort, self.path, altered)[self.case["task_id"]]["arm"], "fixed_strong")
        altered["rows"][0]["routing_text_sha256"] = "changed"
        with self.assertRaises(ValueError):
            runner.frozen_routes(self.cohort, self.path, altered)
        other = {**self.case, "task_id": "another-issue", "prompt": "different"}
        bad = self.work / "invalid.json"
        runner.write(bad, {**self.cohort, "tasks": [self.case, other]})
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.load_cohort(bad)

    def test_bounds_fail_closed_and_the_cost_of_failed_tasks_is_preserved(self):
        raw = {"accepted": False, "private_seconds": 1., "run_seconds": 3.,
               "provider": {"cost_complete": False, "cost_usd": None, "known_cost_subtotal_usd": .001,
                            "cost_upper_bound_usd": .01}}
        row = runner.cost_row(self.case, "relay", raw, {}, 3.6)
        self.assertIsNone(row["cost_usd"])
        self.assertAlmostEqual(row["known_cost_subtotal_usd"], .005)
        self.assertAlmostEqual(row["cost_upper_bound_usd"], .014)
        self.assertIsNone(runner.summarize_policy([row])["cost_per_accepted_usd"])
        self.assertIsNone(runner.cost_row(self.case, "relay", raw, {}, None)["cost_upper_bound_usd"])
        raw["provider"]["cost_upper_bound_usd"] = None
        self.assertIsNone(runner.cost_row(self.case, "relay", raw, {}, 3.6)["cost_upper_bound_usd"])

    def test_passing_workspace_without_verified_post_cannot_count_as_a_model_success(self):
        run = {"accounting": {"calls": []}, "selected_attempt": 1, "selected_model": runner.CHEAP,
               "attempts": [{"call_ids": [], "status": "completed", "verification": {"status": "passed"}}],
               "reason": "operational_checks_passed", "initial_verification": None,
               "escalated": False, "operational_passed": True}
        raw = runner.observation(run, 1., "passed", 1.)
        self.assertFalse(raw["accepted"])
        self.assertTrue(raw["selected_without_verified_response"])

    def test_unpriced_secondary_control_blocks_pilot_and_report_excludes_raw_content(self):
        cases = [{**self.case, "task_id": f"task-{i}", "group_id": f"group-{i}"} for i in range(20)]
        routes = {t["task_id"]: {"arm": "fixed_strong", "router_seconds": .01, "amortized_load_seconds": .01} for t in cases}
        observations = {}
        for arm, amount in (("relay", 1.), ("fixed_strong", 2.), ("fixed_cheap", None)):
            observations[arm] = {"accepted": True, "run_seconds": 1., "private_seconds": .1,
                   "operational_verifier_seconds": .1, "private_status": "passed", "errors_before_dispatch": [],
                   "public_check_inconclusive": False, "selected_without_verified_response": False,
                   "provider": {"cost_usd": amount, "cost_complete": amount is not None,
                                "known_cost_subtotal_usd": amount if amount is not None else .1,
                                "cost_upper_bound_usd": 3., "http_calls": 1}}
        frozen = {"freeze_id": "fixture", "contract": {"routes": routes, "usd_per_dedicated_host_hour": .50}}
        with patch.object(runner, "load_episode", return_value=observations):
            result = runner.report({"cohort_id": "offline-test", "tasks": cases}, self.work, frozen)
        self.assertFalse(result["all_operating_costs_point_identified"])
        self.assertFalse(result["pilot_criteria_met"])
        self.assertTrue(result["savings_at_least_10_percent_even_at_adverse_cost_endpoints"])
        self.assertNotIn(self.case["prompt"], str(result))
        self.assertNotIn("prompt_cache", str(result))

    def test_real_cascade_three_paired_arms_and_resumption_without_repeat(self):
        snapshot = self.work / "snapshot"
        snapshot.mkdir()
        (snapshot / "value.txt").write_text("original")
        (snapshot / "check.py").write_text("import pathlib, sys\nsys.exit(pathlib.Path('value.txt').read_text() != 'fixed')")
        verifier = CommandVerifier.capture(snapshot, checks=[CheckCommand("public", (sys.executable, "-I", "-B", "check.py"))],
                                           protected_files=["check.py"])
        calls, private = [], []

        def make_attempt(*args, **kwargs):
            def attempt(context):
                self.assertEqual((context.workspace / "value.txt").read_text(), "original")
                calls.append(context.model)
                call = context.budget.begin_call(context.model, .01)
                context.budget.finish_call(call, cost_usd=.002, input_tokens=10, output_tokens=10, http_status=200)
                if context.model == runner.STRONG:
                    (context.workspace / "value.txt").write_text("fixed")
                return AttemptOutcome("completed")
            return attempt

        def private_verdict(task, selected):
            self.assertEqual(len(calls), 4)
            self.assertEqual((selected / "value.txt").read_text(), "fixed")
            private.append(selected)
            return "passed"

        dummy_host_path = self.work / "host.py"
        dummy_host_path.write_text("# test host whose calls are injected offline")
        output = self.work / "campaign"
        mock_host = types.SimpleNamespace(
            prepare=lambda task: (snapshot, verifier, {"private": dummy_host_path}),
            build_agent=None, private_verdict=private_verdict)
        with patch.object(runner, "make_openhands_attempt", side_effect=make_attempt), \
                patch.object(runner, "load_host", return_value=mock_host), \
                patch.object(runner, "SOURCES", ("tools/evaluate_repository_cascade.py",)), \
                patch.object(runner.importlib.metadata, "version", return_value="fixture"):
            runner.freeze(self.path, self.score_path, dummy_host_path, output, 3.6, "fixture host rate")
            mock_host.prepare_acquisition = Mock(side_effect=[ValueError("missing fixture key"), None])
            with self.assertRaisesRegex(ValueError, "missing fixture key"):
                runner.acquire(self.path, self.score_path, dummy_host_path, output)
            self.assertFalse((output / "episodes").exists())
            first = runner.acquire(self.path, self.score_path, dummy_host_path, output)
            second = runner.acquire(self.path, self.score_path, dummy_host_path, output)
            self.assertEqual(first, second)
            self.assertEqual(mock_host.prepare_acquisition.call_count, 2)
            self.assertEqual(len(private), 2)
            self.assertEqual(first["summary"]["relay"]["accepted"], 1)
            self.assertEqual(first["summary"]["fixed_cheap"]["accepted"], 0)
            raw = first["rows"]["relay"][0]
            self.assertAlmostEqual(raw["provider"]["cost_usd"], .004)
            self.assertTrue(raw["escalated"])
            self.assertGreater(raw["host_cost_scenario_usd"], 0)
            self.assertEqual(first["rows"][runner.BERT][0]["router_seconds"], 6.)
            self.assertEqual(first["rows"][runner.BERT][0]["provider"], first["rows"]["fixed_cheap"][0]["provider"])
            self.assertGreater(first["rows"][runner.BERT][0]["cost_usd"], first["rows"]["fixed_cheap"][0]["cost_usd"])
            self.assertFalse(first["pilot_criteria_met"])
            log = output / "episodes" / self.case["task_id"] / "relay" / "transport.jsonl"
            with log.open("a") as stream:
                stream.write("changed")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                runner.acquire(self.path, self.score_path, dummy_host_path, output)
            self.assertEqual(len(calls), 4)

    def test_public_preflight_all_originals_only_and_zero_opportunity(self):
        host_path = self.work / "preflight-host.py"
        host_path.write_text("# test public host")
        cases = [{**self.case, "task_id": f"repo-{n}", "group_id": f"group-{n}", "source": f"repo-{n}",
                  "prompt": f"public issue {n}"} for n in range(3)]
        for n, case in enumerate(cases):
            snapshot = self.work / case["task_id"]
            snapshot.mkdir()
            (snapshot / "public.py").write_text(
                f"import pathlib, sys\npathlib.Path('copy-only').write_text('ran')\nsys.exit({n})")

        def prepare_public(task):
            snapshot = self.work / task["task_id"]
            verifier = CommandVerifier.capture(snapshot, checks=[CheckCommand("public", (sys.executable, "-I", "-B", "public.py"))],
                                               protected_files=["public.py"])
            return snapshot, verifier

        dummy = types.SimpleNamespace(prepare_public=Mock(side_effect=prepare_public),
            prepare=lambda task: (*prepare_public(task), {"private": host_path}),
            build_agent=Mock(side_effect=AssertionError("no agent")), private_verdict=Mock(side_effect=AssertionError("no private access")),
            prepare_acquisition=Mock(side_effect=AssertionError("no model credential")))
        for chosen in (cases, cases[:1]):
            casepath = self.work / f"{len(chosen)}-cases.json"
            out = self.work / f"{len(chosen)}-preflight"
            runner.write(casepath, {"cohort_id": "initial-routes-fixture", "tasks": chosen})
            routes = {t["task_id"]: {"arm": "fixed_strong", "router_seconds": 0., "amortized_load_seconds": 0.} for t in chosen}
            scorepath = self.work / f"{len(chosen)}-scores.json"
            runner.write(scorepath, {"routes": routes, "threshold": .50, "truncated_route": "fixed_strong"})
            with patch.object(runner, "load_host", return_value=dummy), \
                    patch.object(runner, "SOURCES", ("tools/evaluate_repository_cascade.py",)), \
                    patch.object(runner.importlib.metadata, "version", return_value="test"):
                with patch.object(runner, "frozen_routes", return_value=routes):
                    runner.freeze(casepath, scorepath, host_path, out, None, None)
                with patch.object(dummy, "prepare", side_effect=AssertionError("private provisioning was called")), \
                        patch.object(runner, "frozen_routes", side_effect=AssertionError("BERT was consulted")):
                    report = runner.preflight_routes(casepath, host_path, out)
                    self.assertEqual(report, runner.preflight_routes(casepath, host_path, out))
            self.assertEqual(report["tasks"], len(chosen))
            self.assertEqual(report["provider_requests"], 0)
            self.assertEqual(report["direct_120b"], 1)
            self.assertIsNone(report["acceptance"])
            self.assertIsNone(report["savings"])
            self.assertFalse(any((self.work / t["task_id"] / "copy-only").exists() for t in chosen))
            self.assertGreater(report["total_verifier_wall_seconds"], 0)
            self.assertEqual([r["initial_route"] for r in report["rows"]], ["direct_120b", "candidate_20b", "no_dispatch"][:len(chosen)])
            if len(chosen) == 1:
                self.assertEqual(report["candidate_20b_opportunities"], 0)
                self.assertIn("no_observed", report["finding"])
        self.assertEqual(dummy.prepare_public.call_count, 4)
        dummy.build_agent.assert_not_called()
        dummy.private_verdict.assert_not_called()
        dummy.prepare_acquisition.assert_not_called()


if __name__ == "__main__":
    unittest.main()
