"""Local fixtures exercise external-runner gates without provider requests."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from test_agent_trajectory_learning import MODELS, rows
from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor
from run_agent_api_v8 import _policies
from relay_trajectory_local import paired_training_rows
from train_agent_trajectory import parse_dataset


class TrajectoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def configuration(self, *, assignment="paired_state"):
        # Declaring paired fixture rows here tests plumbing only. They are never
        # published as observed trajectories or passed to an API generator.
        fixture_rows = lambda part: [replace(row, assignment=assignment) for row in rows(part)]
        predictor = TrajectorySuccessPredictor(MODELS).fit(fixture_rows("fit"), fixture_rows("calibration"))
        predictor.evaluate(fixture_rows("development"))
        artifact = self.directory / "fixture.joblib"
        sha = predictor.save(artifact)
        return {"models": {"strong": MODELS[-1], "weak": MODELS[0]}, "model_catalog": list(MODELS),
            "evaluation_split": "development", "policies": ["always_strong", "relay_v8_trajectory"],
            "trajectory_policy": {"artifact_path": str(artifact), "artifact_sha256": sha,
                "min_probability": .5, "operating_point_source": "unit fixture only"}}

    @staticmethod
    def tasks(part):
        return [{"source_group": row.source_group, "prompt": row.request.prompt}
                for row in rows(part) if row.model == MODELS[0]]

    def test_external_runner_rejects_synthetic_training_fixture(self):
        config = self.configuration(assignment="synthetic_fixture")
        with self.assertRaisesRegex(ValueError, "not synthetic training fixtures"):
            _policies(config, self.tasks("unseen"))

    def test_tasks_required_for_overlap_audit(self):
        config = self.configuration()
        with self.assertRaisesRegex(ValueError, "requires tasks"):
            _policies(config)
        with self.assertRaisesRegex(ValueError, "Audit requires tasks"):
            _policies(config, [])

    def test_fit_cal_and_development_have_distinct_evaluation_guards(self):
        config = self.configuration()
        for part in ("fit", "calibration"):
            with self.subTest(part=part), self.assertRaisesRegex(ValueError, "fit/calibration"):
                _policies(config, self.tasks(part))
        policies = _policies(config, self.tasks("development"))
        self.assertEqual([policy.name for policy in policies], config["policies"])
        holdout = {**config, "evaluation_split": "evaluation"}
        with self.assertRaisesRegex(ValueError, "previously inspected development"):
            _policies(holdout, self.tasks("development"))
        self.assertEqual(len(_policies(holdout, self.tasks("unseen"))), 2)

    def test_runner_requires_catalog_order_and_external_artifact_hash(self):
        config = self.configuration()
        with self.assertRaisesRegex(ValueError, "catalog in order"):
            _policies({**config, "model_catalog": list(MODELS[::-1])}, self.tasks("unseen"))
        changed = deepcopy(config)
        changed["trajectory_policy"].pop("artifact_sha256")
        with self.assertRaisesRegex(ValueError, "external SHA-256"):
            _policies(changed, self.tasks("unseen"))

    def test_collector_rows_parse_as_schema_two_and_censor_entire_pair(self):
        from test_agent_rollouts import configuration, tariffs
        from routerbench.agent_rollouts import initial_anchors, collect_paired_rollouts
        from routerbench.agent_simulation import ScriptedToolFixture
        from routerbench.agent_tasks import build_tasks
        config = configuration()
        models = config["model_catalog"]
        tasks = [build_tasks(per_family=1)[1]]
        result = collect_paired_rollouts(tasks, initial_anchors(tasks, config), models, ScriptedToolFixture(),
            config, self.directory / "collection", max_generation_calls=128, tariffs=tariffs(models), execute=True)
        collected, excluded = paired_training_rows(result, models)
        self.assertEqual(excluded, 0)
        self.assertEqual(len(collected), len(models))
        _, partitions = parse_dataset({"schema_version": 2, "model_catalog": models,
            "fit": collected, "calibration": collected, "development": collected})
        self.assertEqual(partitions["fit"][0].assignment, "synthetic_fixture")
        # Parser accepts structure; fitting these repeated partitions would fail
        # overlap checks. No one-class fixture labels are made trainable.
        incomplete = deepcopy(result)
        incomplete["branches"][0]["censored_reason"] = "infrastructure_failure"
        self.assertEqual(paired_training_rows(incomplete, models), ([], 1))


if __name__ == "__main__":
    unittest.main()
