"""Frozen ensembles validate compatibility and blend without mutating members."""
import copy
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from routerbench.ensemble import FrozenRegretEnsemble
from routerbench.pairwise import OUTCOME_CLASSES


class FittedToyRouter:
    def __init__(self, probabilities, costs, index=(11, 22)):
        self.strong, self.weak = "strong", "weak"
        self.models_ = (self.strong, self.weak)
        self.quality_threshold = .75
        self.fit_rows_, self.calibration_rows_ = 6, 3
        self.target_counts_ = {"fit": dict.fromkeys(OUTCOME_CLASSES, 2),
                               "calibration": dict.fromkeys(OUTCOME_CLASSES, 1)}
        self.probabilities = pd.DataFrame(probabilities, columns=OUTCOME_CLASSES, index=index)
        self.probabilities["expected_quality_gain"] = self.probabilities["strong_only"]-self.probabilities["weak_only"]
        self.costs = pd.DataFrame(costs, columns=self.models_, index=index)

    def predict_tables(self, inputs):
        return self.probabilities, self.costs


def members():
    return [FittedToyRouter([[.6, .1, .3], [.2, .4, .4]], [[.01, .001], [.02, .002]]),
            FittedToyRouter([[.2, .2, .6], [.4, .1, .5]], [[.02, .002], [.01, .001]])]


def inputs():
    return pd.DataFrame({"prompt_text": ["question one", "question two"]}, index=[11, 22])


def replace_column(router, frame_name, column, values):
    frame = getattr(router, frame_name)
    frame.loc[:, column] = values


class EnsembleTests(unittest.TestCase):
    def test_weighted_blend_recomputes_gain_and_does_not_mutate_members(self):
        routers = members()
        before = [(router.probabilities.copy(deep=True), router.costs.copy(deep=True),
                   copy.deepcopy(router.target_counts_)) for router in routers]
        ensemble = FrozenRegretEnsemble(routers, [1, 3])
        p, cost = ensemble.predict_tables(inputs())
        expected = .25*before[0][0].loc[:, OUTCOME_CLASSES]+.75*before[1][0].loc[:, OUTCOME_CLASSES]
        np.testing.assert_allclose(p.loc[:, OUTCOME_CLASSES], expected)
        np.testing.assert_allclose(cost, .25*before[0][1]+.75*before[1][1])
        np.testing.assert_allclose(p["expected_quality_gain"], p["strong_only"]-p["weak_only"])
        self.assertTrue(np.allclose(p.loc[:, OUTCOME_CLASSES].sum(axis=1), 1))
        self.assertEqual(ensemble.weights_, (.25, .75))
        self.assertEqual(ensemble.models_, ("strong", "weak"))
        self.assertFalse(hasattr(ensemble, "fit"))
        for router, (probability, costs, targets) in zip(routers, before):
            pd.testing.assert_frame_equal(router.probabilities, probability)
            pd.testing.assert_frame_equal(router.costs, costs)
            self.assertEqual(router.target_counts_, targets)
        ensemble.target_counts_["fit"]["equal"] = 0
        self.assertEqual(routers[0].target_counts_["fit"]["equal"], 2)

    def test_joblib_roundtrip_and_large_finite_cost_average(self):
        routers = members()
        routers[0].costs.loc[:, :] = np.finfo(float).max
        routers[1].costs.loc[:, :] = np.finfo(float).max
        ensemble = FrozenRegretEnsemble(routers, [1e308, 1e308])
        original = ensemble.predict_tables(inputs())
        self.assertTrue(np.isfinite(original[1].to_numpy()).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.joblib"
            joblib.dump(ensemble, path)
            restored = joblib.load(path)
            replay = restored.predict_tables(inputs())
        for expected, actual in zip(original, replay):
            pd.testing.assert_frame_equal(expected, actual)

    def test_invalid_weights_and_incompatible_member_metadata_are_rejected(self):
        for weights in ([1], [0, 1], [-1, 2], [np.nan, 1], [np.inf, 1], [[1, 2]], [True, True], ["1", "2"], [1e-320, 1e308]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                FrozenRegretEnsemble(members(), weights)
        with self.assertRaises(ValueError):
            FrozenRegretEnsemble(members()[:1])
        for attribute, value in (("models_", ("weak", "strong")), ("strong", "wrong"),
                                 ("quality_threshold", .5), ("fit_rows_", 9),
                                 ("calibration_rows_", 9),
                                 ("target_counts_", {"fit": {"strong_only": 1, "weak_only": 2, "equal": 3},
                                                     "calibration": dict.fromkeys(OUTCOME_CLASSES, 1)})):
            routers = members()
            setattr(routers[1], attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                FrozenRegretEnsemble(routers)

    def test_invalid_predictions_costs_and_row_alignment_are_rejected(self):
        changes = [lambda router: setattr(router, "probabilities", router.probabilities.iloc[::-1]),
                   lambda router: replace_column(router, "probabilities", "equal", [.9, .9]),
                   lambda router: replace_column(router, "probabilities", "strong_only", [np.nan, .2]),
                   lambda router: replace_column(router, "probabilities", "expected_quality_gain", [0, 0]),
                   lambda router: setattr(router, "costs", router.costs.iloc[::-1]),
                   lambda router: replace_column(router, "costs", "strong", [-1, .02]),
                   lambda router: replace_column(router, "costs", "weak", [np.inf, .002])]
        for mutate in changes:
            routers = members()
            ensemble = FrozenRegretEnsemble(routers)
            mutate(routers[1])
            with self.subTest(change=mutate), self.assertRaises(ValueError):
                ensemble.predict_tables(inputs())
        routers = members()
        ensemble = FrozenRegretEnsemble(routers)
        routers[1].quality_threshold = .5
        with self.assertRaisesRegex(ValueError, "cambiaron"):
            ensemble.predict_tables(inputs())


if __name__ == "__main__":
    unittest.main()
