import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from relay_router import RelayRouter, load_bundle
from routerbench.utility import choose_cost_utility, cost_utility_score


class PairDecider:
    strong, weak = 'strong', 'weak'

    def predict_tables(self, frame):
        return (pd.DataFrame({'strong_only': .2, 'weak_only': .1, 'equal': .7,
                              'expected_quality_gain': .1}, index=frame.index),
                pd.DataFrame({'strong': 3., 'weak': 1.}, index=frame.index))


class UtilitySDKTests(unittest.TestCase):
    def setUp(self):
        self.decider = PairDecider()
        self.p, self.c = self.decider.predict_tables(pd.DataFrame({'prompt_text': ['case']}))

    def test_signed_threshold_and_ties_reproduce_score_policy(self):
        score = cost_utility_score(self.p, self.c, 'strong', 'weak').iloc[0]
        for threshold, expected in [(-1., 'strong'), (np.nextafter(score, -np.inf), 'strong'),
                                    (score, 'weak'), (np.nextafter(score, np.inf), 'weak')]:
            decision = choose_cost_utility(self.p, self.c, 'strong', 'weak', threshold)
            self.assertEqual(decision.selected_model.iloc[0], expected)
        self.assertAlmostEqual(score, .05)

    def test_equal_outcomes_do_not_force_the_strong_model(self):
        self.p.loc[:, ['strong_only', 'weak_only', 'expected_quality_gain']] = 0.
        self.p['equal'] = 1.
        self.assertEqual(choose_cost_utility(self.p, self.c, 'strong', 'weak', 0).selected_model.iloc[0], 'weak')

    def test_policy_rejects_invalid_extra_cost_simplex_or_declared_gain(self):
        for costs in [self.c.assign(strong=1.), self.c.assign(strong=.5), self.c.assign(weak=np.nan)]:
            with self.assertRaises(ValueError):
                choose_cost_utility(self.p, costs, 'strong', 'weak', .1)
        for probabilities in [self.p.assign(equal=.8), self.p.assign(expected_quality_gain=1.), self.p.assign(future_score=1.)]:
            with self.assertRaises(ValueError):
                choose_cost_utility(probabilities, self.c, 'strong', 'weak', .1)
        for cutoff in [True, np.nan, np.inf, None]:
            with self.assertRaises(ValueError):
                choose_cost_utility(self.p, self.c, 'strong', 'weak', cutoff)

    def test_sdk_utility_preserves_auditable_estimates_and_calls_only_callback(self):
        router = RelayRouter(self.decider, {'strong': 'strong', 'weak': 'weak'},
                             policy_mode='pairwise_utility', tradeoff=.1, cost_unit='historical_cost')
        with patch('socket.create_connection') as network:
            result = router.run('full\n    input', lambda model, prompt: (model, prompt))
        network.assert_not_called()
        self.assertEqual(result['response'], ('weak', 'full\n    input'))
        self.assertAlmostEqual(result['decision']['expected_quality_gain'], .1)
        self.assertAlmostEqual(result['decision']['operating_score'], .05)
        self.assertIsNone(result['decision']['harm_probability'])
        json.dumps(result['decision'], allow_nan=False)

    def test_utility_bundle_loads_and_invalid_cutoff_rejected_before_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            artifact = directory/'router.joblib'
            joblib.dump(self.decider, artifact)
            manifest = dict(schema_version=1, artifact=artifact.name, artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            strong='strong', weak='weak', policy_mode='pairwise_utility', tradeoff=.1)
            path = directory/'bundle.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(load_bundle(directory).route('case').selected_model, 'weak')
            for cutoff in [None, True, float('nan'), float('inf')]:
                manifest['tradeoff'] = cutoff
                path.write_text(json.dumps(manifest))
                with patch('relay_router.client.joblib.load') as loader:
                    with self.assertRaises(ValueError):
                        load_bundle(directory)
                    loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
