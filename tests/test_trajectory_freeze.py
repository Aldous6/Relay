"""Regression for mutable inference parameters omitted by fitted-state hashes.

Runtime inference options belong to the frozen predictor manifest. These tests
use fixtures; archived live collections retain their original source snapshots.
"""
from copy import deepcopy
import unittest

from relay_router.trajectory import TrajectoryValuePolicy
from routerbench.agent_trajectory_learning import TrajectorySuccessPredictor
from test_agent_trajectory_learning import MODELS, rows


class TrajectoryFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitted = TrajectorySuccessPredictor(MODELS, max_text_features=120).fit(rows("fit"), rows("cal"))

    def parts(self, predictor):
        union = predictor._features.named_steps["union"]
        branches = dict(union.transformer_list)
        return union, branches["text"].named_steps["tfidf"], branches["state"].named_steps["vectorizer"]

    def test_manifest_task_audit_and_prediction_are_read_only_after_policy_construction(self):
        predictor = deepcopy(self.fitted)
        policy = TrajectoryValuePolicy(predictor, .8, "offline fixture threshold")
        before = predictor.manifest()
        request = rows("unseen")[0].request
        predictor.audit_task_groups([{"source_group": "unseen-group", "prompt": "unseen task"}])
        predictor.predict_outcomes(request)
        policy.select(request)
        self.assertEqual(before, predictor.manifest())

    def test_effective_transformer_norm_change_cannot_silently_change_forecasts(self):
        predictor = deepcopy(self.fitted)
        policy = TrajectoryValuePolicy(predictor, .8, "offline fixture threshold")
        request = rows("unseen")[0].request
        before_manifest = predictor.manifest()
        before_forecast = predictor.predict_outcomes(request)
        _, tfidf, _ = self.parts(predictor)
        # The fitted transformer, not just the vectorizer constructor option,
        # controls normalization during inference.
        tfidf._tfidf.norm = None
        self.assertNotEqual(before_forecast, predictor.predict_outcomes(request))
        self.assertNotEqual(before_manifest, predictor.manifest())
        with self.assertRaisesRegex(ValueError, "changed after policy construction"):
            policy.select(request)

    def test_feature_parameters_and_runtime_weighting_are_part_of_the_freeze(self):
        def change_lowercase(union, tfidf, state):
            tfidf.lowercase = False
        def change_token_pattern(union, tfidf, state):
            tfidf.token_pattern = r"(?u)\b[a-zA-Z]+\b"
        def change_ngrams(union, tfidf, state):
            tfidf.ngram_range = (1, 1)
        def change_effective_tf(union, tfidf, state):
            tfidf._tfidf.sublinear_tf = False
        def change_weights(union, tfidf, state):
            union.transformer_weights = {"text": 2.0}
        def change_state_separator(union, tfidf, state):
            state.separator = ":"
        def change_configured_vocabulary(union, tfidf, state):
            tfidf.vocabulary = {"new": 0}
        for mutate in (change_lowercase, change_token_pattern, change_ngrams, change_effective_tf,
                       change_weights, change_state_separator, change_configured_vocabulary):
            predictor = deepcopy(self.fitted)
            policy = TrajectoryValuePolicy(predictor, .8, "offline fixture threshold")
            before = predictor.manifest()
            mutate(*self.parts(predictor))
            with self.subTest(change=mutate.__name__):
                self.assertNotEqual(before, predictor.manifest())
                with self.assertRaisesRegex(ValueError, "changed after policy construction"):
                    policy.select(rows("unseen")[0].request)

    def test_fitted_vocabulary_and_coefficients_remain_guarded(self):
        for name in ("vocabulary", "coefficient"):
            predictor = deepcopy(self.fitted)
            policy = TrajectoryValuePolicy(predictor, .8, "offline fixture threshold")
            if name == "vocabulary":
                _, tfidf, _ = self.parts(predictor)
                term = next(iter(tfidf.vocabulary_))
                tfidf.vocabulary_[term] += 1
            else:
                predictor._heads[MODELS[0]][2].coef_[0] += 1
            with self.subTest(change=name), self.assertRaisesRegex(ValueError, "changed after policy construction"):
                policy.select(rows("unseen")[0].request)

    def test_unserializable_parameter_is_rejected_without_invoking_it(self):
        predictor = deepcopy(self.fitted)
        policy = TrajectoryValuePolicy(predictor, .8, "offline fixture threshold")
        invoked = []
        def callback(text):
            invoked.append(text)
            return text
        _, tfidf, _ = self.parts(predictor)
        tfidf.preprocessor = callback
        with self.assertRaisesRegex(ValueError, "Unsupported frozen inference parameter"):
            policy.select(rows("unseen")[0].request)
        self.assertFalse(invoked)


if __name__ == "__main__":
    unittest.main()
