"""Frozen BERT feature contracts with deterministic mocks, never native models."""
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from routerbench.bert_features import FrozenBERTFeatures
from routerbench.features import extract_features


class MockBERTEncoder:
    def __init__(self, config):
        self.calls = []

    def encode_features(self, texts):
        self.calls.append(list(texts))
        values = np.zeros((len(texts), 768), dtype=np.float32)
        for position, text in enumerate(texts):
            values[position, :3] = [len(text), ord(text[-1]), sum(map(ord, text)) % 997]
        return values

    def __getstate__(self):
        raise AssertionError("The native encoder handle must not enter a pickle.")


def mock_bert_factory(config):
    return MockBERTEncoder(config)


def frame(texts, precomputed=False):
    result = pd.DataFrame({"prompt_text": texts})
    return pd.concat([result, extract_features(texts)], axis=1) if precomputed else result


def transformer():
    return FrozenBERTFeatures(reference_config={}, encoder_factory=mock_bert_factory)


class FrozenBERTFeatureTests(unittest.TestCase):
    def test_native_flow_batches_and_reuses_each_backbone_forward_for_preference(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            def __getitem__(self, item):
                return FakeTensor(self.values[item])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        hidden_batches = []

        def base_model(texts):
            hidden = np.zeros((len(texts), 2, 768), dtype=np.float32)
            hidden[:, 0, 0] = [int(text.rsplit("-", 1)[1]) for text in texts]
            hidden[:, 1, 0] = -100
            tensor = FakeTensor(hidden)
            hidden_batches.append(tensor)
            return SimpleNamespace(last_hidden_state=tensor)

        native_model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=768, model_type="xlm-roberta"),
            base_model=Mock(side_effect=base_model),
            classifier=Mock(side_effect=lambda hidden: FakeTensor(np.tile([2., 0., -1.], (len(hidden.values), 1)))))
        tokenizer = Mock(side_effect=lambda texts, **kwargs: {"texts": texts})
        reference = SimpleNamespace(model=native_model, tokenizer=tokenizer,
                                    torch=SimpleNamespace(inference_mode=nullcontext))
        config = {"model_dir": "artifacts/mock", "threads": 4,
                  "reference_tokenization": {"max_length": 512, "truncation": True,
                                             "padding": True, "prefix": None}}
        texts = [f"native-{index}" for index in range(10)]
        with patch("routerbench.routellm_reference.LocalBERTReference", return_value=reference) as loader, \
                patch("routerbench.bert_features.perf_counter", side_effect=[100., 103.5]) as clock:
            features = FrozenBERTFeatures(config).fit(frame(texts))
            loader.assert_not_called()
            actual = features.transform(frame(texts))
        self.assertEqual(actual.shape, (10, 768))
        np.testing.assert_array_equal(actual[:, 0], np.arange(10))
        self.assertEqual(native_model.base_model.call_count, 2)
        self.assertEqual(native_model.classifier.call_count, 2)
        self.assertEqual([len(call.args[0]) for call in tokenizer.call_args_list], [8, 2])
        for call in tokenizer.call_args_list:
            self.assertEqual(call.kwargs, {"padding": True, "truncation": True,
                                           "max_length": 512, "return_tensors": "pt"})
        for call, hidden in zip(native_model.classifier.call_args_list, hidden_batches):
            self.assertIs(call.args[0], hidden)
        preference = features.encoder_.last_preference_probabilities_
        self.assertEqual(preference.shape, (10,))
        self.assertTrue(np.isfinite(preference).all())
        self.assertTrue(((preference >= 0) & (preference <= 1)).all())
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(features.encoder_.last_native_seconds_, 3.5)

    def test_fit_is_lazy_and_exposes_the_fixed_feature_schema(self):
        factory = Mock(side_effect=AssertionError("fit must not load an encoder"))
        inputs = frame(["first", "second"], precomputed=True)
        result = FrozenBERTFeatures(reference_config={}, encoder_factory=factory).fit(inputs)
        factory.assert_not_called()
        self.assertFalse(hasattr(result, "encoder_"))
        self.assertEqual(result.n_features_out_, 768)
        self.assertEqual(len(result.get_feature_names_out()), 768)
        self.assertEqual(result.n_features_in_, len(inputs.columns))

    def test_complete_text_and_source_order_reach_the_encoder(self):
        long_prompt = "Word with spacing\n"*1500+"EVIDENCE_AT_THE_END"
        inputs = frame([long_prompt, "second"]).set_axis([91, 7])
        original = inputs.copy(deep=True)
        features = transformer().fit(inputs)
        actual = features.transform(inputs)
        self.assertEqual(features.encoder_.calls[-1], [long_prompt, "second"])
        self.assertEqual(actual.shape, (2, 768))
        self.assertEqual(actual.dtype, np.float64)
        self.assertTrue(np.isfinite(actual).all())
        reverse = features.transform(inputs.iloc[::-1])
        self.assertEqual(features.encoder_.calls[-1], ["second", long_prompt])
        np.testing.assert_array_equal(reverse, actual[::-1])
        pd.testing.assert_frame_equal(inputs, original)

    def test_clone_is_unfitted_and_feature_names_are_not_mutable_state(self):
        inputs = frame(["first", "second"])
        features = transformer().fit(inputs)
        features.transform(inputs)
        copied = clone(features)
        self.assertEqual(copied.get_params(), features.get_params())
        self.assertFalse(hasattr(copied, "encoder_"))
        with self.assertRaises(NotFittedError):
            copied.transform(inputs)
        names = features.get_feature_names_out()
        names[0] = "mutated"
        self.assertNotEqual(features.get_feature_names_out()[0], "mutated")

    def test_joblib_excludes_runtime_without_mutating_the_running_transformer(self):
        inputs = frame(["first", "second"], precomputed=True)
        features = transformer()
        expected = features.fit_transform(inputs)
        runtime = features.encoder_
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bert_features.joblib"
            joblib.dump(features, path)
            self.assertIs(features.encoder_, runtime)
            restored = joblib.load(path)
        self.assertFalse(hasattr(restored, "encoder_"))
        np.testing.assert_array_equal(restored.transform(inputs), expected)
        self.assertTrue(hasattr(restored, "encoder_"))

    def test_invalid_output_width_rows_and_nonfinite_values_are_rejected(self):
        inputs = frame(["first", "second"])
        features = transformer().fit(inputs)
        features.transform(inputs)
        for bad in (np.zeros((2, 767)), np.zeros((2, 769)), np.zeros((1, 768)),
                    np.zeros(1536), np.full((2, 768), np.nan), np.full((2, 768), np.inf),
                    np.full((2, 768), "not numeric")):
            with self.subTest(shape=bad.shape), patch.object(features.encoder_, "encode_features", return_value=bad):
                with self.assertRaises((ValueError, TypeError)):
                    features.transform(inputs)

    def test_predictor_boundary_empty_and_ambiguous_rows_are_rejected(self):
        valid = frame(["first", "second"])
        invalid = [valid.assign(score=1), valid.assign(eval_name="benchmark"), valid.assign(n_chars=3),
                   valid.iloc[:0], pd.concat([valid.iloc[[0]]]*2), frame([" "]), frame([None]), "a prompt"]
        for bad in invalid:
            with self.subTest(type=type(bad).__name__), self.assertRaises((ValueError, TypeError)):
                transformer().fit(bad)
        features = transformer().fit(valid)
        for bad in invalid:
            with self.subTest(type=type(bad).__name__), self.assertRaises((ValueError, TypeError)):
                features.transform(bad)
        np.testing.assert_array_equal(features.transform(valid), features.transform(frame(["first", "second"], precomputed=True)))

    def test_refit_drops_an_old_runtime_and_remains_lazy(self):
        features = transformer()
        features.fit_transform(frame(["first", "second"]))
        features.fit(frame(["new fitting inputs", "different"]))
        self.assertFalse(hasattr(features, "encoder_"))
        self.assertEqual(features.transform(frame(["another"])).shape, (1, 768))


if __name__ == "__main__":
    unittest.main()
