"""Frozen checkpoint CLS features for an explicitly adapted BERT control.

This representation does not reproduce the published preference router after a
new local head is fitted. It only reuses its pinned, locally verified backbone.
No model, package backend, network client or dataset is loaded by fit/import.
"""
from __future__ import annotations

import copy
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .pairwise import _inputs
from .representations import PromptRepresentation

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIMENSION = 768
MAX_LENGTH = 512
BATCH_SIZE = 8


def _texts(frame):
    _inputs(frame)
    texts, precomputed = PromptRepresentation._validate_frame(frame)
    if precomputed:
        # These columns are not BERT features, but keep the shared input
        # boundary strict instead of accepting invalid numeric metadata.
        PromptRepresentation._structural_values(frame, texts, True)
    return texts


class _LocalFrozenBERTEncoder:
    """Native CPU CLS extraction, created only by transform on demand.

    Uses the reference's exact tokenizer truncation at 512, not sliding windows.
    Optional preference probabilities are telemetry from the same hidden state;
    they are never appended to the 768 returned predictor columns.
    """

    def __init__(self, reference_config):
        if reference_config.get("reference_tokenization") != {
                "max_length": MAX_LENGTH, "truncation": True,
                "padding": True, "prefix": None}:
            raise ValueError("El control BERT conserva la tokenización fijada a 512 tokens.")
        if reference_config.get("threads") != 4:
            raise ValueError("El control BERT fija cuatro hilos de CPU.")
        directory = (ROOT / reference_config["model_dir"]).resolve()
        if not directory.is_relative_to(ROOT):
            raise ValueError("El checkpoint BERT debe estar dentro del proyecto.")
        # This existing loader verifies local model/tokenizer file hashes and
        # fixed packages and explicitly forbids remote model loading.
        from .routellm_reference import LocalBERTReference

        self.reference = LocalBERTReference(directory, reference_config)
        config = self.reference.model.config
        if getattr(config, "hidden_size", None) != FEATURE_DIMENSION or getattr(config, "model_type", None) != "xlm-roberta":
            raise ValueError("Se requiere el backbone XLM-RoBERTa fijado de 768 columnas.")

    def encode_features(self, texts):
        from .routellm_reference import reference_probabilities

        reference = self.reference
        started = perf_counter()
        chunks, preferences = [], []
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = reference.tokenizer(texts[start:start+BATCH_SIZE], padding=True,
                truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with reference.torch.inference_mode():
                hidden = reference.model.base_model(**encoded).last_hidden_state
                cls = hidden[:, 0, :].detach().cpu().numpy()
                # XLM-RoBERTa's classifier consumes its full sequence tensor.
                # This saves another backbone pass for the frozen reference.
                logits = reference.model.classifier(hidden).detach().cpu().numpy()
            chunks.append(cls)
            preferences.append(reference_probabilities(logits))
        self.last_preference_probabilities_ = np.concatenate(preferences)
        features = np.concatenate(chunks, axis=0)
        self.last_call_ = {"prompts": len(texts), "batch_size": BATCH_SIZE,
                           "max_length": MAX_LENGTH, "truncation": True,
                           "feature_dimension": FEATURE_DIMENSION,
                           "scope": "raw CLS features; frozen preference head is telemetry only"}
        self.last_native_seconds_ = perf_counter() - started
        return features


class FrozenBERTFeatures(TransformerMixin, BaseEstimator):
    """Return raw 768-column CLS features from the pinned frozen backbone.

    fit validates input and records metadata without loading a model. transform
    passes complete strings in their original order to a lazy encoder. The
    native encoder applies the published 512-token truncation at tokenization;
    full strings reaching it do not imply full retained context.

    encoder_factory is a testing/integration injection: factory(config) returns
    an object exposing encode_features(texts). Preference telemetry is optional
    on injected encoders and never part of this feature contract.
    """

    def __init__(self, reference_config, encoder_factory=None):
        self.reference_config = reference_config
        self.encoder_factory = encoder_factory

    def fit(self, X, y=None):
        _texts(X)
        if not isinstance(self.reference_config, dict):
            raise ValueError("reference_config debe ser un diccionario.")
        if self.encoder_factory is not None and not callable(self.encoder_factory):
            raise TypeError("encoder_factory debe ser callable o None.")
        self.__dict__.pop("encoder_", None)
        self.reference_config_ = copy.deepcopy(self.reference_config)
        self.n_features_out_ = FEATURE_DIMENSION
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        self.feature_names_out_ = np.asarray(
            [f"bert_cls__{position:03d}" for position in range(FEATURE_DIMENSION)], dtype=object)
        return self

    def _get_encoder(self):
        if not hasattr(self, "encoder_"):
            factory = _LocalFrozenBERTEncoder if self.encoder_factory is None else self.encoder_factory
            self.encoder_ = factory(self.reference_config_)
            if not callable(getattr(self.encoder_, "encode_features", None)):
                self.__dict__.pop("encoder_", None)
                raise TypeError("El encoder BERT debe proporcionar encode_features.")
        return self.encoder_

    def transform(self, X):
        check_is_fitted(self, ("reference_config_", "feature_names_out_"))
        texts = _texts(X)
        if self.reference_config != self.reference_config_:
            raise ValueError("La configuración BERT cambió; vuelve a ajustar explícitamente.")
        encoded = self._get_encoder().encode_features(texts)
        if hasattr(encoded, "shape") and encoded.shape != (len(texts), FEATURE_DIMENSION):
            raise ValueError("El encoder BERT debe devolver exactamente n × 768 columnas.")
        if np.iscomplexobj(encoded):
            raise ValueError("El encoder BERT debe devolver atributos reales.")
        try:
            result = np.asarray(encoded, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("El encoder BERT debe devolver una matriz numérica densa.") from error
        if result.shape != (len(texts), FEATURE_DIMENSION) or not np.isfinite(result).all():
            raise ValueError("Los atributos BERT deben tener forma n × 768 y ser finitos.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def __getstate__(self):
        state = dict(super().__getstate__())
        state.pop("encoder_", None)
        return state
