"""Frozen semantic features with a train-only optional structural block.

The encoder is loaded lazily and excluded from serialized estimator state.
Only prompt text and the existing fourteen structural columns cross the input
boundary. Cached embeddings remain an encoder concern, outside this DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_NAMES
from .representations import PromptRepresentation

EMBEDDING_DIMENSION = 384


class SemanticPromptRepresentation(TransformerMixin, BaseEstimator):
    """Encode complete prompts, optionally combined with fourteen features.

    ``encoder_factory`` is an optional callable accepting ``encoder_spec`` and
    returning an object with ``encode(texts, use_cache=...)``. Production defaults
    to the local WindowedE5Encoder. Fit never calls or loads the encoder.
    """

    def __init__(self, mode: str = "semantic", encoder_spec: dict | None = None,
                 encoder_factory=None, use_cache: bool = True):
        self.mode = mode
        self.encoder_spec = encoder_spec
        self.encoder_factory = encoder_factory
        self.use_cache = use_cache

    def fit(self, X: pd.DataFrame, y=None):
        if self.mode not in {"semantic", "semantic_hybrid"}:
            raise ValueError("mode debe ser semantic o semantic_hybrid.")
        if not isinstance(self.encoder_spec, dict):
            raise ValueError("encoder_spec debe ser un diccionario de configuración.")
        if self.encoder_factory is not None and not callable(self.encoder_factory):
            raise TypeError("encoder_factory debe ser callable o None.")
        texts, precomputed = PromptRepresentation._validate_frame(X)
        self.mode_ = self.mode
        # A repeated fit may have a different encoder configuration. Do not
        # retain a runtime session instantiated during the previous fit.
        self.__dict__.pop("encoder_", None)
        names = [f"semantic__embedding_{index:03d}" for index in range(EMBEDDING_DIMENSION)]
        if self.mode_ == "semantic_hybrid":
            values = PromptRepresentation._structural_values(X, texts, precomputed)
            self.structural_scaler_ = MaxAbsScaler().fit(values)
            names.extend(f"structural__{name}" for name in FEATURE_NAMES)
        else:
            self.__dict__.pop("structural_scaler_", None)
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self.n_features_out_ = len(names)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def _get_encoder(self):
        if not hasattr(self, "encoder_"):
            if self.encoder_factory is None:
                from .semantic_encoder import WindowedE5Encoder

                self.encoder_ = WindowedE5Encoder(self.encoder_spec)
            else:
                self.encoder_ = self.encoder_factory(self.encoder_spec)
        return self.encoder_

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ("mode_", "n_features_out_"))
        texts, precomputed = PromptRepresentation._validate_frame(X)
        if not isinstance(self.use_cache, (bool, np.bool_)):
            raise ValueError("use_cache debe ser booleano.")
        encoded = self._get_encoder().encode(texts, use_cache=bool(self.use_cache))
        try:
            vectors = np.asarray(encoded, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("El encoder debe devolver una matriz numérica.") from exc
        if vectors.shape != (len(texts), EMBEDDING_DIMENSION):
            raise ValueError(f"El encoder debe devolver forma ({len(texts)}, {EMBEDDING_DIMENSION}).")
        if not np.isfinite(vectors).all():
            raise ValueError("El encoder devolvió embeddings no finitos.")
        blocks = [normalize(sparse.csr_matrix(vectors, dtype=np.float32), norm="l2")]
        if self.mode_ == "semantic_hybrid":
            values = PromptRepresentation._structural_values(X, texts, precomputed)
            scaled = self.structural_scaler_.transform(values)
            blocks.append(normalize(sparse.csr_matrix(scaled, dtype=np.float32), norm="l2"))
        combined = sparse.hstack(blocks, format="csr", dtype=np.float32)
        return normalize(combined, norm="l2", copy=False).tocsr()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def __getstate__(self):
        # object.__getstate__ can expose the live __dict__ on Python 3.12.
        # Excluding a runtime from a pickle must not mutate the running model.
        state = dict(super().__getstate__())
        state.pop("encoder_", None)
        return state
