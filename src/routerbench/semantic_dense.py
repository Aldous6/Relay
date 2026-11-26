"""Explicit bounded densification of the existing full-prompt semantic features."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .pairwise import _inputs
from .semantic_representation import SemanticPromptRepresentation


class DenseSemanticRepresentation(TransformerMixin, BaseEstimator):
    """Wrap the existing semantic transformer for dense-only estimators.

    Fitting only fits the nested transformer's optional structural scaler. It
    never loads or calls the encoder. Transformation preserves the existing
    text/window/cache contract and converts its finite 384/398-column CSR output
    into float64 only after checking a hard 512-column densification boundary.
    This wrapper does not add truncation or change semantic normalization.

    Runtime serialization is delegated to the nested transformer's __getstate__,
    which excludes its encoder handle. sklearn.clone reconstructs this wrapper
    from constructor parameters and carries no fitted nested transformer.
    """

    def __init__(self, mode="semantic_hybrid", encoder_spec=None,
                 encoder_factory=None, use_cache=True):
        self.mode = mode
        self.encoder_spec = encoder_spec
        self.encoder_factory = encoder_factory
        self.use_cache = use_cache

    def fit(self, X, y=None):
        _inputs(X)
        self.semantic_ = SemanticPromptRepresentation(
            mode=self.mode, encoder_spec=self.encoder_spec,
            encoder_factory=self.encoder_factory, use_cache=self.use_cache).fit(X, y)
        self.n_features_out_ = self.semantic_.n_features_out_
        if self.n_features_out_ not in (384, 398) or self.n_features_out_ > 512:
            raise ValueError("La representación semántica densa admite únicamente 384 o 398 columnas y nunca más de 512.")
        self.mode_ = self.semantic_.mode_
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        self.feature_names_out_ = self.semantic_.get_feature_names_out()
        return self

    def transform(self, X):
        check_is_fitted(self, ("semantic_", "mode_", "n_features_out_"))
        _inputs(X)
        # Allow callers to disable the encoder cache for honest latency probes
        # after fitting, exactly as with the existing semantic representation.
        self.semantic_.use_cache = self.use_cache
        matrix = self.semantic_.transform(X)
        if not sparse.isspmatrix_csr(matrix):
            raise ValueError("La representación semántica original debe devolver una matriz CSR.")
        if matrix.ndim != 2 or matrix.shape[0] != len(X) or matrix.shape[1] > 512 or matrix.shape[1] != self.n_features_out_:
            raise ValueError("Las dimensiones semánticas no coinciden con las filas o el ancho denso permitido.")
        if not np.isfinite(matrix.data).all():
            raise ValueError("La representación semántica contiene valores no finitos.")
        result = matrix.astype(np.float64, copy=False).toarray()
        if not np.isfinite(result).all():
            raise ValueError("La matriz semántica densa contiene valores no finitos.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ("semantic_", "feature_names_out_"))
        return self.feature_names_out_.copy()
