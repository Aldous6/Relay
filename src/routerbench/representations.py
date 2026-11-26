"""Prompt-only structural and lexical representations for the new experiment.

This module does not change the fourteen-column v1 dataset. Vocabulary and
scaling are learned only when ``fit`` is called by the training procedure. Text
is never truncated, although a fitted vocabulary necessarily ignores unseen
terms. Lexical vectors are sparse; no embedding model or network is required.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_NAMES, extract_features


class PromptRepresentation(TransformerMixin, BaseEstimator):
    """Build structural, lexical or equally weighted hybrid prompt features.

    Input is a DataFrame with ``prompt_text`` and optionally all fourteen
    precomputed structural columns. Any other column is rejected to maintain the
    predictor boundary. Structural columns are used in FEATURE_NAMES order.
    Each block is normalized per row; the combined output is normalized again.
    ``max_features`` is a separate limit for the word and character vocabularies.
    """

    def __init__(self, mode: str = "hybrid", max_features: int = 12000, min_df: int | float = 2):
        self.mode = mode
        self.max_features = max_features
        self.min_df = min_df

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> tuple[list[str], bool]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("La entrada debe ser un DataFrame con prompt_text.")
        if not frame.columns.is_unique:
            raise ValueError("Las columnas repetidas son ambiguas.")
        if "prompt_text" not in frame.columns:
            raise ValueError("Falta la columna prompt_text con texto ya reconstruido.")
        extras = set(frame.columns) - {"prompt_text", *FEATURE_NAMES}
        if extras:
            raise ValueError(f"Columnas no permitidas como predictores: {sorted(map(str, extras))}")
        present = set(frame.columns) & set(FEATURE_NAMES)
        if present and present != set(FEATURE_NAMES):
            raise ValueError("Proporciona las catorce columnas estructurales completas o ninguna.")
        texts = frame["prompt_text"].tolist()
        if not texts:
            raise ValueError("Se necesita al menos un prompt.")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Cada prompt_text debe ser una cadena de texto.")
        return texts, bool(present)

    @staticmethod
    def _structural_values(frame: pd.DataFrame, texts: list[str], precomputed: bool) -> np.ndarray:
        if precomputed:
            try:
                values = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("Las variables estructurales deben ser numéricas.") from exc
        else:
            values = extract_features(texts).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Las variables estructurales deben ser finitas y no negativas.")
        return values

    def fit(self, X: pd.DataFrame, y=None):
        if self.mode not in {"structural", "lexical", "hybrid"}:
            raise ValueError("mode debe ser structural, lexical o hybrid.")
        if not isinstance(self.max_features, Integral) or isinstance(self.max_features, bool) or self.max_features < 1:
            raise ValueError("max_features debe ser un entero positivo por vocabulario.")
        texts, precomputed = self._validate_frame(X)
        self.mode_ = self.mode
        names = []
        if self.mode_ in {"structural", "hybrid"}:
            values = self._structural_values(X, texts, precomputed)
            self.structural_scaler_ = MaxAbsScaler().fit(values)
            names.extend(f"structural__{name}" for name in FEATURE_NAMES)
        if self.mode_ in {"lexical", "hybrid"}:
            shared = dict(max_features=self.max_features, min_df=self.min_df,
                          sublinear_tf=True, dtype=np.float32)
            self.lexical_ = FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), **shared)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **shared)),
            ])
            try:
                self.lexical_.fit(texts)
            except ValueError as exc:
                raise ValueError("No se pudo ajustar TF-IDF. Revisa el texto y min_df; el vocabulario puede estar vacío.") from exc
            names.extend(self.lexical_.get_feature_names_out().tolist())
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self.n_features_out_ = len(names)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ("mode_", "n_features_out_"))
        texts, precomputed = self._validate_frame(X)
        blocks = []
        if self.mode_ in {"structural", "hybrid"}:
            values = self._structural_values(X, texts, precomputed)
            scaled = self.structural_scaler_.transform(values)
            blocks.append(normalize(sparse.csr_matrix(scaled, dtype=np.float32), norm="l2"))
        if self.mode_ in {"lexical", "hybrid"}:
            blocks.append(normalize(self.lexical_.transform(texts).tocsr(), norm="l2"))
        result = sparse.hstack(blocks, format="csr", dtype=np.float32)
        return normalize(result, norm="l2", copy=False).tocsr()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
