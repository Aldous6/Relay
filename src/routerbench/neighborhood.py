"""V7: fit-only lexical neighbors combined with global pairwise evidence.

This is a local hypothesis inspired by similarity-weighted routing, not the
official RouteLLM SW algorithm. It uses lexical vectors, top-k retrieval and
thresholded outcomes instead of semantic Arena comparisons and per-query BT.
The combination head is fitted on calibration only; its apparent calibration
on that same population is not an independent quality estimate.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.utils.validation import check_is_fitted

from .features import normalize_for_dedup
from .learner import _cost_features
from .pairwise import OUTCOME_CLASSES, _inputs, _names, _outcomes, pairwise_targets
from .representations import PromptRepresentation


def _positive(value, name, integer=False):
    kind = Integral if integer else Real
    if (not isinstance(value, kind) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} debe ser positivo y finito" + (" y entero." if integer else "."))


def _class_prior(codes):
    return (np.bincount(codes, minlength=3).astype(float)+1)/(len(codes)+3)


def _ordered_probabilities(classifier, matrix, prior):
    """Retain smoothed mass for classes absent from the classifier's fit."""
    result = np.tile(prior, (matrix.shape[0], 1))
    if classifier is not None:
        supported = np.asarray(classifier.classes_, dtype=int)
        missing = np.setdiff1d(np.arange(3), supported)
        estimated = np.asarray(classifier.predict_proba(matrix), dtype=np.float64)
        if (not np.isfinite(estimated).all() or (estimated < 0).any() or (estimated > 1).any()
                or (estimated.sum(axis=1) <= 0).any()):
            raise ValueError("El predictor devolvió probabilidades inválidas.")
        # SGD can emit float32 row sums differing from one by a few ulps.
        estimated /= estimated.sum(axis=1, keepdims=True)
        result[:, supported] = estimated * (1-prior[missing].sum())
    if (not np.isfinite(result).all() or (result < 0).any() or (result > 1).any()
            or not np.allclose(result.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("El predictor devolvió probabilidades inválidas.")
    return result


def _neighbor_features(query_vectors, fit_vectors, fit_codes, fit_prior,
                       n_neighbors, temperature, prior_weight, query_batch_size):
    """Return local class evidence and support from fitting neighbors only.

    At most query_batch_size by n_fit similarities are densified. Boundary
    ties choose the earliest fitting row, without inspecting its outcome.
    Vectors are unit-normalized by PromptRepresentation; no outcome or ID is
    part of their coordinates. Raw text is neither required nor retained here.
    """
    n_fit = fit_vectors.shape[0]
    k = min(int(n_neighbors), n_fit)
    result = np.empty((query_vectors.shape[0], 6), dtype=np.float64)
    transposed = fit_vectors.T
    for start in range(0, query_vectors.shape[0], query_batch_size):
        stop = min(start+query_batch_size, query_vectors.shape[0])
        similarities = (query_vectors[start:stop] @ transposed).toarray()
        if not np.isfinite(similarities).all():
            raise ValueError("La similitud de vecinos contiene valores no finitos.")
        # Roundoff may exceed the cosine range by a few float32 ulps.
        np.clip(similarities, -1, 1, out=similarities)
        for offset, row in enumerate(similarities):
            if k == n_fit:
                positions = np.arange(n_fit)
            else:
                boundary = np.partition(row, n_fit-k)[n_fit-k]
                above = np.flatnonzero(row > boundary)
                ties = np.flatnonzero(row == boundary)[:k-len(above)]
                positions = np.concatenate((above, ties))
            positions = positions[np.lexsort((positions, -row[positions]))]
            selected = row[positions].astype(np.float64)
            weights = np.exp((selected-selected[0])/temperature)
            total = weights.sum()
            counts = np.bincount(fit_codes[positions], weights=weights, minlength=3)
            result[start+offset, :3] = (counts+prior_weight*fit_prior)/(total+prior_weight)
            result[start+offset, 3:] = (selected[0], selected.mean(), total**2/(np.square(weights).sum()*k))
    return result


class NeighborhoodRegretRouter(BaseEstimator):
    """Combine a global classifier and fit-only retrieved outcome evidence.

    All fitting examples, including shared failures, enter the global head and
    neighbor bank. A separate multinomial logistic head combines/calibrates
    predictions using only calibration rows. Missing calibration classes keep
    Laplace-smoothed constant mass. With one class the full smoothed prior is
    used. Neither fallback guarantees reliable probabilities out of sample.

    Stored sparse fit vectors and class codes can contain information about
    the training population; absence of raw prompt strings is not a privacy
    guarantee or permission to redistribute training-derived artifacts.
    """

    def __init__(self, strong="gpt-4-1106-preview", weak="mistralai/mixtral-8x7b-chat",
                 seed=42, quality_threshold=.75, max_features=20000, min_df=2,
                 alpha=1e-4, n_neighbors=64, temperature=.05, prior_weight=4.,
                 calibration_C=1., use_neighbors=True, query_batch_size=64):
        self.strong = strong
        self.weak = weak
        self.seed = seed
        self.quality_threshold = quality_threshold
        self.max_features = max_features
        self.min_df = min_df
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.prior_weight = prior_weight
        self.calibration_C = calibration_C
        self.use_neighbors = use_neighbors
        self.query_batch_size = query_batch_size

    def _validate_parameters(self):
        _names(self.strong, self.weak)
        for name in ("alpha", "temperature", "prior_weight", "calibration_C"):
            _positive(getattr(self, name), name)
        for name in ("n_neighbors", "query_batch_size"):
            _positive(getattr(self, name), name, integer=True)
        if self.query_batch_size > 256:
            raise ValueError("query_batch_size no puede superar 256 para limitar memoria.")
        if not isinstance(self.use_neighbors, bool):
            raise ValueError("use_neighbors debe ser booleano.")

    def fit(self, fit_inputs, fit_scores, fit_costs, calibration_inputs, calibration_scores):
        self._validate_parameters()
        _inputs(fit_inputs)
        _inputs(calibration_inputs)
        fit_scores, fit_costs = _outcomes(fit_scores, self.strong, self.weak, fit_costs)
        calibration_scores, _ = _outcomes(calibration_scores, self.strong, self.weak)
        if not fit_inputs.index.equals(fit_scores.index) or not calibration_inputs.index.equals(calibration_scores.index):
            raise ValueError("Inputs y resultados deben estar alineados exactamente.")
        if len(fit_inputs.index.intersection(calibration_inputs.index)):
            raise ValueError("Fit y calibración requieren identificadores disjuntos.")
        groups = []
        for frame in (fit_inputs, calibration_inputs):
            normalized = frame.prompt_text.map(normalize_for_dedup)
            if normalized.duplicated().any():
                raise ValueError("Hay prompts normalizados duplicados en una partición.")
            groups.append(set(normalized))
        if groups[0] & groups[1]:
            raise ValueError("Fit y calibración comparten grupos de prompt.")
        fit_target = pairwise_targets(fit_scores, self.strong, self.weak, self.quality_threshold)
        cal_target = pairwise_targets(calibration_scores, self.strong, self.weak, self.quality_threshold)
        mapping = {name: position for position, name in enumerate(OUTCOME_CLASSES)}
        fit_codes = fit_target.map(mapping).to_numpy(dtype=np.int8)
        cal_codes = cal_target.map(mapping).to_numpy(dtype=np.int8)
        transformer = PromptRepresentation(mode="hybrid", max_features=self.max_features, min_df=self.min_df)
        fit_vectors = transformer.fit_transform(fit_inputs).astype(np.float32).tocsr()
        cal_vectors = transformer.transform(calibration_inputs).astype(np.float32).tocsr()
        fit_prior, cal_prior = _class_prior(fit_codes), _class_prior(cal_codes)
        global_head = None
        if np.unique(fit_codes).size >= 2:
            global_head = SGDClassifier(loss="log_loss", alpha=self.alpha, average=True,
                                        max_iter=1500, tol=1e-4, random_state=self.seed)
            global_head.fit(fit_vectors, fit_codes)
        global_features = _ordered_probabilities(global_head, cal_vectors, fit_prior)
        if self.use_neighbors:
            local_features = _neighbor_features(cal_vectors, fit_vectors, fit_codes, fit_prior,
                self.n_neighbors, self.temperature, self.prior_weight, self.query_batch_size)
            calibration_features = np.column_stack((global_features, local_features))
        else:
            calibration_features = global_features
        calibrator = None
        if np.unique(cal_codes).size >= 2:
            calibrator = LogisticRegression(C=self.calibration_C, solver="lbfgs", max_iter=1000,
                                             random_state=self.seed)
            calibrator.fit(calibration_features, cal_codes)
        constants = {name: float(cal_prior[position]) for position, name in enumerate(OUTCOME_CLASSES)
                     if calibrator is None or position not in calibrator.classes_}
        cost_x = _cost_features(fit_inputs)
        log_cost = np.log(np.maximum(fit_costs.to_numpy(dtype=float), 1e-12))
        cost_regressor = Ridge(alpha=1.).fit(cost_x, log_cost)
        residual = log_cost-np.asarray(cost_regressor.predict(cost_x)).reshape(len(fit_inputs), 2)
        # Publish fitted state together; failures above leave a prior fit intact.
        self.transformer_, self.global_head_, self.calibrator_ = transformer, global_head, calibrator
        self.fit_prior_, self.class_prior_, self.constants_ = fit_prior, cal_prior, constants
        self.fit_vectors_ = fit_vectors if self.use_neighbors else None
        self.fit_codes_ = fit_codes if self.use_neighbors else None
        self.cost_regressor_ = cost_regressor
        self.cost_log_smearing_ = logsumexp(residual, axis=0)-np.log(len(fit_inputs))
        self.models_, self.classes_ = (self.strong, self.weak), np.asarray(OUTCOME_CLASSES)
        self.fit_rows_, self.calibration_rows_ = len(fit_inputs), len(calibration_inputs)
        self.n_neighbors_ = min(self.n_neighbors, self.fit_rows_) if self.use_neighbors else 0
        self.calibration_feature_count_ = calibration_features.shape[1]
        self.target_counts_ = {part: target.value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict()
                               for part, target in (("fit", fit_target), ("calibration", cal_target))}
        self.calibration_notes_ = {
            name: ("smoothed constant calibration mass; fewer than two classes or class absent"
                   if name in constants else "multinomial combination on disjoint calibration rows; no independent reliability guarantee")
            for name in OUTCOME_CLASSES}
        self.feature_names_out_ = np.asarray([f"global__{name}" for name in OUTCOME_CLASSES] +
            ([f"local__{name}" for name in OUTCOME_CLASSES] + ["max_similarity", "mean_similarity", "effective_neighbor_fraction"]
             if self.use_neighbors else []))
        self.fitted_parameters_ = dict(self.get_params(deep=False))
        return self

    def predict_tables(self, inputs):
        check_is_fitted(self, ("transformer_", "models_", "cost_regressor_", "calibrator_"))
        _inputs(inputs)
        self._validate_parameters()
        if self.get_params(deep=False) != self.fitted_parameters_:
            raise ValueError("Los parámetros cambiaron después del ajuste; vuelve a ajustar.")
        transformed = self.transformer_.transform(inputs).astype(np.float32).tocsr()
        features = _ordered_probabilities(self.global_head_, transformed, self.fit_prior_)
        if self.use_neighbors:
            if self.fit_vectors_ is None or self.fit_codes_ is None:
                raise ValueError("El modo de vecinos cambió después del ajuste; vuelve a ajustar.")
            features = np.column_stack((features, _neighbor_features(transformed, self.fit_vectors_,
                self.fit_codes_, self.fit_prior_, self.n_neighbors, self.temperature,
                self.prior_weight, self.query_batch_size)))
        if features.shape[1] != self.calibration_feature_count_:
            raise ValueError("La representación cambió después del ajuste; vuelve a ajustar.")
        probabilities = _ordered_probabilities(self.calibrator_, features, self.class_prior_)
        table = pd.DataFrame(probabilities, index=inputs.index, columns=OUTCOME_CLASSES)
        table["expected_quality_gain"] = table.strong_only-table.weak_only
        log_cost = np.asarray(self.cost_regressor_.predict(_cost_features(inputs))).reshape(len(inputs), 2)
        costs = np.exp(np.clip(log_cost+self.cost_log_smearing_, -50, np.log(np.finfo(float).max)-1))
        return table, pd.DataFrame(costs, index=inputs.index, columns=self.models_)
