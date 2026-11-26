"""V6: learn the risk of downgrading, with a conservative strong fallback.

Two separately calibrated binary estimates describe strong-only success and
weak success. They are marginal estimates, not a jointly calibrated simplex or
a statistical guarantee. The representation is the existing lexical hybrid.
No provider, model discovery, future outcomes or network is used at inference.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.utils.validation import check_is_fitted

from .features import normalize_for_dedup
from .learner import _cost_features
from .pairwise import _inputs, _names, _outcomes
from .representations import PromptRepresentation

RISK_COLUMNS = ("harm_probability", "weak_success_probability")


def _probability(value, name):
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} debe estar en [0,1] y ser finito.")


def guarded_targets(scores, strong, weak, quality_threshold=.75):
    _names(strong, weak)
    _probability(quality_threshold, "quality_threshold")
    scores, _ = _outcomes(scores, strong, weak)
    strong_ok, weak_ok = scores[strong].ge(quality_threshold), scores[weak].ge(quality_threshold)
    return pd.DataFrame({"harm_probability": (strong_ok & ~weak_ok).astype(int),
                         "weak_success_probability": weak_ok.astype(int)}, index=scores.index)


class GuardedPairRouter(BaseEstimator):
    """Prompt-only harm and weak-success estimates for one explicit model pair."""

    def __init__(self, strong="gpt-4-1106-preview", weak="mistralai/mixtral-8x7b-chat",
                 seed=42, quality_threshold=.75, max_features=20000, min_df=2, alpha=1e-4):
        self.strong = strong
        self.weak = weak
        self.seed = seed
        self.quality_threshold = quality_threshold
        self.max_features = max_features
        self.min_df = min_df
        self.alpha = alpha

    def fit(self, fit_inputs, fit_scores, fit_costs, calibration_inputs, calibration_scores):
        _names(self.strong, self.weak)
        _probability(self.quality_threshold, "quality_threshold")
        if (not isinstance(self.alpha, Real) or isinstance(self.alpha, (bool, np.bool_))
                or not np.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("alpha debe ser positivo y finito.")
        _inputs(fit_inputs)
        _inputs(calibration_inputs)
        scores, costs = _outcomes(fit_scores, self.strong, self.weak, fit_costs)
        cal_scores, _ = _outcomes(calibration_scores, self.strong, self.weak)
        if not fit_inputs.index.equals(scores.index) or not calibration_inputs.index.equals(cal_scores.index):
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
        target = guarded_targets(scores, self.strong, self.weak, self.quality_threshold)
        cal_target = guarded_targets(cal_scores, self.strong, self.weak, self.quality_threshold)
        transformer = PromptRepresentation(mode="hybrid", max_features=self.max_features, min_df=self.min_df)
        fit_x = transformer.fit_transform(fit_inputs)
        cal_x = transformer.transform(calibration_inputs)
        predictors, constants, notes = {}, {}, {}
        for position, column in enumerate(RISK_COLUMNS):
            y, cal_y = target[column].to_numpy(), cal_target[column].to_numpy()
            if np.unique(y).size < 2 or np.unique(cal_y).size < 2:
                constants[column] = float((cal_y.sum()+1)/(len(cal_y)+2))
                notes[column] = "smoothed calibration prevalence; a partition lacks both classes"
            else:
                classifier = SGDClassifier(loss="log_loss", alpha=self.alpha, max_iter=1500,
                                           tol=1e-4, average=True, random_state=self.seed+position)
                classifier.fit(fit_x, y)
                predictor = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
                predictor.fit(cal_x, cal_y)
                predictors[column] = predictor
                notes[column] = "sigmoid on disjoint calibration; reliability not guaranteed"
        cost_x = _cost_features(fit_inputs)
        log_cost = np.log(np.maximum(costs.to_numpy(dtype=float), 1e-12))
        cost_regressor = Ridge(alpha=1.0).fit(cost_x, log_cost)
        residual = log_cost-np.asarray(cost_regressor.predict(cost_x)).reshape(len(fit_inputs), 2)
        self.transformer_, self.predictors_, self.constants_ = transformer, predictors, constants
        self.cost_regressor_ = cost_regressor
        self.cost_log_smearing_ = logsumexp(residual, axis=0)-np.log(len(fit_inputs))
        self.calibration_notes_ = notes
        self.models_ = (self.strong, self.weak)
        self.fit_rows_, self.calibration_rows_ = len(fit_inputs), len(calibration_inputs)
        self.target_counts_ = {part: {column: {str(label): int((frame[column] == label).sum())
                                             for label in (0, 1)} for column in RISK_COLUMNS}
                               for part, frame in (("fit", target), ("calibration", cal_target))}
        return self

    def predict_tables(self, inputs):
        check_is_fitted(self, ("transformer_", "cost_regressor_", "models_"))
        _inputs(inputs)
        values = self.transformer_.transform(inputs)
        predictions = {}
        for column in RISK_COLUMNS:
            if column in self.constants_:
                predictions[column] = np.full(len(inputs), self.constants_[column])
            else:
                predictor = self.predictors_[column]
                predictions[column] = predictor.predict_proba(values)[:, list(predictor.classes_).index(1)]
        log_cost = np.asarray(self.cost_regressor_.predict(_cost_features(inputs))).reshape(len(inputs), 2)
        cost = np.exp(np.clip(log_cost+self.cost_log_smearing_, -50, np.log(np.finfo(float).max)-1))
        return (pd.DataFrame(predictions, index=inputs.index),
                pd.DataFrame(cost, index=inputs.index, columns=self.models_))


def choose_guarded(predictions, costs, strong, weak, max_harm, min_weak_success):
    """Choose weak only when both probability gates and a cost saving hold.

Thresholds concern fitted probabilities; neither is the observed-score
threshold, an uncertainty bound, or a guaranteed maximum failure rate.
"""
    _names(strong, weak)
    _probability(max_harm, "max_harm")
    _probability(min_weak_success, "min_weak_success")
    for frame, columns in ((predictions, set(RISK_COLUMNS)), (costs, {strong, weak})):
        if (not isinstance(frame, pd.DataFrame) or frame.empty or not frame.index.is_unique
                or not frame.columns.is_unique or set(frame.columns) != columns):
            raise ValueError("Se requieren tablas no vacías con columnas e índices únicos del par.")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Las predicciones deben ser finitas y no negativas.")
    if not predictions.index.equals(costs.index) or (predictions.to_numpy(dtype=float) > 1).any():
        raise ValueError("Probabilidades fuera de rango o tablas desalineadas.")
    low_risk = predictions.harm_probability.le(max_harm)
    sufficient_success = predictions.weak_success_probability.ge(min_weak_success)
    cheaper = costs[weak].lt(costs[strong])
    use_weak = low_risk & sufficient_success & cheaper
    reason = np.where(~low_risk, "harm_guard", np.where(~sufficient_success, "weak_quality_guard",
                       np.where(~cheaper, "no_predicted_saving", "guarded_weak")))
    return pd.DataFrame({"selected_model": np.where(use_weak, weak, strong), "decision": reason},
                        index=predictions.index)
