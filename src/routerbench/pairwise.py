"""Prompt-only expected regret for a fixed pair of models.

The target is the difference in thresholded success, not relative answer style.
Both models failing is an equal-outcome example, rather than a strong-model win.
This module never reads a dataset, a benchmark name or held-out outcomes.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.utils.validation import check_is_fitted

from .learner import _check_outcomes, _cost_features
from .representations import PromptRepresentation

OUTCOME_CLASSES = ("strong_only", "weak_only", "equal")


def _names(strong, weak):
    if any(not isinstance(name, str) or not name.strip() or name != name.strip()
           for name in (strong, weak)) or strong == weak:
        raise ValueError("Se requieren dos identificadores de modelos distintos y no vacíos.")


def _inputs(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.index.is_unique:
        raise ValueError("Se requieren inputs no vacíos con índice único.")
    texts, _ = PromptRepresentation._validate_frame(frame)
    if any(not text.strip() for text in texts):
        raise ValueError("Los prompts deben contener texto no vacío.")


def _outcomes(scores, strong, weak, costs=None):
    if not isinstance(scores, pd.DataFrame) or not {strong, weak}.issubset(scores.columns):
        raise ValueError("Faltan los scores del par de modelos.")
    if costs is not None:
        if not isinstance(costs, pd.DataFrame) or not costs.columns.is_unique or set(costs.columns) != set(scores.columns):
            raise ValueError("Scores y costos deben contener los mismos modelos únicos.")
        costs = costs.loc[:, scores.columns]
    _check_outcomes(scores, costs)
    return scores.loc[:, [strong, weak]], None if costs is None else costs.loc[:, [strong, weak]]


def pairwise_targets(scores: pd.DataFrame, strong: str, weak: str,
                     quality_threshold: float = .75) -> pd.Series:
    """Encode strong-only, weak-only and equal thresholded outcomes."""
    _names(strong, weak)
    if not isinstance(quality_threshold, Real) or isinstance(quality_threshold, (bool, np.bool_)) or not np.isfinite(quality_threshold) or not 0 <= quality_threshold <= 1:
        raise ValueError("El umbral de calidad debe ser finito y estar en [0,1].")
    pair, _ = _outcomes(scores, strong, weak)
    strong_success = pair[strong].ge(quality_threshold).to_numpy()
    weak_success = pair[weak].ge(quality_threshold).to_numpy()
    return pd.Series(np.where(strong_success & ~weak_success, "strong_only",
                              np.where(weak_success & ~strong_success, "weak_only", "equal")),
                     index=scores.index, name="pairwise_outcome")


class PairwiseRegretRouter(BaseEstimator):
    """Estimate expected success gain and costs with disjoint calibration.

    Three one-vs-rest classifiers receive independent sigmoid calibration; the
    calibrated outputs are normalized to a probability simplex. Normalization
    does not guarantee calibration of the final three-way probabilities.
    A class absent from either fitting partition uses the explicitly recorded
    Laplace-smoothed calibration prevalence, including when all outcomes tie.
    """

    def __init__(self, strong="gpt-4-1106-preview", weak="mistralai/mixtral-8x7b-chat",
                 representation="hybrid", seed=42, quality_threshold=.75,
                 max_features=20000, min_df=2, alpha=1e-4, model_type="sgd",
                 feature_transformer=None):
        self.strong = strong
        self.weak = weak
        self.representation = representation
        self.seed = seed
        self.quality_threshold = quality_threshold
        self.max_features = max_features
        self.min_df = min_df
        self.alpha = alpha
        self.model_type = model_type
        self.feature_transformer = feature_transformer

    def fit(self, fit_inputs, fit_scores, fit_costs, calibration_inputs,
            calibration_scores):
        _names(self.strong, self.weak)
        if self.model_type not in {"sgd", "logistic"}:
            raise ValueError("model_type debe ser sgd o logistic.")
        if not isinstance(self.alpha, Real) or isinstance(self.alpha, (bool, np.bool_)) or not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha debe ser positivo y finito.")
        _inputs(fit_inputs)
        _inputs(calibration_inputs)
        fit_scores, fit_costs = _outcomes(fit_scores, self.strong, self.weak, fit_costs)
        calibration_scores, _ = _outcomes(calibration_scores, self.strong, self.weak)
        if not fit_inputs.index.equals(fit_scores.index) or not calibration_inputs.index.equals(calibration_scores.index):
            raise ValueError("Inputs y resultados deben estar alineados exactamente.")
        if len(fit_inputs.index.intersection(calibration_inputs.index)):
            raise ValueError("Ajuste y calibración requieren identificadores disjuntos.")
        fit_target = pairwise_targets(fit_scores, self.strong, self.weak, self.quality_threshold)
        calibration_target = pairwise_targets(calibration_scores, self.strong, self.weak, self.quality_threshold)
        self.transformer_ = (clone(self.feature_transformer) if self.feature_transformer is not None else
                             PromptRepresentation(self.representation, self.max_features, self.min_df))
        fit_x = self.transformer_.fit_transform(fit_inputs)
        calibration_x = self.transformer_.transform(calibration_inputs)
        self.predictors_, self.constants_, self.calibration_notes_ = {}, {}, {}
        self.target_counts_ = {"fit": fit_target.value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict(),
                               "calibration": calibration_target.value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict()}
        self.class_prior_ = np.array([(calibration_target.eq(name).sum()+1)/(len(calibration_target)+3)
                                     for name in OUTCOME_CLASSES], dtype=float)
        for position, name in enumerate(OUTCOME_CLASSES):
            y_fit = fit_target.eq(name).to_numpy(dtype=int)
            y_calibration = calibration_target.eq(name).to_numpy(dtype=int)
            if np.unique(y_fit).size < 2 or np.unique(y_calibration).size < 2:
                self.constants_[name] = float((y_calibration.sum()+1)/(len(y_calibration)+2))
                self.calibration_notes_[name] = "constant smoothed calibration prevalence; fit or calibration lacks both binary classes"
                continue
            if self.model_type == "sgd":
                classifier = SGDClassifier(loss="log_loss", alpha=self.alpha, max_iter=1500,
                                           tol=1e-4, average=True, random_state=self.seed+position)
            else:
                # LogisticRegression's summed-loss C is scaled to the fit size.
                classifier = LogisticRegression(C=1/(self.alpha*len(fit_inputs)), solver="liblinear",
                                                max_iter=1500, tol=1e-4, random_state=self.seed+position)
            classifier.fit(fit_x, y_fit)
            calibrated = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
            calibrated.fit(calibration_x, y_calibration)
            self.predictors_[name] = calibrated
            self.calibration_notes_[name] = "one-vs-rest sigmoid on disjoint calibration; normalized jointly at inference"
        # The future call's response length/cost never enters these predictors.
        cost_x = _cost_features(fit_inputs)
        log_cost = np.log(np.maximum(fit_costs.to_numpy(dtype=float), 1e-12))
        self.cost_regressor_ = Ridge(alpha=1.).fit(cost_x, log_cost)
        fitted_log_cost = np.asarray(self.cost_regressor_.predict(cost_x)).reshape(len(fit_inputs), 2)
        self.cost_log_smearing_ = logsumexp(log_cost-fitted_log_cost, axis=0)-np.log(len(fit_inputs))
        self.models_ = (self.strong, self.weak)
        self.classes_ = np.asarray(OUTCOME_CLASSES)
        self.fit_rows_, self.calibration_rows_ = len(fit_inputs), len(calibration_inputs)
        return self

    def predict_tables(self, inputs):
        check_is_fitted(self, ("models_", "transformer_", "cost_regressor_"))
        _inputs(inputs)
        transformed = self.transformer_.transform(inputs)
        values = []
        for name in OUTCOME_CLASSES:
            if name in self.constants_:
                values.append(np.full(len(inputs), self.constants_[name]))
            else:
                calibrated = self.predictors_[name]
                positive = list(calibrated.classes_).index(1)
                values.append(calibrated.predict_proba(transformed)[:, positive])
        probabilities = np.column_stack(values)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("El modelo devolvió probabilidades inválidas.")
        normalizer = probabilities.sum(axis=1, keepdims=True)
        zero_rows = normalizer[:, 0] <= 0
        probabilities[zero_rows] = self.class_prior_
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        advantage = pd.DataFrame(probabilities, columns=OUTCOME_CLASSES, index=inputs.index)
        advantage["expected_quality_gain"] = advantage["strong_only"]-advantage["weak_only"]
        log_estimate = np.asarray(self.cost_regressor_.predict(_cost_features(inputs))).reshape(len(inputs), 2)
        costs = np.exp(np.clip(log_estimate+self.cost_log_smearing_, -50, np.log(np.finfo(float).max)-1))
        return advantage, pd.DataFrame(costs, columns=self.models_, index=inputs.index)


def choose_pairwise(advantage: pd.DataFrame, predicted_costs: pd.DataFrame,
                    strong: str, weak: str, tradeoff: float) -> pd.DataFrame:
    """Choose strong iff expected quality gain exceeds tradeoff × extra cost.

    The cost difference is signed: a less expensive strong model can be chosen
    even with equal expected quality. An exact utility tie selects weak.
    ``tradeoff`` is quality probability per unit of predicted historical cost.
    Observed scores and costs have no role in this selection.
    """
    _names(strong, weak)
    if not isinstance(tradeoff, Real) or isinstance(tradeoff, (bool, np.bool_)) or not np.isfinite(tradeoff) or tradeoff < 0:
        raise ValueError("tradeoff debe ser finito y no negativo.")
    if not isinstance(advantage, pd.DataFrame) or advantage.empty or not advantage.index.is_unique or not advantage.columns.is_unique:
        raise ValueError("Se requieren probabilidades no vacías con ejes únicos.")
    allowed = {*OUTCOME_CLASSES, "expected_quality_gain"}
    if not set(OUTCOME_CLASSES).issubset(advantage.columns) or set(advantage.columns)-allowed:
        raise ValueError("La tabla debe contener solo las probabilidades de resultados pareados y su diferencia.")
    if not isinstance(predicted_costs, pd.DataFrame) or not predicted_costs.index.equals(advantage.index) or not predicted_costs.index.is_unique or not predicted_costs.columns.is_unique or set(predicted_costs.columns) != {strong, weak}:
        raise ValueError("Probabilidades y costos del par deben estar alineados exactamente.")
    probabilities = advantage.loc[:, OUTCOME_CLASSES].to_numpy(dtype=float)
    costs = predicted_costs.loc[:, [strong, weak]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Las probabilidades pareadas deben ser finitas, estar en [0,1] y sumar uno.")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Los costos previstos deben ser finitos y no negativos.")
    gain = probabilities[:, 0]-probabilities[:, 1]
    if "expected_quality_gain" in advantage:
        supplied = advantage["expected_quality_gain"].to_numpy(dtype=float)
        if not np.isfinite(supplied).all() or not np.allclose(supplied, gain, atol=1e-10, rtol=0):
            raise ValueError("La ganancia declarada no coincide con las probabilidades.")
    extra_cost = costs[:, 0]-costs[:, 1]
    with np.errstate(over="ignore"):
        choose_strong = gain > float(tradeoff)*extra_cost
    return pd.DataFrame({"selected_model": pd.array(np.where(choose_strong, strong, weak), dtype="string"),
                         "predicted_quality_gain": gain, "predicted_extra_cost": extra_cost,
                         "decision": np.where(choose_strong, "pairwise_strong", "pairwise_weak")},
                        index=advantage.index)
