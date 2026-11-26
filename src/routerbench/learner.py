"""Local per-model success prediction with disjoint probability calibration.

This is a new routing experiment. It does not replace the original tier target.
No scores, observed costs or model responses are inputs at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge, SGDClassifier

from .features import WORD_PATTERN
from .representations import PromptRepresentation


def _check_outcomes(scores: pd.DataFrame, costs: pd.DataFrame | None = None) -> None:
    if not scores.index.is_unique or not scores.columns.is_unique or scores.empty:
        raise ValueError("Los resultados deben tener filas y modelos únicos, no vacíos.")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Los scores deben estar en [0,1] y ser finitos.")
    if costs is not None:
        if not costs.index.equals(scores.index) or not costs.columns.equals(scores.columns):
            raise ValueError("Scores y costos desalineados.")
        if not np.isfinite(costs.to_numpy(dtype=float)).all() or (costs.to_numpy(dtype=float) < 0).any():
            raise ValueError("Los costos deben ser finitos y no negativos.")


def _cost_features(inputs: pd.DataFrame) -> np.ndarray:
    # These are all available before generation. Stored output tokens/costs are
    # never used as prediction inputs. Three features keep cost inference cheap.
    names = ["n_chars", "n_words", "n_lines"]
    if set(names).issubset(inputs.columns):
        values = inputs.loc[:, names].to_numpy(dtype=float)
    else:
        texts = inputs["prompt_text"].tolist()
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Los prompts deben ser strings.")
        values = np.array([[len(text), len(WORD_PATTERN.findall(text)), len(text.splitlines())] for text in texts], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Atributos de longitud inválidos.")
    return np.log1p(values)


class SuccessCostRouter:
    """Estimate per-model success and historical cost from prompt-only inputs.

    Calibration is a fitted procedure, not a guarantee of reliable uncertainty.
    Its out-of-sample behavior must be measured on development validation.
    """

    def __init__(self, representation: str = "hybrid", seed: int = 42,
                 quality_threshold: float = 0.75, max_features: int = 12000,
                 min_df: int = 2, alpha: float = 0.0001, feature_transformer=None):
        if not np.isfinite(quality_threshold) or not 0 <= quality_threshold <= 1:
            raise ValueError("quality_threshold debe estar en [0,1].")
        self.representation = representation
        self.seed = seed
        self.quality_threshold = quality_threshold
        self.max_features = max_features
        self.min_df = min_df
        self.alpha = alpha
        self.feature_transformer = feature_transformer

    def fit(self, fit_inputs: pd.DataFrame, fit_scores: pd.DataFrame,
            fit_costs: pd.DataFrame, calibration_inputs: pd.DataFrame,
            calibration_scores: pd.DataFrame) -> "SuccessCostRouter":
        _check_outcomes(fit_scores, fit_costs)
        _check_outcomes(calibration_scores)
        if not fit_inputs.index.equals(fit_scores.index) or not calibration_inputs.index.equals(calibration_scores.index):
            raise ValueError("Inputs y outcomes desalineados.")
        if len(fit_inputs.index.intersection(calibration_inputs.index)):
            raise ValueError("Fit y calibración deben tener identificadores disjuntos.")
        if set(fit_scores.columns) != set(calibration_scores.columns):
            raise ValueError("Los conjuntos deben evaluar los mismos modelos.")
        self.models_ = tuple(sorted(fit_scores.columns))
        self.transformer_ = (clone(self.feature_transformer) if self.feature_transformer is not None else
                             PromptRepresentation(mode=self.representation, max_features=self.max_features, min_df=self.min_df))
        fit_x = self.transformer_.fit_transform(fit_inputs)
        calibration_x = self.transformer_.transform(calibration_inputs)
        self.predictors_ = {}
        self.constants_ = {}
        self.calibration_notes_ = {}
        for position, model in enumerate(self.models_):
            fit_target = fit_scores[model].ge(self.quality_threshold).to_numpy(dtype=int)
            calibration_target = calibration_scores[model].ge(self.quality_threshold).to_numpy(dtype=int)
            if np.unique(fit_target).size < 2 or np.unique(calibration_target).size < 2:
                self.constants_[model] = float((calibration_target.sum() + 1) / (len(calibration_target) + 2))
                self.calibration_notes_[model] = "constant smoothed calibration prevalence; a partition lacks both classes"
                continue
            classifier = SGDClassifier(loss="log_loss", alpha=self.alpha,
                                       max_iter=1500, tol=1e-4, average=True,
                                       random_state=self.seed + position)
            classifier.fit(fit_x, fit_target)
            calibrator = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
            calibrator.fit(calibration_x, calibration_target)
            self.predictors_[model] = calibrator
            self.calibration_notes_[model] = "sigmoid fitted on disjoint calibration rows"
        # Log-linear historical cost model with a train-only residual smearing
        # correction. This is an estimate, never the observed future row cost.
        cost_x = _cost_features(fit_inputs)
        observed = fit_costs.loc[:, list(self.models_)].to_numpy(dtype=float)
        log_cost = np.log(np.maximum(observed, 1e-12))
        self.cost_regressor_ = Ridge(alpha=1.0).fit(cost_x, log_cost)
        predicted_log_cost = np.asarray(self.cost_regressor_.predict(cost_x)).reshape(len(fit_inputs), len(self.models_))
        residual = log_cost - predicted_log_cost
        self.cost_log_smearing_ = logsumexp(residual, axis=0) - np.log(len(fit_inputs))
        self.fit_rows_ = len(fit_inputs)
        self.calibration_rows_ = len(calibration_inputs)
        return self

    def predict_tables(self, inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, "models_"):
            raise ValueError("El router todavía no está entrenado.")
        if not inputs.index.is_unique or inputs.empty:
            raise ValueError("Se requieren inputs no vacíos con índice único.")
        transformed = self.transformer_.transform(inputs)
        probabilities = {}
        for model in self.models_:
            if model in self.constants_:
                probabilities[model] = np.full(len(inputs), self.constants_[model])
            else:
                classifier = self.predictors_[model]
                positive_column = list(classifier.classes_).index(1)
                probabilities[model] = classifier.predict_proba(transformed)[:, positive_column]
        log_estimate = np.asarray(self.cost_regressor_.predict(_cost_features(inputs))).reshape(len(inputs), len(self.models_))
        estimated_cost = np.exp(np.clip(log_estimate + self.cost_log_smearing_, -50, np.log(np.finfo(float).max) - 1))
        return (pd.DataFrame(probabilities, index=inputs.index),
                pd.DataFrame(estimated_cost, columns=self.models_, index=inputs.index))
