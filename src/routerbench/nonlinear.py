"""Dense prompt representations and nonlinear, calibrated pairwise regret.

Vocabulary, SVD, structural scaling, predictors and cost estimates use fitting
rows only. Calibration uses its own disjoint rows. Neither class reads data or
selects policies from observed future outcomes.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_NAMES
from .learner import _cost_features
from .pairwise import OUTCOME_CLASSES, _inputs, _names, _outcomes, pairwise_targets
from .representations import PromptRepresentation


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo.")


class DensePromptRepresentation(TransformerMixin, BaseEstimator):
    """Fit-only lexical SVD plus magnitude-preserving structural attributes.

    ``lexical_svd`` concatenates a randomized truncated SVD of full-prompt
    word/character TF-IDF and fourteen log1p/MaxAbs structural columns.
    ``structural`` returns only those fourteen columns. No dense output row is
    normalized: prompt length remains available to the nonlinear predictor.
    The SVD rank is capped by fitting rows and lexical columns. It is not an
    assertion of semantic understanding or exact reconstruction of the text.
    """

    def __init__(self, mode="lexical_svd", n_components=64, max_features=12000,
                 min_df=2, seed=42):
        self.mode = mode
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.seed = seed

    def fit(self, X, y=None):
        _inputs(X)
        if self.mode not in {"lexical_svd", "structural"}:
            raise ValueError("mode debe ser lexical_svd o structural.")
        _positive_integer(self.n_components, "n_components")
        _positive_integer(self.max_features, "max_features")
        texts, precomputed = PromptRepresentation._validate_frame(X)
        values = PromptRepresentation._structural_values(X, texts, precomputed)
        self.structural_scaler_ = MaxAbsScaler().fit(np.log1p(values))
        self.mode_ = self.mode
        self.n_components_ = 0
        if self.mode_ == "lexical_svd":
            if len(X) < 2:
                raise ValueError("SVD requiere al menos dos filas de ajuste.")
            self.lexical_ = PromptRepresentation(mode="lexical", max_features=self.max_features,
                                                  min_df=self.min_df)
            lexical = self.lexical_.fit_transform(X)
            if lexical.shape[1] < 2:
                raise ValueError("SVD requiere al menos dos columnas léxicas.")
            self.n_components_ = min(int(self.n_components), len(X)-1, lexical.shape[1]-1)
            self.svd_ = TruncatedSVD(n_components=self.n_components_, algorithm="randomized",
                                    n_iter=5, random_state=self.seed).fit(lexical)
        self.feature_names_out_ = np.asarray(
            [f"svd__{i}" for i in range(self.n_components_)]+
            [f"structural__{name}" for name in FEATURE_NAMES], dtype=object)
        self.n_features_out_ = len(self.feature_names_out_)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, ("mode_", "structural_scaler_", "n_features_out_"))
        _inputs(X)
        texts, precomputed = PromptRepresentation._validate_frame(X)
        values = PromptRepresentation._structural_values(X, texts, precomputed)
        structural = self.structural_scaler_.transform(np.log1p(values))
        if self.mode_ == "lexical_svd":
            result = np.column_stack((self.svd_.transform(self.lexical_.transform(X)), structural))
        else:
            result = structural
        if result.shape != (len(X), self.n_features_out_) or not np.isfinite(result).all():
            raise ValueError("La representación densa devolvió valores o dimensiones inválidos.")
        return np.asarray(result, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class NonlinearRegretRouter(BaseEstimator):
    """HistGradientBoosting for three threshold-success outcomes of a fixed pair.

    Early stopping is disabled so no implicit fitting/calibration split occurs.
    A multiclass sigmoid calibrator uses the separate calibration population.
    Classes absent from either partition receive explicit Laplace-smoothed
    calibration prevalence. Supported classes are trained and calibrated on
    their fitting/calibration rows and share the remaining probability mass.
    With fewer than two supported classes, all predictions use a three-class
    smoothed calibration prior. This fallback is recorded, not silently ignored.
    """

    def __init__(self, strong="gpt-4-1106-preview", weak="mistralai/mixtral-8x7b-chat",
                 representation="lexical_svd", n_components=64, max_features=12000,
                 min_df=2, max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
                 l2_regularization=1., seed=42, quality_threshold=.75,
                 feature_transformer=None):
        self.strong = strong
        self.weak = weak
        self.representation = representation
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_leaf_nodes = max_leaf_nodes
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.seed = seed
        self.quality_threshold = quality_threshold
        self.feature_transformer = feature_transformer

    @staticmethod
    def _dense(values, n_rows):
        if sparse.issparse(values):
            raise ValueError("El clasificador no lineal requiere una representación densa explícita.")
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[0] != n_rows or not array.shape[1] or not np.isfinite(array).all():
            raise ValueError("La representación debe ser una matriz densa finita alineada.")
        return array

    def fit(self, fit_inputs, fit_scores, fit_costs, calibration_inputs, calibration_scores):
        _names(self.strong, self.weak)
        for value, name in ((self.max_leaf_nodes, "max_leaf_nodes"), (self.max_iter, "max_iter"),
                            (self.min_samples_leaf, "min_samples_leaf")):
            _positive_integer(value, name)
        if self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes debe ser al menos dos.")
        if not isinstance(self.l2_regularization, Real) or isinstance(self.l2_regularization, (bool, np.bool_)) or not np.isfinite(self.l2_regularization) or self.l2_regularization < 0:
            raise ValueError("l2_regularization debe ser finita y no negativa.")
        _inputs(fit_inputs)
        _inputs(calibration_inputs)
        fit_scores, fit_costs = _outcomes(fit_scores, self.strong, self.weak, fit_costs)
        calibration_scores, _ = _outcomes(calibration_scores, self.strong, self.weak)
        if not fit_inputs.index.equals(fit_scores.index) or not calibration_inputs.index.equals(calibration_scores.index):
            raise ValueError("Inputs y resultados deben estar alineados exactamente.")
        if len(fit_inputs.index.intersection(calibration_inputs.index)):
            raise ValueError("Ajuste y calibración requieren identificadores disjuntos.")
        fit_target = pairwise_targets(fit_scores, self.strong, self.weak, self.quality_threshold)
        cal_target = pairwise_targets(calibration_scores, self.strong, self.weak, self.quality_threshold)
        codes = {name: position for position, name in enumerate(OUTCOME_CLASSES)}
        y_fit = fit_target.map(codes).to_numpy(dtype=int)
        y_cal = cal_target.map(codes).to_numpy(dtype=int)
        self.transformer_ = (clone(self.feature_transformer) if self.feature_transformer is not None else
                             DensePromptRepresentation(self.representation, self.n_components,
                                                       self.max_features, self.min_df, self.seed))
        fit_x = self._dense(self.transformer_.fit_transform(fit_inputs), len(fit_inputs))
        cal_x = self._dense(self.transformer_.transform(calibration_inputs), len(calibration_inputs))
        self.target_counts_ = {"fit": fit_target.value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict(),
                               "calibration": cal_target.value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_dict()}
        self.class_prior_ = np.asarray([(cal_target.eq(name).sum()+1)/(len(cal_target)+3)
                                        for name in OUTCOME_CLASSES], dtype=float)
        self.supported_classes_ = np.intersect1d(np.unique(y_fit), np.unique(y_cal))
        self.predictor_ = None
        self.constants_ = {}
        self.calibration_notes_ = {}
        self.classifier_fit_rows_ = self.classifier_calibration_rows_ = 0
        if len(self.supported_classes_) >= 2:
            fit_mask = np.isin(y_fit, self.supported_classes_)
            cal_mask = np.isin(y_cal, self.supported_classes_)
            classifier = HistGradientBoostingClassifier(max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization, early_stopping=False, random_state=self.seed)
            classifier.fit(fit_x[fit_mask], y_fit[fit_mask])
            self.predictor_ = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
            self.predictor_.fit(cal_x[cal_mask], y_cal[cal_mask])
            self.classifier_fit_rows_ = int(fit_mask.sum())
            self.classifier_calibration_rows_ = int(cal_mask.sum())
            for position, name in enumerate(OUTCOME_CLASSES):
                if position not in self.supported_classes_:
                    self.constants_[name] = float(self.class_prior_[position])
                    self.calibration_notes_[name] = "class absent from fit or calibration; smoothed calibration prevalence"
                else:
                    self.calibration_notes_[name] = "multiclass sigmoid on disjoint supported calibration rows; unsupported class mass held constant"
        else:
            self.constants_ = dict(zip(OUTCOME_CLASSES, self.class_prior_.tolist()))
            self.calibration_notes_ = {name: "fewer than two shared classes; full smoothed calibration prior"
                                       for name in OUTCOME_CLASSES}
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
        transformed = self._dense(self.transformer_.transform(inputs), len(inputs))
        probabilities = np.tile(self.class_prior_, (len(inputs), 1))
        if self.predictor_ is not None:
            supported = self.predictor_.predict_proba(transformed)
            retained_mass = 1-sum(self.constants_.values())
            for position, code in enumerate(self.predictor_.classes_):
                probabilities[:, int(code)] = supported[:, position]*retained_mass
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("El modelo devolvió probabilidades inválidas.")
        advantage = pd.DataFrame(probabilities, index=inputs.index, columns=OUTCOME_CLASSES)
        advantage["expected_quality_gain"] = advantage["strong_only"]-advantage["weak_only"]
        log_estimate = np.asarray(self.cost_regressor_.predict(_cost_features(inputs))).reshape(len(inputs), 2)
        costs = np.exp(np.clip(log_estimate+self.cost_log_smearing_, -50, np.log(np.finfo(float).max)-1))
        return advantage, pd.DataFrame(costs, columns=self.models_, index=inputs.index)
