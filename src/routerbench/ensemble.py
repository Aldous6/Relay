"""Inference-only convex blends of already fitted pairwise regret routers."""
from __future__ import annotations

import copy
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .pairwise import OUTCOME_CLASSES, _inputs


def _metadata(router):
    if not callable(getattr(router, "predict_tables", None)):
        raise ValueError("Cada miembro debe proporcionar predict_tables.")
    models = getattr(router, "models_", None)
    if not isinstance(models, (list, tuple)) or len(models) != 2 or len(set(models)) != 2 or any(not isinstance(name, str) or not name for name in models):
        raise ValueError("Cada miembro debe estar entrenado con dos modelos distintos y ordenados.")
    if getattr(router, "strong", None) != models[0] or getattr(router, "weak", None) != models[1]:
        raise ValueError("Los identificadores strong/weak deben coincidir con models_.")
    quality = getattr(router, "quality_threshold", None)
    if not isinstance(quality, Real) or isinstance(quality, (bool, np.bool_)) or not np.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError("El umbral de calidad de cada miembro debe estar en [0,1].")
    counts = []
    for attribute in ("fit_rows_", "calibration_rows_"):
        value = getattr(router, attribute, None)
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError("Los miembros requieren cantidades de ajuste y calibración positivas.")
        counts.append(int(value))
    targets = getattr(router, "target_counts_", None)
    if not isinstance(targets, dict) or set(targets) != {"fit", "calibration"}:
        raise ValueError("Faltan los soportes de clases de ajuste/calibración del miembro.")
    normalized_targets = {}
    for partition, n_rows in zip(("fit", "calibration"), counts):
        values = targets[partition]
        if not isinstance(values, dict) or set(values) != set(OUTCOME_CLASSES):
            raise ValueError("Los soportes deben contener exactamente las tres clases pareadas.")
        if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 0 for value in values.values()) or sum(values.values()) != n_rows:
            raise ValueError("Los soportes no coinciden con el número de filas del miembro.")
        normalized_targets[partition] = {name: int(values[name]) for name in OUTCOME_CLASSES}
    return {"models": tuple(models), "quality": float(quality), "rows": tuple(counts),
            "target_counts": normalized_targets}


class FrozenRegretEnsemble:
    """Blend fitted routers without fitting predictors or inspecting outcomes.

    Weights must be finite and strictly positive and are normalized once. All
    members must use the same strong/weak ordering, quality threshold and
    reported fit/calibration class supports. Matching counts are a compatibility
    check, not proof of identical row identities; experiment manifests must
    establish that provenance separately. This class has no fit method.

    Convexly blending separately calibrated members does not guarantee that the
    resulting three-class probabilities are calibrated jointly. Members remain
    caller-owned fitted objects and must not be refit externally after assembly.
    """

    def __init__(self, routers, weights=None):
        if not isinstance(routers, (list, tuple)) or len(routers) < 2:
            raise ValueError("El ensemble requiere al menos dos routers entrenados.")
        self.routers = tuple(routers)
        metadata = [_metadata(router) for router in self.routers]
        if any(item != metadata[0] for item in metadata[1:]):
            raise ValueError("Los miembros deben compartir par ordenado, umbral y soportes de ajuste/calibración.")
        if weights is None:
            weights = np.ones(len(self.routers), dtype=float)
        try:
            raw_weights = np.asarray(weights, dtype=object)
            if raw_weights.ndim != 1 or any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in raw_weights):
                raise ValueError("Los pesos deben ser números reales, no indicadores booleanos.")
            values = np.asarray(weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Los pesos deben ser números positivos finitos.") from exc
        if values.ndim != 1 or len(values) != len(self.routers) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Se requiere un peso positivo finito por miembro.")
        # Scaling first also handles valid weights whose direct sum overflows.
        values = values/values.max()
        if (values == 0).any():
            raise ValueError("La escala relativa de los pesos no es representable sin anular un miembro.")
        values = values/values.sum()
        self.weights_ = tuple(values.tolist())
        self.weights = self.weights_
        self._member_metadata = copy.deepcopy(metadata[0])
        self.models_ = metadata[0]["models"]
        self.strong, self.weak = self.models_
        self.quality_threshold = metadata[0]["quality"]
        self.fit_rows_, self.calibration_rows_ = metadata[0]["rows"]
        self.target_counts_ = copy.deepcopy(metadata[0]["target_counts"])
        self.classes_ = np.asarray(OUTCOME_CLASSES)
        self.calibration_notes_ = {name: "fixed convex blend of fitted member probabilities; no additional calibration and no joint calibration guarantee"
                                   for name in OUTCOME_CLASSES}
        self.provenance_note_ = "matching class supports/counts do not verify matching row identities; verify source experiment manifests"

    def predict_tables(self, inputs):
        _inputs(inputs)
        if any(_metadata(router) != self._member_metadata for router in self.routers):
            raise ValueError("Los metadatos de un miembro cambiaron después de ensamblarlo.")
        probabilities, costs = [], []
        required = {*OUTCOME_CLASSES, "expected_quality_gain"}
        for router in self.routers:
            first, predicted_costs = router.predict_tables(inputs)
            if not isinstance(first, pd.DataFrame) or not first.index.equals(inputs.index) or not first.index.is_unique or not first.columns.is_unique or set(first.columns) != required:
                raise ValueError("Las probabilidades de cada miembro deben tener las mismas filas y las tres clases más su ganancia.")
            matrix = first.loc[:, list(OUTCOME_CLASSES)].to_numpy(dtype=float)
            gain = first["expected_quality_gain"].to_numpy(dtype=float)
            if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any() or not np.allclose(matrix.sum(axis=1), 1, atol=1e-8, rtol=0):
                raise ValueError("Cada miembro debe devolver probabilidades finitas en el simplex.")
            if not np.isfinite(gain).all() or not np.allclose(gain, matrix[:, 0]-matrix[:, 1], atol=1e-10, rtol=0):
                raise ValueError("La ganancia de un miembro no coincide con sus probabilidades.")
            if not isinstance(predicted_costs, pd.DataFrame) or not predicted_costs.index.equals(inputs.index) or not predicted_costs.index.is_unique or not predicted_costs.columns.is_unique or set(predicted_costs.columns) != set(self.models_):
                raise ValueError("Los costos de cada miembro deben corresponder a las mismas filas y al mismo par.")
            cost_matrix = predicted_costs.loc[:, list(self.models_)].to_numpy(dtype=float)
            if not np.isfinite(cost_matrix).all() or (cost_matrix < 0).any():
                raise ValueError("Los costos previstos de cada miembro deben ser finitos y no negativos.")
            probabilities.append(matrix)
            costs.append(cost_matrix)
        weights = np.asarray(self.weights_)
        blended = np.tensordot(weights, np.stack(probabilities), axes=(0, 0))
        blended /= blended.sum(axis=1, keepdims=True)
        # A scaled convex average avoids overflowing on finite near-max costs.
        cost_stack = np.stack(costs)
        scale = cost_stack.max(axis=0)
        scaled = np.divide(cost_stack, scale, out=np.zeros_like(cost_stack), where=scale > 0)
        scaled_mean = np.tensordot(weights, scaled, axes=(0, 0))/weights.sum()
        mean_costs = np.clip(scaled_mean, 0, 1)*scale
        first = pd.DataFrame(blended, columns=OUTCOME_CLASSES, index=inputs.index)
        first["expected_quality_gain"] = first["strong_only"]-first["weak_only"]
        return first, pd.DataFrame(mean_costs, columns=self.models_, index=inputs.index)
