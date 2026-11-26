"""Explicit activity routing for Relay's first maintenance integration.

This is a configurable operational policy, not a learned quality predictor.
The task text is never scanned for commands or model-routing instructions.
"""
from __future__ import annotations

import time


def _model_name(value: str) -> str:
    if (type(value) is not str or not value or value != value.strip()
            or any(char.isspace() for char in value)):
        raise ValueError("El modelo debe tener un identificador exacto sin espacios.")
    return value


class ActivityRouter:
    """Choose the configured model for the harness's current activity.

    Activity comes from the harness, never from model-generated instructions.
    ``always_strong`` means the configured editing model on every activity;
    the label is a control role, not a guarantee of that model's quality.
    """

    def __init__(self, inspection_model: str, editing_model: str, mode: str = "activity"):
        self.inspection_model = _model_name(inspection_model)
        self.editing_model = _model_name(editing_model)
        if type(mode) is not str or mode not in {"activity", "always_strong", "verified"}:
            raise ValueError("mode debe ser activity, always_strong o verified.")
        self.mode = mode

    def route(self, activity: str, *, attempt: int = 0) -> dict:
        if type(activity) is not str or activity not in {"inspection", "editing", "repair"}:
            raise ValueError("Actividad de mantenimiento desconocida.")
        if type(attempt) is not int or attempt < 0:
            raise ValueError("attempt debe ser un entero no negativo.")
        started = time.perf_counter()
        if self.mode == "always_strong":
            selected = self.editing_model
            reason = "Modelo de edición fijo para todas las actividades."
        elif activity == "inspection":
            selected = self.inspection_model
            reason = "Modelo configurado para inspección y planificación."
        elif self.mode == "verified" and activity == "editing":
            selected = self.inspection_model
            reason = "Primer intento económico; las pruebas públicas determinan si hace falta reparación fuerte."
        elif activity == "repair":
            selected = self.editing_model
            reason = "Modelo configurado para corregir el cambio tras la verificación."
        else:
            selected = self.editing_model
            reason = "Modelo configurado para escribir el cambio."
        return {
            "selected_model": selected,
            "reason": reason,
            "policy_mode": {"activity": "activity_policy_v1", "always_strong": "always_strong", "verified": "verified_cascade_v1"}[self.mode],
            "activity": activity,
            "attempt": attempt,
            "router_seconds": max(0., time.perf_counter() - started),
            "evaluation_scope": "configured_activity_policy_not_learned",
        }

    def manifest(self) -> dict:
        return {
            "mode": self.mode,
            "inspection_model": self.inspection_model,
            "editing_model": self.editing_model,
            "policy": "verified_cascade_v1" if self.mode == "verified" else "activity_policy_v1",
            "learned": False,
            "quality_validated": False,
        }
