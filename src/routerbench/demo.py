"""Local Relay demo: frozen historical router plus explicit catalog projection.

The live local catalog is an integration demonstration, not a calibrated
transfer of RouterBench scores or historical dollar costs to current models.
"""
from __future__ import annotations

import hashlib
import json
import time
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

from .data import sha256_file
from .live import discover_ollama, generate_ollama
from .policy import choose_models


def assemble_messages(prompt: str, history: list, document: str = "") -> list[dict]:
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(document, str):
        raise ValueError("Escribe una consulta de texto no vacía.")
    if not isinstance(history, list):
        raise ValueError("El historial debe ser una lista.")
    messages = [{"role": "system", "content": "Eres un asistente útil. Responde en el idioma del usuario. Si usas documentos, distingue lo que dicen de tus inferencias y no inventes citas."}]
    for message in history:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
            raise ValueError("El historial contiene mensajes inválidos.")
        model_content = message.get("model_content", message["content"])
        if not isinstance(model_content, str):
            raise ValueError("El contenido conservado para el modelo debe ser texto.")
        messages.append({"role": message["role"], "content": model_content})
    content = ("DOCUMENTO DE REFERENCIA:\n" + document + "\n\nCONSULTA:\n" + prompt) if document else prompt
    messages.append({"role": "user", "content": content})
    return messages


def conservative_context_bound(messages: list[dict]) -> int:
    """Conservative UTF-8 budget with template reserve; not exact tokenization."""
    return sum(len(item["content"].encode("utf-8")) + 256 for item in messages) + 1024


class RelayDemoService:
    def __init__(self, root: str | Path, research_run: str | Path | None = None):
        self.root = Path(root).resolve()
        self.catalog = json.loads((self.root / "configs/live_catalog.json").read_text(encoding="utf-8"))
        self.protocol = json.loads((self.root / "configs/final_protocol.json").read_text(encoding="utf-8"))
        self.primary = next(policy for policy in self.protocol["policies"] if policy["role"] == "primary_confirmatory_comparison")
        self.tier_map = json.loads((self.root / "data/processed/manifest.json").read_text(encoding="utf-8"))["tier_map"]
        self._router = None
        self.router_mode = "frozen_v2"
        self._research_directory = None
        if research_run is not None:
            self._configure_research(research_run)

    def _configure_research(self, research_run):
        directory = (self.root / research_run).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("El experimento debe estar dentro del proyecto.")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(summary.get(name) is not False for name in
               ("outer_test_accessed_for_v3", "confirmation_of_superiority", "automatic_live_promotion")):
            raise ValueError("La demo v3 requiere un experimento exclusivamente de desarrollo.")
        configuration = summary["configuration"]
        if configuration.get("outer_test_evaluation_allowed") is not False or configuration.get("automatic_live_promotion_allowed") is not False:
            raise ValueError("La configuración v3 no permite usar test ni promover el candidato automáticamente.")
        digest = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != summary["configuration_sha256"]:
            raise ValueError("La configuración v3 no coincide con su hash.")
        selection = summary["selected_for_future_evaluation"]
        candidate = summary["candidates"].get(selection["candidate_id"])
        if not candidate or selection.get("requires_independent_evaluation") is not True:
            raise ValueError("No hay un candidato v3 entrenado y seleccionado para evaluar.")
        specification = candidate["specification"]
        configured = [entry for entry in configuration["candidates"] if entry["id"] == selection["candidate_id"]]
        if configured != [specification] or specification["objective"] not in {"absolute", "pairwise"}:
            raise ValueError("La selección v3 no coincide con sus candidatos declarados.")
        if selection["point"] != candidate["selected_point"] or selection["point"]["candidate_id"] != specification["id"]:
            raise ValueError("El punto operativo v3 no coincide con la selección registrada.")
        parameter = selection["point"]["parameter"]
        if not isinstance(parameter, Real) or isinstance(parameter, bool) or not pd.notna(parameter) or not 0 <= parameter < float("inf") or (specification["objective"] == "absolute" and parameter > 1):
            raise ValueError("El parámetro de selección v3 es inválido.")
        grid_name = "absolute_probability_grid" if specification["objective"] == "absolute" else "pairwise_tradeoff_grid"
        if parameter not in configuration[grid_name] or selection["point"] not in candidate["curve"] or selection["point"]["metrics"]["success_rate"] < summary["quality_floor"]:
            raise ValueError("El punto v3 no pertenece a su grilla factible de desarrollo.")
        artifact = selection.get("artifact")
        if not isinstance(artifact, str) or Path(artifact).name != artifact or Path(artifact).suffix != ".joblib" or artifact != candidate["artifact"] or selection["artifact_sha256"] != candidate["artifact_sha256"] or summary["files_sha256"].get(artifact) != selection["artifact_sha256"]:
            raise ValueError("El artefacto v3 no coincide con su selección.")
        pair = configuration["common_pair"]
        if pair != {"strong": "gpt-4-1106-preview", "weak": "mistralai/mixtral-8x7b-chat"} or self.tier_map.get(pair["strong"]) != 2 or self.tier_map.get(pair["weak"]) != 1:
            raise ValueError("La proyección experimental v3 requiere el par histórico GPT-4/Mixtral.")
        self._research_directory = directory
        self._research_summary_sha256 = sha256_file(summary_path)
        self._research_selection = selection
        self._research_specification = specification
        self._research_pair = pair
        self.router_mode = "experimental_v3"

    def _load_router(self):
        if self._router is None:
            if getattr(self, "_research_directory", None) is not None:
                directory = self._research_directory
                selection = self._research_selection
                if sha256_file(directory / "summary.json") != self._research_summary_sha256 or sha256_file(directory / selection["artifact"]) != selection["artifact_sha256"]:
                    raise ValueError("El router v3 no coincide con el artefacto seleccionado.")
                artifact = directory / selection["artifact"]
            else:
                directory = self.root / self.primary["run_dir"]
                if sha256_file(directory / "summary.json") != self.primary["source_summary_sha256"] or sha256_file(directory / self.primary["artifact"]) != self.primary["artifact_sha256"]:
                    raise ValueError("El router no coincide con el artefacto congelado.")
                artifact = directory / self.primary["artifact"]
            self._router = joblib.load(artifact)
        return self._router

    def status(self) -> dict:
        experimental = getattr(self, "router_mode", "frozen_v2") == "experimental_v3"
        variant = self._research_selection["candidate_id"] if experimental else self.primary["variant"]
        threshold = self.primary["minimum_predicted_success"]
        if experimental:
            threshold = self._research_selection["point"]["parameter"] if self._research_specification["objective"] == "absolute" else None
        try:
            available = discover_ollama(self.catalog["base_url"])
            names = [item["model"] for item in available]
            candidates = [item["model"] for item in self.catalog["candidate_models"]
                          if item["model"] in names and (not experimental or item["tier"] in {1, 2})]
            return {"available_models": candidates, "live_ready": bool(candidates),
                    "primary_variant": variant, "threshold": threshold,
                    "backend_error": None, "mapping_status": self.catalog["mapping_status"],
                    "router_mode": getattr(self, "router_mode", "frozen_v2")}
        except Exception as error:
            return {"available_models": [], "live_ready": False, "primary_variant": variant, "threshold": threshold,
                    "backend_error": str(error), "mapping_status": self.catalog["mapping_status"],
                    "router_mode": getattr(self, "router_mode", "frozen_v2")}

    def _route(self, prompt, history, document):
        messages = assemble_messages(prompt, history, document)
        start = time.perf_counter()
        model = self._load_router()
        available = {item["model"]: item for item in discover_ollama(self.catalog["base_url"], include_context=True)}
        bound = conservative_context_bound(messages)
        reserve = self.catalog["default_max_output_tokens"]
        eligible = []
        for candidate in self.catalog["candidate_models"]:
            info = available.get(candidate["model"])
            if not info:
                continue
            native = info.get("context_length")
            if not isinstance(native, int) or native <= reserve:
                continue
            context_window = min(native, self.catalog.get("maximum_runtime_context", 16384))
            if bound + reserve <= context_window:
                eligible.append({**candidate, "context_window": context_window})
        if not eligible:
            raise ValueError("No hay un modelo local instalado con contexto suficiente según la estimación conservadora. El texto no fue recortado. Revisa el catálogo o procesa el documento por etapas.")
        live_by_tier = {item["tier"]: item for item in eligible}
        experimental = getattr(self, "router_mode", "frozen_v2") == "experimental_v3"
        compatible_pair = []
        if experimental:
            compatible_pair = [name for name in self._research_pair.values() if self.tier_map[name] in live_by_tier]
            if not compatible_pair:
                raise ValueError("No hay un modelo local compatible con el par experimental v3 y el contexto completo.")
        if experimental and len(compatible_pair) == 1:
            historical_model = compatible_pair[0]
            decision = "capability_fallback"
        else:
            router_text = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
            probabilities, predicted_costs = model.predict_tables(pd.DataFrame({"prompt_text": [router_text]}))
            if experimental:
                strong, weak = self._research_pair["strong"], self._research_pair["weak"]
                parameter = self._research_selection["point"]["parameter"]
                if self._research_specification["objective"] == "pairwise":
                    from .pairwise import choose_pairwise
                    route = choose_pairwise(probabilities, predicted_costs.loc[:, [strong, weak]], strong, weak, parameter).iloc[0]
                else:
                    route = choose_models(probabilities.loc[:, [strong, weak]], predicted_costs.loc[:, [strong, weak]], parameter).iloc[0]
            else:
                allowed = pd.DataFrame({name: [self.tier_map[name] in live_by_tier] for name in probabilities}, index=probabilities.index)
                route = choose_models(probabilities, predicted_costs, self.primary["minimum_predicted_success"], allowed).iloc[0]
            if pd.isna(route["selected_model"]):
                raise ValueError("No hay una ruta compatible disponible.")
            historical_model = str(route["selected_model"])
            decision = str(route["decision"])
        historical_tier = self.tier_map[historical_model]
        live = live_by_tier[historical_tier]
        result = {"live_model": live["model"], "historical_model": historical_model,
                  "historical_tier": historical_tier, "decision": decision,
                  "router_seconds": time.perf_counter()-start, "mapping_status": self.catalog["mapping_status"],
                  "context_window": live["context_window"], "conservative_input_bound": bound,
                  "router_mode": getattr(self, "router_mode", "frozen_v2"),
                  "warning": "La conexión al catálogo local es experimental. Los resultados históricos no validan la calidad ni el ahorro de estos modelos locales."}
        if experimental:
            result["research_candidate"] = self._research_selection["candidate_id"]
            result["warning"] += " Relay v3 fue elegido con desarrollo reutilizado y todavía necesita evaluación independiente."
        return result, messages

    def route_only(self, prompt: str, history: list, document: str = "") -> dict:
        result, _ = self._route(prompt, history, document)
        return result

    def respond(self, prompt: str, history: list, document: str = "") -> dict:
        result, messages = self._route(prompt, history, document)
        response = generate_ollama(result["live_model"], messages,
                                   max_output_tokens=self.catalog["default_max_output_tokens"], timeout=180,
                                   base_url=self.catalog["base_url"], context_window=result["context_window"])
        return {**result, "answer": response["content"], "generation_seconds": response["duration_seconds"],
                "usage": response["usage"], "cost_usd": response["cost_usd"],
                "provider": response["provider"], "finish_reason": response["finish_reason"], "mode": "live"}
