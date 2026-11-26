"""Local Ollama document conversations, loaded only when this view is used."""
from __future__ import annotations

import hashlib
from ipaddress import ip_address
import json
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlsplit

from .errors import LiveBackendError


_RESEARCH = "reports/relay-v3-development-v1"
_INSTALL = "Esta vista necesita la instalación completa de Relay con sus artefactos de investigación y Ollama."
_LIMITATION = ("La selección proyecta el catálogo histórico sobre modelos locales. "
               "El desempeño y el ahorro de esos modelos requieren una evaluación propia.")


class DocumentsUnavailable(RuntimeError):
    """The bundled research profile or its local runtime is unavailable."""


def _artifact_exists(directory, filename):
    return (isinstance(filename, str) and Path(filename).name == filename
            and Path(filename).suffix == ".joblib"
            and (directory / filename).resolve().parent == directory.resolve()
            and (directory / filename).is_file())


def _v3_available(root):
    directory = root / _RESEARCH
    try:
        manifest = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        return _artifact_exists(directory, manifest["selected_for_future_evaluation"]["artifact"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _load_service(root, profile):
    if not (root / "configs/live_catalog.json").is_file():
        raise DocumentsUnavailable(_INSTALL)
    try:
        from routerbench.demo import RelayDemoService

        service = RelayDemoService(root, research_run=_RESEARCH if profile == "experimental_v3" else None)
        if profile == "frozen_v2":
            run = (root / service.primary["run_dir"]).resolve()
            if not run.is_relative_to(root) or not _artifact_exists(run, service.primary["artifact"]):
                raise DocumentsUnavailable(_INSTALL)
        host = urlsplit(service.catalog["base_url"]).hostname
        if host != "localhost" and not ip_address(host).is_loopback:
            raise DocumentsUnavailable("El catálogo de esta vista requiere una dirección local de Ollama.")
        return service
    except (ImportError, OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise DocumentsUnavailable(_INSTALL) from exc


def _history(value):
    if not isinstance(value, list):
        raise ValueError("El historial debe ser una lista de mensajes.")
    history = []
    for row in value:
        if (not isinstance(row, dict) or row.get("role") not in ("user", "assistant")
                or not isinstance(row.get("content"), str)):
            raise ValueError("El historial contiene un mensaje inválido.")
        msg = {"role": row["role"], "content": row["content"]}
        if "model_content" in row:
            if not isinstance(row["model_content"], str):
                raise ValueError("El contenido del historial debe ser texto.")
            msg["model_content"] = row["model_content"]
        digest = row.get("document_sha256")
        if digest is not None:
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("La huella del documento del historial es inválida.")
            msg["document_sha256"] = digest
        history.append(msg)
    return history


class DocumentService:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._services = {}
        self._load_lock = Lock()
        self._generation_lock = Lock()

    def _service(self, profile, v3_available):
        if profile == "experimental_v3" and not v3_available:
            raise DocumentsUnavailable("El artefacto experimental v3 no está disponible en esta instalación.")
        with self._load_lock:
            if profile not in self._services:
                self._services[profile] = _load_service(self.root, profile)
            return self._services[profile]

    def request(self, action: str, data: dict) -> dict:
        if action not in ("documentos-state", "documentos-send") or not isinstance(data, dict):
            raise ValueError("Solicitud de documentos desconocida.")
        profile = data.get("profile", "frozen_v2")
        if profile not in ("frozen_v2", "experimental_v3"):
            raise ValueError("Elige un perfil de Relay disponible.")
        v3_available = _v3_available(self.root)
        if action == "documentos-state":
            try:
                service = self._service(profile, v3_available)
                status = service.status()
            except DocumentsUnavailable as exc:
                return {"available": False, "ready": False, "models": [], "v3_available": v3_available, "notice": str(exc)}
            models = status["available_models"]
            ready = bool(status["live_ready"] and models)
            notice = "" if ready else (status.get("backend_error") or "Conecta Ollama e instala un modelo del catálogo local.")
            return {"available": True, "ready": ready, "models": models, "v3_available": v3_available, "notice": notice}

        prompt = data.get("prompt")
        document = data.get("document", "")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(document, str):
            raise ValueError("Escribe una consulta y usa texto UTF-8 para el documento.")
        history = _history(data.get("history", []))
        document = document.removeprefix("\ufeff")
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest() if document else None
        if digest and any(msg.get("document_sha256") == digest for msg in history):
            document = ""
        service = self._service(profile, v3_available)
        try:
            with self._generation_lock:
                result = service.respond(prompt, history=history, document=document)
        except (LiveBackendError, OSError, ImportError) as exc:
            raise DocumentsUnavailable(f"No se pudo completar la respuesta local. {exc}") from exc
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise DocumentsUnavailable("El modelo local no devolvió una respuesta de texto.")
        content = f"DOCUMENTO DE REFERENCIA:\n{document}\n\nCONSULTA:\n{prompt}" if document else prompt
        detail = {key: result.get(key) for key in (
            "live_model", "historical_model", "historical_tier", "decision", "router_seconds",
            "generation_seconds", "cost_usd", "finish_reason", "router_mode", "mapping_status", "warning",
        )}
        detail["usage"] = {key: (result.get("usage") or {}).get(key) for key in ("input_tokens", "output_tokens")}
        detail["limitation"] = _LIMITATION
        return {"answer": answer, "user_message": {"role": "user", "content": prompt, "model_content": content,
                                                   "document_sha256": digest},
                "assistant_message": {"role": "assistant", "content": answer}, "detail": detail}
