"""Command line for the usable Relay maintenance integration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys

from .maintenance import run_maintenance, _save, _render_report
from .maintenance_workspace import Workspace
from .maintenance_routing import ActivityRouter
from .maintenance_provider import GroqMaintenanceProvider

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "maintenance_demo.json"
DEMO_TASK = ("Corrige normalize_labels: elimina espacios exteriores y entradas vacías, normaliza con casefold "
             "y elimina duplicados conservando el orden de primera aparición. No modifiques la lista recibida.")

def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "repository", "source_files", "test_files", "models", "limits", "spending"}
    if (type(config) is not dict or set(config) != required
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("Configuración de mantenimiento incompleta o con campos desconocidos.")
    if type(config["repository"]) is not str or not config["repository"].strip():
        raise ValueError("repository debe ser una ruta explícita.")
    root = (path.parent / config["repository"]).resolve()
    if not root.is_dir():
        raise ValueError("No existe la carpeta del repositorio.")
    for key in ("source_files", "test_files"):
        if type(config[key]) is not list or not config[key] or any(type(p) is not str for p in config[key]):
            raise ValueError(f"{key} debe contener rutas de archivos Python explícitos.")
    models = config["models"]
    if type(models) is not dict or set(models) != {"inspection", "editing"}:
        raise ValueError("models requiere inspection y editing.")
    allowed_models = {"groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"}
    if any(type(name) is not str or name not in allowed_models for name in models.values()):
        raise ValueError("Esta entrega utiliza el catálogo Groq GPT-OSS 20B/120B declarado.")
    limits = config["limits"]
    bounds = {"max_repairs": (0, 2), "max_output_tokens": (256, 4096),
              "max_requests": (2, 6), "request_budget": (3000, 8000),
              "minimum_interval_seconds": (60, 300), "request_timeout_seconds": (10, 180),
              "test_timeout_seconds": (1, 120)}
    if type(limits) is not dict or set(limits) != set(bounds):
        raise ValueError("Límites incompletos.")
    for key, (low, high) in bounds.items():
        if type(limits[key]) is not int or not low <= limits[key] <= high:
            raise ValueError(f"Límite inválido: {key}.")
    if limits["max_requests"] < 2 + limits["max_repairs"]:
        raise ValueError("max_requests no cubre inspección, edición y reparaciones declaradas.")
    spending = config["spending"]
    if (type(spending) is not dict or set(spending) != {"account_is_free_tier", "max_provider_spend_usd"}
            or spending["account_is_free_tier"] is not True
            or type(spending["max_provider_spend_usd"]) is not int
            or spending["max_provider_spend_usd"] != 0):
        raise ValueError("Esta entrega requiere la cuenta Groq Free ya autorizada y presupuesto US$0.")
    # Validation-only construction: never creates a run or reads credentials.
    Workspace(root, root / "relay-validation-unused", config["source_files"], config["test_files"]).validate_inputs()
    return config, root


def _key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        if not sys.stdin.isatty():
            raise ValueError("Falta GROQ_API_KEY. Ejecuta en una terminal para introducirla de forma oculta.")
        key = getpass.getpass("Clave de tu cuenta Groq Free (entrada oculta, solo en memoria): ").strip()
    if not key:
        raise ValueError("No se introdujo una clave.")
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description="Relay: mantenimiento de repositorios Python.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Comprueba la configuración local sin APIs.")
    check.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    run = sub.add_parser("run", help="Inspecciona, prepara cambios y prueba una copia del repositorio.")
    run.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    run.add_argument("--task", help="Petición concreta. El ejemplo incluye una petición predeterminada.")
    run.add_argument("--output", type=Path)
    run.add_argument("--mode", choices=("activity", "always_strong", "verified"), default="activity")
    run.add_argument("--inspection-mode", choices=("model", "direct"), default="model")
    run.add_argument("--output-mode", choices=("strict", "prompt_json"), default="strict")
    apply = sub.add_parser("apply", help="Aplica una propuesta revisada cuyos tests pasan.")
    apply.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "apply":
            directory = args.run_dir.resolve()
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            if report.get("kind") != "relay_maintenance" or report.get("status") != "ready_for_review":
                raise ValueError("La ejecución no contiene una propuesta aprobada por las pruebas y pendiente de revisión.")
            workspace = Workspace.reopen(directory)
            result = workspace.apply_to_original()
            _save(directory / "application.json", result)
            report.update(status="applied", original_modified=True,
                          message="Propuesta aplicada después de comprobar los archivos originales.")
            _save(directory / "report.json", report)
            _render_report(directory, report)
            print(f"Cambio aplicado. Registro: {directory / 'application.json'}")
            return 0
        config, root = load_config(args.config)
        if args.command == "check":
            print(f"Repositorio: {root}\nEditables: {len(config['source_files'])}; tests: {len(config['test_files'])}.")
            print(f"Inspección: {config['models']['inspection']}\nEdición y reparación: {config['models']['editing']}")
            print("Clave disponible en el proceso." if os.environ.get("GROQ_API_KEY") else
                  "La clave se solicitará de forma oculta al ejecutar; no está en este proceso.")
            print("Configuración válida. No se hicieron llamadas a modelos.")
            return 0
        limits = config["limits"]
        task = args.task
        if task is None:
            if args.config.resolve() != DEFAULT_CONFIG.resolve():
                raise ValueError("Declara --task para trabajar en un repositorio propio.")
            task = DEMO_TASK
        directory = (args.output or Path.cwd() / "outputs" / "maintenance" /
                     datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
        if directory.exists():
            raise ValueError("La carpeta de salida ya existe; usa una carpeta nueva.")
        print(f"Repositorio: {root}\nSalida: {directory}")
        print("Se enviarán únicamente los archivos Python declarados a tu cuenta Groq Free.")
        print("Las pruebas ejecutan código del proyecto con tus permisos. Usa repositorios de confianza.")
        provider = GroqMaintenanceProvider(_key(), max_output_tokens=limits["max_output_tokens"],
            output_mode=args.output_mode,
            timeout=limits["request_timeout_seconds"], max_requests=limits["max_requests"],
            request_budget=limits["request_budget"], minimum_interval_seconds=limits["minimum_interval_seconds"],
            on_wait=lambda seconds: print(f"Esperando {seconds:.0f} s para respetar la cuota del modelo…", flush=True))
        router = ActivityRouter(config["models"]["inspection"], config["models"]["editing"], mode=args.mode)
        report = run_maintenance(root=root, source_files=config["source_files"], test_files=config["test_files"],
            task=task, run_dir=directory, provider=provider, router=router,
            timeout_seconds=limits["test_timeout_seconds"], max_repairs=limits["max_repairs"],
            progress=lambda message: print(message, flush=True), configuration=config, inspection_mode=args.inspection_mode)
        print(report["message"])
        print(f"Informe: {directory / 'report.md'}")
        if (directory / "changes.diff").is_file():
            print(f"Diff: {directory / 'changes.diff'}")
        if report["status"] == "ready_for_review":
            print(f'Para aplicar después de revisar: python tools/relay_maintain.py apply "{directory}"')
            return 0
        return 2
    except (ValueError, OSError, KeyError) as exc:
        print(f"Relay: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
