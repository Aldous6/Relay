"""Ask for the Groq key locally, without writing it, and run a small strong control."""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_agent_api_v8 import main as run_api


def configuration(free_confirmed=False):
    config = json.loads((ROOT / "configs/agent_api_v8.groq_free.example.json").read_text(encoding="utf-8"))
    config["run_id"] = "groq-strong-local-diagnostic"
    config["per_family"] = 1
    config["generation_options"]["max_output_tokens"] = 2048
    config["generation_options"]["action_format"] = "native_tools"
    config["policies"], config["primary_policy"] = ["always_strong"], "always_strong"
    config["console_progress"] = True
    config["pacing"] = {"min_interval_seconds": 2.1, "max_request_budget": 8000,
                        "tokens_per_minute": 8000}
    config["account_limits_provenance"] = {
        "source": "user-supplied Groq account limits, 2026-11-12",
        "requests_per_minute": 30, "requests_per_day": 1000,
        "tokens_per_minute": 8000, "tokens_per_day": 200000,
        "scope": "limits for each declared GPT-OSS model; other account usage is not tracked"}
    config["spending"]["account_is_free_tier"] = free_confirmed
    config["spending"]["account_tier_source"] = (
        "User explicitly confirmed Free in Groq Console Billing at " + datetime.now(timezone.utc).isoformat()
        if free_confirmed else None)
    config["scope"] = "Three synthetic known-family development diagnostics, no independent quality or ROI claim"
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--controls", action="store_true", help="Compare strong, weak and reactive cascade on the three development tasks.")
    parser.add_argument("--diagnose", action="store_true", help="One connection diagnostic, no benchmark or retries.")
    parser.add_argument("--diagnose-last-failure", action="store_true", help="Diagnose one failed request from the last completed run.")
    args = parser.parse_args(argv)
    if args.diagnose_last_failure:
        args.diagnose = True
    if args.prepare_only and args.diagnose:
        raise ValueError("Elige preparar o diagnosticar.")
    if args.controls and args.diagnose:
        raise ValueError("Elige controles o diagnóstico de una llamada.")
    if not args.prepare_only and not sys.stdin.isatty():
        raise RuntimeError("Ejecuta este comando en tu terminal local para introducir la clave oculta.")
    confirmed = False
    if not args.prepare_only:
        print("Comprueba Groq Console > Billing. La tabla de cuotas no confirma el plan.")
        confirmed = input("Si tu cuenta dice Free, escribe FREE para continuar: ").strip() == "FREE"
        if not confirmed:
            raise RuntimeError("No se ejecutó: falta confirmar el plan gratuito.")
    config = configuration(confirmed)
    if args.controls:
        config["run_id"] = "groq-local-three-controls-development"
        config["policies"] = ["always_strong", "always_weak", "cheap_then_escalate"]
        config["primary_policy"] = "cheap_then_escalate"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "outputs" / "groq-local" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    arguments = ["--config", str(path), "--output", str(directory / "run")]
    if args.prepare_only:
        return run_api(arguments)
    key = getpass.getpass("Pega tu Groq API key (entrada oculta, no se guarda): ").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("Clave vacía o con espacios; no se llamó al proveedor.")
    previous = os.environ.get("GROQ_API_KEY")
    try:
        os.environ["GROQ_API_KEY"] = key
        del key
        if args.diagnose:
            if args.diagnose_last_failure:
                return diagnose(config, directory, replay_last_failure=True)
            return diagnose(config, directory)
        print("Iniciando tres tareas con " + ("los tres controles." if args.controls else "GPT-OSS 120B.")
              + " Las esperas por cuota se calculan con el consumo registrado.")
        print("Los resultados quedan en:", directory / "run")
        return run_api(arguments + ["--execute"])
    finally:
        if previous is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous


def diagnose(config, directory, replay_last_failure=False):
    from relay_router.agent import AgentRequest, AgentStateV8, parse_action
    from relay_router.errors import LiveBackendError
    from routerbench.agent_litellm import LiteLLMEpisodeGenerator
    generator = LiteLLMEpisodeGenerator(config["models_metadata"], config["generation_options"], capture_error_reason=True)
    model = config["models"]["strong"]
    request = AgentRequest('Responde con una acción final cuyo answer sea ok.', (), AgentStateV8(0, 0, (), (), 1))
    replay = None
    if replay_last_failure:
        from report_agent_benchmark import audit_run
        candidates = sorted((ROOT / "outputs/groq-local").glob("*/run/summary.json"), reverse=True)
        for candidate in candidates:
            manifest, journal, _, _ = audit_run(candidate.parent)
            failures = [record for record in journal["records"] if record["kind"] == "call_completed"
                        and record["data"].get("backend_failure", {}).get("status_code") == 400]
            if not failures:
                continue
            record = failures[0]
            if record["data"]["model"] != model:
                raise ValueError("El modelo de la solicitud guardada difiere del catálogo actual.")
            payload = record["data"]["request"]
            state = dict(payload["state"])
            if state.pop("context_length_unit", None) != "utf8_bytes_prompt_and_history":
                raise ValueError("Unidad de contexto guardada inesperada.")
            for name in ("previous_errors", "available_tools", "infrastructure_errors", "failed_tiers"):
                state[name] = tuple(state[name])
            request = AgentRequest(payload["prompt"], tuple(payload["history"]), AgentStateV8(**state))
            replay = {"run": str(candidate.parent), "freeze_sha256": manifest["freeze_sha256"],
                      "episode_id": record["episode_id"], "call_index": record["data"]["call_index"],
                      "scope": "diagnostic only; never added to the original experiment"}
            break
        if replay is None:
            raise ValueError("No hay una solicitud HTTP 400 guardada en una corrida completada.")
    result = {"scope": "one connection diagnostic, no benchmark, no retries", "model": model,
              "sdk_version": generator.manifest()["litellm_version"],
              "diagnostic_time_utc": datetime.now(timezone.utc).isoformat()}
    if replay is not None:
        result["replayed_request"] = replay
    print("Comprobando una sola llamada; no se ejecutará la batería de tareas.", flush=True)
    try:
        reply = generator(model, request)
        result.update(outcome="response_received", usage=reply.usage)
        try:
            parse_action(reply.content, request.state.available_tools)
            result["action_parse_valid"] = True
        except (ValueError, TypeError):
            result["action_parse_valid"] = False
    except LiveBackendError as error:
        result.update(outcome="failed", failure=error.details())
        reason = getattr(error, 'operator_reason', None)
        if reason is not None:
            result['provider_message_redacted'] = reason
    except Exception as error:
        result.update(outcome="configuration_failure", exception_type=type(error).__name__)
    path = directory / "diagnostic.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("Diagnóstico guardado en:", path, flush=True)
    return result


if __name__ == "__main__":
    main()
