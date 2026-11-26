"""Run independent text requests with Relay's experimental per-prompt selector.

The local BERT scorer stays loaded between requests. Groq generation happens
only when explicitly running this client without --route-only. The existing
pilot profile is for GPT-OSS 20B/120B; it is not a profile for arbitrary models.
"""
import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router import Relay
from relay_router.integrations.routellm import load_groq_pilot_selector

MODELS = {"groq/openai/gpt-oss-20b": (.075, .30), "groq/openai/gpt-oss-120b": (.15, .60)}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def groq_generate(model, *, prompt, api_key):
    if model not in MODELS:
        raise ValueError("Modelo fuera del par configurado.")
    body = {"model": model.removeprefix("groq/"), "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_completion_tokens": 1024, "reasoning_effort": "low", "stream": False}
    request = Request("https://api.groq.com/openai/v1/chat/completions",
                      data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                      headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=90) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Respuesta demasiado grande.")
        value = json.loads(raw)
    except HTTPError as error:
        raise RuntimeError(f"Groq rechazó la solicitud (HTTP {error.code}); no se reintentó.") from None
    except (URLError, OSError):
        raise RuntimeError("La conexión con Groq no terminó; no se reintentó.") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("Groq devolvió una respuesta inválida.") from None
    try:
        if value.get("model") != model.removeprefix("groq/"):
            raise ValueError
        content = value["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError
    except (AttributeError, KeyError, TypeError, IndexError, ValueError):
        raise RuntimeError("Groq no devolvió una respuesta de texto visible.") from None
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
    incoming = incoming if type(incoming) is int and incoming >= 0 else None
    outgoing = outgoing if type(outgoing) is int and outgoing >= 0 else None
    rates = MODELS[model]
    cost = (incoming * rates[0] + outgoing * rates[1]) / 1e6 if incoming is not None and outgoing is not None else None
    return {"content": content, "measurement": {"input_tokens": incoming, "output_tokens": outgoing,
            "cost_usd": cost, "cost_basis": "tariff_estimate" if cost is not None else "unknown"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, help="Consulta UTF-8; si se omite, solicita texto en la terminal.")
    parser.add_argument("--route-only", action="store_true", help="Seleccionar sin llamar a Groq ni solicitar clave.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/routellm-bert")
    parser.add_argument("--log", type=Path, help="Ledger opcional de metadatos; nunca guarda consultas ni respuestas.")
    args = parser.parse_args(argv)
    initial = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None
    print("Cargando selector local una sola vez...", flush=True)
    selector, provenance = load_groq_pilot_selector(args.checkpoint)
    relay = Relay(selector.strong_model, prompt_selector=selector, mode="apply", allow_experimental=True,
                  log_path=args.log)
    print(f"Selector listo ({provenance['load_seconds']:.1f} s de carga). Perfil experimental GPT-OSS 20B/120B.")
    print("Cada consulta es independiente. Escribe :salir para terminar.")
    api_key = None
    try:
        while True:
            try:
                prompt = initial if initial is not None else input("Consulta: ")
            except EOFError:
                break
            if prompt.strip() == ":salir":
                break
            if not prompt.strip():
                if initial is not None:
                    raise ValueError("El archivo no contiene una consulta.")
                continue
            task = relay.task("text-request", routing_text=prompt, eligible_models=MODELS, max_calls=1)
            print(f"Modelo: {task.model}; selección: {task.decision['routing_seconds']:.3f} s.")
            if args.route_only:
                task.finish(accepted=None)
            else:
                if api_key is None:
                    api_key = os.environ.get("GROQ_API_KEY") or getpass("Clave Groq (oculta, en memoria): ")
                if not api_key.strip():
                    task.finish(accepted=None)
                    raise ValueError("Se requiere una clave Groq para generar la respuesta.")
                try:
                    response = task.call(groq_generate, prompt=prompt, api_key=api_key,
                                         measure=lambda value: value["measurement"])
                except Exception:
                    task.finish(accepted=False, cost_complete=False)
                    raise
                observation = task.finish(accepted=None, cost_complete=True)
                print(response["content"])
                cost = observation["cost_usd"]
                print("Costo de llamada desconocido." if cost is None else
                      f"Costo de llamada estimado sin caché: USD {cost:.8f}; excluye cómputo del router.")
            if initial is not None:
                break
    finally:
        api_key = None
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        # Avoid printing arbitrary provider payloads or local file contents.
        print(str(error) if isinstance(error, (ValueError, RuntimeError)) else "No se pudo leer el archivo solicitado.", file=sys.stderr)
        raise SystemExit(1)
