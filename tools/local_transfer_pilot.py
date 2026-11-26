"""Explicit prospective synthetic pilot with two installed local models only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SYSTEM = ("Responde preguntas sobre el material recibido. Devuelve solamente el dato o número solicitado, "
          "sin explicación ni formato adicional. Puedes calcular a partir de los datos. "
          "Si el dato no se puede obtener del material, responde exactamente NO_EVIDENCE.")
MODELS = {"strong": "llama3:latest", "weak": "qwen2.5:1.5b"}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def cases(seed):
    """Eight fixed templates, eight independently generated instances each."""
    rng = random.Random(seed)
    result = []
    for category in range(8):
        for index in range(8):
            nonce = f"P{rng.getrandbits(40):010X}"
            if category == 0:
                codes = [f"K{rng.getrandbits(24):06X}" for _ in range(24)]
                values = [rng.randint(100, 9999) for _ in codes]
                position = (0, 12, 23)[index % 3]
                text = "\n".join(f"Registro {code}: saldo {value}." for code, value in zip(codes, values))
                question = f"¿Cuál es el saldo del registro {codes[position]}?"
                answer = str(values[position])
            elif category == 1:
                quantities, prices = [rng.randint(2, 13) for _ in range(4)], [rng.randint(7, 37) for _ in range(4)]
                text = "\n".join(f"Lote {i}: {q} unidades a {p} pesos cada una." for i, (q, p) in enumerate(zip(quantities, prices)))
                question, answer = "¿Cuántos pesos cuestan los cuatro lotes juntos?", str(sum(q*p for q, p in zip(quantities, prices)))
            elif category == 2:
                entries = [(i, rng.choice(["activo", "inactivo"]), rng.choice(["norte", "sur"]), rng.randint(10, 89)) for i in range(12)]
                text = "\n".join(f"Cuenta {i}: estado {s}, región {r}, saldo {v}." for i, s, r, v in entries)
                question = "¿Cuál es la suma de saldos de las cuentas con estado activo y región norte?"
                answer = str(sum(v for _, s, r, v in entries if s == "activo" and r == "norte"))
            elif category == 3:
                updates = [(day, rng.randint(100, 999)) for day in [2, 7, 14, 22]]
                cutoff = rng.choice([8, 15, 23])
                answer = str(max((d, v) for d, v in updates if d <= cutoff)[1])
                rng.shuffle(updates)
                text = "\n".join(f"{d:02d}/03/2025: el cupo aprobado pasa a {v}." for d, v in updates)
                question = f"¿Cuál era el cupo vigente al terminar el {cutoff:02d}/03/2025? Usa la actualización más reciente hasta esa fecha."
            elif category == 4:
                owner, room = f"E{rng.getrandbits(24):06X}", rng.randint(100, 999)
                text = f"El proyecto {nonce} está a cargo de {owner}.\n{owner} tiene asignada la sala {room}.\nEl proyecto OTRO está a cargo de OTRA_PERSONA, que usa la sala {room+1}."
                question, answer = f"¿Qué sala usa la persona a cargo de {nonce}?", str(room)
            elif category == 5:
                old, new = rng.randint(100, 999), rng.randint(1000, 1999)
                text = f"Acta del 02/04/2025: presupuesto de {nonce} = {old}.\nActa del 17/04/2025: presupuesto de {nonce} = {new}.\nRegla: cuando dos actas discrepan, prevalece la de fecha más reciente."
                question, answer = f"¿Cuál es el presupuesto vigente de {nonce}?", str(new)
            elif category == 6:
                text = f"El proyecto {nonce} tiene 7 integrantes y trabaja en la sede CENTRO. No se proporciona un presupuesto."
                question, answer = f"¿Cuál es el presupuesto de {nonce}?", "NO_EVIDENCE"
            else:
                first, revised = rng.randint(10, 99), rng.randint(100, 999)
                text = f"Historial:\nUsuario: Mi clave temporal es {first}.\nAsistente: Entendido.\nUsuario: Corrijo la clave: ahora es {revised}.\nAsistente: Actualizada.\nUsuario: Usa siempre mi última corrección."
                question, answer = "¿Cuál es la clave temporal que debo usar ahora?", str(revised)
            prompt = f"Caso sintético {nonce}.\nMATERIAL:\n{text}\n\nPREGUNTA:\n{question}\nDevuelve únicamente la respuesta solicitada."
            result.append({"id": f"task{category}_{index:02d}", "prompt": prompt, "reference": answer, "grade": "exact"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Generate real local responses after freezing all routes.")
    args = parser.parse_args()
    from routerbench.data import sha256_file
    from routerbench.live import discover_ollama

    directory = ROOT / "reports/local-transfer-pilot-v1"
    directory.mkdir(parents=True, exist_ok=True)
    reference_selection_path = ROOT / "reports/routellm-bert-secondary-v1/selection.json"
    reference_selection = json.loads(reference_selection_path.read_text(encoding="utf-8"))
    source_dir = ROOT / "reports/relay-v3-development-v1"
    source = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    selected = source["selected_for_future_evaluation"]
    if selected["candidate_id"] != "pairwise_hybrid_a5" or selected["point"]["parameter"] != 20:
        raise ValueError("El piloto conserva el punto v3 predeclarado; no se elige con datos locales nuevos.")
    artifact = source_dir / selected["artifact"]
    if sha256_file(artifact) != selected["artifact_sha256"]:
        raise ValueError("El modelo Relay cambió.")
    available = {item["model"]: item for item in discover_ollama(include_context=True)}
    for model in MODELS.values():
        if model not in available or available[model].get("context_length", 0) < 8192:
            raise ValueError(f"Falta {model} con contexto suficiente; no se descarga ni sustituye.")
    lock_path = directory / "policy_lock.json"
    immutable = {"scope": "controlled synthetic local transfer pilot, not original RouterBench comparison or production validation",
        "router_artifact_sha256": selected["artifact_sha256"], "router_source_summary_sha256": sha256_file(source_dir / "summary.json"),
        "relay_tradeoff": 20, "bert_threshold": .5, "reference_selection_sha256": sha256_file(reference_selection_path),
        "bert_weights_sha256": reference_selection["configuration"]["weights_sha256"],
        "models": {alias: {"name": model, "digest": available[model]["digest"]} for alias, model in MODELS.items()},
        "alias_note": "strong/weak indicate inherited router actions and model size, not verified local quality ordering",
        "system": SYSTEM, "generation_options": {"num_ctx": 8192, "num_predict": 96, "temperature": 0, "seed": 42},
        "cases": 64, "categories": 8, "synthetic": True, "monetary_cost_usd": None,
        "quality_metric": "strict normalized exact response on controlled synthetic tasks; not broad answer quality",
        "generation_sampling": "one cached response per case/model, shared across routing policies; no stochastic variance estimate",
        "api_sources": ["https://docs.ollama.com/api/chat", "https://docs.ollama.com/modelfile"],
        "automatic_training_or_promotion": False, "paid_api_calls": False,
        "generator_source_sha256": sha256_file(Path(__file__).resolve())}
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if {k: v for k, v in lock.items() if k != "cohort_seed"} != immutable:
            raise ValueError("La política, modelo, generación o código cambió; se conserva la ejecución anterior.")
    else:
        lock = {**immutable, "cohort_seed": secrets.randbits(64)}
        write_new(lock_path, lock)  # Commit the policy before generating cases.
    dataset_path = directory / "cohort.jsonl"
    expected = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in cases(lock["cohort_seed"]))
    if dataset_path.exists():
        if dataset_path.read_text(encoding="utf-8") != expected:
            raise ValueError("La cohorte cambió; no se regenera para mejorar resultados.")
    else:
        with dataset_path.open("x", encoding="utf-8") as stream:
            stream.write(expected)
    if not args.execute:
        print(json.dumps({"cohort": str(dataset_path), "policy_lock": str(lock_path), "generation_executed": False}))
        return

    import joblib
    import pandas as pd
    from routerbench.demo import conservative_context_bound
    from routerbench.live import _request_json, _usage
    from routerbench.pairwise import choose_pairwise
    from routerbench.prospective import run_prospective
    from routerbench.routellm_reference import LocalBERTReference

    model = joblib.load(artifact)
    bert = LocalBERTReference(ROOT / reference_selection["configuration"]["model_dir"], reference_selection["configuration"])
    known = set(pd.read_parquet(ROOT / "data/processed/rows.parquet")["prompt_group"])
    timings = {"relay": [], "bert": []}

    def router_text(prompt):
        return f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}"

    def route_relay(prompt):
        tick = time.perf_counter()
        p, c = model.predict_tables(pd.DataFrame({"prompt_text": [router_text(prompt)]}))
        choice = choose_pairwise(p, c, model.strong, model.weak, 20).iloc[0]["selected_model"]
        timings["relay"].append(time.perf_counter()-tick)
        return "strong" if choice == model.strong else "weak"

    def route_bert(prompt):
        tick = time.perf_counter()
        value = bert.predict_batch([router_text(prompt)])[0]
        timings["bert"].append(time.perf_counter()-tick)
        return "strong" if value["probability"] >= .5 else "weak"

    count = 0

    def generate(alias, prompt):
        nonlocal count
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        if conservative_context_bound(messages)+96 > 8192:
            raise ValueError("El prompt no cabe según el presupuesto conservador; no se recorta.")
        current = {item["model"]: item for item in discover_ollama()}
        name = MODELS[alias]
        if current.get(name, {}).get("digest") != lock["models"][alias]["digest"]:
            raise ValueError("El digest local cambió durante la ejecución.")
        tick = time.perf_counter()
        response = _request_json("http://127.0.0.1:11434/api/chat", {"model": name, "messages": messages,
            "stream": False, "options": lock["generation_options"]}, timeout=120)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Respuesta sin contenido textual.")
        count += 1
        print(f"Respuestas nuevas en esta ejecución: {count} · {name}", flush=True)
        return {"content": content, "usage": _usage(response.get("prompt_eval_count"), response.get("eval_count")),
                "duration_seconds": time.perf_counter()-tick, "cost_usd": None, "finish_reason": response.get("done_reason")}

    evaluation = run_prospective(dataset_path, directory / "evaluation", lock, route_relay, route_bert, generate,
        known_prompt_groups=known, collect_both_models=True)
    timing_path = directory / "route_timings.json"
    if not timing_path.exists():
        write_new(timing_path, {"scope": "routing callback wall times; includes extra BERT coverage audit; not a matched latency benchmark",
            "seconds": timings, "resumed_routes_may_be_absent_from_this_invocation": True})
    print(json.dumps(evaluation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
