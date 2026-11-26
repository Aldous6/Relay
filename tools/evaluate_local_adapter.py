"""Freeze locally adapted routers before a new synthetic local evaluation.

Both Qwen responses are collected once per case and shared by all policies.
This is a new-instance pilot within eight known templates, not production proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_transfer_pilot import SYSTEM, cases
from routerbench.data import sha256_file
from routerbench.features import normalize_for_dedup
from routerbench.live import _request_json, _usage, discover_ollama
from routerbench.prospective import _hash, _read_cases, _write, prepare_prospective, run_prospective

MODELS = {"strong": "qwen2.5:1.5b", "weak": "qwen2.5:0.5b"}
OPTIONS = {"num_ctx": 8192, "num_predict": 96, "temperature": 0, "seed": 42}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def persist_identical(path, value):
    if path.exists():
        if load(path) != value:
            raise ValueError(f"La evidencia congelada cambió: {path.name}.")
    else:
        _write(path, value)


def route_selected(model, specification, frame):
    if specification["mode"] == "fixed_strong_fallback":
        return "strong"
    if specification["mode"] != "pairwise" or specification["cost_unit"] != "seconds":
        raise ValueError("Política local no reconocida.")
    from routerbench.pairwise import choose_pairwise

    probabilities, durations = model.predict_tables(frame)
    chosen = choose_pairwise(probabilities, durations, model.strong, model.weak,
                             specification["tradeoff"]).iloc[0]["selected_model"]
    if chosen not in MODELS.values():
        raise ValueError("El adaptador eligió un modelo ajeno al catálogo.")
    return "strong" if chosen == MODELS["strong"] else "weak"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    import pandas as pd

    output = ROOT / "reports/local-adapter-evaluation-v1"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reports/local-adapter-development-v1"
    development = load(source / "summary.json")
    candidate = development
    control = development["controls"]["bert_adapted"]
    for policy in (candidate, control):
        if sha256_file(source / policy["artifact"]) != policy["artifact_sha256"]:
            raise ValueError("El artefacto local cambió.")
        specification = policy["selected_specification"]
        if {alias: specification[alias] for alias in MODELS} != MODELS:
            raise ValueError("El adaptador no corresponde a este catálogo.")
        if specification["system_sha256"] != hashlib.sha256(SYSTEM.encode()).hexdigest():
            raise ValueError("La instrucción no corresponde al entrenamiento.")
    available = {row["model"]: row for row in discover_ollama(include_context=True)}
    for name in MODELS.values():
        if available.get(name, {}).get("context_length", 0) < OPTIONS["num_ctx"]:
            raise ValueError("Los dos Qwen instalados requieren contexto suficiente; no se sustituyen.")
    # Archive the actual source before the fresh seed is sampled. Existing
    # archives are verified byte-for-byte against the current source on resume.
    paths = sorted([*Path(ROOT / "src/routerbench").glob("*.py"),
                    Path(__file__).resolve(), ROOT / "tools/local_transfer_pilot.py",
                    ROOT / "tools/train_local_adapter.py", ROOT / "configs/local_adapter_v1.json"])
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    bundle = output / "source_bundle.zip"
    if not bundle.exists():
        with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        if set(archive.namelist()) != set(source_hashes) or any(
                hashlib.sha256(archive.read(name)).hexdigest() != digest for name, digest in source_hashes.items()):
            raise ValueError("El código difiere del snapshot de ejecución; conservar la evidencia anterior.")
    immutable = {
        "scope": "fresh instances of eight known synthetic templates after local development; not new task families or production validation",
        "adapter_source_summary_sha256": sha256_file(source / "summary.json"),
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "control_artifact_sha256": control["artifact_sha256"],
        "candidate_specification": candidate["selected_specification"],
        "control_specification": control["selected_specification"],
        "candidate_role": "predeclared raw fourteen structural attributes; no architecture selection on this cohort",
        "reference_role": "locally adapted BERT CLS features, same development rows and local head; not published RouteLLM",
        "published_bert_shadow_threshold": .5,
        "bert_selection_sha256": development["bert_reference_selection_sha256"],
        "bert_weights_sha256": development["bert_weights_sha256"],
        "models": {alias: {"name": name, "digest": available[name]["digest"]} for alias, name in MODELS.items()},
        "system": SYSTEM, "generation_options": OPTIONS,
        "cases": 64, "templates": 8, "synthetic": True,
        "generation_sampling": "one cached response per case/model; both potential outcomes shared by all policies",
        "metric": "normalized exact response; generation errors count as incorrect",
        "cost_unit": "measured generation callback seconds; not money, tokens or end-to-end service latency",
        "monetary_cost_usd": None, "paid_api_calls": False,
        "automatic_training_or_promotion": False,
        "native_cpu_threads": 4,
        "source_files_sha256": source_hashes, "source_bundle_sha256": sha256_file(bundle),
        "previous_cohort_sha256": sha256_file(ROOT / "reports/local-transfer-pilot-v1/cohort.jsonl")}
    lock_path = output / "policy_lock.json"
    if lock_path.exists():
        lock = load(lock_path)
        if {key: value for key, value in lock.items() if key != "cohort_seed"} != immutable:
            raise ValueError("La política congelada cambió; no se regenera la cohorte.")
    else:
        lock = {**immutable, "cohort_seed": secrets.randbits(64)}
        _write(lock_path, lock)
    dataset = output / "cohort.jsonl"
    expected = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in cases(lock["cohort_seed"]))
    if dataset.exists():
        if dataset.read_text(encoding="utf-8") != expected:
            raise ValueError("La cohorte congelada cambió.")
    else:
        with dataset.open("x", encoding="utf-8") as stream:
            stream.write(expected)
    previous = [json.loads(line) for line in (ROOT / "reports/local-transfer-pilot-v1/cohort.jsonl").read_text(encoding="utf-8").splitlines()]
    known = set(pd.read_parquet(ROOT / "data/processed/rows.parquet")["prompt_group"])
    known.update(hashlib.sha256(normalize_for_dedup(row["prompt"]).encode()).hexdigest() for row in previous)
    freeze = prepare_prospective(dataset, output / "evaluation", lock, known, True)
    if not args.execute:
        print(json.dumps({"prepared": str(dataset), "fresh_cases": freeze["n_cases"], "generated": False}))
        return
    # Callback models are lazy-loaded so a completed replay needs no runtime.
    import joblib
    from routerbench.demo import conservative_context_bound

    rows, _, dataset_hash = _read_cases(dataset, known)
    ids = {row["prompt_sha256"]: row["id"] for row in rows}
    shadow_path = output / "published_bert_routes.json"
    shadow_header = {"policy_lock_sha256": sha256_file(lock_path), "dataset_sha256": dataset_hash, "threshold": .5}
    shadow = load(shadow_path) if shadow_path.exists() else {**shadow_header, "routes": [], "payload_sha256": _hash([])}
    if {key: shadow[key] for key in shadow_header} != shadow_header or shadow["payload_sha256"] != _hash(shadow["routes"]):
        raise ValueError("Las rutas shadow cambiaron.")
    shadow_by_hash = {row["prompt_sha256"]: row for row in shadow["routes"]}
    models = {}
    timings = []

    def frame(prompt):
        return pd.DataFrame({"prompt_text": [f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}"]})

    def runtime(key, specification):
        if key not in models:
            model = joblib.load(source / specification["artifact"])
            # Neutral warmup is outside the fresh cohort and all timers.
            for text in ("¿Cuál es el dato del material recibido?", "Material: registro A = 7. Consulta el registro A."):
                model.predict_tables(frame(text))
            models[key] = model
        return models[key]

    def route_candidate(prompt):
        model = runtime("candidate", candidate)
        tick = time.perf_counter()
        action = route_selected(model, candidate["selected_specification"], frame(prompt))
        timings.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "candidate_seconds": time.perf_counter()-tick})
        return action

    def route_control(prompt):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_hash in shadow_by_hash:
            return shadow_by_hash[prompt_hash]["adapted_route"]
        model = runtime("control", control)
        tick = time.perf_counter()
        action = route_selected(model, control["selected_specification"], frame(prompt))
        elapsed = time.perf_counter()-tick
        if control["selected_specification"]["mode"] == "fixed_strong_fallback":
            # Shadow telemetry must not change the adapted fallback decision.
            model.predict_tables(frame(prompt))
        encoder = model.transformer_.encoder_
        probability = float(encoder.last_preference_probabilities_[0])
        record = {"id": ids[prompt_hash], "prompt_sha256": prompt_hash,
                  "model": "strong" if probability >= .5 else "weak", "probability": probability,
                  "adapted_route": action, "adapted_callback_seconds": elapsed,
                  "native_bert_seconds": encoder.last_native_seconds_}
        shadow["routes"].append(record)
        shadow_by_hash[prompt_hash] = record
        shadow["payload_sha256"] = _hash(shadow["routes"])
        _write(shadow_path, shadow)
        return action

    count = 0

    def generate(alias, prompt):
        nonlocal count
        if len(shadow["routes"]) != len(rows):
            raise ValueError("Todas las rutas deben guardarse antes de generar respuestas.")
        before = output / "all_routes_before_generation.json"
        if not before.exists():
            persist_identical(before, {"dataset_sha256": dataset_hash, "n_cases": len(rows),
                "published_bert_routes_sha256": sha256_file(shadow_path),
                "evaluation_routes_sha256": sha256_file(output / "evaluation/routes.json"),
                "statement": "written in the first generation callback before its first Ollama request"})
            _write(output / "route_timings.json", {"scope": "resident sequential CPU routing before any cohort generation; excludes model load and neutral warmup; no production SLA",
                "candidate": timings, "bert": "adapted_callback_seconds and native_bert_seconds in published_bert_routes.json"})
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        if conservative_context_bound(messages)+OPTIONS["num_predict"] > OPTIONS["num_ctx"]:
            raise ValueError("El prompt no cabe en el presupuesto conservador; no se recorta.")
        current = {row["model"]: row for row in discover_ollama()}
        name = MODELS[alias]
        if current.get(name, {}).get("digest") != lock["models"][alias]["digest"]:
            raise ValueError("El digest cambió durante la ejecución.")
        tick = time.perf_counter()
        response = _request_json("http://127.0.0.1:11434/api/chat", {"model": name, "messages": messages,
            "stream": False, "options": OPTIONS}, timeout=120)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Respuesta sin contenido textual.")
        count += 1
        if count % 8 == 0:
            print(f"Respuestas nuevas: {count}/128", flush=True)
        return {"content": content, "usage": _usage(response.get("prompt_eval_count"), response.get("eval_count")),
                "duration_seconds": time.perf_counter()-tick}

    result = run_prospective(dataset, output / "evaluation", lock, route_candidate, route_control,
                             generate, known_prompt_groups=known, collect_both_models=True)
    print(json.dumps({"completed": True, "calls_this_invocation": count,
                      "result_sha256": result["result_sha256"], "policies": result["policies"]}, ensure_ascii=False))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        main()
