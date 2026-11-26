"""Compare resident CPU routing latency on the same development prompts.

This is an explicit offline benchmark, never a model-quality evaluation. Run
after training has stopped, using the pinned .venv-routellm environment. It does
not generate LLM responses, use paid APIs, or load the consumed outer test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Se requiere un objeto JSON: {path}")
    return value


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or _hash(path) != expected:
        raise ValueError(f"Fuente fuera del proyecto o SHA-256 distinto: {relative}")
    return path


def _statistics(samples: list[float]) -> dict:
    import numpy as np

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Se requieren al menos dos tiempos finitos y no negativos.")
    return {"n": len(values), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, .95)),
            "min_seconds": float(values.min()), "max_seconds": float(values.max()),
            "total_seconds": float(values.sum())}


def benchmark(root: Path, run_dir: Path, output: Path) -> dict:
    root, run_dir, output = root.resolve(), run_dir.resolve(), output.resolve()
    if not run_dir.is_relative_to(root / "reports") or output.parent != run_dir:
        raise ValueError("La ejecución y la salida deben estar en el mismo directorio de informes del proyecto.")
    if output.exists():
        raise FileExistsError("La medición existente se conserva; elige una salida nueva para repetirla explícitamente.")

    import joblib
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    from routerbench.development import load_development_inputs
    from routerbench.improvement import _decisions, activate_full_fit
    from routerbench.routellm_reference import LocalBERTReference, reference_probabilities

    summary_path = run_dir / "summary.json"
    summary_hash = _hash(summary_path)
    summary = _json(summary_path)
    config = summary["configuration"]
    configuration_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if configuration_hash != summary["configuration_sha256"] or config["outer_test_evaluation_allowed"] is not False:
        raise ValueError("La configuración no corresponde a un experimento exclusivo de desarrollo.")
    if summary["outer_test_accessed_for_v3"] is not False or summary["confirmation_of_superiority"] is not False:
        raise ValueError("La medición necesita el informe exploratorio original, sin evaluación de test nueva.")
    selected = summary["selected_for_future_evaluation"]
    if selected.get("artifact") is None:
        raise ValueError("No hay un candidato entrenado seleccionado para medir.")
    candidate = summary["candidates"][selected["candidate_id"]]
    if (candidate["artifact"] != selected["artifact"] or candidate["artifact_sha256"] != selected["artifact_sha256"]
            or candidate["selected_point"] != selected["point"]):
        raise ValueError("El candidato y el punto seleccionados no coinciden con el manifiesto.")
    artifact_name = selected["artifact"]
    if Path(artifact_name).name != artifact_name or summary["files_sha256"].get(artifact_name) != selected["artifact_sha256"]:
        raise ValueError("El artefacto seleccionado no tiene una ruta y huella verificables.")
    artifact_path = _verified(root, str((run_dir / artifact_name).relative_to(root)), selected["artifact_sha256"])
    reference_selection_path = _verified(root, config["reference"]["selection_file"], config["reference"]["selection_sha256"])
    reference_selection = _json(reference_selection_path)
    if reference_selection["test_outcomes_used_for_selection"] is not False:
        raise ValueError("La referencia no conservó selección exclusiva con desarrollo.")
    # Use the pinned configuration inside the already hashed reference selection,
    # rather than a potentially revised configuration file elsewhere.
    reference_config = reference_selection["configuration"]
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    if pair != [reference_config["common_pair"]["strong"], reference_config["common_pair"]["weak"]]:
        raise ValueError("El benchmark debe conservar el mismo par histórico.")
    reference_parameter = summary["routellm_bert"]["selected_point"]["parameter"]
    if reference_parameter not in config["reference"]["threshold_grid"]:
        raise ValueError("El punto de BERT no pertenece a la búsqueda declarada.")

    inputs, _, _, rows, manifest = load_development_inputs(root, config)
    rows = activate_full_fit(rows)
    if summary["source_prepared_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("La preparación difiere de la usada para elegir el candidato.")
    counts = rows["development_split"].value_counts().to_dict()
    if counts != config["expected_population"] or counts != summary["partition_counts"]:
        raise ValueError("Las particiones de desarrollo no corresponden al informe.")
    validation_ids = rows.index[rows["development_split"].eq("validation")]
    sample_size = config["latency_sample_size"]
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 2:
        raise ValueError("La muestra de latencia debe contener al menos dos solicitudes.")
    sample_ids = validation_ids[:min(sample_size, len(validation_ids))]
    if len(sample_ids) < 2:
        raise ValueError("No hay suficientes prompts de validación.")
    # Identical source order and sample size to the development runner. DataFrame
    # assembly and reading are outside timing for both resident backends.
    frames = [inputs.loc[[row_id], ["prompt_text"]] for row_id in sample_ids]
    texts = [frame.iloc[0]["prompt_text"] for frame in frames]
    prompt_keys = [{"source_row": int(row_id), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                   for row_id, text in zip(sample_ids, texts)]

    tick = time.perf_counter()
    relay = joblib.load(artifact_path)
    relay_load = time.perf_counter()-tick
    tick = time.perf_counter()
    bert = LocalBERTReference(root / reference_config["model_dir"], reference_config)
    bert_load = time.perf_counter()-tick
    specification = candidate["specification"]
    parameter = selected["point"]["parameter"]

    def relay_call(index: int):
        first, predicted_costs = relay.predict_tables(frames[index])
        routes = _decisions(specification, first, predicted_costs, pair, parameter)
        if len(routes) != 1 or routes.iloc[0] not in pair:
            raise ValueError("El candidato no devolvió una ruta válida del par común.")

    def bert_call(index: int):
        # Only the faithful native routing work is timed. predict_batch also
        # tokenizes the full prompt to audit coverage; that extra audit is
        # intentionally excluded from this inference comparison.
        encoded = bert.tokenizer([texts[index]], padding=True, truncation=True,
                                 max_length=bert.max_length, return_tensors="pt")
        with bert.torch.inference_mode():
            logits = bert.model(**encoded).logits.detach().cpu().numpy()
        probability = reference_probabilities(logits)[0]
        route = pair[0] if probability >= reference_parameter else pair[1]
        if route not in pair:
            raise ValueError("BERT no devolvió una ruta válida del par común.")

    calls = {"relay": relay_call, "routellm_bert": bert_call}
    samples = {name: [] for name in calls}
    per_prompt = []
    threads = reference_config["threads"]
    # Torch is set to the pinned CPU thread count by LocalBERTReference. Cap
    # native BLAS pools identically for both resident backends in this process.
    with threadpool_limits(limits=threads):
        pools = [{key: pool.get(key) for key in ("internal_api", "num_threads", "version")}
                 for pool in threadpool_info()]
        for index in range(2):
            order = ("relay", "routellm_bert") if index % 2 == 0 else ("routellm_bert", "relay")
            for name in order:
                calls[name](index)
        for index, row_id in enumerate(sample_ids):
            order = ("relay", "routellm_bert") if index % 2 == 0 else ("routellm_bert", "relay")
            record = {"source_row": int(row_id), "order": list(order), "input_characters": len(texts[index])}
            for name in order:
                tick = time.perf_counter()
                calls[name](index)
                elapsed = time.perf_counter()-tick
                samples[name].append(elapsed)
                record[name+"_seconds"] = elapsed
            per_prompt.append(record)
            if (index+1) % 10 == 0 or index+1 == len(sample_ids):
                print(f"CPU latency: {index+1}/{len(sample_ids)} prompts en ambos routers.", flush=True)

    if _hash(summary_path) != summary_hash or _hash(artifact_path) != selected["artifact_sha256"]:
        raise ValueError("El resumen o artefacto cambió durante la medición; no se publica.")
    relevant_sources = ["tools/benchmark_v3.py", "src/routerbench/improvement.py",
                        "src/routerbench/pairwise.py", "src/routerbench/learner.py",
                        "src/routerbench/representations.py", "src/routerbench/routellm_reference.py"]
    report = {
        "kind": "descriptive resident CPU latency on selected development prompts",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_sha256": summary_hash,
        "source_configuration_sha256": configuration_hash,
        "selected_candidate": selected["candidate_id"],
        "selected_parameter": parameter,
        "selected_artifact_sha256": selected["artifact_sha256"],
        "bert_threshold": reference_parameter,
        "bert_checkpoint": reference_config["checkpoint"],
        "bert_checkpoint_revision": reference_config["checkpoint_revision"],
        "bert_weights_sha256": reference_config["weights_sha256"],
        "reference_selection_sha256": config["reference"]["selection_sha256"],
        "reference_manifest_sha256": _hash(root / reference_config["model_dir"] / "manifest.json"),
        "sample": {"n": len(sample_ids), "selection": "first validation IDs in original source order, matching v3 runner",
                   "prompt_keys_sha256": hashlib.sha256(json.dumps(prompt_keys, sort_keys=True).encode()).hexdigest(),
                   "prompt_keys": prompt_keys,
                   "input_characters_min": min(map(len, texts)),
                   "input_characters_median": float(np.median(list(map(len, texts)))),
                   "input_characters_max": max(map(len, texts))},
        "warmups_per_backend": 2,
        "execution_order": "alternating backend order for each prompt; sequential in one process",
        "feature_or_prediction_cache_used": False,
        "latency": {name: _statistics(values) for name, values in samples.items()},
        "cold_initialization_seconds": {"relay_joblib_load": relay_load, "bert_reference_initialization": bert_load},
        "cold_initialization_scope": {"relay": "joblib deserialization after external artifact hash preflight",
                                      "bert": "pinned LocalBERTReference initialization, including its file-hash verification, optional-library imports, tokenizer and CPU model loading",
                                      "comparability": "different initialization boundaries and unreset OS file caches; do not compare these as controlled cold-start times"},
        "timed_work": {"relay": "full-text prompt representation, predictions, predicted costs and fixed pair policy",
                       "routellm_bert": "native tokenizer, truncated forward inference, official three-class aggregation and fixed threshold decision; no extra full-token coverage audit"},
        "bert_tokenization": {"max_length": bert.max_length, "truncation": True, "padding": True,
                              "note": "BERT uses its official 512-token limit; Relay processes full prompt text. Equal source text does not imply equal retained context."},
        "hardware_environment": {"platform": platform.platform(), "processor": platform.processor(),
                                 "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                                 "torch_num_threads": bert.torch.get_num_threads(), "native_pool_limit": threads,
                                 "native_pools_during_measurement": pools,
                                 "packages": {name: importlib.metadata.version(name) for name in
                                              ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "torch", "transformers", "tokenizers")}},
        "per_prompt_timings": per_prompt,
        "source_files_sha256": {name: _hash(root / name) for name in relevant_sources},
        "outer_test_evaluated": False,
        "model_responses_generated": False,
        "paid_api_calls": False,
        "limitations": ["small nonrandom development sample, not a serving SLA or population confidence interval",
                        "resident models and fitted vocabularies are reused; input representations and forward predictions are recomputed",
                        "background CPU load and OS scheduling are not controlled or certified by this script; run after training stops",
                        "model loading is separate; no network, queueing, LLM generation, retries or application rendering is timed",
                        "latency does not establish quality superiority, monetary net savings or production readiness"],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    # Exclusive creation prevents normal benchmark execution from replacing an
    # earlier result, including if another invocation completed during timing.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/relay-v3-development-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "latency_cpu.json"
    report = benchmark(ROOT, args.run_dir, output)
    print(json.dumps({"output": str(output), "candidate": report["selected_candidate"],
                      "latency": report["latency"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
