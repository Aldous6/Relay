"""Pinned offline RouteLLM BERT transfer reference, separate from primary results.

The original BERT router returns 1 - sum(softmax(logits)[-2:]) and sends
probabilities >= its threshold to the strong model. Its target is relative
preference, unlike Relay's score-threshold success probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .comparison import paired_bootstrap
from .data import load_verified_pickle, sha256_file, write_json
from .features import parse_prompt
from .labels import outcome_tables
from .policy import choose_models, evaluate_policy

ROOT = Path(__file__).resolve().parents[2]


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reference_probabilities(logits) -> np.ndarray:
    """Faithful official three-class aggregation, with stable softmax."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Se requieren tres logits finitos por prompt.")
    probabilities = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.clip(1 - probabilities[:, -2:].sum(axis=1), 0, 1)


def reference_routes(probabilities: pd.Series, threshold: float, pair: dict) -> pd.Series:
    values = probabilities.to_numpy(dtype=float)
    if not probabilities.index.is_unique or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilidades BERT inválidas.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Umbral BERT inválido.")
    return pd.Series(np.where(values >= threshold, pair["strong"], pair["weak"]),
                     index=probabilities.index, name="selected_model")


def select_operating_point(curve: list[dict], strong_success: float, margin: float) -> dict:
    """Use development alone: minimum cost meeting the declared quality floor."""
    if not curve or not np.isfinite([strong_success, margin]).all() or not 0 <= strong_success <= 1 or not 0 <= margin <= 1:
        raise ValueError("Regla de selección inválida.")
    for point in curve:
        values = [point["threshold"], point["success_rate"], point["cost_per_1000"]]
        if not np.isfinite(values).all() or not 0 <= values[0] <= 1 or not 0 <= values[1] <= 1 or values[2] < 0:
            raise ValueError("Punto de desarrollo inválido.")
    floor = strong_success - margin
    eligible = [point for point in curve if point["success_rate"] >= floor]
    # Quality is a constraint rather than a second search objective when feasible.
    selected = min(eligible, key=lambda p: (p["cost_per_1000"], p["threshold"])) if eligible else min(
        curve, key=lambda p: (-p["success_rate"], p["cost_per_1000"], p["threshold"]))
    return {"threshold": selected["threshold"], "development_quality_floor": floor,
            "quality_constraint_met": bool(eligible), "fallback_used": not bool(eligible),
            "development_metrics": selected}


def verify_local_model(directory: Path, config: dict) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["checkpoint"] != config["checkpoint"] or manifest["revision"] != config["checkpoint_revision"]:
        raise ValueError("El checkpoint local no corresponde a la revisión fijada.")
    required = {config["weights_file"], "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "LICENSE"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Faltan archivos en el manifiesto local.")
    for name in required:
        if Path(name).name != name or sha256_file(directory / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Archivo del checkpoint modificado: {name}.")
    if manifest["files"][config["weights_file"]]["sha256"] != config["weights_sha256"]:
        raise ValueError("Los pesos no tienen el hash fijado.")
    return manifest


class LocalBERTReference:
    """CPU only; no package-level RouteLLM imports and no remote model loading."""

    def __init__(self, directory: Path, config: dict):
        self.config = config
        self.manifest = verify_local_model(directory, config)
        # Must be set before importing transformers/hub. Local files are also
        # required explicitly, so an absent file cannot silently use the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if torch.__version__.split("+")[0] != config["packages"]["torch"] or importlib.metadata.version("transformers") != config["packages"]["transformers"]:
            raise ValueError("Ejecuta este módulo con el entorno .venv-routellm fijado.")
        torch.set_num_threads(config["threads"])
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(directory), num_labels=3, local_files_only=True, use_safetensors=True).to("cpu").eval()
        self.max_length = config["reference_tokenization"]["max_length"]

    def predict_batch(self, prompts: list[str]) -> list[dict]:
        # Count the full original prompt independently of the official truncation.
        full = self.tokenizer(prompts, truncation=False, padding=False, add_special_tokens=True)["input_ids"]
        encoded = self.tokenizer(prompts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        retained = encoded["attention_mask"].sum(dim=1).tolist()
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits.detach().cpu().numpy()
        probabilities = reference_probabilities(logits)
        return [{"probability": float(probability), "input_tokens": len(tokens),
                 "retained_tokens": int(count), "truncated_tokens": max(0, len(tokens)-int(count))}
                for probability, tokens, count in zip(probabilities, full, retained)]


def cached_probabilities(prompts: pd.Series, predictor, cache_path: Path,
                         specification: dict, batch_size: int = 8) -> pd.DataFrame:
    """Resume only an exact text/order/checkpoint/protocol match; checkpoint /256."""
    if not prompts.index.is_unique or prompts.empty or not prompts.map(lambda x: isinstance(x, str) and bool(x)).all() or not 1 <= batch_size <= 8:
        raise ValueError("Prompts, índice o tamaño de lote inválidos.")
    prompt_keys = [{"source_row": int(row), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                   for row, text in prompts.items()]
    spec = {**specification, "prompts_sha256": object_hash(prompt_keys), "n_prompts": len(prompts)}
    records = []
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["specification"] != spec or object_hash(cached["rows"]) != cached["rows_sha256"]:
            raise ValueError("Caché BERT incompatible o modificada.")
        records = cached["rows"]
        if len(records) > len(prompt_keys) or any(record["source_row"] != key["source_row"] for record, key in zip(records, prompt_keys)):
            raise ValueError("Filas de la caché BERT desalineadas.")
    def save():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        write_json(temporary, {"specification": spec, "rows": records, "rows_sha256": object_hash(records)})
        temporary.replace(cache_path)
    tick = time.perf_counter()
    last_saved = len(records)
    if records:
        print(f"BERT {specification['partition']}: caché {len(records)}/{len(prompts)}.", flush=True)
    for start in range(len(records), len(prompts), batch_size):
        batch = prompts.iloc[start:start+batch_size]
        values = predictor.predict_batch(batch.tolist())
        if len(values) != len(batch):
            raise ValueError("El encoder devolvió otro número de filas.")
        for (row, _), value in zip(batch.items(), values):
            probability = value["probability"]
            if not np.isfinite(probability) or not 0 <= probability <= 1 or not 0 <= value["retained_tokens"] <= value["input_tokens"] or value["truncated_tokens"] != value["input_tokens"] - value["retained_tokens"]:
                raise ValueError("Salida BERT inválida.")
            records.append({"source_row": int(row), **value})
        if len(records)-last_saved >= 256 or len(records) == len(prompts):
            save()
            last_saved = len(records)
            print(f"BERT {specification['partition']}: {len(records)}/{len(prompts)}; {time.perf_counter()-tick:.1f}s en esta ejecución.", flush=True)
    return pd.DataFrame(records).set_index("source_row").loc[prompts.index]


def coverage_summary(frame: pd.DataFrame) -> dict:
    return {"n": len(frame), "input_tokens": int(frame["input_tokens"].sum()),
            "retained_tokens": int(frame["retained_tokens"].sum()),
            "truncated_tokens": int(frame["truncated_tokens"].sum()),
            "prompts_truncated": int(frame["truncated_tokens"].gt(0).sum()),
            "fraction_prompts_truncated": float(frame["truncated_tokens"].gt(0).mean()),
            "token_coverage": float(frame["retained_tokens"].sum()/frame["input_tokens"].sum()),
            "max_input_tokens": int(frame["input_tokens"].max()),
            "scope": "official tokenizer truncation at 512 including special tokens; no sliding windows"}


def _verified_preparation(root: Path, config: dict):
    directory = root / config["prepared_dir"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_sha256"] != config["dataset"]["sha256"]:
        raise ValueError("Datos preparados incompatibles.")
    for name, checksum in manifest["files_sha256"].items():
        if sha256_file(directory / name) != checksum:
            raise ValueError(f"Preparación modificada: {name}.")
    rows = pd.read_parquet(directory / "rows.parquet").set_index("source_row", drop=False)
    if not rows.index.is_unique:
        raise ValueError("Filas preparadas repetidas.")
    return rows, manifest


def _pair_predictions(directory: Path, probability_file: str, cost_file: str,
                      ids: pd.Index, pair: list[str], hashes: dict | None = None):
    frames, evidence = [], {}
    for name in (probability_file, cost_file):
        checksum = sha256_file(directory / name)
        if hashes is not None and checksum != hashes[name]:
            raise ValueError(f"Predicciones Relay modificadas: {name}.")
        frame = pd.read_parquet(directory / name)
        if not frame.index.equals(ids) or not frame.index.is_unique or not frame.columns.is_unique or not set(pair).issubset(frame.columns):
            raise ValueError("Predicciones Relay desalineadas o con otro par.")
        frames.append(frame.loc[:, pair])
        evidence[name] = checksum
    return *frames, evidence


def _pair_outcomes(raw: pd.DataFrame, ids: pd.Index, pair: list[str]):
    # Only the requested partition and common pair are inspected.
    columns = pair + [model + "|total_cost" for model in pair]
    frame = raw.iloc[ids.to_numpy()].loc[:, columns].copy()
    frame.index = ids
    return outcome_tables(frame)


def require_frozen_selection(path: Path, config_hash: str) -> dict:
    if not path.is_file():
        raise ValueError("Congela selection.json antes de acceder al test secundario.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_before_secondary_test" or selection.get("configuration_sha256") != config_hash:
        raise ValueError("La selección secundaria no está congelada para esta configuración.")
    return selection


def run_reference(root: Path, config: dict, development_only: bool = False) -> dict:
    started = time.perf_counter()
    if config["threshold_grid"] != [point/10 for point in range(1, 10)] or config["reference_tokenization"] != {"max_length": 512, "truncation": True, "padding": True, "prefix": None}:
        raise ValueError("La referencia requiere la grilla fijada y el tokenizador oficial.")
    config_hash = object_hash(config)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if (output / "summary.json").exists():
        raise FileExistsError("La comparación secundaria ya terminó; conserva sus resultados.")
    rows, prepared = _verified_preparation(root, config)
    source_dir = root / config["relay_development_run"]
    source = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if source["dataset_sha256"] != prepared["dataset_sha256"] or source["source_prepared_files_sha256"] != prepared["files_sha256"]:
        raise ValueError("Desarrollo Relay usa otra preparación.")
    metadata_file = source_dir / "development_rows.parquet"
    if sha256_file(metadata_file) != source["files_sha256"][metadata_file.name]:
        raise ValueError("La partición de desarrollo fue modificada.")
    development = pd.read_parquet(metadata_file).set_index("source_row")
    if not development.index.is_unique or set(development.index) != set(rows.index[rows["split"].eq("train")]) or not development["prompt_group"].equals(rows.loc[development.index, "prompt_group"]):
        raise ValueError("Las particiones no corresponden al train congelado.")
    dev_ids = development.index[development["development_split"].eq("validation")]
    if len(dev_ids) != config["expected_population"]["development_validation"] or rows["split"].eq("test").sum() != config["expected_population"]["outer_test"]:
        raise ValueError("Las poblaciones no corresponden al protocolo secundario.")
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    variant = config["relay_variant"]
    dev_probs, dev_predicted_costs, dev_hashes = _pair_predictions(source_dir, f"{variant}_probabilities.parquet",
        f"{variant}_predicted_costs.parquet", dev_ids, pair, source["files_sha256"])
    raw = load_verified_pickle(root / config["dataset"]["path"], config["dataset"]["sha256"])
    predictor = LocalBERTReference(root / config["model_dir"], config)
    cache_spec = {"configuration_sha256": config_hash, "checkpoint_revision": config["checkpoint_revision"],
                  "weights_sha256": config["weights_sha256"], "partition": "development_validation"}
    dev_prompts = raw["prompt"].iloc[dev_ids.to_numpy()].map(parse_prompt)
    dev_prompts.index = dev_ids
    dev_bert = cached_probabilities(dev_prompts, predictor, output / "development_bert_cache.json", cache_spec, config["batch_size"])
    dev_scores, dev_costs = _pair_outcomes(raw, dev_ids, pair)
    quality = config["quality_score_threshold"]
    strong_dev = evaluate_policy(pd.Series(pair[0], index=dev_ids), dev_scores, dev_costs, quality)["metrics"]
    curves = {"routellm_bert": [], "relay_common_pair": []}
    for threshold in config["threshold_grid"]:
        actions = {"routellm_bert": reference_routes(dev_bert["probability"], threshold, config["common_pair"]),
                   "relay_common_pair": choose_models(dev_probs, dev_predicted_costs, threshold)["selected_model"]}
        for method, selected in actions.items():
            curves[method].append({"threshold": threshold, **evaluate_policy(selected, dev_scores, dev_costs, quality)["metrics"]})
    selection = {"status": "frozen_before_secondary_test", "protocol_id": config["protocol_id"],
                 "configuration_sha256": config_hash, "configuration": config,
                 "n_development_validation": len(dev_ids), "strong_development": strong_dev,
                 "development_curves": curves, "development_coverage": coverage_summary(dev_bert),
                 "source_prepared_files_sha256": prepared["files_sha256"],
                 "relay_source_summary_sha256": sha256_file(source_dir / "summary.json"),
                 "relay_development_prediction_sha256": dev_hashes,
                 "reference_manifest_sha256": sha256_file(root / config["model_dir"] / "manifest.json"),
                 "selected": {method: select_operating_point(curve, strong_dev["success_rate"], config["development_quality_loss_margin"])
                              for method, curve in curves.items()},
                 "test_outcomes_used_for_selection": False, "primary_protocol_changed": False}
    if selection_path.exists():
        frozen = require_frozen_selection(selection_path, config_hash)
        if {key: value for key, value in frozen.items() if key != "frozen_at_utc"} != selection:
            raise ValueError("La selección ya congelada cambió; no se sobrescribe.")
    else:
        selection["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        # Exclusive creation makes this immutable to the normal evaluation path.
        with selection_path.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    if development_only:
        return require_frozen_selection(selection_path, config_hash)
    # This gate precedes accessing test texts, predictions, or outcomes.
    selection = require_frozen_selection(selection_path, config_hash)
    selection_hash = sha256_file(selection_path)
    test_ids = rows.index[rows["split"].eq("test")]
    test_prompts = raw["prompt"].iloc[test_ids.to_numpy()].map(parse_prompt)
    test_prompts.index = test_ids
    test_bert = cached_probabilities(test_prompts, predictor, output / "test_bert_cache.json",
        {**cache_spec, "partition": "outer_test", "selection_sha256": selection_hash}, config["batch_size"])
    final_dir = root / config["relay_final_dir"]
    test_probs, test_predicted_costs, test_hashes = _pair_predictions(final_dir,
        config["relay_final_probability_file"], config["relay_final_predicted_cost_file"], test_ids, pair)
    actions = {"routellm_bert": reference_routes(test_bert["probability"], selection["selected"]["routellm_bert"]["threshold"], config["common_pair"]),
               "relay_common_pair": choose_models(test_probs, test_predicted_costs, selection["selected"]["relay_common_pair"]["threshold"])["selected_model"],
               "fixed_strong": pd.Series(pair[0], index=test_ids), "fixed_weak": pd.Series(pair[1], index=test_ids)}
    for method, selected in actions.items():
        selected.to_frame("selected_model").to_parquet(output / f"{method}_routes.parquet")
    # All test actions are fixed and persisted before observed scores or costs.
    if sha256_file(selection_path) != selection_hash:
        raise ValueError("La selección cambió durante la ejecución.")
    test_scores, test_costs = _pair_outcomes(raw, test_ids, pair)
    evaluations = {method: evaluate_policy(selected, test_scores, test_costs, quality) for method, selected in actions.items()}
    for method, evaluation in evaluations.items():
        evaluation["rows"].to_parquet(output / f"{method}_observed_results.parquet")
    test_bert.to_parquet(output / "test_bert_probabilities_and_coverage.parquet")
    dev_bert.to_parquet(output / "development_bert_probabilities_and_coverage.parquet")
    comparisons = {}
    for candidate, reference in (("relay_common_pair", "routellm_bert"), ("relay_common_pair", "fixed_strong"), ("routellm_bert", "fixed_strong")):
        result = paired_bootstrap(evaluations[candidate]["rows"], evaluations[reference]["rows"], config["bootstrap_replicates"], config["seed"])
        result["scope"] = "secondary held-out paired prompt bootstrap; adapted Relay vs externally pretrained BERT transfer; historical call cost excludes overhead"
        comparisons[f"{candidate}_vs_{reference}"] = result
    report = {"protocol_id": config["protocol_id"], "kind": config["scope"],
              "selection_sha256": selection_hash, "configuration_sha256": config_hash,
              "n_development_validation": len(dev_ids), "n_test": len(test_ids),
              "selected": selection["selected"], "metrics": {method: result["metrics"] for method, result in evaluations.items()},
              "comparisons": comparisons, "coverage": {"development": coverage_summary(dev_bert), "test": coverage_summary(test_bert)},
              "reference": {"checkpoint": config["checkpoint"], "revision": config["checkpoint_revision"],
                            "source_revision": config["source_revision"], "implementation": "local faithful BERT inference adapter; original RouteLLM package not imported"},
              "relay_test_prediction_sha256": test_hashes, "paid_api_calls": False,
              "primary_protocol_changed": False, "test_outcomes_used_for_selection": False,
              "elapsed_seconds_this_execution": time.perf_counter()-started,
              "limitations": config["limitations"], "evaluated_at_utc": datetime.now(timezone.utc).isoformat()}
    report["files_sha256"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file() and path.suffix != ".tmp"}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/routellm_reference.json")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_reference(ROOT, config, args.development_only)
    print(json.dumps({"output": config["output_dir"], "selected": report["selected"],
                      "n_test": report.get("n_test"), "metrics": report.get("metrics")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
