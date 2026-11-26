"""Train a small local adapter from completed, already observed catalog data.

No generation, provider or network is invoked. All observed durations are seconds
and remain distinct from dollars. Calibration selects a development policy; its
post-selection intervals do not constitute an independent evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_transfer_pilot import SYSTEM
from routerbench.comparison import paired_bootstrap
from routerbench.data import sha256_file, write_json
from routerbench.features import normalize_for_dedup
from routerbench.nonlinear import NonlinearRegretRouter
from routerbench.pairwise import choose_pairwise
from routerbench.prospective import _hash, _read_cases


def local_partition(groups, seed=42, fit_fraction=.7):
    if not 0 < fit_fraction < 1:
        raise ValueError("Las fracciones fit/cal deben ser positivas.")
    return groups.map(lambda group: "fit" if int(hashlib.sha256(
        f"local-adapter:{seed}:{group}".encode()).hexdigest(), 16)/2**256 < fit_fraction else "calibration")


def router_text(prompt):
    return f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_local_development(root, config):
    source = root / config["source_dir"]
    dataset = root / config["dataset_path"]
    rows, _, data_hash = _read_cases(dataset)
    if len(rows) != config["expected_cases"] or any(row["grade"] != "exact" for row in rows):
        raise ValueError("La adaptación necesita la cohorte exact original de tamaño declarado.")
    freeze, summary, checkpoint = [_json(source / name) for name in ("freeze.json", "summary.json", "responses.json")]
    freeze_body = {name: value for name, value in freeze.items() if name != "freeze_sha256"}
    summary_body = {name: value for name, value in summary.items() if name != "result_sha256"}
    if _hash(freeze_body) != freeze["freeze_sha256"] or _hash(summary_body) != summary["result_sha256"]:
        raise ValueError("Las huellas de freeze o summary no coinciden.")
    if freeze["dataset_sha256"] != data_hash or summary["dataset_sha256"] != data_hash or summary["freeze_sha256"] != freeze["freeze_sha256"]:
        raise ValueError("La cohorte no coincide con la auditoría completada.")
    if summary["responses_sha256"] != sha256_file(source / "responses.json") or summary["routes_sha256"] != sha256_file(source / "routes.json"):
        raise ValueError("Los checkpoints de la auditoría cambiaron.")
    responses = checkpoint["payload"]
    if checkpoint["freeze_sha256"] != freeze["freeze_sha256"] or checkpoint["payload_sha256"] != _hash(responses):
        raise ValueError("La caché de respuestas no corresponde a la auditoría.")
    for alias, name in config["models"].items():
        if not freeze["policy_manifest"]["models"][alias].startswith(name):
            raise ValueError("El catálogo observado no coincide con los modelos del adaptador.")
    ids = pd.Index([row["id"] for row in rows], name="case_id")
    metadata = pd.DataFrame({"prompt_group": [hashlib.sha256(normalize_for_dedup(row["prompt"]).encode()).hexdigest() for row in rows],
                             "source_prompt_sha256": [row["prompt_sha256"] for row in rows]}, index=ids)
    metadata["split"] = local_partition(metadata["prompt_group"], **{key: config["split"][key] for key in ("seed", "fit_fraction")})
    inputs = pd.DataFrame({"prompt_text": [router_text(row["prompt"]) for row in rows]}, index=ids)
    metadata["router_input_sha256"] = inputs["prompt_text"].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    scores = pd.DataFrame(index=ids)
    durations = pd.DataFrame(index=ids)
    for alias in ("strong", "weak"):
        outcomes = summary["outcomes"]["baseline_"+alias]
        if len(outcomes) != len(rows) or len({outcome["id"] for outcome in outcomes}) != len(rows):
            raise ValueError("Los resultados del baseline contienen filas ambiguas.")
        by_id = {outcome["id"]: outcome for outcome in outcomes}
        values, seconds = [], []
        for row in rows:
            outcome = by_id[row["id"]]
            key = _hash([row["id"], row["prompt_sha256"], alias])
            record = responses[key]
            correct = normalize_for_dedup(record["content"] or "") == normalize_for_dedup(row["reference"])
            duration = record["duration_seconds"]
            if outcome["model"] != alias or outcome["grade"] != "exact" or outcome["response_key"] != key or not isinstance(outcome["success"], bool) or outcome["success"] != correct or record["id"] != row["id"] or record["model"] != alias or record["prompt_sha256"] != row["prompt_sha256"] or record["status"] != "ok":
                raise ValueError("Outcomes y respuestas locales no coinciden.")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Se requieren duraciones de generación positivas y finitas, en segundos.")
            values.append(float(correct))
            seconds.append(duration)
        scores[config["models"][alias]] = values
        durations[config["models"][alias]] = seconds
    provenance = {name: sha256_file(source / name) for name in ("freeze.json", "summary.json", "responses.json", "routes.json")}
    provenance["dataset_sha256"] = data_hash
    return inputs, scores, durations, metadata, provenance


def evaluate_seconds(selected, scores, durations):
    successes, seconds = [], []
    for row, model in selected.items():
        successes.append(bool(scores.loc[row, model] >= .75))
        seconds.append(float(durations.loc[row, model]))
    observed = pd.DataFrame({"success": successes, "cost": seconds}, index=selected.index)
    return {"n": len(selected), "success_rate": float(observed["success"].mean()),
            "total_selected_generation_seconds": float(observed["cost"].sum()),
            "mean_selected_generation_seconds": float(observed["cost"].mean()),
            "cost_unit": "seconds", "monetary_cost_usd": None}, observed


def make_bert_transformer(reference_config):
    from routerbench.bert_features import FrozenBERTFeatures
    return FrozenBERTFeatures(reference_config)


def _train_one(model, inputs, scores, durations, fit, calibration, config, output, prefix, artifact_name):
    model.fit(inputs.loc[fit], scores.loc[fit], durations.loc[fit], inputs.loc[calibration], scores.loc[calibration])
    artifact = output / artifact_name
    joblib.dump(model, artifact)
    p, estimated_seconds = model.predict_tables(inputs.loc[calibration])
    p.to_parquet(output / f"{prefix}calibration_probabilities.parquet")
    estimated_seconds.to_parquet(output / f"{prefix}calibration_predicted_seconds.parquet")
    fixed = pd.Series(model.strong, index=calibration)
    fixed_metrics, fixed_rows = evaluate_seconds(fixed, scores.loc[calibration], durations.loc[calibration])
    curve, actions = [], {}
    for parameter in config["tradeoff_grid"]:
        actions[str(parameter)] = choose_pairwise(p, estimated_seconds, model.strong, model.weak, parameter)["selected_model"]
    pd.DataFrame(actions).to_parquet(output / f"{prefix}calibration_routes_all_points.parquet")
    for parameter in config["tradeoff_grid"]:
        selected = actions[str(parameter)]
        metrics, _ = evaluate_seconds(selected, scores.loc[calibration], durations.loc[calibration])
        curve.append({"tradeoff": parameter, "metrics": metrics, "weak_fraction": float(selected.eq(model.weak).mean())})
    feasible = [point for point in curve if point["metrics"]["success_rate"] >= fixed_metrics["success_rate"]]
    if feasible:
        chosen = min(feasible, key=lambda point: (point["metrics"]["total_selected_generation_seconds"], -point["metrics"]["success_rate"], point["tradeoff"]))
        selected = actions[str(chosen["tradeoff"])]
        specification = {"mode": "pairwise", "policy": "signed_pairwise_utility", "tradeoff": chosen["tradeoff"], "cost_unit": "seconds"}
    else:
        selected = fixed
        specification = {"mode": "fixed_strong_fallback", "policy": "fixed_strong", "tradeoff": None, "cost_unit": "seconds"}
    specification.update(input_template=config["router_input"], system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
                         strong=model.strong, weak=model.weak, quality_threshold=model.quality_threshold)
    metrics, observed = evaluate_seconds(selected, scores.loc[calibration], durations.loc[calibration])
    selected.to_frame("selected_model").to_parquet(output / f"{prefix}selected_calibration_routes.parquet")
    comparison = paired_bootstrap(observed, fixed_rows, config["bootstrap_replicates"], config["split"]["seed"])
    diagnostic = {"success_rate_delta": comparison["success_rate_delta"], "success_rate_delta_95ci": comparison["success_rate_delta_95ci"],
                  "mean_selected_seconds_delta": comparison["cost_per_1000_delta"]/1000,
                  "mean_selected_seconds_delta_95ci": [value/1000 for value in comparison["cost_per_1000_delta_95ci"]],
                  "relative_selected_duration_saving": comparison["relative_call_cost_saving"],
                  "relative_selected_duration_saving_95ci": comparison["relative_call_cost_saving_95ci"],
                  "scope": "descriptive post-selection calibration diagnostics on a small reused cohort; seconds, not money"}
    return {"artifact": artifact.name, "artifact_sha256": sha256_file(artifact),
            "target_counts": model.target_counts_, "calibration_notes": model.calibration_notes_,
            "fit_rows": model.fit_rows_, "calibration_rows": model.calibration_rows_,
            "classifier_fit_rows": model.classifier_fit_rows_, "classifier_calibration_rows": model.classifier_calibration_rows_,
            "selected_specification": specification, "selected_calibration_metrics": metrics,
            "fixed_strong_calibration_metrics": fixed_metrics, "curve": curve,
            "calibration_comparison_vs_strong": diagnostic}


def train_adapter(root=ROOT, config=None, output=None):
    root = Path(root)
    config = _json(root / "configs/local_adapter_v1.json") if config is None else config
    output = root / config["output_dir"] if output is None else Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("La adaptación existente se conserva; no se sobrescribe.")
    if config["cost_unit"] != "seconds" or any(config[name] is not False for name in ("automatic_live_promotion", "independent_superiority_claim", "paid_api_calls")):
        raise ValueError("Este adaptador solo permite desarrollo local con duraciones, sin promoción automática.")
    if config["tradeoff_grid"] != [0, .01, .02, .05, .1, .2, .5, 1, 2]:
        raise ValueError("La grilla local debe conservar sus nueve puntos predeclarados.")
    if config["router_input"] != "SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}":
        raise ValueError("La serialización debe conservar SYSTEM y USER como se usaron en el controlador.")
    reference_path = root / config["bert_reference_selection_file"]
    if sha256_file(reference_path) != config["bert_reference_selection_sha256"]:
        raise ValueError("La referencia BERT fijada cambió.")
    reference = _json(reference_path)
    if reference["test_outcomes_used_for_selection"] is not False:
        raise ValueError("La referencia BERT no conserva selección original de desarrollo.")
    reference_config = reference["configuration"]
    inputs, scores, durations, rows, provenance = load_local_development(root, config)
    fit, calibration = [rows.index[rows["split"].eq(name)] for name in ("fit", "calibration")]
    if min(len(fit), len(calibration)) < 2:
        raise ValueError("El hash no produjo soporte suficiente; no se cambia la partición.")
    output.mkdir(parents=True, exist_ok=True)
    started = {"configuration": config, "configuration_sha256": _hash(config), "source_files_sha256": provenance,
               "system": SYSTEM, "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
               "serialization": config["router_input"], "scope": "known synthetic cohort reused for development, not independent validation",
               "bert_reference_selection_sha256": sha256_file(reference_path),
               "bert_checkpoint_revision": reference_config["checkpoint_revision"],
               "bert_weights_sha256": reference_config["weights_sha256"],
               "same_local_head_hyperparameters": {name: value for name, value in config["learner"].items() if name != "representation"}}
    write_json(output / "training_started.json", started)
    rows.to_parquet(output / "development_rows.parquet")
    strong, weak = config["models"]["strong"], config["models"]["weak"]
    model = NonlinearRegretRouter(strong=strong, weak=weak, **config["learner"])
    primary = _train_one(model, inputs, scores, durations, fit, calibration, config, output, "", "adapter.joblib")
    print("Adaptador primario estructural completo; entrenando control BERT adaptado con las mismas filas.", flush=True)
    control_parameters = {**config["learner"], "representation": "bert_cls"}
    control = NonlinearRegretRouter(strong=strong, weak=weak, feature_transformer=make_bert_transformer(reference_config), **control_parameters)
    control_report = _train_one(control, inputs, scores, durations, fit, calibration, config, output,
                                "bert_adapted_", "bert_adapted.joblib")
    control_report["role"] = "locally adapted BERT feature control; not the published preference router"
    report = {**started, **primary, "models": config["models"],
              "partition_counts": {"fit": len(fit), "calibration": len(calibration)},
              "partition_prompt_groups": {name: rows.loc[rows["split"].eq(name), "prompt_group"].tolist() for name in ("fit", "calibration")},
              "controls": {"bert_adapted": control_report}, "primary_candidate_not_selected_by_architecture": True,
              "independent_validation": False,
              "production_validation": False, "monetary_cost_usd": None,
              "limitations": ["64 already observed synthetic cases reused for development; hash splitting does not restore independence",
                              "calibration both calibrates the model and selects its operating point; reported intervals are descriptive",
                              "cached callback durations include hardware/warmup/order variation and are not prices",
                              "new cohort needed after freezing this artifact, policy, local model digests and generation settings"],
              "script_sha256": sha256_file(Path(__file__).resolve()),
              "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_adapter_v1.json")
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        result = train_adapter(ROOT, _json(args.config))
    print(json.dumps({"artifact": result["artifact"], "selected_specification": result["selected_specification"],
                      "calibration_metrics": result["selected_calibration_metrics"], "independent_validation": False}, indent=2))


if __name__ == "__main__":
    main()
