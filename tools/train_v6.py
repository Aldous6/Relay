"""Train/evaluate v6 on reused development only; never generate or open test outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.comparison import paired_bootstrap
from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs
from routerbench.improvement import activate_full_fit
from routerbench.policy import evaluate_policy


def make_router(pair, parameters):
    from routerbench.guarded import GuardedPairRouter
    return GuardedPairRouter(strong=pair[0], weak=pair[1], **parameters)


def guarded_routes(probabilities, costs, pair, max_harm, min_weak_success):
    from routerbench.guarded import choose_guarded
    return choose_guarded(probabilities, costs, pair[0], pair[1], max_harm, min_weak_success)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _verify(root, relative, digest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
        raise ValueError(f"Fuente modificada o fuera del proyecto: {relative}")
    return path


def select_point(points, quality_floor, maximum_cost=None):
    """Select from means, never from post-selection confidence intervals."""
    feasible = [point for point in points if point["metrics"]["success_rate"] >= quality_floor
                and (maximum_cost is None or point["metrics"]["cost_per_1000"] < maximum_cost)]
    return min(feasible, key=lambda point: (point["metrics"]["cost_per_1000"],
                                           -point["metrics"]["success_rate"], point["index"])) if feasible else None


def _validate(config):
    for flag in ("outer_test_evaluation_allowed", "automatic_live_promotion_allowed", "paid_api_calls_authorized"):
        if config.get(flag) is not False:
            raise ValueError("V6 no permite test, promoción automática ni APIs pagadas.")
    if config["selection"]["quality_loss_margin"] != 0:
        raise ValueError("V6 conserva el margen de selección cero predeclarado.")
    if (config["max_harm_grid"] != [.01, .03, .05, .1] or
            config["min_weak_success_grid"] != [.5, .65, .8, .9] or
            config["reference"]["threshold_grid"] != [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, 1]):
        raise ValueError("Las grillas v6 deben conservar sus 16 puntos predeclarados.")
    if config["learner"]["quality_threshold"] != config["quality_threshold"]:
        raise ValueError("El objetivo de entrenamiento debe coincidir con el score de evaluación.")
    pair = config["common_pair"]
    if set(pair) != {"strong", "weak"} or any(not isinstance(value, str) or not value for value in pair.values()) or pair["strong"] == pair["weak"]:
        raise ValueError("Se requiere un par explícito de modelos distintos.")


def _freeze_sources(output, config, source_hashes):
    paths = [Path(__file__).resolve(), ROOT / "configs/relay_v6.json", ROOT / "docs/relay_v6_protocolo.md",
             *sorted((ROOT / "src").rglob("*.py"))]
    hashes = {}
    with zipfile.ZipFile(output / "source_bundle.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            hashes[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(relative, content)
        archive.writestr("configuration.json", json.dumps(config, ensure_ascii=False, indent=2))
    frozen = {"configuration": config, "configuration_sha256": _hash(config),
              "source_files_sha256": hashes, "input_files_sha256": source_hashes,
              "source_bundle_sha256": sha256_file(output / "source_bundle.zip"),
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow")},
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "outer_test_access_allowed": False}
    write_json(output / "development_started.json", frozen)
    return frozen


def run_v6(root, config, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    _validate(config)
    if not output.is_relative_to(root):
        raise ValueError("La salida debe estar dentro del proyecto.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("V6 no sobrescribe ejecuciones existentes; usa otra salida.")
    sources = {config["reference_split_run"] + "/summary.json": config["reference_split_summary_sha256"]}
    reference = config["reference"]
    sources.update({reference["selection_file"]: reference["selection_sha256"],
                    reference["development_probabilities_file"]: reference["development_probabilities_sha256"]})
    control_config = config.get("previous_bert_control")
    if control_config:
        sources.update({control_config[name]: control_config[name + "_sha256"] for name in ("summary_file", "curve_file", "actions_file")})
    for path, digest in sources.items():
        _verify(root, path, digest)
    prior = json.loads((root / reference["selection_file"]).read_text(encoding="utf-8"))
    if prior["test_outcomes_used_for_selection"] is not False:
        raise ValueError("La referencia debe estar seleccionada exclusivamente con desarrollo.")
    prior_config = prior["configuration"]
    if ({key: prior_config["common_pair"][key] for key in ("strong", "weak")} != config["common_pair"] or
            prior_config["quality_score_threshold"] != config["quality_threshold"] or
            prior_config["dataset"]["sha256"] != config["dataset"]["sha256"]):
        raise ValueError("El par, objetivo o dataset no coincide con la referencia BERT histórica.")
    inputs, scores, costs, rows, manifest = load_development_inputs(root, config)
    rows = activate_full_fit(rows)
    ids = {part: rows.index[rows["development_split"].eq(part)] for part in ("fit", "calibration", "validation")}
    counts = {part: len(index) for part, index in ids.items()}
    if counts != config["expected_population"] or min(counts.values()) < 2:
        raise ValueError("La población no coincide con las particiones declaradas.")
    if not all(frame.index.equals(rows.index) for frame in (inputs, scores, costs)):
        raise ValueError("Los inputs y outcomes no están alineados con las particiones.")
    pair = [config["common_pair"]["strong"], config["common_pair"]["weak"]]
    prior_control = None
    if control_config:
        prior_summary = json.loads((root / control_config["summary_file"]).read_text(encoding="utf-8"))
        prior_curve = json.loads((root / control_config["curve_file"]).read_text(encoding="utf-8"))["curve"]
        prior_actions = pd.read_parquet(root / control_config["actions_file"])
        # sweep_v3.routes persisted booleans: True selects strong, False weak.
        # Decode this documented representation before model-name validation.
        if all(pd.api.types.is_bool_dtype(dtype) for dtype in prior_actions.dtypes) and not prior_actions.isna().any().any():
            prior_actions = pd.DataFrame(np.where(prior_actions.to_numpy(dtype=bool), pair[0], pair[1]),
                                         index=prior_actions.index, columns=prior_actions.columns)
        if (prior_summary["common_pair"] != config["common_pair"] or prior_summary["quality_score_threshold"] != config["quality_threshold"] or
                prior_summary["source_prepared_files_sha256"] != manifest["files_sha256"] or
                prior_summary["source_reference_probabilities_sha256"] != reference["development_probabilities_sha256"] or
                prior_summary["outer_test_accessed"] is not False or not prior_actions.index.equals(ids["validation"]) or
                len(prior_curve) != prior_actions.shape[1] or prior_actions.isna().any().any() or
                not prior_actions.isin(pair).all().all()):
            raise ValueError("El control BERT previo no coincide con población, par o procedencia.")
        prior_control = (prior_curve, prior_actions)
    bert = pd.read_parquet(root / reference["development_probabilities_file"])
    if (not bert.index.equals(ids["validation"]) or not bert.index.is_unique or
            "probability" not in bert or not np.isfinite(bert["probability"]).all() or not bert["probability"].between(0, 1).all()):
        raise ValueError("Predicciones BERT desalineadas o inválidas.")
    sources.update({config.get("prepared_dir", "data/processed") + "/" + name: digest
                    for name, digest in manifest["files_sha256"].items()})
    if "dataset" in config:
        sources[config["dataset"]["path"]] = config["dataset"]["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    frozen = _freeze_sources(output, config, sources)
    rows.to_parquet(output / "development_rows.parquet")
    tick = time.perf_counter()
    router = make_router(pair, config["learner"])
    print(f"V6: fit={counts['fit']}, calibración={counts['calibration']}, validación={counts['validation']}.", flush=True)
    router.fit(inputs.loc[ids["fit"]], scores.loc[ids["fit"], pair], costs.loc[ids["fit"], pair],
               inputs.loc[ids["calibration"]], scores.loc[ids["calibration"], pair])
    training_seconds = time.perf_counter() - tick
    artifact = output / "router.joblib"
    joblib.dump(router, artifact)
    tick = time.perf_counter()
    probabilities, predicted_costs = router.predict_tables(inputs.loc[ids["validation"]])
    batch_seconds = time.perf_counter() - tick
    probabilities.to_parquet(output / "validation_probabilities.parquet")
    predicted_costs.to_parquet(output / "validation_predicted_costs.parquet")
    parameters = [(harm, success) for harm in config["max_harm_grid"] for success in config["min_weak_success_grid"]]
    actions = pd.DataFrame({str(index): guarded_routes(probabilities, predicted_costs, pair, harm, success)["selected_model"]
                            for index, (harm, success) in enumerate(parameters)}, index=ids["validation"])
    bert_actions = pd.DataFrame({str(index): np.where(bert["probability"] >= threshold, pair[0], pair[1])
                                for index, threshold in enumerate(reference["threshold_grid"])}, index=ids["validation"])
    fixed_actions = pd.DataFrame({"strong": pair[0], "weak": pair[1]}, index=ids["validation"])
    if prior_control:
        prior_control[1].to_parquet(output / "previous_bert_all_routes.parquet")
    for name, table in (("relay_all_routes.parquet", actions), ("bert_all_routes.parquet", bert_actions),
                        ("fixed_routes.parquet", fixed_actions)):
        table.to_parquet(output / name)
    action_files = ["relay_all_routes.parquet", "bert_all_routes.parquet", "fixed_routes.parquet"]
    if prior_control:
        action_files.append("previous_bert_all_routes.parquet")
    write_json(output / "actions_frozen.json", {"configuration_sha256": frozen["configuration_sha256"],
               "files_sha256": {name: sha256_file(output / name) for name in action_files},
               "selection_and_metrics_computed": False})
    # Outcomes are consulted for evaluation only after every candidate action was persisted.
    val_scores, val_costs = scores.loc[ids["validation"], pair], costs.loc[ids["validation"], pair]
    quality = config["quality_threshold"]
    fixed = {name: evaluate_policy(fixed_actions[name], val_scores, val_costs, quality) for name in ("strong", "weak")}
    bert_curve = [{"index": index, "threshold": threshold,
                   "metrics": evaluate_policy(bert_actions[str(index)], val_scores, val_costs, quality)["metrics"]}
                  for index, threshold in enumerate(reference["threshold_grid"])]
    fixed_floor = fixed["strong"]["metrics"]["success_rate"]
    bert_point = select_point(bert_curve, fixed_floor)
    if bert_point is None:
        raise ValueError("La grilla BERT debe incluir la referencia fija fuerte (umbral cero).")
    quality_floor = max(fixed_floor, bert_point["metrics"]["success_rate"])
    curve = [{"index": index, "max_harm": harm, "min_weak_success": success,
              "weak_fraction": float(actions[str(index)].eq(pair[1]).mean()),
              "metrics": evaluate_policy(actions[str(index)], val_scores, val_costs, quality)["metrics"]}
             for index, (harm, success) in enumerate(parameters)]
    chosen = select_point(curve, quality_floor, fixed["strong"]["metrics"]["cost_per_1000"])
    selected = fixed_actions["strong"] if chosen is None else actions[str(chosen["index"])]
    selected.to_frame("selected_model").to_parquet(output / "selected_routes.parquet")
    bert_selected = bert_actions[str(bert_point["index"])]
    bert_selected.to_frame("selected_model").to_parquet(output / "bert_selected_routes.parquet")
    observed = evaluate_policy(selected, val_scores, val_costs, quality)
    bert_observed = evaluate_policy(bert_selected, val_scores, val_costs, quality)
    observed["rows"].to_parquet(output / "selected_observed_results.parquet")
    bert_observed["rows"].to_parquet(output / "bert_observed_results.parquet")
    comparisons = {"vs_fixed_strong": paired_bootstrap(observed["rows"], fixed["strong"]["rows"], config["bootstrap_replicates"], config["seed"]),
                   "vs_routellm_bert": paired_bootstrap(observed["rows"], bert_observed["rows"], config["bootstrap_replicates"], config["seed"])}
    extra_controls = {}
    if prior_control:
        prior_curve, prior_actions = prior_control
        prior_point = select_point([{**point, "index": index} for index, point in enumerate(prior_curve)], fixed_floor)
        if prior_point is None:
            raise ValueError("La curva previa BERT no incluye un punto factible.")
        control_observed = evaluate_policy(prior_actions.iloc[:, prior_point["index"]], val_scores, val_costs, quality)
        if control_observed["metrics"] != prior_point["metrics"]:
            raise ValueError("El punto BERT previo no reproduce sus métricas con estas filas.")
        comparisons["vs_previous_bert_101"] = paired_bootstrap(observed["rows"], control_observed["rows"], config["bootstrap_replicates"], config["seed"])
        extra_controls["previous_bert_101"] = {"selected_point": prior_point, "selection_floor": fixed_floor,
                                                 "scope": "previous 101-point development search; descriptive control, not equal search budget"}
    default_index = reference["threshold_grid"].index(.5)
    extra_controls["published_bert_threshold_0_5"] = bert_curve[default_index]
    selected_weak = selected.eq(pair[1])
    strong_ok, weak_ok = val_scores[pair[0]].ge(quality), val_scores[pair[1]].ge(quality)
    diagnostics = {"weak_route_fraction": float(selected_weak.mean()), "weak_routes": int(selected_weak.sum()),
                   "strong_only_on_weak_routes": int((selected_weak & strong_ok & ~weak_ok).sum()),
                   "weak_only_on_weak_routes": int((selected_weak & ~strong_ok & weak_ok).sum()),
                   "weak_successes_on_weak_routes": int((selected_weak & weak_ok).sum()),
                   "brier_harm": float(np.mean((probabilities["harm_probability"] - (strong_ok & ~weak_ok).astype(float)) ** 2)),
                   "brier_weak_success": float(np.mean((probabilities["weak_success_probability"] - weak_ok.astype(float)) ** 2))}
    bundle = {"schema_version": 1, "artifact": artifact.name, "artifact_sha256": sha256_file(artifact),
              "strong": pair[0], "weak": pair[1], "policy_mode": "guarded" if chosen else "fixed_strong",
              "max_harm": chosen["max_harm"] if chosen else .01,
              "min_weak_success": chosen["min_weak_success"] if chosen else .9,
              "scope": "reused historical development; no production or new-model validation", "independent_validation": False,
              "cost_unit": "historical_cost",
              "model_identity_status": "historical_pair_only_not_validated_for_new_models"}
    write_json(output / "bundle.json", bundle)
    latency = []
    for row_id in ids["validation"][:config.get("latency_sample_size", 60)]:
        tick = time.perf_counter()
        if chosen:
            p, c = router.predict_tables(inputs.loc[[row_id], ["prompt_text"]])
            guarded_routes(p, c, pair, chosen["max_harm"], chosen["min_weak_success"])
        else:
            str(pair[0])
        latency.append(time.perf_counter() - tick)
    for path, digest in frozen["source_files_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("El código cambió durante entrenamiento; conserva el snapshot y no declares finalizada esta ejecución.")
    report = {"experiment_id": config["experiment_id"], "configuration": config,
              "configuration_sha256": frozen["configuration_sha256"], "partition_counts": counts,
              "source_prepared_files_sha256": manifest["files_sha256"], "quality_floor": quality_floor,
              "fixed_models": {name: result["metrics"] for name, result in fixed.items()},
              "routellm_bert": {"curve": bert_curve, "selected_point": bert_point}, "curve": curve,
              "selected_point": chosen, "selected_metrics": observed["metrics"], "policy_mode": bundle["policy_mode"],
              "development_gate_passed": chosen is not None, "quality_floor_met": observed["metrics"]["success_rate"] >= quality_floor,
              "comparisons": comparisons, "additional_controls": extra_controls, "routing_diagnostics": diagnostics,
              "training_seconds": training_seconds, "batch_inference_seconds": batch_seconds,
              "single_request_latency": {"n": len(latency), "p50_seconds": float(np.median(latency)) if latency else None,
                                         "p95_seconds": float(np.quantile(latency, .95)) if latency else None,
                                         "scope": "resident selected policy; fixed fallback skips predictor; no generation or serving SLA"},
              "outer_test_accessed": False, "independent_validation": False, "confirmation_of_superiority": False,
              "production_validation": False, "automatic_live_promotion": False, "paid_api_calls": False,
              "limitations": ["Validation reused across v2-v6; selecting on these means cannot establish independent superiority.",
                              "RouteLLM BERT uses published external supervision; Relay uses local outcomes; not an equal-supervision comparison and not RouteLLM MF.",
                              "The coarse 16-point BERT grid includes 0.5; equal point counts do not equalize tuning or supervision. The prior 101-point search is retained as a descriptive control.",
                              "Historical selected-call costs exclude router compute, training, retries and current pricing.",
                              "Quality constraints apply to observed development means, not guaranteed per-request probabilities or population quality.",
                              "The historical candidate identities cannot be relabeled as current local/API models.",
                              "A fixed fallback may fail the higher BERT quality floor; it is not a successful v6 result."],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/relay_v6.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(config.get("native_threads", 4)):
        result = run_v6(ROOT, config, args.output or ROOT / config["output_dir"])
    print(json.dumps({name: result[name] for name in ("policy_mode", "development_gate_passed", "selected_point", "selected_metrics")}, indent=2))


if __name__ == "__main__":
    main()
