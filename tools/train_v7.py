"""V7 similarity-conditioned regret: reused development, never outer-test evaluation."""
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
sys.path.insert(0, str(ROOT / "tools"))

from routerbench.comparison import paired_bootstrap
from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs
from routerbench.improvement import activate_full_fit
from routerbench.policy import evaluate_policy
from sweep_v3 import quantile_actions


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify(root, name, digest):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
        raise ValueError(f"Fuente modificada o fuera del proyecto: {name}")
    return path


def make_router(pair, parameters):
    from routerbench.neighborhood import NeighborhoodRegretRouter
    return NeighborhoodRegretRouter(strong=pair[0], weak=pair[1], **parameters)


def operating_score(probabilities, costs, pair):
    from routerbench.utility import cost_utility_score
    return cost_utility_score(probabilities, costs, pair[0], pair[1])


def route_one(probabilities, costs, pair, parameter):
    from routerbench.utility import choose_cost_utility
    return choose_cost_utility(probabilities, costs, pair[0], pair[1], parameter)


def named_actions(table, pair, index):
    if not isinstance(table, pd.DataFrame) or not table.index.equals(index) or not table.index.is_unique or not table.columns.is_unique or table.empty or table.isna().any().any():
        raise ValueError("Acciones de control inválidas o desalineadas.")
    if all(pd.api.types.is_bool_dtype(dtype) for dtype in table.dtypes):
        table = pd.DataFrame(np.where(table.to_numpy(), pair[0], pair[1]), index=table.index, columns=table.columns)
    if not table.isin(pair).all().all():
        raise ValueError("Las acciones no pertenecen al par histórico.")
    return table


def select_point(curve, floor):
    points = [point for point in curve if point["metrics"]["success_rate"] >= floor]
    return min(points, key=lambda p: (p["metrics"]["cost_per_1000"], -p["metrics"]["success_rate"], p["index"])) if points else None


def validate(config):
    if any(config.get(flag) is not False for flag in ("outer_test_evaluation_allowed", "automatic_live_promotion_allowed", "paid_api_calls_authorized")):
        raise ValueError("V7 solo permite desarrollo offline sin test ni promoción.")
    if config["selection"]["quality_loss_margin"] != 0 or config["quantiles"] != 101:
        raise ValueError("Conservar piso sin pérdida y 101 cuantiles.")
    if config["primary_candidate_id"] != "neighborhood_primary" or config["candidates"] != [
            {"id": "neighborhood_primary", "use_neighbors": True}, {"id": "global_control", "use_neighbors": False}]:
        raise ValueError("La primaria no se sustituye por la ablación global.")
    if config["learner"]["quality_threshold"] != config["quality_threshold"]:
        raise ValueError("Objetivo y umbral de evaluación incompatibles.")
    if set(config["controls"]) != {"bert101", "ensemble_v4", "v6"}:
        raise ValueError("Se requieren todos los controles predeclarados.")


def freeze(output, config, sources):
    paths = [Path(__file__).resolve(), ROOT / "tools/sweep_v3.py", ROOT / "configs/relay_v7.json",
             ROOT / "docs/relay_v7_protocolo.md", ROOT / "references/routellm_paper.json",
             *sorted((ROOT / "src").rglob("*.py"))]
    digests = {}
    with zipfile.ZipFile(output / "source_bundle.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative, content = path.relative_to(ROOT).as_posix(), path.read_bytes()
            digests[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(relative, content)
        archive.writestr("configuration.json", json.dumps(config, ensure_ascii=False, indent=2))
    result = {"configuration": config, "configuration_sha256": object_hash(config), "source_files_sha256": digests,
              "input_files_sha256": sources, "source_bundle_sha256": sha256_file(output / "source_bundle.zip"),
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "scipy", "pyarrow")},
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "outer_test_allowed": False}
    write_json(output / "development_started.json", result)
    return result


def run_v7(root, config, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    validate(config)
    if not output.is_relative_to(root):
        raise ValueError("La salida debe quedar en el proyecto.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("No se sobrescriben ejecuciones v7.")
    ref = config["reference"]
    sources = {config["reference_split_run"] + "/summary.json": config["reference_split_summary_sha256"],
               ref["selection_file"]: ref["selection_sha256"], ref["development_probabilities_file"]: ref["development_probabilities_sha256"]}
    for control in config["controls"].values():
        sources.update({control["directory"] + "/" + name: digest for name, digest in control["files_sha256"].items()})
    for name, digest in sources.items():
        verify(root, name, digest)
    reference = read(root / ref["selection_file"])
    previous = reference["configuration"]
    if (reference["test_outcomes_used_for_selection"] is not False
            or {role: previous["common_pair"][role] for role in ("strong", "weak")} != config["common_pair"]
            or previous["quality_score_threshold"] != config["quality_threshold"] or previous["dataset"]["sha256"] != config["dataset"]["sha256"]):
        raise ValueError("Referencia BERT incompatible con par, objetivo o dataset.")
    inputs, scores, costs, rows, manifest = load_development_inputs(root, config)
    rows = activate_full_fit(rows)
    ids = {part: rows.index[rows.development_split.eq(part)] for part in ("fit", "calibration", "validation")}
    counts = {part: len(index) for part, index in ids.items()}
    if counts != config["expected_population"] or min(counts.values()) < 2 or any(not frame.index.equals(rows.index) for frame in (inputs, scores, costs)):
        raise ValueError("Particiones de desarrollo o índices incompatibles.")
    pair = [config["common_pair"][role] for role in ("strong", "weak")]
    controls, prior_curves, control_summaries = {}, {}, {}
    for name, entry in config["controls"].items():
        directory = root / entry["directory"]
        summary = read(directory / entry["summary"])
        control_summaries[name] = summary
        if summary["source_prepared_files_sha256"] != manifest["files_sha256"] or summary["outer_test_accessed"] is not False:
            raise ValueError("El control no conserva preparación y separación de test.")
        control_pair = summary.get("common_pair", summary.get("configuration", {}).get("common_pair"))
        control_quality = summary.get("quality_score_threshold", summary.get("configuration", {}).get("quality_threshold"))
        if control_pair != config["common_pair"] or control_quality != config["quality_threshold"]:
            raise ValueError("Par u objetivo del control incompatible.")
        if name in ("bert101", "ensemble_v4"):
            controls[name] = named_actions(pd.read_parquet(directory / entry["actions"]), pair, ids["validation"])
            prior_curves[name] = read(directory / entry["curve"])["curve"]
            if len(prior_curves[name]) != controls[name].shape[1]:
                raise ValueError("Curvas y acciones del control no coinciden.")
        else:
            controls["relay_v6"] = named_actions(pd.read_parquet(directory / entry["routes"]), pair, ids["validation"])
            controls["bert16_v6"] = named_actions(pd.read_parquet(directory / entry["bert_routes"]), pair, ids["validation"])
            if any(table.shape[1] != 1 for table in (controls["relay_v6"], controls["bert16_v6"])):
                raise ValueError("Los controles congelados v6 requieren una política cada uno.")
    if control_summaries["bert101"]["source_reference_probabilities_sha256"] != ref["development_probabilities_sha256"]:
        raise ValueError("El control BERT usa otra caché.")
    sources.update({config["prepared_dir"] + "/" + name: digest for name, digest in manifest["files_sha256"].items()})
    sources[config["dataset"]["path"]] = config["dataset"]["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    frozen = freeze(output, config, sources)
    rows.to_parquet(output / "development_rows.parquet")
    models, predictions, candidate_actions, parameters, timings = {}, {}, {}, {}, {}
    for spec in config["candidates"]:
        name = spec["id"]
        print(f"V7 {name}: fit={counts['fit']}, calibración={counts['calibration']}, validación={counts['validation']}.", flush=True)
        model = make_router(pair, {**config["learner"], "use_neighbors": spec["use_neighbors"]})
        tick = time.perf_counter()
        model.fit(inputs.loc[ids["fit"]], scores.loc[ids["fit"], pair], costs.loc[ids["fit"], pair],
                  inputs.loc[ids["calibration"]], scores.loc[ids["calibration"], pair])
        training_seconds = time.perf_counter()-tick
        joblib.dump(model, output / (name + ".joblib"))
        tick = time.perf_counter()
        p, c = model.predict_tables(inputs.loc[ids["validation"]])
        inference_seconds = time.perf_counter()-tick
        if not p.index.equals(ids["validation"]) or not c.index.equals(ids["validation"]):
            raise ValueError("El candidato alteró las filas de validación.")
        p.to_parquet(output / (name + "_probabilities.parquet"))
        c.to_parquet(output / (name + "_predicted_costs.parquet"))
        ratio = operating_score(p, c, pair)
        cuts, actions = quantile_actions(ratio, config["quantiles"])
        actions = named_actions(actions, pair, ids["validation"])
        ratio.to_frame().to_parquet(output / (name + "_operating_scores.parquet"))
        actions.to_parquet(output / (name + "_all_routes.parquet"))
        models[name], predictions[name], candidate_actions[name], parameters[name] = model, (p, c), actions, cuts
        timings[name] = {"training_seconds": training_seconds, "batch_inference_seconds": inference_seconds}
    fixed_actions = pd.DataFrame({"strong": pair[0], "weak": pair[1]}, index=ids["validation"])
    fixed_actions.to_parquet(output / "fixed_routes.parquet")
    for name, table in controls.items():
        table.to_parquet(output / (name + "_all_routes.parquet"))
    action_names = ["fixed_routes.parquet", *[name + "_all_routes.parquet" for name in [*controls, *candidate_actions]]]
    write_json(output / "actions_frozen.json", {"configuration_sha256": frozen["configuration_sha256"],
               "selection_and_metrics_computed": False, "files_sha256": {name: sha256_file(output / name) for name in action_names}})
    # First evaluation of the new candidate decisions happens after every action is frozen.
    val_scores, val_costs = scores.loc[ids["validation"], pair], costs.loc[ids["validation"], pair]
    quality = config["quality_threshold"]
    def evaluate(selected):
        return evaluate_policy(selected, val_scores, val_costs, quality)
    fixed = {name: evaluate(fixed_actions[name]) for name in ("strong", "weak")}
    observed, control_reports = {}, {}
    for name in ("bert101", "ensemble_v4"):
        curve = [{**point, "index": index} for index, point in enumerate(prior_curves[name])]
        floor = fixed["strong"]["metrics"]["success_rate"] if name == "bert101" else quality_floor
        point = select_point(curve, floor)
        if point is None:
            observed[name] = fixed["strong"]
            selected = fixed_actions["strong"]
        else:
            selected = controls[name].iloc[:, point["index"]]
            observed[name] = evaluate(selected)
            if observed[name]["metrics"] != point["metrics"]:
                raise ValueError("El control no reproduce las métricas archivadas.")
        control_reports[name] = {"selected_point": point, "metrics": observed[name]["metrics"], "quality_floor": floor}
        selected.to_frame("selected_model").to_parquet(output / (name + "_selected_routes.parquet"))
        if name == "bert101":
            if point is None:
                raise ValueError("BERT101 debe contener el extremo fuerte fijo.")
            quality_floor = max(floor, observed[name]["metrics"]["success_rate"])
    for name, expected in (("relay_v6", control_summaries["v6"]["selected_metrics"]),
                           ("bert16_v6", control_summaries["v6"]["routellm_bert"]["selected_point"]["metrics"])):
        observed[name] = evaluate(controls[name].iloc[:, 0])
        if observed[name]["metrics"] != expected:
            raise ValueError("El control congelado v6 no reproduce sus métricas.")
        control_reports[name] = {"metrics": expected, "scope": "previously frozen v6 control, not reselected"}
    candidates = {}
    for spec in config["candidates"]:
        name, actions = spec["id"], candidate_actions[spec["id"]]
        curve = [{"index": index, "parameter": cut, "weak_fraction": float(actions.iloc[:, index].eq(pair[1]).mean()),
                  "metrics": evaluate(actions.iloc[:, index])["metrics"]} for index, cut in enumerate(parameters[name])]
        point = select_point(curve, quality_floor)
        selected = fixed_actions["strong"] if point is None else actions.iloc[:, point["index"]]
        observed[name] = evaluate(selected)
        selected.to_frame("selected_model").to_parquet(output / (name + "_selected_routes.parquet"))
        observed[name]["rows"].to_parquet(output / (name + "_observed_results.parquet"))
        weak = selected.eq(pair[1]); strong_ok, weak_ok = val_scores[pair[0]].ge(quality), val_scores[pair[1]].ge(quality)
        p, c = predictions[name]
        diagnostics = {"weak_routes": int(weak.sum()), "weak_fraction": float(weak.mean()),
                       "strong_only_on_weak_routes": int((weak & strong_ok & ~weak_ok).sum()),
                       "weak_only_on_weak_routes": int((weak & ~strong_ok & weak_ok).sum()),
                       "weak_successes_on_weak_routes": int((weak & weak_ok).sum()),
                       "expected_gain_mse": float(np.mean(((p.strong_only-p.weak_only)-(strong_ok.astype(int)-weak_ok.astype(int)))**2))}
        artifact = name + ".joblib"
        candidates[name] = {"specification": spec, "curve": curve, "selected_point": point, "selected_metrics": observed[name]["metrics"],
                            "artifact": artifact, "artifact_sha256": sha256_file(output / artifact),
                            "training_target_counts": getattr(models[name], "target_counts_", {}),
                            "calibration_notes": getattr(models[name], "calibration_notes_", {}),
                            "quality_floor_met": observed[name]["metrics"]["success_rate"] >= quality_floor,
                            "development_gate_passed": point is not None, "diagnostics": diagnostics, **timings[name]}
    primary_name = config["primary_candidate_id"]
    primary, point = candidates[primary_name], candidates[primary_name]["selected_point"]
    primary["beats_bert_cost_on_development"] = point is not None and primary["selected_metrics"]["cost_per_1000"] < observed["bert101"]["metrics"]["cost_per_1000"]
    comparisons = {name: paired_bootstrap(observed[primary_name]["rows"], reference["rows"], config["bootstrap_replicates"], config["seed"])
                   for name, reference in {"fixed_strong": fixed["strong"], **{name: observed[name] for name in [*controls, "global_control"]}}.items()}
    bundle = {"schema_version": 1, "artifact": primary["artifact"], "artifact_sha256": primary["artifact_sha256"],
              "strong": pair[0], "weak": pair[1], "policy_mode": "pairwise_utility" if point else "fixed_strong",
              "tradeoff": point["parameter"] if point else None, "cost_unit": "historical_cost", "independent_validation": False,
              "scope": "reused historical development; no production or new-model validation"}
    write_json(output / "bundle.json", bundle)
    latency = []
    for row_id in ids["validation"][:config["latency_sample_size"]]:
        tick = time.perf_counter()
        if point:
            p, c = models[primary_name].predict_tables(inputs.loc[[row_id], ["prompt_text"]])
            route_one(p, c, pair, point["parameter"])
        latency.append(time.perf_counter()-tick)
    for name, digest in frozen["source_files_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Una fuente cambió durante entrenamiento; preservar snapshot y no declarar finalizado.")
    result = {"experiment_id": config["experiment_id"], "configuration": config, "configuration_sha256": frozen["configuration_sha256"],
              "partition_counts": counts, "source_prepared_files_sha256": manifest["files_sha256"], "quality_floor": quality_floor,
              "fixed_models": {name: value["metrics"] for name, value in fixed.items()}, "controls": control_reports,
              "candidates": candidates, "primary_candidate_id": primary_name, "selected_point": point,
              "selected_metrics": primary["selected_metrics"], "policy_mode": bundle["policy_mode"],
              "development_gate_passed": primary["development_gate_passed"], "beats_bert_cost_on_development": primary["beats_bert_cost_on_development"],
              "comparisons": comparisons, "single_request_latency": {"n": len(latency), "p50_seconds": float(np.median(latency)) if latency else None,
              "p95_seconds": float(np.quantile(latency, .95)) if latency else None, "scope": "resident complete text representation, uncached neighbor search, model and policy; excludes loading/generation; no service SLA"},
              "outer_test_accessed": False, "independent_validation": False, "confirmation_of_superiority": False,
              "production_validation": False, "automatic_live_promotion": False, "paid_api_calls": False,
              "limitations": ["Repeated development selection across v2-v7; bootstrap intervals are post-selection exploratory diagnostics.",
                              "Local similarity outcomes and historical lexical features are not the official RouteLLM similarity-weighted Bradley-Terry method.",
                              "BERT has external supervision, Relay local outcomes; this does not isolate algorithm under equal supervision and does not evaluate MF.",
                              "Historical selected-call cost excludes router overhead, training, retries and current prices.",
                              "The fixed global ablation is not promoted over the predeclared neighborhood primary.",
                              "The stored fit neighbor matrix is an artifact of development data; it needs provenance/licensing review before distribution."],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "files_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/relay_v7.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = read(args.config)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(config["native_threads"]):
        result = run_v7(ROOT, config, args.output or ROOT / config["output_dir"])
    print(json.dumps({name: result[name] for name in ("primary_candidate_id", "selected_metrics", "development_gate_passed", "beats_bert_cost_on_development")}, indent=2))


if __name__ == "__main__":
    main()
