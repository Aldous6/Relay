"""One frozen held-out evaluation; no refitting or policy search on test."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .comparison import paired_bootstrap
from .data import load_verified_pickle, sha256_file, write_json
from .development import ROOT, calibration_diagnostics
from .features import FEATURE_NAMES, parse_prompt
from .labels import outcome_tables
from .policy import choose_models, evaluate_policy


def cluster_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame, groups: pd.Series,
                      replicates: int = 5000, seed: int = 42) -> dict:
    if not candidate.index.equals(reference.index) or not candidate.index.equals(groups.index) or groups.isna().any():
        raise ValueError("Filas y grupos desalineados.")
    differences = candidate[["success", "cost"]].astype(float) - reference[["success", "cost"]].astype(float)
    differences["group"] = groups
    sums = differences.groupby("group")[["success", "cost"]].sum()
    counts = groups.value_counts().reindex(sums.index).to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(sums), size=(replicates, len(sums)))
    delta = sums.to_numpy()[samples].sum(axis=1) / counts[samples].sum(axis=1, keepdims=True)
    return {"resampling_unit": "eval_name", "groups": len(sums), "replicates": replicates, "seed": seed,
            "success_rate_delta_95ci": np.quantile(delta[:, 0], [0.025, 0.975]).tolist(),
            "cost_per_1000_delta_95ci": (1000*np.quantile(delta[:, 1], [0.025, 0.975])).tolist(),
            "scope": "secondary sensitivity; eval_name groups need not be independent benchmark families"}


def assess_primary(comparison: dict, margin: float) -> dict:
    lower, upper = comparison["success_rate_delta_95ci"]
    quality = lower > -margin
    cost = comparison["cost_per_1000_delta_95ci"][1] < 0
    return {"quality_loss_margin": margin, "quality_noninferiority_supported": quality,
            "lower_historical_call_cost_supported": cost, "both_primary_criteria_met": quality and cost,
            "quality_status": "supported_within_declared_margin" if quality else "inferior_beyond_margin" if upper < -margin else "inconclusive",
            "interpretation_scope": "historical held-out RouterBench calls; no net production savings or RouteLLM superiority claim"}


def run_final(protocol_path: Path, output: Path, root: Path = ROOT) -> dict:
    protocol_hash = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_outer_test":
        raise ValueError("Falta un protocolo congelado antes del test.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("La evaluación final ya se inició. Conservar artefactos y no volver a buscar políticas en test.")
    prepared = root / protocol["prepared_dir"]
    for name, checksum in protocol["prepared_files_sha256"].items():
        if sha256_file(prepared / name) != checksum:
            raise ValueError(f"Se modificó la preparación: {name}.")
    for policy in protocol["policies"]:
        run_dir = root / policy["run_dir"]
        if sha256_file(run_dir / "summary.json") != policy["source_summary_sha256"] or sha256_file(run_dir / policy["artifact"]) != policy["artifact_sha256"]:
            raise ValueError(f"Se modificó una política congelada: {policy['id']}.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation_started.json", {"protocol_sha256": protocol_hash,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "policy_search_allowed": False})
    # No test outcomes are inspected before the above persisted freeze check.
    rows = pd.read_parquet(prepared / "rows.parquet")
    features = pd.read_parquet(prepared / "features.parquet")
    mask = rows["split"].eq("test")
    metadata = rows.loc[mask].copy()
    if len(metadata) != protocol["expected_outer_test"]["all_prompts"] or tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("El test no coincide con la población fijada.")
    source_ids = pd.Index(metadata["source_row"].to_numpy(), name="source_row")
    metadata.index = source_ids
    raw = load_verified_pickle(root / protocol["dataset"]["path"], protocol["dataset"]["sha256"])
    heldout = raw.iloc[source_ids.to_numpy()].copy()
    heldout.index = source_ids
    inputs = features.loc[mask].copy()
    inputs.index = source_ids
    inputs.insert(0, "prompt_text", heldout["prompt"].map(parse_prompt))
    metadata.to_parquet(output / "test_rows.parquet", index=False)
    predictions = {}
    for policy in protocol["policies"]:
        print(f"Prediciendo test congelado: {policy['id']} ({len(inputs)} prompts).", flush=True)
        router = joblib.load(root / policy["run_dir"] / policy["artifact"])
        tick = time.perf_counter()
        probabilities, predicted_costs = router.predict_tables(inputs)
        routes = choose_models(probabilities, predicted_costs, policy["minimum_predicted_success"])
        routes.to_parquet(output / f"{policy['id']}_routes.parquet", index=True)
        probabilities.to_parquet(output / f"{policy['id']}_probabilities.parquet", index=True)
        predicted_costs.to_parquet(output / f"{policy['id']}_predicted_costs.parquet", index=True)
        predictions[policy["id"]] = {"policy": policy, "probabilities": probabilities,
                                      "routes": routes, "inference_seconds": time.perf_counter()-tick}
        if hasattr(router.transformer_, "encoder_"):
            pd.DataFrame(router.transformer_.encoder_.last_stats_, index=source_ids).to_parquet(output / f"{policy['id']}_coverage.parquet", index=True)
    # Freeze all decisions first. The following scores/costs are for evaluation,
    # and cannot change the already-saved actions.
    scores, costs = outcome_tables(heldout)
    reference_model = protocol["reference"]["fixed_model"]
    reference = evaluate_policy(pd.Series(reference_model, index=source_ids), scores, costs, protocol["quality_score_threshold"])
    reference["rows"].to_parquet(output / "fixed_reference_observed_results.parquet", index=True)
    report = {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
              "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "n_test": len(inputs),
              "outer_test_performance_evaluated": True, "policy_selection_used_test": False,
              "routellm_official_evaluated": False, "reference_model": reference_model,
              "fixed_reference": reference["metrics"], "policies": {},
              "all_fixed_models": {model: evaluate_policy(pd.Series(model, index=source_ids), scores, costs, protocol["quality_score_threshold"])["metrics"] for model in scores},
              "primary_policy_id": next(policy["id"] for policy in protocol["policies"] if policy["role"] == "primary_confirmatory_comparison"),
              "claims_scope": protocol["estimation"]["cost_scope"]}
    for policy_id, prediction in predictions.items():
        evaluation = evaluate_policy(prediction["routes"]["selected_model"], scores, costs, protocol["quality_score_threshold"])
        evaluation["rows"].to_parquet(output / f"{policy_id}_observed_results.parquet", index=True)
        comparison = paired_bootstrap(evaluation["rows"], reference["rows"], protocol["paired_uncertainty"]["replicates"], protocol["paired_uncertainty"]["seed"])
        comparison["scope"] = "held-out paired prompt bootstrap; primary or secondary status is declared by policy role; historical selected-call cost excludes router compute and retries"
        by_eval = {}
        for group, indices in metadata.groupby("eval_name").groups.items():
            subset = evaluation["rows"].loc[indices]
            by_eval[str(group)] = {"n": len(subset), "success_rate": float(subset["success"].mean()), "cost_per_1000": float(subset["cost"].mean()*1000)}
        length_group = pd.cut(inputs["n_chars"], [-1, 256, 1024, 4096, np.inf], labels=["0-256", "257-1024", "1025-4096", ">4096"])
        by_length = {}
        for group in length_group.dropna().unique():
            subset = evaluation["rows"].loc[length_group.eq(group)]
            by_length[str(group)] = {"unit": "characters", "n": len(subset), "success_rate": float(subset["success"].mean()), "cost_per_1000": float(subset["cost"].mean()*1000)}
        report["policies"][policy_id] = {"variant": prediction["policy"]["variant"], "role": prediction["policy"]["role"],
              "metrics": evaluation["metrics"], "comparison_vs_fixed": comparison,
              "cluster_sensitivity": cluster_bootstrap(evaluation["rows"], reference["rows"], metadata["eval_name"], protocol["paired_uncertainty"]["replicates"], protocol["paired_uncertainty"]["seed"]),
              "calibration_diagnostics": calibration_diagnostics(prediction["probabilities"], scores, protocol["quality_score_threshold"]),
              "fallback_rate": float(prediction["routes"]["decision"].eq("fallback").mean()),
              "selected_model_counts": prediction["routes"]["selected_model"].value_counts().to_dict(),
              "batch_prediction_seconds": prediction["inference_seconds"], "by_eval_name": by_eval, "by_length": by_length}
    primary = report["policies"][report["primary_policy_id"]]
    report["primary_conclusion"] = assess_primary(primary["comparison_vs_fixed"], protocol["selection"]["quality_loss_margin"])
    report["files_sha256"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "summary.json", report)
    print(json.dumps({"primary": primary["metrics"], "conclusion": report["primary_conclusion"]}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/final_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final")
    args = parser.parse_args(argv)
    run_final(args.protocol, args.output)


if __name__ == "__main__":
    main()
