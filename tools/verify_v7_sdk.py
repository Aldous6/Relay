"""Offline SDK replay of frozen v7 and v6 validation predictions.

This checks serialization and harness integration, not routing quality. It reads
only the original development prompts through the SHA-verifying train loader;
it neither evaluates the outer test nor calls a model provider.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay_router import load_bundle
from routerbench.data import sha256_file, write_json
from routerbench.development import load_development_inputs

ATOL, RTOL = 1e-8, 1e-7
SAMPLE_SIZE = 12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def object_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def checked_file(directory, name, digest):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
        raise ValueError(f"Archivo modificado o fuera de su directorio: {name}")
    return path


def checked_report(directory):
    summary = read(directory / "summary.json")
    if summary.get("outer_test_accessed") is not False:
        raise ValueError("El replay exige una ejecucion sin acceso al test exterior.")
    for name, digest in summary["files_sha256"].items():
        checked_file(directory, name, digest)
    return summary


def numeric_difference(actual, expected, label):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Valores o dimensiones invalidos: {label}")
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=label)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.


def checked_table(path, validation_ids):
    frame = pd.read_parquet(path)
    if not frame.index.is_unique or not frame.columns.is_unique or not frame.index.equals(validation_ids):
        raise ValueError(f"La tabla no conserva las filas originales de validacion: {path.name}")
    return frame


def replay_sample(v7_directory, v7, v6_directory, validation_ids):
    """Cover SDK branches using frozen actions, without inspecting outcomes."""
    sample = list(validation_ids[:SAMPLE_SIZE])
    selection = {"initial_source_rows": [int(value) for value in sample], "coverage": [], "near_cutoff": None}
    primary = v7["primary_candidate_id"]
    for version, directory, filename in (("v7", v7_directory, primary + "_selected_routes.parquet"),
                                         ("v6", v6_directory, "selected_routes.parquet")):
        routes = checked_table(directory / filename, validation_ids).selected_model
        for model in dict.fromkeys(routes.astype(str)):
            represented = routes.loc[sample].eq(model)
            added = not represented.any()
            row_id = routes.index[routes.eq(model)][0] if added else represented.index[represented][0]
            if added:
                sample.append(row_id)
            selection["coverage"].append({"version": version, "selected_model": model,
                                          "source_row": int(row_id), "added": bool(added)})
    bundle = read(v7_directory / "bundle.json")
    if bundle["policy_mode"] == "pairwise_utility":
        scores = checked_table(v7_directory / (primary + "_operating_scores.parquet"), validation_ids).operating_score
        row_id = (scores - bundle["tradeoff"]).abs().idxmin()
        added = row_id not in sample
        if added:
            sample.append(row_id)
        selection["near_cutoff"] = {"version": "v7", "source_row": int(row_id), "added": bool(added),
                                     "absolute_score_distance": float(abs(scores.loc[row_id] - bundle["tradeoff"]))}
    return pd.Index(sample, name=validation_ids.name), selection


def replay(directory, summary, prompts, validation_ids, *, version):
    bundle = read(directory / "bundle.json")
    primary = summary["primary_candidate_id"] if version == "v7" else None
    names = ({"probabilities": primary + "_probabilities.parquet",
              "costs": primary + "_predicted_costs.parquet",
              "routes": primary + "_selected_routes.parquet",
              "operating_scores": primary + "_operating_scores.parquet"}
             if primary else
             {"probabilities": "validation_probabilities.parquet",
              "costs": "validation_predicted_costs.parquet", "routes": "selected_routes.parquet"})
    required = {"bundle.json", bundle["artifact"], "development_rows.parquet", *names.values()}
    if not required.issubset(summary["files_sha256"]):
        raise ValueError("El manifiesto no registra todos los archivos utilizados por el replay.")
    tables = {name: checked_table(directory / path, validation_ids).loc[prompts.index]
              for name, path in names.items()}
    rows = pd.read_parquet(directory / "development_rows.parquet")
    if "source_row" in rows.columns:
        rows = rows.set_index("source_row")
    saved_validation = rows.index[rows.development_split.eq("validation")]
    if not saved_validation.equals(validation_ids):
        raise ValueError("La ejecucion no conserva la particion de validacion del loader.")
    # Exercise both documented load entry points without importing provider SDKs.
    router = load_bundle(directory if version == "v7" else directory / "bundle.json")
    if router.model_catalog != {role: bundle[role] for role in ("strong", "weak")}:
        raise ValueError("El SDK altero las identidades historicas del bundle.")
    probabilities, costs = router.decider.predict_tables(prompts)
    differences = {}
    for label, actual in (("probabilities", probabilities), ("costs", costs)):
        expected = tables[label]
        if not actual.index.equals(expected.index) or not actual.columns.equals(expected.columns):
            raise ValueError(f"La prediccion cambio el contrato de filas o columnas: {label}")
        differences[label] = numeric_difference(actual, expected, version + ":" + label)
    differences.update({"selected_cost": 0., "harm_probability": 0.,
                        "weak_success_probability": 0., "expected_quality_gain": 0., "operating_score": 0.})
    cases = []
    for row_id, prompt in prompts.prompt_text.items():
        decision = router.route(prompt)
        expected_model = str(tables["routes"].loc[row_id, "selected_model"])
        if decision.selected_model != expected_model:
            raise AssertionError(f"Ruta SDK distinta de la congelada: {version}/{row_id}")
        if (decision.policy_mode != bundle["policy_mode"] or decision.cost_unit != bundle["cost_unit"]
                or decision.evaluation_scope != bundle["scope"] or decision.router_seconds < 0):
            raise AssertionError("El SDK altero metadatos del bundle.")
        expected_values = {}
        if router.policy_mode != "fixed_strong":
            expected_values["selected_cost"] = (decision.predicted_cost, tables["costs"].loc[row_id, expected_model])
        if router.policy_mode == "guarded":
            expected_values.update({name: (getattr(decision, name), tables["probabilities"].loc[row_id, name])
                                    for name in ("harm_probability", "weak_success_probability")})
        elif router.policy_mode == "pairwise_utility":
            expected_values["expected_quality_gain"] = (decision.expected_quality_gain,
                                                        tables["probabilities"].loc[row_id, "expected_quality_gain"])
            expected_values["operating_score"] = (decision.operating_score,
                                                  tables["operating_scores"].loc[row_id, "operating_score"])
        for name, (actual, expected) in expected_values.items():
            differences[name] = max(differences[name], numeric_difference(actual, expected, version + ":" + name))
        cases.append({"source_row": int(row_id), "selected_model": decision.selected_model, "matches_archive": True})
    calls = []
    first_prompt = prompts.prompt_text.iloc[0]
    def generate_mock(model_name, unchanged_prompt):
        calls.append((model_name, unchanged_prompt))
        return {"offline_mock": True}
    response = router.run(first_prompt, generate_mock)
    first_model = str(tables["routes"].iloc[0]["selected_model"])
    if calls != [(first_model, first_prompt)] or response["response"] != {"offline_mock": True}:
        raise AssertionError("El harness debe invocar exactamente una vez el callback y preservar el prompt.")
    if response["decision"]["selected_model"] != first_model:
        raise AssertionError("La decision del callback no reproduce la ruta congelada.")
    return {"n": len(prompts), "policy_mode": router.policy_mode, "all_routes_match": True,
            "cost_unit": router.cost_unit, "evaluation_scope": router.evaluation_scope,
            "max_absolute_differences": differences, "mock_generation_calls": len(calls),
            "real_generation_calls": 0, "cases": cases,
            "report_summary_sha256": sha256_file(directory / "summary.json"),
            "verified_files_sha256": summary["files_sha256"]}


def run(root, v7_directory, v6_directory, output):
    root, v7_directory, v6_directory, output = (Path(path).resolve() for path in
                                               (root, v7_directory, v6_directory, output))
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("La salida debe ser nueva y quedar dentro del proyecto.")
    v7, v6 = checked_report(v7_directory), checked_report(v6_directory)
    config = v7["configuration"]
    started = read(v7_directory / "development_started.json")
    if object_hash(config) != v7["configuration_sha256"] or config != started["configuration"]:
        raise ValueError("La configuracion no coincide con la congelada antes de entrenar.")
    if v7["source_prepared_files_sha256"] != v6["source_prepared_files_sha256"]:
        raise ValueError("V6 y v7 no comparten la preparacion original.")
    for name, digest in started["source_files_sha256"].items():
        checked_file(root, name, digest)
    checked_file(root, config["reference_split_run"] + "/summary.json", config["reference_split_summary_sha256"])
    # This loader verifies the raw SHA before deserialization and slices to outer
    # train before inspecting any prompt/outcome. Outcomes are unused by replay.
    inputs, scores, observed_costs, rows, manifest = load_development_inputs(root, config)
    del scores, observed_costs
    if manifest["files_sha256"] != v7["source_prepared_files_sha256"]:
        raise ValueError("El loader no reprodujo los archivos preparados de la ejecucion.")
    validation_ids = rows.index[rows.development_split.eq("validation")]
    sample_ids, selection = replay_sample(v7_directory, v7, v6_directory, validation_ids)
    prompts = inputs.loc[sample_ids, ["prompt_text"]].copy()
    del inputs, rows
    if len(prompts) < SAMPLE_SIZE:
        raise ValueError("Se requieren al menos doce prompts originales de validacion.")
    results = {name: replay(directory, summary, prompts, validation_ids, version=name)
               for name, directory, summary in (("v7", v7_directory, v7), ("v6", v6_directory, v6))}
    # Ensure the evidence and frozen implementation stayed unchanged throughout.
    for directory, summary in ((v7_directory, v7), (v6_directory, v6)):
        for name, digest in summary["files_sha256"].items():
            checked_file(directory, name, digest)
    for name, digest in started["source_files_sha256"].items():
        checked_file(root, name, digest)
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "offline archive/SDK replay; not quality evaluation or independent validation",
              "sample_rule": "first 12 validation rows; append first archived row for each missing selected model in v7 then v6; append v7 row closest to frozen cutoff; no outcomes used",
              "sampling": selection,
              "tolerance": {"absolute": ATOL, "relative": RTOL}, "versions": results,
              "dataset_sha256": config["dataset"]["sha256"], "outer_test_accessed": False,
              "paid_api_calls": False, "real_generation_calls": 0,
              "verifier_sha256": sha256_file(Path(__file__)),
              "current_sdk_sha256": sha256_file(root / "src/relay_router/client.py")}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", type=Path, default=ROOT / "reports/relay-v7-development-v1")
    parser.add_argument("--v6", type=Path, default=ROOT / "reports/relay-v6-development-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v7-package/sdk_replay.json")
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        result = run(ROOT, args.v7, args.v6, args.output)
    print(json.dumps({version: {key: report[key] for key in ("n", "all_routes_match", "max_absolute_differences")}
                      for version, report in result["versions"].items()}, indent=2))


if __name__ == "__main__":
    main()
