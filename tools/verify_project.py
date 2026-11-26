"""Offline verification of tests, provenance and prepared artifact integrity."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.data import sha256_file, write_json
from routerbench.features import FEATURE_NAMES
from routerbench.labels import fit_tiers
from routerbench.data import load_verified_pickle


def main():
    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/processed/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(ROOT / "data/processed/features.parquet")
    rows = pd.read_parquet(ROOT / "data/processed/rows.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    assert tuple(features.columns) == FEATURE_NAMES
    assert features.shape == (len(rows), 14)
    assert len(labels) == len(rows) == manifest["deduplicated_rows"]
    assert np.isfinite(features.to_numpy()).all()
    assert rows["prompt_group"].is_unique
    assert set(rows["split"]) == {"train", "test"}
    assert labels["tier"].notna().equals(labels["eligible"])
    assert labels.loc[labels["eligible"], "tier"].isin([0, 1, 2]).all()
    assert not manifest["final_test_evaluated"]
    assert manifest["configuration"] == config
    actual_hashes = {name: sha256_file(ROOT / "data/processed" / name) for name in ("features.parquet", "rows.parquet", "labels.parquet")}
    assert manifest["files_sha256"] == actual_hashes
    raw = load_verified_pickle(ROOT / config["dataset"]["path"])
    train_positions = rows.loc[rows["split"].eq("train"), "source_row"].to_numpy()
    assert fit_tiers(raw.iloc[train_positions], config["tier_sizes"]) == manifest["tier_map"]

    syntax_files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) + list((ROOT / "tools").glob("*.py"))
    for path in syntax_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_result = unittest.TextTestRunner(verbosity=1).run(tests)
    assert test_result.wasSuccessful(), "Fallaron pruebas unitarias."

    notebook = json.loads((ROOT / "notebooks/01_auditoria.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    notebook_output = io.StringIO()
    executed_cells = 0
    with contextlib.redirect_stdout(notebook_output):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"01_auditoria.ipynb:cell{index}", "exec"), namespace)
                executed_cells += 1
    skill_report = json.loads((ROOT / "reports/skills-installed.json").read_text(encoding="utf-8"))
    for skill in skill_report["skills"]:
        for name, checksum in skill["files_sha256"].items():
            assert sha256_file(ROOT / skill["local_path"] / name) == checksum
    final_report_path = ROOT / "reports/final/summary.json"
    final_evaluated = False
    if final_report_path.is_file():
        final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        final_evaluated = final_report["outer_test_performance_evaluated"]
    write_json(ROOT / "reports/verification.json", {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "unit_tests_passed": test_result.testsRun, "python_syntax_files": len(syntax_files),
        "prepared_rows": len(rows), "feature_count": 14, "dataset_sha256_verified": True,
        "tiers_recomputed_from_training_only": True, "parquet_sha256_verified": True,
        "notebook_code_cells_executed_in_order": executed_cells,
        "notebook_execution_method": "standard Python exec, no IPython magics or kernel required",
        "notebook_output": notebook_output.getvalue(),
        "installed_skills_hashes_verified": [skill["name"] for skill in skill_report["skills"]],
        "final_test_performance_evaluated": final_evaluated,
        "application_started_by_verifier": False,
        "live_integration_evidence": "reports/live_smoke.json" if (ROOT / "reports/live_smoke.json").is_file() else None,
    })
    print("Verificación completa. reports/verification.json")


if __name__ == "__main__":
    main()
