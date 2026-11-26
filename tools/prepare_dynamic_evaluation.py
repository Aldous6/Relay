"""Freeze an open-topic evaluation with held-out task families and sources.

This command only validates supplied data/policies and writes provenance. It
does not infer family labels, train a router, start an app or generate answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.data import sha256_file
from routerbench.features import normalize_for_dedup
from routerbench.prospective import _hash, _read_cases, _write, prepare_prospective

SPLITS = ("fit", "calibration", "evaluation")


def inspect_partitions(datasets, metadata):
    """Family/source labels are audit-only metadata and never predictor text."""
    if set(datasets) != set(SPLITS) or set(metadata) != set(SPLITS):
        raise ValueError("Se requieren fit, calibration y evaluation.")
    seen_prompts, seen_sources = set(), set()
    families, counts = {}, {}
    development_groups = set()
    for split in SPLITS:
        rows = datasets[split]
        annotations = metadata[split]
        if not rows or len(rows) != len(annotations):
            raise ValueError("Cada caso requiere una anotación de procedencia.")
        by_id = {}
        for item in annotations:
            if (not isinstance(item, dict) or set(item) != {"id", "task_family", "source_group"} or
                    any(not isinstance(value, str) or not value.strip() for value in item.values()) or item["id"] in by_id):
                raise ValueError("Anotación inválida: id, task_family, source_group únicos por partición.")
            by_id[item["id"]] = item
        if len({row["id"] for row in rows}) != len(rows) or set(by_id) != {row["id"] for row in rows}:
            raise ValueError("Las anotaciones no corresponden a los casos.")
        groups = {hashlib.sha256(normalize_for_dedup(row["prompt"]).encode()).hexdigest() for row in rows}
        if len(groups) != len(rows) or groups & seen_prompts:
            raise ValueError("Hay prompts normalizados duplicados dentro o entre particiones.")
        source_groups = {item["source_group"] for item in annotations}
        if source_groups & seen_sources:
            raise ValueError("Un documento o conversación fuente cruza particiones.")
        seen_prompts.update(groups)
        seen_sources.update(source_groups)
        if split != "evaluation":
            development_groups.update(groups)
        families[split] = {item["task_family"] for item in annotations}
        counts[split] = {"cases": len(rows), "source_groups": len(source_groups),
                         "families": sorted(families[split]),
                         "cases_by_family": {family: sum(item["task_family"] == family for item in annotations)
                                             for family in sorted(families[split])}}
    if families["evaluation"] & (families["fit"] | families["calibration"]):
        raise ValueError("Esta prueba exige familias de evaluación ausentes de fit y calibración.")
    return counts, development_groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("La salida debe estar dentro del proyecto.")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (set(manifest) != {"partitions", "policy_manifest", "artifacts"} or
            set(manifest["partitions"]) != set(SPLITS) or
            not isinstance(manifest["policy_manifest"], dict) or
            manifest["policy_manifest"].get("paid_api_calls") is not False or
            not isinstance(manifest["artifacts"], list) or not manifest["artifacts"]):
        raise ValueError("Declarar particiones, políticas sin APIs pagadas y artefactos congelados.")
    if any(not isinstance(name, str) or not name.strip() for name in manifest["artifacts"]):
        raise ValueError("Los artefactos deben ser rutas explícitas.")

    def resolve(name):
        path = (manifest_path.parent / name).resolve()
        if not path.is_file():
            raise ValueError("Falta un archivo declarado en el manifiesto.")
        return path

    datasets, metadata, sources = {}, {}, {}
    for split, entry in manifest["partitions"].items():
        if set(entry) != {"cases", "metadata"}:
            raise ValueError("Cada partición declara cases y metadata.")
        cases_path, metadata_path = resolve(entry["cases"]), resolve(entry["metadata"])
        datasets[split], _, _ = _read_cases(cases_path)
        metadata[split] = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8-sig").splitlines()]
        sources[split] = {"cases_sha256": sha256_file(cases_path), "metadata_sha256": sha256_file(metadata_path)}
    counts, known = inspect_partitions(datasets, metadata)
    artifacts = {name: sha256_file(resolve(name)) for name in manifest["artifacts"]}
    policy = {"configuration": manifest["policy_manifest"], "manifest_sha256": sha256_file(manifest_path),
              "source_files_sha256": sources, "artifacts_sha256": artifacts,
              "preparer_sha256": sha256_file(Path(__file__).resolve()),
              "partition_audit": counts,
              "family_scope": "evaluation task families absent from fit and calibration according to supplied annotations",
              "source_scope": "declared source_group disjoint across partitions",
              "independence_certified": False,
              "metadata_used_as_predictors": False, "production_validation": False}
    result = prepare_prospective(resolve(manifest["partitions"]["evaluation"]["cases"]), output,
                                 policy, known_prompt_groups=known, collect_both_models=True)
    audit = {"freeze_sha256": result["freeze_sha256"], "partition_audit": counts,
             "paid_api_calls": False, "generation_executed": False,
             "limits": "Checks supplied annotations and exact normalized overlap; does not establish label truth, semantic novelty, representative coverage or statistical power."}
    audit_path = output / "partition_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")) != audit:
        raise ValueError("La auditoría previa cambió; no se sobrescribe.")
    if not audit_path.exists():
        _write(audit_path, audit)
    print(json.dumps({"prepared": str(output), "audit_sha256": _hash(audit), "generation_executed": False}))


if __name__ == "__main__":
    main()
