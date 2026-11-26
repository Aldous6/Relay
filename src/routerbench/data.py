"""Local ingestion with pinned provenance, schema checks and prompt groups."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import normalize_for_dedup, parse_prompt
from .labels import outcome_tables

OFFICIAL_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_verified_pickle(path: str | Path, expected_sha256: str = OFFICIAL_SHA256) -> pd.DataFrame:
    """Deserialize only this verified research artifact, never arbitrary pickle."""
    actual = sha256_file(path)
    if actual != expected_sha256 or expected_sha256 != OFFICIAL_SHA256:
        raise ValueError(f"SHA-256 no autorizado: {actual}. No se abrió el pickle.")
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("El archivo no contiene un DataFrame.")
    required = {"sample_id", "prompt", "eval_name"}
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("Esquema inválido: faltan identificadores o hay columnas repetidas.")
    return frame


def prompt_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    texts = frame["prompt"].map(parse_prompt)
    normalized = texts.map(normalize_for_dedup)
    if normalized.eq("").any():
        raise ValueError("Hay prompts vacíos después de reconstruir el texto.")
    return pd.DataFrame({
        "source_row": np.arange(len(frame)),
        "sample_id": frame["sample_id"].astype(str).to_numpy(),
        "eval_name": frame["eval_name"].astype(str).to_numpy(),
        "prompt_text": texts.to_numpy(),
        "prompt_group": normalized.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()).to_numpy(),
    })


def hash_partition(groups: pd.Series, seed: int = 42, test_fraction: float = 0.2) -> pd.Series:
    """Assign groups without inspecting costs, scores, labels or benchmark names."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction debe estar entre 0 y 1.")
    cutoff = int(test_fraction * 2**256)
    return groups.map(lambda group: "test" if int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) < cutoff else "train")


def numeric_results(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    scores, costs = outcome_tables(frame)
    return list(scores.columns), scores, costs


def audit_frame(frame: pd.DataFrame, thresholds: list[float]) -> dict:
    if not thresholds or any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Los umbrales de auditoría deben ser finitos y estar en [0,1].")
    models, scores, costs = numeric_results(frame)
    metadata = prompt_metadata(frame)
    duplicate_mask = metadata["prompt_group"].duplicated(keep=False)
    conflict_groups = 0
    for _, indices in metadata.loc[duplicate_mask].groupby("prompt_group").groups.items():
        if pd.concat([scores.iloc[indices], costs.iloc[indices]], axis=1).drop_duplicates().shape[0] > 1:
            conflict_groups += 1
    unique_rows = ~metadata["prompt_group"].duplicated()
    return {
        "rows": len(frame), "columns": len(frame.columns), "models_count": len(models),
        "models": models, "evaluations": int(frame["eval_name"].nunique()),
        "empty_prompts": int(metadata["prompt_text"].str.strip().eq("").sum()),
        "null_scores": int(scores.isna().sum().sum()), "null_costs": int(costs.isna().sum().sum()),
        "duplicate_extra_rows": int((~unique_rows).sum()),
        "duplicate_groups_with_conflicting_results": conflict_groups,
        "rows_with_partial_score": int(((scores > 0) & (scores < 1)).any(axis=1).sum()),
        "thresholds": {
            str(threshold): {
                "ineligible_raw": int((~scores.ge(threshold).any(axis=1)).sum()),
                "eligible_raw": int(scores.ge(threshold).any(axis=1).sum()),
                "eligible_after_dedup": int((scores.ge(threshold).any(axis=1).to_numpy() & unique_rows.to_numpy()).sum()),
            } for threshold in thresholds
        },
        "median_estimated_cost_global_descriptive_only": {model: float(costs[model].median()) for model in models},
        "dedup_rule": "NFKC + casefold + whitespace collapse; first source row, independent of results",
    }


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
