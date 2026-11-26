"""Verify saved semantic results and perform a local long-text smoke check.

This checks software coverage and serialization, not answer quality on new text.
It never opens outcomes from the reserved outer test and makes no network calls.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routerbench.data import sha256_file, write_json
from routerbench.recommend import recommend
from routerbench.semantic_encoder import WindowedE5Encoder


def main():
    directory = ROOT / "reports/relay-semantic-development-v1"
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    source_directory = ROOT / report["configuration"]["reference_split_run"]
    source_report = json.loads((source_directory / "summary.json").read_text(encoding="utf-8"))
    for name, checksum in report["files_sha256"].items():
        assert sha256_file(directory / name) == checksum, name
    current_rows = pd.read_parquet(directory / "development_rows.parquet").sort_values("source_row").reset_index(drop=True)
    original_rows = pd.read_parquet(source_directory / "development_rows.parquet").sort_values("source_row").reset_index(drop=True)
    pd.testing.assert_frame_equal(current_rows, original_rows)
    assert report["source_prepared_files_sha256"] == source_report["source_prepared_files_sha256"]
    assert report["outer_test_performance_evaluated"] is False
    assert report["routellm_official_evaluated"] is False
    for variant in report["variants"]:
        coverage = pd.read_parquet(directory / f"{variant}_coverage.parquet")
        assert coverage["truncated_tokens"].eq(0).all()
        assert coverage["input_tokens"].equals(coverage["covered_tokens"])
        assert coverage["n_windows"].ge(1).all()
        probabilities = pd.read_parquet(directory / f"{variant}_probabilities.parquet")
        assert coverage.index.equals(probabilities.index)
        artifact = directory / report["variants"][variant]["artifact"]
        router = joblib.load(artifact)
        assert not hasattr(router.transformer_, "encoder_")
    spec = json.loads((ROOT / "configs/semantic_encoder.json").read_text(encoding="utf-8"))
    encoder = WindowedE5Encoder(spec)
    long_prompt = ("Document section: compare the evidence and assumptions across the sources.\n" * 900
                   + "Final instruction: identify contradictions and cite the relevant sections.")
    started = time.perf_counter()
    embedding = encoder.encode([long_prompt], use_cache=False)
    long_stats = encoder.last_stats_[0]
    assert long_stats["input_tokens"] > 8192
    assert long_stats["n_windows"] > 1
    assert long_stats["truncated_tokens"] == 0
    assert long_stats["input_tokens"] == long_stats["covered_tokens"]
    assert embedding.shape == (1, 384) and np.isfinite(embedding).all()
    assert np.isclose(np.linalg.norm(embedding[0]), 1.0)
    long_elapsed = time.perf_counter() - started
    short = "<s> literal boundary. Compara la evidencia y explica tus supuestos."
    first = encoder.encode([short], use_cache=False)
    second = encoder.encode([short], use_cache=False)
    assert np.array_equal(first, second)
    # Exact model input parity on a single-window prompt, including a literal
    # special-token boundary that previously revealed an assembly error.
    tokenizer = encoder._tokenizer
    full_ids = tokenizer.encode(spec["prefix"] + short, add_special_tokens=False).ids
    payload = full_ids[len(encoder._prefix_ids):]
    assembled = [encoder._cls, *encoder._prefix_ids, *payload, encoder._sep]
    reference_ids = tokenizer.encode(spec["prefix"] + short, add_special_tokens=True).ids
    assert assembled == reference_ids
    recommendation = recommend(directory, "semantic", "Explain how experimental evidence can support a hypothesis.")
    assert recommendation["llm_called"] is False
    assert recommendation["selected_model"] in report["fixed_model_baselines"]
    result = {
        "python": platform.python_version(), "saved_artifact_hashes_verified": True,
        "exact_reference_partitions_preserved": True, "validation_full_coverage_verified": True,
        "serialization_excludes_runtime": True,
        "long_text_software_test": {**long_stats, "elapsed_seconds": long_elapsed,
                                    "embedding_shape": list(embedding.shape),
                                    "scope": "synthetic coverage smoke check, not measured LLM answer quality"},
        "single_window_tokenization_matches_reference": True,
        "same_text_uncached_repeat_bitwise_equal_in_smoke_check": True,
        "model_recommendation": recommendation,
        "outer_test_performance_evaluated": False, "routellm_evaluated": False,
    }
    write_json(directory / "native_integration.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
