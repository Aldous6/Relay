"""Explicit callback evaluation of an external JSONL, with durable routing.

No provider, network client or LLM is imported. One writer may use an output
directory. Gold references never enter routing or generation callbacks. Grades
are reported separately; they are not RouterBench scores or a superiority test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .features import normalize_for_dedup

POLICIES = ("candidate", "reference")
MODELS = {"strong", "weak"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_cases(path, known_groups=None):
    known = set() if known_groups is None else set(known_groups)
    rows, ids, groups = [], set(), set()
    raw = Path(path).read_bytes()
    for line in raw.decode("utf-8-sig").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"id", "prompt", "reference", "grade"}:
            raise ValueError("Cada fila requiere exclusivamente id, prompt, reference y grade.")
        if any(not isinstance(row[name], str) or not row[name].strip() for name in row):
            raise ValueError("Los campos JSONL deben ser texto no vacío.")
        if row["grade"] not in {"exact", "contains"}:
            raise ValueError("grade debe ser exact o contains.")
        group = hashlib.sha256(normalize_for_dedup(row["prompt"]).encode()).hexdigest()
        if row["id"] in ids or group in groups or group in known:
            raise ValueError("ID/prompt duplicado o solapamiento exacto con prompts conocidos.")
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
        ids.add(row["id"])
        groups.add(group)
        rows.append(row)
    if not rows:
        raise ValueError("El JSONL debe contener casos.")
    return rows, known, hashlib.sha256(raw).hexdigest()


def prepare_prospective(dataset_path, output_dir, policy_manifest, known_prompt_groups=None,
                        collect_both_models=False):
    rows, known, dataset_hash = _read_cases(dataset_path, known_prompt_groups)
    if not isinstance(policy_manifest, dict) or not policy_manifest or not isinstance(collect_both_models, bool):
        raise ValueError("Se requiere un manifiesto JSON no vacío y collect_both_models booleano.")
    body = {"schema_version": 1, "dataset_sha256": dataset_hash, "n_cases": len(rows),
            "policy_manifest": policy_manifest, "collect_both_models": collect_both_models,
            "known_prompt_groups_sha256": _hash(sorted(known)),
            "overlap_scope": "exact normalized prompt hashes only; not a certificate of independence"}
    expected = {**body, "freeze_sha256": _hash(body)}
    output = Path(output_dir)
    freeze = output / "freeze.json"
    if freeze.exists():
        if json.loads(freeze.read_text(encoding="utf-8")) != expected:
            raise ValueError("El input o manifiesto difiere de la evaluación congelada.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("No se escribe sobre un directorio no vacío sin freeze.json.")
        output.mkdir(parents=True, exist_ok=True)
        _write(freeze, expected)
    return expected


def _checkpoint(path, freeze_key, default):
    if not path.exists():
        return default
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["freeze_sha256"] != freeze_key or saved["payload_sha256"] != _hash(saved["payload"]):
        raise ValueError("Checkpoint modificado o de otro experimento.")
    return saved["payload"]


def _save(path, freeze_key, payload):
    _write(path, {"freeze_sha256": freeze_key, "payload_sha256": _hash(payload), "payload": payload})


def _token(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _record(row, model):
    return {"id": row["id"], "model": model, "prompt_sha256": row["prompt_sha256"],
            "status": "pending", "content": None, "usage": {name: None for name in
            ("input_tokens", "output_tokens", "total_tokens")}, "duration_seconds": None, "error_type": None}


def _aggregate(outcomes, responses):
    result = {}
    for policy in outcomes:
        selected = outcomes[policy]
        records = [responses[row["response_key"]] for row in selected]
        by_grade = {}
        for grade in ("exact", "contains"):
            subset = [row for row in selected if row["grade"] == grade]
            if subset:
                wins = sum(row["success"] for row in subset)
                by_grade[grade] = {"n": len(subset), "successes": wins, "success_rate": wins/len(subset)}
        tokens = {}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            values = [record["usage"][name] for record in records if record["usage"][name] is not None]
            tokens[name] = {"known_total": sum(values) if values else None,
                            "known_cases": len(values), "unknown_cases": len(records)-len(values)}
        duration = [record["duration_seconds"] for record in records if record["duration_seconds"] is not None]
        result[policy] = {"by_grade": by_grade, "n_cases": len(selected),
                          "generation_failures": sum(record["status"] != "ok" for record in records),
                          "selected_model_counts": {model: sum(row["model"] == model for row in selected) for model in sorted(MODELS)},
                          "tokens": tokens, "selected_generation_seconds": sum(duration) if duration else None,
                          "cost_usd": None}
    return result


def run_prospective(dataset_path, output_dir, policy_manifest, route_candidate, route_reference,
                    generate, known_prompt_groups=None, collect_both_models=False):
    freeze = prepare_prospective(dataset_path, output_dir, policy_manifest, known_prompt_groups, collect_both_models)
    cases, _, dataset_hash = _read_cases(dataset_path, known_prompt_groups)
    if dataset_hash != freeze["dataset_sha256"]:
        raise ValueError("El input cambió después de congelarlo.")
    output, key = Path(output_dir), freeze["freeze_sha256"]
    routes_path, responses_path = output / "routes.json", output / "responses.json"
    summary_path = output / "summary.json"
    if summary_path.exists() and (not routes_path.is_file() or not responses_path.is_file()):
        raise ValueError("Un resultado completado requiere sus checkpoints intactos.")
    routes = _checkpoint(routes_path, key, [])
    if not isinstance(routes, list) or len(routes) > len(cases):
        raise ValueError("Checkpoint de rutas inválido.")
    for position, saved in enumerate(routes):
        row = cases[position]
        if not isinstance(saved, dict) or set(saved) != {"id", "prompt_sha256", "routes"} or saved["id"] != row["id"] or saved["prompt_sha256"] != row["prompt_sha256"] or not isinstance(saved["routes"], dict) or set(saved["routes"])-set(POLICIES) or any(not isinstance(value, str) or value not in MODELS for value in saved["routes"].values()):
            raise ValueError("Las rutas guardadas no coinciden con los casos.")
    if summary_path.exists() and (len(routes) != len(cases) or any(set(saved["routes"]) != set(POLICIES) for saved in routes)):
        raise ValueError("Las rutas de un resultado completado están incompletas.")
    for position, row in enumerate(cases):
        if position == len(routes):
            routes.append({"id": row["id"], "prompt_sha256": row["prompt_sha256"], "routes": {}})
        for policy, callback in zip(POLICIES, (route_candidate, route_reference)):
            if policy not in routes[position]["routes"]:
                model = callback(row["prompt"])
                if not isinstance(model, str) or model not in MODELS:
                    raise ValueError("Los routers deben seleccionar strong o weak.")
                routes[position]["routes"][policy] = model
                _save(routes_path, key, routes)
    # All selected actions have now been persisted before the first generation.
    required = {}
    for row, action in zip(cases, routes):
        models = MODELS if collect_both_models else set(action["routes"].values())
        for model in sorted(models):
            response_key = _hash([row["id"], row["prompt_sha256"], model])
            required[response_key] = (row, model)
    responses = _checkpoint(responses_path, key, {})
    if not isinstance(responses, dict) or set(responses)-set(required):
        raise ValueError("Checkpoint de respuestas contiene llamadas ajenas.")
    for response_key, saved in responses.items():
        expected = _record(*required[response_key])
        if not isinstance(saved, dict) or set(saved) != set(expected) or any(saved[name] != expected[name] for name in ("id", "model", "prompt_sha256")) or saved["status"] not in {"pending", "ok", "error"}:
            raise ValueError("Respuesta guardada inválida o desalineada.")
        if saved["status"] == "ok" and (not isinstance(saved["content"], str) or not saved["content"].strip()):
            raise ValueError("Una respuesta exitosa debe conservar contenido.")
        if not isinstance(saved["usage"], dict) or set(saved["usage"]) != set(expected["usage"]) or any(value is not None and _token(value) is None for value in saved["usage"].values()):
            raise ValueError("Usage guardado inválido.")
        duration = saved["duration_seconds"]
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0):
            raise ValueError("Duración guardada inválida.")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        body = {name: value for name, value in summary.items() if name != "result_sha256"}
        if summary["freeze_sha256"] != key or summary["result_sha256"] != _hash(body) or summary["responses_sha256"] != _file_hash(responses_path) or summary["routes_sha256"] != _file_hash(routes_path) or set(responses) != set(required) or any(record["status"] == "pending" for record in responses.values()):
            raise ValueError("El resultado completado o sus registros cambió.")
        return summary
    for response_key, (row, model) in required.items():
        if response_key in responses:
            if responses[response_key]["status"] == "pending":
                responses[response_key].update(status="error", error_type="InterruptedGeneration")
                _save(responses_path, key, responses)
            continue
        record = _record(row, model)
        responses[response_key] = record
        _save(responses_path, key, responses)  # In-flight calls are never retried silently.
        start = time.perf_counter()
        try:
            answer = generate(model, row["prompt"])
            if not isinstance(answer, dict) or not isinstance(answer.get("content"), str) or not answer["content"].strip():
                raise ValueError("Invalid generation content")
            record.update(status="ok", content=answer["content"])
            usage = answer.get("usage") if isinstance(answer.get("usage"), dict) else {}
            record["usage"] = {name: _token(usage.get(name)) for name in record["usage"]}
            provided = answer.get("duration_seconds")
            if isinstance(provided, (int, float)) and not isinstance(provided, bool) and math.isfinite(provided) and provided >= 0:
                record["duration_seconds"] = provided
        except Exception as error:
            record.update(status="error", error_type=type(error).__name__)
        if record["duration_seconds"] is None:
            record["duration_seconds"] = time.perf_counter()-start
        _save(responses_path, key, responses)
    if _file_hash(dataset_path) != freeze["dataset_sha256"]:
        raise ValueError("El input cambió durante la evaluación; se conservan checkpoints sin publicar resultado.")
    outcomes = {policy: [] for policy in POLICIES}
    if collect_both_models:
        outcomes.update(baseline_strong=[], baseline_weak=[])
    for row, action in zip(cases, routes):
        chosen = dict(action["routes"])
        if collect_both_models:
            chosen.update(baseline_strong="strong", baseline_weak="weak")
        for policy, model in chosen.items():
            response_key = _hash([row["id"], row["prompt_sha256"], model])
            answer = responses[response_key]
            text, gold = normalize_for_dedup(answer["content"] or ""), normalize_for_dedup(row["reference"])
            success = answer["status"] == "ok" and (text == gold if row["grade"] == "exact" else gold in text)
            outcomes[policy].append({"id": row["id"], "model": model, "grade": row["grade"],
                                     "success": success, "response_key": response_key})
    summary = {"freeze_sha256": key, "dataset_sha256": freeze["dataset_sha256"],
               "policies": _aggregate(outcomes, responses), "outcomes": outcomes,
               "unique_generation_calls": len(responses), "cost_usd": None,
               "grading_scope": "exact and contains reported separately; not RouterBench scores",
               "duration_scope": "selected generation callbacks only; shared measurements reused across policies; no router overhead",
               "production_validation": False, "superiority_test_performed": False,
               "responses_sha256": _file_hash(responses_path), "routes_sha256": _file_hash(routes_path)}
    summary["result_sha256"] = _hash(summary)
    _write(summary_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and freeze an external JSONL; never call models.")
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--known-prompt-groups", type=Path)
    parser.add_argument("--collect-both-models", action="store_true")
    args = parser.parse_args(argv)
    known = json.loads(args.known_prompt_groups.read_text(encoding="utf-8")) if args.known_prompt_groups else None
    manifest = json.loads(args.policy_manifest.read_text(encoding="utf-8"))
    print(json.dumps(prepare_prospective(args.input, args.output, manifest, known, args.collect_both_models), indent=2))


if __name__ == "__main__":
    main()
