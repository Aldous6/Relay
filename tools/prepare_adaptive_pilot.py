"""Prepare consumed development and an untouched OpenHands holdout, offline.

This command never calls a model and never changes the public-v1/v2 artifacts.
The 24 fresh IDs were fixed by the existing hash/eligibility rule before any
fresh model outcomes. A changed preflight fails instead of choosing substitutes.
"""
from __future__ import annotations

import argparse
import ast
import doctest
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unicodedata
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/imported/harness-public-v1"
DEFAULT_OUTPUT = ROOT / "data/imported/harness-adaptive-v1"
PINNED_INPUTS = {
    "configs/harness_public_pilot_v2.json": "0c5ffbadab1619944df1fcb94aebe138c60f80177a69b4ea4cb06c89afd1720f",
    "data/imported/harness-public-v1/sources.json": "a7dbaa8f7034782fbc8dc6fba90ea76eb0c0e225abd159ac3702d3c6de3a8a82",
    "data/imported/harness-public-v1/frozen_suite_v2.json": "131d77f26699bb8a641d8f179dc17202152067144d873f46419e4c8fc66bd3fc",
    "outputs/harness-live/public-v2/observations.jsonl": "2708440944b231a4b9050e50146caa30600456fee36f12bcf654d7a0ff955fdd",
    "outputs/harness-live/public-v2/run_manifest.json": "4d454b2adcb60c88af3a089845f860658e14239f376ecf6faea9eecc5acb536f",
}
SELECTED = {
    "function-implementation": tuple("HumanEval-" + value for value in
        ("15", "31", "30", "21", "19", "35", "27", "43", "100", "2", "53", "17")),
    "function-repair": tuple("QuixBugs-" + value for value in (
        "powerset", "find_first_in_sorted", "wrap", "bucketsort", "longest_common_subsequence",
        "quicksort", "gcd", "next_permutation", "mergesort", "possible_change", "rpn_eval", "kth")),
}
NORMALIZATION = "NFKC, Unicode casefold, collapsed whitespace; duplicate key only"
HISTORICAL_ROLE = (
    "All 24 public-v2 tasks, including its previously inspected validation split, "
    "are consumed development. They are not reserved evidence for this pilot."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_prompt_digest(prompt):
    # Same conservative rule as routerbench.features.normalize_for_dedup,
    # without importing pandas into the stdlib-only preparation environment.
    return digest(" ".join(unicodedata.normalize("NFKC", prompt).casefold().split()).encode())


def helpers():
    """Reuse the frozen harness rank, source guard, test runner and prompt."""
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("relay_original_freezer", ROOT / "tools/freeze_harness_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pinned_inputs():
    values, provenance = {}, []
    for relative, expected in PINNED_INPUTS.items():
        path = ROOT / relative
        content = path.read_bytes()
        if digest(content) != expected:
            raise ValueError(f"Historical/source input changed: {relative}")
        text = content.decode("utf-8")
        values[relative] = ([json.loads(line) for line in text.splitlines() if line.strip()]
                            if path.suffix == ".jsonl" else json.loads(text))
        provenance.append({"path": relative, "sha256": expected, "bytes": len(content)})
    return values, provenance


def verify_imports(source_manifest):
    """Check originals before reading them, then verify the derived local imports."""
    for entry in source_manifest["sources"]:
        path = SOURCE / entry["file"]
        if path.parent.resolve() != SOURCE.resolve():
            raise ValueError("Source path outside the pinned source directory")
        content = path.read_bytes()
        if digest(content) != entry["sha256"] or len(content) != entry["bytes"]:
            raise ValueError(f"Source hash/length mismatch: {entry['file']}")
    human = [json.loads(line) for line in gzip.decompress((SOURCE / "HumanEval.jsonl.gz").read_bytes()).decode().splitlines()]
    if human != json.loads((SOURCE / "humaneval.json").read_text(encoding="utf-8")):
        raise ValueError("Imported HumanEval differs from the pinned compressed original")
    verified = 0
    with zipfile.ZipFile(SOURCE / "quixbugs.zip") as archive:
        for member in archive.namelist():
            relative = member.partition("/")[2]
            if member.endswith("/") or not relative:
                continue
            if not (relative.startswith(("python_programs/", "correct_python_programs/", "json_testcases/", "python_testcases/"))
                    or relative in {"LICENSE", "README.md", "legal_notes.txt"}):
                continue
            path = SOURCE / "quixbugs" / relative
            if not path.resolve().is_relative_to((SOURCE / "quixbugs").resolve()):
                raise ValueError("Unsafe archive path")
            if path.read_bytes().replace(b"\r\n", b"\n") != archive.read(member).replace(b"\r\n", b"\n"):
                raise ValueError(f"Imported QuixBugs differs from the pinned archive: {relative}")
            verified += 1
    return {"humaneval_tasks": len(human), "quixbugs_files_verified": verified}


def candidates(helper):
    """Apply the original freezer's source-only eligibility rules unchanged."""
    rows = []
    for row in json.loads((SOURCE / "humaneval.json").read_text(encoding="utf-8")):
        initial = row["prompt"]
        try:
            tree = helper.check_source(initial)
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == row["entry_point"])
            doc = ast.get_docstring(function) or ""
            if not doctest.DocTestParser().get_examples(doc) or len(initial) > 3000:
                continue
            gold = initial + row["canonical_solution"]
            helper.check_source(gold)
        except (ValueError, SyntaxError, StopIteration):
            continue
        public = ("import doctest\n_test = doctest.DocTestParser().get_doctest(" + repr(doc)
                  + ", globals(), 'public_examples', None, 0)\n"
                  "_result = doctest.DocTestRunner().run(_test)\nassert _result.failed == 0\n")
        rows.append(dict(task_id=row["task_id"].replace("/", "-"), workflow="function-implementation",
                         initial_source=initial, public_tests=public,
                         private_tests=row["test"] + "\ncheck(" + row["entry_point"] + ")\n",
                         instruction="Implement the specified Python function in solution.py. Preserve its signature and behavior.",
                         gold_source=gold, upstream="openai/human-eval"))
    quix = SOURCE / "quixbugs"
    for path in sorted((quix / "json_testcases").glob("*.json")):
        name = path.stem
        initial = (quix / "python_programs" / (name + ".py")).read_text(encoding="utf-8")
        gold = (quix / "correct_python_programs" / (name + ".py")).read_text(encoding="utf-8")
        test = (quix / "python_testcases" / ("test_" + name + ".py")).read_text(encoding="utf-8")
        expected = "assert " + name + "(*input_data) == expected"
        if expected not in test or "testdata = [[" in test or len(initial) > 3000:
            continue
        try:
            helper.check_source(initial)
            helper.check_source(gold)
            cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (ValueError, SyntaxError):
            continue
        if len(cases) < 2:
            continue
        def checks(values):
            return "for input_data, expected in " + repr(values) + ":\n    " + expected + "\n"
        rows.append(dict(task_id="QuixBugs-" + name, workflow="function-repair", initial_source=initial,
                         public_tests=checks(cases[:1]), private_tests=checks(cases[1:]), gold_source=gold,
                         instruction="Repair the defect in solution.py according to its specification. Preserve function names and interfaces.",
                         upstream="jkoppel/QuixBugs"))
    return rows


def freeze_fresh_tasks(rows, historical, helper):
    previous = {row["task_id"] for row in historical}
    chosen, exclusions = [], []
    for workflow, fixed_ids in SELECTED.items():
        accepted = []
        eligible = sorted((row for row in rows if row["workflow"] == workflow and row["task_id"] not in previous),
                          key=lambda row: helper.rank(row["task_id"]))
        for row in eligible:
            with tempfile.TemporaryDirectory(prefix="relay-adaptive-preflight-") as tmp:
                public = helper.run_check(row["gold_source"], row["public_tests"], directory=Path(tmp) / "public")
                private = helper.run_check(row["gold_source"], row["private_tests"], directory=Path(tmp) / "private")
                original = helper.run_check(row["initial_source"], row["private_tests"], directory=Path(tmp) / "original", timeout=2)
            if not public["passed"] or not private["passed"] or original["passed"]:
                exclusions.append({"task_id": row["task_id"], "gold_public": public["passed"],
                                   "gold_private": private["passed"], "original_private": original["passed"],
                                   "categories": [public["category"], private["category"], original["category"]]})
                continue
            if row["task_id"] != fixed_ids[len(accepted)]:
                raise ValueError("Preflight changed the fixed task selection; do not substitute or tune the holdout")
            item = {key: value for key, value in row.items() if key != "gold_source"}
            prompt = helper.task_prompt(item)
            item.update(group_id=item["task_id"], split="validation", prompt_text=prompt,
                        routing_text_sha256=digest(prompt.encode()), normalized_prompt_sha256=normalized_prompt_digest(prompt))
            accepted.append(item)
            if len(accepted) == len(fixed_ids):
                break
        if len(accepted) != len(fixed_ids):
            raise ValueError(f"Insufficient fixed eligible tasks: {workflow}")
        chosen.extend(accepted)
    assert_disjoint(historical, chosen, helper.task_prompt)
    return chosen, exclusions


def assert_disjoint(historical, fresh, task_prompt):
    for key in ("task_id", "group_id"):
        previous = {row[key] for row in historical}
        current = [row[key] for row in fresh]
        if len(set(current)) != len(current) or previous.intersection(current):
            raise ValueError(f"Historical/fresh overlap or duplicate {key}")
    previous = {normalized_prompt_digest(task_prompt(row)) for row in historical}
    current = [normalized_prompt_digest(task_prompt(row)) for row in fresh]
    if len(set(current)) != len(current) or previous.intersection(current):
        raise ValueError("Historical/fresh overlap or duplicate normalized prompt")


def development_records(suite, observations, task_prompt):
    tasks = {row["task_id"]: row for row in suite["tasks"]}
    expected = {(task_id, model) for task_id in tasks for model in suite["models"]}
    seen, records = set(), []
    for observation in observations:
        pair = (observation["task_id"], observation["model"])
        if pair not in expected or pair in seen:
            raise ValueError("Unexpected or repeated historical task/model observation")
        seen.add(pair)
        task = tasks[pair[0]]
        for key in ("group_id", "workflow", "split"):
            if observation[key] != task[key]:
                raise ValueError(f"Historical observation differs from frozen task: {key}")
        if observation["evaluation_id"] != suite["evaluation_id"] or type(observation["accepted"]) is not bool:
            raise ValueError("Invalid historical evaluation or accepted value")
        prompt = task_prompt(task)
        if digest(prompt.encode()) != task["routing_text_sha256"]:
            raise ValueError("Historical routing text differs from its frozen hash")
        cost = observation["cost_usd"]
        if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            raise ValueError("Historical cost must be finite nonnegative or null")
        if type(observation["cost_complete"]) is not bool or not isinstance(observation["cost_basis"], str):
            raise ValueError("Invalid historical cost provenance")
        if observation["cost_complete"] != (cost is not None):
            raise ValueError("Historical cost completeness disagrees with nullable total")
        record = {key: observation[key] for key in (
            "task_id", "group_id", "workflow", "model", "accepted", "cost_usd", "cost_basis", "cost_complete", "cost_scope")}
        record.update(prompt_text=prompt, routing_text_sha256=task["routing_text_sha256"],
                      normalized_prompt_sha256=normalized_prompt_digest(prompt), split="development",
                      historical_split=task["split"], historical_evaluation_id=suite["evaluation_id"])
        for key in ("known_recorded_cost_subtotal_usd", "logical_calls", "http_attempts", "unknown_http_costs",
                    "wall_seconds", "verification_seconds", "quota_wait_seconds", "transport_seconds", "accounting_source"):
            if key in observation:
                record[key] = observation[key]
        records.append(record)
    if seen != expected:
        raise ValueError("Historical observations are incomplete: retain every task/model, including failures")
    return records


def write_bundle(output, suite, development, provenance):
    """Exclusive creation: an existing preparation is never overwritten."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    written = []
    for name, value in (("fresh_suite.json", suite), ("development.json", development)):
        payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        with (output / name).open("xb") as stream:
            stream.write(payload)
        written.append({"file": name, "sha256": digest(payload), "bytes": len(payload)})
    manifest = {"schema_version": 1, "kind": "adaptive_pilot_preparation", "files": written,
                "sources": provenance, "historical_validation_consumed": True,
                "historical_role": HISTORICAL_ROLE, "model_calls": 0}
    with (output / "preparation_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return manifest


def prepare(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    allowed = [DEFAULT_OUTPUT.resolve(), (ROOT / "outputs/harness-adaptive/v1").resolve()]
    if not any(output == base or output.is_relative_to(base) for base in allowed):
        raise ValueError("Output must be under data/imported/harness-adaptive-v1 or outputs/harness-adaptive/v1")
    if output.exists():
        raise FileExistsError("The preparation directory already exists; frozen artifacts are never overwritten")
    values, provenance = read_pinned_inputs()
    source_manifest = values["data/imported/harness-public-v1/sources.json"]
    config = values["configs/harness_public_pilot_v2.json"]
    historical = values["data/imported/harness-public-v1/frozen_suite_v2.json"]
    run_manifest = values["outputs/harness-live/public-v2/run_manifest.json"]
    if (config["task_file_sha256"] != PINNED_INPUTS["data/imported/harness-public-v1/frozen_suite_v2.json"]
            or config["source_manifest_sha256"] != PINNED_INPUTS["data/imported/harness-public-v1/sources.json"]
            or run_manifest["suite"] != config):
        raise ValueError("Historical suite, source, or run provenance mismatch")
    verification = verify_imports(source_manifest)
    helper = helpers()
    records = development_records(historical, values["outputs/harness-live/public-v2/observations.jsonl"], helper.task_prompt)
    tasks, excluded = freeze_fresh_tasks(candidates(helper), historical["tasks"], helper)
    for relative in ("tools/prepare_adaptive_pilot.py", "tools/freeze_harness_suite.py", "src/relay_router/harness_episode.py"):
        provenance.append({"path": relative, "sha256": digest((ROOT / relative).read_bytes())})
    metadata = {"sources": provenance, "source_manifest": source_manifest, "source_verification": verification,
                "historical_validation_consumed": True, "historical_role": HISTORICAL_ROLE,
                "duplicate_normalization": NORMALIZATION,
                "limitation": "Exploratory public historical function tasks; contamination is possible; not production approval."}
    suite = {key: historical[key] for key in (
        "schema_version", "models", "baseline_model", "max_calls_per_episode", "max_total_http_attempts",
        "max_output_tokens", "max_observed_tokens_per_model", "temperature", "reasoning_effort", "stop_on_public_pass",
        "production_policy_gate_unchanged", "source_manifest_sha256")}
    suite.update(metadata, kind="frozen_adaptive_harness_pilot", evaluation_id="openhands-adaptive-v1",
                 scope="Fresh reserved validation; historical public-v2 is consumed development",
                 task_selection="First 12 eligible per workflow in original SHA-256 order after excluding historical tasks; IDs fixed before fresh outcomes",
                 selection="Frozen prompt policy fitted only on consumed development before any fresh episodes",
                 order="Freeze all routes before fresh episodes; preserve original hash order within workflow",
                 validation_groups_per_workflow=12, excluded_preflight=excluded, tasks=tasks)
    development = dict(metadata, schema_version=1, kind="consumed_harness_development", models=historical["models"],
                       baseline_model=historical["baseline_model"], records=records,
                       unknown_cost_records=sum(row["cost_usd"] is None for row in records),
                       cost_limitation="Uncached tariff scenarios, not Groq Free billed expenditure; unknown totals remain null.")
    manifest = write_bundle(output, suite, development, provenance)
    return {"output": str(output), "fresh_tasks": len(tasks), "development_records": len(records),
            "unknown_development_costs": development["unknown_cost_records"], "files": manifest["files"], "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
