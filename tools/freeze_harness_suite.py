"""Freeze a small independent-source exploratory cohort before model generation."""
import ast
import doctest
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.harness_episode import check_source, run_check, task_prompt

SOURCE = ROOT / "data/imported/harness-public-v1"
OUT = ROOT / "configs/harness_public_pilot_v1.json"
COUNT = 12


def rank(value):
    return hashlib.sha256(("relay-harness-public-v1|" + value).encode()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("The cohort is frozen already; do not overwrite or resplit it.")
    candidates = []
    for row in json.loads((SOURCE / "humaneval.json").read_text(encoding="utf-8")):
        initial = row["prompt"]
        try:
            tree = check_source(initial)
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == row["entry_point"])
            doc = ast.get_docstring(function) or ""
            examples = doctest.DocTestParser().get_examples(doc)
            if not examples or len(initial) > 3000:
                continue
            gold = initial + row["canonical_solution"]
            check_source(gold)
        except (ValueError, SyntaxError, StopIteration):
            continue
        public = ("import doctest\n_test = doctest.DocTestParser().get_doctest(" + repr(doc)
                  + ", globals(), 'public_examples', None, 0)\n"
                  "_result = doctest.DocTestRunner().run(_test)\nassert _result.failed == 0\n")
        candidates.append(dict(task_id=row["task_id"].replace("/", "-"), workflow="function-implementation",
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
        # Preserve upstream semantics; omit tests requiring bespoke adapters.
        expected = "assert " + name + "(*input_data) == expected"
        if expected not in test or "testdata = [[" in test or len(initial) > 3000:
            continue
        try:
            check_source(initial)
            check_source(gold)
            cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (ValueError, SyntaxError):
            continue
        if len(cases) < 2:
            continue
        def checks(values):
            return ("for input_data, expected in " + repr(values) + ":\n    " + expected + "\n")
        candidates.append(dict(task_id="QuixBugs-" + name, workflow="function-repair", initial_source=initial,
                               public_tests=checks(cases[:1]), private_tests=checks(cases[1:]), gold_source=gold,
                               instruction="Repair the defect in solution.py according to its specification. Preserve function names and interfaces.",
                               upstream="jkoppel/QuixBugs"))
    chosen, excluded = [], []
    for workflow in ("function-implementation", "function-repair"):
        eligible = sorted((row for row in candidates if row["workflow"] == workflow), key=lambda row: rank(row["task_id"]))
        accepted = []
        for row in eligible:
            with tempfile.TemporaryDirectory() as tmp:
                pub = run_check(row["gold_source"], row["public_tests"], directory=Path(tmp) / "public")
                private = run_check(row["gold_source"], row["private_tests"], directory=Path(tmp) / "private")
                original = run_check(row["initial_source"], row["private_tests"], directory=Path(tmp) / "original", timeout=2)
            if not pub["passed"] or not private["passed"] or original["passed"]:
                excluded.append({"task_id": row["task_id"], "gold_public": pub["passed"],
                                 "gold_private": private["passed"], "original_private": original["passed"]})
                continue
            row.pop("gold_source")
            row["group_id"] = row["task_id"]
            row["split"] = "development" if len(accepted) < COUNT // 2 else "validation"
            row["routing_text_sha256"] = hashlib.sha256(task_prompt(row).encode()).hexdigest()
            accepted.append(row)
            if len(accepted) == COUNT:
                break
        if len(accepted) < COUNT:
            raise ValueError(f"Insufficient eligible tasks for {workflow}: {len(accepted)}")
        chosen.extend(accepted)
    config = {"schema_version": 1, "kind": "frozen_public_harness_pilot", "evaluation_id": "openhands-public-v1",
              "scope": "exploratory public benchmarks; not production approval or independent proof of no quality loss",
              "models": ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"],
              "baseline_model": "groq/openai/gpt-oss-120b", "max_calls_per_episode": 3,
              "max_total_http_attempts": 144, "max_output_tokens": 1024,
              "max_observed_tokens_per_model": 180000, "temperature": 0, "reasoning_effort": "low",
              "selection": "workflow alternative only if development has no acceptance loss and >=10% lower uncached tariff cost",
              "development_groups_per_workflow": 6, "validation_groups_per_workflow": 6,
              "production_policy_gate_unchanged": {"min_groups_per_workflow_per_split": 20},
              "order": "development before frozen plan before validation; hash task ordering; model order hash-balanced",
              "task_selection": "hash order; public function/source-size constraints; gold tests pass and original held-out tests fail; no model outcomes",
              "excluded_preflight": excluded,
              "source_manifest_sha256": hashlib.sha256((SOURCE / "sources.json").read_bytes()).hexdigest(),
              "tasks": chosen}
    # Keep task texts/tests local; committed config is a small manifest pointing to hashed data.
    payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    frozen = SOURCE / "frozen_suite.json"
    frozen.write_text(payload, encoding="utf-8", newline="\n")
    summary = {key: value for key, value in config.items() if key != "tasks"}
    summary.update(task_file=str(frozen.relative_to(ROOT)).replace("\\", "/"),
                   task_file_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                   tasks=[{key: row[key] for key in ("task_id", "group_id", "workflow", "split", "routing_text_sha256")} for row in chosen])
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUT), "tasks": len(chosen), "episodes": len(chosen) * 2,
                      "excluded_preflight": len(excluded)}))


if __name__ == "__main__":
    main()
