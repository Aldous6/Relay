"""Prepare paired challenge observations; execution uses an in-memory Groq key.

Initial states plus fixed decision 2 on cheap-probe reference trajectories are
declared before outcomes. Reserved tasks are never executed by this command.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from groq_local import configuration
from routerbench.agent_challenges import build_challenge_tasks, CHALLENGE_FAMILIES
from routerbench.agent_evaluation import object_hash
from routerbench.agent_tasks import validate_task_splits

PARTITIONS = ("fit", "calibration", "development")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def make_splits(seed):
    tasks = build_challenge_tasks(seed=seed, per_family=6)
    splits = {name: [] for name in (*PARTITIONS, "evaluation")}
    for family in CHALLENGE_FAMILIES:
        cases = [task for task in tasks if task["family"] == family]
        for index, task in enumerate(cases):
            part = "fit" if index < 2 else "calibration" if index == 2 else "development" if index == 3 else "evaluation"
            splits[part].append(task)
    validate_task_splits(splits)
    return splits


def prepare(directory, seed=20261113):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    splits = make_splits(seed)
    config = configuration(False)
    config.update(run_id="trajectory-paired-development", seed=seed, task_splits_path=str(directory / "task_splits.json"),
                  scope="Paired synthetic known-family reference continuations; not independent validation")
    # A study declaration, not a heuristic to select states from their results.
    plan = {"schema_version": 1, "seed": seed, "partitions": {k: len(v) for k, v in splits.items()},
            "decision_indices": [0, 2], "followup_source": "cheap probe followed by reactive reference",
            "models": config["model_catalog"], "task_splits_sha256": object_hash(splits),
            "configuration": config, "free_account_confirmation_required_for_execution": True,
            "reserved_partition_executed": False, "independent_validation": False,
            "scope": "Acquisition only; final trajectory labels under one reference policy, not learned policy performance"}
    write_new(directory / "task_splits.json", splits)
    write_new(directory / "plan.json", {"payload": plan, "payload_sha256": object_hash(plan)})
    return plan


def paired_training_rows(result, models):
    """Drop an entire incomplete pair for fit, retaining all branches in audit."""
    groups = {}
    for branch in result["branches"]:
        groups.setdefault(branch["anchor_index"], []).append(branch)
    rows, excluded = [], 0
    for branches in groups.values():
        if (len(branches) != len(models) or {b["candidate"] for b in branches} != set(models)
                or any(b["censored_reason"] is not None for b in branches)):
            excluded += 1
        else:
            rows.extend(b["row"] for b in branches)
    return rows, excluded


def execute(directory, *, free_confirmed=False):
    from routerbench.agent_litellm import LiteLLMEpisodeGenerator
    from routerbench.agent_pacing import PacedEpisodeGenerator
    from routerbench.agent_rollouts import initial_anchors, select_followup_anchors, collect_paired_rollouts
    directory = Path(directory).resolve()
    sealed = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    plan = sealed["payload"]
    if sealed["payload_sha256"] != object_hash(plan):
        raise ValueError("El plan fue alterado.")
    splits = json.loads((directory / "task_splits.json").read_text(encoding="utf-8"))
    if object_hash(splits) != plan["task_splits_sha256"]:
        raise ValueError("Las particiones cambiaron después de preparar.")
    validate_task_splits(splits)
    if not free_confirmed:
        raise ValueError("Se requiere confirmar el plan Free; no basta la tabla de cuotas.")
    if (directory / "observations.json").exists() or any((directory / part).exists() for part in PARTITIONS):
        raise FileExistsError("Usa un plan nuevo; este helper no reejecuta colecciones parciales.")
    config = deepcopy(plan["configuration"])
    config["spending"].update(account_is_free_tier=True,
        account_tier_source="User explicitly confirmed Groq Billing Free and authorized execution",
        max_total_generation_calls=336)
    config["collector_runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base = LiteLLMEpisodeGenerator(config["models_metadata"], config["generation_options"])
    generate = PacedEpisodeGenerator(base, **config["pacing"],
        max_output_tokens=config["generation_options"]["max_output_tokens"], show_progress=True)
    config["generation_manifest"] = generate.manifest()
    tariffs = json.loads((ROOT / "configs/groq_paid_tariffs_20261112.json").read_text(encoding="utf-8"))
    models = tuple(config["model_catalog"])
    dataset = {"schema_version": 2, "model_catalog": list(models)}
    audit = {"plan_sha256": sealed["payload_sha256"], "collections": [], "excluded_censored_pairs": 0,
             "reserved_partition_executed": False, "independent_validation": False}
    for part in PARTITIONS:
        tasks = splits[part]
        part_config = deepcopy(config)
        # Harness execution modes have no fit label; data membership is recorded separately.
        part_config.update(evaluation_split="development", data_partition=part)
        initial = initial_anchors(tasks, part_config)
        dataset[part] = []
        for stage in ("initial", "followup"):
            bundle = initial if stage == "initial" else select_followup_anchors(
                directory / part / "initial", decision_indices=(2,), probe_model=models[0])
            if not bundle["anchors"]:
                audit["collections"].append({"partition": part, "stage": stage, "unavailable": bundle["unavailable"]})
                continue
            bound = sum(anchor["request"]["state"]["remaining_calls"] * len(models) for anchor in bundle["anchors"])
            output = directory / part / stage
            print(json.dumps({"stage": stage, "partition": part, "anchors": len(bundle["anchors"]),
                              "maximum_generation_calls": bound, "output": str(output)}), flush=True)
            result = collect_paired_rollouts(tasks, bundle, models, generate, part_config, output,
                max_generation_calls=bound, tariffs=tariffs, execute=True)
            rows, excluded = paired_training_rows(result, models)
            dataset[part].extend(rows)
            audit["excluded_censored_pairs"] += excluded
            audit["collections"].append({"partition": part, "stage": stage, "directory": str(output),
                "freeze_sha256": result["collection_freeze_sha256"], "generation_calls": result["generation_calls"],
                "censored_branches": len(result["censored"]), "excluded_censored_pairs": excluded,
                "unavailable": result["unavailable"]})
            print(json.dumps({"completed": stage, "partition": part, "generation_calls": result["generation_calls"],
                              "training_rows": len(rows), "censored_pairs": excluded}), flush=True)
    audit["class_support"] = {part: {model: dict(Counter(str(row["success"]).lower() for row in dataset[part]
        if row["model"] == model)) for model in models} for part in PARTITIONS}
    audit["observations_sha256"] = object_hash(dataset)
    audit["training_class_support_ready"] = all(set(audit["class_support"][part][model]) == {"true", "false"}
        for part in ("fit", "calibration") for model in models)
    write_new(directory / "observations.json", dataset)
    write_new(directory / "acquisition_audit.json", audit)
    print(json.dumps({"observations": str(directory / "observations.json"), **audit}, indent=2), flush=True)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--free-account-confirmed", action="store_true")
    args = parser.parse_args(argv)
    directory = args.output or ROOT / "outputs/groq-trajectory" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not directory.exists():
        plan = prepare(directory, args.seed)
        print(json.dumps({"plan_directory": str(directory), "partitions": plan["partitions"], "provider_calls": 0}), flush=True)
    if not args.execute:
        return directory
    if not args.free_account_confirmed:
        raise ValueError("Confirma Billing Free con --free-account-confirmed antes de ejecutar.")
    if not sys.stdin.isatty():
        raise RuntimeError("Usa una terminal para la entrada oculta de la clave.")
    key = getpass.getpass("Groq API key (oculta; solo en memoria): ").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("Clave vacía o con espacios.")
    previous = os.environ.get("GROQ_API_KEY")
    try:
        os.environ["GROQ_API_KEY"] = key
        del key
        return execute(directory, free_confirmed=True)
    finally:
        if previous is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous


if __name__ == "__main__":
    main()
