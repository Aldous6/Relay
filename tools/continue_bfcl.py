"""Continue only unattempted BFCL pairs in a new, audited session.

Prepare is the default and never requests credentials or generates responses.
Closed, censored and interrupted attempts are never selected again. Sources,
cases, policies and generation settings are inherited; only the declared local
session token allocation may change. This is not a provider quota reset.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from routerbench.agent_evaluation import _fingerprint, _seal, _sources, object_hash
from routerbench.bfcl_environment import BFCLRuntime
from routerbench.bfcl_harness import prepare_bfcl_episodes, run_bfcl_episodes
from routerbench.bfcl_litellm import BFCLPacedGenerator
from routerbench.bfcl_session import BFCLSessionGenerator
from report_bfcl import POLICIES, Snapshot, _manifest, audit_bfcl
from run_bfcl import build_generator, _check_no_inline_secrets, _write_frozen

KIND = "bfcl_pending_continuation_v1"


def _token_limit(value):
    if type(value) is not int or not 1 <= value <= 200000:
        raise ValueError("max-session-tokens requiere un entero entre 1 y 200000.")
    return value


def build_session(config):
    base = build_generator(config)
    paced = BFCLPacedGenerator(base, **config["pacing"],
        max_output_tokens=config["generation_options"]["max_output_tokens"],
        show_progress=config.get("console_progress") is True)
    return BFCLSessionGenerator(paced, **config["session_limits"])


def split_pending(cases, policy_sequence, pending):
    """At most two rectangles reproduce an exact suffix of case-major pairs."""
    planned = [[case["id"], name] for case in cases for name in policy_sequence]
    if pending and pending != planned[len(planned)-len(pending):]:
        raise ValueError("Los pendientes deben ser una cola ordenada de pares congelados.")
    if not pending:
        return []
    ids = [case["id"] for case in cases]
    index = ids.index(pending[0][0])
    names = [name for identifier, name in pending if identifier == ids[index]]
    groups = []
    if names != policy_sequence:
        groups.append({"case_ids": [ids[index]], "policies": names})
        index += 1
    if index < len(ids):
        groups.append({"case_ids": ids[index:], "policies": list(policy_sequence)})
    for number, group in enumerate(groups):
        group["directory"] = f"run-{number+1:02d}"
        group["planned_pairs"] = [[identifier, name] for identifier in group["case_ids"] for name in group["policies"]]
    if [pair for group in groups for pair in group["planned_pairs"]] != pending:
        raise ValueError("La particion de continuacion cambio los pares pendientes.")
    return groups


def _load_run(directory, snapshot):
    report = audit_bfcl(directory)
    if not report["summary_present"] or not report["summary_current"] or report["status"] not in {"complete", "stopped"}:
        raise ValueError("La corrida previa debe tener un summary cerrado y vigente; no se reenvian pending.")
    manifest = _manifest(directory, snapshot)
    summary = snapshot.read(directory / "summary.json", sealed=True)
    if "unattempted_pairs" not in summary:
        raise ValueError("El resumen previo debe declarar sus pares pendientes explicitamente.")
    if (directory / "journal.json").exists():
        snapshot.read(directory / "journal.json", sealed=True)
    statuses = {(row["episode_id"], row["policy"]): row["status"] for row in report["episodes"]}
    if any(statuses.get(tuple(pair)) != "not_started" for pair in summary["unattempted_pairs"]):
        raise ValueError("Un pendiente declarado ya contiene intentos; no se puede repetir.")
    return {"configuration": manifest["configuration"], "cases": manifest["cases"],
        "policy_sequence": manifest["policy_sequence"], "pending": summary["unattempted_pairs"],
        "completed": summary["completed_pairs"], "manifest": manifest,
        "run_directories": [str(directory)], "summary": summary}


def _changed_config(config, limit, generator_factory):
    config = deepcopy(config)
    if (type(config.get("session_limits")) is not dict
            or set(config["session_limits"]) != {"max_tokens_per_model", "max_provider_calls"}
            or type(config["session_limits"]["max_provider_calls"]) is not int
            or config["session_limits"]["max_provider_calls"] < 1
            or config.get("session_stop_on_infrastructure") is not True):
        raise ValueError("La corrida previa requiere limites de sesion y detencion de infraestructura congelados.")
    prior = generator_factory(deepcopy(config))
    if prior.manifest() != config["generation_manifest"]:
        raise ValueError("El generador actual no reproduce el manifiesto previo.")
    config["session_limits"]["max_tokens_per_model"] = _token_limit(limit)
    generate = generator_factory(deepcopy(config))
    config["generation_manifest"] = generate.manifest()
    _check_no_inline_secrets(config)
    return config, generate, prior


def _load_previous(directory, snapshot, *, visited=None):
    directory = Path(directory).resolve()
    visited = set() if visited is None else visited
    if directory in visited or len(visited) >= 64:
        raise ValueError("La cadena de continuaciones tiene un ciclo o excede el limite.")
    visited.add(directory)
    if not (directory / "continuation.json").exists():
        return _load_run(directory, snapshot)
    plan = snapshot.read(directory / "continuation.json", sealed=True)
    summary = snapshot.read(directory / "summary.json", sealed=True)
    if plan.get("kind") != KIND or summary.get("continuation_sha256") != object_hash(plan):
        raise ValueError("Plan de continuacion incompatible o alterado.")
    parent = _load_previous(plan["parent"]["directory"], snapshot, visited=visited)
    _verify_parent_hashes(plan["parent"]["file_sha256"], snapshot)
    if (plan["ordered_pending_pairs"] != parent["pending"]
            or plan["policy_sequence"] != parent["policy_sequence"]
            or plan["source_files_sha256"] != parent["manifest"]["source_files_sha256"]
            or plan["ancestor_run_directories"] != parent["run_directories"]):
        raise ValueError("La continuacion no coincide con los pendientes del padre.")
    groups = split_pending(parent["cases"], parent["policy_sequence"], parent["pending"])
    if [{k: v for k, v in group.items() if k != "freeze_sha256"} for group in plan["runs"]] != groups:
        raise ValueError("Los bloques de continuacion cambiaron.")
    allowed = deepcopy(parent["configuration"])
    allowed["session_limits"]["max_tokens_per_model"] = _token_limit(plan["max_session_tokens"])
    allowed["generation_manifest"] = deepcopy(plan["configuration"]["generation_manifest"])
    if plan["configuration"] != allowed:
        raise ValueError("La continuacion cambio parametros fuera del limite de sesion.")
    completed, count, closed_dirs = [], 0, []
    by_id = {case["id"]: case for case in parent["cases"]}
    for group in plan["runs"]:
        path = directory / group["directory"]
        manifest = _manifest(path, snapshot)
        if (manifest["configuration"] != plan["configuration"]
                or manifest["cases"] != [by_id[identifier] for identifier in group["case_ids"]]
                or manifest["policy_sequence"] != group["policies"]
                or manifest["freeze_sha256"] != group["freeze_sha256"]):
            raise ValueError("Un bloque no reproduce su plan congelado.")
        if (path / "summary.json").exists():
            child = _load_run(path, snapshot)
            completed.extend(child["completed"])
            count += child["summary"]["generation_calls"]
            closed_dirs.append(str(path))
        elif (path / "journal.json").exists():
            if snapshot.read(path / "journal.json", sealed=True)["records"]:
                raise ValueError("La continuacion contiene un bloque sin cerrar; no se reenvian pending.")
    expected = _root_summary(plan, completed, count, closed_dirs, summary.get("session_stop_reason"))
    if summary != expected:
        raise ValueError("El resumen de continuacion no reproduce sus bloques cerrados.")
    return {**parent, "configuration": plan["configuration"], "pending": summary["unattempted_pairs"],
        "completed": summary["completed_pairs"], "run_directories": parent["run_directories"] + closed_dirs,
        "summary": summary}


def _verify_parent_hashes(hashes, snapshot):
    for path, expected in hashes.items():
        if hashlib.sha256(snapshot.raw(path)).hexdigest() != expected:
            raise ValueError("Un artefacto del padre cambio tras preparar la continuacion.")


def _root_summary(plan, completed, calls, closed_dirs, reason):
    pending = plan["ordered_pending_pairs"]
    if completed != pending[:len(completed)]:
        raise ValueError("Los cierres de continuacion no forman un prefijo de los pendientes.")
    unattempted = pending[len(completed):]
    if unattempted and (type(reason) is not dict or not reason):
        raise ValueError("La continuacion parcial requiere una causa explicita.")
    return {"schema_version": 1, "kind": KIND, "continuation_sha256": object_hash(plan),
        "planned_pairs": pending, "completed_pairs": completed, "unattempted_pairs": unattempted,
        "generation_calls": calls, "run_directories": closed_dirs,
        "session_stopped": bool(unattempted), "session_stop_reason": reason if unattempted else None,
        "execution_complete": not unattempted, "independent_validation": False}


def prepare_continuation(previous, output, *, max_session_tokens=160000,
                         generator_factory=build_session, runtime_factory=BFCLRuntime):
    """Audit closed artifacts and prepare exact remaining pairs without API calls."""
    _token_limit(max_session_tokens)
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("La continuacion necesita una carpeta NUEVA.")
    snapshot = Snapshot()
    parent = _load_previous(previous, snapshot)
    source = parent["manifest"]
    if (parent["configuration"].get("execution_mode") == "external"
            and (parent["configuration"].get("evaluation_split") != "development"
                 or parent["configuration"].get("independent_validation") is not False)):
        raise ValueError("La continuacion externa solo admite el desarrollo ya declarado.")
    if source["source_files_sha256"] != _sources():
        raise ValueError("Las fuentes actuales no coinciden con la corrida previa.")
    config, generate, prior = _changed_config(parent["configuration"], max_session_tokens, generator_factory)
    if (_fingerprint(prior) != source["generation_source"]
            or _fingerprint(generate) != source["generation_source"]
            or _fingerprint(runtime_factory) != source["runtime_source"]):
        raise ValueError("La implementacion del generador o runtime cambio.")
    sequence = parent["policy_sequence"]
    for name in sequence:
        policy = POLICIES[name](config["model_catalog"])
        if policy.manifest() != source["policies"][name] or _fingerprint(policy) != source["policy_sources"][name]:
            raise ValueError("La politica actual no coincide con el control previo.")
    groups = split_pending(parent["cases"], sequence, parent["pending"])
    snapshot.verify()
    output.mkdir(parents=True, exist_ok=False)
    by_id = {case["id"]: case for case in parent["cases"]}
    provenance = dict(zip((case["id"] for case in source["cases"]), source["runtime_provenance"]))
    for group in groups:
        cases = [by_id[identifier] for identifier in group["case_ids"]]
        policies = [POLICIES[name](config["model_catalog"]) for name in group["policies"]]
        frozen = prepare_bfcl_episodes(cases, policies, output / group["directory"], config,
            generate=generate, runtime_factory=runtime_factory)
        if frozen["runtime_provenance"] != [provenance[identifier] for identifier in group["case_ids"]]:
            raise ValueError("La procedencia o los limites efectivos del runtime cambiaron.")
        group["freeze_sha256"] = frozen["freeze_sha256"]
    plan = {"schema_version": 1, "kind": KIND,
        "parent": {"directory": str(previous), "file_sha256": deepcopy(snapshot.hashes)},
        "configuration": config, "policy_sequence": sequence,
        "ordered_pending_pairs": parent["pending"], "runs": groups,
        "ancestor_run_directories": parent["run_directories"], "max_session_tokens": max_session_tokens,
        "source_files_sha256": source["source_files_sha256"],
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provider_calls": 0, "independent_validation": False,
        "scope": "Exact ordered unattempted development pairs; one shared local session allocation, no provider quota reset."}
    snapshot.verify()
    _write_frozen(output / "continuation.json", _seal(plan))
    if not groups:
        _write_frozen(output / "summary.json", _seal(_root_summary(plan, [], 0, [], None)))
    return {"plan": plan, "generate": generate, "cases": by_id, "output": output,
            "runtime_factory": runtime_factory, "parent_snapshot": snapshot}


def execute_continuation(prepared):
    plan, output, generate = prepared["plan"], prepared["output"], prepared["generate"]
    prepared["parent_snapshot"].verify()
    if (output / "summary.json").exists():
        snapshot = Snapshot()
        closed = _load_previous(output, snapshot)
        snapshot.verify()
        return closed["summary"]
    completed, calls, closed_dirs, reason = [], 0, [], None
    for group in plan["runs"]:
        reason = generate.session_stop_reason()
        if reason is not None:
            break
        cases = [prepared["cases"][identifier] for identifier in group["case_ids"]]
        policies = [POLICIES[name](plan["configuration"]["model_catalog"]) for name in group["policies"]]
        path = output / group["directory"]
        result = run_bfcl_episodes(cases, policies, generate, path, plan["configuration"],
            runtime_factory=prepared["runtime_factory"])
        completed.extend(result["completed_pairs"])
        calls += result["generation_calls"]
        closed_dirs.append(str(path))
        if result["session_stopped"]:
            reason = result["session_stop_reason"]
            break
        # A shared outage on the last pair of a rectangle is still terminal for
        # the enclosing session, even though that rectangle itself is complete.
        from routerbench.bfcl_harness import _shared_session_failure
        reason = _shared_session_failure(result["episodes"][-1]) if result["episodes"] else None
        if reason is not None:
            break
    summary = _root_summary(plan, completed, calls, closed_dirs, reason)
    _write_frozen(output / "summary.json", _seal(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True, help="Closed BFCL RUN or closed continuation root.")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are rejected.")
    parser.add_argument("--max-session-tokens", type=int, default=160000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    prepared = prepare_continuation(args.previous, args.output, max_session_tokens=args.max_session_tokens)
    plan = prepared["plan"]
    result = {"output": str(prepared["output"]), "mode": "prepare", "generation_calls": 0,
        "pending_pairs": len(plan["ordered_pending_pairs"]), "subruns": len(plan["runs"]),
        "continuation_sha256": object_hash(plan), "max_session_tokens": args.max_session_tokens}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not args.execute or not plan["runs"]:
        return result
    config = plan["configuration"]
    spend = config.get("spending", {})
    if (config.get("execution_mode") != "external" or config.get("evaluation_split") != "development"
            or config.get("independent_validation") is not False
            or spend.get("account_is_free_tier") is not True or spend.get("max_provider_spend_usd") != 0
            or set(config.get("models_metadata", {})) != set(config["model_catalog"])
            or any(not name.startswith("groq/") for name in config["model_catalog"])
            or any(row.get("api_key_env") != "GROQ_API_KEY" for row in config.get("models_metadata", {}).values())):
        raise ValueError("Execute requiere la configuracion Groq Free autorizada y desarrollo reservado.")
    previous_key = os.environ.get("GROQ_API_KEY")
    try:
        key = previous_key
        if not key:
            if not sys.stdin.isatty():
                raise RuntimeError("Usa terminal para entrada oculta o GROQ_API_KEY autorizado en el entorno.")
            key = getpass.getpass("Groq API key (oculta; solo en memoria): ")
        if not key or any(character.isspace() for character in key):
            raise ValueError("La clave debe ser no vacia y sin espacios.")
        os.environ["GROQ_API_KEY"] = key
        del key
        summary = execute_continuation(prepared)
        result.update(mode="execute", generation_calls=summary["generation_calls"],
            remaining_pairs=len(summary["unattempted_pairs"]), summary=str(prepared["output"] / "summary.json"))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if previous_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous_key


if __name__ == "__main__":
    main()
