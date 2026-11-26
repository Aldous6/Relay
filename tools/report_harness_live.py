"""Produce an auditable pilot report; unknown costs never become zero savings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay_router.harness_comparison import compare, freeze_plan


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run.resolve()
    manifest = json.loads((folder / "run_manifest.json").read_text())
    config = manifest["suite"]
    raw = (ROOT / config["task_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["task_file_sha256"]:
        raise ValueError("Cohort changed")
    suite = json.loads(raw)
    rows = load_lines(folder / "observations.jsonl")
    attempts = load_lines(folder / "http_attempts.jsonl")
    starts = list((folder / "attempt_starts").glob("*.json"))
    if len(attempts) != len(starts):
        raise ValueError("Unresolved dispatch; report cannot treat it as a completed collection")
    dispatches = {row["attempt_id"]: row for row in attempts}
    started = [json.loads(path.read_text()) for path in starts]
    if len(dispatches) != len(attempts) or {row["attempt_id"] for row in started} != set(dispatches):
        raise ValueError("Duplicate or unmatched HTTP attempt identifiers")
    for row in started:
        if any(row[field] != dispatches[row["attempt_id"]][field] for field in ("episode_id", "model", "started_at_utc")):
            raise ValueError("HTTP write-ahead record differs from its outcome")
    plan = json.loads((folder / "frozen_plan.json").read_text())
    result = compare(suite, plan, rows)
    index = {(row["task_id"], row["model"]): row for row in rows}
    expected = {(task["task_id"], model) for task in suite["tasks"] for model in suite["models"]}
    if set(index) != expected:
        raise ValueError("The primary cohort is incomplete; retain partial observations without declaring completion")
    expected_episodes = {task + "-" + model.rsplit("/", 1)[-1] for task, model in expected}
    if {row["episode_id"] for row in attempts} != expected_episodes:
        raise ValueError("HTTP records do not match the complete episode cohort")
    for row in rows:
        episode_id = row["task_id"] + "-" + row["model"].rsplit("/", 1)[-1]
        episode_attempts = [item for item in attempts if item["episode_id"] == episode_id]
        if row["http_attempts"] != len(episode_attempts):
            raise ValueError("Episode accounting omitted an HTTP attempt")
        known = all(item["input_tokens"] is not None and item["output_tokens"] is not None for item in episode_attempts)
        if row["cost_complete"] != known or (not known and row["cost_usd"] is not None):
            raise ValueError("Unknown HTTP usage was converted to a complete episode cost")
        if known:
            weak = row["model"] == "groq/openai/gpt-oss-20b"
            total = sum((item["input_tokens"] * (.075 if weak else .15) + item["output_tokens"] * (.30 if weak else .60)) / 1e6 for item in episode_attempts)
            if not math.isclose(row["cost_usd"], total, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Episode cost does not match all measured HTTP tokens")
        if row["accepted"] != (row["public_passed"] and row["private_passed"]):
            raise ValueError("Acceptance does not match public and reserved verification")
    selected_tasks = [task for task in suite["tasks"] if task["split"] == "validation"]
    for name, choices in plan["policies"].items():
        selected = [index.get((task["task_id"], choices[task["task_id"]])) for task in selected_tasks]
        if all(row is not None for row in selected):
            result["policies"][name]["median_transport_seconds"] = statistics.median(row["transport_seconds"] for row in selected)
            result["policies"][name]["median_episode_seconds_without_quota_wait"] = statistics.median(
                max(0, row["wall_seconds"] - row["quota_wait_seconds"]) for row in selected)
    # Reprice the diagnostics from their measured tokens using exact model IDs.
    previous = []
    for relative in ("outputs/harness-live/public-v1-complete/http_attempts.jsonl", "outputs/harness-live/public-v1-measured/http_attempts.jsonl"):
        path = ROOT / relative
        if path.exists():
            previous.extend(load_lines(path))
    all_attempts = previous + attempts
    known_subtotal = 0.
    for row in all_attempts:
        if row["input_tokens"] is not None and row["output_tokens"] is not None:
            weak = row["model"] == "groq/openai/gpt-oss-20b"
            known_subtotal += (row["input_tokens"] * (.075 if weak else .15) + row["output_tokens"] * (.30 if weak else .60)) / 1e6
    result["acquisition"] = {"primary_http_attempts": len(attempts), "prior_http_attempts": len(previous),
                             "all_http_attempts": len(all_attempts),
                             "unknown_usage_attempts": sum(row["input_tokens"] is None or row["output_tokens"] is None for row in all_attempts),
                             "known_uncached_tariff_subtotal_usd": known_subtotal,
                             "complete_monetary_cost_usd": None,
                             "note": "Includes retained setup/development diagnostics. Unknown usage and local compute are not imputed."}
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["sources"] = json.loads((ROOT / "data/imported/harness-public-v1/sources.json").read_text())
    result["reference"] = manifest["reference"]
    scores = json.loads((ROOT / "data/imported/harness-public-v1/bert_scores.json").read_text())
    if freeze_plan(suite, [row for row in rows if row["split"] == "development"], scores) != plan:
        raise ValueError("The frozen plan cannot be reproduced from development alone")
    validation_ids = {task["task_id"] for task in selected_tasks}
    result["routing_overhead"] = {
        "bert_validation_inference_seconds": sum(row["router_seconds"] for row in scores["rows"] if row["task_id"] in validation_ids),
        "bert_cold_load_seconds": scores["reference"]["load_seconds"],
        "relay_isolated_lookup_seconds": None, "local_compute_monetary_cost": None,
        "note": "Transport latency fields exclude policy CPU inference; Relay lookup overhead was not independently timed."}
    validation_episodes = {task["task_id"] + "-" + model.rsplit("/", 1)[-1] for task in selected_tasks for model in suite["models"]}
    first_validation = min(datetime.fromisoformat(row["started_at_utc"]) for row in attempts if row["episode_id"] in validation_episodes)
    frozen_at = datetime.fromtimestamp((folder / "frozen_plan.json").stat().st_mtime, tz=timezone.utc)
    if frozen_at > first_validation:
        raise ValueError("The route plan file was modified after validation began")
    sources_unchanged = all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest for relative, digest in manifest["source_sha256"].items())
    result["audit"] = {"cohort_sha256": config["task_file_sha256"], "plan_file_sha256": hashlib.sha256((folder / "frozen_plan.json").read_bytes()).hexdigest(),
                       "observations_sha256": hashlib.sha256((folder / "observations.jsonl").read_bytes()).hexdigest(),
                       "attempts_sha256": hashlib.sha256((folder / "http_attempts.jsonl").read_bytes()).hexdigest(),
                       "episodes": len(rows), "all_expected_episodes_present": set(index) == expected,
                       "all_dispatches_reconciled": True, "episode_costs_recomputed_from_tokens": True,
                       "plan_reproduced_from_development_only": True,
                       "runtime_sources_match_start": sources_unchanged,
                       "plan_file_written_at_utc": frozen_at.isoformat(), "first_validation_http_at_utc": first_validation.isoformat(),
                       "plan_precedes_validation": True}
    result["workflow_routes"] = plan["workflow_routes"]
    relay, cheap, strong = [result["policies"][name] for name in ("relay_exploratory", "fixed_cheap", "fixed_strong")]
    same_routes = all(plan["policies"]["relay_exploratory"][task["task_id"]] == plan["policies"]["fixed_cheap"][task["task_id"]] for task in selected_tasks)
    same_strong_routes = all(plan["policies"]["relay_exploratory"][task["task_id"]] == plan["policies"]["fixed_strong"][task["task_id"]] for task in selected_tasks)
    result["incremental_routing_value"] = {"same_decisions_as_fixed_cheap": same_routes,
                                           "same_decisions_as_fixed_strong": same_strong_routes,
                                           "claim": ("No incremental routing advantage over fixed cheap in this cohort" if same_routes
                                                     else "No routing savings over fixed strong in this cohort" if same_strong_routes
                                                     else "Inspect paired quality and complete costs; differences alone do not establish general superiority")}
    result["development_by_workflow"] = {
        workflow: {model: {"tasks": len(subset), "accepted": sum(row["accepted"] for row in subset),
                           "unknown_cost_episodes": sum(not row["cost_complete"] for row in subset)}
                   for model in suite["models"]
                   for subset in [[row for row in rows if row["split"] == "development" and row["workflow"] == workflow and row["model"] == model]]}
        for workflow in plan["workflow_routes"]}
    destination = ROOT / "reports/harness-live-public-v2.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    def cost(value):
        return "desconocido" if value is None else f"USD {value:.8f}"
    def pct(value):
        return "desconocida" if value is None else f"{value*100:.2f} %"
    text = ["# Relay y OpenHands: resultado del piloto público\n",
            f"Se completaron {len(rows)} episodios sobre 24 tareas públicas. La selección usó doce tareas de desarrollo; los resultados siguientes pertenecen a las doce reservadas. Las decisiones quedaron congeladas antes de evaluar esas tareas.\n"]
    for name, stats in result["policies"].items():
        text.append(f"{name}: {stats['accepted']}/{stats['tasks_expected']} tareas aceptadas; costo LLM tarifario total {cost(stats['llm_tariff_total_usd'])}; por tarea aceptada {cost(stats['llm_tariff_cost_per_accepted_task_usd'])}; ahorro relativo por tarea aceptada frente a 120B fijo {pct(stats['cost_per_accepted_savings_vs_strong'])}.\n")
    if same_routes:
        text.append("Relay tomó las mismas decisiones que usar siempre 20B en la evaluación. Cualquier ahorro frente a 120B también lo obtiene el modelo barato fijo: no se demostró una ventaja incremental de routing.\n")
    elif same_strong_routes:
        text.append("La regla de desarrollo conservó el modelo fuerte. En esta evaluación no produjo ahorro frente a usarlo siempre.\n")
    else:
        text.append("Relay seleccionó una combinación de modelos. Su diferencia observada debe leerse junto con aceptación, intentos desconocidos y tamaño de muestra; no equivale a superioridad general.\n")
    bert = result["policies"].get("bert_transferred_development_calibrated")
    if (bert and relay["llm_tariff_cost_per_accepted_task_usd"] and bert["llm_tariff_cost_per_accepted_task_usd"] is not None):
        savings = 1 - bert["llm_tariff_cost_per_accepted_task_usd"] / relay["llm_tariff_cost_per_accepted_task_usd"]
        text.append(f"La referencia BERT calibrada obtuvo {bert['accepted']}/12 aceptaciones frente a {relay['accepted']}/12 de Relay y un costo de llamadas por tarea aceptada {savings*100:.2f} % menor. Su selección usa el prompt de cada tarea; Relay fija una ruta para toda la actividad. Este resultado no valida una ventaja de Relay. La inferencia local de BERT tardó {result['routing_overhead']['bert_validation_inference_seconds']:.2f} segundos en las doce tareas, más {result['routing_overhead']['bert_cold_load_seconds']:.2f} segundos de carga inicial; ese cómputo no está monetizado.\n")
    text.extend([
        "Los importes son un escenario sin caché con tokens informados por Groq y tarifas publicadas, no una factura pagada de Groq Free. Incluyen las llamadas de las tareas fallidas cuando hay uso informado; no convierten intentos sin uso en costo cero. El cómputo local, el desarrollo y la adquisición de evidencia no están monetizados.\n",
        f"La adquisición completa conservó {len(previous)} intentos previos y {len(attempts)} del protocolo principal; {result['acquisition']['unknown_usage_attempts']} no informaron uso. El subtotal tarifario conocido de toda esa adquisición es {cost(known_subtotal)} y no representa su costo completo.\n",
        "El harness usa el agente y el ciclo reales de OpenHands. Se detiene al pasar ejemplos públicos y aplica después tests reservados, sin mostrarlos al modelo. Esta revisión eliminó el cierre adicional detectado en desarrollo; no se ejecutó una comparación completa del cierre antiguo contra el nuevo.\n",
        "La comparación de políticas reutiliza ramas pareadas con el mismo agente y presupuesto. BERT es el checkpoint publicado de RouteLLM transferido al par GPT-OSS, con supervisión externa; el umbral calibrado se eligió únicamente en desarrollo. No se ejecutaron Not Diamond, MF ni SW.\n",
        "HumanEval y QuixBugs son benchmarks históricos públicos que pueden estar presentes en el entrenamiento de los modelos. Esta muestra pequeña no acredita ausencia de deterioro poblacional, adopción comercial ni resultados en repositorios grandes. La política exploratoria no se aprobó para producción y los mínimos del SDK no se redujeron.\n",
        "El [reporte JSON](harness-live-public-v2.json) conserva métricas, intentos, hashes y procedencia. El [protocolo](../docs/harness_public_pilot_v1.md) documenta la revisión antes de la validación.\n"])
    (ROOT / "reports/harness-live-public-v2.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"report": str(destination), "policies": result["policies"], "incremental": result["incremental_routing_value"]}))


if __name__ == "__main__":
    main()
