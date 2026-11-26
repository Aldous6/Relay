"""Build Relay's final report and static figures from completed result files.

No training, threshold search or test evaluation occurs here. The primary policy
comes from the frozen protocol; secondary policies never replace it after test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Falta un resultado completo: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"El resultado no es un objeto JSON: {path}")
    return value


def number(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("El informe no admite métricas no finitas.")
    return result


def pct(value, decimals=2) -> str:
    return f"{100*number(value):.{decimals}f} %"


def pp(value, decimals=2) -> str:
    return f"{100*number(value):+.{decimals}f} pp"


def interval(values, unit="pp") -> str:
    if len(values) != 2 or number(values[0]) > number(values[1]):
        raise ValueError("Intervalo inválido.")
    formatter = pp if unit == "pp" else (lambda value: f"{number(value):+.6f}")
    return f"[{formatter(values[0])}, {formatter(values[1])}]"


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def policy_label(policy_id: str, entry: dict, primary_id: str) -> str:
    if policy_id == primary_id:
        return f"Relay primaria ({entry['variant']})"
    return {"structural": "Estructural, secundaria", "semantic": "Semántica, secundaria",
            "semantic_hybrid": "Semántica + 14 variables, secundaria"}.get(entry.get("variant"), str(policy_id))


def optional_metric_rows(value: dict, prefix: str = "", depth: int = 0) -> list[dict]:
    """Read reported metrics without assuming which secondary policy won."""
    if depth > 3:
        return []
    if "success_rate" in value and "cost_per_1000" in value:
        return [{"label": prefix or "metrics", "success_rate": number(value["success_rate"]),
                 "cost_per_1000": number(value["cost_per_1000"])}]
    rows = []
    for key, item in value.items():
        if isinstance(item, dict) and key not in {"by_eval_name", "by_length", "files_sha256", "packages", "configuration", "protocol"}:
            rows.extend(optional_metric_rows(item, f"{prefix}.{key}".strip("."), depth+1))
    return rows


def source_record(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def export_figures(output: Path, direct: dict, tiers: dict, protocol: dict, provenance: dict) -> dict:
    """Produce two provisional report figures; no journal compliance is claimed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        return {"generated": False, "reason": f"Dependencia gráfica no disponible: {type(exc).__name__}"}

    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    primary_id = direct["primary_policy_id"]
    primary = direct["policies"][primary_id]
    comparison = primary["comparison_vs_fixed"]
    cluster = primary["cluster_sensitivity"]
    margin = number(protocol["selection"]["quality_loss_margin"])
    points = []
    for policy_id, entry in direct["policies"].items():
        points.append({"policy_id": policy_id, "label": policy_label(policy_id, entry, primary_id),
                       "success_rate": number(entry["metrics"]["success_rate"]),
                       "cost_per_1000": number(entry["metrics"]["cost_per_1000"]), "primary": policy_id == primary_id})
    fixed = [{"model": name, "success_rate": number(metrics["success_rate"]),
              "cost_per_1000": number(metrics["cost_per_1000"])} for name, metrics in direct["all_fixed_models"].items()]
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
             "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
             "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#182024",
             "axes.labelcolor": "#182024", "xtick.color": "#182024", "ytick.color": "#182024"}
    files = []
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained")
        left, right = axes
        other_fixed = [entry for entry in fixed if entry["model"] != direct["reference_model"]]
        left.scatter([entry["cost_per_1000"] for entry in other_fixed], [100*entry["success_rate"] for entry in other_fixed],
                     marker="x", s=35, color="#7b858a", label="Otros modelos fijos")
        reference = direct["fixed_reference"]
        left.scatter([reference["cost_per_1000"]], [100*reference["success_rate"]], marker="P", s=85,
                     color="#222222", label="Referencia fija elegida en fit", zorder=4)
        marks = [("o", "#0072B2"), ("s", "#D55E00"), ("^", "#009E73"), ("D", "#6D4E8E")]
        for index, point in enumerate(points):
            marker, color = ("*", "#171d1c") if point["primary"] else marks[index % len(marks)]
            left.scatter([point["cost_per_1000"]], [100*point["success_rate"]], marker=marker,
                         s=150 if point["primary"] else 65, color=color, edgecolors="#182024", linewidths=.6,
                         label=point["label"], zorder=6 if point["primary"] else 5)
        left.set(xlabel="Costo histórico estimado por 1,000 prompts", ylabel="Prompts con score ≥ 0.75 (%)",
                 title="A. Costo y éxito observados", xlim=(0, max(entry["cost_per_1000"] for entry in fixed+points)*1.10), ylim=(0, 100))
        left.grid(axis="y", color="#e3e6e7", linewidth=.6)
        left.legend(loc="lower right", fontsize=7.3, frameon=True, facecolor="white", framealpha=.96)

        estimates = [comparison["success_rate_delta_95ci"], cluster["success_rate_delta_95ci"]]
        delta = 100*number(comparison["success_rate_delta"])
        for position, (bounds, color, marker) in enumerate(zip(estimates, ["#222222", "#D55E00"], ["o", "s"])):
            low, high = [100*number(value) for value in bounds]
            right.hlines(position, low, high, color=color, linewidth=2)
            right.vlines([low, high], position-.075, position+.075, color=color, linewidth=1)
            right.plot(delta, position, marker=marker, markersize=6, color=color)
        right.axvline(0, color="#747c80", linestyle=":", linewidth=1, label="Sin diferencia")
        right.axvline(-100*margin, color="#0072B2", linestyle="--", linewidth=1.3, label=f"Margen: −{100*margin:g} pp")
        all_x = [100*number(value) for bounds in estimates for value in bounds]+[-100*margin, 0]
        padding = max(.3, (max(all_x)-min(all_x))*.15)
        right.set(xlim=(min(all_x)-padding, max(all_x)+padding), ylim=(1.6, -.6),
                  yticks=[0, 1], yticklabels=["Por prompt\n(primario)", "Por eval_name\n(sensibilidad)"],
                  xlabel="Diferencia de éxito: Relay − fijo (pp)", title="B. Política primaria, IC del 95 %")
        right.grid(axis="x", color="#e3e6e7", linewidth=.6)
        right.legend(loc="lower right", fontsize=8, frameon=False)
        fig.suptitle(f"Relay · test reservado, n={direct['n_test']:,} prompts", fontsize=13)
        path = directory / "routing_cost_quality.png"
        temporary = path.with_name(path.name + ".part")
        fig.savefig(temporary, format="png", dpi=200, facecolor="white", transparent=False,
                    metadata={"Title": "Relay held-out routing results", "Description": "Stored historical call costs and observed success; paired bootstrap intervals; no net production cost claim."})
        temporary.replace(path)
        plt.close(fig)
        files.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path), "width_inches": 11.8,
                      "height_inches": 5.2, "dpi": 200, "transparent": False,
                      "alt_text": "Puntos de costo histórico frente a éxito de las políticas congeladas y once modelos fijos; a la derecha, diferencia de éxito de la política primaria con intervalos por prompt y por grupo, cero y margen de pérdida declarado."})

        selected = tiers["selected_model"]
        selected_result = tiers["primary_test_result"]
        matrix = np.asarray(selected_result["confusion_matrix"], dtype=int)
        labels = selected_result["confusion_matrix_labels"]
        if matrix.shape != (len(labels), len(labels)) or (matrix < 0).any():
            raise ValueError("Matriz de confusión inválida.")
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(100*matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        fig, ax = plt.subplots(figsize=(6.0, 5.0), layout="constrained")
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        for row in range(len(labels)):
            for column in range(len(labels)):
                ax.text(column, row, f"{matrix[row,column]:,}\n{normalized[row,column]:.1f} %", ha="center", va="center",
                        color="white" if normalized[row,column] >= 55 else "#182024", fontsize=10)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
               xticklabels=[f"Tier {label}" for label in labels], yticklabels=[f"Tier {label}" for label in labels],
               xlabel="Tier predicho", ylabel="Tier real",
               title=f"Experimento original · {selected}\nn={selected_result['n']:,} prompts elegibles")
        colorbar = fig.colorbar(image, ax=ax, shrink=.86)
        colorbar.set_label("Porcentaje dentro de cada tier real")
        path = directory / "tier_confusion.png"
        temporary = path.with_name(path.name + ".part")
        fig.savefig(temporary, format="png", dpi=200, facecolor="white", transparent=False,
                    metadata={"Title": "Relay original tier experiment confusion matrix", "Description": "Row-normalized colors, raw counts and percentages annotated; model selected by training CV."})
        temporary.replace(path)
        plt.close(fig)
        files.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path), "width_inches": 6.0,
                      "height_inches": 5.0, "dpi": 200, "transparent": False,
                      "alt_text": "Matriz de confusión del modelo de tiers seleccionado en validación cruzada. Cada celda muestra conteo y porcentaje de su fila; el color se normaliza por tier real para hacer visible la clase minoritaria."})
    figure_data = {"routing_policies": points, "all_fixed_models": fixed,
                   "primary_difference": comparison, "cluster_sensitivity": cluster, "quality_loss_margin": margin,
                   "tier_confusion_counts": matrix.tolist(), "tier_confusion_labels": labels,
                   "tier_confusion_row_percentages": normalized.tolist()}
    atomic_text(directory / "figure_data.json", json.dumps(figure_data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    manifest = {"sources": provenance["sources"], "matplotlib_version": matplotlib.__version__,
                "figures": files, "source_data": "figure_data.json",
                "transformations": ["success fractions multiplied by 100 for percent/percentage-point displays",
                                    "all fixed and predeclared direct-policy point estimates retained",
                                    "bootstrap intervals read unchanged from evaluated summaries",
                                    "confusion heatmap normalized by true-tier row; annotations retain raw counts"],
                "uncertainty": {"level": .95, "method": "percentile paired bootstrap", **protocol["paired_uncertainty"]},
                "destination": "provisional standalone report figures; no journal requirements specified",
                "not_certified": ["scientific validity", "WCAG compliance", "publisher compliance"]}
    atomic_text(directory / "provenance.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return {"generated": True, "files": files, "provenance": "figures/provenance.json"}


def build_report(root: Path = ROOT) -> dict:
    output = root / "reports/final"
    direct_path, tiers_path = output / "summary.json", root / "reports/final-tiers/summary.json"
    # Read both complete required artifacts before creating any presentation output.
    direct, tiers = read_json(direct_path), read_json(tiers_path)
    protocol_path = root / "configs/final_protocol.json"
    protocol = read_json(protocol_path)
    if direct["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("El protocolo actual no coincide con la evaluación final.")
    if direct["policy_selection_used_test"] is not False or tiers["selection_used_test"] is not False:
        raise ValueError("No se puede presentar como congelada una selección que utilizó test.")
    if direct["outer_test_performance_evaluated"] is not True or tiers["final_test_evaluated"] is not True:
        raise ValueError("Falta una evaluación final completa.")
    primary_id = direct["primary_policy_id"]
    frozen_primary = next(entry for entry in protocol["policies"] if entry["role"] == "primary_confirmatory_comparison")
    if frozen_primary["id"] != primary_id or direct["policies"][primary_id]["variant"] != frozen_primary["variant"]:
        raise ValueError("La política primaria cambió después de fijar el protocolo.")
    primary = direct["policies"][primary_id]
    comparison = primary["comparison_vs_fixed"]
    cluster = primary["cluster_sensitivity"]
    conclusion = direct["primary_conclusion"]
    margin = number(protocol["selection"]["quality_loss_margin"])
    sources = [source_record(root, path) for path in (direct_path, tiers_path, protocol_path)]
    permutation_path = root / "reports/final-tiers/permutation_importance.parquet"
    import pandas as pd
    if sha256(permutation_path) != tiers["files_sha256"]["permutation_importance.parquet"]:
        raise ValueError("La importancia por permutación no coincide con el resultado fijado.")
    permutation = pd.read_parquet(permutation_path)
    sources.append(source_record(root, permutation_path))
    optional_path = root / "reports/routellm-bert-secondary-v1/summary.json"
    optional = read_json(optional_path) if optional_path.is_file() else None
    if optional is not None:
        if optional["test_outcomes_used_for_selection"] is not False or optional["primary_protocol_changed"] is not False or optional["n_test"] != direct["n_test"]:
            raise ValueError("La referencia secundaria no conserva los contratos declarados.")
        for name, digest in optional["files_sha256"].items():
            if Path(name).name != name or sha256(optional_path.parent / name) != digest:
                raise ValueError(f"Referencia secundaria modificada: {name}")
        sources.append(source_record(root, optional_path))
    policy_path = root / "reports/final-tier-policy-v1/summary.json"
    tier_policy = read_json(policy_path) if policy_path.is_file() else None
    if tier_policy is not None:
        if tier_policy["source_tier_summary_sha256"] != sha256(tiers_path) or tier_policy["selection_used_test"] is not False or tier_policy["n_test"] != direct["n_test"]:
            raise ValueError("P3 no corresponde al clasificador congelado.")
        for name, digest in tier_policy["files_sha256"].items():
            if Path(name).name != name or sha256(policy_path.parent / name) != digest:
                raise ValueError(f"P3 modificado: {name}")
        sources.append(source_record(root, policy_path))
    correction = output / "scope_metadata_correction.json"
    if correction.is_file():
        sources.append(source_record(root, correction))
    provenance = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                  "primary_policy_frozen_before_test": primary_id,
                  "secondary_winner_reselection": False,
                  "tool_reference": {"authors": "Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner",
                                     "year": 2026, "title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
                                     "doi": "10.48550/arXiv.2609.00065", "url": "https://arxiv.org/abs/2609.00065"}}
    figures = export_figures(output, direct, tiers, protocol, provenance)
    selected = tiers["selected_model"]
    text = [
        f"Relay · Resultados finales del estudio offline. Protocolo {protocol['protocol_id']}. Fecha de referencia: {protocol.get('client_date', 'no registrada')}.",
        f"La política primaria fue {primary['variant']} con probabilidad mínima {frozen_primary['minimum_predicted_success']:.2f}, elegida únicamente con desarrollo y congelada antes del test. Las variantes secundarias se conservan como comparaciones descriptivas y no reemplazan a la primaria según sus resultados de test.",
        f"La selección directa se evaluó sobre {direct['n_test']:,} prompts deduplicados del test, incluidos los que no tienen ningún modelo que alcance el umbral. Éxito significa que el score guardado del modelo seleccionado fue al menos {protocol['quality_score_threshold']:.2f}. La referencia fija {direct['reference_model']} fue elegida con el subconjunto de ajuste, no con test.",
        "| Política congelada | Éxito | Costo histórico / 1,000 prompts |\n| --- | ---: | ---: |",
        f"| Referencia fija: {cell(direct['reference_model'])} | {pct(direct['fixed_reference']['success_rate'])} | {number(direct['fixed_reference']['cost_per_1000']):.6f} |",
    ]
    for policy_id, entry in direct["policies"].items():
        text.append(f"| {cell(policy_label(policy_id, entry, primary_id))} | {pct(entry['metrics']['success_rate'])} | {number(entry['metrics']['cost_per_1000']):.6f} |")
    # Join table rows explicitly rather than adding paragraph gaps inside tables.
    table_end = len(text)
    initial = "\n\n".join(text[:3])+"\n\n"+"\n".join(text[3:table_end])
    paragraphs = [initial,
        f"Para la primaria, la diferencia de éxito respecto al fijo fue {pp(comparison['success_rate_delta'])}, IC del 95 % por bootstrap pareado por prompt {interval(comparison['success_rate_delta_95ci'])}. La reducción relativa del costo histórico de las llamadas fue {pct(comparison['relative_call_cost_saving'])}; su intervalo fue [{pct(comparison['relative_call_cost_saving_95ci'][0])}, {pct(comparison['relative_call_cost_saving_95ci'][1])}]. No se incluyeron cómputo del router, entrenamiento, reintentos ni tarifas actuales de proveedores. Este resultado no representa ahorro neto en producción.",
        f"El margen de pérdida admisible fue {100*margin:g} punto(s) porcentual(es), fijado como decisión operativa antes de consultar test. El criterio de calidad exigía que el límite inferior del IC de la diferencia superara −{100*margin:g} pp; el de costo exigía que el límite superior de su diferencia estuviera por debajo de cero. Calidad dentro del margen respaldada: {'sí' if conclusion['quality_noninferiority_supported'] else 'no'}. Menor costo histórico respaldado: {'sí' if conclusion['lower_historical_call_cost_supported'] else 'no'}. Ambos criterios primarios cumplidos: {'sí' if conclusion['both_primary_criteria_met'] else 'no'}. Estado de calidad reportado: {conclusion['quality_status']}. No satisfacer no inferioridad no permite afirmar calidad equivalente.",
        f"La sensibilidad secundaria agrupada por eval_name produjo un IC de diferencia de éxito {interval(cluster['success_rate_delta_95ci'])} y de costo por 1,000 {interval(cluster['cost_per_1000_delta_95ci'], 'cost')}, usando {cluster['groups']} grupos. Los intervalos por prompt y por grupo describen unidades de remuestreo distintas. Los nombres de evaluación no necesariamente equivalen a familias independientes. Las comparaciones secundarias no tienen ajuste de multiplicidad y no justifican elegir un nuevo ganador después del test.",
        f"El experimento original de tres tiers se reporta por separado sobre {tiers['test_eligible_rows']:,} prompts elegibles. Se excluyeron {tiers['excluded_ineligible']['test']} casos sin etiqueta de tier; esos casos sí permanecen en la evaluación de política directa. La selección original por F1-macro en validación cruzada fue {selected}, sin selección con test.",
    ]
    tier_table = ["| Modelo original | F1-macro CV (train) | F1-macro test | Recall Tier 2 test | F1 Tier 2 test |",
                  "| --- | ---: | ---: | ---: | ---: |"]
    for name, entry in tiers["test"].items():
        tier_table.append(f"| {cell(name)}{' (seleccionado)' if name == selected else ''} | {number(tiers['cv'][name]['metrics_mean']['f1_macro']):.4f} | {number(entry['f1_macro']):.4f} | {pct(entry['tier2_recall'])} | {number(entry['tier2_f1']):.4f} |")
    paragraphs.append("\n".join(tier_table))
    primary_tier = tiers["primary_test_result"]
    paragraphs.append(f"P1 queda respondida por la comparación de los modelos tabulares con las referencias: el seleccionado obtuvo F1-macro {number(primary_tier['f1_macro']):.4f}. Su recall de Tier 2 fue {pct(primary_tier['tier2_recall'])}; por tanto, la métrica agregada no oculta aquí que una fracción importante de Tier 2 no se recuperó. No se sustituye por el modelo con mayor recall observado después de evaluar test. La desviación entre folds es variabilidad descriptiva de folds correlacionados y no un intervalo de confianza de población.")
    importance = tiers["permutation_importance"]
    importance_table = ["| Atributo P2 | Caída media de F1 Tier 2 | SD descriptiva | Fracción de folds en top 3 |",
                        "| --- | ---: | ---: | ---: |"]
    for feature in importance["top_three"]:
        matches = permutation.loc[permutation["feature"].eq(feature)]
        if len(matches) != 1:
            raise ValueError(f"Ranking de importancia ambiguo: {feature}")
        row = matches.iloc[0]
        importance_table.append(f"| {cell(feature)} | {number(row['mean_f1_tier2_drop']):.4f} | {number(row['std_f1_tier2_drop']):.4f} | {pct(row['top3_fold_fraction'])} |")
    paragraphs.extend([f"P2 usa importancia por permutación calculada solo dentro del train, con {importance['folds']} folds y {importance['repeats_per_fold']} repeticiones por fold. Los tres atributos del ranking y su estabilidad son:",
                       "\n".join(importance_table),
                       "La caída mide dependencia predictiva del modelo, no un efecto causal ni una dificultad intrínseca del prompt. La SD y la frecuencia top 3 no son intervalos de confianza ni probabilidades de que un atributo sea importante; un tercer puesto inestable se presenta como tal."])
    if tier_policy is None:
        paragraphs.append("La política P3 del clasificador original todavía no tiene un resumen completo. La evaluación de selección directa es una ampliación separada y no sustituye esa simulación.")
    else:
        representatives = tier_policy["representative_selection"]["representatives"]
        paragraphs.append("P3 del experimento original se ejecutó con el clasificador ya congelado y representantes elegidos solo con train: " +
                          "; ".join(f"T{tier}: {model}" for tier, model in representatives.items()) +
                          ". Las rutas se persistieron antes de consultar scores y costos de test. Acierto de tier y éxito del representante son resultados diferentes. [Manifiesto P3](../final-tier-policy-v1/summary.json).")
        policy_table = ["| Política original | Éxito en todo el test | Costo histórico / 1,000 |", "| --- | ---: | ---: |"]
        for label, metrics in [("Clasificador de tiers", tier_policy["policy"]), ("Premium fijo", tier_policy["premium_reference"])]:
            policy_table.append(f"| {label} | {pct(metrics['success_rate'])} | {number(metrics['cost_per_1000']):.6f} |")
        paragraphs.append("\n".join(policy_table))
        difference = tier_policy["comparison_vs_premium"]
        paragraphs.append(f"Frente al Premium fijo, la política original tuvo una diferencia de éxito de {pp(difference['success_rate_delta'])}, IC del 95 % {interval(difference['success_rate_delta_95ci'])}, y reducción relativa de costo histórico de {pct(difference['relative_call_cost_saving'])}. El oracle del manifiesto es una cota retrospectiva que usa resultados observados y se abstiene sin costo de llamada cuando ningún modelo alcanza el umbral; no es una política desplegable ni una referencia justa de costo operativo.")
        sensitivity_table = ["| Umbral de score para evaluar las mismas llamadas | Éxito de política original | Costo histórico / 1,000 |", "| --- | ---: | ---: |"]
        for threshold, entry in tier_policy["sensitivity"]["thresholds"].items():
            metrics = entry["policy"]
            sensitivity_table.append(f"| {threshold} | {pct(metrics['success_rate'])} | {number(metrics['cost_per_1000']):.6f} |")
        paragraphs.extend(["\n".join(sensitivity_table), "Esta sensibilidad cambia únicamente la definición de éxito: conserva clasificador, representantes y llamadas. No es un estudio reentrenado por umbral. Los balances de etiquetas de train/test y los once modelos fijos están en el manifiesto P3."])
    paragraphs.append("Los scores proceden de tareas heterogéneas y un umbral común no necesariamente significa la misma calidad semántica en cada benchmark.")
    if optional is None:
        paragraphs.append("La comparación ejecutada contra RouteLLM no está disponible entre las fuentes de este informe. No se afirma superioridad frente a RouteLLM. La selección directa entre once candidatos tampoco se compara como si fuera el mismo problema que una referencia binaria entre dos LLMs.")
    else:
        paragraphs.append("Existe una referencia secundaria adicional en [el manifiesto de RouteLLM BERT](../routellm-bert-secondary-v1/summary.json). Su población, implementación y régimen de entrenamiento deben leerse con ese manifiesto. Se mantiene separada de la comparación primaria de once candidatos; su existencia no acredita una reproducción completa del artículo ni superioridad universal de Relay.")
        names = {"routellm_bert": "RouteLLM BERT", "relay_common_pair": "Relay con el par común", "fixed_strong": "GPT-4 fijo", "fixed_weak": "Mixtral fijo"}
        reference_rows = [{"label": names.get(key, key), **value} for key, value in optional["metrics"].items()]
        if reference_rows:
            secondary_table = ["| Métrica/política de la referencia secundaria | Éxito reportado | Costo histórico / 1,000 reportado |",
                               "| --- | ---: | ---: |"]
            secondary_table.extend(f"| {cell(row['label'])} | {pct(row['success_rate'])} | {row['cost_per_1000']:.6f} |" for row in reference_rows)
            paragraphs.append("\n".join(secondary_table))
            difference = optional["comparisons"]["relay_common_pair_vs_routellm_bert"]
            coverage = optional["coverage"]["test"]
            paragraphs.append(f"Relay frente a BERT obtuvo una diferencia de éxito de {pp(difference['success_rate_delta'])}, IC del 95 % {interval(difference['success_rate_delta_95ci'])}; el intervalo incluye cero. Su costo histórico fue {pct(-difference['relative_call_cost_saving'])} mayor. Esta referencia no demuestra superioridad de Relay. Ambos umbrales se seleccionaron únicamente con validación. La comparación conserva el par GPT-4/Mixtral; Relay recibió ajuste local y BERT supervisión externa, con posible contaminación de benchmarks sin resolver. BERT truncó {coverage['prompts_truncated']} de {coverage['n']} prompts al límite oficial de 512 tokens, reteniendo {pct(coverage['token_coverage'])} de los tokens. No se evalúa aquí la variante MF ni se aísla el efecto del método bajo supervisión idéntica.")
        else:
            paragraphs.append("El esquema de esa referencia no expone pares success_rate/cost_per_1000 reconocibles para esta tabla. Sus métricas se conservan en el manifiesto enlazado, sin inventar una equivalencia de tareas o una cifra de comparación.")
    paragraphs.extend([
        "La demo con modelos locales muestra una respuesta generada realmente por el modelo elegido. Su proyección desde grupos históricos a otro catálogo sigue sin validación específica de calidad y costo. Estos resultados de RouterBench no son resultados de esos modelos locales y no acreditan preparación para producción. Ahorrar dinero y consumir menos tokens son resultados distintos.",
        "Siguen pendientes el reentrenamiento por otros umbrales y una evaluación con familias completas fuera del entrenamiento. La sensibilidad de llamadas congeladas de P3 tiene un alcance más limitado. La limitación de preentrenamiento del encoder y la comparabilidad de scores se conservan. El resultado más favorable de una variante secundaria no se transforma en conclusión primaria.",
    ])
    if figures["generated"]:
        paragraphs.extend([
            "![Costo histórico y éxito, junto con diferencia primaria e intervalos de incertidumbre.](figures/routing_cost_quality.png)",
            f"Figura 1. Puntos observados sobre el mismo test, conservando todas las políticas predeclaradas y los once modelos fijos. La estrella identifica la primaria elegida antes del test. El panel derecho muestra IC del 95 % de la diferencia, con {protocol['paired_uncertainty']['replicates']:,} réplicas pareadas y semilla {protocol['paired_uncertainty']['seed']}; el remuestreo por eval_name es una sensibilidad secundaria. No se trazan curvas ni interpolaciones entre puntos aislados.",
            "![Matriz de confusión del modelo original de tiers seleccionado mediante validación cruzada.](figures/tier_confusion.png)",
            "Figura 2. Cada celda conserva el conteo bruto y su porcentaje dentro de la fila de tier real. El color normaliza por fila para comparar clases con distinto soporte. Las fuentes numéricas y la procedencia están en [figure_data.json](figures/figure_data.json) y [provenance.json](figures/provenance.json).",
        ])
    else:
        paragraphs.append(f"Las figuras no se generaron: {figures['reason']}. Las tablas conservan los resultados completos requeridos.")
    paragraphs.extend([
        "Fuentes locales: [políticas directas](summary.json), [experimento original de tiers](../final-tiers/summary.json), [protocolo congelado](../../configs/final_protocol.json). Los hashes de los archivos fuente y las transformaciones gráficas quedan en el manifiesto de este informe.",
        "Procedencia de herramientas: las guías de integridad y exportación de la skill scientific-visualization se utilizaron para estas figuras. Referencia verificada: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel y Aubrey M. Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). Esta cita documenta la herramienta y no respalda las métricas de Relay.",
    ])
    atomic_text(output / "relay_results.md", "\n\n".join(paragraphs)+"\n")
    provenance["figures"] = figures
    provenance["output_report"] = source_record(root, output / "relay_results.md")
    provenance["script_sha256"] = sha256(Path(__file__).resolve())
    atomic_text(output / "report_provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return {"report": str(output / "relay_results.md"), "figures_generated": figures["generated"],
            "primary_policy": primary_id, "source_count": len(sources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
