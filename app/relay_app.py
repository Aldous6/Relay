"""Relay's local demonstration. Start it explicitly with Streamlit.

Importing this module never generates a model response. A response is requested
only after the user submits the conversation form.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


STYLE = """
<style>
.stApp {
  --paper: #161d33; --rail: #11182c; --panel: #202941; --field: #182138;
  --edge: #3c4866; --ink: #f1f3ff; --muted: #c0c8de; --lavender: #bdc6ff;
  --display: "Segoe UI Variable Display", "Segoe UI", "Source Sans 3", sans-serif;
  color-scheme: dark; color: var(--ink); background: var(--paper);
  font-family: "Segoe UI Variable Text", "Segoe UI", "Source Sans 3", sans-serif;
}
[data-testid="stMain"], [data-testid="stAppViewContainer"] {background: var(--paper)}
[data-testid="stHeader"] {background: transparent; pointer-events: none}
[data-testid="stHeader"] button {pointer-events: auto}
[data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"] {display: none}
[data-testid="stSidebar"] {background: var(--rail); border-right: 1px solid #303b54}
[data-testid="stSidebarHeader"] {min-height: 2.65rem; padding-block: .55rem}
[data-testid="stSidebarUserContent"] {padding: .45rem 1.25rem 1.2rem}
.local-brand {
  display: flex; align-items: center; gap: .72rem; padding: 0 .3rem; margin: 0 0 .9rem;
  color: #fff; font: 680 1.88rem/1 var(--display); letter-spacing: -.07em;
}
.local-mark {position: relative; width: 1.05em; height: 1.05em; flex: none; overflow: hidden; color: #a9b6ff}
.local-mark::before {
  content: ""; position: absolute; left: 4%; top: 19%; width: 65%; height: 13%;
  background: currentColor; border-radius: 0 .15em .15em 0; transform: skew(-25deg) rotate(-15deg);
  box-shadow: .07em .24em 0 currentColor, .15em .48em 0 currentColor;
}
[data-testid="stMainBlockContainer"] {
  max-width: 67rem; margin-inline: auto; padding: 3.25rem clamp(1.4rem, 4.2vw, 4rem) 3.5rem;
}
.st-key-relay_local_titlebar {
  padding-bottom: 1rem; border-bottom: 1px solid var(--edge);
  align-items: center; justify-content: space-between; gap: .7rem;
}
.st-key-relay_local_titlebar h1 {
  color: var(--ink); padding: 0; font: 650 clamp(1.8rem, 3.5vw, 2.65rem)/1.1 var(--display);
  letter-spacing: -.05em;
}
.local-status {display: inline-flex; align-items: center; gap: .45rem; color: var(--muted); font-size: .85rem}
.local-status::before {content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: #dba967}
.local-status.ready::before {background: #80cdae}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: .25rem; border: 0; padding: .2rem 0 .5rem}
[data-testid="stTabs"] [data-baseweb="tab"] {
  flex: 0 0 auto; padding: .4rem .85rem; height: 2.45rem; border-radius: .55rem;
  color: var(--muted); background: transparent;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {background: #242f4a; color: white}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {background: #343d69; color: #f1f2ff}
[data-testid="stTabs"] [data-baseweb="tab"] p {color: inherit}
[data-testid="stTabs"] :is([data-baseweb="tab-border"], [data-baseweb="tab-highlight"]) {display: none}
[data-testid="stMain"] h3, [data-testid="stSidebar"] h3 {
  color: var(--ink); font: 650 clamp(1.18rem, 2.1vw, 1.55rem)/1.25 var(--display); letter-spacing: -.025em;
}
[data-testid="stForm"] {
  padding: clamp(1rem, 2.7vw, 1.75rem); border: 1px solid var(--edge); border-radius: 1rem; background: var(--panel);
}
[data-testid="stForm"] [data-baseweb="textarea"] {background: var(--field); border-color: #53617e; border-radius: .65rem}
[data-testid="stForm"] textarea {color: var(--ink); caret-color: var(--lavender); font-size: 1rem; line-height: 1.55}
[data-testid="stForm"] textarea::placeholder {color: #acb6cf}
[data-testid="stForm"] [data-baseweb="textarea"]:focus-within {border-color: var(--lavender); box-shadow: 0 0 0 1px var(--lavender)}
[data-testid="stFileUploaderDropzone"] {
  background: #1c253d; border: 1px dashed #63708d; border-radius: .65rem; color: var(--muted);
}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stDownloadButton"] button, [data-testid="stSidebar"] .stButton button {
  border: 1px solid #54617e; border-radius: .55rem; color: var(--ink); background: #29344e;
}
[data-testid="stFormSubmitButton"] button {
  min-height: 2.55rem; padding-inline: 1.25rem; border: 1px solid #bfc7ff; border-radius: .6rem;
  background: #c5cdff; color: #162149; font-weight: 650;
}
[data-testid="stFormSubmitButton"] button:enabled:hover {background: #e0e4ff; color: #101b40}
[data-testid="stFormSubmitButton"] button:disabled {background: #303952; color: #c0c8db; border-color: #444f68}
[data-testid="stForm"] [data-testid="stWidgetLabel"] p {color: var(--ink); font-weight: 600}
[data-testid="stCaptionContainer"] {color: var(--muted)}
[data-testid="stChatMessage"] {padding: .85rem 1rem; background: #1c253e; border-radius: .8rem}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background: #282e50}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {color: #172046; background: #abb8ff}
[data-testid="stMetric"] {padding: .75rem .8rem; background: #242d46; border-radius: .7rem}
[data-testid="stMetricLabel"] p {color: var(--muted)}
[data-testid="stMetricValue"] {font: 620 1.45rem/1.4 var(--display); color: var(--ink)}
[data-testid="stExpander"] details {border-color: var(--edge); background: #1c253c; border-radius: .7rem}
[data-testid="stExpander"] summary:hover {background: var(--panel)}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) > div:first-child,
[data-testid="stSidebar"] [data-baseweb="checkbox"]:has(input:checked) > div:first-child {
  background: #6676cf; border-color: #c1c9ff;
}
.stApp :focus-visible {outline: 2px solid var(--lavender); outline-offset: 2px}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] {padding: 3.4rem .95rem 2.5rem}
  .st-key-relay_local_titlebar {align-items: flex-start; flex-wrap: wrap}
  [data-testid="stTabs"] [data-baseweb="tab"] {padding: .4rem .65rem}
  [data-testid="stForm"] {padding: .85rem}
}
</style>
"""


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _v3_available():
    directory = ROOT / "reports/relay-v3-development-v1"
    summary = _read_json(directory / "summary.json")
    if not isinstance(summary, dict):
        return False
    selection = summary.get("selected_for_future_evaluation", {})
    artifact = selection.get("artifact")
    return bool(isinstance(artifact, str) and Path(artifact).name == artifact and
                Path(artifact).suffix == ".joblib" and (directory / artifact).is_file())


def _service(experimental_v3=False):
    key = "_relay_service_v3" if experimental_v3 else "_relay_service"
    if key not in st.session_state:
        from routerbench.demo import RelayDemoService

        st.session_state[key] = RelayDemoService(ROOT, research_run="reports/relay-v3-development-v1") if experimental_v3 else RelayDemoService(ROOT)
    return st.session_state[key]


def _available_names(status: dict) -> list[str]:
    names = []
    for model in status.get("available_models", []):
        value = model if isinstance(model, str) else model.get("name", model.get("model"))
        if value:
            names.append(str(value))
    return names


def _duration(value, scale: float = 1, suffix: str = "s") -> str:
    if value is None:
        return "Sin dato"
    try:
        return f"{float(value) * scale:,.1f} {suffix}"
    except (TypeError, ValueError):
        return "Sin dato"


def _usage_value(value) -> str:
    return "Sin dato" if value is None else f"{int(value):,}"


def _new_document(text: str, history: list[dict]) -> tuple[str, str | None]:
    if not text:
        return "", None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    already_present = any(message.get("document_sha256") == digest for message in history)
    return ("" if already_present else text), digest


def _render_latest_run(result: dict):
    st.divider()
    st.markdown("### La ruta de esta respuesta")
    st.write(f"**{result.get('live_model', 'Modelo no identificado')}**")
    columns = st.columns(4)
    columns[0].metric("Decisión del router", _duration(result.get("router_seconds"), 1000, "ms"))
    columns[1].metric("Generación", _duration(result.get("generation_seconds")))
    usage = result.get("usage") or {}
    columns[2].metric("Tokens de entrada", _usage_value(usage.get("input_tokens")))
    columns[3].metric("Tokens de salida", _usage_value(usage.get("output_tokens")))
    if result.get("cost_usd") is None:
        st.caption("Ejecución local. El costo de hardware y energía no está estimado.")
    else:
        st.caption(f"Costo reportado de esta ejecución: {float(result['cost_usd']):.6f} USD.")
    if result.get("warning"):
        st.info(str(result["warning"]))
    if result.get("finish_reason") == "length":
        st.warning("La respuesta alcanzó el límite de salida configurado para esta demo y puede estar incompleta.")
    with st.expander("Ver la decisión de investigación"):
        st.write(f"Modelo de referencia: {result.get('historical_model', 'Sin dato')}")
        tier = result.get("historical_tier")
        st.write(f"Grupo histórico de costo: {'T' + str(tier) if tier is not None else 'Sin dato'}")
        labels = {"threshold_met": "Umbral de selección alcanzado", "fallback": "Ruta de respaldo", "abstain": "Sin candidato elegible",
                  "pairwise_strong": "Mayor utilidad prevista del modelo amplio", "pairwise_weak": "Mayor utilidad prevista del modelo intermedio",
                  "capability_fallback": "Única opción compatible con el contexto"}
        st.write(labels.get(result.get("decision"), str(result.get("decision", "Sin dato"))))
        if result.get("router_mode") == "experimental_v3":
            st.caption("Respuesta en modo Relay v3 experimental. La selección del candidato procede de desarrollo reutilizado.")
        st.caption("La predicción histórica orienta esta demo. No es una probabilidad de éxito validada para el modelo local.")


def _conversation(service, status: dict):
    st.markdown("### Nueva consulta")
    st.write("Escribe una pregunta o adjunta un archivo .txt o .md en UTF-8. Relay selecciona un modelo del catálogo local.")
    history = st.session_state.setdefault("_relay_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("relay_prompt_form", clear_on_submit=False):
        prompt = st.text_area("Tu solicitud", placeholder="Compara las hipótesis de este documento y señala qué evidencia falta.",
                              height=130, key="relay_prompt")
        document = st.file_uploader("Documento opcional", type=["txt", "md"], key="relay_document",
                                    help="Texto UTF-8 completo, sujeto al contexto del modelo. El mismo documento se conserva una vez en el historial.")
        with st.container(horizontal=True, horizontal_alignment="right"):
            send = st.form_submit_button("Enviar a Relay", type="primary", disabled=not bool(status.get("live_ready")))

    st.caption("Demo experimental: el piloto local mostró límites de transferencia del router histórico. Este perfil aún no está validado para producción; revisa la pestaña Resultados.")
    if not status.get("live_ready"):
        st.info("La conversación necesita Ollama y los modelos del catálogo local. Puedes revisar los resultados mientras se completa la conexión.")
    if send:
        if not prompt.strip():
            st.warning("Escribe una solicitud antes de enviarla.")
            return
        try:
            text = document.getvalue().decode("utf-8-sig") if document is not None else ""
        except UnicodeDecodeError:
            st.error("El documento no está codificado en UTF-8. Guárdalo como texto UTF-8 y vuelve a cargarlo.")
            return
        text, document_digest = _new_document(text, history)
        try:
            with st.spinner("Relay está preparando la respuesta…"):
                result = service.respond(prompt, history=[dict(item) for item in history], document=text)
            answer = result.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                st.error("El modelo no devolvió una respuesta de texto. La conversación se conserva.")
                return
            model_content = ("DOCUMENTO DE REFERENCIA:\n" + text + "\n\nCONSULTA:\n" + prompt) if text else prompt
            history.extend([{"role": "user", "content": prompt, "model_content": model_content,
                             "document_sha256": document_digest}, {"role": "assistant", "content": answer}])
            st.session_state["_relay_latest_run"] = result
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as exc:
            # The backend returns user-safe messages and never includes keys.
            st.error(f"No se pudo completar la respuesta: {exc}")
    if st.session_state.get("_relay_latest_run"):
        _render_latest_run(st.session_state["_relay_latest_run"])


def _policy_table(summary: dict) -> list[dict]:
    policies = summary.get("policies", summary.get("policy_results", {}))
    if isinstance(policies, list):
        items = [(entry.get("name", str(index)), entry) for index, entry in enumerate(policies)]
    elif isinstance(policies, dict):
        items = policies.items()
    else:
        items = []
    rows = []
    reference = summary.get("fixed_reference")
    if isinstance(reference, dict):
        rows.append({"Política": "Modelo fijo de referencia", "Éxito (%)": round(100*float(reference["success_rate"]), 2),
                     "Costo histórico / 1,000": round(float(reference["cost_per_1000"]), 6)})
    labels = {"relay_primary": "Relay · Política principal", "structural_secondary": "Estructural · Secundaria",
              "semantic_secondary": "Semántica · Secundaria", "semantic_hybrid_secondary": "Semántica + estructura · Secundaria"}
    for name, entry in items:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics", entry)
        if not isinstance(metrics, dict):
            continue
        success = metrics.get("success_rate")
        cost = metrics.get("cost_per_1000")
        if success is None and cost is None:
            continue
        rows.append({"Política": labels.get(str(name), str(name)), "Éxito (%)": round(100*float(success), 2) if success is not None else None,
                     "Costo histórico / 1,000": round(float(cost), 6) if cost is not None else None})
    return rows


def _results():
    st.markdown("### Evaluación RouterBench")
    st.write("La evidencia científica se reporta por separado de las respuestas de la demo local.")
    summary = _read_json(ROOT / "reports/final/summary.json")
    if summary is None:
        st.info("Los resultados finales aún no están disponibles en esta copia. Las comparaciones de desarrollo se conservan en reports/.")
        _v3_results()
        return
    rows = _policy_table(summary)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    conclusion = summary.get("primary_conclusion", {})
    if conclusion:
        if conclusion.get("both_primary_criteria_met"):
            st.success("La política principal respalda menor costo histórico y calidad dentro del margen predefinido en este test.")
        else:
            st.info("El resultado no respalda conjuntamente los dos criterios predefinidos. Revisa el intervalo de calidad antes de concluir equivalencia.")
    st.caption("Costos históricos estimados del benchmark. No incluyen el cómputo del router ni acreditan ahorro monetario de los modelos locales.")
    with st.expander("Protocolo, incertidumbre y detalle"):
        st.json(summary, expanded=False)
    st.download_button("Descargar resultados finales", json.dumps(summary, ensure_ascii=False, indent=2),
                       file_name="relay_resultados_finales.json", mime="application/json")
    report = ROOT / "reports/final/relay_results.md"
    if report.is_file():
        st.download_button("Descargar informe de investigación", report.read_text(encoding="utf-8"),
                           file_name="relay_results.md", mime="text/markdown")
    figure = ROOT / "reports/final/figures/routing_cost_quality.png"
    if figure.is_file():
        st.image(str(figure), caption="Políticas congeladas e incertidumbre de la comparación principal.", width="stretch")
    tiers = _read_json(ROOT / "reports/final-tiers/summary.json")
    if tiers is not None:
        with st.expander("Experimento original de tres tiers"):
            result = tiers.get("primary_test_result", {})
            st.write(f"Modelo seleccionado en validación cruzada: {tiers.get('selected_model', 'Sin dato')}.")
            if result:
                st.write(f"F1-macro: {result['f1_macro']:.4f}. Recall de Tier 2: {100*result['tier2_recall']:.2f} %.")
            tier_policy = _read_json(ROOT / "reports/final-tier-policy-v1/summary.json")
            if tier_policy is not None:
                st.write(f"Al convertir los tiers en llamadas, la política obtuvo {100*tier_policy['policy']['success_rate']:.2f} % de éxito frente a {100*tier_policy['premium_reference']['success_rate']:.2f} % del Premium fijo.")
                st.caption("Esta simulación conserva el clasificador y elige representantes solo con entrenamiento. Una predicción correcta de tier no garantiza que su representante responda con éxito.")
            st.json(tiers, expanded=False)
    reference = _read_json(ROOT / "reports/routellm-bert-secondary-v1/summary.json")
    with st.expander("Comparación secundaria con RouteLLM BERT"):
        if reference is None:
            st.write("La comparación secundaria todavía está ejecutándose o no está disponible en esta copia.")
        else:
            labels = {"relay_common_pair": "Relay · par común", "routellm_bert": "RouteLLM · checkpoint BERT",
                      "fixed_strong": "GPT-4 fijo", "fixed_weak": "Mixtral fijo"}
            st.dataframe([{"Política": labels.get(name, name), "Éxito (%)": round(100*metrics["success_rate"], 2),
                           "Costo histórico / 1,000": round(metrics["cost_per_1000"], 6)}
                          for name, metrics in reference["metrics"].items()], hide_index=True, width="stretch")
            st.caption("Mismo par de candidatos y test. Relay se ajustó con datos locales y BERT usa supervisión externa; esta comparación de transferencia no aísla el efecto del método ni evalúa todas las variantes de RouteLLM.")
            st.json(reference, expanded=False)
    _v3_results()
    _local_pilot_results()
    _local_adapter_results()


def _local_adapter_results():
    directory = ROOT / "reports/local-adapter-evaluation-v1"
    comparison = _read_json(directory / "comparison.json")
    with st.expander("Adaptación a Qwen · evaluación nueva"):
        st.caption("Otra cohorte de 64 casos, con las mismas ocho plantillas y nuevas instancias. Relay y el control BERT se adaptaron antes con datos locales; el BERT publicado conserva su checkpoint de preferencias. Esta prueba no valida producción.")
        if comparison is None:
            st.write("El informe de la adaptación local todavía no está disponible en esta copia.")
            return
        labels = {"candidate": "Relay · 14 atributos adaptados", "reference": "BERT adaptado local",
                  "published_bert": "RouteLLM BERT publicado · transferencia",
                  "baseline_strong": "Siempre Qwen 1.5B", "baseline_weak": "Siempre Qwen 0.5B"}
        rows = []
        for name, label in labels.items():
            metric = comparison["policies"][name]
            exact = metric["by_grade"]["exact"]
            counts = metric["selected_model_counts"]
            rows.append({"Política": label, "Respuestas exactas": f"{exact['successes']}/{exact['n']}",
                         "Qwen 1.5B / 0.5B": f"{counts['strong']} / {counts['weak']}",
                         "Generación seleccionada (s)": round(metric["selected_generation_seconds"], 2),
                         "Tokens totales reportados": metric["tokens"]["total_tokens"]["known_total"],
                         "Fallos de generación": metric["generation_failures"]})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Las políticas comparten las mismas respuestas por caso/modelo. Los tiempos suman generación seleccionada y excluyen el router; no representan dinero ni latencia de servicio. Esta adaptación no sustituyó el perfil de conversación.")
        report = directory / "local_adapter.md"
        if report.is_file():
            st.download_button("Descargar evaluación de adaptación local", report.read_text(encoding="utf-8"),
                               file_name="relay_adaptacion_local.md", mime="text/markdown")
        st.json(comparison, expanded=False)


def _local_pilot_results():
    summary = _read_json(ROOT / "reports/local-transfer-pilot-v1/evaluation/summary.json")
    with st.expander("Prueba local con prompts nuevos"):
        st.caption("Piloto de 64 casos sintéticos y ocho plantillas. Las políticas se fijaron antes de crear los casos. Esta prueba de transferencia no valida producción ni demuestra superioridad general.")
        if summary is None:
            st.write("La colección local aún no tiene un resumen completo. No se presentan resultados parciales como finales.")
            return
        labels = {"candidate": "Relay v3", "reference": "RouteLLM BERT", "baseline_strong": "Siempre Llama 3",
                  "baseline_weak": "Siempre Qwen 2.5 1.5B"}
        rows = []
        for name, metric in summary["policies"].items():
            exact = metric["by_grade"].get("exact")
            if exact is not None:
                rows.append({"Política": labels.get(name, name), "Respuestas exactas": f"{exact['successes']}/{exact['n']}",
                             "Exactitud (%)": round(100*exact["success_rate"], 2),
                             "Tokens de salida reportados": metric["tokens"]["output_tokens"]["known_total"],
                             "Fallos de generación": metric["generation_failures"]})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Se comparte una respuesta por caso/modelo entre las políticas. Las salidas se califican por igualdad normalizada; explicaciones adicionales pueden contar como error. Los tokens usan distintos tokenizadores y el costo monetario local es desconocido.")
        report = ROOT / "reports/local-transfer-pilot-v1/local_transfer.md"
        if report.is_file():
            st.download_button("Descargar informe del piloto local", report.read_text(encoding="utf-8"),
                               file_name="relay_piloto_local.md", mime="text/markdown")
        st.json(summary, expanded=False)


def _v3_results():
    summary = _read_json(ROOT / "reports/relay-v3-development-v1/summary.json")
    with st.expander("Relay v3 · desarrollo experimental"):
        st.caption("Estos puntos se eligieron con la misma validación usada para desarrollar. No son una confirmación independiente de superioridad ni una validación del catálogo local.")
        if summary is None:
            st.write("El experimento v3 todavía no tiene un resumen completo en esta copia.")
            return
        points = []
        reference = summary.get("routellm_bert", {}).get("selected_point")
        if reference:
            points.append(("RouteLLM BERT · desarrollo", reference))
        for name, entry in summary.get("candidates", {}).items():
            point = entry.get("selected_point")
            if point:
                points.append((name, point))
        if points:
            st.dataframe([{"Candidato": label, "Éxito de desarrollo (%)": round(100*point["metrics"]["success_rate"], 2),
                           "Costo histórico / 1,000": round(point["metrics"]["cost_per_1000"], 6)}
                          for label, point in points], hide_index=True, width="stretch")
        selection = summary.get("selected_for_future_evaluation", {})
        st.write(f"Candidato conservado en esta corrida: {selection.get('candidate_id', 'Sin dato')}.")
        st.json(summary, expanded=False)


def _methodology():
    st.markdown("### Qué estamos evaluando")
    st.write("Relay estudia cómo seleccionar un modelo con información disponible antes de generar su respuesta. La investigación compara señales estructurales, léxicas y semánticas en un catálogo histórico de once LLMs.")
    st.write("Las particiones de ajuste, calibración y validación se mantienen separadas. La elección final se congela antes de evaluar el test reservado. Scores, costos observados y respuestas no entran como predictores.")
    st.write("El estudio original de tres niveles de costo se conserva como experimento separado. La selección directa de modelos es una ampliación de ese trabajo.")
    st.write("La demo usa un catálogo local distinto. Su mapeo desde grupos históricos es experimental y requiere evaluación propia antes de afirmar calidad o ahorro en producción.")
    st.caption("Una respuesta en vivo no sustituye una evaluación. La referencia secundaria usa el checkpoint BERT publicado por RouteLLM mediante inferencia local; conserva su supervisión externa y su truncamiento a 512 tokens. Su estado y resultados están en la pestaña Resultados.")


def main():
    direct_chat = "--opencode" in sys.argv[1:]
    st.set_page_config(page_title="Relay" if direct_chat else "Relay · Espacio local",
                       page_icon="↗", layout="wide")
    environment = "OpenCode"
    if not direct_chat:
        brand = st.sidebar.empty()
        environment = st.sidebar.radio("Entorno", ["Demo local", "OpenCode"], key="relay_environment", horizontal=True)
    if environment == "OpenCode":
        from relay_router.opencode_ui import render_opencode

        render_opencode(ROOT)
        return
    st.html(STYLE)
    brand.html('<div class="local-brand"><span class="local-mark" aria-hidden="true"></span>Relay</div>')
    with st.sidebar:
        st.markdown("### Selección de modelos")
        ready_v3 = _v3_available()
        if not ready_v3:
            st.session_state["_relay_use_v3"] = False
        experimental_v3 = st.checkbox("Probar Relay v3 (experimental)", value=False,
                                      key="_relay_use_v3", disabled=not ready_v3)
        if experimental_v3:
            st.caption("Versión elegida solo con desarrollo; requiere evaluación independiente. Usa las opciones intermedia y amplia del catálogo local.")
        elif not ready_v3:
            st.caption("V3 estará disponible cuando termine su resumen y exista un candidato entrenado. La versión congelada sigue activa.")
    mode = "experimental_v3" if experimental_v3 else "frozen_v2"
    if st.session_state.get("_relay_active_mode", mode) != mode:
        st.session_state.pop("_relay_latest_run", None)
    st.session_state["_relay_active_mode"] = mode
    try:
        service = _service(experimental_v3)
        status = service.status()
    except Exception:
        service, status = None, {"available_models": [], "live_ready": False}
    names = _available_names(status)
    with st.container(key="relay_local_titlebar", horizontal=True, vertical_alignment="center"):
        with st.container(width="stretch"):
            st.title("Espacio de trabajo local")
        status_class = "ready" if status.get("live_ready") else "unavailable"
        inventory = f"{len(names)} {'modelo disponible' if len(names) == 1 else 'modelos disponibles'}"
        st.html(f'<span class="local-status {status_class}">Ollama · {inventory}</span>', width="content")
    with st.sidebar:
        st.write("**Catálogo local**")
        if names:
            for name in names:
                st.write(name)
        else:
            st.caption("Sin conexión disponible")
        st.divider()
        st.caption("La selección utiliza Relay v3 experimental." if experimental_v3 else "La selección utiliza el router congelado para esta demostración.")
        if st.button("Reiniciar conversación", width="stretch"):
            st.session_state["_relay_history"] = []
            st.session_state.pop("_relay_latest_run", None)
            st.rerun()
    conversation, results, methodology = st.tabs(["Conversar", "Resultados", "Metodología"])
    with conversation:
        _conversation(service, status)
    with results:
        _results()
    with methodology:
        _methodology()


if __name__ == "__main__":
    main()
