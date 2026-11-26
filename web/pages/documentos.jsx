import Head from "next/head";
import { useEffect, useRef, useState } from "react";
import { api } from "../components/api";
import { Icon } from "../components/ChatParts";
import { AnswerByline, answerBlocks } from "../components/Messages";
import WorkspaceHeader from "../components/WorkspaceHeader";

const storageKey = "relay.documents.thread";
const defaultProfile = "frozen_v2";
const initialState = { available: false, ready: false, models: [], v3_available: false };
const decisions = {
  threshold_met: "Umbral de selección alcanzado", fallback: "Ruta de respaldo", abstain: "Sin candidato elegible",
  pairwise_strong: "Mayor utilidad prevista del modelo amplio", pairwise_weak: "Mayor utilidad prevista del modelo intermedio",
  capability_fallback: "Única opción compatible con el contexto",
};

function RouteDetail({ detail }) {
  const usage = detail.usage || {};
  const number = (value, options) => value == null ? "Sin dato" : Number(value).toLocaleString("es-MX", options);
  const rows = [
    ["Decisión del router", detail.router_seconds == null ? "Sin dato" : number(detail.router_seconds * 1000, { maximumFractionDigits: 1 }) + " ms"],
    ["Generación", detail.generation_seconds == null ? "Sin dato" : number(detail.generation_seconds, { maximumFractionDigits: 1 }) + " s"],
    ["Tokens de entrada", number(usage.input_tokens)], ["Tokens de salida", number(usage.output_tokens)],
  ];
  return <details className="usage"><summary>Ruta y consumo del turno</summary>
    <dl>{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    <p>{detail.cost_usd == null ? "El costo de hardware y energía queda sin estimar." : "Costo reportado: " + number(detail.cost_usd, { maximumFractionDigits: 6 }) + " USD."}</p>
    {detail.historical_model && <p>Referencia histórica: {detail.historical_model}{detail.historical_tier != null && " · T" + detail.historical_tier}.</p>}
    {detail.decision && <p>{decisions[detail.decision] || detail.decision}.</p>}
    <p>{detail.limitation || "La selección histórica orienta esta demo. Su resultado individual requiere evaluación en los modelos locales."}</p>
  </details>;
}

export default function Documents() {
  const [profile, setProfile] = useState(defaultProfile);
  const [state, setState] = useState(initialState);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [history, setHistory] = useState([]);
  const [details, setDetails] = useState([]);
  const [draft, setDraft] = useState("");
  const [attachment, setAttachment] = useState(null);
  const [reading, setReading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [storageWarning, setStorageWarning] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);
  const fileRef = useRef(null);
  const fileVersion = useRef(0);
  const sending = useRef(false);
  const answerRef = useRef(null);
  const promptRef = useRef(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey));
      if (saved && Array.isArray(saved.history) && Array.isArray(saved.details)
        && saved.history.every((msg, i) => msg?.role === (i % 2 ? "assistant" : "user") && typeof msg.content === "string")
        && saved.history.length === saved.details.length * 2) {
        setHistory(saved.history);
        setDetails(saved.details);
      }
    } catch { /* A new tab or browser storage restriction still allows an in-memory session. */ }
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    api("documentos-state", { profile }, controller.signal).then(data => {
      if (!controller.signal.aborted) setState(data);
    }).catch(exc => {
      if (!controller.signal.aborted) setState({ ...initialState, notice: exc.message });
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [profile, revision]);

  async function attach(file) {
    if (!file || sending.current) return;
    const version = ++fileVersion.current;
    setError(""); setReading(true); setAttachment(null);
    try {
      if (!/\.(txt|md)$/i.test(file.name)) throw new Error("Elige un archivo .txt o .md.");
      let text;
      try { text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer()); }
      catch { throw new Error("No se pudo leer el archivo. Guárdalo con codificación UTF-8 y vuelve a adjuntarlo."); }
      if (fileVersion.current === version) setAttachment({ name: file.name, text });
    } catch (exc) {
      if (fileVersion.current === version) setError(exc.message);
    } finally {
      if (fileVersion.current === version) setReading(false);
    }
  }
  function removeFile() {
    fileVersion.current++; setReading(false); setAttachment(null);
    if (fileRef.current) fileRef.current.value = "";
  }
  function clearThread() {
    setConfirmReset(false); setHistory([]); setDetails([]); setDraft(""); setError(""); setStorageWarning(""); removeFile();
    try { sessionStorage.removeItem(storageKey); } catch { /* Storage is optional. */ }
    promptRef.current?.focus();
  }
  async function send(e) {
    e.preventDefault();
    if (!state.ready || loading || sending.current || reading || !draft.trim()) return;
    sending.current = true; setBusy(true); setError(""); setConfirmReset(false);
    try {
      const data = await api("documentos-send", { profile, prompt: draft, document: attachment?.text || "", history });
      const nextHistory = [...history, data.user_message, data.assistant_message];
      const nextDetails = [...details, data.detail];
      setHistory(nextHistory); setDetails(nextDetails); setDraft(""); removeFile();
      try {
        sessionStorage.setItem(storageKey, JSON.stringify({ history: nextHistory, details: nextDetails }));
        setStorageWarning("");
      } catch {
        setStorageWarning("El navegador conserva este turno en memoria. Al recargar podrías perderlo.");
      }
      requestAnimationFrame(() => answerRef.current?.scrollIntoView({ block: "nearest" }));
    } catch (exc) { setError(exc.message); }
    finally { sending.current = false; setBusy(false); }
  }
  let status = "Servicio local sin conexión";
  if (loading) status = "Consultando Ollama…";
  else if (state.ready) status = "Ollama disponible";
  else if (state.available) status = "Ollama necesita atención";
  return <>
    <Head><title>Documentos · Relay</title><meta name="viewport" content="width=device-width, initial-scale=1" /></Head>
    <div className="reading-room">
      <WorkspaceHeader active="documents" />
      <main className="reading-layout">
        <div className="reading-desk">
          <header className="reading-title">
            <h1>Mesa de lectura</h1>
            <p>Adjunta un texto y hazle preguntas con los modelos locales de Ollama.</p>
          </header>
          {history.length > 0 && <section className="reading-history" aria-label="Conversación sobre documentos">
            {history.map((msg, i) => {
              if (msg.role === "user") return <div key={i} className="reading-question">
                <span>Tú</span><p>{msg.content}</p>
              </div>;
              const detail = details[Math.floor(i / 2)];
              return <article key={i} className="reading-answer" ref={i === history.length - 1 ? answerRef : undefined}>
                <AnswerByline model={detail?.live_model} />
                <div className="answer-text">{answerBlocks(msg.content)}</div>
                {detail?.warning && <p className="reading-notice">{detail.warning}</p>}
                {detail?.finish_reason === "length" && <p className="reading-notice">La respuesta alcanzó el límite de salida y puede estar incompleta.</p>}
                {detail && <RouteDetail detail={detail} />}
              </article>;
            })}
          </section>}
          <form className="reading-editor" onSubmit={send}>
            <div className="reading-attachment" data-dragging={dragging} data-attached={Boolean(attachment)}
              onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true); }}
              onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false); }} onDrop={e => {
                e.preventDefault(); setDragging(false);
                if (busy) return;
                if (e.dataTransfer.files.length > 1) setError("Adjunta un documento por turno.");
                else attach(e.dataTransfer.files[0]);
              }}>
              <input ref={fileRef} type="file" hidden accept=".txt,.md" aria-label="Archivo de referencia"
                disabled={busy} onChange={e => { attach(e.target.files[0]); e.target.value = ""; }} />
              <button type="button" className="reading-file" disabled={busy} onClick={() => fileRef.current?.click()}
                aria-label={attachment ? "Cambiar documento adjunto" : "Adjuntar documento"}>
                <span className="reading-paper" aria-hidden="true" />
                <span><strong>{attachment?.name || (reading ? "Leyendo archivo…" : "Adjuntar documento")}</strong>
                  <small>{attachment ? "Cambiar archivo" : "Elige o arrastra un .txt o .md en UTF-8"}</small></span>
              </button>
              {attachment && <button type="button" className="reading-icon" disabled={busy} aria-label="Quitar documento" onClick={removeFile}><Icon name="close" /></button>}
            </div>
            <label htmlFor="doc-prompt">Tu pregunta</label>
            <textarea ref={promptRef} id="doc-prompt" rows={3} value={draft} readOnly={busy} onChange={e => setDraft(e.target.value)}
              placeholder="¿Qué hipótesis plantea el texto y qué evidencia las apoya?" />
            <div className="reading-send">
              <button type="submit" disabled={busy || reading || loading || !state.ready || !draft.trim()}>
                {busy ? "Generando…" : "Enviar pregunta"}<Icon name="up" /></button>
            </div>
          </form>
          {error && <p className="reading-notice reading-error" role="alert">{error}</p>}
          {storageWarning && <p className="reading-notice" role="status">{storageWarning}</p>}
          {busy && <p role="status" className="reading-notice">El modelo local está preparando la respuesta.</p>}
          {history.length > 0 && <div className="reading-reset">
            {confirmReset ? <><span>¿Borrar la conversación de esta pestaña?</span>
              <button type="button" onClick={() => setConfirmReset(false)}>Conservarla</button>
              <button type="button" disabled={busy} onClick={clearThread}>Borrar</button></> :
              <button type="button" disabled={busy} onClick={() => setConfirmReset(true)}>Reiniciar conversación</button>}
          </div>}
        </div>
        <aside className="reading-shelf" aria-label="Recursos de lectura">
          {(state.has_report || state.has_slides) && <section>
            <h2>Material del estudio</h2>
            {state.has_report && <StudyLink href="/estudio/informe.pdf" title="Guía de estudio" format="Guía en PDF" />}
            {state.has_slides && <StudyLink href="/estudio/presentacion.pdf" title="Presentación" format="Diapositivas en PDF" slides />}
          </section>}
          <section><h2>Modelos locales</h2>
            <div className="reading-state"><span data-ready={!loading && state.ready} role="status">{status}</span>
              <button className="reading-icon" type="button" aria-label="Actualizar catálogo local" disabled={busy || loading}
                onClick={() => setRevision(v => v + 1)}><Icon name="refresh" /></button></div>
            {state.notice && <p className="reading-service-note" role="status">{state.notice}</p>}
            {state.models?.length > 0 && <ul className="reading-models">{state.models.map(model => <li key={model}>{model}</li>)}</ul>}
            {(state.v3_available || profile === "experimental_v3") && <label className="reading-profile"><input type="checkbox"
              checked={profile === "experimental_v3"} disabled={busy || loading} onChange={e => setProfile(e.target.checked ? "experimental_v3" : defaultProfile)} />
              Usar el perfil v3 experimental</label>}
          </section>
        </aside>
      </main>
    </div>
  </>;
}

function StudyLink({ href, title, format, slides = false }) {
  return <a className="reading-resource" href={href} target="_blank" rel="noreferrer">
    <span className="reading-cover" data-slides={slides} aria-hidden="true" />
    <span><strong>{title}</strong><small>{format}</small></span>
    <Icon name="chevron" />
  </a>;
}
