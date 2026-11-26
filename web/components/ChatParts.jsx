import { useEffect, useRef } from "react";

const paths = {
  plus: "M12 5v14M5 12h14",
  search: "M15.5 15.5 21 21M17 10.5a6.5 6.5 0 1 1-13 0 6.5 6.5 0 0 1 13 0Z",
  folder: "M3 7V5a1 1 0 0 1 1-1h5l2 3h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z",
  settings: "M3 6h9m4 0h5M3 12h3m4 0h11M3 18h12m4 0h2M12 3v6M6 9v6m9 0v6",
  menu: "M3 5h18v14H3zM9 5v14",
  close: "m6 6 12 12M18 6 6 18",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7m4-7v7",
  up: "M12 20V4m-6.5 6.5L12 4l6.5 6.5",
  branch: "M6 19V5m0 8h6a5 5 0 0 0 5-5V5M4 5a2 2 0 1 0 4 0 2 2 0 0 0-4 0m11 0a2 2 0 1 0 4 0 2 2 0 0 0-4 0",
  changes: "M5 5h9v12H5zM10 17v3h9V8h-3M8 9h3m-3 4h3",
  debug: "m3 17 4-5 4 3 6-10m-3 0h4v4m-15 11h18",
  chat: "M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9l-5 4v-4H3V6a2 2 0 0 1 2-2Z",
  document: "M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6",
  route: "M3 17h4c4 0 4-10 9-10h4m-4-4 4 4-4 4",
  refresh: "M20 7v5h-5M4 17v-5h5m-4.5-3a8 8 0 0 1 13-3l2.5 3M4.5 15l2 2a8 8 0 0 0 13-2",
  chevron: "m9 5 7 7-7 7",
  history: "M3.3 11a8.6 8.6 0 1 1 1.6 6.1M3 5v6h6m3-4v5l3.5 2",
};

export function Icon({ name }) {
  return <svg className="icon" aria-hidden="true" viewBox="0 0 24 24" width="20" height="20" fill="none"
    stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round">
    <path d={paths[name]} />
  </svg>;
}

export function projectName(path) {
  const parts = (path || "").split(/[\\/]/).filter(Boolean);
  return parts.at(-1) || path || "Proyecto";
}

export function providerOf(id) {
  return id.split("/")[0];
}

export function matchingModels(models, prefs) {
  return models.filter(m => (!prefs.provider || providerOf(m.id) === prefs.provider)
    && (!prefs.free_only || m.free) && (!prefs.restrict || prefs.allowed.includes(m.id)));
}

export function Composer({ inputRef, draft, change, disabled, busy, confirm, submit, inThread }) {
  const formRef = useRef(null);
  useEffect(() => {
    const node = inputRef.current;
    if (node) {
      node.style.height = "auto";
      node.style.height = Math.min(node.scrollHeight, 155) + "px";
    }
  }, [draft, inputRef]);
  const label = confirm ? "Revisar modelo" : "Enviar mensaje";
  return <form ref={formRef} className="composer" onSubmit={e => { e.preventDefault(); submit(); }}>
    <textarea ref={inputRef} value={draft} onChange={e => change(e.target.value)}
      id="relay-prompt" aria-label="Mensaje para Relay" placeholder={inThread ? "Continúa la conversación…" : "Escribe lo que quieres hacer en este proyecto…"} rows={2}
      readOnly={busy} onKeyDown={e => {
        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing
          && (e.ctrlKey || e.metaKey || window.matchMedia("(pointer: fine)").matches)) {
          e.preventDefault();
          if (!disabled && draft.trim()) formRef.current.requestSubmit();
        }
      }} />
    <div className="composer-footer"><span className="composer-mode"><Icon name="route" />
      {confirm ? "Revisión de modelo" : "Modelo automático"}</span>
      <button type="submit" className="send-button" disabled={disabled || !draft.trim()} aria-label={label} title={label}>
        <span>{confirm ? "Revisar" : "Enviar"}</span><Icon name="up" />
      </button>
    </div>
  </form>;
}

const starters = [
  ["branch", "Entender el proyecto", "Explícame cómo está organizado este proyecto y por dónde empezar a leerlo."],
  ["changes", "Revisar cambios", "Revisa los cambios pendientes y señala posibles fallos. Explica tus hallazgos antes de editar archivos."],
  ["debug", "Investigar un fallo", "Ayúdame a diagnosticar este fallo en el proyecto: "],
];

export function Starters({ useDraft, disabled }) {
  return <aside className="starters" aria-label="Tareas sugeridas">{starters.map(([icon, label, prompt]) =>
    <button className="starter" type="button" key={label} disabled={disabled} onClick={() => useDraft(prompt)}>
      <Icon name={icon} /><span>{label}</span>
    </button>)}</aside>;
}

export function Review({ review, busy, changeModel, edit, send }) {
  const selector = useRef(null);
  useEffect(() => { selector.current?.focus(); }, []);
  return <section className="review" aria-labelledby="review-heading">
    <h2 id="review-heading">Revisa el envío</h2>
    <blockquote className="review-prompt">{review.prompt}</blockquote>
    <label className="field">Modelo para este mensaje
      <select ref={selector} disabled={busy} value={review.manual || ""} onChange={e => changeModel(e.target.value)}>
        <option value="">Automático · {review.automatic}</option>
        {review.decision.eligible_model_ids.map(id => <option key={id}>{id}</option>)}
      </select>
    </label>
    <p>{review.decision.reason}</p>
    {review.preview_only && <p className="notice">La vista previa está disponible. Para generar una respuesta, reconecta un proveedor en OpenCode.</p>}
    {review.decision.model === "opencode-go/iceman-contributor-free" && <p className="notice">
      Este modelo permite al proveedor usar tus mensajes y respuestas para mejorar el modelo.{" "}
      <a href="https://opencode.ai/docs/console/go/#privacy" target="_blank" rel="noreferrer">Condiciones del proveedor</a>
    </p>}
    <div className="review-actions">
      <button className="button" type="button" disabled={busy} onClick={edit}>Volver a editar</button>
      <button className="button button-primary" type="button" disabled={busy || review.preview_only} onClick={send}>
        Enviar con este modelo
      </button>
    </div>
  </section>;
}
