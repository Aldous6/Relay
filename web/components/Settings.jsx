import { useEffect, useRef, useState } from "react";
import { Icon, providerOf } from "./ChatParts";

function Toggle({ label, description, checked, onChange }) {
  return <label className="pref-toggle">
    <span><strong>{label}</strong>{description && <small>{description}</small>}</span>
    <input type="checkbox" role="switch" checked={checked} onChange={e => onChange(e.target.checked)} />
  </label>;
}

export default function Settings({ data, busy, error, close, openProject, save, refresh, reenable }) {
  const panel = useRef(null);
  const [path, setPath] = useState(data.workspace);
  const [prefs, setPrefs] = useState(data.preferences);
  useEffect(() => {
    const dialog = panel.current;
    dialog.showModal();
    return () => dialog.close();
  }, []);

  const switchPath = path.trim() !== data.workspace;
  const providerIds = Array.from(new Set(data.models.map(model => providerOf(model.id))));
  if (prefs.provider && !providerIds.includes(prefs.provider)) providerIds.push(prefs.provider);
  const choices = data.models.filter(m => (!prefs.provider || providerOf(m.id) === prefs.provider)
    && (!prefs.free_only || m.free));
  const missing = prefs.allowed.filter(id => !choices.some(m => m.id === id));

  function setPref(field, value) {
    setPrefs(previous => ({ ...previous, [field]: value }));
  }
  function toggleAllowed(id, checked) {
    setPrefs(previous => ({ ...previous, allowed: checked
      ? [...previous.allowed, id] : previous.allowed.filter(item => item !== id) }));
  }
  function submit(event) {
    event.preventDefault();
    if (busy || !path.trim()) return;
    if (switchPath) openProject(path.trim());
    else save(prefs);
  }
  function outsideClick(event) {
    if (busy || event.target !== panel.current) return;
    const rect = panel.current.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right ||
      event.clientY < rect.top || event.clientY > rect.bottom) close();
  }

  return <dialog ref={panel} className="pref-sheet" aria-labelledby="settings-title" onClick={outsideClick}
    onCancel={event => { event.preventDefault(); if (!busy) close(); }}>
    <header className="pref-header">
      <h2 id="settings-title">Ajustes del proyecto</h2>
      <button type="button" className="pref-dismiss" aria-label="Cerrar ajustes" onClick={close} disabled={busy}>
        <Icon name="close" />
      </button>
    </header>
    <form className="pref-form" onSubmit={submit}>
      <div className="pref-scroll">
        {error && <p className="notice error" role="alert">{error}</p>}
        <label className="pref-field">Carpeta del proyecto
          <span className="pref-path"><Icon name="folder" />
            <input value={path} onChange={e => setPath(e.target.value)} required disabled={busy}
              placeholder="Ruta completa a la carpeta" autoComplete="off" spellCheck={false} />
          </span>
        </label>
        {switchPath && <p className="pref-help" role="status">Abre la carpeta para ver sus modelos y ajustes.</p>}
        <fieldset className="pref-section" disabled={busy || switchPath}>
          <legend>Catálogo de modelos</legend>
          <div className="pref-catalog">
            <span>{data.models.length === 1 ? "1 modelo anunciado" : data.models.length + " modelos anunciados"} por OpenCode</span>
            <button className="pref-link" type="button" onClick={refresh}><Icon name="refresh" />Actualizar</button>
          </div>
          {data.catalog_error && <p role="alert" className="notice error">{data.catalog_error}</p>}
          <label className="pref-field">Proveedor
            <select value={prefs.provider || ""} onChange={e => setPref("provider", e.target.value || null)}>
              <option value="">Todos los proveedores</option>
              {providerIds.sort().map(id => <option key={id}>{id}</option>)}
            </select>
          </label>
          <Toggle label="Solo modelos gratuitos" description="Según las tarifas que anuncia cada proveedor."
            checked={prefs.free_only} onChange={value => setPref("free_only", value)} />
          <Toggle label="Elegir modelos permitidos" checked={prefs.restrict}
            onChange={value => setPref("restrict", value)} />
          {prefs.restrict && <fieldset className="pref-models">
            <legend>Modelos permitidos</legend>
            <div className="pref-model-list">
              {choices.map(model => <label key={model.id}>
                <input type="checkbox" checked={prefs.allowed.includes(model.id)}
                  onChange={e => toggleAllowed(model.id, e.target.checked)} />
                <span>{model.id}</span>
              </label>)}
              {missing.map(id => <label key={id}>
                <input type="checkbox" checked onChange={() => toggleAllowed(id, false)} />
                <span>{id}<small>Fuera de los filtros o del catálogo actual</small></span>
              </label>)}
              {!choices.length && !missing.length && <p className="pref-help">No hay modelos con estos filtros.</p>}
            </div>
          </fieldset>}
          {prefs.free_only && !prefs.restrict && choices.length > 0 && <details className="pref-available">
            <summary>Ver modelos gratuitos ({choices.length})</summary>
            <ul>{choices.map(model => <li key={model.id}>{model.id}</li>)}</ul>
          </details>}
          {Object.entries(data.blocked).map(([provider, through]) => <div className="pref-blocked" key={provider}>
            <p>{provider} está en pausa tras un error de acceso. Reconéctalo con <code>/connect</code> en OpenCode.</p>
            <button className="pref-link" type="button" onClick={() => reenable(provider, through)}>Volver a habilitar {provider}</button>
          </div>)}
        </fieldset>
        <fieldset className="pref-section" disabled={busy || switchPath}>
          <legend>Envío de mensajes</legend>
          <Toggle label="Revisar el modelo antes de enviar" description="Puedes aceptar la selección de Relay o elegir otro modelo disponible."
            checked={prefs.confirm} onChange={value => setPref("confirm", value)} />
        </fieldset>
      </div>
      <footer className="pref-footer">
        <button className="pref-cancel" type="button" disabled={busy} onClick={close}>Cancelar</button>
        <button className="pref-save" type="submit" disabled={busy || !path.trim()}>
          {switchPath ? "Abrir carpeta" : "Guardar ajustes"}
        </button>
      </footer>
    </form>
  </dialog>;
}
