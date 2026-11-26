import { useEffect, useRef, useState } from "react";
import { Icon, projectName } from "./ChatParts";

export function ConversationRow({ chat, current, busy, openChat, requestDelete, compact = false }) {
  return <div className="chat-list-row">
    <button type="button" className={compact ? "rail-chat" : "history-item"} disabled={busy}
      aria-current={current ? "true" : undefined} title={chat.title} onClick={() => openChat(chat.id)}>
      {compact ? chat.title : <><Icon name="chat" /><span>{chat.title}</span></>}
    </button>
    <button type="button" className="chat-delete" disabled={busy} aria-haspopup="dialog"
      aria-label={"Eliminar conversación: " + chat.title} title="Eliminar conversación" onClick={() => requestDelete(chat)}>
      <Icon name="trash" />
    </button>
  </div>;
}

export function DeleteChatDialog({ chat, busy, error, cancel, confirm }) {
  const dialog = useRef(null);
  const cancelButton = useRef(null);
  useEffect(() => {
    const node = dialog.current;
    node.showModal();
    cancelButton.current?.focus();
    return () => node.close();
  }, []);

  return <dialog ref={dialog} className="delete-dialog" aria-labelledby="delete-heading" aria-describedby="delete-description"
    onCancel={e => { e.preventDefault(); if (!busy) cancel(); }}>
    <h2 id="delete-heading">¿Eliminar conversación?</h2>
    <p id="delete-description">Se eliminarán los mensajes de <strong>«{chat.title}»</strong> del historial de Relay.
      Esta acción no se puede deshacer.</p>
    {error && <p className="notice error" role="alert">{error}</p>}
    <div className="delete-actions">
      <button ref={cancelButton} className="button" type="button" onClick={cancel} disabled={busy}>Cancelar</button>
      <button className="button delete-confirm" type="button" onClick={confirm} disabled={busy}>
        {busy ? "Eliminando…" : "Eliminar"}</button>
    </div>
  </dialog>;
}

export default function History({ data, busy, close, openChat, createChat, createChatRef, requestDelete, error }) {
  const dialog = useRef(null);
  const [query, setQuery] = useState("");
  const chats = data.conversations.filter(chat =>
    chat.title.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));

  useEffect(() => {
    const node = dialog.current;
    node.showModal();
    return () => node.close();
  }, []);

  return <dialog ref={dialog} className="history-dialog" aria-labelledby="history-heading"
    onCancel={e => { e.preventDefault(); close(); }} onClick={e => { if (e.target === e.currentTarget) close(); }}>
    <div className="history-panel">
      <div className="history-heading"><div><h2 id="history-heading">Historial</h2>
        {data.workspace && <p><Icon name="folder" />{projectName(data.workspace)}</p>}</div>
        <button type="button" className="icon-button" onClick={close} aria-label="Cerrar historial"><Icon name="close" /></button></div>
      <button ref={createChatRef} className="button button-primary new-chat" type="button" disabled={busy || !data.workspace}
        onClick={createChat}><Icon name="plus" />Nueva conversación</button>
      {data.conversations.length > 0 && <label className="history-search"><Icon name="search" />
        <input type="search" aria-label="Buscar conversación" placeholder="Buscar conversación" value={query}
          onChange={e => setQuery(e.target.value)} /></label>}
      {error && <p className="notice error" role="alert">{error}</p>}
      <div className="history-list" aria-label="Conversaciones guardadas">
        {chats.map(chat => <ConversationRow key={chat.id} chat={chat} current={chat.id === data.chat}
          busy={busy} openChat={openChat} requestDelete={requestDelete} />)}
        {!chats.length && <p className="history-empty">{query ? "Ninguna conversación coincide con tu búsqueda." : "Aquí aparecerán las conversaciones que inicies en este proyecto."}</p>}
      </div>
    </div>
  </dialog>;
}
