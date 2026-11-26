import Head from "next/head";
import { useEffect, useRef, useState } from "react";
import { Composer, Icon, matchingModels, projectName, providerOf, Review, Starters } from "../components/ChatParts";
import { Turn } from "../components/Messages";
import Settings from "../components/Settings";
import History, { ConversationRow, DeleteChatDialog } from "../components/History";
import WorkspaceHeader from "../components/WorkspaceHeader";
import { api } from "../components/api";

const defaults = { provider: null, restrict: false, allowed: [], free_only: false, confirm: false };
const blank = { workspace: "", chat: "", turns: [], conversations: [], preferences: defaults, models: [], blocked: {} };

function lastLocation() {
  try { return JSON.parse(sessionStorage.getItem("relay.chat.location")) || {}; }
  catch { return {}; }
}

function Status({ children }) {
  return <p className="notice" role="status">{children}</p>;
}

export default function Chat() {
  const [data, setData] = useState(blank);
  const [draft, setDraft] = useState("");
  const [review, setReview] = useState(null);
  const [stage, setStage] = useState("loading");
  const [error, setError] = useState("");
  const [syncNeeded, setSyncNeeded] = useState(false);
  const [settings, setSettings] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [pending, setPending] = useState("");
  const drafts = useRef(new Map());
  const input = useRef(null);
  const scroll = useRef(null);
  const navToggle = useRef(null);
  const historyNewChat = useRef(null);
  const locked = useRef(false);
  const busy = stage !== "";
  const chatKey = d => JSON.stringify([d.workspace, d.chat]);
  const address = { workspace: data.workspace, chat: data.chat };
  const matches = matchingModels(data.models, data.preferences);
  const sendable = matches.filter(m => !(providerOf(m.id) in data.blocked));
  const previewOnly = data.preferences.confirm && matches.length > 0 && !sendable.length;
  const turns = data.turns;
  const last = turns.at(-1);
  const rejected = Boolean(last?.auth_rejected);
  const unfinished = Boolean(last && !last.result);
  const lostSession = last && !unfinished && !rejected && !turns.some(t => t.result?.session_id);
  const canCompose = data.workspace && !unfinished && !lostSession && !syncNeeded && (sendable.length > 0 || previewOnly);
  const title = data.conversations.find(c => c.id === data.chat)?.title || "Conversación";
  const isThread = turns.length > 0 || Boolean(pending);
  const status = stage === "loading" ? "Abriendo proyecto" : previewOnly ? "Solo vista previa" :
    sendable.length === 1 ? "1 modelo" : sendable.length ? sendable.length + " modelos" : "Sin modelos";

  function apply(value) {
    setData(value);
    try { sessionStorage.setItem("relay.chat.location", JSON.stringify({ workspace: value.workspace, chat: value.chat })); }
    catch { /* Private browsing can disable session storage. */ }
  }

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const previous = lastLocation();
        let value;
        try { value = await api("state", previous, controller.signal); }
        catch (e) {
          if (!previous.workspace || controller.signal.aborted) throw e;
          value = await api("state", {}, controller.signal);
        }
        if (!controller.signal.aborted) apply(value);
      } catch (e) {
        if (!controller.signal.aborted) setError(e.message);
      } finally {
        if (!controller.signal.aborted) setStage("");
      }
    }
    load();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (scroll.current && (turns.length || pending)) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [data.chat, turns.at(-1)?.id, turns.at(-1)?.result?.status, pending]);

  async function act(label, operation) {
    if (locked.current) return;
    locked.current = true;
    setStage(label);
    setError("");
    try { await operation(); }
    catch (e) { setError(e.message || "No se pudo contactar con Relay."); }
    finally { locked.current = false; setStage(""); }
  }

  function focusDraft(text) {
    setDraft(text);
    requestAnimationFrame(() => {
      input.current?.focus();
      input.current?.setSelectionRange(text.length, text.length);
    });
  }

  function changeChat(payload) {
    act("navigation", async () => {
      const next = await api("state", payload);
      drafts.current.set(chatKey(data), draft);
      apply(next);
      setReview(null);
      setSyncNeeded(false);
      setDraft(drafts.current.get(chatKey(next)) || "");
      setNavOpen(false);
    });
  }

  function newChat() {
    changeChat({ workspace: data.workspace, chat: crypto.randomUUID() });
  }

  function openChat(chat) {
    changeChat({ workspace: data.workspace, chat });
  }

  function requestDelete(chat) {
    setError("");
    setDeleteTarget(chat);
  }

  function deleteChat() {
    if (!deleteTarget) return;
    const target = { workspace: data.workspace, chat: deleteTarget.id };
    const deletingActive = target.chat === data.chat;
    act("deleting", async () => {
      const nextChat = deletingActive ? crypto.randomUUID() : data.chat;
      let recovered;
      try {
        await api("delete-chat", target);
      } catch (failure) {
        // The server may have committed the deletion before the response was lost.
        try { recovered = await api("state", { workspace: data.workspace, chat: nextChat }); }
        catch { throw failure; }
        if (recovered.conversations.some(chat => chat.id === target.chat)) throw failure;
      }
      drafts.current.delete(chatKey(target));
      const next = recovered ?? {
        ...data, chat: nextChat, turns: deletingActive ? [] : data.turns,
        conversations: data.conversations.filter(chat => chat.id !== target.chat),
      };
      apply(next);
      if (next.chat !== data.chat) {
        setReview(null);
        setDraft("");
        setSyncNeeded(false);
      }
      setDeleteTarget(null);
      requestAnimationFrame(() => (navOpen ? historyNewChat : navToggle).current?.focus());
    });
  }

  function refresh() {
    act("refresh", async () => {
      apply(await api("state", { ...(data.workspace ? address : {}), refresh: true }));
      setSyncNeeded(false);
      setReview(null);
    });
  }

  function openReview(prompt, manual = "", automatic = "") {
    act("routing", async () => {
      const proposal = await api("preview", { ...address, prompt, model: manual || null });
      setDraft(prompt);
      setReview({ ...proposal, prompt, manual, automatic: manual ? automatic : proposal.decision.model,
        previous: last?.id ?? null });
    });
  }

  function send(prompt, proposal = null) {
    const previous = proposal ? proposal.previous : last?.id ?? null;
    act("sending", async () => {
      setPending(prompt);
      try {
        await api("send", { ...address, prompt, previous, model: proposal?.manual || null,
          expected_model: proposal?.decision.model || null });
        setDraft("");
        drafts.current.delete(chatKey(data));
        setReview(null);
        setSyncNeeded(true);
        apply(await api("state", address));
        setSyncNeeded(false);
      } catch (original) {
        let current;
        try { current = await api("state", address); }
        catch {
          setSyncNeeded(true);
          throw new Error(original.message + " Actualiza el historial antes de volver a enviar.");
        }
        apply(current);
        setSyncNeeded(false);
        setReview(null);
        const added = current.turns.filter(t => t.id > (previous ?? 0));
        if (added.some(t => t.prompt === prompt)) {
          setDraft("");
          drafts.current.delete(chatKey(data));
        } else {
          setDraft(prompt);
          throw original;
        }
      } finally { setPending(""); }
    });
  }

  function submit() {
    if (busy || !canCompose || !draft.trim()) return;
    if (data.preferences.confirm) openReview(draft);
    else send(draft);
  }

  const composer = review ? <Review review={review} busy={busy} changeModel={model => openReview(review.prompt, model, review.automatic)}
    edit={() => { setReview(null); focusDraft(review.prompt); }} send={() => send(review.prompt, review)} /> :
    <Composer inputRef={input} draft={draft} change={setDraft} disabled={!canCompose || busy} busy={busy}
      confirm={data.preferences.confirm} submit={submit} inThread={isThread} />;
  const feedback = <>
    {error && !settings && !navOpen && !deleteTarget && <div role="alert" className="notice error">{error}
      {(!data.workspace || syncNeeded) && <button className="button" disabled={busy} onClick={refresh}>Actualizar historial</button>}
    </div>}
    {stage === "routing" && <Status>Consultando qué modelo puede atender esta tarea…</Status>}
    {unfinished && <Status>Hay un envío pendiente. <button className="button" disabled={busy} onClick={refresh}>Actualizar historial</button></Status>}
    {lostSession && <Status>OpenCode terminó sin una sesión que se pueda continuar. Abre un chat nuevo.</Status>}
    {data.workspace && !sendable.length && !error && <Status>
      {previewOnly ? "Puedes revisar la ruta de un mensaje. Hay proveedores en pausa." :
        "Revisa el catálogo de OpenCode y los filtros para habilitar un modelo."}{" "}
      <button type="button" className="button" onClick={() => setSettings(true)} disabled={busy}>Abrir ajustes</button>
    </Status>}
  </>;

  return <>
    <Head><title>Relay · Chat de proyecto</title><meta name="viewport" content="width=device-width, initial-scale=1" /></Head>
    <div className={"relay-app" + (isThread ? " has-thread" : "")}>
      <WorkspaceHeader active="chat" documentsAvailable={Boolean(data.has_documents)}
        newChat={<button className="rail-new" type="button" disabled={busy || !data.workspace} onClick={newChat}>
          <Icon name="plus" />Nuevo chat</button>}
        history={<button ref={navToggle} className="rail-action rail-history" type="button" aria-haspopup="dialog"
          onClick={() => setNavOpen(true)} aria-label="Abrir historial" title="Historial">
          <Icon name="history" /><span>Historial</span></button>}
        recent={data.conversations.length > 0 && <section className="rail-recent" aria-labelledby="recent-chats">
          <h2 id="recent-chats">Recientes</h2>
          {data.conversations.slice(0, 6).map(chat => <ConversationRow key={chat.id} chat={chat}
            current={data.chat === chat.id} busy={busy} openChat={openChat} requestDelete={requestDelete} compact />)}
        </section>}>
        <button className="rail-action" type="button" disabled={busy || !data.workspace} onClick={() => setSettings(true)} aria-haspopup="dialog" aria-label="Abrir ajustes" title="Ajustes">
          <Icon name="settings" /><span>Ajustes</span></button>
      </WorkspaceHeader>
      <main className="workspace">
        <div className="context-bar"><div className="project-name" title={data.workspace || undefined}>
          <Icon name="folder" /><span>{data.workspace ? projectName(data.workspace) : "Proyecto local"}</span></div>
          <div className="context-actions"><span className="model-status" role="status" data-ready={sendable.length > 0}>{status}</span>
            {isThread && <button className="button compact mobile-new" type="button" disabled={busy || !data.workspace} onClick={newChat}>
              <Icon name="plus" />Nuevo chat</button>}</div></div>
        <div ref={scroll} className={isThread ? "transcript-scroll" : "start-scroll"}>
          {!isThread ? <div className="start-content">
            <header className="start-heading">
              <h1>¿Qué quieres resolver?</h1>
              <p>Relay elige el modelo para cada tarea de tu proyecto.</p>
            </header>
            <section className="write-area" aria-label="Redactar mensaje">
              {composer}{feedback}
            </section>
            {!review && <Starters disabled={busy} useDraft={focusDraft} />}
          </div> :
            <section className="thread"><header className="thread-header"><h1>{title}</h1></header>
              {turns.map(turn => <Turn key={turn.id} turn={turn} />)}
              {pending && <div className="turn pending"><div className="user-message">{pending}</div>
                <p className="notice" role="status">Relay está esperando la respuesta de OpenCode…</p></div>}
            </section>}
        </div>
        {isThread && <div className="thread-dock"><div className="compose-dock">{feedback}
          {rejected && !review && !pending && sendable.length > 0 && <div className="notice">
            Puedes revisar otro modelo para el último mensaje. El proveedor puede cobrar el reintento.{" "}
            <button className="button" disabled={busy} onClick={() => openReview(last.prompt)}>Revisar reintento</button></div>}
          {!pending && composer}
        </div></div>}
      </main>
      {navOpen && <History data={data} busy={busy} error={deleteTarget ? "" : error} createChat={newChat}
        createChatRef={historyNewChat} requestDelete={requestDelete}
        close={() => { setNavOpen(false); requestAnimationFrame(() => navToggle.current?.focus()); }}
        openChat={openChat} />}
      {deleteTarget && <DeleteChatDialog chat={deleteTarget} busy={busy} error={error} confirm={deleteChat}
        cancel={() => { setDeleteTarget(null); setError(""); }} />}
      {settings && <Settings key={data.workspace} data={data} busy={busy} error={error}
        close={() => { setSettings(false); setError(""); }}
        openProject={workspace => changeChat({ workspace })} refresh={refresh}
        save={preferences => act("saving", async () => {
          apply(await api("preferences", { ...address, preferences })); setSettings(false); setReview(null);
        })} reenable={(provider, through_turn) => act("saving", async () => {
          apply(await api("reenable", { ...address, provider, through_turn })); setReview(null);
        })} />}
    </div>
  </>;
}
