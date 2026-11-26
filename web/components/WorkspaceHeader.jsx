import { Icon } from "./ChatParts";

export default function WorkspaceHeader({ active, documentsAvailable = true, newChat, history, recent, children }) {
  return <header className="workspace-header">
    <a className="wordmark" href="/" aria-label="Relay, inicio">Relay</a>
    {newChat}
    <nav className="workspace-nav" aria-label="Secciones de Relay">
      <a href="/" aria-current={active === "chat" ? "page" : undefined}><Icon name="chat" />Chat</a>
      {documentsAvailable && <a href="/documentos/" aria-current={active === "documents" ? "page" : undefined}>
        <Icon name="document" />Documentos</a>}
    </nav>
    {history}
    {recent}
    {children && <div className="rail-footer">{children}</div>}
  </header>;
}
