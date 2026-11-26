import { Fragment } from "react";

function inline(text) {
  return text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g).map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (link) return <a key={i} href={link[2]} target="_blank" rel="noreferrer">{link[1]}</a>;
    return part;
  });
}

const fence = /^ {0,3}```/;
const heading = /^(#{1,6})\s+(.+)$/;
const listItem = /^\s*([-+*]|\d+\.)\s+(.+)$/;
const tableRule = /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/;

function cells(row) {
  return row.trim().replace(/^\||\|$/g, "").split("|").map(cell => cell.trim());
}

export function answerBlocks(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  for (let index = 0; index < lines.length;) {
    const start = index;
    const line = lines[index++];
    if (!line.trim()) continue;
    if (fence.test(line)) {
      const code = [];
      while (index < lines.length && !fence.test(lines[index])) code.push(lines[index++]);
      if (index < lines.length) index++;
      blocks.push(<pre key={start}><code>{code.join("\n")}</code></pre>);
    } else if (line.includes("|") && tableRule.test(lines[index] || "")) {
      const columns = cells(line);
      const rows = [];
      index++;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) rows.push(cells(lines[index++]));
      blocks.push(<div className="table-scroll" key={start}><table><thead><tr>
        {columns.map((cell, i) => <th key={i} scope="col">{inline(cell)}</th>)}
      </tr></thead><tbody>{rows.map((row, n) => <tr key={n}>{columns.map((_, i) =>
        <td key={i}>{inline(row[i] || "")}</td>)}</tr>)}</tbody></table></div>);
    } else if (line.startsWith("> ")) {
      const rows = [line.slice(2)];
      while (index < lines.length && lines[index].startsWith("> ")) rows.push(lines[index++].slice(2));
      blocks.push(<blockquote key={start}>{inline(rows.join(" "))}</blockquote>);
    } else if (heading.test(line)) {
      blocks.push(<h3 key={start}>{inline(line.match(heading)[2])}</h3>);
    } else if (listItem.test(line)) {
      const first = line.match(listItem);
      const ordered = /^\d/.test(first[1]);
      const items = [first[2]];
      while (index < lines.length) {
        const item = lines[index].match(listItem);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(item[2]);
        index++;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(<List key={start} start={ordered ? parseInt(first[1], 10) : undefined}>
        {items.map((item, i) => <li key={i}>{inline(item)}</li>)}</List>);
    } else {
      const paragraph = [line];
      while (index < lines.length && lines[index].trim() && !fence.test(lines[index])
        && !heading.test(lines[index]) && !listItem.test(lines[index]) && !lines[index].startsWith("> ")
        && !(lines[index].includes("|") && tableRule.test(lines[index + 1] || ""))) paragraph.push(lines[index++]);
      blocks.push(<p key={start}>{paragraph.map((row, i) =>
        <Fragment key={i}>{i > 0 && <br />}{inline(row)}</Fragment>)}</p>);
    }
  }
  return blocks;
}

function Usage({ turn }) {
  const { result, reason } = turn;
  const usage = result.usage || {};
  const fields = [
    ["Entrada", usage.input_tokens == null ? "Sin dato" : usage.input_tokens + " tokens"],
    ["Salida", usage.output_tokens == null ? "Sin dato" : usage.output_tokens + " tokens"],
    ["Costo reportado", usage.reported_cost_usd == null ? "Sin dato" : Number(usage.reported_cost_usd).toFixed(6) + " USD"],
  ];
  if (result.duration_seconds != null) fields.push(["Duración", Number(result.duration_seconds).toFixed(1) + " s"]);
  return <details className="usage"><summary>Ruta y consumo</summary>
    {reason && <p>{reason}</p>}
    <dl>{fields.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>
    <small>OpenCode reporta los subtotales. El proveedor determina el cargo final.</small>
    {result.session_id && <p className="session-id">Sesión <code>{result.session_id}</code></p>}
    {result.tool_calls?.length > 0 && <><p className="usage-subtitle">Herramientas utilizadas</p>
      <ul>{result.tool_calls.map((tool, i) => <li key={i}>{tool.name} · {tool.status}</li>)}</ul></>}
  </details>;
}

export function AnswerByline({ model }) {
  return <div className="answer-meta">
    <strong>Relay</strong>
    {model && <span className="answer-model" aria-label={"Modelo: " + model}>{model}</span>}
  </div>;
}

export function Turn({ turn }) {
  const { result } = turn;
  return <article className="turn">
    <div className="user-message" aria-label="Tu mensaje">{turn.prompt}</div>
    <div className="assistant-message">
      <AnswerByline model={turn.model} />
      {!result ? <p className="notice">Este turno está pendiente. Actualiza el historial para ver su resultado.</p> :
        <>
          {result.status === "error" && <p className="notice error">{result.error || "OpenCode terminó con un error."}</p>}
          {result.permission_denied && <p className="notice">OpenCode rechazó una operación que requiere aprobación interactiva.</p>}
          {result.text ? <div className="answer-text">{answerBlocks(result.text)}</div> :
            result.status !== "error" && <p>OpenCode terminó sin texto de respuesta.</p>}
          <Usage turn={turn} />
        </>}
    </div>
  </article>;
}
