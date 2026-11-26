export async function api(action, payload, signal) {
  const response = await fetch("/api/" + action, {
    method: "POST", cache: "no-store", signal,
    headers: { "Content-Type": "application/json", "X-Relay-Client": "relay-web" },
    body: JSON.stringify(payload),
  });
  let body;
  try { body = await response.json(); }
  catch { throw new Error("Relay no pudo leer la respuesta del servicio local."); }
  if (!response.ok || body.error) throw new Error(body.error || "Relay no pudo completar la solicitud.");
  return body;
}
