"""Relay's local, project-scoped chat with OpenCode."""
from __future__ import annotations

from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path
import sqlite3
from uuid import uuid4

import streamlit as st

from .integrations.opencode import OpenCodeError, discover_models, run_prompt
from .opencode_chat_style import CHAT_STYLE
from .opencode_history import DEFAULT_PREFERENCES, ChatConflict, ChatHistory, last_project, provider_rejected_auth
from .opencode_selection import advertised_free, select_opencode_model


_INPUT = "opencode_chat_input"
_MARK = '<span class="relay-mark" aria-hidden="true"></span>'
_BUTTON_ESCAPES = str.maketrans({char: "\\" + char for char in "\\`*_[]<>!~|"})
_PROVIDER_NAMES = {"opencode": "OpenCode Zen", "opencode-go": "OpenCode Go"}
_STARTERS = (
    ("Entender el proyecto", ":material/account_tree:", "Explícame cómo está organizado este proyecto y dónde empieza la ejecución."),
    ("Revisar mis cambios", ":material/difference:", "Revisa los cambios pendientes del proyecto y señala posibles fallos."),
    ("Investigar un fallo", ":material/troubleshoot:", "Ayúdame a diagnosticar este fallo en el proyecto: "),
)


def _switch_chat(chat=None):
    state = st.session_state
    state["_opencode_chat"] = chat or uuid4().hex
    state.pop("_opencode_preview", None)
    state[_INPUT] = ""


def _draft(text):
    st.session_state[_INPUT] = text


def _use_starter(key, text):
    # Removing the starter row also resets its buttons on the next rerun.
    if st.session_state.get(key):
        _draft(text)


def _save_choice(history, field, key):
    try:
        history.update_preference(field, st.session_state[key])
    except (OSError, sqlite3.Error) as exc:
        st.session_state["_opencode_preference_error"] = str(exc)


def _choice(history, preferences, field):
    key = "opencode_confirm_model" if field == "confirm" else f"opencode_{field}"
    value = preferences[field]
    if key not in st.session_state or st.session_state[key] != value:
        st.session_state[key] = value
    return {"key": key, "on_change": _save_choice, "args": (history, field, key)}


def _request_catalog_check():
    st.session_state["_opencode_check_before_send"] = True


def _save_preview_model():
    st.session_state["_opencode_preview"]["manual_model"] = st.session_state["opencode_manual_model"]


def _review_model(preview, automatic, models):
    eligible = automatic["eligible_model_ids"]
    manual = preview.get("manual_model")
    options = [None, *eligible]
    if manual is not None and manual not in eligible:
        options.append(manual)
    key = "opencode_manual_model"
    if key not in st.session_state or st.session_state[key] != manual:
        st.session_state[key] = manual

    def label(model):
        if model is None:
            return f"Automático · {automatic['model']}"
        return model if model in eligible else f"{model} (ya no disponible)"

    st.selectbox("Modelo para este mensaje", options, index=options.index(manual), format_func=label,
                 key=key, on_change=_save_preview_model)
    if manual is None:
        return automatic
    try:
        return select_opencode_model(preview["prompt"], models, selected_model=manual)
    except ValueError as exc:
        st.warning(str(exc))
        return None


def _catalog(workspace, history, preferences):
    state = st.session_state
    refresh = st.button("Actualizar catálogo", key="opencode_refresh", icon=":material/refresh:")
    sending = state.pop("_opencode_check_before_send", False)
    if refresh or (sending and preferences["free_only"]) or "_opencode_catalog" not in state:
        try:
            with st.spinner("Consultando OpenCode…"):
                state["_opencode_catalog"] = discover_models(workspace)
        except (OpenCodeError, OSError, ValueError) as exc:
            state.pop("_opencode_catalog", None)
            st.error(str(exc))
            return []
    models = state["_opencode_catalog"]
    if not models:
        st.warning("OpenCode no anunció modelos. Revisa su configuración y actualiza el catálogo.")
        return []
    st.caption(f"{len(models)} modelos en el catálogo. El acceso depende de tu plan y cuota.")
    providers = sorted({m.model_id.partition("/")[0] for m in models})
    # Keep a missing selection visible so a catalog refresh cannot silently
    # replace it with the first available provider or an unapproved model.
    configured = preferences["provider"]
    if configured and configured not in providers:
        providers.append(configured)
    provider = st.selectbox("Proveedor", [None, *providers], format_func=lambda value: value or "Todos los proveedores",
                            **_choice(history, preferences, "provider"))
    if provider:
        models = [m for m in models if m.model_id.partition("/")[0] == provider]
    if st.checkbox("Solo modelos gratuitos", **_choice(history, preferences, "free_only")):
        models = [m for m in models if advertised_free(m)]
        if models:
            with st.expander(f"Modelos gratuitos del catálogo ({len(models)})"):
                st.text("\n".join(sorted(m.model_id for m in models)))
        else:
            st.warning("El proveedor seleccionado no anuncia modelos gratuitos en este catálogo.")
    if st.checkbox("Limitar a mis modelos", **_choice(history, preferences, "restrict")):
        available = {m.model_id for m in models}
        choices = sorted(available.union(preferences["allowed"]))
        allowed = st.multiselect("Modelos permitidos", choices, placeholder="Elige los de tu plan",
                                 format_func=lambda m: m if m in available else f"{m} (fuera del catálogo seleccionado)",
                                 **_choice(history, preferences, "allowed"))
        models = [m for m in models if m.model_id in allowed]
    return models


def _question(prompt):
    with st.chat_message("user"):
        st.html(f'<div class="relay-question">{escape(prompt)}</div>')


def _answer(turn):
    result = turn["result"]
    with st.chat_message("assistant"):
        st.html(f'<div class="relay-meta">{_MARK}<strong>Relay</strong>'
                f'<span class="relay-model">{escape(turn["model"])}</span></div>')
        if result is None:
            st.info("Hay un envío pendiente de confirmación. Puedes volver a abrir este chat para actualizarlo.")
            return
        if result.get("status") == "error":
            st.error(result.get("error") or "OpenCode terminó con un error.")
        if result.get("permission_denied"):
            st.warning("OpenCode rechazó una operación que requería aprobación interactiva.")
        if result.get("text"):
            st.markdown(result["text"])
        elif result.get("status") != "error":
            st.caption("OpenCode terminó sin una respuesta de texto.")
        with st.expander("Modelo y consumo"):
            st.write(turn["reason"])
            usage = result.get("usage") or {}
            cols = st.columns(3)
            for col, key, label in zip(cols, ("input_tokens", "output_tokens", "reported_cost_usd"),
                                       ("Tokens de entrada", "Tokens de salida", "USD reportados")):
                value = usage.get(key)
                if key == "reported_cost_usd" and value is not None:
                    value = f"{value:.6f}"
                col.metric(label, "Sin dato" if value is None else str(value))
            st.caption("Subtotales de OpenCode para este turno. El cargo total queda sin verificar.")
            if (seconds := result.get("duration_seconds")) is not None:
                st.caption(f"Duración de este turno: {seconds:.1f} s")
            if sid := result.get("session_id"):
                st.text(f"Sesión  {sid}")
            for tool in result.get("tool_calls", []):
                st.text(f"{tool['name']} · {tool['status']}")


def _dispatch(history, chat, turns, prompt, decision, workspace):
    try:
        turn_id = history.begin(chat, turns[-1]["id"] if turns else None, prompt, decision)
    except (ChatConflict, sqlite3.Error) as exc:
        st.error(f"No se pudo iniciar el envío. {exc}")
        return
    after_rejection = bool(turns and provider_rejected_auth(turns[-1]["result"], turns[-1]["model"]))
    # An initial rejected request has no useful conversation to replay.
    # When earlier exchanges succeeded, keep their verified OpenCode session.
    session_id = next((t["result"]["session_id"] for t in reversed(turns)
                       if t["result"] and t["result"].get("session_id")
                       and (not after_rejection or t["result"]["status"] == "completed")), None)
    model = decision["model"]
    _question(prompt)
    with st.spinner(f"OpenCode está trabajando con {model}…", show_time=True):
        try:
            args = {"session_id": session_id} if session_id else {}
            result = run_prompt(prompt, model, workspace, **args)
        except (OpenCodeError, OSError, ValueError) as exc:
            result = {"status": "error", "text": "", "error": str(exc)}
    # A local write retry must never repeat a model call.
    try:
        history.finish(chat, turn_id, result)
    except sqlite3.Error:
        st.session_state["_opencode_unsaved"] = (workspace, chat, turn_id, result)
        _answer({"model": model, "reason": decision["reason"], "result": result})
        st.error("Relay recibió el resultado y aún no pudo guardarlo. Vuelve a abrir el chat para reintentar el guardado.")
        return
    st.rerun()


def _auth_settings(history, blocked):
    for provider, rejected_turn in sorted(blocked.items()):
        name = _PROVIDER_NAMES.get(provider, provider)
        st.caption(f"{name} está en pausa en este proyecto tras un 401. "
                   f"Para volver a usarlo, abre OpenCode, escribe /connect y reconecta {name}.")
        if st.button(f"Volver a habilitar {name}", key=f"opencode_reenable_{provider}"):
            try:
                history.reenable_provider(provider, rejected_turn)
            except sqlite3.Error as exc:
                st.error(f"No se pudo guardar el cambio. {exc}")
            else:
                st.rerun()


def _auth_retry(history, chat, turns, models, workspace):
    failed = turns[-1]
    if not models:
        st.info("Para recuperar este mensaje, elige otro proveedor permitido en Ajustes de OpenCode "
                "o reconecta el que rechazó la petición.")
        return
    try:
        decision = select_opencode_model(failed["prompt"], models)
    except ValueError as exc:
        st.info(str(exc))
        return
    st.caption("Puedes volver a enviar este mensaje. El proveedor puede cobrar el nuevo intento.")
    model = decision["model"]
    if model == "opencode-go/iceman-contributor-free":
        st.caption("Iceman Contributor Free permite al proveedor usar tus mensajes y sus respuestas para mejorar "
                   "el modelo. Puedes elegir otro desde Ajustes de OpenCode. "
                   "[Condiciones de OpenCode Go](https://opencode.ai/docs/console/go/#privacy)")
    if st.button(f"Reintentar con {model}", key=f"opencode_retry_auth_{failed['id']}_{model}",
                 on_click=_request_catalog_check):
        _dispatch(history, chat, turns, failed["prompt"], decision, workspace)


@lru_cache(maxsize=1)
def _hero_image():
    asset = Path(__file__).resolve().parents[2] / "app" / "assets" / "relay-routing.png"
    return b64encode(asset.read_bytes()).decode("ascii")


def _welcome(title, instruction):
    st.html('<section class="relay-hero">'
            f'<img class="relay-hero-art" src="data:image/png;base64,{_hero_image()}" alt="">'
            f'<div class="relay-hero-copy"><h1>{escape(title)}</h1><p>{escape(instruction)}</p></div></section>')


def _project_bar(slot, workspace, status, ready=False):
    project = escape(workspace.name or str(workspace)) if workspace else "Elige una carpeta"
    path = escape(str(workspace)) if workspace else ""
    style = "ready" if ready else "limited"
    slot.html(f'<div class="relay-top"><span class="relay-project" title="{path}">'
              f'<span class="relay-folder-icon" aria-hidden="true"></span><strong>{project}</strong></span>'
              f'<span class="relay-vendor {style}">{status}</span></div>')


def _compare_examples(models, expanded):
    with st.expander("Comparar dos ejemplos sin enviar", expanded=expanded):
        for prompt in ("Dame un titular para la landing de Relay.",
                       "Define la arquitectura de la landing de Relay y explica el flujo de registro."):
            st.write(prompt)
            try:
                model = select_opencode_model(prompt, models)["model"]
            except ValueError as exc:
                st.caption(str(exc))
            else:
                st.code(model, language=None)


def render_opencode(default_workspace: Path):
    st.html(CHAT_STYLE)
    state = st.session_state
    if "opencode_directory" not in state:
        state["opencode_directory"] = str(last_project(default_workspace))
    with st.sidebar:
        st.html(f'<div class="relay-brand">{_MARK}<span class="relay-brand-name">Relay</span></div>')
        st.button("Nuevo chat", icon=":material/add:", key="opencode_new_chat",
                  width="stretch", on_click=_switch_chat)
        nav = st.container()
    with st.container(key="relay_toolbar", horizontal=True, vertical_alignment="center"):
        with st.container(width="stretch"):
            project_bar = st.empty()
        with st.popover("Ajustes de OpenCode", icon=":material/tune:"):
            directory = st.text_input("Carpeta del proyecto", key="opencode_directory")
            try:
                workspace = Path(directory).expanduser().resolve(strict=True)
                if not directory.strip() or not workspace.is_dir():
                    raise ValueError
            except (OSError, RuntimeError, ValueError):
                workspace = None
            if state.get("_opencode_workspace") != workspace:
                state["_opencode_workspace"] = workspace
                state.pop("_opencode_catalog", None)
                state.pop("opencode_history_search", None)
                _switch_chat()
            if workspace:
                if "_opencode_catalog" not in state:
                    _project_bar(project_bar, workspace, "Consultando catálogo…")
                try:
                    history = ChatHistory(workspace)
                    preferences = history.get_preferences() or DEFAULT_PREFERENCES
                except (OSError, sqlite3.Error) as exc:
                    st.error(f"No se pudieron cargar los ajustes del proyecto. {exc}")
                    return
                if error := state.pop("_opencode_preference_error", None):
                    st.error(f"No se pudo guardar el cambio de ajustes. {error}")
                models = _catalog(workspace, history, preferences)
                confirm = st.checkbox("Confirmar modelo antes de enviar", **_choice(history, preferences, "confirm"))
            auth_settings = st.container()

    if workspace is None:
        _project_bar(project_bar, None, "Sin proyecto")
        _welcome("Elige tu proyecto", "Abre los ajustes e indica la ruta a una carpeta que exista en este equipo.")
        st.chat_input("Escribe un mensaje a Relay", key=_INPUT, disabled=True)
        return
    try:
        unsaved = state.get("_opencode_unsaved")
        if unsaved:
            old_project, old_chat, old_turn, old_result = unsaved
            ChatHistory(old_project).finish(old_chat, old_turn, old_result)
            del state["_opencode_unsaved"]
        conversations = history.conversations()
        chat = state["_opencode_chat"]
        turns = history.turns(chat)
        blocked = history.blocked_providers()
    except (OSError, sqlite3.Error) as exc:
        st.error(f"No se pudo abrir el historial local de Relay. {exc}")
        st.chat_input("Escribe un mensaje a Relay", key=_INPUT, disabled=True)
        return
    with auth_settings:
        _auth_settings(history, blocked)
    send_models = [m for m in models if m.model_id.partition("/")[0] not in blocked]
    preview_only = bool(confirm and models and not send_models)
    if not preview_only:
        models = send_models
    with nav:
        st.html('<p class="relay-rail-label">Conversaciones</p>')
        matching_chats = conversations
        if len(conversations) >= 6:
            query = st.text_input("Buscar conversación", key="opencode_history_search",
                                  placeholder="Buscar conversación", label_visibility="collapsed",
                                  icon=":material/search:")
            matching_chats = [(cid, title) for cid, title in conversations if query.casefold() in title.casefold()]
        if conversations:
            with st.container(key="relay_history_list"):
                for cid, title in matching_chats:
                    # Streamlit parses button labels as Markdown; chat titles are plain user text.
                    plain_title = title.translate(_BUTTON_ESCAPES)
                    st.button(plain_title, key=f"opencode_history_{cid}", width="stretch", icon=":material/chat_bubble_outline:",
                              type="primary" if cid == chat else "tertiary",
                              on_click=_switch_chat, args=(cid,))
                if not matching_chats:
                    st.html('<p class="relay-rail-empty">Ninguna conversación coincide.</p>')
        else:
            st.html('<p class="relay-rail-empty">Aquí se guardan los chats de esta carpeta.</p>')
    status = "Solo vista previa" if preview_only else "Sin modelos"
    if send_models:
        status = "1 modelo" if len(send_models) == 1 else f"{len(send_models)} modelos"
    _project_bar(project_bar, workspace, status, ready=bool(send_models))
    if turns:
        title = next((title for cid, title in conversations if cid == chat), "Conversación")
        st.html(f'<h1 class="relay-conversation-title">{escape(title)}</h1>')
    for turn in turns:
        _question(turn["prompt"])
        _answer(turn)

    waiting = bool(turns and turns[-1]["result"] is None)
    auth_failed = bool(turns and provider_rejected_auth(turns[-1]["result"], turns[-1]["model"]))
    no_session = bool(turns and not waiting and not auth_failed
                      and not any((t["result"] or {}).get("session_id") for t in turns))
    if no_session:
        st.info("OpenCode no devolvió una sesión que se pueda continuar. Abre un chat nuevo para enviar otro mensaje.")
    if turns and not models and not auth_failed:
        st.info("Abre Ajustes de OpenCode para revisar el proveedor y actualizar su catálogo.")
    if turns and preview_only:
        st.info("Puedes comparar modelos con la vista previa local. Para recibir respuestas, reconecta el proveedor en OpenCode.")
    disabled = not models or waiting or no_session
    preview = state.get("_opencode_preview")
    if not confirm or disabled or (preview and preview["previous"] != (turns[-1]["id"] if turns else None)):
        state.pop("_opencode_preview", None)
        preview = None
    if auth_failed and not preview:
        _auth_retry(history, chat, turns, send_models, workspace)
    intro = st.empty() if not turns else None
    if intro is not None and not preview:
        with intro.container():
            if preview_only:
                _welcome("Prueba una consulta", "Puedes ver qué modelo elegiría Relay. Para recibir respuestas, reconecta tu proveedor en OpenCode.")
            elif not models:
                _welcome("Conecta un modelo", "Abre Ajustes de OpenCode para revisar tu proveedor, los filtros y el catálogo de esta carpeta.")
            else:
                _welcome("¿En qué trabajamos?", "Describe una tarea de tu proyecto. Relay elegirá un modelo para responderte.")
    if preview:
        try:
            decision = select_opencode_model(preview["prompt"], models)
        except ValueError as exc:
            st.warning(str(exc))
            state.pop("_opencode_preview", None)
            preview = None
        else:
            review = st.empty()
            with review.container(), st.container(key="relay_review_panel"):
                st.html(f'<div class="relay-review-heading">{_MARK}<h2>Revisa el envío</h2></div>'
                        f'<div class="relay-review-prompt">{escape(preview["prompt"])}</div>')
                decision = _review_model(preview, decision, models)
                if decision:
                    who = "Modelo elegido para este mensaje" if preview.get("manual_model") else "Modelo elegido por Relay"
                    st.write(f"**{who}**", decision["model"])
                    st.caption(decision["reason"])
                with st.container(horizontal=True):
                    send = st.button("Enviar con este modelo", key="opencode_confirm_send", type="primary",
                                     disabled=preview_only or decision is None, on_click=_request_catalog_check)
                    cancel = st.button("Volver a editar", key="opencode_confirm_cancel")
            previous_model = preview.get("model")
            preview["model"] = decision["model"] if decision else None
            if send and (preview_only or decision is None or previous_model != decision["model"]):
                st.warning("La ruta cambió. Revisa el modelo que aparece antes de confirmar.")
                return
            if send or cancel:
                state.pop("_opencode_preview", None)
                if cancel:
                    _draft(preview["prompt"])
                    st.rerun()
                review.empty()
                _dispatch(history, chat, turns, preview["prompt"], decision, workspace)
                return
    if preview:
        return
    placeholder = "Continúa la conversación" if turns else "Describe tu tarea o haz una pregunta…"
    if turns:
        prompt = st.chat_input(placeholder, key=_INPUT, disabled=disabled, on_submit=_request_catalog_check)
    else:
        first_input = st.empty()
        with first_input.container():
            prompt = st.chat_input(placeholder, key=_INPUT, disabled=disabled, on_submit=_request_catalog_check)
            if not disabled:
                with st.container(key="relay_starters", horizontal=True):
                    for index, (label, icon, suggestion) in enumerate(_STARTERS):
                        key = f"opencode_starter_{index}"
                        st.button(label, key=key, icon=icon, help="Prepara un borrador que puedes editar.",
                                  on_click=_use_starter, args=(key, suggestion))
                if confirm and (preferences["restrict"] or preferences["free_only"]) and len(models) >= 2:
                    _compare_examples(models, expanded=preview_only)
    if prompt and prompt.strip() and not disabled:
        try:
            decision = select_opencode_model(prompt, models)
        except ValueError as exc:
            st.warning(str(exc))
            return
        if confirm:
            state["_opencode_preview"] = {"prompt": prompt, "model": decision["model"],
                                          "previous": turns[-1]["id"] if turns else None}
            st.rerun()
        if intro is not None:
            intro.empty()
            first_input.empty()
        _dispatch(history, chat, turns, prompt, decision, workspace)
