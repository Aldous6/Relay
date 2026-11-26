"""Visual theme for the local OpenCode workspace and its portalled controls."""

CHAT_STYLE = """
<style>
:root {
  --relay-paper: #f5f6fa;
  --relay-ink: #182044;
  --relay-muted: #525d76;
  --relay-line: #d4d8e5;
  --relay-indigo: #4148c8;
  --relay-ultramarine: #2633a9;
  --relay-rail: #192039;
  --relay-font: "Segoe UI Variable Text", "Segoe UI", "Source Sans 3", sans-serif;
  --relay-display: "Segoe UI Variable Display", "Segoe UI", "Source Sans 3", sans-serif;
}
.stApp, [data-testid="stPopoverBody"] {
  color-scheme: light;
  font-family: var(--relay-font);
  color: var(--relay-ink);
  --surface: white; --text: var(--relay-ink); --secondary: var(--relay-muted);
  --line: var(--relay-line); --action: var(--relay-indigo); --hover: #edf0fb;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--relay-paper);
}
[data-testid="stDecoration"], [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {display: none}
[data-testid="stHeader"] {background: transparent; height: 2.65rem; pointer-events: none}
[data-testid="stHeader"] button {pointer-events: auto}
[data-testid="stExpandSidebarButton"] {
  background: white; border: 1px solid var(--relay-line); border-radius: .55rem;
}
[data-testid="stExpandSidebarButton"] span {color: var(--relay-ink)}
[data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {
  width: 100%; max-width: 68rem; margin-inline: auto;
  padding-inline: clamp(1.35rem, 4vw, 4.25rem); box-sizing: border-box;
}
[data-testid="stMainBlockContainer"] {padding-top: 3.1rem; padding-bottom: 3.5rem}
[data-testid="stBottom"], [data-testid="stBottom"] > div {background: var(--relay-paper)}
[data-testid="stBottomBlockContainer"] {padding-top: .75rem; padding-bottom: 1.2rem}

[data-testid="stSidebar"] {
  color-scheme: dark; background: var(--relay-rail); color: #f5f6fe;
  --surface: #252d49; --text: #f5f6fe; --secondary: #c0c6dc;
  --line: #46516f; --action: #bdc5ff; --hover: #333d60;
}
[data-testid="stSidebarContent"] {background: var(--relay-rail); padding-bottom: 1.5rem}
[data-testid="stSidebarHeader"] {min-height: 2.6rem; padding-block: .55rem}
[data-testid="stSidebarCollapseButton"] button {color: var(--secondary)}
[data-testid="stSidebarCollapseButton"] button span {color: inherit}
[data-testid="stSidebarUserContent"] {padding: .45rem 1.25rem 1.2rem}
.relay-brand {
  display: inline-flex; align-items: center; gap: .72rem; color: #fff;
  padding: 0 .35rem; margin: 0 0 1.05rem; font: 680 1.88rem/1 var(--relay-display);
  letter-spacing: -.07em;
}
.relay-mark {
  position: relative; display: inline-block; width: 1.05em; height: 1.05em;
  flex: 0 0 auto; color: #a9b6ff; overflow: hidden;
}
.relay-mark::before {
  content: ""; position: absolute; left: 4%; top: 19%; width: 65%; height: 13%;
  background: currentColor; border-radius: 0 .15em .15em 0; transform: skew(-25deg) rotate(-15deg);
  box-shadow: .07em .24em 0 currentColor, .15em .48em 0 currentColor;
}
.relay-rail-label, .relay-section, .relay-rail-empty {line-height: 1.45}
.relay-rail-label, [data-testid="stSidebar"] .relay-section {
  color: #c6cde1; font-weight: 650; font-size: .82rem; margin: 1.25rem .3rem .05rem;
}
.relay-rail-empty {
  max-width: 24ch; margin: 0 .3rem; color: #aeb8d1; font-size: .86rem;
}
.st-key-opencode_new_chat button {
  width: 100%; justify-content: flex-start; min-height: 2.65rem; padding-inline: .85rem;
  border: 1px solid #bfc6ff; border-radius: .65rem; background: #dce1ff; color: #182252;
  font-weight: 650;
}
.st-key-opencode_new_chat button:hover {border-color: #fff; background: #f1f3ff; color: #111a44}
.st-key-relay_history_list {max-height: 52vh; overflow: auto; scrollbar-width: thin; scrollbar-color: #566384 transparent}
.st-key-relay_history_list, .st-key-relay_history_list [data-testid="stVerticalBlock"] {gap: .17rem}
.st-key-relay_history_list button {
  min-height: 2.45rem; justify-content: flex-start; text-align: left; padding: .35rem .65rem;
  background: transparent; border: 1px solid transparent; border-radius: .55rem; color: #c5cbe0;
}
.st-key-relay_history_list button:hover, .st-key-relay_history_list button[kind="primary"] {
  background: #2e3858; color: #fff;
}
.st-key-relay_history_list button[kind="primary"] {border-inline-start: 3px solid #aab8ff}
.st-key-relay_history_list button [data-testid="stMarkdownContainer"] {min-width: 0}
.st-key-relay_history_list button p {
  display: block; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.st-key-relay_toolbar {
  width: 100%; align-items: center; justify-content: space-between; flex-wrap: nowrap;
  gap: .75rem; border-bottom: 1px solid var(--relay-line); padding-bottom: .9rem; margin-bottom: .65rem;
}
.st-key-relay_toolbar > :has(.relay-top) {min-width: 0; flex: 1 1 auto}
.st-key-relay_toolbar [data-testid="stPopover"] {width: auto}
.relay-top {
  min-width: 0; display: flex; align-items: center; flex-wrap: wrap; gap: .4rem 1.1rem;
  color: var(--relay-muted); font-size: .89rem;
}
.relay-project {display: inline-flex; align-items: center; gap: .55rem; min-width: 0; max-width: 100%}
.relay-project strong {overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--relay-ink)}
.relay-folder-icon {
  display: inline-block; width: 1.2rem; height: .86rem; position: relative; flex: none;
  border: 1.6px solid #646b95; border-radius: .17rem; margin-top: .15rem;
}
.relay-folder-icon::before {
  content: ""; position: absolute; width: 42%; height: 25%; left: -1.6px; top: -.25rem;
  background: var(--relay-paper); border: 1.6px solid #646b95; border-bottom: 0;
  border-radius: .12rem .16rem 0 0;
}
.relay-vendor {display: inline-flex; align-items: center; gap: .4rem; font-size: .8rem; color: var(--relay-muted)}
.relay-vendor::before {content: ""; width: .48rem; height: .48rem; flex: none; background: #767f94; border-radius: 50%}
.relay-vendor.ready::before {background: #26815f}
.relay-vendor.limited::before {background: #ad6925}
.st-key-relay_toolbar [data-testid="stPopoverButton"] {
  min-height: 2.3rem; border: 1px solid #cbd0e0; border-radius: .58rem;
  background: white; color: var(--relay-ink); white-space: nowrap;
}
.st-key-relay_toolbar [data-testid="stPopoverButton"]:hover {border-color: var(--relay-indigo); color: var(--relay-indigo)}

.relay-hero {
  isolation: isolate; position: relative; overflow: hidden; display: flex; align-items: center;
  min-height: clamp(14rem, 24vw, 19rem); padding: clamp(1.55rem, 3.8vw, 3.2rem);
  box-sizing: border-box; border-radius: 1.35rem; background: var(--relay-ultramarine);
}
.relay-hero-art {
  position: absolute; z-index: -1; right: 0; top: 0; height: 100%; width: 57%;
  max-width: none; object-fit: cover; object-position: center; pointer-events: none;
}
.relay-hero-copy {max-width: 48%}
.relay-hero-copy h1 {
  max-width: 10ch; padding: 0; margin: 0; color: white; font: 680 clamp(2.6rem, 5.1vw, 4.15rem)/.99 var(--relay-display);
  letter-spacing: -.062em; overflow-wrap: normal;
}
.relay-hero-copy p {
  max-width: 34ch; padding: 0; margin: 1rem 0 0; color: #eef0ff;
  font-size: clamp(.91rem, 1.4vw, 1.04rem); line-height: 1.5;
}
.relay-conversation-title {
  color: var(--relay-ink); margin: .6rem 0 .7rem; padding: 0; font: 650 clamp(1.6rem, 3vw, 2.25rem)/1.2 var(--relay-display);
  letter-spacing: -.045em; overflow-wrap: anywhere;
}
.relay-section {color: var(--relay-muted); font-size: .88rem; font-weight: 650; margin: .2rem 0}
.st-key-relay_starters {justify-content: flex-start; flex-wrap: wrap; gap: .5rem}
.st-key-relay_starters button {
  display: flex; min-height: 2.6rem; padding: .45rem .85rem; border: 1px solid #d3d7e5;
  border-radius: .66rem; background: white; color: #323c64; justify-content: flex-start;
}
.st-key-relay_starters button [data-testid="stIconMaterial"] {color: var(--relay-indigo)}
.st-key-relay_starters button:hover {border-color: #a4abe4; background: #eef0fc; color: #28339b}

[data-testid="stChatInput"] {
  padding: .4rem; border: 1px solid #aeb6cf; border-radius: 1.12rem; background: white;
  box-shadow: 0 4px 14px rgb(31 42 85 / 6%);
}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
  background: transparent; border: none; box-shadow: none;
}
[data-testid="stChatInput"]:focus-within {border-color: var(--relay-indigo); box-shadow: 0 0 0 2px #d5d9ff}
[data-testid="stChatInput"] textarea {
  background: transparent; color: var(--relay-ink); caret-color: var(--relay-indigo);
  -webkit-text-fill-color: currentColor; font: 1rem/1.5 var(--relay-font);
}
[data-testid="stChatInput"] textarea::placeholder {color: #5d677e; opacity: 1}
[data-testid="stChatInputSubmitButton"] {
  height: 2.45rem; width: 2.45rem; margin: .12rem; border: 0;
  background: var(--relay-indigo); border-radius: .68rem; color: white;
}
[data-testid="stChatInputSubmitButton"]:disabled {background: #eaedf6; color: #596681}
[data-testid="stChatInputSubmitButton"]:enabled:hover {color: white; background: #3038aa}
[data-testid="stChatInput"]:has(textarea:disabled) {background: #edeff5; border-color: #c7cddd; box-shadow: none}
[data-testid="stChatMessage"] {background: transparent; border: none; padding: .5rem 0; gap: 0}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {display: none}
[data-testid="stChatMessage"]:has(.relay-question) {width: fit-content; max-width: 84%; margin-inline-start: auto}
[data-testid="stChatMessageContent"] {min-width: 0}
.relay-question, .relay-review-prompt {white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.55}
.relay-question {
  padding: .74rem 1.05rem; color: #28254e; background: #e6e8fa;
  border-radius: 1.15rem 1.15rem .28rem 1.15rem;
}
.relay-meta {
  display: flex; align-items: center; flex-wrap: wrap; gap: .48rem;
  margin: .15rem 0 .12rem; color: var(--relay-ink); font-size: .95rem;
}
.relay-meta .relay-mark, .relay-review-heading .relay-mark {color: var(--relay-indigo); font-size: 1.3rem}
.relay-model {font-size: .8rem; color: var(--relay-muted); overflow-wrap: anywhere}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {color: var(--relay-ink)}
[data-testid="stChatMessageContent"] :is(p, li) {line-height: 1.65}
[data-testid="stChatMessageContent"] :is(h1, h2, h3, h4, h5, h6) {color: var(--relay-ink)}
[data-testid="stChatMessageContent"] a, [data-testid="stCaptionContainer"] a {color: var(--action); text-decoration: underline}
[data-testid="stChatMessageContent"] :not(pre) > code {background: #e9ecf5; color: #343565}
[data-testid="stChatMessageContent"] blockquote {color: var(--relay-muted); border-inline-start-color: #aab1d6}
[data-testid="stChatMessageContent"] table {color: var(--relay-ink); background: white}
[data-testid="stChatMessageContent"] :is(td, th) {border-color: var(--relay-line); overflow-wrap: anywhere}
[data-testid="stChatMessageContent"] th {background: #e9ecf5}
[data-testid="stMain"] [data-testid="stCode"] {color-scheme: dark}

.st-key-relay_review_panel {
  background: white; color: var(--relay-ink); border: 1px solid #cdd1e7;
  border-inline-start: 4px solid var(--relay-indigo); border-radius: 1rem;
  padding: clamp(1rem, 3vw, 1.75rem);
}
.relay-review-heading {display: flex; align-items: center; gap: .55rem}
.relay-review-heading h2 {
  margin: 0; padding: 0; color: var(--relay-ink); font: 650 1.45rem/1.2 var(--relay-display); letter-spacing: -.035em;
}
.relay-review-prompt {margin-top: .8rem; padding: .75rem .95rem; border-radius: .65rem; background: #f0f1fb; color: #393d65}
[data-testid="stCaptionContainer"] {color: var(--secondary)}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stMetric"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: var(--text)}
[data-testid="stMetricValue"] {font-family: var(--relay-display); font-size: 1.65rem; font-weight: 600}
[data-testid="stExpander"] details {border: 1px solid var(--line); background: var(--surface); border-radius: .75rem}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary span {color: var(--text)}
[data-testid="stExpander"] summary:hover {background: var(--hover); border-radius: .75rem}
[data-testid="stMain"] [data-testid="stExpanderDetails"] {color: var(--relay-ink)}
:is(.stApp, [data-testid="stPopoverBody"]) :is([data-baseweb="input"], [data-baseweb="select"] > div) {
  border-color: var(--line); border-radius: .6rem; background: var(--surface); color: var(--text);
}
:is(.stApp, [data-testid="stPopoverBody"]) :is([data-baseweb="input"], [data-baseweb="select"]) :is(div, input) {
  color: var(--text);
}
:is(.stApp, [data-testid="stPopoverBody"]) [data-baseweb="base-input"] {background: transparent}
:is(.stApp, [data-testid="stPopoverBody"]) [data-baseweb="input"] input {background: transparent; color: var(--text)}
:is(.stApp, [data-testid="stPopoverBody"]) [data-baseweb="input"] input::placeholder {color: var(--secondary)}
:is(.stApp, [data-testid="stPopoverBody"]) [data-baseweb="select"] svg {fill: var(--secondary)}
:is(.stApp, [data-testid="stPopoverBody"]) [data-baseweb="select"] [data-baseweb="tag"] {
  background: #e8eafa; color: #323a7c;
}
:is(.stApp, [data-testid="stPopoverBody"]) [data-baseweb="select"] [data-baseweb="tag"] span {color: #323a7c}
[data-testid="stPopoverBody"] {
  width: min(24rem, calc(100vw - 2rem)); background: white; border: 1px solid var(--relay-line);
  border-radius: .9rem; box-shadow: 0 .6rem 2rem rgb(27 33 73 / 16%);
}
[data-testid="stPopoverBody"] > div {background: white; color: var(--relay-ink)}
[data-testid="stPopoverBody"] [data-testid="stCheckbox"] p {color: var(--relay-ink)}
[data-testid="stPopoverBody"] [data-baseweb="checkbox"] > div:first-child {
  border-color: #75809b; background-color: white;
}
[data-testid="stPopoverBody"] [data-baseweb="checkbox"]:has(input:checked) > div:first-child {
  background-color: var(--relay-indigo); border-color: var(--relay-indigo);
}
[data-baseweb="menu"], [role="listbox"], [data-baseweb="menu"] [role="option"], [role="listbox"] [role="option"] {
  background: white; color: var(--relay-ink);
}
[role="listbox"] [role="option"]:is(:hover, [aria-selected="true"]), [data-baseweb="menu"] li:hover {
  background: #e8ebfa; color: #222965;
}
:is([data-testid="stPopoverBody"], .st-key-relay_review_panel, [data-testid="stChatMessageContent"])
  button[kind="secondary"] {
  color: var(--relay-ink); background: white; border-color: #c3c9da; border-radius: .59rem;
}
.st-key-relay_review_panel button[kind="primary"] {
  min-height: 2.5rem; background: var(--relay-indigo); color: white; border-color: var(--relay-indigo); border-radius: .59rem;
}
.st-key-relay_review_panel button[kind="primary"]:enabled:hover {background: #3139a9; color: white}
.st-key-relay_review_panel button:disabled {background: #e5e8f1; border-color: #d1d6e4; color: #56627b}
[data-testid="stAlertContainer"] {
  color: #313d78; background: #e9ecff; border-radius: .62rem; overflow-wrap: anywhere;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {color: #705017; background: #faf0d8}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {color: #872c3b; background: #fbe7ea}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {color: #215b49; background: #e3f2e9}
[data-testid="stAlertContainer"] :is([data-testid^="stAlertContent"], p) {color: inherit}
:is(.stApp, [data-testid="stPopoverBody"]) :focus-visible {outline: 2px solid var(--action); outline-offset: 2px}
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {padding-inline: 1.15rem}
  [data-testid="stMainBlockContainer"] {padding-top: 3.25rem}
  .relay-hero {min-height: 13rem; padding: 1.55rem}
  .relay-hero-copy h1 {font-size: clamp(2.55rem, 7.8vw, 3.65rem)}
  .relay-top {gap: .3rem .7rem}
  [data-testid="stChatMessage"]:has(.relay-question) {max-width: 92%}
}
@media (max-width: 490px) {
  [data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {padding-inline: .9rem}
  .st-key-relay_toolbar {gap: .45rem; margin-bottom: .1rem}
  .relay-top {display: grid; gap: .25rem}
  .relay-vendor {font-size: .75rem}
  .relay-hero {padding: 1.35rem; border-radius: 1.05rem; min-height: 13.5rem}
  .relay-hero-copy {max-width: 70%}
  .relay-hero-copy p {max-width: 24ch}
  .relay-hero-art {right: -9%; width: 53%; mask-image: linear-gradient(90deg, transparent 1%, black 33%)}
  .st-key-relay_starters {gap: .4rem}
  .st-key-relay_starters button {padding-inline: .65rem; min-height: 2.45rem}
  .relay-model {max-width: 100%}
}
@media (max-width: 380px) {
  .relay-hero-copy h1 {font-size: 2.12rem}
  .relay-hero-art {right: -14%; width: 51%}
}
@media (forced-colors: active) {
  .relay-question, .relay-hero, .st-key-relay_review_panel {border: 1px solid CanvasText}
  .relay-vendor::before, .relay-mark::before {background: CanvasText}
}
</style>
"""
