// Audio8 tag highlighting for the Audio8 Script node's text widget.
//
// The multiline widget is a DOMWidget: its container is already positioned
// and scaled with the canvas, so we insert a highlight mirror INSIDE that
// container behind the textarea. No canvas overlay, no per-frame rect sync.
//
// Rules mirror audio8_script.py exactly (keep in sync + shared fixtures):
//   [tag]      green when the tag is in the wired voice set
//   [unknown]  red when a voice set is wired, plain when the set is empty
//   [[  ]]     dim literal bracket escapes
// Duplicate voice-set tags resolve to _1/_2 suffixes in chain order, the
// same rule the Python side applies.
import { app } from "../../../scripts/app.js";

const CLASS = "Audio8Script";
const TOKEN = /\[\[|\]\]|\[([^\[\]\n]{1,64})\]/g;
const STYLE = `
.audio8-tag-mirror { position:absolute; inset:0; margin:0; border:0;
  background:transparent; pointer-events:none; white-space:pre-wrap;
  overflow:hidden; color:inherit; z-index:0; }
.audio8-tag-mirror > .audio8-tag-inner { display:block; }
.audio8-tag-known { color:#15803d; font-weight:600; }
.audio8-tag-unknown { color:#b91c1c; font-weight:600; }
.audio8-tag-escape { color:#9ca3af; }
.audio8-tag-src-hidden textarea.comfy-multiline-input { color:transparent; }
`;

function styleOnce() {
  if (document.getElementById("audio8-tag-style")) return;
  const style = document.createElement("style");
  style.id = "audio8-tag-style";
  style.textContent = STYLE;
  document.head.appendChild(style);
}

function graph() {
  return app.graph || app.canvas?.graph;
}

function tagWidgetValue(node) {
  const widget = (node.widgets || []).find(w => w.name === "tag");
  return widget ? String(widget.value ?? "") : "";
}

function upstreamNode(node, inputName) {
  const input = (node.inputs || []).find(i => i.name === inputName);
  if (!input || input.link == null) return null;
  const link = graph()?.links?.[input.link];
  if (!link) return null;
  return graph().getNodeById(link.origin_id) || null;
}

// Chain order: root set first; duplicates gain _1/_2 exactly like Python.
function collectTags(scriptNode) {
  let add = upstreamNode(scriptNode, "voice_set");
  const chain = [];
  let guard = 0;
  while (add && add.type === "Audio8VoiceSetAdd" && guard++ < 200) {
    chain.unshift(tagWidgetValue(add).trim());
    add = upstreamNode(add, "voice_set");
  }
  const used = new Set();
  for (const tag of chain) {
    let final = tag;
    let index = 1;
    while (used.has(final)) final = `${tag}_${index++}`;
    used.add(final);
  }
  return { tags: used, wired: chain.length > 0 };
}

function paint(state) {
  const { ta, inner } = state;
  const value = ta.value;
  const { tags, wired } = collectTags(state.node);
  const fragment = document.createDocumentFragment();
  let position = 0;
  let match;
  TOKEN.lastIndex = 0;
  while ((match = TOKEN.exec(value)) !== null) {
    if (match.index > position) {
      fragment.append(document.createTextNode(value.slice(position, match.index)));
    }
    const token = match[0];
    const span = document.createElement("span");
    if (token === "[[" || token === "]]") {
      span.className = "audio8-tag-escape";
      span.textContent = token;
    } else if (tags.has(match[1])) {
      span.className = "audio8-tag-known";
      span.textContent = token;
    } else {
      span.className = wired ? "audio8-tag-unknown" : "";
      span.textContent = token;
    }
    fragment.append(span);
    position = match.end ?? (match.index + token.length);
  }
  if (position < value.length) {
    fragment.append(document.createTextNode(value.slice(position)));
  }
  inner.replaceChildren(fragment);
}

function copyFont(ta, mirror) {
  const cs = getComputedStyle(ta);
  for (const prop of ["fontFamily", "fontSize", "fontWeight", "fontStyle",
                      "lineHeight", "letterSpacing", "wordSpacing",
                      "whiteSpace", "wordBreak", "tabSize"]) {
    mirror.style[prop] = cs[prop];
  }
  // color:transparent on the textarea also hides an auto caret; pin the
  // caret colour and the mirror's plain-text colour to the real text
  // colour captured before the transparent class applies.
  ta.style.caretColor = cs.color;
  mirror.style.color = cs.color;
  mirror.style.boxSizing = "border-box";
}

function syncBox(ta, mirror) {
  // The mirror sits on the textarea's border box but has no border and no
  // scrollbar. Fold border and (vertical) scrollbar width into the mirror
  // padding so text starts at the exact same pixel. Recomputed per paint:
  // scrollbar presence changes with content height.
  const cs = getComputedStyle(ta);
  const borderLeft = parseFloat(cs.borderLeftWidth) || 0;
  const borderRight = parseFloat(cs.borderRightWidth) || 0;
  const borderTop = parseFloat(cs.borderTopWidth) || 0;
  const borderBottom = parseFloat(cs.borderBottomWidth) || 0;
  const scrollbar = Math.max(0,
    ta.offsetWidth - ta.clientWidth - borderLeft - borderRight);
  mirror.style.paddingTop = `${(parseFloat(cs.paddingTop) || 0) + borderTop}px`;
  mirror.style.paddingRight = `${(parseFloat(cs.paddingRight) || 0) + borderRight + scrollbar}px`;
  mirror.style.paddingBottom = `${(parseFloat(cs.paddingBottom) || 0) + borderBottom}px`;
  mirror.style.paddingLeft = `${(parseFloat(cs.paddingLeft) || 0) + borderLeft}px`;
}

const hookedElements = new WeakSet();

function scriptNodes() {
  const g = graph();
  if (!g) return [];
  return (g._nodes || []).filter(n => n.type === CLASS || n.comfyClass === CLASS);
}

function hookWidget(node, widget) {
  const element = widget.element;
  if (!element || hookedElements.has(element)) return;
  // Vue may rebuild children; cap mirrors globally at one per script node
  // so a rebuild storm cannot accumulate layers. If a mirror truly vanishes,
  // the next scan re-adds one (self-healing).
  if (document.querySelectorAll(".audio8-tag-mirror").length >= scriptNodes().length) return;
  const ta = element.querySelector("textarea") ||
             (element.tagName === "TEXTAREA" ? element : null);
  if (!ta) return;
  hookedElements.add(element);

  const mirror = document.createElement("div");
  mirror.className = "audio8-tag-mirror";
  const inner = document.createElement("div");
  inner.className = "audio8-tag-inner";
  mirror.append(inner);

  const host = ta.parentElement || element;
  if (getComputedStyle(host).position === "static") host.style.position = "relative";
  host.prepend(mirror);
  copyFont(ta, mirror);
  host.classList.add("audio8-tag-src-hidden");

  const state = { node, ta, inner, mirror };
  const repaint = () => {
    try { syncBox(state.ta, state.mirror); paint(state); }
    catch (err) { console.warn("[audio8-tag]", err); }
  };
  ta.addEventListener("input", repaint);
  ta.addEventListener("scroll", () => {
    inner.style.transform = `translate(${-ta.scrollLeft}px, ${-ta.scrollTop}px)`;
  });
  repaint();
}

function scan() {
  for (const node of scriptNodes()) {
    for (const widget of node.widgets || []) {
      if (widget.name === "text" && widget.element) hookWidget(node, widget);
    }
  }
}

app.registerExtension({
  name: "audio8.tagHighlight",
  setup() {
    styleOnce();
    window.__audio8TagHighlight = "loaded";
    // Widget DOM mounts late and nodes rebuild often: cheap periodic scan.
    setInterval(() => { try { scan(); } catch (err) { console.warn("[audio8-tag]", err); } }, 1000);
  },
});
