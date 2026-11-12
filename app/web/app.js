"use strict";
// Page logic: upload + progress, views (upload / results / bark detail / how it works), model diagram.
// Loaded after charts.js; the chart builders there are only called from functions here.

// Category order of every probability array (woofdecoder.taxonomy.CATEGORIES).
const CATEGORIES = ["alert_aggressive", "playful_excited", "distressed_lonely", "attention_seeking", "other"];
const DISPLAY = {
  alert_aggressive: "Alert / Aggressive",
  playful_excited: "Playful / Excited",
  distressed_lonely: "Distressed / Lonely",
  attention_seeking: "Attention-seeking",
  other: "Other",
};
const SHORT = {
  alert_aggressive: "Alert",
  playful_excited: "Playful",
  distressed_lonely: "Distressed",
  attention_seeking: "Wants attention",
  other: "Other",
};
// One short line per situation; the longer text is shown as a tooltip.
const MEANING = {
  alert_aggressive: "Warning or guarding",
  playful_excited: "Play, greeting or excitement",
  distressed_lonely: "Anxious, alone or scared",
  attention_seeking: "Asking for food, a walk, a toy…",
  other: "Vocal, but none of the above",
};
const MEANING_LONG = {
  alert_aggressive: "The dog is warning or guarding: for example at a stranger, another animal or a noise outside.",
  playful_excited: "The dog is having fun or is excited: during play, greeting someone, or expecting something good.",
  distressed_lonely: "The dog seems upset or anxious: for example when left alone, frightened or uncomfortable.",
  attention_seeking: "The dog is asking for something: food, a walk, a toy, to be let in or out, or simply attention.",
  other: "The dog is vocalizing, but not in one of the four situations above (for example while resting or during a bath). This does not mean the model is unsure.",
};
const BRANCHES = ["audio", "body", "scene"];
const PLAIN_BRANCH = { audio: "Sound", body: "Body", scene: "Scene" };
const CLUE_HINT = {
  audio: "How the bark sounds: pitch, harshness, length.",
  body: "Posture: tail, ears and head position.",
  scene: "What is around the dog in the video.",
};
const POLL_MS = 1000;

// One sentence per step, from the README model-structure section (shown under "Model internals").
const LEADS = {
  preprocessing: "Audio → 16 kHz; a pretrained bark detector (PretrainedSED frame_mn10) finds each dog sound; frames are sampled inside that segment.",
  audio: "Frozen BEATs embedding → linear heads for situation, arousal and valence.",
  body: "Fine-tuned YOLO11m-pose keypoints → pose decoder (MLP) → category probabilities.",
  scene: "Frozen CLIP ViT-B/32, zero-shot against one text prompt per category, combined over frames.",
  fusion: "Logistic-regression head on 15 branch probabilities + 3 present flags, with a fitted temperature.",
};

const ICONS = {
  audio: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M2 6h3l4-3v10L5 10H2z"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5" stroke="currentColor" fill="none" stroke-width="1.5"/></svg>',
  body: '<svg viewBox="0 0 16 16" fill="currentColor"><circle cx="4" cy="5" r="1.6"/><circle cx="7" cy="3" r="1.6"/><circle cx="10" cy="3" r="1.6"/><circle cx="13" cy="5" r="1.6"/><path d="M8.5 7c2.5 0 4.5 3 4 5s-2 1.5-4 1.5S4.5 14 4 12s1.5-5 4.5-5z"/></svg>',
  scene: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1.5" y="2.5" width="13" height="11" rx="1.5"/><path d="M2 12l4-4 3 3 2-2 3 3"/><circle cx="11" cy="6" r="1.2" fill="currentColor"/></svg>',
};

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;
const secs = (s) => `${Number(s).toFixed(2)} s`;
const span = (s) => `${s.start_s.toFixed(2)}–${s.end_s.toFixed(2)} s`;
const signed = (x, digits = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
const argmax = (values) => values.reduce((best, v, i) => (v > values[best] ? i : best), 0);
const info = (text) => `<span class="info" tabindex="0" data-tip="${esc(text)}" aria-label="${esc(text)}">i</span>`;

let job = null;
let clip = null;       // finished job status (filename, results) or null
let segments = [];
let busy = false;

// ---------- upload + progress ----------

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function setBusy(on) {
  busy = on;
  $("drop").hidden = on;
  $("progress").hidden = !on;
  if (on) redrawDiagrams();
}

function stageText(s) {
  if (s.status === "queued") return "Waiting in line…";
  if (s.stage === "extracting audio/frames") return s.total > 1 ? `Frames · bark ${s.done + 1} / ${s.total}` : "Extracting audio";
  if (s.stage === "detecting barks") return "Finding barks";
  if (BRANCHES.includes(s.stage)) return `Bark ${s.done + 1} / ${s.total} · ${PLAIN_BRANCH[s.stage].toLowerCase()}`;
  if (s.stage === "fusion") return `Bark ${s.done + 1} / ${s.total} · combining`;
  return "Working…";
}

function stageFraction(s) {
  if (s.status === "queued") return 0.01;
  if (s.stage === "extracting audio/frames") return s.total > 1 ? 0.1 + 0.08 * (s.done / s.total) : 0.03;
  if (s.stage === "detecting barks") return 0.08;
  const order = ["audio", "body", "scene", "fusion"].indexOf(s.stage);
  if (order >= 0 && s.total > 0) return 0.18 + 0.82 * ((s.done + order / 4) / s.total);
  return 0.02;
}

function setProgress(text, fraction) {
  const value = Math.min(100, Math.round(fraction * 100));
  $("stage").textContent = text;
  $("pct").textContent = `${value}%`;
  $("ring-fill").style.strokeDashoffset = String(100 - value);
}

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (_) { /* not JSON */ }
  return `The server answered ${response.status} ${response.statusText}.`;
}

async function startUpload(file) {
  if (!file || busy) return;
  showError("");
  clip = null;
  segments = [];
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
  route();
  $("progress-file").textContent = file.name;
  setBusy(true);
  liveDiagram.setActive(["input"]);
  setProgress("Uploading…", 0);
  try {
    const form = new FormData();
    form.append("clip", file);
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    job = (await response.json()).job_id;
    poll();
  } catch (error) {
    failed(error instanceof TypeError ? "Could not reach the server. Is uvicorn still running?" : error.message);
  }
}

function failed(message) {
  setBusy(false);
  liveDiagram.setActive(null);
  $("clip").value = "";
  showError(message);
}

$("clip").addEventListener("change", () => startUpload($("clip").files[0]));
["dragenter", "dragover"].forEach((type) => $("drop").addEventListener(type, (e) => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach((type) => $("drop").addEventListener(type, () => $("drop").classList.remove("over")));
$("drop").addEventListener("drop", (e) => { e.preventDefault(); startUpload(e.dataTransfer.files[0]); });

async function poll() {
  try {
    const response = await fetch(`/api/jobs/${job}`);
    if (!response.ok) throw new Error(await readError(response));
    const status = await response.json();
    if (status.status === "error") throw new Error(status.error || "The analysis failed.");
    if (status.status === "done") {
      setProgress("Done", 1);
      setBusy(false);
      liveDiagram.setActive(null);
      $("clip").value = "";
      showResults(status);
      return;
    }
    setProgress(stageText(status), stageFraction(status));
    liveDiagram.setActive(activeIds(status));
    setTimeout(poll, POLL_MS);
  } catch (error) {
    failed(error instanceof TypeError ? "Lost connection to the server." : error.message);
  }
}

// ---------- tabs ----------

const tabMemory = {};  // tabset name -> last chosen tab, so it survives bark-to-bark navigation

// tabs: [{key, label, html}]
function tabset(name, tabs) {
  const active = tabs.some((t) => t.key === tabMemory[name]) ? tabMemory[name] : tabs[0].key;
  return `<div class="tabset" data-tabset="${name}">
    <div class="tabs" role="tablist">${tabs.map((t) => `<button type="button" role="tab" class="tab${t.key === active ? " on" : ""}"
      data-tab="${t.key}" aria-selected="${t.key === active}">${t.label}</button>`).join("")}</div>
    ${tabs.map((t) => `<div class="tab-panel" role="tabpanel" data-panel="${t.key}"${t.key === active ? "" : " hidden"}>${t.html}</div>`).join("")}
  </div>`;
}

document.addEventListener("click", (e) => {
  const button = e.target.closest(".tab[data-tab]");
  if (!button) return;
  const set = button.closest(".tabset");
  set.querySelectorAll(":scope > .tabs > .tab").forEach((b) => {
    b.classList.toggle("on", b === button);
    b.setAttribute("aria-selected", String(b === button));
  });
  set.querySelectorAll(":scope > .tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== button.dataset.tab; });
  tabMemory[set.dataset.tabset] = button.dataset.tab;
  redrawDiagrams();
});

// ---------- model diagram ----------

// Short explanation of every box in the diagram (tech = the model behind it).
const STEPS = {
  input: {
    title: "Your clip", sub: "video or audio",
    plain: "Sound alone is enough. With video, the app also reads the dog and its surroundings.",
    tech: "Any common video or audio format; audio is resampled to 16 kHz.",
  },
  pre: {
    title: "Find the barks", sub: "bark detector",
    plain: "Marks every bark, growl, whine or howl. Each one is judged on its own.",
    tech: LEADS.preprocessing,
  },
  "audio-enc": {
    title: "Listen", sub: "BEATs", branch: "audio",
    plain: "A large pretrained sound model turns the bark into numbers: pitch, harshness, rhythm.",
    tech: "Frozen BEATs audio encoder → one embedding per segment.",
  },
  "audio-head": {
    title: "Score the sound", sub: "linear heads", branch: "audio",
    plain: "Scores the 5 situations, plus how worked up (arousal) and how positive (valence) it sounds.",
    tech: "Linear heads for situation (5 categories), arousal and valence.",
  },
  "body-enc": {
    title: "Spot the pose", sub: "YOLO11-pose", branch: "body",
    plain: "Finds nose, ears, tail and paws. Only frames with exactly one dog are used.",
    tech: "YOLO11m-pose, fine-tuned on dog keypoints.",
  },
  "body-head": {
    title: "Read the body", sub: "pose network", branch: "body",
    plain: "Reads tail, ear and head height and scores the 5 situations.",
    tech: "Pose decoder (MLP) on keypoint features, averaged over the usable frames.",
  },
  "scene-enc": {
    title: "Look around", sub: "CLIP", branch: "scene",
    plain: "A picture-and-text model looks at each whole frame: who and what is around.",
    tech: "Frozen CLIP ViT-B/32 image encoder.",
  },
  "scene-head": {
    title: "Match the scene", sub: "text prompts", branch: "scene",
    plain: "Matches each frame against a description of every situation. Weakest clue on its own.",
    tech: LEADS.scene,
  },
  fusion: {
    title: "Combine", sub: "fusion head",
    plain: "Weighs the 3 clues, trusting some more than others. Still works when one is missing.",
    tech: LEADS.fusion,
  },
  result: {
    title: "Answer", sub: "5 situations",
    plain: "A percentage per situation, adding up to 100%. The highest is the answer.",
    tech: "Softmax of the temperature-scaled fusion logits.",
  },
};

// [id, column (left→right), row (-1 top, 0 middle, 1 bottom)]
const LAYOUT = [
  ["input", 0, 0], ["pre", 1, 0],
  ["audio-enc", 2, -1], ["audio-head", 3, -1],
  ["body-enc", 2, 0], ["body-head", 3, 0],
  ["scene-enc", 2, 1], ["scene-head", 3, 1],
  ["fusion", 4, 0], ["result", 5, 0],
];
const EDGES = [
  ["input", "pre"],
  ["pre", "audio-enc"], ["pre", "body-enc"], ["pre", "scene-enc"],
  ["audio-enc", "audio-head"], ["body-enc", "body-head"], ["scene-enc", "scene-head"],
  ["audio-head", "fusion"], ["body-head", "fusion"], ["scene-head", "fusion"],
  ["fusion", "result"],
];
const NODE_W = 116;
const NODE_H = 48;
const CYCLE_S = 5;  // one pass of the dots from clip to answer
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let diagrams = [];

function geometry(vertical) {
  const pad = 8;
  const gapX = vertical ? 10 : 30;
  const gapY = vertical ? 28 : 16;
  const at = {};
  LAYOUT.forEach(([id, col, row]) => {
    at[id] = vertical
      ? { x: pad + (row + 1) * (NODE_W + gapX), y: pad + col * (NODE_H + gapY) }
      : { x: pad + col * (NODE_W + gapX), y: pad + (row + 1) * (NODE_H + gapY) };
  });
  const width = vertical ? 2 * pad + 3 * NODE_W + 2 * gapX : 2 * pad + 6 * NODE_W + 5 * gapX;
  const height = vertical ? 2 * pad + 6 * NODE_H + 5 * gapY : 2 * pad + 3 * NODE_H + 2 * gapY;
  return { at, width, height };
}

function edgePath(a, b, vertical) {
  if (vertical) {
    const [x1, y1, x2, y2] = [a.x + NODE_W / 2, a.y + NODE_H, b.x + NODE_W / 2, b.y];
    const my = (y1 + y2) / 2;
    return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
  }
  const [x1, y1, x2, y2] = [a.x + NODE_W, a.y + NODE_H / 2, b.x, b.y + NODE_H / 2];
  const mx = (x1 + x2) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

// Fraction of the cycle during which a dot travels out of column `col`.
const travel = (col) => [0.02 + col * 0.16, 0.15 + col * 0.16];
const column = Object.fromEntries(LAYOUT.map(([id, col]) => [id, col]));
const f3 = (x) => x.toFixed(3);

function dot(path, col) {
  const [s, e] = travel(col);
  return `<circle r="4" class="ddot" opacity="0">
    <animateMotion dur="${CYCLE_S}s" repeatCount="indefinite" path="${path}" calcMode="linear"
      keyPoints="0;0;1;1" keyTimes="0;${f3(s)};${f3(e)};1"/>
    <animate attributeName="opacity" dur="${CYCLE_S}s" repeatCount="indefinite"
      values="0;0;1;1;0;0" keyTimes="0;${f3(s)};${f3(s + 0.01)};${f3(e - 0.01)};${f3(e)};1"/></circle>`;
}

function glow(col) {
  const t = col === 0 ? 0.005 : travel(col - 1)[1];
  return `<animate attributeName="opacity" dur="${CYCLE_S}s" repeatCount="indefinite"
    values="0;0;.3;0;0" keyTimes="0;${f3(t)};${f3(t + 0.02)};${f3(t + 0.14)};1"/>`;
}

// opts.subs: per-node subtitle overrides; opts.off: ids of nodes that were not used.
function diagramSVG(vertical, opts) {
  const { at, width, height } = geometry(vertical);
  const off = new Set(opts.off || []);
  const edges = EDGES.map(([from, to]) => {
    const path = edgePath(at[from], at[to], vertical);
    const isOff = off.has(from) || off.has(to);
    return `<g class="dedge-g" data-to="${to}"><path class="dedge${isOff ? " off" : ""}" data-to="${to}" d="${path}"/>
      ${reduceMotion || isOff ? "" : dot(path, column[from])}</g>`;
  }).join("");
  const nodes = LAYOUT.map(([id, col]) => {
    const step = STEPS[id];
    const { x, y } = at[id];
    const sub = (opts.subs && opts.subs[id]) || step.sub;
    return `<g class="dnode${step.branch ? ` b-${step.branch}` : ""}${off.has(id) ? " off" : ""}" data-id="${id}"
        tabindex="0" role="button" aria-label="${esc(step.title)}: ${esc(sub)}">
      <rect class="box" x="${x}" y="${y}" width="${NODE_W}" height="${NODE_H}" rx="8"/>
      <rect class="glow" x="${x}" y="${y}" width="${NODE_W}" height="${NODE_H}" rx="8">${reduceMotion || off.has(id) ? "" : glow(col)}</rect>
      <rect class="stripe" x="${x}" y="${y + 8}" width="4" height="${NODE_H - 16}" rx="2"/>
      <text class="t" x="${x + NODE_W / 2 + 2}" y="${y + 20}" text-anchor="middle">${esc(step.title)}</text>
      <text class="s" x="${x + NODE_W / 2 + 2}" y="${y + 37}" text-anchor="middle">${esc(sub)}</text></g>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Model diagram: clip, find the barks, three clues (sound, body, scene), combine, answer">
    ${edges}${nodes}</svg>`;
}

function explainHTML(id, prefix = "") {
  const step = STEPS[id];
  return `<h5>${prefix}${esc(step.title)}</h5><p>${esc(step.plain)}</p><p class="tech">${esc(step.tech)}</p>`;
}

const IDLE_NOTE = '<p class="note">Tap a box.</p>';

// Draws a diagram into `host` and keeps it in the right orientation when the window resizes.
// opts.explain = false hides the explanation box.
function mountDiagram(host, opts = {}) {
  const d = { host, opts, vertical: null, selected: null, active: null };
  host.innerHTML = `<div class="svg-host"></div>${opts.explain === false ? "" : '<div class="explain" aria-live="polite"></div>'}`;
  d.explain = host.querySelector(".explain");
  if (d.explain) d.explain.innerHTML = IDLE_NOTE;
  d.render = () => {
    if (host.clientWidth === 0) return;  // hidden: draw once it is shown
    const vertical = host.clientWidth < 600;
    if (vertical === d.vertical) return;
    d.vertical = vertical;
    host.querySelector(".svg-host").innerHTML = diagramSVG(vertical, opts);
    host.querySelectorAll(".dnode").forEach((node) => {
      const pick = () => d.select(node.dataset.id);
      node.addEventListener("click", pick);
      node.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    });
    d.paint();
  };
  d.select = (id) => {
    d.selected = id;
    if (d.explain) d.explain.innerHTML = explainHTML(id);
    d.paint();
  };
  // Highlight the boxes that are running now (null = show everything).
  d.setActive = (ids) => {
    d.active = ids;
    if (d.explain) d.explain.innerHTML = ids ? explainHTML(ids[0], "Now: ") : IDLE_NOTE;
    d.paint();
  };
  d.paint = () => {
    host.querySelectorAll(".dnode").forEach((n) => {
      n.classList.toggle("sel", n.dataset.id === d.selected);
      n.classList.toggle("on", !!d.active && d.active.includes(n.dataset.id));
    });
    host.querySelectorAll(".dedge").forEach((e) => e.classList.toggle("on", !!d.active && d.active.includes(e.dataset.to)));
    if (d.active) host.dataset.active = ""; else delete host.dataset.active;
  };
  diagrams.push(d);
  d.render();
  return d;
}

// Draw again every diagram that is attached (needed after it was hidden or the width changed).
function redrawDiagrams() {
  diagrams.forEach((d) => {
    if (!document.body.contains(d.host)) return;
    d.vertical = null;
    d.render();
  });
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => diagrams.forEach((d) => (document.body.contains(d.host) ? d.render() : null)), 150);
});

// Which diagram boxes match a job's current stage.
function activeIds(s) {
  if (!s || s.status === "queued") return ["input"];
  if (s.stage === "extracting audio/frames" || s.stage === "detecting barks") return ["pre"];
  if (BRANCHES.includes(s.stage)) return [`${s.stage}-enc`, `${s.stage}-head`];
  if (s.stage === "fusion") return ["fusion"];
  return ["result"];
}

// ---------- static pages ----------

$("clue-chips").innerHTML = BRANCHES.map((b) => `<span class="clue-chip ${b}">${ICONS[b]}${PLAIN_BRANCH[b]}</span>`).join("");

$("numbers").innerHTML = [
  ["16", "kHz audio"], ["3", "clues"], ["18", "fusion inputs"], ["5", "situations"],
].map(([n, label]) => `<div class="card"><b class="num">${n}</b><span>${label}</span></div>`).join("");

$("situations").innerHTML = CATEGORIES.map((c) => `<div class="sit" style="--c:${catColor(c)}" tabindex="0" data-tip="${esc(MEANING_LONG[c])}">
  <span class="dotcell c-${c}" style="background:${catColor(c)}">${ABBR[c]}</span>
  <div><b>${esc(DISPLAY[c])}</b><span>${esc(MEANING[c])}</span></div></div>`).join("");

const howDiagram = mountDiagram($("how-diagram"));
const liveDiagram = mountDiagram($("live-diagram"));

// ---------- views + routing ----------
//   #/how      how it works
//   #bark-N    one bark (once results exist)
//   anything else: results if there are any, else the upload page

function showView(name) {
  ["upload", "results", "how"].forEach((v) => { $(`view-${v}`).hidden = v !== name; });
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === (name === "how" ? "how" : "home")));
  redrawDiagrams();
}

let openedFromList = false;
let detailIndex = null;

function route() {
  if (location.hash === "#/how") { showView("how"); return; }
  if (!clip) { showView("upload"); return; }
  const match = location.hash.match(/^#bark-(\d+)$/);
  const index = match ? segments.findIndex((s) => s.number === Number(match[1])) : -1;
  const detail = index >= 0;
  showView("results");
  $("list-view").hidden = detail;
  $("list-nav").hidden = detail;
  $("detail-view").hidden = !detail;
  $("detail-nav").hidden = !detail;
  if (detail) {
    renderDetail(index);
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }
  openedFromList = false;
  detailIndex = null;
  renderListSide();
  renderTimeline();
}

window.addEventListener("hashchange", route);

function openSegment(index) {
  openedFromList = true;
  location.hash = `bark-${segments[index].number}`;
}

function goToSegment(index) {
  if (index < 0 || index >= segments.length) return;
  history.replaceState(null, "", `#bark-${segments[index].number}`);
  route();
}

function backToList() {
  if (openedFromList) history.back();
  else { history.replaceState(null, "", location.pathname + location.search); route(); }
}

document.addEventListener("keydown", (e) => {
  if ($("detail-view").hidden || $("view-results").hidden || e.target.closest("input, video, [role=tab]")) return;
  if (e.key === "Escape") backToList();
  if (e.key === "ArrowLeft") goToSegment(detailIndex - 1);
  if (e.key === "ArrowRight") goToSegment(detailIndex + 1);
});
$("back").addEventListener("click", backToList);
$("prev").addEventListener("click", () => goToSegment(detailIndex - 1));
$("next").addEventListener("click", () => goToSegment(detailIndex + 1));
$("new-clip").addEventListener("click", () => {
  clip = null;
  segments = [];
  $("player").removeAttribute("src");
  $("player").load();
  history.replaceState(null, "", location.pathname + location.search);
  route();
});

// ---------- player + timeline ----------

let stopAt = null;

// Clip length in seconds: the player's once it has loaded, else the end of the last bark.
function clipDuration() {
  const d = $("player").duration;
  const lastEnd = Math.max(0, ...segments.map((s) => s.end_s));
  return Number.isFinite(d) && d > 0 ? Math.max(d, lastEnd) : lastEnd * 1.05 || 1;
}

function bindOpeners(root) {
  root.querySelectorAll("[data-index].tl-bark, [data-index].mix-col, [data-index].bark-card").forEach((el) =>
    el.addEventListener("click", (e) => { e.stopPropagation(); openSegment(Number(el.dataset.index)); }));
}

// Seek the video when the empty part of a timeline lane is clicked.
function bindSeek(root) {
  root.querySelectorAll(".timeline").forEach((tl) => tl.querySelector(".tl-lane").addEventListener("click", (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    stopAt = null;
    $("player").currentTime = ((e.clientX - box.left) / box.width) * Number(tl.dataset.duration);
  }));
}

function renderTimeline() {
  const host = $("timeline-host");
  if (!segments.length) { host.innerHTML = ""; return; }
  host.innerHTML = clipTimeline(segments, clipDuration(), { focus: detailIndex });
  bindOpeners(host);
  bindSeek(host);
  movePlayhead();
}

function movePlayhead() {
  const t = $("player").currentTime || 0;
  document.querySelectorAll(".timeline").forEach((tl) => {
    const head = tl.querySelector(".playhead");
    if (head) head.style.left = `${Math.min(100, (t / Number(tl.dataset.duration)) * 100)}%`;
  });
}

$("player").addEventListener("loadedmetadata", () => {
  renderTimeline();
  if (detailIndex == null && clip) renderListSide();
});
$("player").addEventListener("timeupdate", () => {
  movePlayhead();
  if (stopAt != null && $("player").currentTime >= stopAt) { $("player").pause(); stopAt = null; }
});
$("player").addEventListener("seeking", () => { if (!$("player").dataset.ours) stopAt = null; delete $("player").dataset.ours; });

function playSegment(seg) {
  const player = $("player");
  player.dataset.ours = "1";
  player.currentTime = Math.max(0, seg.start_s - 0.25);
  stopAt = seg.end_s + 0.25;
  player.play();
}

// ---------- results: list ----------

function certainty(p) {
  if (p >= 0.7) return "confident";
  if (p >= 0.45) return "fairly sure";
  return "unsure";
}

function ranked(probs) {
  return probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
}

function showResults(status) {
  clip = status;
  segments = status.results || [];
  $("clip-name").textContent = status.filename;
  $("player").src = `/api/jobs/${job}/video`;
  $("player-card").classList.toggle("audio-only", segments.length > 0 && !segments.some((s) => s.has_video));
  renderList();
  if (location.hash !== "#/how") history.replaceState(null, "", location.pathname + location.search);
  route();
}

function renderListSide() {
  $("side").innerHTML = segments.length
    ? clipSummary(segments)
    : `<div class="headline"><span class="kicker">Result</span><b>No barks found</b></div>`;
}

function renderList() {
  if (!segments.length) {
    $("bark-grid").innerHTML = `<div class="card empty">
      <span class="empty-icon">${ICONS.audio}</span><b>No barks found</b><span>No dog sound was detected in this clip.</span></div>`;
    return;
  }
  $("bark-grid").innerHTML = segments.map((seg, i) => barkCard(seg, i)).join("");
  bindOpeners($("bark-grid"));
}

// ---------- results: one bark (#bark-N) ----------

// How much a branch pushed the chosen category above the others in the fusion head.
// Softmax ignores a shift shared by all categories, so compare with the branch's mean push.
function branchEffect(f, branch, best) {
  const c = f.contributions[branch];
  const mean = c.reduce((a, b) => a + b, 0) / c.length;
  return c[best] - mean;
}

function notUsedReason(seg, branch) {
  if (branch === "audio") return "sound could not be used";
  if (branch === "body") return seg.body_reason || "no usable frames";
  return "no video frames";
}

function renderDetail(index) {
  const seg = segments[index];
  const player = $("player");
  player.pause();
  if (player.readyState > 0) player.currentTime = seg.start_s;
  $("detail-pos").textContent = `${index + 1} / ${segments.length}`;
  $("prev").disabled = index === 0;
  $("next").disabled = index === segments.length - 1;
  detailIndex = index;

  $("side").innerHTML = answerSide(seg);
  $("play-seg").addEventListener("click", () => playSegment(seg));
  renderTimeline();

  $("detail").innerHTML = `<h3 class="section-title"><span>What each clue said</span></h3>${cluesPanel(seg)}
    <details class="card tech" id="tech"${techOpen ? " open" : ""}>
      <summary>Technical details</summary>${internals(seg)}</details>`;
  $("tech").addEventListener("toggle", () => { techOpen = $("tech").open; });
}

let techOpen = false;

function answerSide(seg) {
  const play = `<button type="button" id="play-seg" class="play-btn">▶ Play bark</button>`;
  const when = `<span class="time">Bark #${seg.number} · ${span(seg)}</span>`;
  if (!seg.fused.present) {
    return `<div class="headline"><span class="kicker">Result</span><b>No answer</b></div>
      <p class="meaning">None of the 3 clues could be used.</p><div class="side-foot">${play}${when}</div>`;
  }
  const order = ranked(seg.fused.probs);
  const [p1] = order[0];
  const [p2, i2] = order[1];
  const close = p1 - p2 < 0.15
    ? `<div class="close-call">Close call with <b>${esc(SHORT[CATEGORIES[i2]])}</b> (${pct(p2, 0)})</div>` : "";
  return `<div class="headline" style="--c:${catColor(seg.category)}">
      <span class="kicker">Most likely</span><b><span class="dot"></span>${esc(DISPLAY[seg.category])}</b>
      <span class="meaning" data-tip="${esc(MEANING_LONG[seg.category])}">${esc(MEANING[seg.category])}</span></div>
    <div class="answer-viz">${donut(seg.fused.probs, pct(p1, 0), certainty(p1), (i) => `${DISPLAY[CATEGORIES[i]]}: ${pct(seg.fused.probs[i])}`)}
      ${probLegend(seg.fused.probs)}</div>
    ${close}
    <div class="side-foot">${play}${when}</div>`;
}

function cluesPanel(seg) {
  const f = seg.fusion;
  const best = seg.fused.present ? argmax(seg.fused.probs) : null;
  const cards = BRANCHES.map((b) => {
    const out = seg.branches[b];
    const head = `<div class="clue-head">${ICONS[b]}<span>${PLAIN_BRANCH[b]}</span>${info(CLUE_HINT[b])}</div>`;
    if (!out.present) {
      return `<div class="clue ${b} off">${head}<div class="clue-off">Not used</div><p class="note">${esc(notUsedReason(seg, b))}</p></div>`;
    }
    const top = argmax(out.probs);
    const category = CATEGORIES[top];
    const agrees = top === best;
    return `<div class="clue ${b}">${head}
      <div class="clue-pick"><span class="clue-pct num">${pct(out.probs[top], 0)}</span>
        <span class="clue-cat"><span class="dot" style="--c:${catColor(category)}"></span>${esc(SHORT[category])}</span>
        ${best == null ? "" : agrees ? '<span class="agree-tag yes">✓ agrees</span>' : '<span class="agree-tag">differs</span>'}</div>
      ${miniColumns(out.probs, PLAIN_BRANCH[b])}
      ${f && best != null ? effectBadge(branchEffect(f, b, best)) : ""}</div>`;
  }).join("");
  return `<div class="clues">${cards}</div>`;
}

// ---------- model internals (one sub-tab per step) ----------

function internals(seg) {
  return tabset("internals", [
    { key: "detect", label: "Detect", html: preprocessingSection(seg) },
    { key: "audio", label: "Sound", html: audioSection(seg) },
    { key: "body", label: "Body", html: bodySection(seg) },
    { key: "scene", label: "Scene", html: sceneSection(seg) },
    { key: "fusion", label: "Fusion", html: fusionSection(seg) },
  ]);
}

function section(key, body) {
  return `<p class="lead">${esc(LEADS[key])}</p>${body}`;
}

function bar(label, value, top = false) {
  return `<div class="row${top ? " top" : ""}"><span>${esc(label)}</span>
    <div class="track"><div class="fill" style="width:${(value * 100).toFixed(1)}%"></div></div>
    <span class="num">${pct(value)}</span></div>`;
}

function namedBars(values) {
  const names = Object.keys(values);
  const best = names.reduce((a, b) => (values[b] > values[a] ? b : a), names[0]);
  return names.map((n) => bar(n, values[n], n === best)).join("");
}

function frameStrip(seg, caption) {
  if (!seg.frames.length) return '<p class="note">No video frames (audio-only clip).</p>';
  return `<div class="frames">${seg.frames.map((f) => `
    <div class="frame ${caption ? (f.body_used ? "used" : "unused") : ""}">
      <img alt="frame at ${secs(f.time_s)}" src="data:image/jpeg;base64,${f.jpeg}">
      ${secs(f.time_s)}${caption ? `<br>${f.body_used ? "✓ used" : `✗ ${f.dogs ?? "?"} dogs`}` : ""}
    </div>`).join("")}</div>`;
}

function preprocessingSection(seg) {
  return section("preprocessing", `
    ${statTiles([
      { label: "Length", value: `${(seg.end_s - seg.start_s).toFixed(2)}<small> s</small>`, note: span(seg) },
      { label: "Detector score", value: seg.confidence == null ? "–" : seg.confidence.toFixed(2), note: "mean, not calibrated" },
      { label: "Frames", value: String(seg.num_frames), note: seg.has_video ? "sampled" : "no video" },
    ])}
    <h4>Sound sections</h4>${soundLanes(seg)}
    <h4>Sampled frames</h4>${frameStrip(seg, false)}`);
}

function audioSection(seg) {
  const out = seg.branches.audio;
  if (!out.present) return section("audio", '<p class="note">Not used: the segment has no usable audio.</p>');
  const x = out.extras;
  const nudge = x.nudge ? CATEGORIES.map((c) => {
    const v = x.nudge[c];
    return `<div class="row"><span>${esc(DISPLAY[c])}</span><span></span>
      <span class="num ${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${signed(v)}</span></div>`;
  }).join("") : "";
  const zeroNudge = x.nudge && Object.values(x.nudge).every((v) => v === 0);
  return section("audio", `
    <div class="split">
      ${x.arousal ? `<div><h4>Arousal</h4>${namedBars(x.arousal)}</div>` : ""}
      ${x.valence ? `<div><h4>Valence ${info("Near chance on unseen dogs (notebook 24): a weak hint.")}</h4>${namedBars(x.valence)}</div>` : ""}
    </div>
    ${nudge ? `<h4>Emotion nudge (logit shift) ${info(zeroNudge
      ? "All 0: the saved emotion-nudge weights are zero, so arousal and valence do not change the situation scores."
      : "How much arousal and valence moved each category's logit before the softmax.")}</h4>${nudge}` : ""}`);
}

function bodySection(seg) {
  const out = seg.branches.body;
  const strip = `<h4>Frames <span class="note">green = used (exactly one dog)</span></h4>${frameStrip(seg, true)}`;
  if (!out.present) {
    return section("body", `<p><strong>Not used</strong>: ${esc(seg.body_reason || "unavailable")}.</p>${seg.has_video ? strip : ""}`);
  }
  const x = out.extras;
  const cues = [["tail_height", "Tail height"], ["ear_height", "Ear height"], ["head_height", "Head height"]]
    .filter(([name]) => typeof x[name] === "number")
    .map(([name, label]) => ({ label, value: x[name], tip: `${label}: ${signed(x[name], 3)}\n${x[name] >= 0 ? "above" : "below"} its reference point` }));
  const used = seg.frames.filter((f) => f.body_used).length;
  return section("body", `
    ${statTiles([{ label: "Frames used", value: `${used}<small> / ${seg.frames.length}</small>`, note: "exactly one dog" }])}
    <h4>Pose cues ${info("Mean over used frames, box-normalised. Positive tail height = tail tip above its base. Still poses, no wagging.")}</h4>
    ${cues.length ? divergingBars(cues, "var(--body)") : '<p class="note">No pose cues.</p>'}
    ${strip}`);
}

function sceneSection(seg) {
  const out = seg.branches.scene;
  const s = seg.scene || {};
  if (!out.present) return section("scene", "<p><strong>Not used</strong>: no video frames.</p>");
  const thumbs = Object.fromEntries(seg.frames.map((f) => [f.time_s.toFixed(3), f.jpeg]));
  const rows = (s.frames || []).map((f) => {
    const jpeg = thumbs[f.time_s.toFixed(3)];
    return `<tr><td>${secs(f.time_s)}</td>
      <td>${jpeg ? `<img alt="" class="thumb" src="data:image/jpeg;base64,${jpeg}">` : ""}</td>
      <td class="wrap">${esc(f.top_prompt)}</td><td>${pct(f.score)}</td></tr>`;
  }).join("");
  return section("scene", `
    ${statTiles([
      { label: "Aggregation", value: `<span class="tile-text">${esc(s.aggregation || out.extras.aggregation)}</span>` },
      ...(s.confidence_weight != null ? [{ label: "Confidence weight", value: String(s.confidence_weight) }] : []),
      { label: "Temperature", value: String(s.temperature) },
    ])}
    <p class="note">Raw zero-shot scene scores are below chance on our data; the fusion head learned how much to trust them.</p>
    ${sceneFrameHeatmap(seg)}
    ${rows ? `<h4>Best prompt per frame</h4><div class="scroll"><table>
      <tr><th>Time</th><th></th><th class="wrap">Top prompt</th><th>Score</th></tr>${rows}</table></div>` : ""}`);
}

function contributionChart(f) {
  const parts = ["bias", ...BRANCHES];
  const values = (i) => ({ bias: f.bias[i], audio: f.contributions.audio[i], body: f.contributions.body[i], scene: f.contributions.scene[i] });
  let extent = 0;
  CATEGORIES.forEach((_, i) => {
    const v = Object.values(values(i));
    extent = Math.max(extent, v.filter((x) => x > 0).reduce((a, b) => a + b, 0), -v.filter((x) => x < 0).reduce((a, b) => a + b, 0));
  });
  extent = extent || 1;
  const rows = CATEGORIES.map((c, i) => {
    let right = 0;
    let left = 0;
    const segs = parts.map((p) => {
      const v = values(i)[p];
      const width = (Math.abs(v) / extent) * 50;
      const start = v >= 0 ? 50 + right : 50 - left - width;
      if (v >= 0) right += width; else left += width;
      return `<div class="seg" data-tip="${p}: ${signed(v, 3)}" style="left:${start}%;width:${width}%;background:var(--${p})"></div>`;
    }).join("");
    return `<div class="contrib"><span>${esc(DISPLAY[c])}</span><div class="lane"><div class="zero"></div>${segs}</div>
      <span class="num">${signed(f.logits[i])}</span></div>`;
  }).join("");
  const legend = parts.map((p) => `<span><span class="swatch" style="background:var(--${p})"></span>${p}</span>`).join("");
  return `<div class="legend">${legend}</div>${rows}
    <div class="contrib-axis"><span></span><div><span>−${extent.toFixed(1)}</span><span>0</span><span>+${extent.toFixed(1)}</span></div><span class="num">logit</span></div>`;
}

function fusionSection(seg) {
  const f = seg.fusion;
  if (!f) return section("fusion", '<p class="note">No branch could run, so there is nothing to fuse.</p>');
  const present = `<div class="chips">${BRANCHES.map((b) => `<span class="chip${f.present[b] ? "" : " off"}">${b}${f.present[b] ? " present" : " missing"}</span>`).join("")}</div>`;
  const zExtent = Math.max(1, ...f.features.map((x) => Math.abs(x.standardized)));
  const featureRows = f.features.map((x) => {
    const [branch, name] = x.name.split(":");
    const absent = !f.present[branch];
    const label = name === "present" ? "present flag" : DISPLAY[name] || name;
    return `<tr class="${absent ? "absent" : ""}"><td>${esc(branch)} · ${esc(label)}</td>
      <td>${x.raw.toFixed(3)}</td><td>${x.standardized.toFixed(3)}</td><td class="mini">${miniDiverging(x.standardized, zExtent)}</td></tr>`;
  }).join("");
  const best = argmax(f.probs);
  const logitRows = CATEGORIES.map((c, i) => `<tr class="${i === best ? "top" : ""}"><td>${esc(DISPLAY[c])}</td>
    <td>${signed(f.bias[i])}</td>${BRANCHES.map((b) => `<td>${signed(f.contributions[b][i])}</td>`).join("")}
    <td>${signed(f.logits[i])}</td><td>${signed(f.scaled_logits[i])}</td><td>${pct(f.probs[i])}</td></tr>`).join("");
  return section("fusion", `
    ${present}
    <h4>What pushed each logit ${info("Each branch's share = weights × standardised inputs (its 5 inputs + present flag); bias is the head's constant. Right raises, left lowers.")}</h4>
    ${contributionChart(f)}
    <h4>Logits ÷ T (${f.temperature.toFixed(3)}) → softmax</h4>
    <div class="scroll"><table><tr><th>Category</th><th>Bias</th><th>Audio</th><th>Body</th><th>Scene</th>
      <th>Logit</th><th>÷ T</th><th>Probability</th></tr>${logitRows}</table></div>
    <h4>The 18 inputs ${info("Raw: centred log-probabilities per branch (zeros when missing), then the present flags. Standardised: (raw − training mean) ÷ training std.")}</h4>
    <div class="scroll"><table><tr><th>Input</th><th>Raw</th><th>Std.</th><th>−${zExtent.toFixed(1)} · 0 · +${zExtent.toFixed(1)}</th></tr>${featureRows}</table></div>`);
}

route();
