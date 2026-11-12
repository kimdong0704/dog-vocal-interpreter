"use strict";

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
// What each situation means, in everyday words (see woofdecoder.taxonomy).
const MEANING = {
  alert_aggressive: "The dog is warning or guarding: for example at a stranger, another animal or a noise outside.",
  playful_excited: "The dog is having fun or is excited: during play, greeting someone, or expecting something good.",
  distressed_lonely: "The dog seems upset or anxious: for example when left alone, frightened or uncomfortable.",
  attention_seeking: "The dog is asking for something: food, a walk, a toy, to be let in or out, or simply attention.",
  other: "The dog is vocalizing, but not in one of the four situations above (for example while resting or during a bath). This does not mean the model is unsure.",
};
const BRANCHES = ["audio", "body", "scene"];
const PLAIN_BRANCH = { audio: "Sound", body: "Body language", scene: "Surroundings" };
const CLUE_HINT = {
  audio: "How the bark sounds: pitch, harshness, length.",
  body: "Posture: tail, ears and head position.",
  scene: "What is around the dog in the video.",
};
const POLL_MS = 1000;

// One plain sentence per step, from the README model-structure section.
const LEADS = {
  preprocessing: "The audio is converted to 16 kHz, a pretrained bark detector (PretrainedSED frame_mn10) finds each dog sound, and frames are sampled from the video inside that segment.",
  audio: "Frozen BEATs turns the sound into an embedding, and linear heads score the situation, arousal and valence.",
  body: "A fine-tuned YOLO11m-pose finds the dog's keypoints in each frame, and a small pose decoder (MLP) turns them into category probabilities.",
  scene: "Frozen CLIP ViT-B/32 compares each frame with one text prompt per category (zero-shot), and the scores are combined over the frames.",
  fusion: "A learned logistic-regression head combines the 15 branch probabilities and 3 branch-present flags, with a fitted temperature.",
  result: "The fused probabilities over the 5 categories for this vocalization.",
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
const signed = (x, digits = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
const argmax = (values) => values.reduce((best, v, i) => (v > values[best] ? i : best), 0);

let job = null;
let segments = [];

// ---------- upload + progress ----------

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function setBusy(busy) {
  $("analyze").disabled = busy;
  $("analyze").textContent = busy ? "Analyzing…" : "Analyze";
  $("progress").hidden = !busy;
}

function stageText(s) {
  if (s.status === "queued") return "Waiting for the previous clip to finish…";
  if (s.stage === "extracting audio/frames") {
    return s.total > 1 ? `Extracting frames for segment ${s.done + 1}/${s.total}…` : "Extracting audio…";
  }
  if (s.stage === "detecting barks") return "Detecting barks…";
  if (BRANCHES.includes(s.stage) || s.stage === "fusion") return `Segment ${s.done + 1}/${s.total}: ${s.stage}…`;
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
  $("stage").textContent = text;
  $("pct").textContent = `${Math.round(fraction * 100)}%`;
  $("bar").style.width = `${Math.min(100, fraction * 100)}%`;
}

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (_) { /* not JSON */ }
  return `The server answered ${response.status} ${response.statusText}.`;
}

$("upload").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("clip").files[0];
  if (!file) return;
  showError("");
  $("results").hidden = true;
  segments = [];
  history.replaceState(null, "", location.pathname + location.search);
  $("how").open = true;
  howDiagram.setActive(["input"]);
  setBusy(true);
  setProgress("Uploading…", 0);
  try {
    const form = new FormData();
    form.append("clip", file);
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    job = (await response.json()).job_id;
    poll();
  } catch (error) {
    setBusy(false);
    howDiagram.setActive(null);
    showError(error instanceof TypeError ? "Could not reach the server. Is uvicorn still running?" : error.message);
  }
});

async function poll() {
  try {
    const response = await fetch(`/api/jobs/${job}`);
    if (!response.ok) throw new Error(await readError(response));
    const status = await response.json();
    if (status.status === "error") throw new Error(status.error || "The analysis failed.");
    if (status.status === "done") {
      setProgress("Done", 1);
      setBusy(false);
      showResults(status);
      return;
    }
    setProgress(stageText(status), stageFraction(status));
    howDiagram.setActive(activeIds(status));
    setTimeout(poll, POLL_MS);
  } catch (error) {
    setBusy(false);
    howDiagram.setActive(null);
    showError(error instanceof TypeError ? "Lost connection to the server." : error.message);
  }
}

// ---------- shared pieces ----------

function bar(label, value, top = false, color = null) {
  const style = color ? `background:${color}` : "";
  return `<div class="row${top ? " top" : ""}"><span>${esc(label)}</span>
    <div class="track"><div class="fill" style="width:${(value * 100).toFixed(1)}%;${style}"></div></div>
    <span class="num">${pct(value)}</span></div>`;
}

function categoryBars(probs) {
  const best = argmax(probs);
  return CATEGORIES.map((c, i) => bar(DISPLAY[c], probs[i], i === best)).join("");
}

function namedBars(values) {
  const names = Object.keys(values);
  const best = names.reduce((a, b) => (values[b] > values[a] ? b : a), names[0]);
  return names.map((n) => bar(n, values[n], n === best)).join("");
}

function icons(seg) {
  return BRANCHES.map((b) => {
    const on = seg.branches[b].present;
    return `<span class="icon${on ? "" : " off"}" title="${PLAIN_BRANCH[b]}: ${on ? "used" : "not used"}">${ICONS[b]}${b === "audio" ? "sound" : b}</span>`;
  }).join("");
}

function frameStrip(seg, caption) {
  if (!seg.frames.length) return '<p class="note">No video frames (audio-only clip).</p>';
  return `<div class="frames">${seg.frames.map((f) => `
    <div class="frame ${caption ? (f.body_used ? "used" : "unused") : ""}">
      <img alt="frame at ${secs(f.time_s)}" src="data:image/jpeg;base64,${f.jpeg}">
      ${secs(f.time_s)}${caption ? `<br>${f.body_used ? "✓ used" : `✗ ${f.dogs ?? "?"} dogs`}` : ""}
    </div>`).join("")}</div>`;
}

function section(step, title, key, body) {
  return `<div class="card"><h3><span class="step">${step}</span>${esc(title)}</h3>
    <p class="lead">${esc(LEADS[key])}</p>${body}</div>`;
}

// ---------- model diagram ----------

// Plain-language explanation of every box in the diagram (tech = the model behind it).
const STEPS = {
  input: {
    title: "Your clip", sub: "video or audio",
    plain: "The video (or audio file) you upload. Sound alone is enough, but with video the app can also look at the dog and its surroundings.",
    tech: "Any common video or audio format; the audio is resampled to 16 kHz.",
  },
  pre: {
    title: "Find the barks", sub: "bark detector",
    plain: "The app listens through the whole clip and marks every moment the dog barks, growls, whines or howls. Each moment is judged on its own, and a few still frames are taken from the video for it.",
    tech: LEADS.preprocessing,
  },
  "audio-enc": {
    title: "Listen", sub: "BEATs", branch: "audio",
    plain: "A large sound model, already trained on a huge library of everyday sounds, turns the bark into a list of numbers that describe how it sounds: pitch, harshness, rhythm and so on.",
    tech: "Frozen BEATs audio encoder → one embedding per segment.",
  },
  "audio-head": {
    title: "Score the sound", sub: "linear heads", branch: "audio",
    plain: "Small layers trained on our labelled barks turn those numbers into a score for each of the five situations. They also estimate how worked up the dog sounds (arousal) and whether the mood seems positive or negative (valence).",
    tech: "Linear heads for situation (5 categories), arousal and valence.",
  },
  "body-enc": {
    title: "Spot the pose", sub: "YOLO11-pose", branch: "body",
    plain: "A pose model finds the dog in each frame and marks points such as the nose, ears, tail and paws. Only frames with exactly one dog are used, so the app knows whose body it is reading.",
    tech: "YOLO11m-pose, fine-tuned on dog keypoints.",
  },
  "body-head": {
    title: "Read the body", sub: "pose network", branch: "body",
    plain: "A small network reads the posture (how high the tail, ears and head are) and scores the five situations.",
    tech: "Pose decoder (MLP) on keypoint features, averaged over the usable frames.",
  },
  "scene-enc": {
    title: "Look around", sub: "CLIP", branch: "scene",
    plain: "A model that understands both pictures and words looks at each frame as a whole: who and what is around the dog.",
    tech: "Frozen CLIP ViT-B/32 image encoder.",
  },
  "scene-head": {
    title: "Match the scene", sub: "text prompts", branch: "scene",
    plain: "Each frame is compared with a short written description of every situation, and the best matches are combined over the frames. This is the weakest clue on its own.",
    tech: LEADS.scene,
  },
  fusion: {
    title: "Combine", sub: "fusion head",
    plain: "The three opinions are weighed together. The combiner was trained on our labelled videos, so it has learned how much to trust each clue, and it still works when a clue is missing (for example, no dog in view).",
    tech: LEADS.fusion,
  },
  result: {
    title: "Answer", sub: "5 situations",
    plain: "The final estimate: a percentage for each of the five situations, adding up to 100%. The highest one is shown as the answer.",
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
const diagrams = [];

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
  return `<h5>${prefix}${esc(step.title)}</h5><p>${esc(step.plain)}</p><p class="tech">Under the hood: ${esc(step.tech)}</p>`;
}

// Draws a diagram into `host` and keeps it in the right orientation when the window resizes.
function mountDiagram(host, opts = {}) {
  const d = { host, opts, vertical: null, selected: null, active: null };
  host.innerHTML = '<div class="svg-host"></div><div class="explain" aria-live="polite"></div>';
  d.explain = host.querySelector(".explain");
  d.explain.innerHTML = '<p class="note">Tap any box to see what it does.</p>';
  d.render = () => {
    const vertical = host.clientWidth > 0 && host.clientWidth < 600;
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
    d.explain.innerHTML = explainHTML(id);
    d.paint();
  };
  // Highlight the boxes that are running now (null = show everything).
  d.setActive = (ids) => {
    d.active = ids;
    if (ids) d.explain.innerHTML = explainHTML(ids[0], "Now: ");
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

const howDiagram = mountDiagram($("how-diagram"));
$("glossary").innerHTML = CATEGORIES.map((c) => `<dt>${esc(DISPLAY[c])}</dt><dd>${esc(MEANING[c])}</dd>`).join("");
// The panel may start hidden (closed); draw again once it has a width.
$("how").addEventListener("toggle", () => { howDiagram.vertical = null; howDiagram.render(); });

// ---------- plain-language helpers ----------

function certainty(p) {
  if (p >= 0.7) return "confident";
  if (p >= 0.45) return "fairly sure";
  return "unsure";
}

function ranked(probs) {
  return probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
}

// How much a branch pushed the chosen category above the others in the fusion head.
// Softmax ignores a shift shared by all categories, so compare with the branch's mean push.
function branchEffect(f, branch, best) {
  const c = f.contributions[branch];
  const mean = c.reduce((a, b) => a + b, 0) / c.length;
  return c[best] - mean;
}

function effectChip(value) {
  if (value > 1) return '<span class="effect pos">Strongly supported the answer</span>';
  if (value > 0.25) return '<span class="effect pos">Supported the answer</span>';
  if (value >= -0.25) return '<span class="effect">Little effect on the answer</span>';
  return '<span class="effect neg">Weighed against the answer</span>';
}

function notUsedReason(seg, branch) {
  if (branch === "audio") return "the sound could not be used";
  if (branch === "body") return seg.body_reason || "no usable frames";
  return "there are no video frames";
}

// ---------- results: list page ----------

function showResults(status) {
  segments = status.results || [];
  $("results").hidden = false;
  $("how").open = false;
  howDiagram.setActive(null);
  howDiagram.explain.innerHTML = '<p class="note">Tap any box to see what it does.</p>';
  $("player").src = `/api/jobs/${job}/video`;
  history.replaceState(null, "", location.pathname + location.search);
  route();
  const found = segments.filter((s) => s.fused.present);
  $("overview").innerHTML = "";
  $("clip-charts").innerHTML = "";
  if (!segments.length) {
    $("summary").textContent = `${status.filename}: no barks found.`;
    $("segments").innerHTML = '<div class="card empty" style="margin:0;border:0">No dog vocalization was detected in this clip.</div>';
    return;
  }
  const counts = {};
  found.forEach((s) => { counts[s.category] = (counts[s.category] || 0) + 1; });
  const byCount = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  // Only name a "most often" category when it clearly leads (no tie, more than one bark).
  const common = byCount.length && counts[byCount[0]] > 1 && counts[byCount[0]] > (counts[byCount[1]] || 0) ? byCount[0] : null;
  $("summary").innerHTML = `<strong>${esc(status.filename)}</strong>: found ${segments.length} bark${segments.length === 1 ? "" : "s"}.`
    + (common ? ` Most often: <strong>${esc(DISPLAY[common])}</strong> (${counts[common]} of ${segments.length}).` : "")
    + " Click a bark to see why.";
  $("segments").innerHTML = segments.map((seg, i) => {
    const verdict = seg.fused.present
      ? `${esc(DISPLAY[seg.category])}<span class="pct">${pct(seg.probability, 0)}</span><span class="sure">· ${certainty(seg.probability)}</span>`
      : '<span class="time">no clue could be used</span>';
    return `<button type="button" class="seg-row" data-index="${i}" aria-label="Open bark ${seg.number}">
      <span class="n">#${seg.number}</span>
      <span class="time">${seg.start_s.toFixed(2)}–${seg.end_s.toFixed(2)} s</span>
      <span class="verdict-cell">${verdict}</span>
      <span class="icons">${icons(seg)}</span>
      <span class="chev" aria-hidden="true">›</span></button>`;
  }).join("");
  document.querySelectorAll(".seg-row").forEach((row) => row.addEventListener("click", () => openSegment(Number(row.dataset.index))));
  renderOverview();
}

// Clip length in seconds: the player's once it has loaded, else the end of the last bark.
function clipDuration() {
  const d = $("player").duration;
  const lastEnd = Math.max(0, ...segments.map((s) => s.end_s));
  return Number.isFinite(d) && d > 0 ? Math.max(d, lastEnd) : lastEnd * 1.05 || 1;
}

function bindOpeners(root) {
  root.querySelectorAll("[data-index].tl-bark, [data-index].mix-col").forEach((el) =>
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

function renderOverview() {
  if (!segments.length) return;
  const duration = clipDuration();
  $("overview").innerHTML = clipStats(segments, duration)
    + `<div class="card"><h3>When the dog vocalized</h3>
      <p class="lead">Each block is one bark, colored by its answer. Click a block to open it, or the lane to jump the video there.</p>
      ${clipTimeline(segments, duration)}${categoryLegend()}</div>`;
  $("clip-charts").innerHTML = `<div class="card"><h3>Situations in this clip</h3>${situationCounts(segments)}</div>`
    + `<div class="card"><h3>Situation mix per bark</h3>
      <p class="lead">Each column is one bark's full answer, adding up to 100%. Click a column to open the bark.</p>
      ${mixColumns(segments)}</div>`
    + `<div class="card"><h3>Did the clues agree?</h3>
      <p class="lead">Each clue's own top pick for every bark, against the combined answer (✓ = same as the answer, – = clue not used).</p>
      ${agreementMatrix(segments)}</div>`;
  [$("overview"), $("clip-charts")].forEach((root) => { bindOpeners(root); bindSeek(root); });
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
  renderOverview();
  if (!$("detail-view").hidden) renderDetailTimeline();
});

let detailIndex = null;

function renderDetailTimeline() {
  const host = $("seg-timeline");
  if (!host || detailIndex == null) return;
  host.innerHTML = clipTimeline(segments, clipDuration(), { focus: detailIndex });
  bindOpeners(host);
  bindSeek(host);
  movePlayhead();
}

// ---------- results: detail page (#bark-N in the address bar) ----------

let openedFromList = false;
let stopAt = null;

function openSegment(index) {
  openedFromList = true;
  location.hash = `bark-${segments[index].number}`;
}

function goToSegment(index) {
  history.replaceState(null, "", `#bark-${segments[index].number}`);
  route();
}

function backToList() {
  if (openedFromList) history.back();
  else { history.replaceState(null, "", location.pathname + location.search); route(); }
}

function route() {
  const match = location.hash.match(/^#bark-(\d+)$/);
  const index = match ? segments.findIndex((s) => s.number === Number(match[1])) : -1;
  const showDetail = index >= 0;
  $("list-view").hidden = showDetail;
  $("detail-view").hidden = !showDetail;
  if (!showDetail) {
    openedFromList = false;
    diagrams.splice(1);  // drop the detail diagram, keep "How it works"
    return;
  }
  renderDetail(index);
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

window.addEventListener("hashchange", route);
document.addEventListener("keydown", (e) => {
  if ($("detail-view").hidden || e.target.closest("input, video")) return;
  if (e.key === "Escape") backToList();
});
$("back").addEventListener("click", backToList);
$("prev").addEventListener("click", () => goToSegment(Number($("prev").dataset.index)));
$("next").addEventListener("click", () => goToSegment(Number($("next").dataset.index)));

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

function renderDetail(index) {
  const seg = segments[index];
  const player = $("player");
  player.pause();
  player.currentTime = seg.start_s;
  $("detail-pos").textContent = `Bark ${index + 1} of ${segments.length}`;
  $("prev").disabled = index === 0;
  $("next").disabled = index === segments.length - 1;
  $("prev").dataset.index = index - 1;
  $("next").dataset.index = index + 1;

  $("detail").innerHTML = [
    answerCard(seg), cluesCard(seg), opinionMatrix(seg),
    `<div class="card"><h3>How this answer was made</h3>
      <p class="lead">The same steps as in "How it works", filled in for this bark. Dashed boxes were skipped.</p>
      <div id="seg-diagram" class="diagram"></div></div>`,
    `<details class="card tech-details"><summary>Technical details <span class="note">(model outputs, for the curious)</span></summary>
      ${[preprocessingSection(seg), audioSection(seg), bodySection(seg), sceneSection(seg), fusionSection(seg), resultSection(seg)].join("")}
    </details>`,
  ].join("");
  $("play-seg").addEventListener("click", () => playSegment(seg));
  detailIndex = index;
  renderDetailTimeline();
  diagrams.splice(1);
  mountDiagram($("seg-diagram"), segmentDiagramOpts(seg));
}

function answerCard(seg) {
  const head = `<div class="answer-head"><div><div class="kicker">Bark #${seg.number} · <span class="nocaps">${seg.start_s.toFixed(2)}–${seg.end_s.toFixed(2)} s</span></div>`;
  const play = '<button type="button" id="play-seg">▶ Play this bark</button></div>';
  if (!seg.fused.present) {
    return `<div class="card">${head}<div class="answer">No answer</div></div>${play}
      <p class="meaning">None of the three clues could be used for this moment, so the app cannot make a guess.</p>
      <h4>Where this bark is in the clip</h4><div id="seg-timeline"></div></div>`;
  }
  const order = ranked(seg.fused.probs);
  const [p1, i1] = order[0];
  const [p2, i2] = order[1];
  const word = certainty(p1);
  const close = p1 - p2 < 0.15
    ? ` <strong>${esc(DISPLAY[CATEGORIES[i2]])}</strong> is a close second (${pct(p2, 0)}), so it could be either.`
    : ` The next most likely is ${esc(DISPLAY[CATEGORIES[i2]])} (${pct(p2, 0)}).`;
  return `<div class="card">${head}
      <div class="kicker">Most likely situation</div>
      <div class="answer">${esc(DISPLAY[seg.category])} <span class="pct">${pct(p1, 0)}</span></div></div>${play}
    <p class="meaning">${esc(MEANING[seg.category])}</p>
    <p class="certainty">The model is <strong>${word}</strong>: it gives this ${pct(p1, 0)}. With five options, a blind guess would get 20%.${close}</p>
    ${statTiles([
      { label: "Top chance", value: pct(p1, 0), note: `${(p1 / 0.2).toFixed(1)}× a blind guess` },
      { label: "Lead over 2nd", value: `${((p1 - p2) * 100).toFixed(0)}<small> pts</small>`, note: `ahead of ${SHORT[CATEGORIES[i2]]}` },
      { label: "Uncertainty", value: pct(uncertainty(seg.fused.probs), 0), note: "0% = certain, 100% = no idea" },
      { label: "Clues used", value: `${BRANCHES.filter((b) => seg.branches[b].present).length}<small> / 3</small>`, note: `${(seg.end_s - seg.start_s).toFixed(2)} s long` },
    ])}
    <h4>Chance of each situation</h4>${categoryBars(seg.fused.probs)}
    <p class="note">These percentages are the model's estimate, not a certainty. The app reads patterns in sound and video; it does not translate barking.</p>
    <h4>Where this bark is in the clip</h4><div id="seg-timeline"></div>
  </div>`;
}

function cluesCard(seg) {
  const f = seg.fusion;
  const best = seg.fused.present ? argmax(seg.fused.probs) : null;
  const tiles = BRANCHES.map((b) => {
    const out = seg.branches[b];
    const title = `<h5>${ICONS[b]}${PLAIN_BRANCH[b]}</h5>`;
    if (!out.present) {
      return `<div class="clue ${b} off">${title}<p><strong>Not used</strong>: ${esc(notUsedReason(seg, b))}.</p>
        <p class="note">The answer was made without this clue.</p></div>`;
    }
    const top = argmax(out.probs);
    const agrees = top === best;
    return `<div class="clue ${b}">${title}
      <p>Leaned toward <strong>${esc(DISPLAY[CATEGORIES[top]])}</strong> (${pct(out.probs[top], 0)})${agrees ? ", same as the answer" : ""}.</p>
      ${f && best != null ? effectChip(branchEffect(f, b, best)) : ""}
      <p class="note" style="margin-top:6px">${esc(CLUE_HINT[b])}</p></div>`;
  }).join("");
  return `<div class="card"><h3>What each clue said</h3>
    <p class="lead">Each clue makes its own guess. The combiner then weighs them, trusting some clues more than others.</p>
    <div class="clues">${tiles}</div></div>`;
}

function segmentDiagramOpts(seg) {
  const off = BRANCHES.filter((b) => !seg.branches[b].present).flatMap((b) => [`${b}-enc`, `${b}-head`]);
  const head = (b) => {
    const out = seg.branches[b];
    if (!out.present) return "not used";
    const top = argmax(out.probs);
    return `${SHORT[CATEGORIES[top]]} ${pct(out.probs[top], 0)}`;
  };
  const usedFrames = seg.frames.filter((fr) => fr.body_used).length;
  const nClues = BRANCHES.filter((b) => seg.branches[b].present).length;
  if (!seg.fused.present) off.push("fusion", "result");
  return {
    off,
    subs: {
      input: "this clip",
      pre: `${seg.start_s.toFixed(2)}–${seg.end_s.toFixed(2)} s`,
      "audio-enc": seg.branches.audio.present ? "sound found" : "not used",
      "audio-head": head("audio"),
      "body-enc": seg.branches.body.present ? `${usedFrames} of ${seg.frames.length} frames` : "not used",
      "body-head": head("body"),
      "scene-enc": seg.branches.scene.present ? `${seg.frames.length} frames` : "not used",
      "scene-head": head("scene"),
      fusion: `${nClues} of 3 clues`,
      result: seg.fused.present ? `${SHORT[seg.category]} ${pct(seg.probability, 0)}` : "no answer",
    },
  };
}

// ---------- technical detail sections ----------

function preprocessingSection(seg) {
  const chips = soundLanes(seg);
  return section(1, "Preprocessing", "preprocessing", `
    <dl class="kv">
      <dt>Time</dt><dd>${seg.start_s.toFixed(2)} – ${seg.end_s.toFixed(2)} s (${(seg.end_s - seg.start_s).toFixed(2)} s)</dd>
      <dt>Bark-detector confidence</dt><dd>${seg.confidence == null ? "unknown" : seg.confidence.toFixed(2)}
        <span class="note">(mean dog-vocalization score, not a calibrated probability)</span></dd>
      <dt>Frames sampled</dt><dd>${seg.num_frames}${seg.has_video ? "" : " (no video)"}</dd>
    </dl>
    <h4>Sound sections</h4>${chips}
    <h4>Sampled frames</h4>${frameStrip(seg, false)}`);
}

function audioSection(seg) {
  const out = seg.branches.audio;
  if (!out.present) return section(2, "Audio", "audio", '<p class="note">Not used: the segment has no usable audio.</p>');
  const x = out.extras;
  const nudge = x.nudge ? CATEGORIES.map((c) => {
    const v = x.nudge[c];
    return `<div class="row"><span>${esc(DISPLAY[c])}</span><span></span>
      <span class="num ${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${signed(v)}</span></div>`;
  }).join("") : "";
  return section(2, "Audio", "audio", `
    <h4>Situation (5 categories)</h4>${categoryBars(out.probs)}
    ${x.arousal ? `<h4>Arousal</h4>${namedBars(x.arousal)}` : ""}
    ${x.valence ? `<h4>Valence</h4>${namedBars(x.valence)}
      <p class="note">Valence is near chance on unseen dogs (notebook 24): treat it as a weak hint.</p>` : ""}
    ${nudge ? `<h4>Emotion nudge (logit shift)</h4>${nudge}
      <p class="note">${Object.values(x.nudge).every((v) => v === 0)
        ? "All shifts are 0: the saved emotion-nudge weights are zero, so arousal and valence do not change the situation scores."
        : "How much arousal and valence moved each category's logit before the softmax."}</p>` : ""}`);
}

function bodySection(seg) {
  const out = seg.branches.body;
  const strip = `<h4>Frames (green = used: exactly one dog)</h4>${frameStrip(seg, true)}`;
  if (!out.present) {
    return section(3, "Body", "body", `<p><strong>Not used</strong>: ${esc(seg.body_reason || "unavailable")}.</p>
      ${seg.has_video ? strip : ""}`);
  }
  const x = out.extras;
  const cues = [["tail_height", "Tail height"], ["ear_height", "Ear height"], ["head_height", "Head height"]]
    .filter(([name]) => typeof x[name] === "number")
    .map(([name, label]) => ({ label, value: x[name], tip: `${label}: ${signed(x[name], 3)}\n${x[name] >= 0 ? "above" : "below"} its reference point` }));
  const used = seg.frames.filter((f) => f.body_used).length;
  return section(3, "Body", "body", `
    <h4>Category probabilities</h4>${categoryBars(out.probs)}
    ${statTiles([
      { label: "Frames used", value: `${used}<small> / ${seg.frames.length}</small>`, note: "frames with exactly one dog" },
    ])}
    ${strip}
    <h4>Pose cues (mean over used frames)</h4>
    ${cues.length ? divergingBars(cues, "var(--body)") : '<p class="note">No pose cues.</p>'}
    <p class="note">Box-normalised; positive tail height means the tail tip is above its base. Each frame is a still pose (no wagging).</p>`);
}

function sceneSection(seg) {
  const out = seg.branches.scene;
  const s = seg.scene || {};
  if (!out.present) return section(4, "Scene", "scene", '<p><strong>Not used</strong>: no video frames.</p>');
  const thumbs = Object.fromEntries(seg.frames.map((f) => [f.time_s.toFixed(3), f.jpeg]));
  const rows = (s.frames || []).map((f) => {
    const jpeg = thumbs[f.time_s.toFixed(3)];
    return `<tr><td>${secs(f.time_s)}</td>
      <td>${jpeg ? `<img alt="" style="height:40px;max-width:72px;object-fit:cover;border-radius:4px;vertical-align:middle" src="data:image/jpeg;base64,${jpeg}">` : ""}</td>
      <td style="text-align:left;white-space:normal">${esc(f.top_prompt)}</td><td>${pct(f.score)}</td></tr>`;
  }).join("");
  return section(4, "Scene", "scene", `
    <h4>Category probabilities</h4>${categoryBars(out.probs)}
    <dl class="kv"><dt>Aggregation</dt><dd>${esc(s.aggregation || out.extras.aggregation)}</dd>
      ${s.confidence_weight != null ? `<dt>Confidence weight</dt><dd>${s.confidence_weight}</dd>` : ""}
      <dt>Temperature</dt><dd>${s.temperature}</dd></dl>
    <p class="note">Raw zero-shot scene scores are below chance on our data; the fusion head learned how much to trust them.</p>
    ${sceneFrameHeatmap(seg)}
    ${rows ? `<h4>Best prompt per frame</h4><div class="scroll"><table>
      <tr><th>Time</th><th></th><th style="text-align:left">Top prompt</th><th>Score</th></tr>${rows}</table></div>` : ""}`);
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
      return `<div class="seg" title="${p}: ${signed(v, 3)}" style="left:${start}%;width:${width}%;background:var(--${p})"></div>`;
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
  if (!f) return section(5, "Fusion", "fusion", '<p class="note">No branch could run, so there is nothing to fuse.</p>');
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
  return section(5, "Fusion", "fusion", `
    <h4>Branches</h4>${present}
    <h4>The 18 inputs</h4>
    <p class="note">Raw: centred log-probabilities per branch (zeros when the branch is missing), then the present flags.
      Standardised: (raw − training mean) ÷ training std, which is what the head multiplies by its weights.</p>
    <div class="scroll"><table><tr><th>Input</th><th>Raw</th><th>Standardised</th><th>−${zExtent.toFixed(1)} · 0 · +${zExtent.toFixed(1)}</th></tr>${featureRows}</table></div>
    <h4>What pushed each category's logit</h4>
    <p class="note">Each branch's share = weights × standardised inputs over its 5 inputs and its present flag; bias is the head's constant.
      Right of the line raises the category, left lowers it.</p>
    ${contributionChart(f)}
    <h4>Contribution matrix (source × category)</h4>
    <p class="note">The same numbers as a grid: blue raises a category's logit, red lowers it; the bottom row is their sum.</p>
    ${contributionMatrix(f)}
    <h4>Logits ÷ temperature (${f.temperature.toFixed(3)}) → softmax</h4>
    <div class="scroll"><table><tr><th>Category</th><th>Bias</th><th>Audio</th><th>Body</th><th>Scene</th>
      <th>Logit</th><th>÷ T</th><th>Probability</th></tr>${logitRows}</table></div>`);
}

function resultSection(seg) {
  if (!seg.fused.present) return section(6, "Result", "result", '<p class="note">No prediction: no branch could run on this segment.</p>');
  return section(6, "Result", "result", `
    <div class="verdict">${esc(DISPLAY[seg.category])} <span class="pct">${pct(seg.probability, 0)}</span></div>
    ${categoryBars(seg.fused.probs)}`);
}
