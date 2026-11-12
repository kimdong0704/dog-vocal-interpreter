"use strict";
// Charts for the results pages: plain HTML/CSS, no libraries. Loaded after app.js and uses its
// constants (CATEGORIES, DISPLAY, BRANCHES, PLAIN_BRANCH) and helpers (esc, pct, signed, argmax).
//
// Colors (app.css): --c-<category> is categorical (identity of a situation), --seq is the
// one-hue sequential ramp (probabilities), --div-pos / --div-neg the diverging pair (logit pushes).
// Situation colors never carry meaning alone: every mark also has a legend, a letter or a tooltip.

const ABBR = {
  alert_aggressive: "A",
  playful_excited: "P",
  distressed_lonely: "D",
  attention_seeking: "W",
  other: "O",
};
const catColor = (c) => `var(--c-${c})`;

// ---------- tooltip: any element with data-tip="text" ----------

const tip = document.createElement("div");
tip.id = "tip";
tip.setAttribute("role", "tooltip");
tip.hidden = true;
document.body.appendChild(tip);

function placeTip(x, y) {
  const gap = 12;
  const box = tip.getBoundingClientRect();
  let left = x + gap;
  let top = y + gap;
  if (left + box.width > innerWidth - 8) left = x - box.width - gap;
  if (top + box.height > innerHeight - 8) top = y - box.height - gap;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

function showTip(event) {
  const target = event.target.closest && event.target.closest("[data-tip]");
  if (!target) { tip.hidden = true; return; }
  if (tip.textContent !== target.dataset.tip) tip.textContent = target.dataset.tip;
  tip.hidden = false;
  placeTip(event.clientX, event.clientY);
}

document.addEventListener("pointermove", showTip);
document.addEventListener("pointerdown", showTip);
document.addEventListener("focusin", (event) => {
  const target = event.target.closest && event.target.closest("[data-tip]");
  if (!target) return;
  tip.textContent = target.dataset.tip;
  tip.hidden = false;
  const box = target.getBoundingClientRect();
  placeTip(box.left, box.bottom);
});
document.addEventListener("focusout", () => { tip.hidden = true; });
document.addEventListener("scroll", () => { tip.hidden = true; }, true);

// ---------- small pieces ----------

// items: [{label, value (HTML), note}]
function statTiles(items) {
  return `<div class="tiles">${items.map((t) => `<div class="tile">
    <div class="tile-label">${esc(t.label)}</div>
    <div class="tile-value">${t.value}</div>
    ${t.note ? `<div class="tile-note">${esc(t.note)}</div>` : ""}</div>`).join("")}</div>`;
}

function categoryLegend() {
  return `<div class="legend">${CATEGORIES.map((c) =>
    `<span><span class="swatch" style="background:${catColor(c)}"></span>${esc(DISPLAY[c])}</span>`).join("")}</div>`;
}

function seqLegend(low = "0%", high = "100%") {
  return `<div class="scale"><span>${low}</span><span class="grad seq"></span><span>${high}</span></div>`;
}

function divLegend(extent) {
  return `<div class="scale"><span>−${extent.toFixed(1)} lowers</span><span class="grad div"></span><span>raises +${extent.toFixed(1)}</span></div>`;
}

const seqBackground = (v) => `color-mix(in oklab, var(--seq) ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, var(--card))`;

function divBackground(v, extent) {
  const share = Math.round(Math.min(1, Math.abs(v) / (extent || 1)) * 90);
  return `color-mix(in oklab, var(${v >= 0 ? "--div-pos" : "--div-neg"}) ${share}%, var(--bar))`;
}

// Entropy of a probability vector as a share of the maximum (0 = certain, 1 = uniform).
function uncertainty(probs) {
  const h = -probs.reduce((sum, p) => (p > 0 ? sum + p * Math.log(p) : sum), 0);
  return h / Math.log(probs.length);
}

function niceStep(span, target = 6) {
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].find((m) => m * power >= raw) * power;
}

// Tick labels under a lane that spans [start, end] seconds.
function timeAxis(start, end) {
  const span = end - start || 1;
  const step = niceStep(span);
  const digits = step < 0.1 ? 2 : step < 1 ? 1 : 0;
  const ticks = [];
  for (let t = Math.ceil(start / step) * step; t <= end + 1e-9; t += step) ticks.push(t);
  return `<div class="axis">${ticks.map((t) => {
    const at = ((t - start) / span) * 100;
    const shift = at < 3 ? "0" : at > 97 ? "-100%" : "-50%";
    return `<span style="left:${at}%;transform:translateX(${shift})">${t.toFixed(digits)} s</span>`;
  }).join("")}</div>`;
}

// ---------- clip overview (list page) ----------

function clipStats(segs, duration) {
  const found = segs.filter((s) => s.fused.present);
  const vocal = segs.reduce((sum, s) => sum + (s.end_s - s.start_s), 0);
  const meanTop = found.length ? found.reduce((sum, s) => sum + s.probability, 0) / found.length : null;
  let agree = 0;
  let opinions = 0;
  found.forEach((s) => BRANCHES.forEach((b) => {
    if (!s.branches[b].present) return;
    opinions += 1;
    if (argmax(s.branches[b].probs) === argmax(s.fused.probs)) agree += 1;
  }));
  const withBody = segs.filter((s) => s.branches.body.present).length;
  return statTiles([
    { label: "Barks found", value: String(segs.length), note: `${found.length} with an answer` },
    { label: "Vocal time", value: `${vocal.toFixed(1)}<small> s</small>`, note: `${pct(vocal / duration, 0)} of the ${duration.toFixed(1)} s clip` },
    { label: "Average certainty", value: meanTop == null ? "–" : pct(meanTop, 0), note: "chance of the top answer, averaged" },
    { label: "Clues agree", value: opinions ? pct(agree / opinions, 0) : "–", note: `${agree} of ${opinions} clue opinions matched the answer` },
    { label: "Body visible", value: `${withBody}<small> / ${segs.length}</small>`, note: "barks where the dog's pose could be read" },
  ]);
}

// opts.focus: index of the bark to highlight (the others are dimmed).
function clipTimeline(segs, duration, opts = {}) {
  const blocks = segs.map((s, i) => {
    const left = (s.start_s / duration) * 100;
    const width = ((s.end_s - s.start_s) / duration) * 100;
    const text = `Bark #${s.number} · ${s.start_s.toFixed(2)}–${s.end_s.toFixed(2)} s\n`
      + (s.fused.present ? `${DISPLAY[s.category]} ${pct(s.probability, 0)}` : "No answer");
    const dim = opts.focus != null && opts.focus !== i ? " dim" : "";
    const color = s.fused.present ? catColor(s.category) : "var(--muted)";
    return `<button type="button" class="tl-bark${dim}" data-index="${i}" data-tip="${esc(text)}" aria-label="${esc(text)}"
      style="left:${left}%;width:max(5px, ${width}%);background:${color}"></button>`;
  }).join("");
  return `<div class="timeline" data-duration="${duration}"><div class="tl-lane">${blocks}<div class="playhead"></div></div>
    ${timeAxis(0, duration)}</div>`;
}

function situationCounts(segs) {
  const found = segs.filter((s) => s.fused.present);
  const counts = CATEGORIES.map((c) => found.filter((s) => s.category === c).length);
  const means = CATEGORIES.map((_, i) => (found.length ? found.reduce((sum, s) => sum + s.fused.probs[i], 0) / found.length : 0));
  const maxCount = Math.max(1, ...counts);
  const rows = (values, scale, fmt, tipText) => CATEGORIES.map((c, i) => `<div class="row" data-tip="${esc(tipText(c, i))}">
      <span>${esc(DISPLAY[c])}</span>
      <div class="track"><div class="fill" style="width:${(values[i] / scale) * 100}%;background:${catColor(c)}"></div></div>
      <span class="num">${fmt(values[i])}</span></div>`).join("");
  return `<div class="split">
    <div><h4>Top answer, per bark</h4>
      ${rows(counts, maxCount, String, (c, i) => `${DISPLAY[c]}: top answer for ${counts[i]} of ${found.length} barks`)}</div>
    <div><h4>Average chance over all barks</h4>
      ${rows(means, 1, (v) => pct(v, 0), (c, i) => `${DISPLAY[c]}: ${pct(means[i])} on average`)}</div>
  </div>`;
}

// 100% stacked column per bark: the full answer, not just the top category.
function mixColumns(segs) {
  const columns = segs.map((s, i) => {
    const label = `<span class="mix-x">#${s.number}</span>`;
    if (!s.fused.present) {
      return `<button type="button" class="mix-col empty" data-index="${i}" data-tip="Bark #${s.number}: no answer">
        <span class="mix-bar"></span>${label}</button>`;
    }
    const parts = CATEGORIES.map((c, j) => {
      const p = s.fused.probs[j];
      if (p < 0.005) return "";
      return `<span class="mix-seg" style="flex:${p} 1 0;background:${catColor(c)}"
        data-tip="${esc(`Bark #${s.number} · ${s.start_s.toFixed(2)} s\n${DISPLAY[c]}: ${pct(p)}`)}"></span>`;
    }).join("");
    return `<button type="button" class="mix-col" data-index="${i}" aria-label="Open bark ${s.number}">
      <span class="mix-bar">${parts}</span>${label}</button>`;
  }).join("");
  return `${categoryLegend()}
    <div class="mix-wrap"><div class="mix-y"><span style="top:0">100%</span><span style="top:50%">50%</span><span style="top:100%">0%</span></div>
      <div class="mix">${columns}</div></div>`;
}

// Rows: each clue and the answer; columns: barks. Cell = that row's top situation (letter + color),
// ✓ when a clue agrees with the answer.
function agreementMatrix(segs) {
  const rows = [...BRANCHES.map((b) => [b, PLAIN_BRANCH[b]]), ["fused", "Answer"]];
  const head = `<tr><th></th>${segs.map((s) => `<th>#${s.number}</th>`).join("")}<th>Agrees</th></tr>`;
  const body = rows.map(([key, label]) => {
    let agree = 0;
    let used = 0;
    const cells = segs.map((s) => {
      const out = key === "fused" ? s.fused : s.branches[key];
      if (!out.present) return `<td data-tip="${esc(`Bark #${s.number} · ${label}: not used`)}"><span class="dotcell none">–</span></td>`;
      const top = argmax(out.probs);
      const category = CATEGORIES[top];
      const matches = key !== "fused" && s.fused.present && top === argmax(s.fused.probs);
      if (key !== "fused") { used += 1; if (matches) agree += 1; }
      const text = `Bark #${s.number} · ${label}\n${DISPLAY[category]} ${pct(out.probs[top], 0)}`
        + (key === "fused" ? "" : matches ? "\nSame as the answer" : "\nDiffers from the answer");
      return `<td data-tip="${esc(text)}"><span class="dotcell c-${category}" style="background:${catColor(category)}">${ABBR[category]}</span>${matches ? '<span class="agree">✓</span>' : ""}</td>`;
    }).join("");
    const total = key === "fused" ? "" : used ? `${agree}/${used}` : "–";
    return `<tr${key === "fused" ? ' class="answer-row"' : ""}><th class="rowh">${esc(label)}</th>${cells}<td class="num">${total}</td></tr>`;
  }).join("");
  const key = CATEGORIES.map((c) => `<span><span class="dotcell c-${c}" style="background:${catColor(c)}">${ABBR[c]}</span> ${esc(DISPLAY[c])}</span>`).join("");
  return `<div class="legend">${key}</div><div class="scroll"><table class="matrix agree-m">${head}${body}</table></div>`;
}

// ---------- heatmaps (detail page) ----------

// rows: [{label (HTML), values: number[] | null, off: text when values is null, strong}]
function probHeatmap(rows, { highlightMax = true } = {}) {
  const head = `<tr><th></th>${CATEGORIES.map((c) => `<th>${esc(DISPLAY[c])}</th>`).join("")}</tr>`;
  const body = rows.map((r) => {
    if (!r.values) return `<tr><th class="rowh">${r.label}</th><td class="off" colspan="${CATEGORIES.length}">${esc(r.off)}</td></tr>`;
    const best = argmax(r.values);
    return `<tr${r.strong ? ' class="answer-row"' : ""}><th class="rowh">${r.label}</th>${r.values.map((v, i) =>
      `<td class="cell${v > 0.55 ? " hi" : ""}${highlightMax && i === best ? " max" : ""}" style="background:${seqBackground(v)}"
        data-tip="${esc(`${r.tipLabel || ""}${DISPLAY[CATEGORIES[i]]}: ${pct(v)}`)}">${pct(v, 0)}</td>`).join("")}</tr>`;
  }).join("");
  return `<div class="scroll"><table class="matrix">${head}${body}</table></div>${seqLegend()}`;
}

function opinionMatrix(seg) {
  const rows = BRANCHES.map((b) => ({
    label: `${ICONS[b]} ${esc(PLAIN_BRANCH[b])}`,
    tipLabel: `${PLAIN_BRANCH[b]} · `,
    values: seg.branches[b].present ? seg.branches[b].probs : null,
    off: `not used: ${notUsedReason(seg, b)}`,
  }));
  rows.push({
    label: "Combined answer",
    tipLabel: "Combined · ",
    values: seg.fused.present ? seg.fused.probs : null,
    off: "no answer",
    strong: true,
  });
  return `<div class="card"><h3>Every clue's opinion, side by side</h3>
    <p class="lead">Each row adds up to 100%. Darker = more likely; the outlined cell is that row's top pick.
      The combined answer is not a simple average: the combiner trusts some clues more.</p>
    ${probHeatmap(rows)}</div>`;
}

function sceneFrameHeatmap(seg) {
  const x = seg.branches.scene.extras || {};
  const probs = x.frame_probs || [];
  const times = x.frame_times_s || [];
  if (!probs.length) return "";
  const thumbs = Object.fromEntries(seg.frames.map((f) => [f.time_s.toFixed(3), f.jpeg]));
  const rows = probs.map((values, i) => {
    const time = times[i];
    const jpeg = time != null ? thumbs[Number(time).toFixed(3)] : null;
    return {
      label: `${jpeg ? `<img class="thumb" alt="" src="data:image/jpeg;base64,${jpeg}">` : ""}${time != null ? secs(time) : `frame ${i + 1}`}`,
      tipLabel: `${time != null ? secs(time) : `Frame ${i + 1}`} · `,
      values,
    };
  });
  rows.push({ label: "Combined", tipLabel: "Combined · ", values: seg.branches.scene.probs, strong: true });
  return `<h4>Every frame's match (frames × situations)</h4>${probHeatmap(rows)}`;
}

function contributionMatrix(f) {
  const rows = [["Bias", f.bias], ...BRANCHES.map((b) => [PLAIN_BRANCH[b], f.contributions[b]]), ["Total (logit)", f.logits]];
  const extent = Math.max(1e-6, ...rows.flatMap(([, values]) => values.map(Math.abs)));
  const head = `<tr><th></th>${CATEGORIES.map((c) => `<th>${esc(DISPLAY[c])}</th>`).join("")}</tr>`;
  const body = rows.map(([label, values], r) => `<tr${r === rows.length - 1 ? ' class="answer-row"' : ""}><th class="rowh">${esc(label)}</th>
    ${values.map((v, i) => `<td class="cell${Math.abs(v) / extent > 0.6 ? " hi" : ""}" style="background:${divBackground(v, extent)}"
      data-tip="${esc(`${label} → ${DISPLAY[CATEGORIES[i]]}: ${signed(v, 3)}`)}">${signed(v)}</td>`).join("")}</tr>`).join("");
  return `<div class="scroll"><table class="matrix">${head}${body}</table></div>${divLegend(extent)}`;
}

// ---------- small diverging bars ----------

// rows: [{label, value, tip}]; one series, so one color.
function divergingBars(rows, color, minExtent = 0.5) {
  const extent = Math.max(minExtent, ...rows.map((r) => Math.abs(r.value)));
  return rows.map((r) => {
    const width = (Math.abs(r.value) / extent) * 50;
    const left = r.value >= 0 ? 50 : 50 - width;
    return `<div class="contrib" data-tip="${esc(r.tip || `${r.label}: ${signed(r.value, 3)}`)}"><span>${esc(r.label)}</span>
      <div class="lane"><div class="zero"></div><div class="seg" style="left:${left}%;width:${width}%;background:${color}"></div></div>
      <span class="num">${signed(r.value)}</span></div>`;
  }).join("") + `<div class="contrib-axis"><span></span><div><span>−${extent.toFixed(2)}</span><span>0</span><span>+${extent.toFixed(2)}</span></div><span></span></div>`;
}

function miniDiverging(value, extent) {
  const width = (Math.min(Math.abs(value), extent) / extent) * 50;
  const left = value >= 0 ? 50 : 50 - width;
  return `<div class="mini-lane"><div class="zero"></div><div class="mini-bar ${value >= 0 ? "up" : "down"}" style="left:${left}%;width:${width}%"></div></div>`;
}

// ---------- sound sections inside one bark ----------

function soundLanes(seg) {
  if (!seg.sound_sections.length) return '<p class="note">No sound sections.</p>';
  const start = seg.start_s;
  const end = seg.end_s;
  const span = end - start || 1;
  const labels = [...new Set(seg.sound_sections.map((s) => s.label))];
  const lanes = labels.map((label) => {
    const parts = seg.sound_sections.filter((s) => s.label === label);
    const bars = parts.map((s) => {
      const left = ((Math.max(s.start_s, start) - start) / span) * 100;
      const width = ((Math.min(s.end_s, end) - Math.max(s.start_s, start)) / span) * 100;
      return `<div class="sl-bar" style="left:${left}%;width:max(3px, ${width}%)"
        data-tip="${esc(`${label}\n${s.start_s.toFixed(2)}–${s.end_s.toFixed(2)} s · score ${s.score.toFixed(2)}`)}"></div>`;
    }).join("");
    const best = Math.max(...parts.map((s) => s.score));
    return `<div class="sl-row"><span>${esc(label)}</span><div class="sl-lane">${bars}</div><span class="num">${best.toFixed(2)}</span></div>`;
  }).join("");
  return `${lanes}<div class="sl-row"><span></span>${timeAxis(start, end)}<span class="num note">best score</span></div>`;
}
