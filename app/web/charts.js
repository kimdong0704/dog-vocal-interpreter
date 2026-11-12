"use strict";
// Charts for the results pages: plain HTML/CSS/SVG, no libraries. Loaded before app.js; the
// functions use its constants (CATEGORIES, DISPLAY, SHORT, BRANCHES, PLAIN_BRANCH, ICONS) and helpers
// (esc, pct, signed, argmax, span), so they may only be called once app.js has run.
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

function seqLegend(low = "0%", high = "100%") {
  return `<div class="scale"><span>${low}</span><span class="grad seq"></span><span>${high}</span></div>`;
}

const seqBackground = (v) => `color-mix(in oklab, var(--seq) ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, var(--card))`;

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

// What the whole clip was mostly about: a ring of how many barks got each answer.
function clipSummary(segs) {
  const found = segs.filter((s) => s.fused.present);
  const counts = CATEGORIES.map((c) => found.filter((s) => s.category === c).length);
  const order = counts.map((n, i) => [n, i]).filter(([n]) => n > 0).sort((a, b) => b[0] - a[0]);
  // Only name a "mostly" category when it clearly leads (no tie).
  const lead = order.length && order[0][0] > (order[1] ? order[1][0] : 0) ? CATEGORIES[order[0][1]] : null;
  const headline = lead
    ? `<div class="headline" style="--c:${catColor(lead)}"><span class="kicker">Mostly</span><b><span class="dot"></span>${esc(DISPLAY[lead])}</b></div>`
    : `<div class="headline"><span class="kicker">Mixed</span><b>No single situation</b></div>`;
  if (!found.length) return `<div class="headline"><span class="kicker">Result</span><b>No answer</b></div>`;
  const fracs = counts.map((n) => n / found.length);
  return `${headline}
    <div class="answer-viz">${donut(fracs, String(found.length), found.length === 1 ? "bark" : "barks",
      (i) => `${DISPLAY[CATEGORIES[i]]}: ${counts[i]} of ${found.length} barks`)}
    ${legendRows(order.map(([n, i]) => ({ c: CATEGORIES[i], value: `${n}`, top: CATEGORIES[i] === lead,
      tip: `${DISPLAY[CATEGORIES[i]]}: ${n} of ${found.length} barks` })))}</div>`;
}

// One clickable card per bark: its answer and how sure the model is.
function barkCard(seg, index) {
  const time = `<div class="bc-top"><span>#${seg.number}</span><span class="num">${seg.start_s.toFixed(1)} s</span></div>`;
  if (!seg.fused.present) {
    return `<button type="button" class="bark-card" data-index="${index}" aria-label="Open bark ${seg.number}">
      ${time}<span class="bc-pct muted">–</span><span class="bc-cat muted">No answer</span></button>`;
  }
  return `<button type="button" class="bark-card" style="--c:${catColor(seg.category)}" data-index="${index}"
      aria-label="Open bark ${seg.number}: ${esc(DISPLAY[seg.category])} ${pct(seg.probability, 0)}">
    ${time}<span class="bc-pct num">${pct(seg.probability, 0)}</span>
    <span class="bc-cat"><span class="dot"></span>${esc(SHORT[seg.category])}</span></button>`;
}

// Ring chart of fractions (one arc per category); `center` and `word` sit in the middle.
function donut(fracs, center, word, tipFor) {
  const r = 62;
  const C = 2 * Math.PI * r;
  const best = argmax(fracs);
  let start = 0;
  const arcs = CATEGORIES.map((c, i) => {
    const len = fracs[i] * C;
    if (len <= 0) return "";
    const gap = len > 6 && fracs[i] < 1 ? 2 : 0;
    const arc = `<circle cx="80" cy="80" r="${r}" fill="none" stroke="${catColor(c)}" stroke-width="${i === best ? 22 : 16}"
      stroke-dasharray="${Math.max(0, len - gap).toFixed(2)} ${C.toFixed(2)}" stroke-dashoffset="${(-start).toFixed(2)}"
      data-tip="${esc(tipFor(i))}"/>`;
    start += len;
    return arc;
  }).join("");
  return `<svg class="donut" viewBox="0 0 160 160" role="img" aria-label="${esc(tipFor(best))}">
    <circle cx="80" cy="80" r="${r}" fill="none" stroke="var(--bar)" stroke-width="16"/>
    <g transform="rotate(-90 80 80)">${arcs}</g>
    <text class="donut-num" x="80" y="88" text-anchor="middle">${esc(center)}</text>
    <text class="donut-lab" x="80" y="110" text-anchor="middle">${esc(word)}</text></svg>`;
}

// rows: [{c (category), value (text), top, tip}]
function legendRows(rows) {
  return `<div class="plegend">${rows.map((r) => `<div class="pl-row${r.top ? " top" : ""}" data-tip="${esc(r.tip)}">
    <span class="dot" style="--c:${catColor(r.c)}"></span><span>${esc(SHORT[r.c])}</span><span class="num">${r.value}</span></div>`).join("")}</div>`;
}

// Every category with its chance, highest first.
function probLegend(probs) {
  const best = argmax(probs);
  return legendRows(probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).map(([p, i]) =>
    ({ c: CATEGORIES[i], value: pct(p, 0), top: i === best, tip: `${DISPLAY[CATEGORIES[i]]}: ${pct(p)}` })));
}

// Five columns (one per category) for a clue's opinion; the top one is labelled.
function miniColumns(probs, who) {
  const best = argmax(probs);
  return `<div class="minicols">${CATEGORIES.map((c, i) => `<div class="minicol${i === best ? "" : " dim"}"
      data-tip="${esc(`${who} · ${DISPLAY[c]}: ${pct(probs[i])}`)}">
      <i style="height:${Math.max(3, probs[i] * 100).toFixed(1)}%;background:${catColor(c)}"></i><span>${ABBR[c]}</span></div>`).join("")}</div>`;
}

// How much the combiner listened to a clue for this answer (logit, relative to its average push).
function effectBadge(value) {
  const [cls, word] = value > 0.25 ? ["pos", "▲ Pushed toward the answer"]
    : value >= -0.25 ? ["", "● Little effect"] : ["neg", "▼ Pushed away from it"];
  return `<div class="effect ${cls}" data-tip="${esc(`Push in the combiner: ${signed(value)} (logit)`)}">${word}</div>`;
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
      style="left:${left}%;width:max(6px, ${width}%);background:${color}"></button>`;
  }).join("");
  return `<div class="timeline" data-duration="${duration}"><div class="tl-lane">${blocks}<div class="playhead"></div></div>
    ${timeAxis(0, duration)}</div>`;
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
  return `<h4>Per frame</h4>${probHeatmap(rows)}`;
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
