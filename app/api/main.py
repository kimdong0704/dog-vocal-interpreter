"""Minimal FastAPI app: upload a clip -> preprocessing + audio branch -> predictions.

Run from the repo root:
    uvicorn app.api.main:app
then open http://127.0.0.1:8000
"""

from __future__ import annotations

import shutil
import tempfile
from html import escape
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from woofdecoder.pipeline import Pipeline, SegmentResult
from woofdecoder.taxonomy import CATEGORIES, DISPLAY_NAMES

STYLE = """
:root { --bg:#f6f5f2; --card:#fff; --text:#222; --muted:#777; --line:#e5e2dc;
        --accent:#c2652a; --bar:#e9e5de; --pos:#3c8d5a; --neg:#c0453b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1b1a19; --card:#262523; --text:#eee; --muted:#999; --line:#3a3835;
          --accent:#e08a4f; --bar:#3a3835; --pos:#6cc08a; --neg:#e57a70; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:760px; margin:0 auto; padding:32px 16px 64px; }
h1 { margin:0 0 4px; font-size:28px; }
h1 span { color:var(--accent); }
.sub { color:var(--muted); margin:0 0 24px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:20px; margin-bottom:16px; }
form { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
input[type=file] { flex:1; min-width:0; }
button { background:var(--accent); color:#fff; border:0; border-radius:8px;
         padding:10px 18px; font-size:15px; cursor:pointer; }
button:disabled { opacity:.6; cursor:wait; }
.summary { color:var(--muted); margin:24px 0 12px; }
.seg-head { display:flex; justify-content:space-between; align-items:baseline;
            gap:12px; flex-wrap:wrap; }
.time { color:var(--muted); font-variant-numeric:tabular-nums; }
.verdict { font-size:22px; font-weight:600; margin:4px 0 16px; }
.verdict .pct { color:var(--accent); }
h4 { margin:18px 0 8px; font-size:13px; text-transform:uppercase;
     letter-spacing:.05em; color:var(--muted); }
.row { display:grid; grid-template-columns:150px 1fr 56px; gap:10px;
       align-items:center; margin:4px 0; font-size:14px; }
.track { background:var(--bar); border-radius:4px; height:10px; overflow:hidden; }
.fill { background:var(--muted); height:100%; }
.top .fill { background:var(--accent); }
.top { font-weight:600; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chip { border:1px solid var(--line); border-radius:999px; padding:2px 10px; font-size:13px; }
.note { color:var(--muted); font-size:12px; margin-top:4px; }
.empty { text-align:center; color:var(--muted); }
@media (max-width:480px) { .row { grid-template-columns:110px 1fr 48px; } }
"""

app = FastAPI(title="WoofDecoder")
pipeline = Pipeline()  # load models once at startup


def page(body: str = "") -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WoofDecoder</title><style>{STYLE}</style></head>
<body><main>
<h1>Woof<span>Decoder</span></h1>
<p class="sub">Upload a clip of a dog. Each detected vocalization is classified from its audio.</p>
<div class="card">
  <form action="/predict" method="post" enctype="multipart/form-data"
        onsubmit="this.querySelector('button').disabled=true;
                  this.querySelector('button').textContent='Analyzing…';">
    <input type="file" name="clip" accept="video/*,audio/*" required>
    <button type="submit">Analyze</button>
  </form>
</div>
{body}
</main></body></html>"""


def bar_row(label: str, value: float, top: bool = False) -> str:
    return (
        f'<div class="row{" top" if top else ""}"><span>{escape(label)}</span>'
        f'<div class="track"><div class="fill" style="width:{value * 100:.1f}%"></div></div>'
        f'<span class="num">{value * 100:.1f}%</span></div>'
    )


def segment_card(number: int, result: SegmentResult) -> str:
    segment = result.segment
    extras = result.extras
    probs = result.output.as_dict()

    html = [
        '<div class="card">',
        (
            f'<div class="seg-head"><strong>Segment {number}</strong>'
            f'<span class="time">{segment.start_s:.2f} – {segment.end_s:.2f} s</span></div>'
        ),
        (
            f'<div class="verdict">{escape(DISPLAY_NAMES[result.category])} '
            f'<span class="pct">{result.probability * 100:.0f}%</span></div>'
        ),
        "<h4>All categories</h4>",
    ]
    for name in sorted(CATEGORIES, key=probs.get, reverse=True):
        html.append(bar_row(DISPLAY_NAMES[name], probs[name], top=name == result.category))

    html.append("<h4>Why</h4>")
    if segment.sound_sections:
        chips = "".join(
            f'<span class="chip">{escape(label)} {start:.2f}–{end:.2f}s</span>'
            for label, (start, end), _ in segment.sound_sections
        )
        html.append(f'<div class="chips">{chips}</div>')
    if segment.confidence is not None:
        html.append(f'<p class="note">Bark detector confidence: {segment.confidence:.2f}</p>')

    for key in ("arousal", "valence"):
        values = extras.get(key)
        if not values:
            continue
        html.append(f"<h4>{key.capitalize()}</h4>")
        best = max(values, key=values.get)
        for name, value in values.items():
            html.append(bar_row(name, value, top=name == best))
        if key == "valence":
            html.append('<p class="note">Valence is near chance on unseen dogs (notebook 24).</p>')

    nudge = extras.get("nudge")
    if nudge:
        html.append("<h4>Emotion nudge (logit shift)</h4>")
        for name, shift in nudge.items():
            css = "pos" if shift > 0 else "neg" if shift < 0 else ""
            html.append(
                f'<div class="row"><span>{escape(DISPLAY_NAMES.get(name, name))}</span><span></span>'
                f'<span class="num {css}">{shift:+.2f}</span></div>'
            )

    shown = {"top_category", "arousal", "valence", "nudge"}
    for key, value in extras.items():
        if key not in shown:
            html.append(f'<p class="note">{escape(key)}: {escape(str(value))}</p>')

    html.append("</div>")
    return "".join(html)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return page()


@app.post("/predict", response_class=HTMLResponse)
def predict(clip: Annotated[UploadFile, File()]) -> str:
    suffix = Path(clip.filename or "").suffix
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        with open(path, "wb") as out:
            shutil.copyfileobj(clip.file, out)
        results = pipeline.run(path, verbose=True)

    name = escape(clip.filename or "clip")
    if not results:
        return page(f'<div class="card empty">No dog vocalization detected in {name}.</div>')

    cards = "".join(segment_card(number, result) for number, result in enumerate(results, 1))
    return page(f'<p class="summary">{name} · {len(results)} segment(s)</p>{cards}')
