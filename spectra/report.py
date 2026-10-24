#!/usr/bin/env python3
"""Render data/spectra/<revision>/analysis.json into a single HTML report.

  python -m spectra.report --revision oct-14-26                  # -> data/spectra/oct-14-26/report.html
  python -m spectra.report --revision oct-14-26 --out report.html
Charts load ECharts from cdnjs; everything else (data, styles, the spectra) is inside the file.
"""
import argparse
import json
from pathlib import Path

from spectra import add_run_args, load_config, run_dir as revision_dir

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700;1,400&family=Hanken+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
<style>
:root {
  color-scheme: light;
  --paper: #f2f4f7; --panel: #ffffff; --ink: #1b2333; --muted: #5b6578; --rule: #d5dae2;
  --low: #5e3c99; --low2: #b2abd2; --mid: #f4f4f2; --high2: #fdb863; --high: #e66101;
  --seq0: #e6e9ef; --seq1: #34425e; --focus: #0072b2;
  --serif: "Spectral", Georgia, "Times New Roman", serif;
  --sans: "Hanken Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
  box-sizing: border-box;
  padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper: #141a24; --panel: #1a2130; --ink: #e6e9ef; --muted: #9aa3b5; --rule: #2c3547;
    --low: #9d84d6; --low2: #5b4d86; --mid: #232b3b; --high2: #a8692a; --high: #f08a3c;
    --seq0: #232b3b; --seq1: #c9d3e6; --focus: #56b4e9;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #141a24; --panel: #1a2130; --ink: #e6e9ef; --muted: #9aa3b5; --rule: #2c3547;
  --low: #9d84d6; --low2: #5b4d86; --mid: #232b3b; --high2: #a8692a; --high: #f08a3c;
  --seq0: #232b3b; --seq1: #c9d3e6; --focus: #56b4e9;
}
*, *::before, *::after { box-sizing: inherit; }
html { scroll-padding-top: env(safe-area-inset-top, 0px); }
body { margin: 0; background: var(--paper); color: var(--ink); font: 400 17px/1.55 var(--sans); }
a { color: inherit; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 28px; }
.prose { max-width: 68ch; }
h1, h2, h3 { font-family: var(--serif); font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: clamp(2.6rem, 6vw, 4.6rem); line-height: 1.02; margin: 0 0 .35em; font-weight: 700; }
h2 { font-size: clamp(1.7rem, 3vw, 2.3rem); line-height: 1.15; margin: 0 0 .4em; }
h3 { font-size: 1.25rem; margin: 0 0 .3em; }
p { margin: 0 0 1em; }
.muted { color: var(--muted); }
.small { font-size: .88rem; }
section { padding: 64px 0 24px; border-top: 1px solid var(--rule); }
.demo { background: var(--high); color: #fff; padding: 10px 28px; font-weight: 600; }
.demo span { font-weight: 400; }
.topbar { display: flex; justify-content: space-between; align-items: center; padding: 18px 0 0; }
.runline { font-size: .9rem; color: var(--muted); }
.themebtn { white-space: nowrap; background: none; border: 1px solid var(--rule); color: var(--ink); border-radius: 999px;
  padding: 6px 14px; font: 500 .85rem var(--sans); cursor: pointer; }

/* ---------- hero spectra ---------- */
.hero { padding: 56px 0 40px; }
.lede { font-family: var(--serif); font-size: clamp(1.15rem, 2vw, 1.4rem); line-height: 1.45; max-width: 46ch; color: var(--ink); }
.spectra { margin-top: 40px; }
.spec-head, .spec-row { display: grid; grid-template-columns: 220px 1fr; gap: 18px; align-items: center; }
.spec-labels { display: grid; height: 156px; align-items: end; }
.spec-labels span { writing-mode: vertical-rl; transform: rotate(180deg); font-size: .78rem; line-height: 1; color: var(--muted);
  white-space: nowrap; justify-self: center; align-self: end; max-height: 152px; overflow: hidden; text-overflow: ellipsis;
  padding-top: 6px; cursor: default; }
.spec-labels span.on { color: var(--ink); font-weight: 600; }
.spec-row { margin: 6px 0; }
.spec-name { font-family: var(--serif); font-size: 1.45rem; font-weight: 600; line-height: 1.1;
  display: flex; align-items: center; gap: 10px; min-width: 0; }
.spec-name i { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.spec-name b { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.band { display: grid; height: 46px; border-radius: 3px; overflow: hidden; }
.band .gap, .spec-labels .gap { background: transparent; }
.cell { border: 0; padding: 0; margin: 0; cursor: crosshair; }
.cell:hover, .cell:focus-visible { outline: 2px solid var(--ink); outline-offset: -2px; position: relative; z-index: 1; }
@media (prefers-reduced-motion: no-preference) {
  .band { animation: sweep 1.1s cubic-bezier(.2,.7,.2,1) both; }
  .spec-row:nth-child(2) .band { animation-delay: .08s } .spec-row:nth-child(3) .band { animation-delay: .16s }
  .spec-row:nth-child(4) .band { animation-delay: .24s } .spec-row:nth-child(5) .band { animation-delay: .32s }
  .spec-row:nth-child(6) .band { animation-delay: .40s } .spec-row:nth-child(7) .band { animation-delay: .48s }
  .spec-row:nth-child(8) .band { animation-delay: .56s } .spec-row:nth-child(9) .band { animation-delay: .64s }
  @keyframes sweep { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
}
.readout { display: grid; grid-template-columns: 220px 1fr; gap: 18px; margin-top: 14px; min-height: 3.2em; }
.readout p { margin: 0; font-size: .98rem; }
.legend { display: flex; align-items: center; gap: 12px; font-size: .85rem; color: var(--muted); margin-top: 18px; flex-wrap: wrap; }
.legend .ramp { width: 220px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg, var(--low), var(--low2), var(--mid), var(--high2), var(--high)); border: 1px solid var(--rule); }

/* ---------- profiles ---------- */
.profiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 40px 48px; margin-top: 28px; }
.profile h3 { font-size: 1.6rem; display: flex; align-items: center; gap: 10px; }
.profile h3 i { width: 12px; height: 12px; border-radius: 50%; }
.miniband { display: grid; height: 10px; border-radius: 2px; overflow: hidden; margin: 6px 0 16px; }
.desc { list-style: none; padding: 0; margin: 0 0 14px; }
.desc li { display: grid; grid-template-columns: 54px 1fr; gap: 12px; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--rule); }
.desc .meter { height: 6px; background: var(--seq0); border-radius: 3px; position: relative; }
.desc .meter b { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.desc .meter::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 12px; }
.chip { font-size: .85rem; padding: 2px 10px; border-radius: 999px; background: var(--panel); border: 1px solid var(--rule); }
.facts { font-size: .9rem; color: var(--muted); }

/* ---------- explorer ---------- */
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 16px 0 8px; }
.tabs .grp { width: 100%; font-size: .85rem; color: var(--muted); margin-top: 8px; }
.tab { background: var(--panel); border: 1px solid var(--rule); color: var(--ink); border-radius: 6px;
  padding: 6px 12px; font: 500 .88rem var(--sans); cursor: pointer; }
.tab[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.chart { width: 100%; height: 340px; }
.chart.tall { height: 420px; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; align-items: start; }
.traps { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 28px; margin-top: 18px; }
.trap h3 { font-size: 1.1rem; margin-bottom: 0; }
.trap p { font-size: .88rem; color: var(--muted); margin: 2px 0 0; }
.trap .chart { height: 230px; }
.callout { border-left: 3px solid var(--high); padding: 10px 16px; background: var(--panel); max-width: 68ch; }
.bignums { display: flex; gap: 48px; flex-wrap: wrap; margin: 18px 0 8px; }
.bignums div { font-family: var(--serif); font-size: 3.2rem; font-weight: 700; line-height: 1; }
.bignums small { display: block; font: 400 .9rem var(--sans); color: var(--muted); margin-top: 6px; max-width: 22ch; }

/* ---------- viewer ---------- */
.picker { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 14px 0; }
select, .btn { font: 500 .92rem var(--sans); color: var(--ink); background: var(--panel); border: 1px solid var(--rule);
  border-radius: 6px; padding: 8px 10px; max-width: 100%; }
.btn { cursor: pointer; }
.promptbox { font-family: var(--serif); font-size: 1.15rem; line-height: 1.45; white-space: pre-wrap; max-width: 80ch;
  border-left: 3px solid var(--ink); padding: 4px 0 4px 16px; margin: 10px 0 6px; }
.tagline { font-size: .85rem; color: var(--muted); margin-bottom: 18px; }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
.answer { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; display: flex; flex-direction: column; min-width: 0; }
.answer header { display: flex; justify-content: space-between; gap: 8px; padding: 10px 14px; border-bottom: 1px solid var(--rule); font-weight: 600; }
.answer header span { font-weight: 400; color: var(--muted); font-size: .85rem; white-space: nowrap; }
.answer .body { padding: 12px 14px; font-size: .92rem; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 520px; overflow: auto; }
.answer pre { white-space: pre; overflow-x: auto; background: var(--paper); padding: 8px; border-radius: 4px; font-size: .82rem; }
.answer code { font-size: .86em; }
.tablewrap { overflow-x: auto; }
table.stats { border-collapse: collapse; width: 100%; font-size: .9rem; }
table.stats th, table.stats td { text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
table.stats th:first-child, table.stats td:first-child { text-align: left; }
table.stats th { font-weight: 600; color: var(--muted); }
footer { padding: 40px 0 60px; color: var(--muted); font-size: .85rem; }
.nochart { color: var(--muted); font-size: .9rem; padding: 24px 0; }

/* ---------- usage recommendations ---------- */
.uses { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 22px 28px; margin-top: 18px; }
.use { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; }
.use h3 { font-size: 1.2rem; margin: 0 0 .35em; }
.use .for { font-size: .9rem; color: var(--muted); margin: 0 0 12px; }
.use .line { display: flex; gap: 10px; align-items: baseline; margin: 0 0 6px; font-size: .95rem; }
.use .line b { flex: none; width: 3.4em; color: var(--muted); font-weight: 600; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; }
.use .ev { font-size: .82rem; color: var(--muted); margin: 10px 0 0; }
.drive { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 22px 28px; margin-top: 18px; }
.drive article { border-top: 3px solid var(--rule); padding: 12px 0 4px; }
.drive h3 { font-size: 1.25rem; display: flex; align-items: center; gap: 10px; margin: 0 0 .2em; }
.drive h3 i { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.drive .role { font-size: .9rem; color: var(--muted); margin: 0 0 10px; }
.drive p { font-size: .92rem; margin: 0 0 .7em; }

@media (max-width: 760px) {
  .wrap { padding: 0 18px; }
  .spec-head { display: none; }
  .spec-row, .readout { grid-template-columns: 1fr; gap: 6px; }
  .spec-row { margin: 14px 0; }
  .band { height: 34px; }
  .split { grid-template-columns: 1fr; }
  .profiles { grid-template-columns: 1fr; }
  .uses, .drive { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div id="demo"></div>
<div class="wrap">
  <div class="topbar"><div class="runline" id="runline"></div><button class="themebtn" id="themebtn" type="button">Switch theme</button></div>

  <header class="hero">
    <h1>__TITLE__</h1>
    <p class="lede" id="lede"></p>
    <div class="spectra" id="spectra" role="group" aria-label="Trait spectra for each model"></div>
    <div class="readout"><div></div><p id="readout" aria-live="polite">Tap or point at any band to read it.</p></div>
    <div class="legend"><span>Less than the panel average</span><span class="ramp" aria-hidden="true"></span><span>More than the panel average</span>
      <span>Shade shows effect size; a full shade is a large difference (0.8 standard deviations or more).</span></div>
  </header>

  <section>
    <h2>Profiles</h2>
    <p class="prose muted">Each model's strongest departures from the panel, the words it uses more than the others, and the model whose overall habits sit closest to it.</p>
    <div class="profiles" id="profiles"></div>
  </section>

  <section>
    <h2>Trait explorer</h2>
    <p class="prose muted">Dots are averages across prompts; lines are 95% bootstrap intervals; the dashed line is the panel average.</p>
    <div class="tabs" id="tabs"></div>
    <p class="prose" id="traitdesc"></p>
    <div class="split">
      <div id="traitchart" class="chart"></div>
      <div id="catchart" class="chart"></div>
    </div>
  </section>

  <section>
    <h2>Traps in everyday prompts</h2>
    <p class="prose muted">Some prompts carry a false premise, an error in pasted text, a vague request, the user's own opinion, or an emotional note. These panels show how each model responded when they did. Few prompts carry each trap, so the intervals are wide.</p>
    <div class="traps" id="traps"></div>
  </section>

  <section>
    <h2>Following explicit limits</h2>
    <p class="prose muted">Share of prompts where a stated length or format limit was met, checked by code rather than by a judge.</p>
    <div id="conchart" class="chart"></div>
  </section>

  <section>
    <h2>Style map</h2>
    <p class="prose muted" id="mapnote"></p>
    <div id="mapchart" class="chart tall"></div>
  </section>

  <section>
    <h2>Can you tell them apart?</h2>
    <p class="prose muted">A simple classifier tried to name the model behind each answer, tested only on prompts it never saw during training.</p>
    <div class="bignums" id="bignums"></div>
    <div id="confchart" class="chart tall"></div>
  </section>

  <section>
    <h2>The guessing game</h2>
    <p class="prose muted">Each model sees every answer with the author names removed and matches them to the roster. Just guess uses the prompt and the answers. Guess from the research also gives the measured habits from a Spectra profile. Own answers stay out of both scores.</p>
    <div id="guess-blind">
      <h3>Just guess</h3>
      <p class="prose muted" id="guessnote"></p>
      <div id="guessbody">
      <div class="bignums" id="guessnums"></div>
      <div class="split">
        <div class="tablewrap"><table class="stats" id="guesstable"></table></div>
        <div class="tablewrap"><table class="stats" id="recogtable"></table></div>
      </div>
      <h3 style="margin-top:28px">Who they think wrote it</h3>
      <p class="prose muted">Rows are the model that actually wrote the answer. Columns are the model the others named. A cell is the share of other models' guesses. The diagonal is recognizability.</p>
      <div id="guessheat" class="chart tall"></div>
      </div>
    </div>
    <div id="guess-briefed">
      <h3 style="margin-top:40px">Guess from the research</h3>
      <p class="prose muted" id="guessbnote"></p>
      <div id="guessbbody">
      <div class="bignums" id="guessbnums"></div>
      <div class="split">
        <div class="tablewrap"><table class="stats" id="guessbtable"></table></div>
        <div class="tablewrap"><table class="stats" id="recogbtable"></table></div>
      </div>
      <h3 style="margin-top:28px">Who they think wrote it</h3>
      <p class="prose muted">Same reading as the blind heatmap. This one is after the research notes were added.</p>
      <div id="guessbheat" class="chart tall"></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Same prompt, different answers</h2>
    <div class="picker">
      <label class="small" for="catsel">Category</label><select id="catsel"></select>
      <label class="small" for="psel">Prompt</label><select id="psel"></select>
      <button class="btn" id="randbtn" type="button">Show a random prompt</button>
    </div>
    <div class="promptbox" id="promptbox"></div>
    <div class="tagline" id="tagline"></div>
    <div class="answers" id="answers"></div>
  </section>

  <section>
    <h2>Usage recommendations</h2>
    <p class="prose muted" id="recsmethod"></p>
    <div class="tablewrap" style="margin:18px 0 8px"><table class="stats" id="recstable"></table></div>
    <h3 style="margin-top:36px">By need</h3>
    <div class="uses" id="uses"></div>
    <h3 style="margin-top:40px">How to drive each model</h3>
    <p class="prose muted">These are out-of-the-box API habits with no system prompt. A consumer app can override some of them; the ones that held in every category (length, answering first, vocabulary range, em dashes) are the hardest to talk the model out of.</p>
    <div class="drive" id="drive"></div>
  </section>

  <section>
    <h2>How this was measured</h2>
    <div class="prose">
      <p>Every model received the same prompts with no system prompt, so this profiles out-of-the-box API behavior. Consumer apps add their own instructions and can behave differently.</p>
      <p>Style traits are computed by code from the text itself: length, formatting, hedge words, caveats, openers, closing offers, punctuation habits, stock phrases, readability and vocabulary range. Limits are checked by code too. Behavior traits come from coder models that applied fixed yes-or-no labels; check a sample of their labels by hand before relying on them.</p>
      <p>Differences are shown as effect sizes against the average of the models in this run, not as scores. A model can sit at one end of a trait and still be the right choice for a given job. The prompts are synthetic, English-only and written to resemble everyday use; a different prompt mix would give somewhat different profiles.</p>
      <p id="agreenote"></p>
    </div>
    <h3 style="margin-top:28px">Run details</h3>
    <div class="tablewrap"><table class="stats" id="statstable"></table></div>
  </section>

  <footer id="footer"></footer>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const A = JSON.parse(document.getElementById("data").textContent);
  const models = A.meta.models;
  const PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C6BB1", "#999999", "#B15928", "#1B9E77"];
  const color = m => PALETTE[models.indexOf(m) % PALETTE.length];
  const traits = A.traits;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

  // ---------- formatting
  function fmt(v, unit) {
    if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
    if (unit === "words") return Math.round(v) + " words";
    if (unit === "pct") return Math.round(v * 100) + "%";
    if (unit === "per100") return v.toFixed(2) + " per 100 words";
    if (unit === "score") return v.toFixed(0);
    if (unit === "ratio") return v.toFixed(3);
    return String(v);
  }
  function fmtShort(v, unit) {
    if (v === null || v === undefined || Number.isNaN(v)) return "";
    if (unit === "words") return Math.round(v);
    if (unit === "pct") return Math.round(v * 100) + "%";
    if (unit === "per100") return v.toFixed(1);
    if (unit === "score") return v.toFixed(0);
    if (unit === "ratio") return v.toFixed(2);
    return v;
  }
  function magnitude(d) {
    const a = Math.abs(d), dir = d > 0 ? "more" : "less";
    if (a < 0.2) return "about the same as the panel";
    if (a < 0.5) return "somewhat " + dir + " than the panel";
    if (a < 0.8) return "clearly " + dir + " than the panel";
    return "far " + dir + " than the panel";
  }

  // ---------- color ramp for effect sizes
  function hex2rgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map(x => x + x).join(""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function mix(a, b, t) { const x = hex2rgb(a), y = hex2rgb(b); return "rgb(" + x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",") + ")"; }
  function dColor(d) {
    const t = Math.max(-1, Math.min(1, (d || 0) / 0.8));
    const low = css("--low"), low2 = css("--low2"), mid = css("--mid"), high2 = css("--high2"), high = css("--high");
    if (t < -0.5) return mix(low2, low, (-t - 0.5) * 2);
    if (t < 0) return mix(mid, low2, -t * 2);
    if (t < 0.5) return mix(mid, high2, t * 2);
    return mix(high2, high, (t - 0.5) * 2);
  }

  // ---------- header, demo banner, lede
  if (A.meta.demo) {
    document.getElementById("demo").innerHTML = '<div class="demo">Demo data. <span>These answers were generated by a script to preview this report. They say nothing about any real model.</span></div>';
  }
  const gen = new Date(A.meta.generated);
  const paired = (A.meta.peerrank_files || []).length ? ", paired with the PeerRank run of the same revision" : "";
  document.getElementById("runline").textContent = "Revision " + A.meta.revision + paired + ", generated " + gen.toLocaleDateString(undefined, {year: "numeric", month: "short", day: "numeric"});
  const sampleTxt = A.meta.samples > 1 ? A.meta.samples + " answers per prompt" : "one answer per prompt";
  document.getElementById("lede").textContent = models.length + " models answered the same " + A.meta.n_prompts +
    " everyday prompts, " + sampleTxt + ". Each band shows where a model's habits sit relative to the others. Nothing here is a ranking.";

  // ---------- spectra
  const styleT = traits.filter(t => t.group === "style"), behT = traits.filter(t => t.group === "behavior");
  const cols = styleT.concat(behT.length ? [null] : [], behT);
  const gridCols = cols.map(t => t ? "1fr" : "14px").join(" ");
  const spectra = document.getElementById("spectra");
  const head = document.createElement("div");
  head.className = "spec-head";
  head.innerHTML = "<div></div><div class='spec-labels' style='grid-template-columns:" + gridCols + "'>" +
    cols.map((t, i) => t ? "<span data-col='" + i + "' title='" + esc(t.label + ": " + t.desc) + "'>" + esc(t.short || t.label) + "</span>" : "<span class='gap'></span>").join("") + "</div>";
  spectra.appendChild(head);
  const readout = document.getElementById("readout");
  function bandHTML(m, interactive) {
    return cols.map((t, i) => {
      if (!t) return "<span class='gap'></span>";
      const pm = t.per_model[m];
      const label = t.label + ", " + m + ": " + fmt(pm.value, t.unit) + ", " + magnitude(pm.d);
      return interactive
        ? "<button type='button' class='cell' data-col='" + i + "' data-m='" + esc(m) + "' aria-label='" + esc(label) + "'></button>"
        : "<span class='cell' title='" + esc(label) + "'></span>";
    }).join("");
  }
  function paintBands() {
    document.querySelectorAll(".cell[data-col], .miniband .cell").forEach(el => {
      const m = el.dataset.m || el.parentElement.dataset.m;
      const i = el.dataset.col !== undefined ? +el.dataset.col : +el.dataset.idx;
      const t = cols[i];
      if (t) el.style.background = dColor(t.per_model[m].d);
    });
  }
  models.forEach(m => {
    const row = document.createElement("div");
    row.className = "spec-row";
    row.innerHTML = "<div class='spec-name'><i style='background:" + color(m) + "'></i><b>" + esc(m) + "</b></div>" +
      "<div class='band' style='grid-template-columns:" + gridCols + "'>" + bandHTML(m, true) + "</div>";
    spectra.appendChild(row);
  });
  function showReadout(el) {
    const t = cols[+el.dataset.col], m = el.dataset.m, pm = t.per_model[m];
    readout.innerHTML = "<strong>" + esc(t.label) + ", " + esc(m) + ":</strong> " + esc(fmt(pm.value, t.unit)) +
      " against a panel average of " + esc(fmt(t.panel, t.unit)) + ", " + esc(magnitude(pm.d)) + ". <span class='muted'>" + esc(t.desc) + ".</span>";
    document.querySelectorAll(".spec-labels span").forEach(s => s.classList.toggle("on", s.dataset.col === el.dataset.col));
  }
  spectra.addEventListener("mouseover", e => { if (e.target.matches(".cell[data-col]")) showReadout(e.target); });
  spectra.addEventListener("focusin", e => { if (e.target.matches(".cell[data-col]")) showReadout(e.target); });
  spectra.addEventListener("click", e => { if (e.target.matches(".cell[data-col]")) { selectTrait(cols[+e.target.dataset.col].key); document.getElementById("tabs").scrollIntoView({behavior: "smooth", block: "center"}); } });

  // ---------- profiles
  const prof = document.getElementById("profiles");
  const lengthT = traits.find(t => t.key === "length"), adhT = traits.find(t => t.key === "adherence");
  models.forEach(m => {
    const card = A.cards[m] || {descriptors: []};
    const sig = (A.signature[m] || []).slice(0, 8);
    const desc = card.descriptors.map(d => {
      const w = Math.min(50, Math.abs(d.d) / 1.2 * 50);
      const pos = d.d > 0 ? "left:50%;width:" + w + "%" : "left:" + (50 - w) + "%;width:" + w + "%";
      return "<li><span class='meter' title='effect size " + d.d.toFixed(2) + "'><b style='" + pos + ";background:" + dColor(d.d) + "'></b></span><span>" + esc(d.text) + "</span></li>";
    }).join("") || "<li><span></span><span class='muted'>No trait stands out from the panel by more than a small margin.</span></li>";
    const facts = [];
    if (lengthT) facts.push("Typical length " + fmt(lengthT.per_model[m].value, "words") + ".");
    if (adhT && adhT.per_model[m].value !== null) facts.push("Met " + fmt(adhT.per_model[m].value, "pct") + " of explicit limits.");
    if (card.neighbor) facts.push("Closest overall style: " + esc(card.neighbor) + ".");
    const el = document.createElement("article");
    el.className = "profile";
    el.innerHTML = "<h3><i style='background:" + color(m) + "'></i>" + esc(m) + "</h3>" +
      "<div class='miniband' data-m='" + esc(m) + "' style='grid-template-columns:" + gridCols + "'>" +
      cols.map((t, i) => t ? "<span class='cell' data-idx='" + i + "' title='" + esc(t.label) + "'></span>" : "<span class='gap'></span>").join("") + "</div>" +
      "<ul class='desc'>" + desc + "</ul>" +
      (sig.length ? "<div class='small muted'>Words and phrases it uses more than the others</div><div class='chips'>" + sig.map(s => "<span class='chip'>" + esc(s.term) + "</span>").join("") + "</div>" : "") +
      "<p class='facts'>" + facts.join(" ") + "</p>";
    prof.appendChild(el);
  });

  // ---------- charts
  const charts = {};
  const hasE = typeof window.echarts !== "undefined";
  function chart(id) {
    const el = document.getElementById(id);
    if (!hasE) { el.innerHTML = "<p class='nochart'>Charts need an internet connection to load the charting library.</p>"; el.style.height = "auto"; return null; }
    if (charts[id]) charts[id].dispose();
    charts[id] = echarts.init(el, null, {renderer: "svg"});
    return charts[id];
  }
  const baseText = () => ({color: css("--muted"), fontFamily: "Hanken Grotesk, system-ui, sans-serif", fontSize: 12});
  const axisLine = () => ({lineStyle: {color: css("--rule")}});
  const splitLine = () => ({lineStyle: {color: css("--rule"), type: "dashed"}});

  function dotPlot(id, t, compact) {
    const c = chart(id); if (!c) return;
    const vals = models.map(m => t.per_model[m]);
    const xs = vals.flatMap(v => [v.value, v.lo, v.hi]).filter(v => v !== null && !Number.isNaN(v));
    let lo = Math.min(...xs), hi = Math.max(...xs);
    if (t.unit === "pct") { lo = 0; hi = 1; }
    else {
      const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
      const step = Math.pow(10, Math.floor(Math.log10(hi - lo))) / 2;
      lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
      if (t.unit === "words") lo = Math.max(0, lo);
    }
    c.setOption({
      animation: false,
      grid: {left: compact ? 100 : 130, right: 24, top: 12, bottom: 36},
      tooltip: {trigger: "item", formatter: p => { const v = vals[p.value[0]]; return esc(models[p.value[0]]) + "<br>" + esc(fmt(v.value, t.unit)) + (v.lo !== null ? " (95% interval " + esc(fmtShort(v.lo, t.unit)) + " to " + esc(fmtShort(v.hi, t.unit)) + ")" : "") + "<br>n = " + v.n + " prompts"; }},
      xAxis: {type: "value", min: lo, max: hi, axisLabel: Object.assign(baseText(), {formatter: v => fmtShort(v, t.unit)}), axisLine: axisLine(), splitLine: splitLine()},
      yAxis: {type: "category", data: models, inverse: true, axisLabel: Object.assign(baseText(), {color: css("--ink"), fontSize: compact ? 11 : 13, width: compact ? 90 : 120, overflow: "truncate"}), axisLine: axisLine(), axisTick: {show: false}},
      series: [{
        type: "custom",
        data: vals.map((v, i) => [i, v.value, v.lo === null ? NaN : v.lo, v.hi === null ? NaN : v.hi]),
        encode: {x: [1, 2, 3], y: 0},
        renderItem: (params, api) => {
          const i = api.value(0), v = api.value(1), l = api.value(2), h = api.value(3);
          if (Number.isNaN(v)) return null;
          const col = PALETTE[i % PALETTE.length];
          const p = api.coord([v, i]), kids = [];
          if (!Number.isNaN(l) && !Number.isNaN(h)) {
            const a = api.coord([l, i]), b = api.coord([h, i]);
            kids.push({type: "line", shape: {x1: a[0], y1: a[1], x2: b[0], y2: b[1]}, style: {stroke: col, lineWidth: 4, opacity: 0.35}});
          }
          kids.push({type: "circle", shape: {cx: p[0], cy: p[1], r: compact ? 6 : 8}, style: {fill: col, stroke: css("--paper"), lineWidth: 2}});
          return {type: "group", children: kids};
        },
        markLine: {silent: true, symbol: "none", data: [{xAxis: t.panel}], lineStyle: {color: css("--muted"), type: "dashed"}, label: {show: false}}
      }]
    });
  }
  function seqHeat(id, xcats, ycats, data, opts) {
    const c = chart(id); if (!c) return;
    c.setOption({
      animation: false,
      grid: {left: 8, right: 20, top: 10, bottom: 8, containLabel: true},
      tooltip: {formatter: p => esc(ycats[p.value[1]]) + ", " + esc(xcats[p.value[0]]) + ": " + esc(opts.fmt(p.value[2]))},
      xAxis: {type: "category", data: xcats, axisLabel: Object.assign(baseText(), {interval: 0, rotate: opts.rotate || 0}), axisLine: axisLine(), axisTick: {show: false}, splitArea: {show: false}},
      yAxis: {type: "category", data: ycats, inverse: true, axisLabel: Object.assign(baseText(), {color: css("--ink"), width: (opts.left || 130) - 10, overflow: "truncate"}), axisLine: axisLine(), axisTick: {show: false}},
      visualMap: {show: false, min: opts.min, max: opts.max, inRange: {color: [css("--seq0"), css("--seq1")]}},
      series: [{type: "heatmap",
        data: data.filter(d => d[2] !== null && d[2] !== undefined).map(d => {
          const t = opts.max > opts.min ? (d[2] - opts.min) / (opts.max - opts.min) : 0;
          return {value: d, label: {color: t > 0.55 ? css("--paper") : css("--ink")}};
        }),
        itemStyle: {borderColor: css("--paper"), borderWidth: 2},
        label: {show: true, fontSize: 11, formatter: p => opts.fmtShort(p.value[2])},
        emphasis: {itemStyle: {borderColor: css("--ink"), borderWidth: 2}}}]
    });
  }

  // trait explorer
  const tabs = document.getElementById("tabs");
  let current = traits[0].key;
  function tabGroup(name, list) {
    if (!list.length) return;
    const g = document.createElement("div"); g.className = "grp"; g.textContent = name; tabs.appendChild(g);
    list.forEach(t => {
      const b = document.createElement("button"); b.type = "button"; b.className = "tab"; b.dataset.key = t.key; b.textContent = t.label;
      b.addEventListener("click", () => selectTrait(t.key)); tabs.appendChild(b);
    });
  }
  tabGroup("Style, measured by code", styleT);
  tabGroup("Behavior, labeled by coder models", behT);
  function selectTrait(key) {
    current = key;
    const t = traits.find(x => x.key === key);
    tabs.querySelectorAll(".tab").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
    document.getElementById("traitdesc").innerHTML = "<strong>" + esc(t.label) + ".</strong> " + esc(t.desc) + "." + (t.tag ? " <span class='muted'>Only prompts tagged " + esc(t.tag) + ".</span>" : "");
    dotPlot("traitchart", t, false);
    const cats = A.meta.categories;
    const data = [];
    models.forEach((m, mi) => cats.forEach((cname, ci) => data.push([ci, mi, (t.by_category[m] || {})[cname] ?? null])));
    const vs = data.map(d => d[2]).filter(v => v !== null);
    seqHeat("catchart", cats, models, data, {min: Math.min(...vs), max: Math.max(...vs), rotate: 35, fmt: v => fmt(v, t.unit), fmtShort: v => fmtShort(v, t.unit)});
  }

  // traps
  function renderTraps() {
    const box = document.getElementById("traps");
    const keys = ["premise", "errors", "ambiguity", "agree_user", "emotion"];
    const list = keys.map(k => traits.find(t => t.key === k)).filter(Boolean);
    if (!list.length) {
      box.innerHTML = "<div class='callout'>No behavior codes in this run yet. Run <code>python -m spectra.code_behaviors</code>, then re-run the analysis and this report.</div>";
      return;
    }
    box.innerHTML = list.map(t => {
      const n = Math.max(...models.map(m => t.per_model[m].n || 0));
      return "<div class='trap'><h3>" + esc(t.label) + "</h3><p>" + esc(t.desc) + ". " + n + " prompts.</p><div class='chart' id='trap_" + t.key + "'></div></div>";
    }).join("");
    list.forEach(t => dotPlot("trap_" + t.key, t, true));
  }

  // constraints
  function renderConstraints() {
    const types = A.constraints.types;
    if (!types.length) { document.getElementById("conchart").innerHTML = "<p class='nochart'>No constrained prompts in this run.</p>"; return; }
    const labels = types.map(t => t.label + " (" + t.n_prompts + ")");
    const data = [];
    models.forEach((m, mi) => types.forEach((t, ti) => data.push([ti, mi, A.constraints.per_model[m][t.key]])));
    seqHeat("conchart", labels, models, data, {min: 0, max: 1, rotate: 30, bottom: 80, fmt: v => Math.round(v * 100) + "%", fmtShort: v => Math.round(v * 100) + "%"});
  }

  // style map
  function renderMap() {
    const S = A.style_map;
    const ax = S.axes;
    document.getElementById("mapnote").textContent = "Each dot is one answer, placed by its style features alone. Rightward means more " +
      ax.pc1.pos.join(", ") + "; leftward means more " + ax.pc1.neg.join(", ") + ". Upward means more " + ax.pc2.pos.join(", ") +
      "; downward means more " + ax.pc2.neg.join(", ") + ". Larger rings mark each model's center.";
    const c = chart("mapchart"); if (!c) return;
    const series = models.map((m, i) => ({
      name: m, type: "scatter", symbolSize: 5, itemStyle: {color: color(m), opacity: 0.33},
      data: S.points.filter(p => p[2] === i).map(p => [p[0], p[1], p[3]])
    }));
    series.push({
      name: "centers", type: "scatter", symbolSize: 22, z: 10,
      data: models.map(m => ({value: S.centroids[m], itemStyle: {color: color(m), borderColor: css("--paper"), borderWidth: 3},
        label: {show: true, position: "right", formatter: m, color: css("--ink"), fontFamily: "Spectral, Georgia, serif", fontSize: 14, fontWeight: 600}}))
    });
    c.setOption({
      animation: false,
      legend: {show: false},
      grid: {left: 20, right: 120, top: 10, bottom: 20},
      tooltip: {formatter: p => p.seriesName === "centers" ? esc(models[p.dataIndex]) + " center" : esc(p.seriesName) + ", prompt " + esc(p.value[2])},
      xAxis: {type: "value", axisLabel: {show: false}, splitLine: {show: false}, axisLine: axisLine(), axisTick: {show: false}},
      yAxis: {type: "value", axisLabel: {show: false}, splitLine: {show: false}, axisLine: axisLine(), axisTick: {show: false}},
      series
    });
  }

  // distinctiveness
  function renderDistinct() {
    const D = A.distinctiveness, box = document.getElementById("bignums");
    if (!D) { box.innerHTML = "<p class='nochart'>Not enough data to test distinctiveness.</p>"; return; }
    box.innerHTML = "<div>" + Math.round(D.accuracy * 100) + "%<small>named correctly from words and style together</small></div>" +
      "<div>" + Math.round(D.style_only_accuracy * 100) + "%<small>named correctly from style features alone</small></div>" +
      "<div>" + Math.round(D.chance * 100) + "%<small>expected by chance with " + models.length + " models</small></div>";
    const data = [];
    D.matrix.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
    seqHeat("confchart", D.labels.map(l => "guessed " + l), D.labels, data, {min: 0, max: 1, rotate: 30, bottom: 100, fmt: v => Math.round(v * 100) + "% of its answers", fmtShort: v => Math.round(v * 100) + "%"});
  }

  function fillGuess(G, ids) {
    const note = document.getElementById(ids.note);
    const nums = document.getElementById(ids.nums);
    const heat = document.getElementById(ids.heat);
    if (charts[ids.heat]) { charts[ids.heat].dispose(); delete charts[ids.heat]; }
    const body = document.getElementById(ids.body);
    if (!G) {
      note.textContent = ids.missing;
      if (body) body.hidden = true;
      nums.innerHTML = "";
      document.getElementById(ids.gtable).innerHTML = "";
      document.getElementById(ids.rtable).innerHTML = "";
      heat.innerHTML = "";
      heat.style.height = "auto";
      return;
    }
    if (body) body.hidden = false;
    heat.style.height = "";
    const pct = v => v === null || v === undefined ? "n/a" : Math.round(v * 1000) / 10 + "%";
    const best = G.guessers[0], known = G.recognizable[0];
    note.textContent = G.note;
    nums.innerHTML = "<div>" + pct(G.accuracy) + "<small>of other models' answers named correctly</small></div>" +
      "<div>" + pct(best.accuracy) + "<small>best guesser, " + esc(best.model) + "</small></div>" +
      "<div>" + pct(known.recall) + "<small>most recognizable, " + esc(known.model) + "</small></div>" +
      "<div>" + pct(G.chance) + "<small>expected by chance with " + G.k + " models</small></div>";
    document.getElementById(ids.gtable).innerHTML = "<thead><tr><th>Best guessers</th><th>Accuracy</th><th>Correct</th><th>Found own answer</th></tr></thead><tbody>" +
      G.guessers.map(r => "<tr><td style='color:" + color(r.model) + ";font-weight:600'>" + esc(r.model) + "</td><td>" + pct(r.accuracy) + "</td><td>" + r.correct + " / " + r.n + "</td><td>" + pct(r.self_rate) + "</td></tr>").join("") + "</tbody>";
    document.getElementById(ids.rtable).innerHTML = "<thead><tr><th>Most recognizable</th><th>Named by others</th><th>Correct</th></tr></thead><tbody>" +
      G.recognizable.map(r => "<tr><td style='color:" + color(r.model) + ";font-weight:600'>" + esc(r.model) + "</td><td>" + pct(r.recall) + "</td><td>" + r.correct + " / " + r.n + "</td></tr>").join("") + "</tbody>";
    const data = [];
    G.matrix.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
    seqHeat(ids.heat, G.labels.map(l => "named " + l), G.labels, data, {min: 0, max: 1, rotate: 30, bottom: 100, fmt: v => Math.round(v * 100) + "% of others' guesses", fmtShort: v => Math.round(v * 100) + "%"});
  }

  function renderGuess() {
    const G = A.guessing;
    const blind = G && G.guessers ? G : (G && G.blind) || null;
    const briefed = G && G.guessers ? null : (G && G.briefed) || null;
    const rev = A.meta.revision;
    fillGuess(blind, {
      note: "guessnote", nums: "guessnums", gtable: "guesstable", rtable: "recogtable", heat: "guessheat", body: "guessbody",
      missing: "Not run for this revision. python -m spectra.guess --revision " + rev
    });
    fillGuess(briefed, {
      note: "guessbnote", nums: "guessbnums", gtable: "guessbtable", rtable: "recogbtable", heat: "guessbheat", body: "guessbbody",
      missing: "Not run for this revision. python -m spectra.guess --revision " + rev + " --mode briefed --research " + rev
    });
  }

  // viewer
  const V = A.viewer, catsel = document.getElementById("catsel"), psel = document.getElementById("psel");
  const TAGS = A.meta.tags || {};
  function md(text) {
    const parts = String(text).split(/```/);
    return parts.map((part, i) => {
      if (i % 2 === 1) { const body = part.replace(/^[\w+-]*\n/, ""); return "<pre><code>" + esc(body) + "</code></pre>"; }
      return esc(part).replace(/^#{1,6}\s+(.+)$/gm, "<strong>$1</strong>").replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/`([^`\n]+)`/g, "<code>$1</code>");
    }).join("");
  }
  catsel.innerHTML = "<option value=''>All categories</option>" + A.meta.categories.map(c => "<option>" + esc(c) + "</option>").join("");
  function fillPrompts() {
    const list = V.prompts.filter(p => !catsel.value || p.category === catsel.value);
    psel.innerHTML = list.map(p => "<option value='" + p.id + "'>" + esc(p.id + "  " + p.prompt.replace(/\s+/g, " ").slice(0, 80)) + "</option>").join("");
    showPrompt(psel.value);
  }
  function showPrompt(id) {
    const p = V.prompts.find(x => x.id === id); if (!p) return;
    psel.value = id;
    document.getElementById("promptbox").textContent = p.prompt;
    const bits = [p.category];
    p.tags.forEach(t => bits.push(TAGS[t] || t));
    if (p.constraint) bits.push("limit: " + p.constraint.type.replace(/_/g, " ") + (p.constraint.value !== undefined ? " " + p.constraint.value : ""));
    document.getElementById("tagline").textContent = bits.join("; ");
    const R = V.responses[id] || {};
    document.getElementById("answers").innerHTML = models.map(m => {
      const r = R[m];
      return "<article class='answer'><header><span style='color:" + color(m) + ";font-weight:600'>" + esc(m) + "</span><span>" + (r ? r.words + " words" : "no answer") + "</span></header><div class='body'>" + (r ? md(r.text) : "") + "</div></article>";
    }).join("");
  }
  catsel.addEventListener("change", fillPrompts);
  psel.addEventListener("change", () => showPrompt(psel.value));
  document.getElementById("randbtn").addEventListener("click", () => { const opts = [...psel.options]; showPrompt(opts[Math.floor(Math.random() * opts.length)].value); });
  fillPrompts();

  // usage recommendations (ranked from measured traits; no LLM judge)
  (function renderRecs() {
    const R = A.recommendations; if (!R) return;
    document.getElementById("recsmethod").textContent = R.method;
    const names = m => "<span style='color:" + color(m) + ";font-weight:600'>" + esc(m) + "</span>";
    document.getElementById("recstable").innerHTML = "<thead><tr><th>Need</th><th>Use</th><th>Avoid</th></tr></thead><tbody>" +
      R.uses.map(u => "<tr><td>" + esc(u.need) + "</td><td>" + u.use.map(names).join(", ") + "</td><td>" + (u.avoid.length ? u.avoid.map(names).join(", ") : "—") + "</td></tr>").join("") + "</tbody>";
    document.getElementById("uses").innerHTML = R.uses.map(u =>
      "<article class='use'><h3>" + esc(u.need) + "</h3>" +
      "<p class='for'>" + esc(u.for) + "</p>" +
      "<div class='line'><b>Use</b><span>" + u.use.map(names).join(", ") + "</span></div>" +
      (u.avoid.length ? "<div class='line'><b>Avoid</b><span>" + u.avoid.map(names).join(", ") + "</span></div>" : "") +
      "<div class='line'><b>How</b><span>" + esc(u.how) + "</span></div>" +
      "<div class='line'><b>Watch</b><span>" + esc(u.watch) + "</span></div>" +
      "<p class='ev'>" + esc(u.evidence) + "</p></article>").join("");
    document.getElementById("drive").innerHTML = R.drive.map(d =>
      "<article><h3><i style='background:" + color(d.model) + "'></i>" + esc(d.model) + "</h3>" +
      "<p class='role'>" + esc(d.role) + "</p>" +
      "<p><strong>Do. </strong>" + esc(d.do) + "</p>" +
      "<p><strong>Don't. </strong>" + esc(d.dont) + "</p>" +
      "<p class='ev'>" + esc(d.evidence) + "</p></article>").join("");
  })();

  // run details
  const RS = A.runstats;
  const n = v => v === null || v === undefined ? "n/a" : Math.round(v);
  document.getElementById("statstable").innerHTML = "<thead><tr><th>Model</th><th>Answers</th><th>Errors</th><th>Median words</th><th>Latency</th><th>Output tokens</th><th>Reasoning tokens (reported)</th><th>Cost</th>" + (A.meta.samples > 1 ? "<th title='Median coefficient of variation of length across samples of the same prompt'>Length variation</th>" : "") + "</tr></thead><tbody>" +
    models.map(m => { const r = RS[m]; return "<tr><td>" + esc(m) + "</td><td>" + r.responses + "</td><td>" + r.errors + "</td><td>" + Math.round(r.median_words) + "</td><td>" + (r.median_latency_s === null ? "n/a" : r.median_latency_s.toFixed(1) + " s") + "</td><td>" + n(r.median_output_tokens) + "</td><td>" + n(r.median_reasoning_tokens) + "</td><td>" + (r.total_cost_usd === null ? "n/a" : "$" + r.total_cost_usd.toFixed(2)) + "</td>" + (A.meta.samples > 1 ? "<td>" + (r.length_cv_across_samples === null ? "n/a" : Math.round(r.length_cv_across_samples * 100) + "%") + "</td>" : "") + "</tr>"; }).join("") + "</tbody>";
  if (A.agreement) {
    const ks = Object.entries(A.agreement).filter(([, v]) => v !== null).map(([k, v]) => k.replace(/_/g, " ") + " " + v.toFixed(2));
    document.getElementById("agreenote").textContent = "Agreement between the two coder models (Cohen's kappa; above 0.6 is usually considered good): " + ks.join("; ") + ".";
  }
  document.getElementById("footer").textContent = A.meta.n_responses + " answers analyzed. Coder models: " + (A.meta.coders.length ? A.meta.coders.join(", ") : "none") + ". System prompt: " + (A.meta.system_prompt ? "custom" : "none") + ".";

  // ---------- render + theme
  function renderAll() {
    paintBands();
    selectTrait(current);
    renderTraps(); renderConstraints(); renderMap(); renderDistinct(); renderGuess();
  }
  renderAll();
  const tb = document.getElementById("themebtn");
  tb.addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    renderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  window.addEventListener("resize", () => Object.values(charts).forEach(c => c && c.resize()));
})();
</script>
</body>
</html>
"""


def _trait(analysis, key):
    return next(t for t in analysis["traits"] if t["key"] == key)


def _rank(trait, high=True):
    return sorted(trait["per_model"], key=lambda m: trait["per_model"][m]["value"], reverse=high)


def _val(trait, model):
    return trait["per_model"][model]["value"]


def _constraint_rate(analysis, model):
    types = analysis["constraints"]["types"]
    vals = analysis["constraints"]["per_model"][model]
    num = sum(vals[t["key"]] * t["n_prompts"] for t in types)
    den = sum(t["n_prompts"] for t in types) or 1
    return num / den


def build_recommendations(analysis):
    """Map measured traits to use-cases. No LLM judge: a judge would replay PeerRank taste."""
    models = analysis["meta"]["models"]
    rs = analysis["runstats"]
    length = _trait(analysis, "length")
    answer = _trait(analysis, "answer_first")
    initiative = _trait(analysis, "initiative")
    hedging = _trait(analysis, "hedging")
    errors = _trait(analysis, "errors")
    amb = _trait(analysis, "ambiguity")
    premise = _trait(analysis, "premise")
    adhere = _trait(analysis, "adherence")
    structure = _trait(analysis, "structure")
    warm = _trait(analysis, "warm_tone")
    advice = _trait(analysis, "advice")
    hooks = _trait(analysis, "hooks")
    read = _trait(analysis, "readability")
    vocab = _trait(analysis, "vocab")
    emdash = _trait(analysis, "emdash")
    caveats = _trait(analysis, "caveats")
    agree = _trait(analysis, "agree_user")

    short = _rank(length, high=False)[0]
    long = _rank(length, high=True)[0]
    first = _rank(answer, high=True)[0]
    warmup = _rank(answer, high=False)[0]
    extra = _rank(initiative, high=True)[0]
    justask = _rank(initiative, high=False)[0]
    err_best = [m for m in _rank(errors) if _val(errors, m) >= 0.99]
    err_worst = [m for m in _rank(errors, high=False) if _val(errors, m) <= 0.55]
    amb_best = _rank(amb)[0]
    amb_worst = _rank(amb, high=False)[0]
    adh_best = [m for m in _rank(adhere) if _val(adhere, m) >= max(_val(adhere, x) for x in models) - 1e-9]
    adh_worst = _rank(adhere, high=False)[0]
    struct = _rank(structure)[0]
    plain = _rank(read)[0]
    warmest = _rank(warm)[0]
    fastest = min(models, key=lambda m: rs[m]["median_latency_s"] or 1e9)
    slowest = max(models, key=lambda m: rs[m]["median_latency_s"] or 0)
    cheapest = min(models, key=lambda m: rs[m]["total_cost_usd"] if rs[m]["total_cost_usd"] is not None else 1e9)
    hedge_hi = _rank(hedging)[0]
    hedge_lo = _rank(hedging, high=False)[0]
    prem_lo = _rank(premise, high=False)[0]

    def pct(t, m):
        return f"{100 * _val(t, m):.0f}%"

    def words(m):
        return f"{_val(length, m):.0f}w"

    def lat(m):
        return f"{rs[m]['median_latency_s']:.1f}s"

    def cost(m):
        return f"${rs[m]['total_cost_usd']:.2f}"

    def tok_ratio(m):
        w = rs[m]["median_words"] or 1
        return rs[m]["median_output_tokens"] / w

    uses = [
        {
            "need": "Short, fast first answers",
            "for": "Chat, support, lookup — anything where the first two sentences should be the answer.",
            "use": [short],
            "avoid": [long],
            "how": "Ask the question. Do not add 'be concise'; it already is. Add 'go longer' or 'give options' only if you want depth.",
            "watch": f"{long} writes the longest answer in every category and answers first only {pct(answer, long)} of the time.",
            "evidence": f"{short} is shortest in 10/10 categories ({words(short)} geom. mean), answers first {pct(answer, first)}, median {lat(short)}.",
        },
        {
            "need": "Long teaching or 'do the work for me'",
            "for": "Explanations, how-tos, advice, brainstorms where extra options and warmth are welcome.",
            "use": [long],
            "avoid": [short] if short != long else [],
            "how": "Cap words and say 'start with the answer'. Do not use exact item counts unless you will trim extras. Expect labeled alternatives.",
            "watch": f"Helpfulness overflows hard constraints: {adh_worst} met {pct(adhere, adh_worst)} of limits (fails exact item counts and 'no intro'). Follow-up offers are rare ({pct(hooks, long)}); extras land in the body, then it stops.",
            "evidence": f"{long} longest in 10/10 categories ({words(long)} geom. mean, {rs[long]['median_words']:.0f}w median), does extra {pct(initiative, long)}, warm {pct(warm, long)}, unasked advice {pct(advice, long)}.",
        },
        {
            "need": "Catch errors in pasted text",
            "for": "Editing, invoices, summaries, or any prompt that pastes user text which might be wrong.",
            "use": err_best or [_rank(errors)[0]],
            "avoid": err_worst,
            "how": "You do not need to say the text is wrong. The top models flag it unprompted. If you must use a weaker model, add 'check the source for errors' and verify.",
            "watch": f"{' and '.join(err_worst) if err_worst else 'None'} flag only {pct(errors, err_worst[0]) if err_worst else 'n/a'} of pasted-error traps and miss every error inside a summarize prompt. n=6 traps; intervals are wide.",
            "evidence": f"Pasted-error rate (n=6, kappa={analysis['agreement'].get('error_handling', 0):.2f}): " +
                        ", ".join(f"{m} {pct(errors, m)}" for m in _rank(errors)) + ".",
        },
        {
            "need": "Vague or underspecified requests",
            "for": "Prompts missing audience, length, format, or goal.",
            "use": [amb_best],
            "avoid": [amb_worst],
            "how": "Leave the ambiguity in if you want it surfaced. With a silent-guesser, state assumptions yourself or ask it to list them first.",
            "watch": f"{amb_worst} guesses silently half the time ({pct(amb, amb_worst)}). {prem_lo} is also the only model that ever accepted a false premise ({pct(premise, prem_lo)} vs 100% for the rest).",
            "evidence": f"Ambiguity surfaced (n=11, kappa={analysis['agreement'].get('clarification', 0):.2f}): {amb_best} {pct(amb, amb_best)}, {amb_worst} {pct(amb, amb_worst)}.",
        },
        {
            "need": "Hard format or length limits",
            "for": "Exact N bullets, word caps, 'no intro', tables, code-only, start with yes/no.",
            "use": adh_best,
            "avoid": [adh_worst],
            "how": "State the limit in the prompt. After a draft-aware word-count check, word caps are 89–100% for everyone; the remaining misses are extra bullets and preambles.",
            "watch": f"{adh_worst} will add bonus items and an intro. If you need it anyway, say 'exactly N items, no intro, no extras' and count.",
            "evidence": "Limits met (27 constrained prompts): " +
                        ", ".join(f"{m} {100 * _constraint_rate(analysis, m):.0f}%" for m in sorted(models, key=lambda x: -_constraint_rate(analysis, x))) + ".",
        },
        {
            "need": "Cheap bulk generation",
            "for": "High-volume drafts where latency and error-catching are secondary.",
            "use": [cheapest],
            "avoid": [max(models, key=lambda m: rs[m]["total_cost_usd"] or 0)],
            "how": f"Budget ~{rs[cheapest]['median_latency_s']:.0f}s per call. Add 'flag errors in the source' and 'state your assumptions' — {cheapest} will not do either by default. It is strong on headers and lists.",
            "watch": f"{cheapest} is the slowest model ({lat(cheapest)}) and bills {tok_ratio(cheapest):.1f} output tokens per visible word (hidden thinking). Do not use it to summarize or edit untrusted pasted text.",
            "evidence": f"{cheapest} cost {cost(cheapest)} for 200 prompts vs {cost(max(models, key=lambda m: rs[m]['total_cost_usd'] or 0))} for the most expensive. Formatting {_val(structure, cheapest):.1f}/100w.",
        },
        {
            "need": "Interactive latency",
            "for": "Live chat or any loop where the user is waiting.",
            "use": [fastest],
            "avoid": [slowest],
            "how": "Prefer the fast cluster. Hidden thinking, not visible length, is what makes the slow models slow.",
            "watch": f"{slowest} median {lat(slowest)}; {long} {lat(long)} even though it writes more visible words than {cheapest}.",
            "evidence": "Median latency: " + ", ".join(f"{m} {lat(m)}" for m in sorted(models, key=lambda x: rs[x]["median_latency_s"] or 0)) + ".",
        },
        {
            "need": "Plain drafts in the user's voice",
            "for": "Emails, notes, captions, anything that should read like a person, not a template.",
            "use": [plain],
            "avoid": [struct],
            "how": "Ask for the draft. It already prefers paragraphs over headers. Do not rely on it to catch errors in source text.",
            "watch": f"{plain} is tied-worst on pasted errors ({pct(errors, plain)}). {struct} will dress the same request in headers, bullets and bold ({_val(structure, struct):.1f}/100w).",
            "evidence": f"{plain} highest Flesch reading ease ({_val(read, plain):.1f}), lowest markdown ({_val(structure, plain):.1f}/100w), casual signature vocabulary.",
        },
        {
            "need": "Structured how-tos and docs",
            "for": "Guides that should arrive with headers, numbered steps and bold callouts.",
            "use": [struct],
            "avoid": [plain],
            "how": "Ask for sections. These models lean on markdown even when you do not. Opus also offers a follow-up ({pct(hooks, struct)}).",
            "watch": "Formatting density is not the same as length: the longest model is not the most structured.",
            "evidence": f"{struct} highest markdown density ({_val(structure, struct):.1f}/100w). {plain} lowest ({_val(structure, plain):.1f}/100w).",
        },
        {
            "need": "Warm extras, options, a companion tone",
            "for": "Brainstorms, advice, coaching — when doing more than asked is the point.",
            "use": [warmest],
            "avoid": [justask],
            "how": f"Let it run. Cap length if the extras get expensive. {warmest} almost never opens with praise ({100 * _val(_trait(analysis, 'warm_open'), warmest):.1f}%) — warmth is in the body, not the first line.",
            "watch": f"{justask} does extra only {pct(initiative, justask)} of the time and stays cooler ({pct(warm, justask)}). Hedging is highest in {hedge_hi} ({_val(hedging, hedge_hi):.2f}/100w) and lowest in {hedge_lo} ({_val(hedging, hedge_lo):.2f}/100w) — not the same as warmth.",
            "evidence": f"{warmest} warm {pct(warm, warmest)}, extra {pct(initiative, extra)}, unasked advice {pct(advice, warmest)}. Caveats {pct(caveats, warmest)} vs hedges {_val(hedging, warmest):.2f}/100w.",
        },
        {
            "need": "Two-model pipelines",
            "for": "When one habit should draft and another should check or compress.",
            "use": [short, extra if extra in err_best else (err_best[0] if err_best else _rank(errors)[0])],
            "avoid": [],
            "how": f"Expand with {long}, compress with {short}. Draft cheap with {cheapest}, then error-check with {' or '.join(err_best) if err_best else _rank(errors)[0]}. First pass {short}, deepen with {struct} if the answer needs structure.",
            "watch": "Do not pair two models that share a miss: Flash and Muse both let summarize-errors through.",
            "evidence": f"Opposite poles on the style map: {short} (short, dense, hedged) vs {long} (long, plain, extra). Family confusion: Flash is mistaken for Argon 23.5% of the time but is its safety opposite.",
        },
    ]

    drive = [
        {
            "model": short,
            "role": "The short, answer-first model — closest neighbor " + analysis["cards"][short]["neighbor"] + ".",
            "do": "Use it as the default for chat and lookup. Put hard constraints here. Add 'give three options' or 'explain why' when you want more than a first answer.",
            "dont": "Do not pile on 'be brief' or 'no fluff'. It already does just what was asked "
                    f"({pct(initiative, short)} extra) and wrote zero caveats in 200 answers, while hedging often "
                    f"({_val(hedging, short):.2f}/100w). Denser wording (Flesch {_val(read, short):.1f}).",
            "evidence": f"{words(short)} geom. mean, answer-first {pct(answer, short)} in 10/10 categories, "
                        f"{lat(short)}, {cost(short)} / 200 prompts, {tok_ratio(short):.2f} tokens/word.",
        },
        {
            "model": next((m for m in models if "fable" in m), amb_best),
            "role": "The messy-prompt specialist — surfaces ambiguity and flags pasted errors.",
            "do": "Send underspecified or dirty-source prompts here. Expect a warmer open and em dashes.",
            "dont": "Do not treat it as interchangeable with Opus. Same vendor, opposite punctuation "
                    f"({_val(emdash, next((m for m in models if 'fable' in m), amb_best)):.2f}/100w vs "
                    f"{_val(emdash, next((m for m in models if 'opus' in m), struct)):.3f}/100w).",
            "evidence": f"Ambiguity {pct(amb, amb_best)}, pasted errors {pct(errors, err_best[0] if err_best else amb_best)}, "
                        f"limits {pct(adhere, adh_best[0])}, warm open {100 * _val(_trait(analysis, 'warm_open'), amb_best):.0f}%.",
        },
        {
            "model": struct,
            "role": "Structured extras with error catching — the most identifiable fingerprint in the panel.",
            "do": "Use for how-tos, docs, and any paste that might be wrong. It will format, do extra, and offer another turn.",
            "dont": "Do not expect em dashes or a short reply. It almost never uses the mark Fable treats as default, "
                    f"and hedges the most ({_val(hedging, hedge_hi):.2f}/100w).",
            "evidence": f"Markdown {_val(structure, struct):.1f}/100w, extra {pct(initiative, struct)}, "
                        f"hooks {pct(hooks, struct)}, errors {pct(errors, struct)}, {lat(struct)}, {cost(struct)} / 200 prompts.",
        },
        {
            "model": long,
            "role": "The long, warm, option-heavy model — shortest in PeerRank, longest here.",
            "do": "Use when you want alternatives and a companion tone. Always set a word cap and 'answer first'. Verify item counts.",
            "dont": "Do not send exact-N or no-preamble jobs. Do not confuse low hedging "
                    f"({_val(hedging, long):.2f}/100w, lowest) with caution — it caveats the most ({pct(caveats, long)}) "
                    f"and repeats vocabulary the most (MATTR {_val(vocab, long):.3f}, last in 10/10 categories).",
            "evidence": f"{words(long)} geom. mean, answer-first {pct(answer, long)} (last in 10/10), "
                        f"extra {pct(initiative, long)}, {lat(long)}, {tok_ratio(long):.2f} tokens/word, {cost(long)} / 200 prompts.",
        },
        {
            "model": cheapest,
            "role": "Cheapest, slowest, style-cousin of the long model — and its safety opposite.",
            "do": "Use for bulk structured drafts. Add explicit error-check and 'state assumptions' instructions.",
            "dont": "Do not use it to summarize or edit untrusted text, and do not use it when the user is waiting "
                    f"({lat(cheapest)} median). It is the only model that ever accepted a false premise ({pct(premise, cheapest)}).",
            "evidence": f"{cost(cheapest)} / 200 prompts, {tok_ratio(cheapest):.2f} tokens/word, "
                        f"errors {pct(errors, cheapest)}, ambiguity {pct(amb, cheapest)}, "
                        f"classifier recall lowest in the panel (confused with {long} 23.5% of the time).",
        },
        {
            "model": plain,
            "role": "Plainest conversational prose — neighbor "
                    + analysis["cards"].get(plain, {}).get("neighbor", "—") + ".",
            "do": "Use for drafts that should sound like a person. Ask it to write, not to outline.",
            "dont": "Do not use it as a fact-checker. Tied-worst on pasted errors, casual register "
                    "('perfect', 'get', 'want', 'etc').",
            "evidence": f"Flesch {_val(read, plain):.1f}, markdown {_val(structure, plain):.1f}/100w, "
                        f"em dashes {_val(emdash, plain):.2f}/100w, errors {pct(errors, plain)}, {lat(plain)}, {cost(plain)} / 200 prompts.",
        },
    ]

    # Keep drive in roster order
    order = {m: i for i, m in enumerate(models)}
    drive = sorted(drive, key=lambda d: order.get(d["model"], 99))

    return {
        "method": (
            "Picks are ranked from this run's measured traits (length, answer-first, initiative, "
            "constraint adherence, pasted-error and ambiguity rates, latency, cost) — not from an LLM judge. "
            "A judge would replay PeerRank taste: the same verbosity that put the longest model last under a "
            "200-word cap is the initiative this section treats as a feature. Spectra does not score correctness. "
            "Consumer apps add system prompts and can shift some habits; category-invariant ones (length, "
            "answering first, vocabulary range, em dashes) are the hardest to override. Pasted-error n=6 and "
            "emotion n=5 have wide intervals even when coder agreement is high."
        ),
        "uses": uses,
        "drive": drive,
    }


def main():
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    ap.add_argument("--run-dir", default=None, help="override data/spectra/<revision>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Behavioral spectra")
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else revision_dir(load_config(args.config), args.revision)
    data = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    data["recommendations"] = build_recommendations(data)
    from spectra.guess import load_jsonl as load_guesses, summarize_guesses
    guesses = load_guesses(run_dir / "guesses.jsonl")
    if guesses:
        models = data["meta"]["models"]
        revision = data["meta"].get("revision")
        packed = {}
        for mode in ("blind", "briefed"):
            summary = summarize_guesses(guesses, models, mode, revision)
            if summary:
                packed[mode] = summary
        if packed:
            data["guessing"] = packed
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__TITLE__", args.title).replace("__DATA__", payload)
    out = Path(args.out) if args.out else run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
