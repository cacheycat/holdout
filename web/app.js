import { h, api, follow, fmt, lineChart, flowStrip, traceList, flowStatus } from "/kit/kit.js";

const $ = (id) => document.getElementById(id);
const S = { meta: null, runs: [], runId: null, run: null, events: [], cands: [], sel: null, baseline: null, live: false, stop: null };

const sh = (v) => h("span", { class: v < 0 ? "neg" : v >= 0.5 ? "pos" : "" }, fmt.num(v, 2));
const status = (c) => c.holdout ? (c.holdout.pass ? "pass" : "fail") : c.score.passed ? "gate" : "";
const statusTag = (c) => c.holdout
  ? h("span", { class: `tag ${c.holdout.pass ? "pass" : "fail"}` }, c.holdout.pass ? "holdout pass" : "holdout fail")
  : c.score.passed ? h("span", { class: "tag gate" }, "cleared gates")
  : h("span", { class: "tag" }, `${c.score.failed.length} gate${c.score.failed.length > 1 ? "s" : ""} failed`);

// Candidates come from the finished run's output, or are rebuilt from events while live.
function candidates() {
  if (S.run?.output?.candidates) return S.run.output.candidates;
  const byName = new Map();
  for (const ev of S.events) {
    if (ev.type === "candidate") byName.set(ev.payload.name, { ...ev.payload });
    if (ev.type === "holdout" && byName.has(ev.payload.name)) byName.get(ev.payload.name).holdout = { pass: ev.payload.pass };
  }
  return [...byName.values()];
}

function renderStats() {
  const out = S.run?.output, c = S.cands;
  const peeksUsed = out ? out.peeks_used : S.events.filter((e) => e.type === "holdout").length;
  const budget = S.meta.peek_budget;
  const best = c.reduce((m, x) => Math.max(m, x.score.robust), 0);
  const gens = S.run?.input?.generations ?? "–";
  const gen = c.reduce((m, x) => Math.max(m, x.gen), 0);
  const stat = (label, value, note) => h("dl", { class: "stat" }, h("dt", {}, label), h("dd", {}, value), h("small", {}, note));
  $("stats").replaceChildren(
    stat("Strategies tried", c.length, `generation ${gen} of ${gens}`),
    stat("Cleared all gates", c.filter((x) => x.score.passed).length, "seven checks on train and validation"),
    stat("Best robust score", fmt.num(best, 2), "lower of train and validation Sharpe, penalised"),
    stat("Holdout checks", [`${peeksUsed}/${budget}`, h("span", { class: "pips" }, Array.from({ length: budget }, (_, i) => h("i", { class: i < peeksUsed ? "used" : "" })))],
      "the sealed segment can be read three times per run"),
    stat("Holdout passes", c.filter((x) => x.holdout?.pass).length, "the agent is told pass or fail, never the numbers"));
}

function renderBoard() {
  const rows = [...S.cands].sort((a, b) => b.score.robust - a.score.robust || b.score.val.sharpe - a.score.val.sharpe);
  if (!rows.length) { $("board").replaceChildren(h("caption", { class: "empty" }, S.live ? "Waiting for the first generation…" : "No strategies yet. Start a research run.")); return; }
  $("board").replaceChildren(
    h("thead", {}, h("tr", {}, ["Strategy", "Status", "Robust", "Train", "Valid.", "Trades", "Stability", "Deflated", "Gen"].map((t) => h("th", {}, t)))),
    h("tbody", {}, rows.map((c) => h("tr", { class: c.name === S.sel ? "sel" : "", onclick: () => select(c.name) },
      h("td", { class: "name" }, c.name, h("small", {}, c.parent ? `from ${c.parent}` : "new idea")),
      h("td", {}, statusTag(c)),
      h("td", {}, h("span", { class: "bar" }, h("i", { style: { width: Math.min(100, c.score.robust * 60) + "%" } })), fmt.num(c.score.robust, 2)),
      h("td", {}, sh(c.score.train.sharpe)), h("td", {}, sh(c.score.val.sharpe)),
      h("td", {}, `${c.score.train.trades}+${c.score.val.trades}`),
      h("td", {}, fmt.pct(c.score.stability, 0)), h("td", {}, fmt.num(c.score.dsr, 2)),
      h("td", { class: "dim" }, c.gen)))));
}

async function renderDetail() {
  const c = S.cands.find((x) => x.name === S.sel), el = $("detail");
  if (!c) { el.replaceChildren(h("div", { class: "panel-head" }, h("h2", {}, "Strategy")), h("div", { class: "empty" }, "Select a strategy to inspect it.")); return; }
  const rule = c.rule.split("; ").map((part, i) => h("div", {}, i === 0 ? [h("b", {}, part.split(" when ")[0]), " when " + part.split(" when ")[1]] : part));
  const seg = (label, m, cls = "") => h("div", { class: cls }, h("dt", {}, label),
    h("dd", { class: m ? (m.sharpe < 0 ? "neg" : "") : "dim" }, m ? fmt.num(m.sharpe, 2) : "sealed"),
    h("small", {}, m ? `${fmt.signedPct(m.total_return)} · dd ${fmt.pct(m.max_dd, 0)} · ${m.trades} trades` : "not promoted"));
  const chart = h("div", {});
  el.replaceChildren(
    h("div", { class: "panel-head" }, h("h2", {}, `Generation ${c.gen}${c.parent ? ` · derived from ${c.parent}` : " · new idea"}`), statusTag(c)),
    h("h3", {}, c.name), h("p", { class: "thesis" }, c.thesis || ""),
    h("div", { class: "rule" }, rule),
    h("dl", { class: "split" }, seg("Train Sharpe", c.score.train), seg("Validation Sharpe", c.score.val),
      seg("Holdout Sharpe", c.holdout?.metrics, c.holdout ? "" : "sealed")),
    h("ul", { class: "gates" }, c.score.gates.map((g) => h("li", {}, h("span", { class: g.pass ? "ok" : "no" }), h("span", {}, g.label), h("span", {}, g.detail)))),
    chart,
    h("div", { class: "legend" }, h("span", {}, h("i", { style: { borderColor: "var(--accent)" } }), "strategy"),
      h("span", {}, h("i", { class: "dash", style: { borderColor: "var(--faint)" } }), "buy and hold"),
      h("span", {}, "log scale · costs included")));
  if (!S.run?.output) { chart.append(h("div", { class: "kit-empty" }, "Equity curve is drawn when the run completes.")); return; }
  try {
    const d = await api(`/api/equity?run_id=${S.runId}&name=${encodeURIComponent(c.name)}`);
    if (S.sel !== c.name) return;
    const log = (arr) => d.t.map((t, i) => [t, Math.log(arr[i])]);
    lineChart(chart, {
      height: 210, yFmt: (v) => fmt.signedPct(Math.exp(v) - 1, 0), xFmt: (t) => new Date(t * 1000).toISOString().slice(0, 7), zero: 0,
      series: [{ name: "buy and hold", points: log(d.bench), color: "var(--faint)", dash: "3 3", width: 1 },
               { name: c.name, points: log(d.equity), color: "var(--accent)", width: 1.6 }],
      bands: [{ x0: d.t[0], x1: d.train_end, label: "train" }, { x0: d.train_end, x1: d.val_end, label: "validation", cls: "val" },
              ...(d.holdout_visible ? [{ x0: d.val_end, x1: d.end, label: "holdout", cls: "sealed" }] : [])],
    });
    if (!d.holdout_visible) chart.append(h("div", { class: "kit-empty" }, "The holdout segment is not served for this strategy: it was never promoted."));
  } catch (e) { chart.append(h("div", { class: "kit-empty" }, e.message)); }
}

function renderFrontier() {
  const frontier = S.run?.output?.frontier || (() => {
    const out = []; let best = 0;
    const gens = [...new Set(S.cands.map((c) => c.gen))].sort((a, b) => a - b);
    for (const g of gens) { best = Math.max(best, ...S.cands.filter((c) => c.gen === g).map((c) => c.score.robust)); out.push(best); }
    return out;
  })();
  const b = S.baseline, pts = (arr) => arr.map((v, i) => [i + 1, v]);
  const agentName = S.run?.input?.baseline ? "random search (this run)" : "agent";
  lineChart($("frontier"), {
    height: 170, zero: 0, yFmt: (v) => fmt.num(v, 2), xFmt: (v) => `gen ${Math.round(v)}`,
    series: [...(b ? [{ name: `random, best of ${b.seeds}`, points: pts(b.max), color: "var(--faint)", dash: "2 3", width: 1 },
                      { name: "random, median", points: pts(b.median), color: "var(--cool)", dash: "5 3", width: 1.3 }] : []),
             { name: agentName, points: pts(frontier), color: "var(--accent)", width: 2 }],
  });
  $("frontier-legend").replaceChildren(
    h("span", {}, h("i", { style: { borderColor: "var(--accent)" } }), agentName),
    b ? h("span", {}, h("i", { class: "dash", style: { borderColor: "var(--cool)" } }), `random search, median of ${b.seeds} seeds`) : null,
    b ? h("span", {}, h("i", { class: "dash", style: { borderColor: "var(--faint)" } }), `random search, best seed`) : null,
    b ? h("span", {}, `random cleared the gates in ${b.any_gate_pass} of ${b.seeds} seeds`) : null);
}

function renderLineage() {
  const el = $("lineage"), c = S.cands;
  if (!c.length) { el.replaceChildren(h("div", { class: "kit-empty" }, "No strategies yet")); return; }
  const gens = [...new Set(c.map((x) => x.gen))].sort((a, b) => a - b);
  const NW = 150, NH = 22, GX = 44, GY = 9, pos = {};
  let rows = 0;
  gens.forEach((g, gi) => { const col = c.filter((x) => x.gen === g); rows = Math.max(rows, col.length);
    col.forEach((x, i) => { pos[x.name] = { x: gi * (NW + GX), y: 20 + i * (NH + GY) }; }); });
  const W = gens.length * (NW + GX) - GX, H = 20 + rows * (NH + GY);
  const svg = h("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  gens.forEach((g, gi) => svg.append(h("text", { class: "ln-gen", x: gi * (NW + GX), y: 10 }, `GEN ${g}`)));
  for (const x of c) if (x.parent && pos[x.parent]) {
    const a = pos[x.parent], b = pos[x.name], x1 = a.x + NW, y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2, mx = (x1 + x2) / 2;
    svg.append(h("path", { class: "ln-edge", d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}` }));
  }
  for (const x of c) {
    const p = pos[x.name], label = x.name.length > 21 ? x.name.slice(0, 20) + "…" : x.name;
    svg.append(h("g", { class: `ln-node ${status(x)} ${x.name === S.sel ? "sel" : ""}`, onclick: () => select(x.name) },
      h("title", {}, `${x.name} · robust ${x.score.robust}`),
      h("rect", { x: p.x + .5, y: p.y + .5, width: NW, height: NH, rx: 2 }), h("text", { x: p.x + 8, y: p.y + 15 }, label)));
  }
  el.replaceChildren(svg);
}

async function renderLessons() {
  const rows = await api("/api/memory");
  $("lessons").replaceChildren(...(rows.length ? rows.sort((a, b) => b.utility - a.utility || b.uses - a.uses).map((m) =>
    h("div", { class: "lesson" },
      h("div", { class: "u" }, fmt.num(m.utility, 2), h("span", { class: "bar" }, h("i", { style: { width: m.utility * 100 + "%" } })),
        h("small", {}, `${m.uses} recalls · ${m.wins}↑ ${m.losses}↓`)),
      h("p", {}, m.text))) : [h("div", { class: "kit-empty" }, "No lessons yet. The agent writes them after each generation and they persist across runs.")]));
}

function renderTrace() {
  traceList($("trace"), S.events, (ev) => {
    const p = ev.payload, row = (label, note, cls = "sub") => h("div", { class: `kit-trace-row ${cls}` },
      h("span", { class: "kit-trace-time" }, ""), h("span", { class: "kit-trace-node" }, label), h("span", { class: "kit-trace-note" }, note), h("span", {}));
    if (ev.type === "lesson") return row("lesson written", p.text);
    if (ev.type === "rejected") return row("rejected", `${p.name}: ${p.reason}`, "sub warn");
    if (ev.type === "holdout") return row("holdout check", `${p.name}: ${p.pass ? "PASS" : "FAIL"} · ${p.peeks_left} left`);
    return null;
  });
  const calls = S.events.filter((e) => e.type === "llm" && e.payload.source !== "baseline");
  $("trace-hint").textContent = calls.length ? `${calls.length} model calls · ${fmt.ms(calls.reduce((s, e) => s + e.payload.ms, 0))}` : "";
}

function render() {
  S.cands = candidates();
  if (!S.cands.find((c) => c.name === S.sel)) S.sel = [...S.cands].sort((a, b) => b.score.robust - a.score.robust)[0]?.name || null;
  renderStats(); renderBoard(); renderFrontier(); renderLineage(); renderTrace();
  flowStrip($("flow"), S.meta.graph, flowStatus(S.events));
  renderDetail();
  $("start").disabled = S.live; $("start").textContent = S.live ? "Researching…" : "Start research";
}
function select(name) { S.sel = name; renderBoard(); renderLineage(); renderDetail(); }

async function loadRuns() {
  S.runs = await api("/api/runs?kind=research");
  $("run-select").replaceChildren(...(S.runs.length ? S.runs.map((r) =>
    h("option", { value: r.id, selected: r.id === S.runId }, `#${r.id} · ${r.title}${r.status === "running" ? " · live" : ""}`)) : [h("option", { value: "" }, "no runs yet")]));
}

async function openRun(id) {
  S.stop && S.stop();
  S.runId = id; S.run = await api(`/api/runs/${id}`); S.events = S.run.events; S.live = S.run.status === "running";
  const inp = S.run.input;
  S.baseline = await api(`/api/baseline?asset=${inp.asset}&generations=${inp.generations}&per_gen=${inp.per_gen}`);
  render(); renderLessons();
  if (S.live) {
    S.events = [];
    let timer = null;
    S.stop = follow(id, (ev) => { S.events.push(ev); if (!timer) timer = setTimeout(() => { timer = null; render(); if (ev.type === "lesson") renderLessons(); }, 120); },
      async () => { S.live = false; await loadRuns(); await openRun(id); });
  }
}

async function main() {
  S.meta = await api("/api/meta");
  $("mode").textContent = S.meta.mode === "baseline" ? "no model configured: random search only" : `${S.meta.model} via ${S.meta.mode}`;
  $("asset").replaceChildren(...Object.entries(S.meta.assets).map(([a, v]) => h("option", { value: a }, `${v.symbol} (agent sees “Asset ${a}”)`)));
  await loadRuns();
  if (S.runs.length) await openRun(S.runs[0].id);
  else { S.cands = []; render(); renderLessons(); }
  $("run-select").addEventListener("change", (e) => e.target.value && openRun(+e.target.value));
  $("controls").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const body = { asset: $("asset").value, generations: +$("gens").value, per_gen: +$("per").value };
      if (!(await api(`/api/baseline?asset=${body.asset}&generations=${body.generations}&per_gen=${body.per_gen}`))) api("/api/baseline", body);
      const { run_id } = await api("/api/research", body);
      await loadRuns(); await openRun(run_id); await loadRuns();
    } catch (err) { alert(err.message); }
  });
}
main();
