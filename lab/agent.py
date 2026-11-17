"""The research loop: recall, hypothesise, backtest, critique, reflect, promote.

The model is kept blind on purpose. It sees an alias instead of a ticker, no
dates and no prices, so it cannot lean on what it remembers about a market's
history, and it never sees holdout numbers: a promotion comes back as PASS or
FAIL only. Everything it learns is written to memory as short lessons whose
usefulness is scored by what happened after they were recalled.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from agentkit import END, LLM, Index, Memory, StateGraph, load_cards

from . import critic, data, dsl

PEEK_BUDGET = 3
CARDS = Index(load_cards(Path(__file__).resolve().parent.parent / "knowledge"))

_COND = {"type": "object", "additionalProperties": False, "required": ["ind", "n", "m", "op", "val"],
         "properties": {"ind": {"type": "string", "enum": list(dsl.INDICATORS)}, "n": {"type": "integer"},
                        "m": {"type": ["integer", "null"]}, "op": {"type": "string", "enum": ["<", ">"]},
                        "val": {"type": "number"}}}
PROPOSAL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["candidates"], "properties": {
    "candidates": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["name", "thesis", "parent", "side", "entry", "exit", "stop_atr", "max_hold"],
        "properties": {"name": {"type": "string"}, "thesis": {"type": "string"},
                       "parent": {"type": ["string", "null"]}, "side": {"type": "string", "enum": ["long", "short"]},
                       "entry": {"type": "array", "items": _COND}, "exit": {"type": "array", "items": _COND},
                       "stop_atr": {"type": ["number", "null"]}, "max_hold": {"type": "integer"}}}}}}
REFLECT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["lessons", "next_focus"], "properties": {
    "lessons": {"type": "array", "items": {"type": "string"}}, "next_focus": {"type": "string"}}}

SYSTEM = f"""You are a quantitative researcher designing rule-based strategies for one liquid, volatile asset on 4-hour bars.

You work blind: the asset is an alias, and you are given no dates or prices. A sealed holdout segment exists that you never see. An automated harness you cannot change backtests every proposal with next-bar execution and trading costs, then applies seven gates: enough trades, validation Sharpe, train-to-validation decay, parameter stability under jitter, working on at least one other asset, drawdown, and a deflated Sharpe that gets stricter with every strategy tried.

Strategies are data, not code. Indicators available:
{dsl.vocabulary()}

A strategy has a side (long or short), 1 to {dsl.MAX_ENTRY} entry conditions that must all hold, 0 to {dsl.MAX_EXIT} exit conditions (any one exits), an optional stop in ATR multiples ({dsl.STOP_RANGE[0]} to {dsl.STOP_RANGE[1]}, or null), and max_hold in bars ({dsl.HOLD_RANGE[0]} to {dsl.HOLD_RANGE[1]}). One position at a time.

The score to raise is the robust score: the lower of train and validation Sharpe, scaled down for fragile parameters, single-asset behaviour and thin trade counts. Chasing train Sharpe is the failure mode. Prefer simple rules with an economic reason to exist."""


@lru_cache(maxsize=4)
def context(alias: str) -> dict:
    def pack(a):
        bars = data.load(a)
        return {"bars": bars, "ind": dsl.Indicators(bars), "splits": data.splits(len(bars["c"]))}
    ctx = pack(alias)
    ctx["others"] = [pack(a) for a in data.ASSETS if a != alias]
    ctx["regime"] = data.regime(ctx["bars"], *ctx["splits"]["train"])
    return ctx


def _row(c: dict) -> str:
    s = c["score"]
    verdict = "passes all gates" if s["passed"] else "fails " + ", ".join(s["failed"])
    hold = f" | holdout {'PASS' if c['holdout']['pass'] else 'FAIL'}" if c.get("holdout") else ""
    return (f"- {c['name']} (gen {c['gen']}, parent {c['parent'] or 'none'}): {dsl.describe(c['spec'])} | "
            f"train Sharpe {s['train']['sharpe']:.2f}, validation {s['val']['sharpe']:.2f}, trades {s['train']['trades']}+{s['val']['trades']}, "
            f"stability {s['stability']:.2f}, breadth {s['breadth']:.1f}, deflated {s['dsr']:.2f}, robust {s['robust']:.2f} | {verdict}{hold}")


def research(emit, *, memory: Memory, llm: LLM, alias: str = "A", generations: int = 5, per_gen: int = 3,
             seed: int = 0, use_memory: bool = True) -> dict:
    ctx = context(alias)
    rng = random.Random(seed)
    st: dict = {"gen": 0, "cands": [], "keys": set(), "peeks": PEEK_BUDGET, "frontier": [], "focus": None,
                "recalled": [], "notes": [], "batch": [], "lessons_written": []}
    log_call = lambda rec: emit("llm", rec)
    best = lambda: max((c["score"]["robust"] for c in st["cands"]), default=0.0)

    def recall(_):
        st["gen"] += 1
        query = st["focus"] or "robust simple rules for a volatile trending asset, avoiding overfitting and thin trade counts"
        st["recalled"] = memory.search(query, k=5, kind="lesson") if use_memory else []
        st["notes"] = CARDS.search(query, k=3)
        emit("recall", {"gen": st["gen"], "query": query,
                        "lessons": [{"id": m["id"], "text": m["text"], "utility": m["utility"]} for m in st["recalled"]],
                        "notes": [n["title"] for n in st["notes"]]})
        return {"_note": f"generation {st['gen']}: {len(st['recalled'])} lessons, {len(st['notes'])} notes"}

    def baseline_proposals():
        out, parents = [], sorted(st["cands"], key=lambda c: -c["score"]["robust"])[:3]
        for i in range(per_gen):
            if parents and i % 2 == 0:
                p = rng.choice(parents)
                out.append({**dsl.mutate(p["spec"], rng), "name": f"mut-{st['gen']}{chr(97 + i)}", "thesis": "random mutation of a leading candidate", "parent": p["name"]})
            else:
                out.append({**dsl.random_spec(rng), "name": f"rnd-{st['gen']}{chr(97 + i)}", "thesis": "random draw from the vocabulary", "parent": None})
        return {"candidates": out}

    def hypothesise(_):
        board = sorted(st["cands"], key=lambda c: -c["score"]["robust"])
        shown = board[:6] + [c for c in st["cands"][-per_gen:] if c not in board[:6]]
        user = "\n\n".join(filter(None, [
            f"Generation {st['gen']} of {generations}. Propose exactly {per_gen} strategies. Trials so far: {len(st['cands'])}. Holdout checks left: {st['peeks']}.",
            "Training-segment character of the asset:\n" + json.dumps(ctx["regime"]),
            ("Results so far (best first, then the latest batch):\n" + "\n".join(_row(c) for c in shown)) if shown else "No strategies tested yet.",
            ("Lessons recalled from earlier research (utility is the share of past uses that preceded an improvement):\n"
             + "\n".join(f"- [{m['utility']:.2f}] {m['text']}" for m in st["recalled"])) if st["recalled"] else None,
            "Reference notes:\n" + "\n".join(f"- {n['title']}: {n['text'][:420]}" for n in st["notes"]) if st["notes"] else None,
            "For each candidate give a short unique kebab-case name, a one-sentence thesis, and `parent` (the name of the strategy it modifies, or null for a new idea). "
            "Mix refinement of what is working with at least one structurally different idea.",
        ]))
        out = llm.json("hypothesise", SYSTEM, user, PROPOSAL_SCHEMA, baseline_proposals, on_call=log_call)
        names = {c["name"] for c in st["cands"]}
        batch = []
        for raw in out.get("candidates", [])[:per_gen + 1]:
            try:
                spec = dsl.normalize(raw)
            except (dsl.SpecError, TypeError, ValueError, KeyError) as e:
                emit("rejected", {"name": str(raw.get("name"))[:60], "reason": str(e)})
                continue
            k = dsl.key(spec)
            if k in st["keys"]:
                emit("rejected", {"name": str(raw.get("name"))[:60], "reason": "duplicate of an earlier strategy"})
                continue
            st["keys"].add(k)
            name = str(raw.get("name") or f"s-{k[:5]}")[:40]
            while name in names:
                name += "-b"
            names.add(name)
            parent = raw.get("parent") if raw.get("parent") in names - {name} else None
            batch.append({"key": k, "name": name, "thesis": str(raw.get("thesis") or "")[:300], "parent": parent,
                          "gen": st["gen"], "spec": spec, "rule": dsl.describe(spec)})
        st["batch"] = batch
        return {"_note": f"{len(batch)} candidates: " + ", ".join(b["name"] for b in batch)}

    def backtest(_):
        srs = [c["score"]["dev_sr_bar"] for c in st["cands"]]
        for c in st["batch"]:
            n = len(st["cands"]) + 1
            var = float(np.var(srs)) if len(srs) > 1 else 1e-4
            c["score"] = critic.assess(c["spec"], ctx, n, var)
            srs.append(c["score"]["dev_sr_bar"])
            st["cands"].append(c)
        return {"_note": f"{len(st['batch'])} strategies on train and validation, {critic.PERTURBATIONS} jittered variants and 2 other assets each"}

    def critique(_):
        prev = st["frontier"][-1] if st["frontier"] else 0.0
        for c in st["batch"]:
            s = c["score"]
            emit("candidate", {k: c[k] for k in ("gen", "name", "parent", "rule", "thesis", "score")})
        st["frontier"].append(round(best(), 3))
        improved = st["frontier"][-1] > prev + 1e-9
        if use_memory and st["recalled"]:
            memory.feedback([m["id"] for m in st["recalled"]], improved)
        passed = sum(c["score"]["passed"] for c in st["batch"])
        return {"improved": improved, "_note": f"{passed} of {len(st['batch'])} pass all gates; best robust score {st['frontier'][-1]:.2f}"
                + (" (improved)" if improved else "")}

    def baseline_lessons():
        fails = [f for c in st["batch"] for f in c["score"]["failed"]]
        if not fails:
            return {"lessons": [], "next_focus": "refine the passing candidates with small parameter changes"}
        top = max(set(fails), key=fails.count)
        return {"lessons": [f"The most common rejection in a batch was the {top} gate; proposals should be designed against it first."],
                "next_focus": f"strategies that clear the {top} gate"}

    def reflect(_):
        user = "\n\n".join(filter(None, [
            f"Generation {st['gen']} results:\n" + "\n".join(_row(c) for c in st["batch"]),
            f"Best robust score before this generation: {(st['frontier'][-2] if len(st['frontier']) > 1 else 0):.2f}. After: {st['frontier'][-1]:.2f}.",
            ("Gate detail for this batch:\n" + "\n".join(
                f"- {c['name']}: " + "; ".join(f"{g['label']} {'ok' if g['pass'] else 'FAIL'} ({g['detail']})" for g in c["score"]["gates"])
                for c in st["batch"])),
            "Lessons already recalled this generation:\n" + "\n".join(f"- {m['text']}" for m in st["recalled"]) if st["recalled"] else None,
            "Write at most two lessons that would change what a researcher proposes next time on any asset. Each must be one sentence, "
            "specific and falsifiable (name the rule family, parameter range or gate), and must not repeat a recalled lesson. "
            "Return an empty list if nothing new was learned. Then state `next_focus`: what the next generation should search for, in one sentence.",
        ]))
        out = llm.json("reflect", SYSTEM, user, REFLECT_SCHEMA, baseline_lessons, on_call=log_call)
        written = []
        if use_memory:
            for text in out.get("lessons", [])[:2]:
                if isinstance(text, str) and 20 < len(text) < 400:
                    mid = memory.add("lesson", text, {"gen": st["gen"], "asset": alias, "source": llm.mode})
                    written.append({"id": mid, "text": text})
                    emit("lesson", {"id": mid, "text": text, "gen": st["gen"]})
        st["lessons_written"] += written
        st["focus"] = str(out.get("next_focus") or "")[:300] or None
        return {"_note": (f"{len(written)} lessons written. " if written else "") + (f"Next: {st['focus']}" if st["focus"] else "")}

    def promote(_):
        ready = [c for c in st["cands"] if c["score"]["passed"] and "holdout" not in c]
        if not ready or st["peeks"] <= 0:
            why = "no candidate has cleared every gate" if not ready else "holdout budget is spent"
            return {"_note": f"no promotion: {why}"}
        c = max(ready, key=lambda x: x["score"]["robust"])
        srs = [x["score"]["dev_sr_bar"] for x in st["cands"]]
        again = critic.assess(c["spec"], ctx, len(st["cands"]), float(np.var(srs)) if len(srs) > 1 else 1e-4)
        if not again["passed"]:  # the bar rose as more strategies were tried
            c["score"] = again
            return {"_note": f"{c['name']} no longer clears the deflated Sharpe gate at {len(st['cands'])} trials"}
        st["peeks"] -= 1
        c["holdout"] = critic.holdout_check(c["spec"], ctx)
        if use_memory:
            ids = [m["id"] for m in st["recalled"]]
            memory.feedback(ids, c["holdout"]["pass"])
        emit("holdout", {"name": c["name"], "pass": c["holdout"]["pass"], "peeks_left": st["peeks"]})
        return {"_note": f"{c['name']} checked on the sealed holdout: {'PASS' if c['holdout']['pass'] else 'FAIL'} ({st['peeks']} checks left)"}

    g = StateGraph("holdout-research")
    g.node("recall", recall, "Recall", "memory").node("hypothesise", hypothesise, "Hypothesise", "llm")
    g.node("backtest", backtest, "Backtest", "tool").node("critique", critique, "Critique", "gate")
    g.node("reflect", reflect, "Reflect", "llm").node("promote", promote, "Promote", "gate")
    g.edge("recall", "hypothesise").edge("hypothesise", "backtest").edge("backtest", "critique")
    g.edge("critique", "reflect").edge("reflect", "promote")
    g.route("promote", lambda s: "next generation" if st["gen"] < generations else "done",
            {"next generation": "recall", "done": END})
    emit("graph", g.describe())
    g.run({}, emit, max_steps=generations * 6 + 2)

    passes = [c for c in st["cands"] if c.get("holdout", {}).get("pass")]
    return {"asset": alias, "symbol": data.ASSETS[alias], "mode": llm.mode, "model": llm.model,
            "generations": generations, "per_gen": per_gen, "trials": len(st["cands"]),
            "frontier": st["frontier"], "peeks_used": PEEK_BUDGET - st["peeks"], "peek_budget": PEEK_BUDGET,
            "holdout_passes": len(passes), "gate_passes": sum(c["score"]["passed"] for c in st["cands"]),
            "lessons_written": st["lessons_written"], "candidates": st["cands"],
            "llm_calls": sum(1 for c in llm.calls if c["source"] != "baseline"), "graph": g.describe()}


def describe_graph() -> dict:
    g = StateGraph("holdout-research")
    for name, label, kind in [("recall", "Recall", "memory"), ("hypothesise", "Hypothesise", "llm"), ("backtest", "Backtest", "tool"),
                              ("critique", "Critique", "gate"), ("reflect", "Reflect", "llm"), ("promote", "Promote", "gate")]:
        g.node(name, lambda s: None, label, kind)
    g.edge("recall", "hypothesise").edge("hypothesise", "backtest").edge("backtest", "critique")
    g.edge("critique", "reflect").edge("reflect", "promote")
    g.route("promote", lambda s: "", {"next generation": "recall", "done": END})
    return g.describe()


def random_baseline(alias: str, generations: int, per_gen: int, seeds: int = 8) -> dict:
    """Random search over the same vocabulary, same gates and same trial budget."""
    from agentkit import Store
    import tempfile, os
    runs = []
    for seed in range(seeds):
        scratch = Memory(Store(os.path.join(tempfile.mkdtemp(), "m.db")))
        out = research(lambda *_: None, memory=scratch, llm=LLM(provider=None), alias=alias,
                       generations=generations, per_gen=per_gen, seed=1000 + seed, use_memory=False)
        runs.append({"seed": seed, "frontier": out["frontier"], "gate_passes": out["gate_passes"],
                     "holdout_passes": out["holdout_passes"], "peeks_used": out["peeks_used"]})
    fr = np.array([r["frontier"] for r in runs])
    return {"asset": alias, "generations": generations, "per_gen": per_gen, "seeds": seeds,
            "median": [round(float(x), 3) for x in np.median(fr, axis=0)],
            "p25": [round(float(x), 3) for x in np.percentile(fr, 25, axis=0)],
            "p75": [round(float(x), 3) for x in np.percentile(fr, 75, axis=0)],
            "max": [round(float(x), 3) for x in fr.max(axis=0)],
            "any_gate_pass": sum(r["gate_passes"] > 0 for r in runs),
            "any_holdout_pass": sum(r["holdout_passes"] > 0 for r in runs), "runs": runs}
