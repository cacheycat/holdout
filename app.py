"""Holdout: a blind, self-improving strategy research agent. Run with `python app.py`."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field

from agentkit import LLM, Memory, Runner, Store
from agentkit.serve import make_app
from lab import agent, critic, data
from lab.backtest import buy_and_hold, evaluate

ROOT = Path(__file__).parent
store = Store(ROOT / "data" / "holdout.db")
memory = Memory(store)
runner = Runner(store)
app = make_app("Holdout", ROOT / "web", store, runner)


class ResearchIn(BaseModel):
    asset: str = Field("A", pattern="^[ABC]$")
    generations: int = Field(5, ge=1, le=8)
    per_gen: int = Field(3, ge=2, le=4)
    baseline: bool = False


def _date(ms: float) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")


@app.get("/api/meta")
def meta():
    llm = LLM()
    assets = {}
    for alias, symbol in data.ASSETS.items():
        ctx = agent.context(alias)
        t, sp = ctx["bars"]["t"], ctx["splits"]
        assets[alias] = {"symbol": symbol, "bars": len(t),
                         "segments": {k: [_date(t[sp[k][0]]), _date(t[sp[k][1] - 1])] for k in ("train", "val", "holdout")}}
    return {"mode": llm.mode, "model": llm.model, "graph": agent.describe_graph(), "assets": assets,
            "peek_budget": agent.PEEK_BUDGET, "busy": runner.busy("research"), "knowledge_passages": len(agent.CARDS.docs),
            "gates": {"min_val_sharpe": critic.MIN_VAL_SHARPE, "min_dsr": critic.MIN_DSR}}


@app.post("/api/research")
def start_research(body: ResearchIn):
    if runner.busy("research"):
        raise HTTPException(409, "a research run is already in progress")
    llm = LLM(provider=None) if body.baseline else LLM()

    def job(run_id, emit):
        return agent.research(emit, memory=memory, llm=llm, alias=body.asset, generations=body.generations,
                              per_gen=body.per_gen, seed=run_id, use_memory=not body.baseline)

    title = f"{data.ASSETS[body.asset]} · {body.generations}×{body.per_gen} · {llm.mode}"
    return {"run_id": runner.start("research", title, body.model_dump(), job)}


@app.get("/api/equity")
def equity(run_id: int, name: str):
    """Equity curve for one candidate. Holdout bars are served only if that
    candidate was promoted; otherwise the sealed segment stays sealed."""
    run = store.get_run(run_id)
    cand = next((c for c in (run or {}).get("output", {}).get("candidates", []) if c["name"] == name), None) if run and run.get("output") else None
    if not cand:
        raise HTTPException(404, "candidate not found")
    ctx = agent.context(run["output"]["asset"])
    sp, bars = ctx["splits"], ctx["bars"]
    hi = sp["holdout"][1] if cand.get("holdout") else sp["dev"][1]
    _, eq = evaluate(cand["spec"], bars, ctx["ind"], 0, hi)
    bench = buy_and_hold(bars, 0, hi)
    step = max(1, hi // 700)
    idx = np.arange(0, hi, step)
    return {"t": (bars["t"][idx] / 1000).tolist(), "equity": eq[idx].round(4).tolist(), "bench": bench[idx].round(4).tolist(),
            "train_end": bars["t"][sp["train"][1]] / 1000, "val_end": bars["t"][sp["val"][1] - 1] / 1000,
            "end": bars["t"][-1] / 1000, "holdout_visible": bool(cand.get("holdout"))}


@app.get("/api/memory")
def lessons():
    return memory.all("lesson")


@app.post("/api/baseline")
def run_baseline(body: ResearchIn):
    out = agent.random_baseline(body.asset, body.generations, body.per_gen, seeds=8)
    store.save_eval(f"random-{body.asset}-{body.generations}x{body.per_gen}", {k: v for k, v in out.items() if k != "runs"}, out["runs"])
    return out


@app.get("/api/baseline")
def get_baseline(asset: str = "A", generations: int = 5, per_gen: int = 3):
    row = store.latest_eval(f"random-{asset}-{generations}x{per_gen}")
    return row["summary"] if row else None


app.finish()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8811, log_level="warning")
