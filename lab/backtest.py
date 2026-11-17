"""Bar-by-bar backtester with next-bar execution and costs.

A signal read at the close of bar i is filled at the open of bar i+1, so a rule
can never trade on the bar that produced its signal. Costs are charged on every
entry and exit. One position at a time, fully invested, no leverage.
"""
from __future__ import annotations

import numpy as np

from .data import BARS_PER_YEAR
from .dsl import Indicators

FEE_BPS, SLIP_BPS = 5.0, 2.0
ATR_N = 14


def simulate(spec: dict, bars: dict, ind: Indicators, lo: int, hi: int) -> dict:
    o, h, l, c = bars["o"], bars["h"], bars["l"], bars["c"]
    entry = np.logical_and.reduce([ind.mask(x) for x in spec["entry"]])
    exit_ = np.logical_or.reduce([ind.mask(x) for x in spec["exit"]]) if spec["exit"] else np.zeros(len(c), bool)
    atr = ind.atr(ATR_N)
    side = 1.0 if spec["side"] == "long" else -1.0
    cost = (FEE_BPS + SLIP_BPS) / 1e4

    eq, pos, pending = 1.0, 0, None
    stop_px, held, entry_eq, entry_i = None, 0, 1.0, 0
    equity = np.empty(hi - lo)
    trades = []

    def close_trade(i):
        nonlocal eq, pos, stop_px, held
        eq *= 1 - cost
        trades.append({"entry": entry_i, "exit": i, "ret": eq / entry_eq - 1, "bars": held})
        pos, stop_px, held = 0, None, 0

    for i in range(lo, hi):
        ref = c[i - 1] if i > 0 else o[i]
        if pending == "exit" and pos:
            eq *= 1 + side * (o[i] / ref - 1)
            close_trade(i)
        elif pending == "enter" and not pos:
            pos, entry_eq, entry_i, held = 1, eq, i, 0
            eq *= 1 - cost
            ref = o[i]
            if spec["stop_atr"] and not np.isnan(atr[i - 1]):
                stop_px = o[i] - side * spec["stop_atr"] * atr[i - 1]
        pending = None

        if pos:
            held += 1
            hit = stop_px is not None and ((side > 0 and l[i] <= stop_px) or (side < 0 and h[i] >= stop_px))
            if hit:
                fill = min(o[i], stop_px) if side > 0 else max(o[i], stop_px)  # gaps fill at the open
                eq *= 1 + side * (fill / ref - 1)
                close_trade(i)
            else:
                eq *= 1 + side * (c[i] / ref - 1)
                if exit_[i] or held >= spec["max_hold"]:
                    pending = "exit"
        elif entry[i]:
            pending = "enter"
        equity[i - lo] = eq

    if pos:
        close_trade(hi - 1)
        equity[-1] = eq
    return {"equity": equity, "trades": trades}


def metrics(equity: np.ndarray, trades: list[dict]) -> dict:
    rets = np.diff(equity, prepend=1.0) / np.concatenate(([1.0], equity[:-1]))
    sd = rets.std()
    years = len(equity) / BARS_PER_YEAR
    wins = [t["ret"] for t in trades if t["ret"] > 0]
    losses = [t["ret"] for t in trades if t["ret"] <= 0]
    total = float(equity[-1] - 1)
    return {
        "sharpe": round(float(rets.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0, 3),
        "total_return": round(total, 4),
        "cagr": round(float(equity[-1] ** (1 / years) - 1) if equity[-1] > 0 and years > 0 else -1.0, 4),
        "max_dd": round(float((equity / np.maximum.accumulate(equity) - 1).min()), 4),
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3) if trades else 0.0,
        "profit_factor": round(float(sum(wins) / abs(sum(losses))), 2) if losses and sum(losses) != 0 else None,
        "exposure": round(float(sum(t["bars"] for t in trades) / len(equity)), 3),
        "avg_hold": round(float(np.mean([t["bars"] for t in trades])), 1) if trades else 0.0,
    }


def evaluate(spec: dict, bars: dict, ind: Indicators, lo: int, hi: int) -> tuple[dict, np.ndarray]:
    run = simulate(spec, bars, ind, lo, hi)
    return metrics(run["equity"], run["trades"]), run["equity"]


def buy_and_hold(bars: dict, lo: int, hi: int) -> np.ndarray:
    c = bars["c"][lo:hi]
    return c / bars["o"][lo]
