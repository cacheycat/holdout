"""Market data: 4-hour klines from Binance's public data API, cached on disk.

The cache is the dataset. Once a symbol is downloaded it is never refreshed
implicitly, so every backtest in a session (and every re-run of a spec) sees
identical bars and the train / validation / holdout boundaries do not move.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from agentkit.http import fetch_json

API = "https://data-api.binance.vision/api/v3/klines"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
INTERVAL = "4h"
BARS_PER_YEAR = 6 * 365
TARGET_BARS = 7000  # a little over three years of 4h bars

# The agent only ever sees the alias. The symbol is for the human reading the UI.
ASSETS = {"A": "BTCUSDT", "B": "ETHUSDT", "C": "SOLUSDT"}
SPLIT = (0.60, 0.80)  # train ends at 60%, validation at 80%, the rest is sealed


def _fetch(symbol: str, end_ms: int | None, limit: int = 1000) -> list:
    url = f"{API}?symbol={symbol}&interval={INTERVAL}&limit={limit}"
    if end_ms:
        url += f"&endTime={end_ms}"
    return fetch_json(url)


def download(symbol: str, bars: int = TARGET_BARS) -> list:
    rows, end = [], None
    while len(rows) < bars:
        batch = _fetch(symbol, end)
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    rows = rows[-bars:]
    return rows[:-1]  # drop the still-forming bar


def load(alias: str) -> dict:
    """Return bars for an alias as numpy arrays: t (ms), o, h, l, c, v."""
    symbol = ASSETS[alias]
    path = CACHE / f"{symbol}_{INTERVAL}.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(download(symbol)))
    rows = json.loads(path.read_text())
    cols = np.array([[float(x) for x in r[:6]] for r in rows])
    return {"alias": alias, "symbol": symbol, "t": cols[:, 0], "o": cols[:, 1], "h": cols[:, 2],
            "l": cols[:, 3], "c": cols[:, 4], "v": cols[:, 5]}


def splits(n: int) -> dict:
    a, b = int(n * SPLIT[0]), int(n * SPLIT[1])
    return {"train": (0, a), "val": (a, b), "dev": (0, b), "holdout": (b, n)}


def regime(bars: dict, lo: int, hi: int) -> dict:
    """Coarse, identity-free description of a segment for the blind prompt."""
    c = bars["c"][lo:hi]
    r = np.diff(np.log(c))
    sma = np.convolve(c, np.ones(200) / 200, mode="valid")
    above = float(np.mean(c[199:] > sma))
    ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    return {"bars": int(hi - lo),
            "annualised_vol": round(float(r.std() * np.sqrt(BARS_PER_YEAR)), 2),
            "share_of_bars_above_200_bar_average": round(above, 2),
            "lag1_return_autocorrelation": round(ac, 3),
            "worst_drawdown_buy_and_hold": round(float((c / np.maximum.accumulate(c) - 1).min()), 2)}
