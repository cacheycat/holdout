"""The strategy language: a closed JSON vocabulary, validated and clamped.

The model proposes data in this vocabulary and nothing else. It cannot emit
code, so a proposal can be wrong but it cannot execute anything, and every
candidate is comparable, hashable and mutable by the random-search baseline.
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

INDICATORS = {
    "rsi":      {"n": (5, 30),   "m": None,      "val": (5, 95),     "doc": "Wilder RSI over n bars, 0 to 100"},
    "sma_gap":  {"n": (10, 300), "m": None,      "val": (-25, 25),   "doc": "close versus its n-bar simple average, in percent"},
    "ema_gap":  {"n": (5, 60),   "m": (20, 250), "val": (-15, 15),   "doc": "n-bar EMA versus m-bar EMA, in percent (n < m)"},
    "roc":      {"n": (3, 120),  "m": None,      "val": (-40, 40),   "doc": "percent change of close over n bars"},
    "bb_z":     {"n": (10, 100), "m": None,      "val": (-3.5, 3.5), "doc": "z-score of close against its n-bar mean and deviation"},
    "donchian": {"n": (10, 200), "m": None,      "val": (0, 1),      "doc": "position of close inside the n-bar high-low range, 0 to 1"},
    "atr_pct":  {"n": (7, 50),   "m": None,      "val": (0.2, 12),   "doc": "n-bar average true range as percent of close"},
    "vol_z":    {"n": (10, 100), "m": None,      "val": (-3, 5),     "doc": "z-score of volume over n bars"},
}
MAX_ENTRY, MAX_EXIT = 3, 2
HOLD_RANGE, STOP_RANGE = (6, 240), (1.0, 6.0)


class SpecError(ValueError):
    pass


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _cond(raw: dict) -> dict:
    ind = raw.get("ind")
    if ind not in INDICATORS:
        raise SpecError(f"unknown indicator '{ind}'")
    b = INDICATORS[ind]
    if raw.get("op") not in ("<", ">"):
        raise SpecError("op must be '<' or '>'")
    n = int(_clamp(int(raw.get("n") or b["n"][0]), *b["n"]))
    m = None
    if b["m"]:
        m = int(_clamp(int(raw.get("m") or b["m"][1]), *b["m"]))
        m = max(m, n + 5)
    return {"ind": ind, "n": n, "m": m, "op": raw["op"], "val": round(float(_clamp(float(raw["val"]), *b["val"])), 3)}


def normalize(raw: dict) -> dict:
    """Validate a proposal and return the canonical spec, or raise SpecError."""
    if raw.get("side") not in ("long", "short"):
        raise SpecError("side must be 'long' or 'short'")
    entry = [_cond(c) for c in (raw.get("entry") or [])][:MAX_ENTRY]
    exit_ = [_cond(c) for c in (raw.get("exit") or [])][:MAX_EXIT]
    if not entry:
        raise SpecError("at least one entry condition is required")
    stop = raw.get("stop_atr")
    stop = None if stop in (None, 0) else round(float(_clamp(float(stop), *STOP_RANGE)), 2)
    hold = int(_clamp(int(raw.get("max_hold") or 60), *HOLD_RANGE))
    return {"side": raw["side"], "entry": entry, "exit": exit_, "stop_atr": stop, "max_hold": hold}


def key(spec: dict) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def cond_text(c: dict) -> str:
    args = f"{c['n']},{c['m']}" if c["m"] else f"{c['n']}"
    return f"{c['ind']}({args}) {c['op']} {c['val']:g}"


def describe(spec: dict) -> str:
    parts = [f"{spec['side'].upper()} when " + " and ".join(cond_text(c) for c in spec["entry"])]
    if spec["exit"]:
        parts.append("exit when " + " or ".join(cond_text(c) for c in spec["exit"]))
    parts.append(f"stop {spec['stop_atr']:g} ATR" if spec["stop_atr"] else "no stop")
    parts.append(f"max hold {spec['max_hold']} bars")
    return "; ".join(parts)


def vocabulary() -> str:
    lines = []
    for name, b in INDICATORS.items():
        m = f", m in [{b['m'][0]}, {b['m'][1]}]" if b["m"] else ", m = null"
        lines.append(f"- {name}: {b['doc']}. n in [{b['n'][0]}, {b['n'][1]}]{m}, val in [{b['val'][0]}, {b['val'][1]}]")
    return "\n".join(lines)


# --- random generation and perturbation (baseline search + stability test) ---

def random_cond(rng: random.Random) -> dict:
    ind = rng.choice(list(INDICATORS))
    b = INDICATORS[ind]
    n = rng.randint(*b["n"])
    m = rng.randint(max(b["m"][0], n + 5), b["m"][1]) if b["m"] else None
    return _cond({"ind": ind, "n": n, "m": m, "op": rng.choice("<>"), "val": rng.uniform(*b["val"])})


def random_spec(rng: random.Random) -> dict:
    return normalize({"side": rng.choice(["long", "long", "short"]),
                      "entry": [random_cond(rng) for _ in range(rng.randint(1, 3))],
                      "exit": [random_cond(rng) for _ in range(rng.randint(0, 2))],
                      "stop_atr": rng.choice([None, rng.uniform(*STOP_RANGE)]),
                      "max_hold": rng.randint(*HOLD_RANGE)})


def perturb(spec: dict, rng: random.Random, scale: float = 0.15) -> dict:
    """Jitter every numeric parameter by up to +-scale. Used by the stability gate."""
    def jig(c):
        b = INDICATORS[c["ind"]]
        span = b["val"][1] - b["val"][0]
        return {**c, "n": round(c["n"] * (1 + rng.uniform(-scale, scale))),
                "m": round(c["m"] * (1 + rng.uniform(-scale, scale))) if c["m"] else None,
                "val": c["val"] + rng.uniform(-scale, scale) * span * 0.25}
    return normalize({**spec, "entry": [jig(c) for c in spec["entry"]], "exit": [jig(c) for c in spec["exit"]],
                      "stop_atr": spec["stop_atr"] * (1 + rng.uniform(-scale, scale)) if spec["stop_atr"] else None,
                      "max_hold": round(spec["max_hold"] * (1 + rng.uniform(-scale, scale)))})


def mutate(spec: dict, rng: random.Random) -> dict:
    """One structural or parametric change. Used by the random-search baseline."""
    s = json.loads(json.dumps(spec))
    move = rng.choice(["param", "param", "swap_entry", "exit", "stop", "hold"])
    if move == "param":
        return perturb(s, rng, 0.35)
    if move == "swap_entry":
        s["entry"][rng.randrange(len(s["entry"]))] = random_cond(rng)
    elif move == "exit":
        s["exit"] = [random_cond(rng)] + s["exit"][:1]
    elif move == "stop":
        s["stop_atr"] = rng.choice([None, rng.uniform(*STOP_RANGE)])
    else:
        s["max_hold"] = rng.randint(*HOLD_RANGE)
    return normalize(s)


# --- indicator maths: every value at bar i uses bars <= i only ---

class Indicators:
    def __init__(self, bars: dict):
        self.b = bars
        self._memo: dict = {}

    def get(self, ind: str, n: int, m: int | None = None) -> np.ndarray:
        k = (ind, n, m)
        if k not in self._memo:
            self._memo[k] = getattr(self, "_" + ind)(n, m)
        return self._memo[k]

    def mask(self, c: dict) -> np.ndarray:
        x = self.get(c["ind"], c["n"], c["m"])
        with np.errstate(invalid="ignore"):
            return (x < c["val"]) if c["op"] == "<" else (x > c["val"])

    @staticmethod
    def sma(x, n):
        out = np.full(len(x), np.nan)
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
        return out

    @staticmethod
    def std(x, n):
        mean = Indicators.sma(x, n)
        var = Indicators.sma(x * x, n) - mean ** 2
        return np.sqrt(np.maximum(var, 0))

    @staticmethod
    def ema(x, n):
        out = np.empty(len(x))
        out[0] = x[0]
        a = 2 / (n + 1)
        for i in range(1, len(x)):
            out[i] = a * x[i] + (1 - a) * out[i - 1]
        return out

    @staticmethod
    def wilder(x, n):
        out = np.full(len(x), np.nan)
        if len(x) <= n:
            return out
        out[n] = x[1:n + 1].mean()
        for i in range(n + 1, len(x)):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
        return out

    def atr(self, n):
        k = ("_atr_abs", n)
        if k not in self._memo:
            h, l, c = self.b["h"], self.b["l"], self.b["c"]
            pc = np.roll(c, 1)
            tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
            tr[0] = h[0] - l[0]
            self._memo[k] = self.wilder(tr, n)
        return self._memo[k]

    def _rsi(self, n, _):
        d = np.diff(self.b["c"], prepend=self.b["c"][0])
        up, dn = self.wilder(np.maximum(d, 0), n), self.wilder(np.maximum(-d, 0), n)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100 - 100 / (1 + up / dn)

    def _sma_gap(self, n, _):
        return (self.b["c"] / self.sma(self.b["c"], n) - 1) * 100

    def _ema_gap(self, n, m):
        out = (self.ema(self.b["c"], n) / self.ema(self.b["c"], m) - 1) * 100
        out[:m] = np.nan
        return out

    def _roc(self, n, _):
        c = self.b["c"]
        out = np.full(len(c), np.nan)
        out[n:] = (c[n:] / c[:-n] - 1) * 100
        return out

    def _bb_z(self, n, _):
        c = self.b["c"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (c - self.sma(c, n)) / self.std(c, n)

    def _donchian(self, n, _):
        h, l, c = self.b["h"], self.b["l"], self.b["c"]
        out = np.full(len(c), np.nan)
        hi = np.lib.stride_tricks.sliding_window_view(h, n).max(axis=1)
        lo = np.lib.stride_tricks.sliding_window_view(l, n).min(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = (c[n - 1:] - lo) / (hi - lo)
        return out

    def _atr_pct(self, n, _):
        return self.atr(n) / self.b["c"] * 100

    def _vol_z(self, n, _):
        v = self.b["v"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (v - self.sma(v, n)) / self.std(v, n)
