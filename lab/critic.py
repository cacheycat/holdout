"""The evaluation harness: seven overfitting gates and one robust score.

Nothing here calls a model, and the agent cannot edit it. A candidate is only
ever judged on train and validation bars; the sealed holdout is touched by
`holdout_check`, which the research loop may call a fixed number of times.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist

import numpy as np

from . import dsl
from .backtest import evaluate

_N = NormalDist()
MIN_TRADES_TRAIN, MIN_TRADES_VAL = 30, 8
MIN_VAL_SHARPE, MAX_DECAY, MIN_STABILITY, MAX_DD, MIN_DSR = 0.5, 0.4, 0.6, 0.35, 0.80
PERTURBATIONS = 8
HOLDOUT_MIN_SHARPE, HOLDOUT_MAX_DD = 0.5, 0.35


def per_bar_sharpe(equity: np.ndarray) -> float:
    r = np.diff(equity, prepend=1.0) / np.concatenate(([1.0], equity[:-1]))
    return float(r.mean() / r.std()) if r.std() > 0 else 0.0


def deflated_sharpe(equity: np.ndarray, n_trials: int, trial_sr_var: float) -> float:
    """Probability the true Sharpe is above zero after correcting for the number
    of strategies tried (Bailey and Lopez de Prado). More trials raise the bar."""
    r = np.diff(equity, prepend=1.0) / np.concatenate(([1.0], equity[:-1]))
    sd = r.std()
    if sd == 0 or len(r) < 30:
        return 0.0
    sr = r.mean() / sd
    z = (r - r.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    n = max(n_trials, 2)
    gamma = 0.5772156649
    expected_max = (1 - gamma) * _N.inv_cdf(1 - 1 / n) + gamma * _N.inv_cdf(1 - 1 / (n * math.e))
    sr0 = math.sqrt(max(trial_sr_var, 1e-8)) * expected_max
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-9))
    return round(_N.cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4)


def assess(spec: dict, ctx: dict, n_trials: int, trial_sr_var: float) -> dict:
    """Score one spec. ctx carries bars, indicator caches and split bounds for
    the primary asset plus the two cross-check assets."""
    bars, ind, sp = ctx["bars"], ctx["ind"], ctx["splits"]
    train, _ = evaluate(spec, bars, ind, *sp["train"])
    val, _ = evaluate(spec, bars, ind, *sp["val"])
    dev, dev_eq = evaluate(spec, bars, ind, *sp["dev"])

    rng = random.Random(dsl.key(spec))
    base = dev["sharpe"]
    kept = 0
    for _ in range(PERTURBATIONS):
        alt, _ = evaluate(dsl.perturb(spec, rng), bars, ind, *sp["dev"])
        kept += alt["sharpe"] >= 0.5 * base and alt["sharpe"] > 0
    stability = kept / PERTURBATIONS if base > 0 else 0.0

    others = []
    for other in ctx["others"]:
        m, _ = evaluate(spec, other["bars"], other["ind"], *other["splits"]["dev"])
        others.append(m["sharpe"])
    breadth = sum(s > 0 for s in others) / len(others) if others else 0.0

    dsr = deflated_sharpe(dev_eq, n_trials, trial_sr_var)
    decay_ok = train["sharpe"] <= 0 or val["sharpe"] >= MAX_DECAY * train["sharpe"]
    gates = [
        {"id": "trades", "label": "Enough trades", "pass": train["trades"] >= MIN_TRADES_TRAIN and val["trades"] >= MIN_TRADES_VAL,
         "detail": f"{train['trades']} train, {val['trades']} validation (need {MIN_TRADES_TRAIN} and {MIN_TRADES_VAL})"},
        {"id": "validation", "label": "Validation Sharpe", "pass": val["sharpe"] >= MIN_VAL_SHARPE,
         "detail": f"{val['sharpe']:.2f} (need {MIN_VAL_SHARPE})"},
        {"id": "decay", "label": "Train to validation decay", "pass": bool(decay_ok and train["sharpe"] > 0),
         "detail": f"train {train['sharpe']:.2f}, validation {val['sharpe']:.2f} (validation must keep {int(MAX_DECAY * 100)}% of train)"},
        {"id": "stability", "label": "Parameter stability", "pass": stability >= MIN_STABILITY,
         "detail": f"{kept} of {PERTURBATIONS} jittered variants kept half the Sharpe (need {int(MIN_STABILITY * 100)}%)"},
        {"id": "breadth", "label": "Works on another asset", "pass": breadth > 0,
         "detail": "cross-asset Sharpe " + ", ".join(f"{s:.2f}" for s in others)},
        {"id": "drawdown", "label": "Drawdown", "pass": dev["max_dd"] >= -MAX_DD,
         "detail": f"{dev['max_dd'] * 100:.0f}% worst (limit -{int(MAX_DD * 100)}%)"},
        {"id": "deflated", "label": "Deflated Sharpe", "pass": dsr >= MIN_DSR,
         "detail": f"{dsr:.2f} after {n_trials} trials (need {MIN_DSR})"},
    ]
    floor = max(0.0, min(train["sharpe"], val["sharpe"]))
    robust = floor * (0.5 + 0.5 * stability) * (0.6 + 0.4 * breadth) * min(1.0, train["trades"] / 40)
    return {"train": train, "val": val, "dev": dev, "stability": round(stability, 3), "breadth": round(breadth, 2),
            "cross_sharpes": others, "dsr": dsr, "gates": gates, "passed": all(g["pass"] for g in gates),
            "failed": [g["id"] for g in gates if not g["pass"]], "robust": round(robust, 3),
            "dev_sr_bar": per_bar_sharpe(dev_eq)}


def holdout_check(spec: dict, ctx: dict) -> dict:
    """The one place the sealed segment is read."""
    m, _ = evaluate(spec, ctx["bars"], ctx["ind"], *ctx["splits"]["holdout"])
    ok = m["sharpe"] >= HOLDOUT_MIN_SHARPE and m["max_dd"] >= -HOLDOUT_MAX_DD and m["total_return"] > 0
    return {"metrics": m, "pass": bool(ok)}
