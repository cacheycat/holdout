import numpy as np

from lab import critic, dsl
from lab.backtest import FEE_BPS, SLIP_BPS, buy_and_hold, simulate


def test_entry_fills_on_next_bar_open(bars):
    ind = dsl.Indicators(bars)
    spec = dsl.normalize({"side": "long", "entry": [{"ind": "rsi", "n": 14, "op": "<", "val": 35}],
                          "exit": [], "stop_atr": None, "max_hold": 20})
    res = simulate(spec, bars, ind, 0, len(bars["c"]))
    mask = ind.mask(spec["entry"][0])
    assert res["trades"], "fixture should produce trades"
    for t in res["trades"]:
        assert mask[t["entry"] - 1], "a signal must exist on the bar before the fill"


def test_never_in_market_when_rule_is_unsatisfiable(bars):
    ind = dsl.Indicators(bars)
    spec = dsl.normalize({"side": "long", "entry": [{"ind": "rsi", "n": 14, "op": ">", "val": 95},
                                                    {"ind": "rsi", "n": 14, "op": "<", "val": 5}],
                          "exit": [], "stop_atr": None, "max_hold": 20})
    res = simulate(spec, bars, ind, 0, len(bars["c"]))
    assert res["trades"] == [] and np.allclose(res["equity"], 1.0)


def test_costs_are_charged_on_round_trip(bars):
    ind = dsl.Indicators(bars)
    spec = dsl.normalize({"side": "long", "entry": [{"ind": "rsi", "n": 14, "op": "<", "val": 95}],
                          "exit": [], "stop_atr": None, "max_hold": 6})
    res = simulate(spec, bars, ind, 0, len(bars["c"]))
    per_side = (FEE_BPS + SLIP_BPS) / 1e4
    assert all(t["ret"] < 0.5 for t in res["trades"])
    assert per_side > 0 and len(res["trades"]) > 10


def test_buy_and_hold_tracks_close(bars):
    eq = buy_and_hold(bars, 0, 100)
    assert abs(eq[-1] / eq[0] - bars["c"][99] / bars["c"][0]) < 0.05


def test_deflated_sharpe_falls_as_trials_grow():
    rng = np.random.default_rng(3)
    equity = np.cumprod(1 + rng.normal(0.0004, 0.01, 800))
    few = critic.deflated_sharpe(equity, 2, 0.01)
    many = critic.deflated_sharpe(equity, 200, 0.01)
    assert many <= few
