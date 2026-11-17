import random

import pytest

from lab import dsl

GOOD = {"side": "long", "entry": [{"ind": "rsi", "n": 14, "op": "<", "val": 30}], "exit": [],
        "stop_atr": 2, "max_hold": 40}


def test_normalize_returns_canonical_spec():
    spec = dsl.normalize(GOOD)
    assert spec["side"] == "long" and spec["stop_atr"] == 2.0 and spec["exit"] == []


def test_normalize_clamps_out_of_range_values():
    spec = dsl.normalize({**GOOD, "entry": [{"ind": "rsi", "n": 999, "op": "<", "val": 500}],
                          "stop_atr": 99, "max_hold": 99999})
    assert spec["entry"][0]["n"] == 30 and spec["entry"][0]["val"] == 95
    assert spec["stop_atr"] == 6.0 and spec["max_hold"] == 240


@pytest.mark.parametrize("bad", [
    {**GOOD, "side": "sideways"},
    {**GOOD, "entry": []},
    {**GOOD, "entry": [{"ind": "magic", "n": 5, "op": "<", "val": 1}]},
    {**GOOD, "entry": [{"ind": "rsi", "n": 14, "op": "=", "val": 30}]},
])
def test_normalize_rejects_invalid(bad):
    with pytest.raises(dsl.SpecError):
        dsl.normalize(bad)


def test_condition_counts_are_capped():
    cond = {"ind": "rsi", "n": 14, "op": "<", "val": 30}
    spec = dsl.normalize({**GOOD, "entry": [cond] * 6, "exit": [cond] * 5})
    assert len(spec["entry"]) == dsl.MAX_ENTRY and len(spec["exit"]) == dsl.MAX_EXIT


def test_key_is_stable_and_order_insensitive():
    spec = dsl.normalize(GOOD)
    assert dsl.key(spec) == dsl.key(dict(reversed(list(spec.items()))))


def test_random_perturb_and_mutate_always_produce_valid_specs():
    rng = random.Random(1)
    for _ in range(200):
        spec = dsl.random_spec(rng)
        assert dsl.normalize(spec) == spec
        assert dsl.normalize(dsl.perturb(spec, rng)) is not None
        assert dsl.normalize(dsl.mutate(spec, rng)) is not None


def test_indicators_never_use_future_bars(bars):
    """Value at bar i must be identical when every later bar is removed."""
    cut = 900
    head = {k: (v[:cut] if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in bars.items()}
    full, part = dsl.Indicators(bars), dsl.Indicators(head)
    for name, spec in dsl.INDICATORS.items():
        n = spec["n"][0] + 5
        m = spec["m"][1] if spec["m"] else None
        a, b = full.get(name, n, m)[:cut], part.get(name, n, m)
        mask = ~(a != a)  # compare only where defined
        assert (abs(a[mask] - b[mask]) < 1e-9).all(), name
