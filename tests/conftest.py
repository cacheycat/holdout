import numpy as np
import pytest


@pytest.fixture()
def bars():
    """A deterministic random-walk series with a mild drift."""
    rng = np.random.default_rng(7)
    n = 1500
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    o = np.roll(c, 1)
    o[0] = c[0]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, n))
    return {"alias": "A", "symbol": "TEST", "t": np.arange(n) * 14_400_000.0,
            "o": o, "h": h, "l": low, "c": c, "v": rng.uniform(50, 150, n)}
