import numpy as np
import pandas as pd
import pytest


def make_prices(n: int = 1800, seed: int = 0, drift: float = 0.0004, vol: float = 0.013) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol_ = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol_}, index=idx)


def make_cross(index: pd.Index, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    cols = {
        "oil": 60 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        "gold": 1500 * np.exp(np.cumsum(rng.normal(0, 0.008, n))),
        "vix": 18 + 5 * np.abs(np.sin(np.arange(n) / 50)) + rng.normal(0, 1, n),
        "dollar": 95 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
        "yield10y": 2 + np.cumsum(rng.normal(0, 0.02, n)),
        "spx": 3000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
        "bonds": 120 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
    }
    return pd.DataFrame(cols, index=index)


@pytest.fixture
def px():
    return make_prices()


@pytest.fixture
def cross(px):
    return make_cross(px.index)
