"""Feature engineering.

Feature groups:
  technical   - momentum, trend, volatility, volume
  behavior    - proxies for human behavior: fear (VIX), crowd chasing (extension from trend),
                panic (drawdown + volume spikes), herding (stock correlation with the market)
  cross-asset - oil, gold, dollar, yields, bonds relative to their own history
  macro       - FRED series (rates, yield curve, credit spreads, consumer sentiment)
  news        - historical GDELT tone for the stock and the broad market
All features use only information available at each date (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def technical_features(px: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = px["Close"], px["High"], px["Low"], px["Volume"]
    f = pd.DataFrame(index=px.index)
    ret = c.pct_change()
    for n in (1, 5, 10, 21, 63, 126, 252):
        f[f"ret_{n}d"] = c.pct_change(n)
    for n in (10, 21, 63):
        f[f"vol_{n}d"] = ret.rolling(n).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21d"] / f["vol_63d"]
    f["rsi14"] = _rsi(c)
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma50_vs_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["bb_pos"] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std())
    f["drawdown_252"] = c / c.rolling(252, min_periods=60).max() - 1
    f["range_pos_252"] = (c - c.rolling(252, min_periods=60).min()) / (
        c.rolling(252, min_periods=60).max() - c.rolling(252, min_periods=60).min()
    )
    f["volume_z"] = (np.log1p(v) - np.log1p(v).rolling(63).mean()) / np.log1p(v).rolling(63).std()
    return f


def behavior_features(px: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """Proxies for sentiment and crowd behavior, and for how it is shifting."""
    f = pd.DataFrame(index=px.index)
    c = px["Close"]
    ret = c.pct_change()
    if "vix" in cross:
        vix = cross["vix"].reindex(px.index).ffill()
        f["vix"] = vix
        f["vix_z_252"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        f["vix_chg_5d"] = vix.pct_change(5)
        # regime: is fear rising or falling relative to its own trend
        f["vix_vs_sma50"] = vix / vix.rolling(50).mean() - 1
    if "spx" in cross:
        spx = cross["spx"].reindex(px.index).ffill()
        mret = spx.pct_change()
        f["mkt_ret_21d"] = spx.pct_change(21)
        f["mkt_dist_sma200"] = spx / spx.rolling(200).mean() - 1
        f["rel_strength_63d"] = c.pct_change(63) - spx.pct_change(63)
        # herding: how tightly the stock trades with the market, and how that is changing
        corr = ret.rolling(63).corr(mret)
        f["mkt_corr_63d"] = corr
        f["mkt_corr_chg"] = corr - corr.shift(63)
        f["beta_126d"] = ret.rolling(126).cov(mret) / mret.rolling(126).var()
    # crowd chasing: stretched above trend after a big run
    f["extension_z"] = (c / c.rolling(50).mean() - 1) / (c / c.rolling(50).mean() - 1).rolling(252).std()
    # panic indicator: big down day on heavy volume
    vz = (np.log1p(px["Volume"]) - np.log1p(px["Volume"]).rolling(63).mean()) / np.log1p(px["Volume"]).rolling(63).std()
    f["panic_21d"] = ((ret < -0.03) & (vz > 1)).astype(float).rolling(21).sum()
    # pattern persistence: does the stock tend to trend or mean revert lately?
    f["autocorr_63d"] = ret.rolling(63).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
    return f


def cross_asset_features(px_index: pd.Index, cross: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=px_index)
    for name in ("oil", "gold", "dollar", "bonds"):
        if name in cross:
            s = cross[name].reindex(px_index).ffill()
            f[f"{name}_ret_21d"] = s.pct_change(21)
            f[f"{name}_ret_63d"] = s.pct_change(63)
            f[f"{name}_dist_sma200"] = s / s.rolling(200).mean() - 1
    if "yield10y" in cross:
        y = cross["yield10y"].reindex(px_index).ffill()
        f["yield10y"] = y
        f["yield10y_chg_63d"] = y.diff(63)
    if "gold" in cross and "spx" in cross:
        g, s = cross["gold"].reindex(px_index).ffill(), cross["spx"].reindex(px_index).ffill()
        f["gold_vs_stocks_63d"] = g.pct_change(63) - s.pct_change(63)  # risk-off signal
    return f


def macro_features(px_index: pd.Index, macro: pd.DataFrame) -> pd.DataFrame:
    if macro is None or macro.empty:
        return pd.DataFrame(index=px_index)
    # Macro data is published with a lag; shift by 30 days to avoid look-ahead.
    m = macro.copy()
    m.index = m.index + pd.Timedelta(days=30)
    m = m.reindex(m.index.union(px_index)).ffill().reindex(px_index)
    out = m.copy()
    for col in m.columns:
        out[f"{col}_chg_90d"] = m[col].diff(90)
    return out.add_prefix("macro_")


def news_features(px_index: pd.Index, stock_tone: pd.Series | None, market_tone: pd.Series | None) -> pd.DataFrame:
    f = pd.DataFrame(index=px_index)
    for name, s in (("stock", stock_tone), ("market", market_tone)):
        if s is None or s.empty:
            continue
        s = s.reindex(s.index.union(px_index)).ffill().reindex(px_index)
        f[f"news_{name}_tone"] = s.rolling(3, min_periods=1).mean()
        f[f"news_{name}_tone_7d"] = s.rolling(7, min_periods=3).mean()
        f[f"news_{name}_tone_z"] = (s.rolling(7, min_periods=3).mean() - s.rolling(90, min_periods=30).mean()) / s.rolling(
            90, min_periods=30
        ).std()
    return f


def build_features(
    px: pd.DataFrame,
    cross: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    stock_tone: pd.Series | None = None,
    market_tone: pd.Series | None = None,
) -> pd.DataFrame:
    cross = cross if cross is not None else pd.DataFrame()
    parts = [
        technical_features(px),
        behavior_features(px, cross),
        cross_asset_features(px.index, cross),
        macro_features(px.index, macro),
        news_features(px.index, stock_tone, market_tone),
    ]
    X = pd.concat(parts, axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def make_labels(close: pd.Series, horizon: str) -> pd.DataFrame:
    """Forward return and a Buy/Hold/Sell label.

    The threshold scales with the stock's own recent volatility, so a 'Buy' means a move
    that is meaningful for that stock, not an arbitrary fixed percentage.
    Label: 2 = Buy, 1 = Hold, 0 = Sell. Last `days` rows have NaN labels (future unknown).
    """
    cfg = HORIZONS[horizon]
    d, k = cfg["days"], cfg["k"]
    fwd = close.shift(-d) / close - 1
    daily_vol = close.pct_change().rolling(63).std()
    thresh = k * daily_vol * np.sqrt(d)
    label = pd.Series(1.0, index=close.index)
    label[fwd > thresh] = 2.0
    label[fwd < -thresh] = 0.0
    label[fwd.isna() | thresh.isna()] = np.nan
    return pd.DataFrame({"fwd_ret": fwd, "label": label, "thresh": thresh})
