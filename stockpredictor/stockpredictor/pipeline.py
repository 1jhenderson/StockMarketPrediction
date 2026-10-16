"""Ties data, features, model and backtest together."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import data, sentiment
from .backtest import run_backtest, signal_accuracy, walk_forward_predictions
from .config import HORIZONS
from .features import build_features, make_labels
from .model import SignalModel, decide


def load_feature_matrix(ticker: str, use_news: bool = True, years: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = data.get_prices(ticker, years)
    cross = data.get_cross_assets(years)
    macro = data.get_fred()
    stock_tone = market_tone = None
    if use_news:
        stock_tone = data.get_gdelt_tone(f'"{ticker}" sourcelang:eng', years=min(years, 10))
        market_tone = data.get_gdelt_tone("stock market economy sourcelang:eng", years=min(years, 10))
    X = build_features(px, cross, macro, stock_tone, market_tone)
    return px, X


@dataclass
class HorizonSignal:
    horizon: str
    call: str
    confidence: float
    p_buy: float
    p_hold: float
    p_sell: float
    drivers: pd.Series = field(default_factory=pd.Series)


@dataclass
class TickerSignal:
    ticker: str
    as_of: pd.Timestamp
    price: float
    swing: HorizonSignal
    long: HorizonSignal
    headline_sentiment: dict
    market_sentiment: dict
    history: pd.DataFrame | None = None  # recent close with SMA50/SMA200, for charts


def compute_signal(ticker: str, use_news: bool = True, explain: bool = False, px=None, X=None) -> TickerSignal:
    if px is None or X is None:
        px, X = load_feature_matrix(ticker, use_news)
    close = px["Close"]
    latest = X.dropna(how="all").iloc[[-1]]
    out = {}
    for name, cfg in HORIZONS.items():
        lab = make_labels(close, name)
        m = SignalModel(cfg["days"]).fit(X, lab["label"])
        p = m.predict_proba(latest).iloc[0]
        call, conf = decide(p)
        drivers = m.feature_importance(X, lab["label"]) if explain else pd.Series(dtype=float)
        out[name] = HorizonSignal(name, call, conf, p["p_buy"], p["p_hold"], p["p_sell"], drivers)
    try:
        hs = sentiment.score_headlines(data.get_current_headlines(ticker))
        ms = sentiment.score_headlines(data.get_market_headlines())
    except Exception:
        hs = ms = {"mean": 0.0, "n": 0, "top_pos": None, "top_neg": None}
    hist = pd.DataFrame(
        {"close": close, "sma50": close.rolling(50).mean(), "sma200": close.rolling(200).mean()}
    ).tail(400)
    return TickerSignal(ticker.upper(), latest.index[0], float(close.iloc[-1]), out["swing"], out["long"], hs, ms, hist)


def backtest_ticker(ticker: str, horizon: str = "swing", use_news: bool = True, px=None, X=None) -> dict:
    if px is None or X is None:
        px, X = load_feature_matrix(ticker, use_news)
    preds = walk_forward_predictions(X, px["Close"], horizon)
    res = run_backtest(px["Close"], preds)
    res["signals"] = signal_accuracy(preds, px["Close"], horizon)
    res["ticker"] = ticker.upper()
    res["horizon"] = horizon
    return res
