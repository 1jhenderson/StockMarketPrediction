import numpy as np
import pandas as pd

from stockpredictor.backtest import run_backtest, signal_accuracy, walk_forward_predictions
from stockpredictor.features import build_features, make_labels, news_features, technical_features
from stockpredictor.model import SignalModel, decide
from stockpredictor.sentiment import score_headlines


def test_features_build_and_no_inf(px, cross):
    X = build_features(px, cross)
    assert len(X) == len(px)
    assert not np.isinf(X.to_numpy(dtype=float)).any()
    for col in ("rsi14", "vix_z_252", "mkt_corr_63d", "oil_ret_21d", "extension_z"):
        assert col in X.columns


def test_features_have_no_lookahead(px, cross):
    """Features computed on truncated history must equal features computed on full history."""
    full = build_features(px, cross)
    cut = 1200
    part = build_features(px.iloc[:cut], cross.iloc[:cut])
    a = full.iloc[cut - 1].dropna()
    b = part.iloc[cut - 1].reindex(a.index)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-9)


def test_labels_shape_and_tail_nan(px):
    lab = make_labels(px["Close"], "swing")
    assert lab["label"].iloc[-10:].isna().all()
    assert set(lab["label"].dropna().unique()) <= {0.0, 1.0, 2.0}
    # all three classes should appear on random-walk data
    assert lab["label"].nunique() == 3


def test_rsi_bounds(px):
    f = technical_features(px)
    r = f["rsi14"].dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_news_features_alignment(px):
    tone = pd.Series(np.random.default_rng(0).normal(0, 1, 600), index=pd.bdate_range("2019-01-01", periods=600))
    f = news_features(px.index, tone, tone)
    assert "news_stock_tone_z" in f.columns
    assert len(f) == len(px)


def test_model_predicts_probabilities(px, cross):
    X = build_features(px, cross)
    lab = make_labels(px["Close"], "swing")
    m = SignalModel(10).fit(X.iloc[:1200], lab["label"].iloc[:1200])
    p = m.predict_proba(X.iloc[1300:1310])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert list(p.columns) == ["p_sell", "p_hold", "p_buy"]


def test_decide_requires_edge():
    even = pd.Series({"p_sell": 0.33, "p_hold": 0.34, "p_buy": 0.33})
    assert decide(even)[0] == "HOLD"
    bull = pd.Series({"p_sell": 0.1, "p_hold": 0.3, "p_buy": 0.6})
    assert decide(bull)[0] == "BUY"
    bear = pd.Series({"p_sell": 0.6, "p_hold": 0.3, "p_buy": 0.1})
    assert decide(bear)[0] == "SELL"


def test_walk_forward_never_uses_future(px, cross):
    X = build_features(px, cross)
    preds = walk_forward_predictions(X, px["Close"], "swing", min_train=700, retrain_every=100)
    assert preds.index.min() >= X.index[700]
    assert preds.index.is_monotonic_increasing
    assert not preds.index.duplicated().any()


def test_backtest_runs_and_reports(px, cross):
    X = build_features(px, cross)
    preds = walk_forward_predictions(X, px["Close"], "swing", min_train=700, retrain_every=100)
    res = run_backtest(px["Close"], preds)
    assert {"strategy", "buy_hold", "n_trades", "pct_time_invested"} <= set(res)
    assert -1.0 <= res["strategy"]["max_drawdown"] <= 0.0
    acc = signal_accuracy(preds, px["Close"], "swing")
    assert 0.0 <= acc["accuracy"] <= 1.0


def test_sentiment_direction():
    pos = score_headlines(["Company beats earnings, stock surges to record high"])
    neg = score_headlines(["Company misses estimates, shares plunge amid fraud probe"])
    assert pos["mean"] > 0 > neg["mean"]
    assert score_headlines([])["n"] == 0
