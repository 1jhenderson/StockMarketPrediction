"""Walk-forward backtest.

The model is retrained on an expanding window and only ever predicts dates after its training
data (with a purge gap equal to the horizon). Positions: BUY = long, HOLD = keep previous position,
SELL = flat. Compared against buy-and-hold on the same dates, after transaction costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS, MIN_TRAIN_DAYS, RETRAIN_EVERY, TRANSACTION_COST
from .features import make_labels
from .model import SignalModel, decide


def walk_forward_predictions(
    X: pd.DataFrame, close: pd.Series, horizon: str, min_train: int = MIN_TRAIN_DAYS, retrain_every: int = RETRAIN_EVERY
) -> pd.DataFrame:
    cfg = HORIZONS[horizon]
    lab = make_labels(close, horizon)
    X = X.dropna(how="all")
    idx = X.index.intersection(lab.index)
    X, lab = X.loc[idx], lab.loc[idx]
    gap = cfg["days"]
    preds = []
    start = min_train
    while start < len(idx):
        end = min(start + retrain_every, len(idx))
        train_end = start - gap  # purge so no training label overlaps the test period
        if train_end < min_train // 2:
            break
        m = SignalModel(cfg["days"]).fit(X.iloc[:train_end], lab["label"].iloc[:train_end])
        p = m.predict_proba(X.iloc[start:end])
        preds.append(p)
        start = end
    if not preds:
        raise ValueError("Not enough history for a walk-forward backtest.")
    return pd.concat(preds)


def run_backtest(close: pd.Series, preds: pd.DataFrame, min_edge: float = 0.10) -> dict:
    calls = preds.apply(lambda r: decide(r, min_edge)[0], axis=1)
    position = pd.Series(np.nan, index=preds.index)
    position[calls == "BUY"] = 1.0
    position[calls == "SELL"] = 0.0
    position = position.ffill().fillna(0.0)
    ret = close.pct_change().reindex(preds.index)
    # trade at next day's return: signal at close t is applied to return t+1
    strat_ret = position.shift(1).fillna(0.0) * ret
    trades = position.diff().abs().fillna(0.0)
    strat_ret = strat_ret - trades * TRANSACTION_COST
    bh_ret = ret

    def stats(r: pd.Series) -> dict:
        r = r.dropna()
        if r.empty:
            return {}
        eq = (1 + r).cumprod()
        years = len(r) / 252
        dd = (eq / eq.cummax() - 1).min()
        vol = r.std() * np.sqrt(252)
        return {
            "total_return": float(eq.iloc[-1] - 1),
            "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
            "vol": float(vol),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
            "max_drawdown": float(dd),
        }

    # equity curves (weekly points) for charts, both starting at 1.0
    eq = pd.DataFrame({"strategy": (1 + strat_ret.fillna(0.0)).cumprod(), "buy_hold": (1 + bh_ret.fillna(0.0)).cumprod()})
    eq = eq.iloc[::5]
    curves = {
        "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
        "strategy": [round(float(v), 4) for v in eq["strategy"]],
        "buy_hold": [round(float(v), 4) for v in eq["buy_hold"]],
    }

    return {
        "strategy": stats(strat_ret),
        "buy_hold": stats(bh_ret),
        "curves": curves,
        "n_trades": int((trades > 0).sum()),
        "pct_time_invested": float(position.mean()),
        "start": preds.index[0].date(),
        "end": preds.index[-1].date(),
    }


def signal_accuracy(preds: pd.DataFrame, close: pd.Series, horizon: str, min_edge: float = 0.10) -> dict:
    """How often the calls matched what actually happened, and average forward return per call."""
    lab = make_labels(close, horizon).reindex(preds.index)
    calls = preds.apply(lambda r: decide(r, min_edge)[0], axis=1)
    mapping = {"SELL": 0.0, "HOLD": 1.0, "BUY": 2.0}
    valid = lab["label"].notna()
    acc = float((calls[valid].map(mapping) == lab["label"][valid]).mean()) if valid.any() else float("nan")
    avg = lab.groupby(calls)["fwd_ret"].mean().to_dict()
    counts = calls.value_counts().to_dict()
    base = float(lab["fwd_ret"].mean())
    return {"accuracy": acc, "avg_fwd_return_by_call": avg, "call_counts": counts, "baseline_avg_fwd_return": base}
