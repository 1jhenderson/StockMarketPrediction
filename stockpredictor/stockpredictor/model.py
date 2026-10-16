"""Model: gradient-boosted classifier predicting Buy / Hold / Sell, with calibrated probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


class SignalModel:
    def __init__(self, horizon_days: int, random_state: int = 0):
        self.horizon_days = horizon_days
        self.model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=5.0,
            min_samples_leaf=40,
            random_state=random_state,
        )
        self.columns: list[str] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SignalModel":
        data = pd.concat([X, y.rename("y")], axis=1).dropna(subset=["y"])
        # Purge: drop the last `horizon_days` rows, whose labels peek into the future of the training window end.
        data = data.iloc[: -self.horizon_days] if len(data) > self.horizon_days * 3 else data
        self.columns = list(X.columns)
        self.model.fit(data[self.columns], data["y"].astype(int))
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.model.predict_proba(X[self.columns])
        out = pd.DataFrame(0.0, index=X.index, columns=[0, 1, 2])
        for i, cls in enumerate(self.classes_):
            out[int(cls)] = p[:, i]
        out.columns = ["p_sell", "p_hold", "p_buy"]
        return out

    def feature_importance(self, X: pd.DataFrame, y: pd.Series, n: int = 8, n_repeats: int = 3) -> pd.Series:
        data = pd.concat([X[self.columns], y.rename("y")], axis=1).dropna(subset=["y"]).iloc[-750:]
        r = permutation_importance(
            self.model, data[self.columns], data["y"].astype(int), n_repeats=n_repeats, random_state=0, n_jobs=1
        )
        return pd.Series(r.importances_mean, index=self.columns).sort_values(ascending=False).head(n)


def decide(p: pd.Series, min_edge: float = 0.10) -> tuple[str, float]:
    """Turn probabilities into a call. Requires a clear edge over the alternatives, otherwise HOLD.

    Returns (call, confidence) where confidence is the probability margin of the chosen class.
    """
    p_buy, p_sell, p_hold = float(p["p_buy"]), float(p["p_sell"]), float(p["p_hold"])
    if p_buy - p_sell >= min_edge and p_buy >= p_hold * 0.9:
        return "BUY", p_buy - max(p_sell, p_hold)
    if p_sell - p_buy >= min_edge and p_sell >= p_hold * 0.9:
        return "SELL", p_sell - max(p_buy, p_hold)
    return "HOLD", max(p_hold - max(p_buy, p_sell), 0.0)
