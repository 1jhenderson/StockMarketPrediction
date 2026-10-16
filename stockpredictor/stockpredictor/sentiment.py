"""Headline sentiment scoring. Uses VADER with a small finance lexicon boost."""
from __future__ import annotations

from statistics import mean

_FINANCE_LEXICON = {
    "beats": 2.0, "beat": 1.5, "surge": 2.0, "surges": 2.0, "soar": 2.0, "soars": 2.0, "rally": 1.8,
    "upgrade": 2.0, "upgraded": 2.0, "record": 1.2, "profit": 1.5, "growth": 1.2, "bullish": 2.0,
    "miss": -2.0, "misses": -2.0, "plunge": -2.5, "plunges": -2.5, "slump": -2.0, "downgrade": -2.0,
    "downgraded": -2.0, "lawsuit": -1.8, "probe": -1.5, "recall": -1.8, "layoffs": -1.5, "bearish": -2.0,
    "bankruptcy": -3.0, "fraud": -3.0, "cuts": -1.0, "warning": -1.5, "selloff": -2.2, "crash": -2.8,
}

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(_FINANCE_LEXICON)
    return _analyzer


def score_headlines(headlines: list[str]) -> dict:
    """Return mean sentiment in [-1, 1], count, and the strongest positive/negative headlines."""
    if not headlines:
        return {"mean": 0.0, "n": 0, "top_pos": None, "top_neg": None}
    a = _get_analyzer()
    scored = [(a.polarity_scores(h)["compound"], h) for h in headlines]
    scored.sort()
    return {
        "mean": mean(s for s, _ in scored),
        "n": len(scored),
        "top_pos": scored[-1][1] if scored[-1][0] > 0.2 else None,
        "top_neg": scored[0][1] if scored[0][0] < -0.2 else None,
    }
