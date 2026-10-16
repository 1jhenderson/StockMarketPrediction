"""Central configuration."""
from pathlib import Path

CACHE_DIR = Path.home() / ".stockpredictor_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cross-asset context tickers (all free via yfinance)
CROSS_ASSETS = {
    "oil": "CL=F",
    "gold": "GC=F",
    "vix": "^VIX",
    "dollar": "DX-Y.NYB",
    "yield10y": "^TNX",
    "spx": "^GSPC",
    "bonds": "TLT",
}

# FRED series (free CSV endpoint, no key needed)
FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "yield_curve": "T10Y2Y",
    "credit_spread": "BAMLH0A0HYM2",
    "consumer_sent": "UMCSENT",
    "unemployment": "UNRATE",
}

# Horizons: trading days ahead and the move (in units of recent volatility) that counts as buy/sell
HORIZONS = {
    "swing": {"days": 10, "k": 0.5, "label": "Swing (about 2 weeks)"},
    "long": {"days": 126, "k": 0.5, "label": "Long-term (about 6 months)"},
}

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "XOM", "UNH",
    "V", "WMT", "PG", "HD", "KO",
]

# Backtest settings
TRANSACTION_COST = 0.0005  # 5 bps per position change
MIN_TRAIN_DAYS = 756       # about 3 years before the first walk-forward fold
RETRAIN_EVERY = 63         # retrain roughly quarterly
