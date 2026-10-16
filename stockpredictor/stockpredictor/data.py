"""Data loaders: prices, cross-asset context, macro (FRED), and historical news tone (GDELT).

Everything is cached on disk and fails soft: if a source is unreachable the feature
simply goes missing instead of crashing the pipeline.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import CACHE_DIR, CROSS_ASSETS, FRED_SERIES

HEADERS = {"User-Agent": "stockpredictor/0.1 (educational project)"}


def _cache_path(name: str):
    return CACHE_DIR / f"{name}.parquet"


def _fresh(path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _read_cache(path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(str(path) + ".pkl")


def _write_cache(df: pd.DataFrame, path) -> None:
    try:
        df.to_parquet(path)
    except Exception:
        df.to_pickle(str(path) + ".pkl")


# ---------------------------------------------------------------- prices
def get_prices(ticker: str, years: int = 15, max_age_hours: float = 6) -> pd.DataFrame:
    """Daily OHLCV for a ticker, adjusted for splits and dividends."""
    import yfinance as yf

    safe = ticker.replace("^", "").replace("=", "_").replace("/", "_")
    path = _cache_path(f"px_{safe}")
    if _fresh(path, max_age_hours) or _fresh(type(path)(str(path) + ".pkl"), max_age_hours):
        return _read_cache(path)
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker!r}. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    _write_cache(df, path)
    return df


def get_cross_assets(years: int = 15) -> pd.DataFrame:
    """Close prices for oil, gold, VIX, dollar, yields, S&P 500 and long bonds."""
    cols = {}
    for name, sym in CROSS_ASSETS.items():
        try:
            cols[name] = get_prices(sym, years)["Close"]
        except Exception:
            continue
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- macro
def get_fred(max_age_hours: float = 24) -> pd.DataFrame:
    """Macro series from FRED's public CSV endpoint (no API key)."""
    path = _cache_path("fred")
    if _fresh(path, max_age_hours):
        return _read_cache(path)
    cols = {}
    for name, sid in FRED_SERIES.items():
        try:
            r = requests.get(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers=HEADERS, timeout=20
            )
            r.raise_for_status()
            s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
            cols[name] = pd.to_numeric(s, errors="coerce")
        except Exception:
            continue
    df = pd.DataFrame(cols)
    if not df.empty:
        _write_cache(df, path)
    return df


# ---------------------------------------------------------------- news
def get_gdelt_tone(query: str, years: int = 10, max_age_hours: float = 24) -> pd.Series:
    """Historical daily average news tone for a query, from GDELT (free, no key).

    GDELT tone is roughly -10 (very negative) to +10 (very positive). This is how the model
    learns what the news environment looked like on past dates.
    """
    safe = "".join(c if c.isalnum() else "_" for c in query)[:40]
    path = _cache_path(f"gdelt_{safe}")
    if _fresh(path, max_age_hours):
        return _read_cache(path).iloc[:, 0]
    chunks = []
    end = datetime.now()
    # GDELT timelines work best in windows of a few years
    window = timedelta(days=365 * 3)
    cursor = end - timedelta(days=365 * years)
    while cursor < end:
        stop = min(cursor + window, end)
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": query,
                    "mode": "timelinetone",
                    "format": "csv",
                    "startdatetime": cursor.strftime("%Y%m%d000000"),
                    "enddatetime": stop.strftime("%Y%m%d000000"),
                },
                headers=HEADERS,
                timeout=45,
            )
            r.raise_for_status()
            part = pd.read_csv(io.StringIO(r.text))
            if {"Date", "Value"}.issubset(part.columns):
                part = part[part["Series"].astype(str).str.contains("Tone", case=False, na=False)] if "Series" in part else part
                chunks.append(pd.Series(part["Value"].values, index=pd.to_datetime(part["Date"])))
        except Exception:
            pass
        cursor = stop
        time.sleep(1)
    if not chunks:
        return pd.Series(dtype=float, name="news_tone")
    s = pd.concat(chunks).sort_index()
    s = s[~s.index.duplicated()].rename("news_tone")
    _write_cache(s.to_frame(), path)
    return s


def get_current_headlines(ticker: str, company: str | None = None, limit: int = 25) -> list[str]:
    """Latest headlines from Google News RSS and Yahoo Finance RSS."""
    import feedparser

    queries = [ticker] + ([company] if company else [])
    urls = [f"https://news.google.com/rss/search?q={q}+stock&hl=en-US&gl=US&ceid=US:en" for q in queries]
    urls.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
    seen, out = set(), []
    for u in urls:
        try:
            feed = feedparser.parse(u)
        except Exception:
            continue
        for e in feed.entries:
            t = getattr(e, "title", "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out[:limit]


def get_market_headlines(limit: int = 30) -> list[str]:
    """Broad market headlines."""
    import feedparser

    urls = [
        "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    ]
    seen, out = set(), []
    for u in urls:
        try:
            feed = feedparser.parse(u)
        except Exception:
            continue
        for e in feed.entries:
            t = getattr(e, "title", "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out[:limit]
