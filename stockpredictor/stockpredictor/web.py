"""Local web dashboard. Standard library only; binds to 127.0.0.1.

  python -m stockpredictor web            # http://127.0.0.1:8000
  python -m stockpredictor web --port 9000 --no-browser

Work (signals, backtests) runs in background threads. The page starts a job, then polls it, so
slow first runs (data downloads) never time out the browser.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from .config import DEFAULT_WATCHLIST, HORIZONS
from .pipeline import backtest_ticker, compute_signal

STATIC = Path(__file__).parent / "static"
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,12}$")
SIGNAL_TTL = 1800  # seconds a computed signal is reused
JOB_TTL = 3600

_jobs: dict[str, dict] = {}
_signal_cache: dict[tuple, tuple[float, dict]] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------- serialization
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.Series):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.strftime("%Y-%m-%d")
    return o


def signal_to_dict(s) -> dict:
    def hz(h):
        return {
            "label": HORIZONS[h.horizon]["label"],
            "call": h.call,
            "confidence": h.confidence,
            "p_buy": h.p_buy,
            "p_hold": h.p_hold,
            "p_sell": h.p_sell,
            "drivers": h.drivers,
        }

    hist = s.history
    history = None
    if hist is not None and len(hist):
        history = {
            "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
            "close": [round(float(v), 2) for v in hist["close"]],
            "sma50": [None if pd.isna(v) else round(float(v), 2) for v in hist["sma50"]],
            "sma200": [None if pd.isna(v) else round(float(v), 2) for v in hist["sma200"]],
        }
    return jsonable(
        {
            "ticker": s.ticker,
            "as_of": s.as_of,
            "price": s.price,
            "swing": hz(s.swing),
            "long": hz(s.long),
            "headline_sentiment": s.headline_sentiment,
            "market_sentiment": s.market_sentiment,
            "history": history,
        }
    )


# ---------------------------------------------------------------- jobs
def _prune() -> None:
    now = time.time()
    for k in [k for k, j in _jobs.items() if now - j["created"] > JOB_TTL]:
        _jobs.pop(k, None)


def start_job(fn) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _prune()
        _jobs[job_id] = {"status": "running", "created": time.time(), "result": None, "error": None}

    def run():
        try:
            res = fn()
            with _lock:
                _jobs[job_id].update(status="done", result=res)
        except Exception as e:  # surfaced to the page
            with _lock:
                _jobs[job_id].update(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return job_id


def _signal_job(ticker: str, news: bool) -> dict:
    key = (ticker, news)
    with _lock:
        hit = _signal_cache.get(key)
    if hit and time.time() - hit[0] < SIGNAL_TTL:
        return hit[1]
    res = signal_to_dict(compute_signal(ticker, use_news=news, explain=True))
    with _lock:
        _signal_cache[key] = (time.time(), res)
    return res


def _backtest_job(ticker: str, horizon: str, news: bool) -> dict:
    return jsonable(backtest_ticker(ticker, horizon, use_news=news))


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "StockSignal/0.1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/watchlist":
            self._json({"tickers": DEFAULT_WATCHLIST})
        elif u.path == "/api/start":
            kind, ticker = q.get("kind", ""), q.get("ticker", "").strip().upper()
            news = q.get("news", "1") != "0"
            if not TICKER_RE.match(ticker):
                return self._json({"error": "Invalid ticker symbol."}, 400)
            if kind == "signal":
                job = start_job(lambda: _signal_job(ticker, news))
            elif kind == "backtest":
                hz = q.get("horizon", "swing")
                if hz not in HORIZONS:
                    return self._json({"error": "Invalid horizon."}, 400)
                job = start_job(lambda: _backtest_job(ticker, hz, news))
            else:
                return self._json({"error": "Unknown job kind."}, 400)
            self._json({"job": job})
        elif u.path == "/api/job":
            with _lock:
                j = _jobs.get(q.get("id", ""))
                snap = dict(j) if j else None
            if not snap:
                return self._json({"error": "Unknown job."}, 404)
            self._json({"status": snap["status"], "result": snap["result"], "error": snap["error"]})
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{httpd.server_address[1]}"
    print(f"Stock Signal dashboard running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
