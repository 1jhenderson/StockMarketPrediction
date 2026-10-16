import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from stockpredictor import data, pipeline, web
from stockpredictor.features import build_features


@pytest.fixture
def server(monkeypatch, px, cross):
    X = build_features(px, cross)
    monkeypatch.setattr(data, "get_current_headlines", lambda t, company=None, limit=25: ["Company beats estimates, shares surge"])
    monkeypatch.setattr(data, "get_market_headlines", lambda limit=30: ["Stocks slump as fears grow"])
    monkeypatch.setattr(web, "compute_signal", lambda t, use_news=True, explain=False: pipeline.compute_signal(t, px=px, X=X, explain=explain))
    monkeypatch.setattr(web, "backtest_ticker", lambda t, hz, use_news=True: pipeline.backtest_ticker(t, hz, px=px, X=X))
    web._signal_cache.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.headers, r.read()


def wait_job(base, job, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, _, body = get(f"{base}/api/job?id={job}")
        j = json.loads(body)
        if j["status"] != "running":
            return j
        time.sleep(0.3)
    raise TimeoutError


def test_index_served(server):
    status, headers, body = get(server + "/")
    assert status == 200 and b"Stock Signal" in body
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_signal_job(server):
    _, _, body = get(server + "/api/start?kind=signal&ticker=SYNTH&news=0")
    j = wait_job(server, json.loads(body)["job"])
    assert j["status"] == "done", j
    r = j["result"]
    assert r["swing"]["call"] in {"BUY", "HOLD", "SELL"} and r["long"]["call"] in {"BUY", "HOLD", "SELL"}
    assert abs(r["swing"]["p_buy"] + r["swing"]["p_hold"] + r["swing"]["p_sell"] - 1) < 1e-6
    assert len(r["history"]["dates"]) == len(r["history"]["close"]) == 400
    assert r["swing"]["drivers"]
    assert r["headline_sentiment"]["n"] == 1


def test_backtest_job(server):
    _, _, body = get(server + "/api/start?kind=backtest&ticker=SYNTH&horizon=swing&news=0")
    j = wait_job(server, json.loads(body)["job"])
    assert j["status"] == "done", j
    r = j["result"]
    assert r["curves"]["dates"] and len(r["curves"]["strategy"]) == len(r["curves"]["dates"])
    assert "total_return" in r["strategy"] and "accuracy" in r["signals"]


@pytest.mark.parametrize("path", [
    "/api/start?kind=signal&ticker=../../etc/passwd",
    "/api/start?kind=signal&ticker=",
    "/api/start?kind=bogus&ticker=AAPL",
    "/api/start?kind=backtest&ticker=AAPL&horizon=nope",
])
def test_bad_requests_rejected(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + path)
    assert e.value.code == 400


def test_unknown_job_and_route(server):
    for path, code in (("/api/job?id=nope", 404), ("/nothing", 404)):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server + path)
        assert e.value.code == code


def test_job_error_is_reported(server, monkeypatch):
    def boom(*a, **k):
        raise ValueError("No price data returned for 'ZZZZ'.")

    monkeypatch.setattr(web, "compute_signal", boom)
    _, _, body = get(server + "/api/start?kind=signal&ticker=ZZZZ&news=0")
    j = wait_job(server, json.loads(body)["job"])
    assert j["status"] == "error" and "No price data" in j["error"]
