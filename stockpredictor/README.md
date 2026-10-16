# Stock Signal Tool

A Python CLI that gives Buy / Hold / Sell signals for stocks on two horizons (swing, about 2 weeks, and long-term, about 6 months), plus a walk-forward backtester that shows how the same logic would have performed historically.

This is decision support, not financial advice. No model reliably predicts markets. The backtester exists so you can see honestly whether the signals have any edge before trusting them.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m stockpredictor signal AAPL MSFT          # signals for specific tickers
python -m stockpredictor signal NVDA --explain     # plus the top drivers behind the call
python -m stockpredictor watchlist                 # default watchlist (edit in config.py)
python -m stockpredictor watchlist AAPL TSLA JPM   # your own list
python -m stockpredictor backtest AAPL --horizon both
python -m stockpredictor signal AAPL --no-news     # skip GDELT if it is slow or down
pytest                                             # run tests
```

## Web dashboard

```bash
python -m stockpredictor web                 # opens http://127.0.0.1:8000
python -m stockpredictor web --port 9000 --no-browser
```

Search any ticker for both signals, a price chart with 50/200-day averages, the top drivers, current headline sentiment, and one-click walk-forward backtests with a strategy vs buy-and-hold chart. The watchlist is editable (saved in your browser) and refreshes all signals with one button. The server uses only the standard library and listens on 127.0.0.1, so it is reachable only from your own computer. The first run for a ticker downloads years of data and can take a minute; after that results are cached.

## What goes into a signal

| Group | Inputs |
|---|---|
| Technical | multi-horizon returns, volatility, RSI, MACD, moving-average distance, Bollinger position, drawdown, volume |
| Human behavior | VIX level and change (fear), extension above trend (crowd chasing), panic days (big drop on heavy volume), correlation with the market and how it shifts (herding), return autocorrelation (trending vs mean-reverting regime) |
| Cross-asset | oil, gold, US dollar, 10-year yield, long bonds, gold vs stocks (risk-off) |
| Macro | fed funds, yield curve, credit spreads, consumer sentiment, unemployment (FRED, lagged 30 days to avoid look-ahead) |
| News | historical daily news tone for the stock and the broad market from GDELT, so the model learns how markets reacted to the news climate at the time. Live headlines (Google News and Yahoo Finance RSS) are scored with VADER and shown next to the signal |

## How it avoids fooling itself

- Features use only data available on each date (tested in `test_features_have_no_lookahead`).
- Walk-forward backtest: retrains on an expanding window, only predicts later dates, with a purge gap equal to the horizon so training labels never overlap the test period.
- Buy/Sell labels scale with each stock's own volatility, so a "Buy" means a move that is meaningful for that stock.
- A call needs a clear probability edge, otherwise it defaults to HOLD.
- Backtest results include transaction costs and are compared against simple buy-and-hold.

## Known limits

- Free data: GDELT tone is a coarse sentiment measure, and there is no free source of true historical per-headline news. FRED and yfinance can be slow or occasionally change.
- A single stock's model is trained only on that stock's history. Pooling many tickers into one model is the natural next upgrade.
- If backtests do not beat buy-and-hold after costs, believe the backtest.

## Layout

```
stockpredictor/
  config.py     tickers, horizons, backtest settings
  data.py       prices, cross-asset, FRED, GDELT, RSS headlines (cached)
  features.py   all feature groups and the Buy/Hold/Sell labels
  sentiment.py  headline scoring
  model.py      gradient boosting classifier and decision rule
  backtest.py   walk-forward predictions, strategy stats
  pipeline.py   ties it together
  cli.py        commands
  web.py        local dashboard server and JSON API
  static/       dashboard page (index.html)
tests/          pytest suite on synthetic data
```
