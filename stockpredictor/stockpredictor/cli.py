"""Command line interface.

  python -m stockpredictor signal AAPL MSFT      # buy/hold/sell for swing + long-term
  python -m stockpredictor signal AAPL --explain # also show top drivers
  python -m stockpredictor watchlist             # signals for the default watchlist
  python -m stockpredictor backtest AAPL --horizon swing
"""
from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_WATCHLIST, HORIZONS

DISCLAIMER = (
    "Educational decision support only, not financial advice. Models can and do lose money; "
    "past backtest results do not guarantee future performance."
)

ICON = {"BUY": "[+]", "HOLD": "[=]", "SELL": "[-]"}


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def print_signal(s, explain: bool = False) -> None:
    print(f"\n{s.ticker}  ${s.price:,.2f}  (data as of {s.as_of.date()})")
    for hs in (s.swing, s.long):
        label = HORIZONS[hs.horizon]["label"]
        print(
            f"  {label:<28} {ICON[hs.call]} {hs.call:<5} "
            f"(buy {hs.p_buy:.0%} / hold {hs.p_hold:.0%} / sell {hs.p_sell:.0%})"
        )
    h, m = s.headline_sentiment, s.market_sentiment
    if h["n"]:
        print(f"  Stock headline sentiment: {h['mean']:+.2f} from {h['n']} headlines")
        if h["top_pos"]:
            print(f"    most positive: {h['top_pos']}")
        if h["top_neg"]:
            print(f"    most negative: {h['top_neg']}")
    if m["n"]:
        print(f"  Market headline sentiment: {m['mean']:+.2f} from {m['n']} headlines")
    if explain:
        for hs in (s.swing, s.long):
            if len(hs.drivers):
                print(f"  Top drivers ({hs.horizon}):")
                for k, v in hs.drivers.items():
                    print(f"    {k:<28} {v:.4f}")


def cmd_signal(args) -> int:
    from .pipeline import compute_signal

    rc = 0
    for t in args.tickers:
        try:
            print_signal(compute_signal(t, use_news=not args.no_news, explain=args.explain), args.explain)
        except Exception as e:  # keep going for other tickers
            print(f"\n{t.upper()}: could not compute signal ({e})", file=sys.stderr)
            rc = 1
    print(f"\n{DISCLAIMER}")
    return rc


def cmd_watchlist(args) -> int:
    from .pipeline import compute_signal

    tickers = args.tickers or DEFAULT_WATCHLIST
    print(f"{'Ticker':<8}{'Price':>10}  {'Swing':<7}{'Long-term':<10}")
    for t in tickers:
        try:
            s = compute_signal(t, use_news=not args.no_news)
            print(f"{s.ticker:<8}{s.price:>10,.2f}  {s.swing.call:<7}{s.long.call:<10}")
        except Exception as e:
            print(f"{t.upper():<8}{'error':>10}  {e}", file=sys.stderr)
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_backtest(args) -> int:
    from .pipeline import backtest_ticker

    horizons = list(HORIZONS) if args.horizon == "both" else [args.horizon]
    for t in args.tickers:
        for hz in horizons:
            try:
                r = backtest_ticker(t, hz, use_news=not args.no_news)
            except Exception as e:
                print(f"{t.upper()} {hz}: backtest failed ({e})", file=sys.stderr)
                continue
            st, bh = r["strategy"], r["buy_hold"]
            print(f"\n{r['ticker']}  {HORIZONS[hz]['label']}  walk-forward {r['start']} to {r['end']}")
            print(f"  {'':<14}{'Strategy':>10}{'Buy&Hold':>10}")
            print(f"  {'Total return':<14}{_pct(st['total_return']):>10}{_pct(bh['total_return']):>10}")
            print(f"  {'CAGR':<14}{_pct(st['cagr']):>10}{_pct(bh['cagr']):>10}")
            print(f"  {'Sharpe':<14}{st['sharpe']:>10.2f}{bh['sharpe']:>10.2f}")
            print(f"  {'Max drawdown':<14}{_pct(st['max_drawdown']):>10}{_pct(bh['max_drawdown']):>10}")
            print(f"  Trades: {r['n_trades']}   Time invested: {r['pct_time_invested']:.0%}")
            sig = r["signals"]
            print(f"  Label accuracy: {sig['accuracy']:.1%}   (baseline avg fwd return {_pct(sig['baseline_avg_fwd_return'])})")
            for call, v in sorted(sig["avg_fwd_return_by_call"].items()):
                print(f"    avg {HORIZONS[hz]['days']}d forward return after {call}: {_pct(v)} (n={sig['call_counts'].get(call, 0)})")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_web(args) -> int:
    from .web import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="stockpredictor", description="Buy/Hold/Sell decision support")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("signal", help="signals for one or more tickers")
    s.add_argument("tickers", nargs="+")
    s.add_argument("--explain", action="store_true", help="show top feature drivers")
    s.add_argument("--no-news", action="store_true", help="skip GDELT historical news features")
    s.set_defaults(func=cmd_signal)

    w = sub.add_parser("watchlist", help="signals for a watchlist")
    w.add_argument("tickers", nargs="*")
    w.add_argument("--no-news", action="store_true")
    w.set_defaults(func=cmd_watchlist)

    b = sub.add_parser("backtest", help="walk-forward backtest")
    b.add_argument("tickers", nargs="+")
    b.add_argument("--horizon", choices=["swing", "long", "both"], default="both")
    b.add_argument("--no-news", action="store_true")
    b.set_defaults(func=cmd_backtest)

    wb = sub.add_parser("web", help="run the local web dashboard")
    wb.add_argument("--port", type=int, default=8000)
    wb.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    wb.set_defaults(func=cmd_web)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
