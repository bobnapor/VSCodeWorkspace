# =============================================================================
# crypto_backtest.py — Backtesting Module
# =============================================================================
# Fetches historical OHLCV data from TradingView and simulates the signal bot
# running over every candle. No lookahead bias — each candle only sees data
# that would have been available at that moment in time.
#
# Run:
#   python crypto_backtest.py                     # all configured symbols
#   python crypto_backtest.py --symbol BTC/USDT   # single symbol
#   python crypto_backtest.py --bars 1000 --verbose
# =============================================================================

import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

import crypto_config as cfg
from crypto_signals import (
    SIGNAL_FUNCTIONS, calc_rsi, calc_macd, calc_ema, calc_atr, SignalResult,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def _fetch_history(symbol: str, n_bars: int) -> pd.DataFrame:
    """Fetch n_bars of OHLCV history from TradingView."""
    from crypto_paper import fetch_ohlcv_tradingview
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = fetch_ohlcv_tradingview(symbol, n_bars_override=n_bars)
    return df


# ---------------------------------------------------------------------------
# Single-symbol backtest
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str,
    n_bars: int = 500,
    verbose: bool = False,
    use_trend_filter: bool = None,
) -> dict:
    """
    Simulate the signal bot over historical data for one symbol.
    Returns a results dict, or None if data is insufficient.
    """
    use_trend = (
        use_trend_filter
        if use_trend_filter is not None
        else getattr(cfg, "TREND_FILTER_ENABLED", False)
    )

    print(f"\n{'=' * 62}")
    print(
        f"  Backtest: {symbol}  |  TF:{cfg.TIMEFRAME}  |  "
        f"{n_bars} bars  |  trend_filter={use_trend}"
    )
    print(f"{'=' * 62}")

    # Fetch data
    try:
        df = _fetch_history(symbol, n_bars)
    except Exception as exc:
        print(f"  ❌ Data fetch failed: {exc}")
        return None

    min_warmup = (
        max(cfg.MACD_SLOW + cfg.MACD_SIGNAL, cfg.EMA_LONG, cfg.RSI_PERIOD)
        + getattr(cfg, "EMA_CROSSOVER_CONFIRM_CANDLES", 2) + 5
    )
    if len(df) < min_warmup + 10:
        print(f"  ❌ Insufficient data ({len(df)} bars, need {min_warmup + 10}).")
        return None

    print(
        f"  Fetched {len(df)} bars  "
        f"({df.index[0].strftime('%Y-%m-%d')} → "
        f"{df.index[-1].strftime('%Y-%m-%d')})"
    )

    # Simulation state
    usdt = cfg.PAPER_STARTING_BALANCE
    holdings = 0.0
    avg_cost = 0.0
    trades = []
    portfolio_curve = []
    last_side = {}    # cooldown: {side: last_bar_index}

    cooldown_bars = max(
        1, int(getattr(cfg, "SIGNAL_COOLDOWN_MINUTES", 45) / 15)
    )

    for i in range(min_warmup, len(df)):
        window = df.iloc[: i + 1]
        price = float(window["close"].iloc[-1])
        ts = window.index[-1]

        portfolio_curve.append({"timestamp": ts, "value": usdt + holdings * price})

        # Run all signals on this window
        result = SignalResult(symbol, price)
        for name, fn in SIGNAL_FUNCTIONS.items():
            try:
                result.signals[name] = fn(window)
            except Exception:
                result.signals[name] = None

        # Optionally apply trend filter (uses same rolling window for speed)
        if use_trend:
            try:
                ema = calc_ema(window["close"], cfg.TREND_EMA_PERIOD)
                result.trend_bullish = price > float(ema.iloc[-1])
            except Exception:
                result.trend_bullish = None

        result.compute_consensus()
        side = result.consensus
        if side is None:
            continue

        # Cooldown guard
        last_bar = last_side.get(side, -999)
        if i - last_bar < cooldown_bars:
            continue
        last_side[side] = i

        # Execute simulated trade
        if side == "BUY" and usdt >= 1.0:
            # ATR sizing
            try:
                atr = float(calc_atr(window).iloc[-1])
                atr_pct = atr / price
                port_val = usdt + holdings * price
                spend = min(
                    port_val * cfg.ATR_RISK_PCT / atr_pct,
                    port_val * cfg.MAX_TRADE_PCT,
                    usdt,
                )
            except Exception:
                spend = usdt * cfg.TRADE_AMOUNT_FRACTION

            if spend < 1.0:
                continue
            amt = spend / price
            if holdings == 0:
                avg_cost = price
            else:
                total_units = holdings + amt
                avg_cost = (holdings * avg_cost + spend) / total_units
            holdings += amt
            usdt -= spend
            trades.append({
                "timestamp": str(ts), "side": "BUY",
                "price": price, "amount": amt,
            })
            if verbose:
                print(
                    f"  BUY  {ts.strftime('%Y-%m-%d %H:%M')}  "
                    f"@ ${price:,.2f}  amt={amt:.6f}  cash=${usdt:,.2f}"
                )

        elif side == "SELL" and holdings > 0:
            sell_frac = getattr(cfg, "SELL_AMOUNT_FRACTION", 0.50)
            amt = holdings * sell_frac
            proceeds = amt * price
            pnl = proceeds - amt * avg_cost
            usdt += proceeds
            holdings -= amt
            if holdings < 1e-9:
                holdings = 0.0
            trades.append({
                "timestamp": str(ts), "side": "SELL",
                "price": price, "amount": amt, "pnl": pnl,
            })
            if verbose:
                print(
                    f"  SELL {ts.strftime('%Y-%m-%d %H:%M')}  "
                    f"@ ${price:,.2f}  pnl=${pnl:+,.2f}  cash=${usdt:,.2f}"
                )

    # Close any open position at last price for stats
    final_price = float(df["close"].iloc[-1])
    final_value = usdt + holdings * final_price

    # --- Statistics ---
    buy_n = sum(1 for t in trades if t["side"] == "BUY")
    sell_n = sum(1 for t in trades if t["side"] == "SELL")
    pnls = [t["pnl"] for t in trades if "pnl" in t]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0

    # Max drawdown
    values = [p["value"] for p in portfolio_curve]
    peak = values[0] if values else cfg.PAPER_STARTING_BALANCE
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Annualised Sharpe (using candle-level returns)
    ret_series = pd.Series(values).pct_change().dropna()
    candles_per_year = (60 / 15) * 24 * 365   # for 15m candles
    sharpe = 0.0
    if ret_series.std() > 0:
        sharpe = (ret_series.mean() / ret_series.std()) * (candles_per_year ** 0.5)

    total_ret_pct = (
        (final_value - cfg.PAPER_STARTING_BALANCE) / cfg.PAPER_STARTING_BALANCE * 100
    )

    print(f"\n  Results:")
    print(f"  Start balance    : ${cfg.PAPER_STARTING_BALANCE:>10,.2f}")
    print(f"  Final value      : ${final_value:>10,.2f}")
    print(f"  Total return     : {total_ret_pct:>+.2f}%")
    print(f"  Trades           : {len(trades):>4}  ({buy_n} buys / {sell_n} sells)")
    print(
        f"  Win rate         : {win_rate:>5.1f}%  "
        f"({len(wins)} wins / {len(losses)} losses)"
    )
    print(f"  Max drawdown     : {max_dd * 100:>5.2f}%")
    print(f"  Sharpe ratio     : {sharpe:>6.2f}")
    if wins:
        print(f"  Avg win          : ${sum(wins)/len(wins):>+,.2f}")
    if losses:
        print(f"  Avg loss         : ${sum(losses)/len(losses):>+,.2f}")
    print()

    return {
        "symbol": symbol,
        "total_return_pct": total_ret_pct,
        "final_value": final_value,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "trades": trades,
        "portfolio_curve": portfolio_curve,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto Signal Bot — Backtester"
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Symbol to backtest (e.g. BTC/USDT). Default: all in config.",
    )
    parser.add_argument(
        "--bars", type=int, default=500,
        help="Number of historical candles to fetch (default: 500).",
    )
    parser.add_argument(
        "--no-trend", action="store_true",
        help="Disable trend filter for this backtest run.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every individual trade.",
    )
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else cfg.SYMBOLS
    all_results = []

    for sym in symbols:
        res = run_backtest(
            sym,
            n_bars=args.bars,
            verbose=args.verbose,
            use_trend_filter=not args.no_trend,
        )
        if res:
            all_results.append(res)

    if len(all_results) > 1:
        print("\n" + "=" * 62)
        print("  Summary across all symbols:")
        print("=" * 62)
        print(
            f"  {'SYMBOL':<12}  {'RETURN':>8}  {'WIN%':>6}  "
            f"{'MAX DD':>7}  {'SHARPE':>7}  {'TRADES':>6}"
        )
        print("  " + "─" * 58)
        for r in all_results:
            print(
                f"  {r['symbol']:<12}  {r['total_return_pct']:>+7.2f}%"
                f"  {r['win_rate']:>5.1f}%"
                f"  {r['max_drawdown_pct']:>6.2f}%"
                f"  {r['sharpe']:>7.2f}"
                f"  {r['n_trades']:>6}"
            )
        print()
