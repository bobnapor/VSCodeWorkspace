# =============================================================================
# crypto_portfolio.py — Standalone Paper Portfolio Viewer
# =============================================================================
# Run:  python crypto_portfolio.py
# Shows current holdings, live P&L, and recent trade history.
# =============================================================================

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crypto_config as cfg


def _fetch_prices() -> dict:
    """Attempt to fetch current prices from TradingView. Returns {} on failure."""
    prices = {}
    try:
        from crypto_paper import fetch_ohlcv_tradingview
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for symbol in cfg.SYMBOLS:
                try:
                    df = fetch_ohlcv_tradingview(symbol)
                    base = symbol.split("/")[0]
                    prices[base] = float(df["close"].iloc[-1])
                except Exception:
                    pass
    except Exception:
        pass
    return prices


def display_portfolio(live_prices: bool = True) -> None:
    path = Path(cfg.PAPER_PORTFOLIO_FILE)
    if not path.exists():
        print("No portfolio file found.")
        print(f"Expected: {cfg.PAPER_PORTFOLIO_FILE}")
        print("Run the bot in paper mode first (python crypto_main.py).")
        return

    with open(path) as f:
        portfolio = json.load(f)

    current_prices = _fetch_prices() if live_prices else {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 68)
    print(f"  📋  Paper Portfolio   —   {now}")
    if current_prices:
        print("  (prices live from TradingView)")
    else:
        print("  (live prices unavailable — showing avg cost as estimate)")
    print("=" * 68)

    cash = portfolio["usdt_balance"]
    total_value = cash
    print(f"\n  {'Cash (USDT)':<20}  ${cash:>12,.2f}")

    holdings = portfolio.get("holdings", {})
    if holdings:
        print()
        print(
            f"  {'ASSET':<8}  {'AMOUNT':>12}  {'AVG COST':>12}  "
            f"{'NOW':>12}  {'VALUE':>10}  {'P&L':>10}  {'%':>7}"
        )
        print("  " + "─" * 66)
        for base, pos in holdings.items():
            price = current_prices.get(base, pos["avg_cost"])
            value = pos["amount"] * price
            pnl = value - pos["amount"] * pos["avg_cost"]
            pct = pnl / (pos["amount"] * pos["avg_cost"]) * 100
            total_value += value
            sign = "+" if pnl >= 0 else ""
            print(
                f"  {base:<8}  {pos['amount']:>12.6f}  "
                f"${pos['avg_cost']:>11,.4f}  "
                f"${price:>11,.4f}  "
                f"${value:>9,.2f}  "
                f"{sign}${pnl:>8,.2f}  "
                f"{pct:>+6.1f}%"
            )
    else:
        print("\n  No open positions.")

    total_pnl = total_value - cfg.PAPER_STARTING_BALANCE
    total_pct = total_pnl / cfg.PAPER_STARTING_BALANCE * 100
    sign = "+" if total_pnl >= 0 else ""
    print()
    print(f"  {'Total Portfolio Value':<20}  ${total_value:>12,.2f}  "
          f"  {sign}${total_pnl:,.2f} ({total_pct:+.2f}% vs start)")
    print()

    # --- Trade history ---
    trades = portfolio.get("trade_log", [])
    total_trades = len(trades)
    if trades:
        recent = list(reversed(trades[-15:]))
        print(f"  📜 Trade History  (showing last {len(recent)} of {total_trades} total)")
        print()
        print(
            f"  {'TIME':<19}  {'SYMBOL':<12}  {'SIDE':<5}  "
            f"{'PRICE':>12}  {'AMOUNT':>12}"
        )
        print("  " + "─" * 64)
        for t in recent:
            ts = t.get("timestamp", "")[:16].replace("T", " ")
            side_tag = "BUY " if t["side"] == "BUY" else "SELL"
            print(
                f"  {ts:<19}  {t['symbol']:<12}  {side_tag:<5}  "
                f"${t['price']:>11,.4f}  {t['amount']:>12.6f}"
            )
    else:
        print("  No trades executed yet.")

    print()
    print("=" * 68)
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper Portfolio Viewer")
    parser.add_argument(
        "--no-prices", action="store_true",
        help="Skip fetching live prices (faster, offline-safe)"
    )
    args = parser.parse_args()
    display_portfolio(live_prices=not args.no_prices)
