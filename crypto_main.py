# =============================================================================
# crypto_main.py — Main Runner & Scheduler
# =============================================================================
# Run this script to start the bot.
#   python crypto_main.py
#
# It will:
#   1. Run a signal check immediately on startup.
#   2. Send alerts (email / Telegram) based on crypto_config.ALERT_MODE.
#   3. Optionally execute trades if crypto_config.AUTO_TRADE = True.
#   4. Repeat every crypto_config.CHECK_INTERVAL_MINUTES minutes.
# =============================================================================

import logging
import time
from datetime import datetime

import schedule

import crypto_config as cfg
from crypto_signals import get_exchange, run_all_signals, run_all_signals_paper
from crypto_alerts import send_alerts
from crypto_trader import execute_all_trades
from crypto_paper import paper_execute, get_portfolio_summary, get_current_portfolio


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(cfg.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_indicator_val(result, indicator_name: str) -> str:
    """Return a pre-computed indicator value from SignalResult for display."""
    try:
        if indicator_name == "RSI" and result.rsi_val is not None:
            return f"{result.rsi_val:.1f}"
        if indicator_name == "MACD" and result.macd_hist_val is not None:
            return f"{result.macd_hist_val:+.2f}"
    except Exception:
        pass
    return "—"


# ---------------------------------------------------------------------------

def run_job():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("Running signal check at %s", now_str)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Fetch signals — paper mode (CoinGecko) or live mode (ccxt exchange)
    # ------------------------------------------------------------------
    try:
        if cfg.PAPER_TRADE:
            results = run_all_signals_paper()
        else:
            exchange = get_exchange()
            results = run_all_signals(exchange)
    except Exception as exc:
        logger.error("Signal analysis failed: %s", exc, exc_info=True)
        return

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"  Signal Report — {now_str}  [{cfg.TIMEFRAME} candles]")
    if cfg.PAPER_TRADE:
        print("  Mode: PAPER TRADING (TradingView prices, no real orders)")
    print("=" * 60)

    # Price ticker line — one line per symbol with price + indicator snapshot
    print(f"  {'SYMBOL':<12} {'PRICE':>12}  {'RSI':>6}  {'MACD':>8}  {'DECISION'}")
    print("  " + "-" * 56)
    for r in results:
        rsi_val  = _get_indicator_val(r, "RSI")
        macd_val = _get_indicator_val(r, "MACD")
        decision = r.consensus or "HOLD"
        decision_fmt = (
            f"\033[92m{decision}\033[0m" if decision == "BUY"
            else f"\033[91m{decision}\033[0m" if decision == "SELL"
            else decision
        )
        print(
            f"  {r.symbol:<12} ${r.price:>11,.4f}"
            f"  {rsi_val:>6}  {macd_val:>8}  {decision_fmt}"
        )
    print()

    # Full signal breakdown per symbol
    for r in results:
        print(r.summary())

    # Paper portfolio summary
    if cfg.PAPER_TRADE:
        current_prices = {r.symbol: r.price for r in results}
        portfolio = get_current_portfolio()
        print()
        print(get_portfolio_summary(portfolio, current_prices))
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    actionable = [r for r in results if r.consensus is not None]
    if actionable:
        logger.info(
            "%d actionable signal(s) found. Sending alerts...",
            len(actionable),
        )
        send_alerts(results)
    else:
        logger.info("No actionable signals. No alerts sent.")
        # Uncomment to always send an email every cycle:
        # send_alerts(results)

    # ------------------------------------------------------------------
    # Trade execution (paper or live)
    # ------------------------------------------------------------------
    if cfg.PAPER_TRADE:
        if actionable:
            logger.info("Executing PAPER trades...")
            for r in actionable:
                msg = paper_execute(r.symbol, r.consensus, r.price)
                if msg:
                    print(msg)
            # Print updated portfolio after trades
            current_prices = {r.symbol: r.price for r in results}
            portfolio = get_current_portfolio()
            print()
            print(get_portfolio_summary(portfolio, current_prices))
        else:
            logger.info("No signals — no paper trades executed.")

    elif cfg.AUTO_TRADE:
        if actionable:
            logger.info("AUTO_TRADE enabled. Executing live trades...")
            try:
                exchange = get_exchange()
                orders = execute_all_trades(exchange, results)
                logger.info("%d order(s) placed.", len(orders))
            except Exception as exc:
                logger.error(
                    "Trade execution failed: %s", exc, exc_info=True
                )
        else:
            logger.info(
                "AUTO_TRADE enabled but no signals. Nothing to trade."
            )
    else:
        logger.info("AUTO_TRADE disabled. Alert-only mode.")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def main():
    logger.info("Crypto Signal Bot starting up.")
    logger.info("Symbols  : %s", cfg.SYMBOLS)
    if cfg.PAPER_TRADE:
        logger.info("Mode     : PAPER TRADING (CoinGecko, no exchange needed)")
    else:
        logger.info("Exchange : %s", cfg.EXCHANGE_ID)
    logger.info("Timeframe: %s", cfg.TIMEFRAME)
    logger.info("Interval : every %d minutes", cfg.CHECK_INTERVAL_MINUTES)
    logger.info("Alert    : %s", cfg.ALERT_MODE)
    logger.info(
        "AutoTrade: %s",
        "PAPER" if cfg.PAPER_TRADE else str(cfg.AUTO_TRADE),
    )
    logger.info("-" * 60)

    # Run immediately on startup
    run_job()

    # Schedule recurring runs
    schedule.every(cfg.CHECK_INTERVAL_MINUTES).minutes.do(run_job)

    logger.info(
        "Scheduler started. Next run in %d minutes.",
        cfg.CHECK_INTERVAL_MINUTES,
    )

    while True:
        schedule.run_pending()
        time.sleep(30)  # check scheduler every 30 seconds


if __name__ == "__main__":
    main()
