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
import os
import sys
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
# Helpers
# ---------------------------------------------------------------------------

def _supports_ansi() -> bool:
    """Return True if the terminal can render ANSI colour codes."""
    if sys.platform == "win32":
        return (
            os.environ.get("WT_SESSION") is not None        # Windows Terminal
            or os.environ.get("TERM_PROGRAM") == "vscode"  # VS Code terminal
            or os.environ.get("ANSICON") is not None        # ANSICON wrapper
        )
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_ANSI = _supports_ansi()


def _colorize(text: str, color: str) -> str:
    if not _ANSI:
        return text
    codes = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m"}
    return f"{codes.get(color, '')}{text}\033[0m"


# --- Cooldown and deduplication state (in-memory, resets on restart) ---
_last_signal_times = {}   # {symbol: {"BUY": datetime, "SELL": datetime}}
_last_executed = {}       # {symbol: {"side": str, "time": datetime}}


def _is_on_cooldown(symbol: str, side: str) -> bool:
    cooldown = getattr(cfg, "SIGNAL_COOLDOWN_MINUTES", 45)
    last = _last_signal_times.get(symbol, {}).get(side)
    if last is None:
        return False
    return (datetime.now() - last).total_seconds() / 60 < cooldown


def _record_signal(symbol: str, side: str) -> None:
    _last_signal_times.setdefault(symbol, {})[side] = datetime.now()


def _is_duplicate_trade(symbol: str, side: str) -> bool:
    dedup = getattr(cfg, "TRADE_DEDUP_MINUTES", 15)
    last = _last_executed.get(symbol, {})
    if last.get("side") != side or last.get("time") is None:
        return False
    return (datetime.now() - last["time"]).total_seconds() / 60 < dedup


def _record_executed(symbol: str, side: str) -> None:
    _last_executed[symbol] = {"side": side, "time": datetime.now()}


def _check_position_allowed(symbol: str, price: float) -> bool:
    """Return True if buying more of this asset stays within allocation cap."""
    max_alloc = getattr(cfg, "MAX_POSITION_ALLOCATION", 0.20)
    portfolio = get_current_portfolio()
    base = symbol.split("/")[0]
    total_value = portfolio["usdt_balance"]
    for pos in portfolio["holdings"].values():
        total_value += pos["amount"] * pos.get("avg_cost", 0)
    if total_value <= 0:
        return True
    current_value = portfolio["holdings"].get(base, {}).get("amount", 0) * price
    if current_value / total_value >= max_alloc:
        logger.info(
            "Position cap reached for %s (%.1f%% >= %.0f%%). Skipping BUY.",
            symbol, current_value / total_value * 100, max_alloc * 100,
        )
        return False
    return True


def _validate_config() -> list:
    """Return a list of configuration warnings. Empty = all good."""
    issues = []
    mode = cfg.ALERT_MODE.lower()
    if mode in ("email", "both"):
        if not cfg.EMAIL_SENDER or "your_gmail" in cfg.EMAIL_SENDER:
            issues.append("EMAIL_SENDER is not set in crypto_config.py")
        if not cfg.EMAIL_PASSWORD or "your_app_password" in cfg.EMAIL_PASSWORD:
            issues.append("EMAIL_PASSWORD is not set in crypto_config.py")
        if not cfg.EMAIL_RECIPIENTS or "your_email" in cfg.EMAIL_RECIPIENTS[0]:
            issues.append("EMAIL_RECIPIENTS is not set in crypto_config.py")
    if mode in ("telegram", "both"):
        if not cfg.TELEGRAM_BOT_TOKEN:
            issues.append("TELEGRAM_BOT_TOKEN is not set")
        if not cfg.TELEGRAM_CHAT_ID:
            issues.append("TELEGRAM_CHAT_ID is not set")
    if not cfg.PAPER_TRADE:
        if not cfg.EXCHANGE_API_KEY:
            issues.append("EXCHANGE_API_KEY not set — live trading will fail")
        if not cfg.EXCHANGE_API_SECRET:
            issues.append("EXCHANGE_API_SECRET not set — live trading will fail")
    return issues


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


def send_heartbeat() -> None:
    """Send a daily 'bot is alive' summary via the configured alert channel."""
    from crypto_alerts import send_heartbeat_alert
    portfolio = get_current_portfolio()
    n_trades = len(portfolio.get("trade_log", []))
    cash = portfolio.get("usdt_balance", 0)
    holdings = list(portfolio.get("holdings", {}).keys())
    lines = [
        f"Status   : Running ✅",
        f"Symbols  : {', '.join(cfg.SYMBOLS)}",
        f"Mode     : {'PAPER TRADING' if cfg.PAPER_TRADE else 'LIVE'}",
        f"Timeframe: {cfg.TIMEFRAME}  |  Interval: every {cfg.CHECK_INTERVAL_MINUTES}m",
        f"Cash     : ${cash:,.2f} USDT",
        f"Holdings : {', '.join(holdings) if holdings else 'None'}",
        f"Trades   : {n_trades} total in log",
    ]
    send_heartbeat_alert(lines)
    logger.info("Heartbeat sent.")


# ---------------------------------------------------------------------------

def run_job():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("Running signal check at %s", now_str)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Fetch signals — paper mode (TradingView) or live mode (ccxt exchange)
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
        if decision == "BUY":
            decision_fmt = _colorize(decision, "green")
        elif decision == "SELL":
            decision_fmt = _colorize(decision, "red")
        else:
            decision_fmt = decision
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
            executed_any = False
            for r in actionable:
                side = r.consensus

                # Guard 1 — cooldown
                if _is_on_cooldown(r.symbol, side):
                    logger.info(
                        "Cooldown active for %s %s — skipping.",
                        side, r.symbol,
                    )
                    continue

                # Guard 2 — deduplication
                if _is_duplicate_trade(r.symbol, side):
                    logger.info(
                        "Dedup guard: %s %s executed recently — skipping.",
                        side, r.symbol,
                    )
                    continue

                # Guard 3 — position allocation cap (BUY only)
                if side == "BUY" and not _check_position_allowed(
                    r.symbol, r.price
                ):
                    continue

                msg = paper_execute(
                    r.symbol, side, r.price, atr_val=r.atr_val
                )
                if msg:
                    print(_colorize(msg, "green" if side == "BUY" else "red"))
                    _record_signal(r.symbol, side)
                    _record_executed(r.symbol, side)
                    executed_any = True

            if executed_any:
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
        logger.info("Mode     : PAPER TRADING (TradingView, no exchange needed)")
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

    # --- Startup config validation ---
    issues = _validate_config()
    if issues:
        print("\n" + _colorize("⚠️  Configuration warnings:", "yellow"))
        for issue in issues:
            print(f"   • {issue}")
        print(
            "   Alerts may not work until these are fixed "
            "in crypto_config.py\n"
        )

    # Run immediately on startup
    run_job()

    # Schedule recurring signal checks
    schedule.every(cfg.CHECK_INTERVAL_MINUTES).minutes.do(run_job)

    # Schedule heartbeat (if enabled)
    if getattr(cfg, "HEARTBEAT_ENABLED", False):
        hb_hours = getattr(cfg, "HEARTBEAT_HOURS", 24)
        schedule.every(hb_hours).hours.do(send_heartbeat)
        logger.info("Heartbeat scheduled every %d hour(s).", hb_hours)

    logger.info(
        "Scheduler started. Next run in %d minutes. Press Ctrl+C to stop.\n",
        cfg.CHECK_INTERVAL_MINUTES,
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n" + _colorize("Bot stopped cleanly (Ctrl+C).", "yellow"))
        logger.info("Bot stopped by user.")


if __name__ == "__main__":
    main()
