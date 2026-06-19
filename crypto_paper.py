# =============================================================================
# crypto_paper.py — Paper Trading Engine
# =============================================================================
# Uses TradingView (via tvdatafeed) for real OHLCV data.
# Simulates buys/sells against a virtual portfolio stored in a JSON file.
# No exchange account needed for paper trading.
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

import crypto_config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TradingView timeframe map: cfg.TIMEFRAME string → tvdatafeed Interval
# ---------------------------------------------------------------------------
_TV_INTERVAL_MAP = {
    "1m":  Interval.in_1_minute,
    "3m":  Interval.in_3_minute,
    "5m":  Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "30m": Interval.in_30_minute,
    "45m": Interval.in_45_minute,
    "1h":  Interval.in_1_hour,
    "2h":  Interval.in_2_hour,
    "3h":  Interval.in_3_hour,
    "4h":  Interval.in_4_hour,
    "1d":  Interval.in_daily,
    "1w":  Interval.in_weekly,
    "1M":  Interval.in_monthly,
}

# Single shared TvDatafeed session (created once on first use)
_tv_session = None


def _get_tv():
    """Return a cached TvDatafeed session, creating it if needed."""
    global _tv_session
    if _tv_session is None:
        username = cfg.TV_USERNAME or None
        password = cfg.TV_PASSWORD or None
        if username and password:
            logger.info("Connecting to TradingView as %s ...", username)
            _tv_session = TvDatafeed(username, password)
        else:
            logger.info(
                "Connecting to TradingView without login "
                "(limited data — add TV_USERNAME/TV_PASSWORD in config)."
            )
            _tv_session = TvDatafeed()
    return _tv_session


# ---------------------------------------------------------------------------
# TradingView OHLCV fetch
# ---------------------------------------------------------------------------

def fetch_ohlcv_tradingview(
    symbol: str, timeframe_override: str = None, n_bars_override: int = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from TradingView for a given symbol (e.g. 'BTC/USDT').
    Returns a DataFrame with columns: open, high, low, close, volume.
    Symbol mapping is configured in TV_SYMBOL_MAP in crypto_config.py.

    timeframe_override: use a different timeframe (e.g. '4h' for trend filter)
    n_bars_override:    fetch a specific number of bars (e.g. for backtesting)
    """
    mapping = cfg.TV_SYMBOL_MAP.get(symbol)
    if not mapping:
        raise ValueError(
            f"No TradingView mapping configured for '{symbol}'. "
            "Add it to TV_SYMBOL_MAP in crypto_config.py."
        )
    tv_symbol, tv_exchange = mapping

    tf = timeframe_override or cfg.TIMEFRAME
    interval = _TV_INTERVAL_MAP.get(tf)
    if interval is None:
        raise ValueError(
            f"Unsupported timeframe '{tf}'. "
            f"Choose from: {list(_TV_INTERVAL_MAP.keys())}"
        )

    # Determine how many bars to request
    if n_bars_override:
        n_bars = n_bars_override
    elif timeframe_override:
        # Trend filter: fetch enough bars to warm up the EMA
        trend_period = getattr(cfg, "TREND_EMA_PERIOD", 50)
        n_bars = max(cfg.CANDLE_LIMIT, trend_period + 10)
    else:
        n_bars = cfg.CANDLE_LIMIT

    tv = _get_tv()
    df = tv.get_hist(
        symbol=tv_symbol,
        exchange=tv_exchange,
        interval=interval,
        n_bars=n_bars,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"TradingView returned no data for {tv_symbol} on {tv_exchange}. "
            "Check the symbol/exchange names in TV_SYMBOL_MAP."
        )

    # Standardise column names (tvdatafeed returns lowercase already)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "timestamp"

    logger.debug(
        "Fetched %d candles for %s from TradingView.", len(df), symbol
    )
    return df


# ---------------------------------------------------------------------------
# Paper portfolio
# ---------------------------------------------------------------------------

def _load_portfolio() -> dict:
    """Load portfolio from JSON file, or create a fresh one."""
    path = Path(cfg.PAPER_PORTFOLIO_FILE)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    # Fresh portfolio
    return {
        "usdt_balance": cfg.PAPER_STARTING_BALANCE,
        "holdings": {},      # symbol → {"amount": float, "avg_cost": float}
        "trade_log": [],     # list of trade dicts
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_portfolio(portfolio: dict) -> None:
    """Write portfolio atomically — write to .tmp then rename.
    Prevents JSON corruption if the process is killed mid-write."""
    path = Path(cfg.PAPER_PORTFOLIO_FILE)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(portfolio, f, indent=2)
    tmp_path.replace(path)  # atomic on all platforms


def get_portfolio_summary(portfolio: dict, current_prices: dict) -> str:
    """Return a human-readable portfolio summary string."""
    lines = ["📋 Paper Portfolio Summary"]
    lines.append(f"  Cash (USDT): ${portfolio['usdt_balance']:,.2f}")

    total_value = portfolio["usdt_balance"]
    for sym, pos in portfolio["holdings"].items():
        price = current_prices.get(sym, pos["avg_cost"])
        value = pos["amount"] * price
        pnl = value - pos["amount"] * pos["avg_cost"]
        pnl_pct = (pnl / (pos["amount"] * pos["avg_cost"])) * 100
        total_value += value
        lines.append(
            f"  {sym}: {pos['amount']:.6f} units "
            f"@ avg ${pos['avg_cost']:,.4f} | "
            f"now ${price:,.4f} | "
            f"value ${value:,.2f} | "
            f"PnL ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
        )

    lines.append(f"  ── Total Portfolio Value: ${total_value:,.2f} ──")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper trade execution
# ---------------------------------------------------------------------------

def paper_execute(symbol: str, side: str, price: float, atr_val=None):
    """
    Simulate a BUY or SELL for a symbol at the given price.
    Returns a human-readable trade confirmation string, or None if skipped.
    side:    'BUY' or 'SELL'
    atr_val: latest ATR value for volatility-scaled position sizing
    """
    portfolio = _load_portfolio()
    base = symbol.split("/")[0]
    now = datetime.now(timezone.utc).isoformat()

    if side == "BUY":
        # ATR-based position sizing: risk a fixed % of portfolio per ATR unit
        if (
            getattr(cfg, "ATR_SIZING_ENABLED", False)
            and atr_val is not None
            and atr_val > 0
            and price > 0
        ):
            port_value = portfolio["usdt_balance"]
            for pos in portfolio["holdings"].values():
                port_value += pos["amount"] * pos["avg_cost"]
            atr_pct = atr_val / price          # ATR as fraction of price
            risk_usdt = port_value * cfg.ATR_RISK_PCT
            usdt_to_spend = min(
                risk_usdt / atr_pct,
                port_value * cfg.MAX_TRADE_PCT,
                portfolio["usdt_balance"],
            )
        else:
            usdt_to_spend = (
                portfolio["usdt_balance"] * cfg.TRADE_AMOUNT_FRACTION
            )

        if usdt_to_spend < 1.0:
            logger.warning(
                "Paper BUY skipped for %s: insufficient USDT balance.", symbol
            )
            return None

        amount = usdt_to_spend / price
        portfolio["usdt_balance"] -= usdt_to_spend

        if base not in portfolio["holdings"]:
            portfolio["holdings"][base] = {"amount": 0.0, "avg_cost": price}

        # Update average cost
        existing = portfolio["holdings"][base]
        total_units = existing["amount"] + amount
        total_cost = (
            existing["amount"] * existing["avg_cost"] + usdt_to_spend
        )
        existing["amount"] = total_units
        existing["avg_cost"] = total_cost / total_units

        msg = (
            f"📄 PAPER BUY  {symbol}: "
            f"{amount:.6f} {base} @ ${price:,.4f} "
            f"(spent ${usdt_to_spend:,.2f} USDT)"
        )

    elif side == "SELL":
        if base not in portfolio["holdings"]:
            logger.warning(
                "Paper SELL skipped for %s: no holdings.", symbol
            )
            return None

        existing = portfolio["holdings"][base]
        sell_fraction = getattr(cfg, "SELL_AMOUNT_FRACTION", 0.50)
        amount = existing["amount"] * sell_fraction
        if amount <= 0:
            return None

        proceeds = amount * price
        existing["amount"] -= amount
        portfolio["usdt_balance"] += proceeds

        cost_basis = amount * existing["avg_cost"]
        pnl = proceeds - cost_basis

        if existing["amount"] < 1e-9:
            del portfolio["holdings"][base]

        msg = (
            f"📄 PAPER SELL {symbol}: "
            f"{amount:.6f} {base} @ ${price:,.4f} "
            f"(received ${proceeds:,.2f} USDT | PnL ${pnl:+,.2f})"
        )
    else:
        return None

    trade_record = {
        "timestamp": now,
        "symbol": symbol,
        "side": side,
        "price": price,
        "amount": amount,
    }
    portfolio["trade_log"].append(trade_record)
    _save_portfolio(portfolio)
    logger.info(msg)
    return msg


def get_current_portfolio() -> dict:
    """Public accessor for the current paper portfolio."""
    return _load_portfolio()
