# =============================================================================
# crypto_signals.py — Signal Engine
# Computes RSI, MACD, EMA crossover, and volume spike signals from OHLCV data.
# Supports both live exchange mode (ccxt) and paper mode (CoinGecko public API).
# =============================================================================

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import ccxt

import crypto_config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exchange helper
# ---------------------------------------------------------------------------

def get_exchange() -> ccxt.Exchange:
    """Instantiate and return the configured ccxt exchange."""
    exchange_class = getattr(ccxt, cfg.EXCHANGE_ID)
    params = {
        "apiKey": cfg.EXCHANGE_API_KEY or None,
        "secret": cfg.EXCHANGE_API_SECRET or None,
    }
    exchange = exchange_class(params)
    if cfg.USE_SANDBOX and exchange.has.get("sandbox"):
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str) -> pd.DataFrame:
    """Fetch OHLCV candles and return as a DataFrame."""
    raw = exchange.fetch_ohlcv(
        symbol, timeframe=cfg.TIMEFRAME, limit=cfg.CANDLE_LIMIT
    )
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Individual signal checkers  (return 'BUY', 'SELL', or None)
# ---------------------------------------------------------------------------

def signal_rsi(df: pd.DataFrame):
    rsi = calc_rsi(df["close"], cfg.RSI_PERIOD)
    last = rsi.iloc[-1]
    prev = rsi.iloc[-2]
    logger.debug("RSI last=%.2f prev=%.2f", last, prev)

    if prev >= cfg.RSI_OVERSOLD and last < cfg.RSI_OVERSOLD:
        return "BUY"   # crossed down into oversold
    if prev <= cfg.RSI_OVERBOUGHT and last > cfg.RSI_OVERBOUGHT:
        return "SELL"  # crossed up into overbought
    return None


def signal_macd(df: pd.DataFrame):
    macd, sig, hist = calc_macd(
        df["close"], cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL
    )
    # Bullish crossover: macd crosses above signal
    if hist.iloc[-2] < 0 and hist.iloc[-1] >= 0:
        return "BUY"
    # Bearish crossover: macd crosses below signal
    if hist.iloc[-2] > 0 and hist.iloc[-1] <= 0:
        return "SELL"
    return None


def signal_ema_crossover(df: pd.DataFrame):
    short = calc_ema(df["close"], cfg.EMA_SHORT)
    long_ = calc_ema(df["close"], cfg.EMA_LONG)
    # Golden cross: short EMA crosses above long EMA
    if short.iloc[-2] <= long_.iloc[-2] and short.iloc[-1] > long_.iloc[-1]:
        return "BUY"
    # Death cross: short EMA crosses below long EMA
    if short.iloc[-2] >= long_.iloc[-2] and short.iloc[-1] < long_.iloc[-1]:
        return "SELL"
    return None


def signal_volume_spike(df: pd.DataFrame):
    """Volume spike on up candle → BUY hint; on down candle → SELL hint."""
    vol = df["volume"]
    avg_vol = vol.iloc[-(cfg.VOLUME_SPIKE_WINDOW + 1):-1].mean()
    last_vol = vol.iloc[-1]
    last_close = df["close"].iloc[-1]
    last_open = df["open"].iloc[-1]

    if last_vol > cfg.VOLUME_SPIKE_MULTIPLIER * avg_vol:
        if last_close > last_open:
            return "BUY"
        else:
            return "SELL"
    return None


# ---------------------------------------------------------------------------
# Aggregate all signals for one symbol
# ---------------------------------------------------------------------------

SIGNAL_FUNCTIONS = {
    "RSI": signal_rsi,
    "MACD": signal_macd,
    "EMA Crossover": signal_ema_crossover,
    "Volume Spike": signal_volume_spike,
}


class SignalResult:
    """Holds the aggregated signal analysis for one symbol."""

    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self.price = price
        self.signals = {}  # indicator -> 'BUY' / 'SELL' / None
        self.buy_count = 0
        self.sell_count = 0
        self.consensus = None  # 'BUY', 'SELL', or None
        # Live indicator snapshots — populated during analysis for display
        self.rsi_val = None       # float or None
        self.macd_hist_val = None  # float or None

    def compute_consensus(self):
        self.buy_count = sum(1 for v in self.signals.values() if v == "BUY")
        self.sell_count = sum(1 for v in self.signals.values() if v == "SELL")
        if self.buy_count >= cfg.MIN_SIGNALS_TO_ACT:
            self.consensus = "BUY"
        elif self.sell_count >= cfg.MIN_SIGNALS_TO_ACT:
            self.consensus = "SELL"
        else:
            self.consensus = None

    def summary(self) -> str:
        lines = [
            f"=== {self.symbol} @ ${self.price:,.4f} ===",
            f"  Consensus: {self.consensus or 'NO SIGNAL'} "
            f"(BUY={self.buy_count}, SELL={self.sell_count})",
        ]
        for name, sig in self.signals.items():
            lines.append(f"  {name:20s}: {sig or '—'}")
        return "\n".join(lines)


def analyze_symbol(exchange: ccxt.Exchange, symbol: str) -> SignalResult:
    """Fetch data and run all signals for a single symbol."""
    logger.info("Analyzing %s ...", symbol)
    df = fetch_ohlcv(exchange, symbol)
    price = float(df["close"].iloc[-1])
    result = SignalResult(symbol, price)

    for name, fn in SIGNAL_FUNCTIONS.items():
        try:
            result.signals[name] = fn(df)
        except Exception as exc:
            logger.warning("Signal %s failed for %s: %s", name, symbol, exc)
            result.signals[name] = None

    # Cache indicator snapshots for display
    try:
        result.rsi_val = float(calc_rsi(df["close"]).iloc[-1])
    except Exception:
        pass
    try:
        _, _, hist = calc_macd(df["close"])
        result.macd_hist_val = float(hist.iloc[-1])
    except Exception:
        pass

    result.compute_consensus()
    logger.info(result.summary())
    return result


def run_all_signals(exchange: ccxt.Exchange):
    """Analyze every configured symbol and return results."""
    results = []
    for symbol in cfg.SYMBOLS:
        try:
            results.append(analyze_symbol(exchange, symbol))
        except Exception as exc:
            logger.error("Failed to analyze %s: %s", symbol, exc)
    return results


# ---------------------------------------------------------------------------
# Paper mode: CoinGecko-backed signal runner (no exchange needed)
# ---------------------------------------------------------------------------

def analyze_symbol_paper(symbol: str) -> SignalResult:
    """Fetch TradingView OHLCV data and run all signals for a single symbol."""
    from crypto_paper import fetch_ohlcv_tradingview  # avoid circular import
    logger.info("Analyzing %s (paper/TradingView) ...", symbol)
    df = fetch_ohlcv_tradingview(symbol)
    price = float(df["close"].iloc[-1])
    result = SignalResult(symbol, price)

    for name, fn in SIGNAL_FUNCTIONS.items():
        try:
            result.signals[name] = fn(df)
        except Exception as exc:
            logger.warning(
                "Signal %s failed for %s: %s", name, symbol, exc
            )
            result.signals[name] = None

    # Cache indicator snapshots for display
    try:
        result.rsi_val = float(calc_rsi(df["close"]).iloc[-1])
    except Exception:
        pass
    try:
        _, _, hist = calc_macd(df["close"])
        result.macd_hist_val = float(hist.iloc[-1])
    except Exception:
        pass

    result.compute_consensus()
    logger.info(result.summary())
    return result


def run_all_signals_paper():
    """Analyze every configured symbol using CoinGecko (no exchange needed)."""
    results = []
    for symbol in cfg.SYMBOLS:
        try:
            results.append(analyze_symbol_paper(symbol))
        except Exception as exc:
            logger.error(
                "Failed to analyze %s (paper): %s", symbol, exc
            )
    return results
