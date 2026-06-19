# =============================================================================
# crypto_signals.py — Signal Engine
# Computes RSI, MACD, EMA crossover, volume spike, and ATR from OHLCV data.
# Supports live exchange mode (ccxt) and paper mode (TradingView).
# =============================================================================

import logging
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


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures per-candle volatility."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Individual signal checkers  (return 'BUY', 'SELL', or None)
# ---------------------------------------------------------------------------

def signal_rsi(df: pd.DataFrame):
    rsi = calc_rsi(df["close"], cfg.RSI_PERIOD)
    last = rsi.iloc[-1]
    prev = rsi.iloc[-2]
    logger.debug("RSI last=%.2f prev=%.2f", last, prev)

    # Primary: crossover — RSI just entered the extreme zone this candle
    if prev >= cfg.RSI_OVERSOLD and last < cfg.RSI_OVERSOLD:
        return "BUY"
    if prev <= cfg.RSI_OVERBOUGHT and last > cfg.RSI_OVERBOUGHT:
        return "SELL"

    # Secondary: already in zone (catches signals after a restart/gap)
    if getattr(cfg, "RSI_ENTER_ON_ZONE", True):
        if last < cfg.RSI_OVERSOLD:
            return "BUY"
        if last > cfg.RSI_OVERBOUGHT:
            return "SELL"

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
    n = max(1, getattr(cfg, "EMA_CROSSOVER_CONFIRM_CANDLES", 2))

    # Require crossover to hold for n consecutive candles before firing.
    # This eliminates most noise-driven false crossovers on short timeframes.
    try:
        bull_now = all(
            short.iloc[-(i + 1)] > long_.iloc[-(i + 1)] for i in range(n)
        )
        if bull_now and short.iloc[-(n + 1)] <= long_.iloc[-(n + 1)]:
            return "BUY"
        bear_now = all(
            short.iloc[-(i + 1)] < long_.iloc[-(i + 1)] for i in range(n)
        )
        if bear_now and short.iloc[-(n + 1)] >= long_.iloc[-(n + 1)]:
            return "SELL"
    except IndexError:
        pass  # not enough data for confirmation window
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
        self.signals = {}   # indicator -> 'BUY' / 'SELL' / None
        self.buy_count = 0
        self.sell_count = 0
        self.consensus = None    # 'BUY', 'SELL', or None
        # Indicator snapshots populated during analysis (for display + sizing)
        self.rsi_val = None        # float
        self.macd_hist_val = None  # float
        self.atr_val = None        # float — latest ATR in price terms
        # Trend filter: True=uptrend, False=downtrend, None=filter disabled
        self.trend_bullish = None

    def compute_consensus(self):
        buys = {k for k, v in self.signals.items() if v == "BUY"}
        sells = {k for k, v in self.signals.items() if v == "SELL"}

        # Trend filter: suppress signals that go against the macro trend
        if self.trend_bullish is True:
            sells = set()   # uptrend — ignore SELL signals
        elif self.trend_bullish is False:
            buys = set()    # downtrend — ignore BUY signals

        self.buy_count = len(buys)
        self.sell_count = len(sells)

        if self.buy_count >= cfg.MIN_SIGNALS_TO_ACT:
            self.consensus = "BUY"
        elif self.sell_count >= cfg.MIN_SIGNALS_TO_ACT:
            self.consensus = "SELL"
        else:
            self.consensus = None

    def summary(self) -> str:
        if self.trend_bullish is True:
            trend_str = "  Trend (4h)      : ▲ UPTREND  (SELL signals suppressed)"
        elif self.trend_bullish is False:
            trend_str = "  Trend (4h)      : ▼ DOWNTREND (BUY signals suppressed)"
        else:
            trend_str = None

        lines = [
            f"=== {self.symbol} @ ${self.price:,.4f} ===",
            f"  Consensus: {self.consensus or 'NO SIGNAL'} "
            f"(BUY={self.buy_count}, SELL={self.sell_count})",
        ]
        if trend_str:
            lines.append(trend_str)
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

    # Cache indicator snapshots for display / position sizing
    try:
        result.rsi_val = float(calc_rsi(df["close"]).iloc[-1])
    except Exception:
        pass
    try:
        _, _, hist = calc_macd(df["close"])
        result.macd_hist_val = float(hist.iloc[-1])
    except Exception:
        pass
    try:
        result.atr_val = float(
            calc_atr(df, getattr(cfg, "ATR_PERIOD", 14)).iloc[-1]
        )
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
# Paper mode: TradingView-backed signal runner (no exchange account needed)
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

    # Cache indicator snapshots for display / position sizing
    try:
        result.rsi_val = float(calc_rsi(df["close"]).iloc[-1])
    except Exception:
        pass
    try:
        _, _, hist = calc_macd(df["close"])
        result.macd_hist_val = float(hist.iloc[-1])
    except Exception:
        pass
    try:
        result.atr_val = float(
            calc_atr(df, getattr(cfg, "ATR_PERIOD", 14)).iloc[-1]
        )
    except Exception:
        pass

    # Higher-timeframe trend filter
    if getattr(cfg, "TREND_FILTER_ENABLED", False):
        try:
            trend_df = fetch_ohlcv_tradingview(
                symbol, timeframe_override=cfg.TREND_TIMEFRAME
            )
            trend_ema = calc_ema(trend_df["close"], cfg.TREND_EMA_PERIOD)
            last_close = float(trend_df["close"].iloc[-1])
            ema_val = float(trend_ema.iloc[-1])
            result.trend_bullish = last_close > ema_val
            logger.debug(
                "%s trend (%s EMA%d): %s  close=%.4f  ema=%.4f",
                symbol, cfg.TREND_TIMEFRAME, cfg.TREND_EMA_PERIOD,
                "BULL" if result.trend_bullish else "BEAR",
                last_close, ema_val,
            )
        except Exception as exc:
            logger.warning("Trend filter failed for %s: %s", symbol, exc)
            result.trend_bullish = None

    result.compute_consensus()
    logger.info(result.summary())
    return result


def run_all_signals_paper():
    """Analyze every configured symbol using TradingView (no exchange needed)."""
    results = []
    for symbol in cfg.SYMBOLS:
        try:
            results.append(analyze_symbol_paper(symbol))
        except Exception as exc:
            logger.error(
                "Failed to analyze %s (paper): %s", symbol, exc
            )
    return results
