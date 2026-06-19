# =============================================================================
# crypto_config.py — Configuration for Crypto Signal Bot
# =============================================================================
# Fill in your credentials and preferences below.
# NEVER commit this file to a public repo with real keys in it.
# =============================================================================

import os as _os

# Absolute path to this file's directory — ensures log/portfolio files are
# always written to the project folder, regardless of where the .bat is run.
BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))

# --------------- PAPER TRADING MODE ---------------
# True  = no exchange account needed; uses TradingView (via tvdatafeed)
#         for real OHLCV data and simulates trades in paper_portfolio.json
# False = live mode; uses the ccxt exchange configured below
PAPER_TRADE = True

# --------------- TRADINGVIEW SETTINGS (paper mode data source) ---------------
# Your TradingView login — gives access to more symbols and more history.
# Leave blank to use the unauthenticated feed (limited but functional).
TV_USERNAME = ""
TV_PASSWORD = ""

# TradingView symbol map: "BASE/QUOTE" → ("TV_SYMBOL", "TV_EXCHANGE")
# TV_SYMBOL: the symbol as shown on TradingView (no slash)
# TV_EXCHANGE: the exchange name as shown on TradingView
# Tip: search a pair on TradingView and check the top-left corner for the
#      exact exchange label, e.g. "BINANCE", "COINBASE", "KRAKEN", "BYBIT"
TV_SYMBOL_MAP = {
    "BTC/USDT":  ("BTCUSDT",  "BINANCE"),
    "ETH/USDT":  ("ETHUSDT",  "BINANCE"),
    "SOL/USDT":  ("SOLUSDT",  "BINANCE"),
    "HYPE/USDT": ("HYPEUSDT", "BYBIT"),
}

# Starting paper balance (in USDT) — only used on first run
PAPER_STARTING_BALANCE = 10000.0

# File to persist paper portfolio between runs
PAPER_PORTFOLIO_FILE = _os.path.join(BASE_DIR, "paper_portfolio.json")

# --------------- TRADING SYMBOLS TO WATCH ---------------
# Use CCXT format: 'BASE/QUOTE'  e.g. 'BTC/USDT', 'ETH/USDT', 'SOL/USDT'
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "HYPE/USDT",
]

# --------------- EXCHANGE SETTINGS ---------------
# Supported exchanges (via ccxt): 'binance', 'coinbase', 'kraken', 'bybit', etc.
EXCHANGE_ID = "binance"

# Leave blank if you only want alerts (no live trading)
EXCHANGE_API_KEY = ""
EXCHANGE_API_SECRET = ""

# Set to True to use sandbox/testnet instead of live trading
USE_SANDBOX = True

# --------------- SIGNAL SETTINGS ---------------
TIMEFRAME = "15m"  # OHLCV candle timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
CANDLE_LIMIT = 100  # How many candles to fetch for indicator calculations

# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30    # BUY signal when RSI drops below this
RSI_OVERBOUGHT = 70  # SELL signal when RSI rises above this

# MACD settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# EMA crossover settings (short EMA crosses above long EMA = BUY signal)
EMA_SHORT = 9
EMA_LONG = 21

# Volume spike: signal fires if volume > this multiple of the rolling average
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_SPIKE_WINDOW = 20  # rolling window (candles) for average volume

# EMA crossover confirmation — crossover must hold for N consecutive candles.
# 1 = fire on the first crossover candle, 2+ = reduces noise on short TFs.
EMA_CROSSOVER_CONFIRM_CANDLES = 2

# RSI zone entry — also fire BUY/SELL if RSI is *already* in the extreme zone
# when the bot starts/restarts (not just on the exact crossover candle).
RSI_ENTER_ON_ZONE = True

# --------------- TREND FILTER ---------------
# Only take BUY signals when price is above the higher-TF EMA (uptrend).
# Only take SELL signals when price is below it (downtrend).
# Prevents trading counter-trend, the biggest source of avoidable losses.
TREND_FILTER_ENABLED = True
TREND_TIMEFRAME = "4h"   # higher timeframe to evaluate trend on
TREND_EMA_PERIOD = 50    # EMA period on that higher timeframe

# --------------- ATR POSITION SIZING ---------------
# Scale position size inversely with volatility: less risk when price is
# moving fast, more when it's quiet.
ATR_SIZING_ENABLED = True
ATR_PERIOD = 14
ATR_RISK_PCT = 0.01     # risk this fraction of portfolio per ATR unit
MAX_TRADE_PCT = 0.10    # hard cap: never spend more than 10% per trade

# --------------- SIGNAL COOLDOWN ---------------
# After a signal fires, ignore the same direction for N minutes.
# Prevents pile-driving into a position on repeated signals.
SIGNAL_COOLDOWN_MINUTES = 45   # 3 × 15m candles

# --------------- POSITION LIMITS ---------------
# Never allocate more than this fraction of total portfolio to one asset.
MAX_POSITION_ALLOCATION = 0.20  # 20% max per asset

# --------------- TRADE DEDUPLICATION ---------------
# Don't re-execute the same symbol+side within this window (guards against
# double-execution if the scheduler fires twice after a restart).
TRADE_DEDUP_MINUTES = 15

# --------------- HEARTBEAT ---------------
# Periodically email/message a "bot is alive" summary.
HEARTBEAT_ENABLED = True
HEARTBEAT_HOURS = 24

# Minimum number of signals that must agree before alerting/trading
# 1 = fire on any single signal, 3 = require 3 signals to agree (more conservative)
MIN_SIGNALS_TO_ACT = 2

# --------------- ALERT MODE ---------------
# 'email'    — send Gmail SMTP alerts
# 'telegram' — send Telegram bot messages
# 'both'     — send both
# 'none'     — no alerts (useful if AUTO_TRADE = True and you just want execution)
ALERT_MODE = "email"

# --------------- EMAIL SETTINGS (Gmail SMTP) ---------------
# Use a Gmail "App Password" (not your real password).
# Enable 2FA on your Google account, then generate an App Password at:
# https://myaccount.google.com/apppasswords
EMAIL_SENDER = "your_gmail@gmail.com"
EMAIL_PASSWORD = "your_app_password_here"   # 16-char Gmail App Password
EMAIL_RECIPIENTS = ["your_email@example.com"]  # can be a list of addresses

# --------------- TELEGRAM SETTINGS (optional) ---------------
# Create a bot via @BotFather on Telegram, get the token.
# Get your chat_id by messaging @userinfobot.
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# --------------- AUTO-TRADE SETTINGS ---------------
# Set to True to actually place orders. False = alert-only mode.
AUTO_TRADE = False

# Order type for auto-trading: 'market' or 'limit'
ORDER_TYPE = "market"

# How much of your quote currency (e.g. USDT) to risk per trade (as a fraction)
# 0.05 = 5% of available USDT balance per trade
TRADE_AMOUNT_FRACTION = 0.05

# Fraction of holdings to sell per SELL signal.
# 0.50 = exit half the position (decisive). 1.0 = full exit.
SELL_AMOUNT_FRACTION = 0.50

# Stop-loss percentage below entry price (e.g. 0.03 = 3% stop-loss)
STOP_LOSS_PCT = 0.03

# Take-profit percentage above entry price (e.g. 0.06 = 6% take-profit)
TAKE_PROFIT_PCT = 0.06

# --------------- SCHEDULER SETTINGS ---------------
# How often (in minutes) to check signals
CHECK_INTERVAL_MINUTES = 15  # should align with TIMEFRAME (e.g. 15 for '15m')

# --------------- LOGGING ---------------
LOG_FILE = _os.path.join(BASE_DIR, "crypto_bot.log")
LOG_LEVEL = "INFO"   # DEBUG, INFO, WARNING, ERROR
