# =============================================================================
# crypto_config.py — Configuration for Crypto Signal Bot
# =============================================================================
# Fill in your credentials and preferences below.
# NEVER commit this file to a public repo with real keys in it.
# =============================================================================

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
PAPER_PORTFOLIO_FILE = "paper_portfolio.json"

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

# Stop-loss percentage below entry price (e.g. 0.03 = 3% stop-loss)
STOP_LOSS_PCT = 0.03

# Take-profit percentage above entry price (e.g. 0.06 = 6% take-profit)
TAKE_PROFIT_PCT = 0.06

# --------------- SCHEDULER SETTINGS ---------------
# How often (in minutes) to check signals
CHECK_INTERVAL_MINUTES = 15  # should align with TIMEFRAME (e.g. 15 for '15m')

# --------------- LOGGING ---------------
LOG_FILE = "crypto_bot.log"
LOG_LEVEL = "INFO"   # DEBUG, INFO, WARNING, ERROR
