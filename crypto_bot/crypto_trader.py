# =============================================================================
# crypto_trader.py — Auto-Trade Executor (via ccxt)
# =============================================================================
# Only runs when cfg.AUTO_TRADE = True.
# Supports market or limit orders with optional stop-loss / take-profit.
# =============================================================================

import logging

import ccxt

import crypto_config as cfg
from crypto_signals import SignalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Balance helpers
# ---------------------------------------------------------------------------

def get_quote_balance(exchange: ccxt.Exchange, quote: str = "USDT") -> float:
    """Return free balance for the quote currency (e.g. USDT)."""
    balance = exchange.fetch_balance()
    return float(balance.get(quote, {}).get("free", 0.0))


def get_base_balance(exchange: ccxt.Exchange, base: str) -> float:
    """Return free balance for the base currency (e.g. BTC)."""
    balance = exchange.fetch_balance()
    return float(balance.get(base, {}).get("free", 0.0))


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,      # 'buy' or 'sell'
    amount: float,  # in base currency units
):
    """Place a market or limit order. Returns the order dict or None."""
    if cfg.ORDER_TYPE == "market":
        try:
            order = exchange.create_order(
                symbol, "market", side, amount
            )
            logger.info(
                "Market %s order placed: %s @ market — amount=%.6f",
                side.upper(), symbol, amount,
            )
            return order
        except Exception as exc:
            logger.error("Market order failed (%s %s): %s", side, symbol, exc)
            return None

    elif cfg.ORDER_TYPE == "limit":
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker["bid"]) if side == "buy" else float(ticker["ask"])
            order = exchange.create_order(
                symbol, "limit", side, amount, price
            )
            logger.info(
                "Limit %s order placed: %s @ %.4f — amount=%.6f",
                side.upper(), symbol, price, amount,
            )
            return order
        except Exception as exc:
            logger.error("Limit order failed (%s %s): %s", side, symbol, exc)
            return None

    logger.error("Unknown ORDER_TYPE: %s", cfg.ORDER_TYPE)
    return None


# ---------------------------------------------------------------------------
# Trade logic
# ---------------------------------------------------------------------------

def execute_trade(exchange: ccxt.Exchange, result: SignalResult):
    """
    Execute a trade for a SignalResult with BUY or SELL consensus.
    Returns the order dict or None if no trade was placed.
    """
    if not cfg.AUTO_TRADE:
        logger.info("AUTO_TRADE disabled. Skipping execution for %s.", result.symbol)
        return None

    if result.consensus is None:
        return None

    symbol = result.symbol
    base, quote = symbol.split("/")
    side = result.consensus.lower()  # 'buy' or 'sell'

    # ------ Determine trade amount ------
    if side == "buy":
        quote_balance = get_quote_balance(exchange, quote)
        trade_value = quote_balance * cfg.TRADE_AMOUNT_FRACTION
        if trade_value <= 0:
            logger.warning("Insufficient %s balance to buy %s.", quote, symbol)
            return None
        # Convert USDT → base units
        amount = trade_value / result.price
    else:  # sell
        amount = get_base_balance(exchange, base) * cfg.TRADE_AMOUNT_FRACTION
        if amount <= 0:
            logger.warning("Insufficient %s balance to sell %s.", base, symbol)
            return None

    # Respect exchange minimum order size
    try:
        markets = exchange.load_markets()
        market = markets.get(symbol, {})
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
        if amount < min_amount:
            logger.warning(
                "Amount %.6f below exchange minimum %.6f for %s. Skipping.",
                amount, min_amount, symbol,
            )
            return None
    except Exception as exc:
        logger.warning("Could not check min order size: %s", exc)

    order = place_order(exchange, symbol, side, amount)
    if order:
        entry_price = result.price
        sl = entry_price * (1 - cfg.STOP_LOSS_PCT) if side == "buy" else None
        tp = entry_price * (1 + cfg.TAKE_PROFIT_PCT) if side == "buy" else None
        logger.info(
            "Trade executed for %s: %s %.6f | Entry~$%.4f | SL~$%.4f | TP~$%.4f",
            symbol, side.upper(), amount, entry_price,
            sl or 0.0, tp or 0.0,
        )
        # Note: Actual stop-loss/take-profit orders require exchange support.
        # For production use, implement OCO or conditional orders here.
    return order


# ---------------------------------------------------------------------------
# Batch executor
# ---------------------------------------------------------------------------

def execute_all_trades(exchange: ccxt.Exchange, results):
    """Execute trades for all results that have a consensus signal."""
    orders = []
    for result in results:
        if result.consensus is not None:
            order = execute_trade(exchange, result)
            if order:
                orders.append(order)
    return orders
