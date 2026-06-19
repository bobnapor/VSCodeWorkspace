# =============================================================================
# crypto_dashboard.py — Streamlit Web Dashboard
# =============================================================================
# Run:  streamlit run crypto_dashboard.py
# Opens a browser at http://localhost:8501
# Press R to refresh, or enable auto-refresh via the sidebar.
# =============================================================================

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

import crypto_config as cfg

# --- Page config ---
st.set_page_config(
    page_title="Crypto Signal Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_portfolio():
    path = Path(cfg.PAPER_PORTFOLIO_FILE)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def fetch_live_prices():
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Controls")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Bot Config")
    st.caption(f"**Mode:** {'📄 Paper' if cfg.PAPER_TRADE else '⚡ Live'}")
    st.caption(f"**Timeframe:** {cfg.TIMEFRAME}")
    st.caption(f"**Check every:** {cfg.CHECK_INTERVAL_MINUTES}m")
    st.caption(f"**Alert mode:** {cfg.ALERT_MODE}")
    st.caption(f"**Trend filter:** {getattr(cfg, 'TREND_FILTER_ENABLED', False)}")
    st.caption(f"**ATR sizing:** {getattr(cfg, 'ATR_SIZING_ENABLED', False)}")
    st.caption(f"**Min signals:** {cfg.MIN_SIGNALS_TO_ACT}")
    st.divider()
    st.caption(f"Auto-caches for 60s. Last load: {datetime.now().strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🤖 Crypto Signal Bot Dashboard")

portfolio = load_portfolio()

if portfolio is None:
    st.warning(
        "No portfolio file found at:\n\n"
        f"`{cfg.PAPER_PORTFOLIO_FILE}`\n\n"
        "Run the bot in paper mode first: `python crypto_main.py`"
    )
    st.stop()

with st.spinner("Fetching live prices..."):
    live_prices = fetch_live_prices()

# --- Calculated values ---
cash = portfolio["usdt_balance"]
holdings_val = sum(
    pos["amount"] * live_prices.get(base, pos["avg_cost"])
    for base, pos in portfolio.get("holdings", {}).items()
)
total = cash + holdings_val
start = cfg.PAPER_STARTING_BALANCE
pnl = total - start
pnl_pct = pnl / start * 100

# --- Top metrics row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💵 Cash", f"${cash:,.2f}")
c2.metric("📦 Holdings", f"${holdings_val:,.2f}")
c3.metric("💼 Total", f"${total:,.2f}")
c4.metric("📈 P&L", f"${pnl:+,.2f}", delta=f"{pnl_pct:+.2f}%")
c5.metric("🔢 Trades", len(portfolio.get("trade_log", [])))

st.divider()

# --- Holdings + Portfolio Value side by side ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Open Positions")
    holdings = portfolio.get("holdings", {})
    if holdings:
        rows = []
        for base, pos in holdings.items():
            price = live_prices.get(base, pos["avg_cost"])
            value = pos["amount"] * price
            pnl_pos = value - pos["amount"] * pos["avg_cost"]
            pnl_pct_pos = pnl_pos / (pos["amount"] * pos["avg_cost"]) * 100
            rows.append({
                "Asset": base,
                "Amount": f"{pos['amount']:.6f}",
                "Avg Cost": f"${pos['avg_cost']:,.4f}",
                "Price Now": f"${price:,.4f}",
                "Value": f"${value:,.2f}",
                "P&L": f"${pnl_pos:+,.2f}",
                "P&L %": f"{pnl_pct_pos:+.1f}%",
            })
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True
        )
    else:
        st.info("No open positions.")

with col_right:
    st.subheader("Portfolio Value Over Time")
    trades = portfolio.get("trade_log", [])
    if trades:
        # Build a simple equity curve from trade log
        eq_data = []
        running_cash = start
        running_holdings = {}
        for t in trades:
            base = t["symbol"].split("/")[0]
            if t["side"] == "BUY":
                amt = t["amount"]
                spent = amt * t["price"]
                running_cash -= spent
                running_holdings[base] = running_holdings.get(base, 0) + amt
            else:
                amt = t["amount"]
                running_cash += amt * t["price"]
                running_holdings[base] = max(
                    0, running_holdings.get(base, 0) - amt
                )
            est_val = running_cash + sum(
                a * t["price"] for b, a in running_holdings.items()
            )
            eq_data.append({
                "time": t["timestamp"][:16].replace("T", " "),
                "value": est_val,
            })
        eq_df = pd.DataFrame(eq_data)
        eq_df["value"] = pd.to_numeric(eq_df["value"])
        st.line_chart(eq_df.set_index("time")["value"])
    else:
        st.info("Trade history is empty — equity curve will appear after first trades.")

st.divider()

# --- Trade history table ---
st.subheader("Trade History")
if trades:
    df_trades = pd.DataFrame(trades)
    df_trades["time"] = (
        pd.to_datetime(df_trades["timestamp"])
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    df_trades["price_fmt"] = df_trades["price"].apply(lambda x: f"${x:,.4f}")
    df_trades["amount_fmt"] = df_trades["amount"].apply(lambda x: f"{x:.6f}")
    df_trades["pnl_fmt"] = df_trades.get("pnl", pd.Series(dtype=float)).apply(
        lambda x: f"${x:+,.2f}" if pd.notna(x) else "—"
    )

    display_cols = {
        "time": "Time",
        "symbol": "Symbol",
        "side": "Side",
        "price_fmt": "Price",
        "amount_fmt": "Amount",
        "pnl_fmt": "P&L",
    }
    st.dataframe(
        df_trades[list(display_cols.keys())]
        .rename(columns=display_cols)
        .iloc[::-1]
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    # Trade count by symbol
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Trades by Symbol")
        counts = df_trades["symbol"].value_counts().rename_axis("Symbol").reset_index(name="Count")
        st.bar_chart(counts.set_index("Symbol"))
    with col_b:
        st.subheader("Buy vs Sell")
        sides = df_trades["side"].value_counts().rename_axis("Side").reset_index(name="Count")
        st.bar_chart(sides.set_index("Side"))
else:
    st.info("No trades yet.")

st.divider()
st.caption(
    "Data from `paper_portfolio.json` · Prices from TradingView · "
    "Not financial advice."
)
