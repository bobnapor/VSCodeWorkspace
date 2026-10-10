# =============================================================================
# crypto_dashboard.py — Flask Web Dashboard
# =============================================================================
# Run:  python crypto_dashboard.py
# Then open your browser to:  http://localhost:5000
#
# Shows live portfolio, P&L, open positions, equity curve, trade history.
# Auto-refreshes every 60 seconds.
# =============================================================================

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template_string, request, Response
import crypto_config as cfg

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_portfolio():
    path = Path(cfg.PAPER_PORTFOLIO_FILE)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fetch_prices():
    prices = {}
    try:
        from crypto_paper import fetch_ohlcv_tradingview
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


def _build_data():
    portfolio = _load_portfolio()
    if not portfolio:
        return None
    prices = _fetch_prices()

    cash = portfolio["usdt_balance"]
    holdings_val = sum(
        pos["amount"] * prices.get(base, pos["avg_cost"])
        for base, pos in portfolio.get("holdings", {}).items()
    )
    total = cash + holdings_val
    start = cfg.PAPER_STARTING_BALANCE
    pnl = total - start
    pnl_pct = pnl / start * 100

    positions = []
    for base, pos in portfolio.get("holdings", {}).items():
        price = prices.get(base, pos["avg_cost"])
        value = pos["amount"] * price
        pos_pnl = value - pos["amount"] * pos["avg_cost"]
        pos_pct = pos_pnl / (pos["amount"] * pos["avg_cost"]) * 100
        positions.append({
            "asset": base,
            "amount": "{:.6f}".format(pos["amount"]),
            "avg_cost": "${:,.4f}".format(pos["avg_cost"]),
            "price": "${:,.4f}".format(price),
            "value": "${:,.2f}".format(value),
            "pnl": "${:+,.2f}".format(pos_pnl),
            "pnl_pct": "{:+.1f}%".format(pos_pct),
            "pnl_positive": pos_pnl >= 0,
        })

    # Trade history (most recent first)
    trades = list(reversed(portfolio.get("trade_log", [])))
    trade_rows = []
    for t in trades:
        reasons  = t.get("reasons", {})
        buy_sigs  = reasons.get("buy_signals", [])
        sell_sigs = reasons.get("sell_signals", [])
        sig_parts = buy_sigs + sell_sigs
        sig_summary = ", ".join(sig_parts) if sig_parts else "—"
        if len(sig_summary) > 55:
            sig_summary = sig_summary[:52] + "..."
        trend_str   = reasons.get("trend", "")
        funding_str = reasons.get("funding", "")
        oi_str      = reasons.get("oi", "")
        atr_str     = reasons.get("atr", "")
        trade_rows.append({
            "time":        t.get("timestamp", "")[:16].replace("T", " "),
            "symbol":      t["symbol"],
            "side":        t["side"],
            "price":       "${:,.4f}".format(t["price"]),
            "amount":      "{:.6f}".format(t["amount"]),
            "pnl":         "${:+,.2f}".format(t["pnl"]) if "pnl" in t else "\u2014",
            "is_buy":      t["side"] == "BUY",
            "sig_summary": sig_summary,
            "has_reasons": bool(reasons),
            "trend":       trend_str,
            "funding":     funding_str,
            "oi":          oi_str,
            "atr":         atr_str,
            "count":       reasons.get("count", ""),
        })

    # Equity curve from trade log
    eq_points = []
    running_cash = start
    running_holdings = {}
    for t in portfolio.get("trade_log", []):
        base = t["symbol"].split("/")[0]
        if t["side"] == "BUY":
            running_cash -= t["amount"] * t["price"]
            running_holdings[base] = running_holdings.get(base, 0) + t["amount"]
        else:
            running_cash += t["amount"] * t["price"]
            running_holdings[base] = max(0, running_holdings.get(base, 0) - t["amount"])
        est = running_cash + sum(a * t["price"] for a in running_holdings.values())
        eq_points.append({
            "t": t.get("timestamp", "")[:16].replace("T", " "),
            "v": round(est, 2),
        })

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cash": "${:,.2f}".format(cash),
        "holdings_val": "${:,.2f}".format(holdings_val),
        "total": "${:,.2f}".format(total),
        "pnl": "${:+,.2f}".format(pnl),
        "pnl_pct": "{:+.2f}%".format(pnl_pct),
        "pnl_positive": pnl >= 0,
        "n_trades": len(portfolio.get("trade_log", [])),
        "positions": positions,
        "trades": trade_rows,
        "eq_labels": [p["t"] for p in eq_points],
        "eq_values": [p["v"] for p in eq_points],
        "mode": "PAPER TRADING" if cfg.PAPER_TRADE else "LIVE",
        "timeframe": cfg.TIMEFRAME,
        "symbols": ", ".join(cfg.SYMBOLS),
        "interval": cfg.CHECK_INTERVAL_MINUTES,
        "trend_filter": getattr(cfg, "TREND_FILTER_ENABLED", False),
        "start_balance": "${:,.2f}".format(start),
    }


# ---------------------------------------------------------------------------
# Override and log helpers
# ---------------------------------------------------------------------------

def _read_overrides():
    """Load bot_overrides.json. Returns empty dict if absent or malformed."""
    path = Path(getattr(cfg, "BOT_OVERRIDES_FILE", ""))
    if not str(path) or not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_override(key, value):
    """Set one key in bot_overrides.json, preserving all other keys."""
    path_str = getattr(cfg, "BOT_OVERRIDES_FILE", "")
    if not path_str:
        return
    path = Path(path_str)
    overrides = _read_overrides()
    overrides[key] = value
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2)


def _get_log_tail(n=30):
    """Return the last n lines of the bot log file as a list of strings."""
    try:
        log_path = Path(cfg.LOG_FILE)
        if not log_path.exists():
            return ["(log file not found)"]
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip() for ln in lines[-n:]]
    except Exception as exc:
        return ["Error reading log: {}".format(exc)]


# ---------------------------------------------------------------------------
# HTML template (single-file, no external dependencies beyond Chart.js CDN)
# ---------------------------------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>&#x1F916; Crypto Signal Bot</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e0e0e0; padding: 20px; }
    h1 { font-size: 1.5rem; margin-bottom: 4px; }
    .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 24px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
               gap: 12px; margin-bottom: 24px; }
    .card { background: #1e2130; border-radius: 10px; padding: 16px; }
    .card .label { font-size: 0.75rem; color: #888; text-transform: uppercase;
                   letter-spacing: .05em; margin-bottom: 6px; }
    .card .value { font-size: 1.4rem; font-weight: 600; }
    .card .delta { font-size: 0.85rem; margin-top: 4px; }
    .green { color: #4ade80; }
    .red   { color: #f87171; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
             margin-bottom: 24px; }
    @media(max-width:700px){ .grid2 { grid-template-columns: 1fr; } }
    .panel { background: #1e2130; border-radius: 10px; padding: 16px;
             margin-bottom: 24px; }
    .panel h2 { font-size: 1rem; margin-bottom: 14px; color: #ccc; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { text-align: left; padding: 6px 8px; color: #888;
         border-bottom: 1px solid #2a2d3e; font-weight: 500; }
    td { padding: 7px 8px; border-bottom: 1px solid #1a1d2e; }
    tr:last-child td { border-bottom: none; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-size: 0.78rem; font-weight: 600; }
    .badge-buy  { background: #14532d; color: #4ade80; }
    .badge-sell { background: #450a0a; color: #f87171; }
    .config-bar { background: #1e2130; border-radius: 10px; padding: 12px 16px;
                  font-size: 0.82rem; color: #888; margin-bottom: 24px;
                  display: flex; flex-wrap: wrap; gap: 16px; }
    .config-bar span b { color: #ccc; }
    .refresh-btn { float: right; background: #2563eb; color: #fff; border: none;
                   padding: 6px 16px; border-radius: 6px; cursor: pointer;
                   font-size: 0.85rem; }
    .refresh-btn:hover { background: #1d4ed8; }
    .empty { color: #555; font-style: italic; padding: 12px 0; }
    canvas { max-height: 220px; }
    /* ── Controls panel ─────────────────────────────────────────────────── */
    .ctrl-panel { background:#1e2130; border-radius:10px; padding:14px 16px;
                  margin-bottom:24px; }
    .ctrl-row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .ctrl-label { font-size:0.78rem; color:#888; text-transform:uppercase;
                  letter-spacing:.05em; }
    .btn { background:#374151; color:#e0e0e0; border:none; padding:7px 14px;
           border-radius:6px; cursor:pointer; font-size:0.83rem; font-weight:500;
           transition:background .15s; white-space:nowrap; text-decoration:none;
           display:inline-block; }
    .btn:hover { background:#4b5563; }
    .btn-danger { background:#7f1d1d; color:#fca5a5; }
    .btn-danger:hover { background:#991b1b; }
    .btn-warn { background:#78350f; color:#fcd34d; }
    .btn-warn:hover { background:#92400e; }
    .btn-on { background:#14532d; color:#4ade80; }
    .btn-on:hover { background:#166534; }
    .btn-paused { background:#581c87; color:#d8b4fe; }
    .btn-paused:hover { background:#6b21a8; }
    /* ── Confirm modal ──────────────────────────────────────────────────── */
    #modalOverlay { display:none; position:fixed; inset:0;
                    background:rgba(0,0,0,.65); z-index:999;
                    align-items:center; justify-content:center; }
    .modal-box { background:#1e2130; border-radius:12px; padding:28px 32px;
                 max-width:420px; width:90%; border:1px solid #2a2d3e; }
    /* ── Log viewer ─────────────────────────────────────────────────────── */
    .log-tail { background:#0a0c12; border-radius:8px; padding:12px;
                font-family:monospace; font-size:0.75rem; color:#9ca3af;
                max-height:220px; overflow-y:auto; white-space:pre-wrap;
                word-break:break-all; margin:0; }
    /* ── Tab navigation ─────────────────────────────────────────────────── */
    .tab-bar { display:flex; gap:4px; margin-bottom:20px; flex-wrap:wrap; }
    .tab-btn { background:#1e2130; color:#888; border:none; padding:10px 22px;
               border-radius:8px 8px 0 0; cursor:pointer; font-size:0.88rem;
               font-weight:500; transition:background .15s, color .15s; }
    .tab-btn.active { background:#2563eb; color:#fff; }
    .tab-btn:hover:not(.active) { background:#2a2d3e; color:#ccc; }
    .tab-pane { display:none; }
    .tab-pane.active { display:block; }
    /* ── Risk panel ─────────────────────────────────────────────────────── */
    .risk-flag { padding:9px 14px; border-radius:7px; font-size:0.85rem;
                 margin-bottom:8px; }
    .risk-ok     { background:#14532d22; color:#4ade80; border:1px solid #14532d; }
    .risk-warn   { background:#78350f22; color:#fcd34d; border:1px solid #78350f; }
    .risk-danger { background:#7f1d1d22; color:#fca5a5; border:1px solid #7f1d1d; }
    .rm-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
               gap:10px; margin-bottom:20px; }
    .rm-card { background:#0f1117; border-radius:8px; padding:12px; text-align:center; }
    .rm-label { font-size:0.72rem; color:#666; text-transform:uppercase;
                letter-spacing:.05em; margin-bottom:4px; }
    .rm-value { font-size:1.15rem; font-weight:600; }
    .news-item { padding:10px 0; border-bottom:1px solid #1a1d2e; }
    .news-item:last-child { border-bottom:none; }
    .news-title { color:#93c5fd; text-decoration:none; font-size:0.88rem;
                  line-height:1.4; display:block; }
    .news-title:hover { text-decoration:underline; }
    .news-meta { font-size:0.75rem; color:#666; margin-top:3px; }
    .news-tags { font-size:0.72rem; color:#4b5563; margin-top:2px; }
    /* ── Chart tab ──────────────────────────────────────────────────────── */
    .sym-bar { display:flex; gap:6px; margin-bottom:12px; flex-wrap:wrap; }
    .sym-btn { background:#1e2130; color:#888; border:none; padding:6px 14px;
               border-radius:6px; cursor:pointer; font-size:0.82rem; }
    .sym-btn.active { background:#0e7490; color:#fff; }
    #chartContainer { background:#1e2130; border-radius:10px; min-height:420px; }
    #chartLoading { color:#888; padding:40px; text-align:center; font-size:0.9rem; }
    /* ── Funding table tweaks ───────────────────────────────────────────── */
    .fund-signal-BEARISH { color:#f87171; }
    .fund-signal-BULLISH { color:#4ade80; }
    .fund-signal-NEUTRAL { color:#888; }
    /* ── Reasons tooltip row ─────────────────────────────────────────────── */
    .reasons-row td { background:#0a0c12; font-size:0.78rem; color:#9ca3af;
                      padding:6px 12px; }
    .reasons-row { display:none; }
    .reasons-row.open { display:table-row; }
    .reasons-toggle { cursor:pointer; font-size:0.8rem; color:#60a5fa;
                      background:none; border:none; padding:0; }
  </style>
</head>
<body>

{% if not data %}
  <h1>&#x1F916; Crypto Signal Bot</h1>
  <p style="color:#f87171;margin-top:16px">
    No portfolio file found. Run the bot first with:<br><br>
    <code style="background:#1e2130;padding:4px 8px;border-radius:4px">
      python crypto_main.py
    </code>
  </p>
{% else %}

<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
  <h1>&#x1F916; Crypto Signal Bot Dashboard</h1>
  <button class="refresh-btn" onclick="location.reload()">&#x27F3; Refresh</button>
</div>
<div class="subtitle">
  Last updated: {{ data.as_of }} &nbsp;&middot;&nbsp; Auto-refreshes every 60s
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('portfolio',this)">&#x1F4BC; Portfolio</button>
  <button class="tab-btn" onclick="showTab('charts',this)">&#x1F4C8; Charts</button>
  <button class="tab-btn" onclick="showTab('risk',this)">&#x1F6E1;&#xFE0F; Risk</button>
  <button class="tab-btn" onclick="showTab('funding',this)">&#x1F4B0; Funding &amp; OI</button>
</div>

<div class="config-bar">
  <span><b>Mode:</b> {{ data.mode }}</span>
  <span><b>Symbols:</b> {{ data.symbols }}</span>
  <span><b>Timeframe:</b> {{ data.timeframe }}</span>
  <span><b>Check every:</b> {{ data.interval }}m</span>
  <span><b>Trend filter:</b> {{ data.trend_filter }}</span>
  <span><b>Start balance:</b> {{ data.start_balance }}</span>
</div>

<div class="ctrl-panel">
  <div class="ctrl-row">
    <span class="ctrl-label">&#x1F527; Controls</span>
    <button class="btn" id="btnPause" onclick="togglePause()">&#x23F8;&#xFE0F; Pause Bot</button>
    <button class="btn" id="btnTrend" onclick="toggleTrend()">&#x1F4C8; Trend Filter: <span id="trendState">...</span></button>
    <button class="btn btn-danger" onclick="confirmAction('reset')">&#x26A0;&#xFE0F; Reset Portfolio</button>
    <button class="btn btn-warn" onclick="confirmAction('clearlog')">&#x1F5D1;&#xFE0F; Clear Log</button>
    <a href="/api/export-csv" class="btn">&#x1F4E5; Export CSV</a>
  </div>
  <div id="ctrlStatus" style="font-size:0.82rem;color:#4ade80;margin-top:8px;min-height:18px"></div>
</div>

<div id="modalOverlay">
  <div class="modal-box">
    <h3 id="modalTitle" style="margin-bottom:10px;font-size:1.05rem"></h3>
    <p id="modalBody" style="color:#9ca3af;font-size:0.88rem;margin-bottom:20px;line-height:1.5"></p>
    <div style="display:flex;gap:10px;justify-content:flex-end">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-danger" id="modalConfirmBtn">Confirm</button>
    </div>
  </div>
</div>

<div id="tab-portfolio" class="tab-pane active">
<div class="metrics">
  <div class="card">
    <div class="label">&#x1F4B5; Cash</div>
    <div class="value">{{ data.cash }}</div>
  </div>
  <div class="card">
    <div class="label">&#x1F4E6; Holdings</div>
    <div class="value">{{ data.holdings_val }}</div>
  </div>
  <div class="card">
    <div class="label">&#x1F4BC; Total Value</div>
    <div class="value">{{ data.total }}</div>
  </div>
  <div class="card">
    <div class="label">&#x1F4C8; Total P&amp;L</div>
    <div class="value {{ 'green' if data.pnl_positive else 'red' }}">{{ data.pnl }}</div>
    <div class="delta {{ 'green' if data.pnl_positive else 'red' }}">{{ data.pnl_pct }}</div>
  </div>
  <div class="card">
    <div class="label">&#x1F522; Trades</div>
    <div class="value">{{ data.n_trades }}</div>
  </div>
</div>

<div class="grid2">
  <div class="panel">
    <h2>Open Positions</h2>
    {% if data.positions %}
    <table>
      <tr>
        <th>Asset</th><th>Amount</th><th>Avg Cost</th>
        <th>Price</th><th>Value</th><th>P&amp;L</th>
      </tr>
      {% for p in data.positions %}
      <tr>
        <td><b>{{ p.asset }}</b></td>
        <td>{{ p.amount }}</td>
        <td>{{ p.avg_cost }}</td>
        <td>{{ p.price }}</td>
        <td>{{ p.value }}</td>
        <td class="{{ 'green' if p.pnl_positive else 'red' }}">
          {{ p.pnl }} ({{ p.pnl_pct }})
        </td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p class="empty">No open positions.</p>
    {% endif %}
  </div>

  <div class="panel">
    <h2>Portfolio Equity Curve</h2>
    {% if data.eq_labels %}
    <canvas id="eqChart"></canvas>
    {% else %}
    <p class="empty">No trades yet &mdash; chart will appear after first trades.</p>
    {% endif %}
  </div>
</div>

<div class="panel">
  <h2>Trade History (most recent first)</h2>
  {% if data.trades %}
  <table>
    <tr>
      <th>Time</th><th>Symbol</th><th>Side</th>
      <th>Price</th><th>Amount</th><th>Signals</th><th>P&amp;L</th>
    </tr>
    {% for t in data.trades %}
    <tr>
      <td>{{ t.time }}</td>
      <td>{{ t.symbol }}</td>
      <td><span class="badge {{ 'badge-buy' if t.is_buy else 'badge-sell' }}">
        {{ t.side }}
      </span></td>
      <td>{{ t.price }}</td>
      <td>{{ t.amount }}</td>
      <td style="max-width:200px;font-size:0.78rem;color:#9ca3af">
        {{ t.sig_summary }}
        {% if t.has_reasons %}
        <button class="reasons-toggle" onclick="toggleReasons(this)">&#x2139;&#xFE0F;</button>
        {% endif %}
      </td>
      <td class="{{ 'green' if not t.is_buy else '' }}">{{ t.pnl }}</td>
    </tr>
    {% if t.has_reasons %}
    <tr class="reasons-row">
      <td colspan="7">
        <b>Trade rationale</b><br>
        {% if t.trend %}<span style="color:#94a3b8">Trend:</span> {{ t.trend }}<br>{% endif %}
        {% if t.count %}<span style="color:#94a3b8">Signal count:</span> {{ t.count }}<br>{% endif %}
        {% if t.funding %}<span style="color:#94a3b8">Funding:</span> {{ t.funding }}<br>{% endif %}
        {% if t.oi %}<span style="color:#94a3b8">Open Interest:</span> {{ t.oi }}<br>{% endif %}
        {% if t.atr %}<span style="color:#94a3b8">Volatility:</span> {{ t.atr }}{% endif %}
      </td>
    </tr>
    {% endif %}
    {% endfor %}
  </table>
  {% else %}
  <p class="empty">No trades yet.</p>
  {% endif %}
</div>

<div class="panel">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
    <h2>&#x1F4CB; Bot Log (last 30 lines)</h2>
    <button class="btn" style="font-size:0.78rem;padding:4px 10px" onclick="fetchLogTail()">&#x27F3; Refresh</button>
  </div>
  <pre id="logTail" class="log-tail">(loading...)</pre>
</div>
</div><!-- end tab-portfolio -->

<!-- ── Tab 2: Charts ─────────────────────────────────────────────────── -->
<div id="tab-charts" class="tab-pane">
  <div class="panel">
    <h2>&#x1F4C8; Price Charts</h2>
    <div class="sym-bar" id="symBar">
      {% for sym in data.symbols.split(', ') %}
      <button class="sym-btn {% if loop.first %}active{% endif %}"
              onclick="selectSym(this,'{{ sym }}')">{{ sym }}</button>
      {% endfor %}
    </div>
    <div id="chartContainer">
      <div id="chartLoading">Select a symbol above to load its chart.</div>
    </div>
  </div>
</div><!-- end tab-charts -->

<!-- ── Tab 3: Risk Management ────────────────────────────────────────── -->
<div id="tab-risk" class="tab-pane">
  <div class="panel">
    <h2>&#x1F6E1;&#xFE0F; Risk Flags</h2>
    <div id="riskFlags"><span style="color:#888">Loading...</span></div>
  </div>
  <div class="grid2">
    <div class="panel">
      <h2>&#x1F4CA; Portfolio Risk Metrics</h2>
      <div class="rm-grid" id="riskMetrics"></div>
    </div>
    <div class="panel">
      <h2>&#x1F627; Fear &amp; Greed Index</h2>
      <div id="fearGreed" style="text-align:center;padding:10px"><span style="color:#888">Loading...</span></div>
    </div>
  </div>
  <div class="grid2">
    <div class="panel">
      <h2>&#x26D3;&#xFE0F; Chain TVL (DeFiLlama)</h2>
      <div id="chainTvl"><span style="color:#888">Loading...</span></div>
    </div>
    <div class="panel">
      <h2>&#x1F4B5; Stablecoin Supply</h2>
      <div id="stablecoins" style="padding:4px"><span style="color:#888">Loading...</span></div>
      <p style="font-size:0.72rem;color:#555;margin-top:10px">
        Rising supply = capital entering crypto (bullish macro).<br>
        Falling supply = capital leaving (bearish macro).
      </p>
    </div>
  </div>
  <div class="panel">
    <h2>&#x1F4F0; Crypto News</h2>
    <div id="newsPanel"><span style="color:#888">Loading...</span></div>
  </div>
</div><!-- end tab-risk -->

<!-- ── Tab 4: Funding & OI ───────────────────────────────────────────── -->
<div id="tab-funding" class="tab-pane">
  <div class="panel">
    <h2>&#x1F4B0; Funding Rates</h2>
    <p style="font-size:0.78rem;color:#666;margin-bottom:12px">
      Positive funding → longs pay shorts (crowded longs, mean-reversion risk).<br>
      Negative funding → shorts pay longs (potential short squeeze).<br>
      Extreme positive (&gt;+0.1%/8h) suppresses BUY signals when
      <code>OI_FUNDING_SIGNALS_ENABLED = True</code>.
    </p>
    <div id="fundingTable"><span style="color:#888">Loading...</span></div>
  </div>
  <div class="panel">
    <h2>&#x1F4C8; Open Interest</h2>
    <p style="font-size:0.78rem;color:#666;margin-bottom:12px">
      Rising OI + rising price → trend confirmation (strong momentum).<br>
      Rising OI + falling price → new shorts entering (bearish pressure).<br>
      Falling OI → position unwinding (trend may be exhausting).
    </p>
    <div id="oiTable"><span style="color:#888">Loading...</span></div>
    <p id="fundingAs" style="font-size:0.72rem;color:#555;margin-top:12px"></p>
  </div>
</div><!-- end tab-funding -->

<script>
  {% if data.eq_labels %}
  const ctx = document.getElementById('eqChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: {{ data.eq_labels | tojson }},
      datasets: [{
        label: 'Portfolio Value ($)',
        data: {{ data.eq_values | tojson }},
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74,222,128,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#888', callback: function(v){ return '$' + v.toLocaleString(); } },
          grid: { color: '#2a2d3e' }
        }
      }
    }
  });
  {% endif %}

  // ── Controls JS ──────────────────────────────────────────────────────────
  var _ov = {};
  var _cfgTrend = {{ 'true' if data.trend_filter else 'false' }};
  var _startBal = "{{ data.start_balance }}";

  function loadOverrides() {
    fetch('/api/overrides')
      .then(function(r){ return r.json(); })
      .then(function(d){ _ov = d || {}; renderBtns(); })
      .catch(function(){});
  }

  function renderBtns() {
    var paused = !!_ov.paused;
    var tf = ('trend_filter_enabled' in _ov) ? !!_ov.trend_filter_enabled : _cfgTrend;
    var bp = document.getElementById('btnPause');
    if (bp) {
      bp.innerHTML = paused ? '&#x25B6;&#xFE0F; Resume Bot' : '&#x23F8;&#xFE0F; Pause Bot';
      bp.className  = paused ? 'btn btn-paused' : 'btn';
    }
    var ts = document.getElementById('trendState');
    var bt = document.getElementById('btnTrend');
    if (ts) ts.textContent = tf ? 'ON' : 'OFF';
    if (bt) bt.className = tf ? 'btn btn-on' : 'btn';
  }

  function togglePause() { postOv('paused', !_ov.paused); }

  function toggleTrend() {
    var cur = ('trend_filter_enabled' in _ov) ? _ov.trend_filter_enabled : _cfgTrend;
    postOv('trend_filter_enabled', !cur);
  }

  function postOv(key, val) {
    fetch('/api/override/' + key, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({value: val})
    }).then(function(r){ return r.json(); })
      .then(function(d){ _ov[key] = val; renderBtns(); showStatus(d.message || 'Done.', true); })
      .catch(function(e){ showStatus(String(e), false); });
  }

  // Confirm modal
  var _pendingFn = null;
  function openModal(title, body, fn) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').textContent = body;
    _pendingFn = fn;
    document.getElementById('modalConfirmBtn').onclick = function(){
      closeModal(); if (_pendingFn) _pendingFn();
    };
    document.getElementById('modalOverlay').style.display = 'flex';
  }
  function closeModal() {
    document.getElementById('modalOverlay').style.display = 'none';
  }

  function confirmAction(act) {
    if (act === 'reset') {
      openModal(
        'Reset Paper Portfolio',
        'This will erase ALL paper trades and open positions, resetting the balance to '
          + _startBal + '. The previous data will be archived. This cannot be undone.',
        function(){
          fetch('/api/reset-portfolio', {method:'POST'})
            .then(function(r){ return r.json(); })
            .then(function(d){
              showStatus(d.message, d.ok);
              if (d.ok) setTimeout(function(){ location.reload(); }, 1800);
            });
        }
      );
    } else if (act === 'clearlog') {
      openModal(
        'Clear Bot Log',
        'This will truncate the bot log file. The bot will continue running normally.',
        function(){
          fetch('/api/clear-log', {method:'POST'})
            .then(function(r){ return r.json(); })
            .then(function(d){ showStatus(d.message, d.ok); if (d.ok) fetchLogTail(); });
        }
      );
    }
  }

  function showStatus(msg, ok) {
    var el = document.getElementById('ctrlStatus');
    if (!el) return;
    el.textContent = msg;
    el.style.color = ok ? '#4ade80' : '#f87171';
    setTimeout(function(){ if (el.textContent === msg) el.textContent = ''; }, 4000);
  }

  function fetchLogTail() {
    fetch('/api/log-tail')
      .then(function(r){ return r.json(); })
      .then(function(d){
        var el = document.getElementById('logTail');
        if (el) {
          el.textContent = (d.lines && d.lines.length) ? d.lines.join('\\n') : '(log is empty)';
          el.scrollTop = el.scrollHeight;
        }
      }).catch(function(){});
  }

  function toggleReasons(btn) {
    var tr = btn.closest('tr').nextElementSibling;
    if (tr && tr.classList.contains('reasons-row')) {
      tr.classList.toggle('open');
    }
  }

  // ── Tab switching ─────────────────────────────────────────────────────────
  var _chartsLoaded  = false;
  var _riskLoaded    = false;
  var _fundingLoaded = false;
  var _activeChart   = null;

  function showTab(name, el) {
    document.querySelectorAll('.tab-pane').forEach(function(p){ p.classList.remove('active'); });
    document.querySelectorAll('.tab-btn').forEach(function(b){ b.classList.remove('active'); });
    document.getElementById('tab-' + name).classList.add('active');
    if (el) el.classList.add('active');
    if (name === 'charts'  && !_chartsLoaded)  loadCharts();
    if (name === 'risk'    && !_riskLoaded)    loadRisk();
    if (name === 'funding' && !_fundingLoaded) loadFunding();
  }

  // ── Charts tab ────────────────────────────────────────────────────────────
  function loadCharts() {
    _chartsLoaded = true;
    var firstBtn = document.querySelector('#symBar .sym-btn');
    if (firstBtn) {
      var sym = firstBtn.getAttribute('onclick').match(/'([^']+)'\s*\)/);
      if (sym) loadChart(sym[1]);
    }
  }

  function selectSym(btn, sym) {
    document.querySelectorAll('.sym-btn').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    loadChart(sym);
  }

  function loadChart(sym) {
    var container = document.getElementById('chartContainer');
    container.innerHTML = '<div id="chartLoading">Loading ' + sym + ' ...</div>';
    if (_activeChart) { try { _activeChart.remove(); } catch(e){} _activeChart = null; }

    fetch('/api/chart-data/' + encodeURIComponent(sym))
      .then(function(r){ return r.json(); })
      .then(function(d){
        if (d.error) {
          container.innerHTML = '<div style="color:#f87171;padding:20px">Error: ' + d.error + '</div>';
          return;
        }
        container.innerHTML = '';
        var chart = LightweightCharts.createChart(container, {
          width:  container.offsetWidth,
          height: 420,
          layout:    { background: { type: 'solid', color: '#1e2130' }, textColor: '#888' },
          grid:      { vertLines: { color: '#2a2d3e' }, horzLines: { color: '#2a2d3e' } },
          crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
          rightPriceScale: { borderColor: '#2a2d3e' },
          timeScale: { borderColor: '#2a2d3e', timeVisible: true, secondsVisible: false },
        });
        _activeChart = chart;

        var candles = chart.addCandlestickSeries({
          upColor: '#4ade80', downColor: '#f87171',
          borderUpColor: '#4ade80', borderDownColor: '#f87171',
          wickUpColor: '#4ade80', wickDownColor: '#f87171',
        });
        candles.setData(d.candles);
        if (d.markers && d.markers.length) candles.setMarkers(d.markers);

        var volSeries = chart.addHistogramSeries({
          priceFormat: { type: 'volume' },
          priceScaleId: 'vol',
          scaleMargins: { top: 0.85, bottom: 0 },
        });
        volSeries.setData(d.volumes);
        chart.timeScale().fitContent();

        window.addEventListener('resize', function(){
          chart.resize(container.offsetWidth, 420);
        });
      })
      .catch(function(e){
        container.innerHTML = '<div style="color:#f87171;padding:20px">Failed: ' + e + '</div>';
      });
  }

  // ── Risk tab ──────────────────────────────────────────────────────────────
  function loadRisk() {
    _riskLoaded = true;
    fetch('/api/risk')
      .then(function(r){ return r.json(); })
      .then(renderRisk)
      .catch(function(e){
        document.getElementById('riskFlags').innerHTML = '<span style="color:#f87171">Error: ' + e + '</span>';
      });
  }

  function renderRisk(d) {
    // Risk flags
    var m = d.metrics || {};
    var flagsHtml = (m.flags || []).map(function(f){
      var cls  = f.level === 'danger' ? 'risk-danger' : (f.level === 'warn' ? 'risk-warn' : 'risk-ok');
      var icon = f.level === 'danger' ? '&#x1F6A8;' : (f.level === 'warn' ? '&#x26A0;&#xFE0F;' : '&#x2705;');
      return '<div class="risk-flag ' + cls + '">' + icon + ' ' + f.text + '</div>';
    }).join('');
    document.getElementById('riskFlags').innerHTML = flagsHtml || '<span style="color:#888">—</span>';

    // Metrics grid
    var mg = [
      ['Max Drawdown', m.max_dd_fmt,       m.max_dd > 15 ? 'red' : ''],
      ['Current DD',   m.current_dd_fmt,    m.current_dd > 10 ? 'red' : ''],
      ['Win Rate',     m.win_rate_fmt,      ''],
      ['Profit Factor',m.profit_factor_fmt, ''],
      ['Avg Win',      m.avg_win_fmt,       'green'],
      ['Avg Loss',     m.avg_loss_fmt,      'red'],
      ['Exposure',     m.exposure_fmt,      ''],
      ['Total P&L',    m.pnl_pct_fmt,       m.pnl_positive ? 'green' : 'red'],
      ['Closed Trades',String(m.sell_trades || 0), ''],
    ];
    document.getElementById('riskMetrics').innerHTML = mg.map(function(r){
      return '<div class="rm-card"><div class="rm-label">' + r[0] + '</div>'
           + '<div class="rm-value ' + r[2] + '">' + (r[1] || '—') + '</div></div>';
    }).join('');

    // Fear & Greed
    var fg = d.fear_greed;
    if (fg) {
      var bars = (fg.history || []).map(function(h){
        var c = h.value > 60 ? '#ef4444' : (h.value < 40 ? '#22c55e' : '#eab308');
        var ht = Math.max(6, Math.round(h.value / 2));
        return '<div style="display:flex;flex-direction:column;align-items:center;gap:2px">'
             + '<div style="width:18px;height:' + ht + 'px;background:' + c + ';border-radius:2px;opacity:.8"></div>'
             + '<div style="font-size:.6rem;color:#555">' + h.date + '</div></div>';
      }).join('');
      document.getElementById('fearGreed').innerHTML =
        '<div style="font-size:2.8rem;font-weight:700;color:' + fg.color + '">' + fg.value + '</div>'
        + '<div style="color:#888;font-size:.9rem">' + fg.label + '</div>'
        + '<div style="margin-top:10px;display:flex;gap:3px;align-items:flex-end;justify-content:center">' + bars + '</div>';
    }

    // Chain TVL
    var tvl = d.chain_tvl || [];
    if (tvl.length) {
      var rows = tvl.map(function(c){
        return '<tr><td><b>' + c.name + '</b></td><td>' + c.tvl_fmt + '</td>'
             + '<td class="' + (c.d1_pos ? 'green' : 'red') + '">' + c.d1_fmt + '</td>'
             + '<td class="' + (c.d7 >= 0 ? 'green' : 'red') + '">' + c.d7_fmt + '</td></tr>';
      }).join('');
      document.getElementById('chainTvl').innerHTML =
        '<table><tr><th>Chain</th><th>TVL</th><th>24h</th><th>7d</th></tr>' + rows + '</table>';
    }

    // Stablecoins
    var sc = d.stablecoins;
    if (sc) {
      var scRows = (sc.top || []).map(function(s){
        return '<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:.82rem">'
             + '<span>' + s.symbol + '</span><span>' + s.supply_fmt + '</span></div>';
      }).join('');
      document.getElementById('stablecoins').innerHTML =
        '<div style="font-size:1.3rem;font-weight:600;margin-bottom:4px">' + sc.total_fmt + '</div>'
        + '<div style="color:#888;font-size:.78rem;margin-bottom:10px">Total stablecoin supply</div>'
        + scRows;
    }

    // News
    var news = d.news || [];
    if (news.length) {
      document.getElementById('newsPanel').innerHTML = news.map(function(n){
        var tagHtml = n.tags ? '<div class="news-tags">' + n.tags + '</div>' : '';
        return '<div class="news-item"><a href="' + n.url + '" target="_blank" class="news-title">'
             + n.title + '</a><div class="news-meta">' + n.source + ' &middot; ' + n.published + '</div>'
             + tagHtml + '</div>';
      }).join('');
    } else {
      document.getElementById('newsPanel').innerHTML = '<span style="color:#888">No news loaded.</span>';
    }
  }

  // ── Funding & OI tab ──────────────────────────────────────────────────────
  function loadFunding() {
    _fundingLoaded = true;
    fetch('/api/funding')
      .then(function(r){ return r.json(); })
      .then(renderFunding)
      .catch(function(e){
        document.getElementById('fundingTable').innerHTML = '<span style="color:#f87171">Error: ' + e + '</span>';
      });
  }

  function renderFunding(d) {
    var funding = d.funding || {};
    var syms = Object.keys(funding);
    if (syms.length) {
      var rows = syms.map(function(sym){
        var f = funding[sym];
        var sc = 'fund-signal-' + f.signal;
        return '<tr><td><b>' + sym + '</b></td>'
             + '<td class="' + sc + '">' + f.rate_pct + '</td>'
             + '<td>' + f.rate_ann + '</td>'
             + '<td class="' + sc + '">' + f.signal + '</td>'
             + '<td>' + f.next_funding + '</td>'
             + '<td>$' + (f.mark_price || 0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:4}) + '</td></tr>';
      }).join('');
      document.getElementById('fundingTable').innerHTML =
        '<table><tr><th>Symbol</th><th>Rate/8h</th><th>Annualized</th>'
        + '<th>Signal</th><th>Next Funding</th><th>Mark Price</th></tr>' + rows + '</table>';
    } else {
      document.getElementById('fundingTable').innerHTML = '<span style="color:#888">No data (Binance/Bybit public API).</span>';
    }

    var oi = d.oi || {};
    var oiSyms = Object.keys(oi);
    if (oiSyms.length) {
      var oiRows = oiSyms.map(function(sym){
        var o = oi[sym];
        return '<tr><td><b>' + sym + '</b></td><td>' + o.oi_fmt + '</td>'
             + '<td class="' + (o.change_positive ? 'green' : 'red') + '">' + o.change_fmt + '</td></tr>';
      }).join('');
      document.getElementById('oiTable').innerHTML =
        '<table><tr><th>Symbol</th><th>Open Interest</th><th>4h Change</th></tr>' + oiRows + '</table>';
    }

    if (d.as_of) {
      document.getElementById('fundingAs').textContent = 'Data as of ' + d.as_of;
    }
  }

  // ── Init on page load
  loadOverrides();
  fetchLogTail();

  // Auto-refresh every 60 seconds
  setTimeout(function(){ location.reload(); }, 60000);
</script>
{% endif %}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    data = _build_data()
    return render_template_string(HTML, data=data)


@app.route("/api/data")
def api_data():
    """JSON endpoint — useful for debugging or external integrations."""
    return jsonify(_build_data())


@app.route("/api/reset-portfolio", methods=["POST"])
def api_reset_portfolio():
    """Wipe the paper portfolio and reset to starting balance."""
    try:
        from crypto_paper import reset_paper_portfolio
        msg = reset_paper_portfolio()
        return jsonify({"ok": True, "message": msg})
    except Exception as exc:
        return jsonify({"ok": False, "message": "Error: {}".format(exc)}), 500


@app.route("/api/clear-log", methods=["POST"])
def api_clear_log():
    """Truncate the bot log file."""
    try:
        from crypto_paper import clear_bot_log
        msg = clear_bot_log()
        return jsonify({"ok": True, "message": msg})
    except Exception as exc:
        return jsonify({"ok": False, "message": "Error: {}".format(exc)}), 500


@app.route("/api/overrides")
def api_overrides():
    """Return the current bot_overrides.json contents."""
    return jsonify(_read_overrides())


@app.route("/api/override/<key>", methods=["POST"])
def api_set_override(key):
    """Set a single key in bot_overrides.json."""
    allowed = {"paused", "trend_filter_enabled"}
    if key not in allowed:
        return jsonify({"ok": False, "message": "Unknown key: {}".format(key)}), 400
    body = request.get_json(silent=True) or {}
    value = body.get("value")
    try:
        _write_override(key, value)
        labels = {"paused": "Bot", "trend_filter_enabled": "Trend filter"}
        if key == "paused":
            state = "paused" if value else "resumed"
        else:
            state = "ON" if value else "OFF"
        return jsonify({"ok": True, "message": "{} {}.".format(labels[key], state)})
    except Exception as exc:
        return jsonify({"ok": False, "message": "Error: {}".format(exc)}), 500


@app.route("/api/log-tail")
def api_log_tail():
    """Return the last N lines of the bot log file (default 30)."""
    n = request.args.get("n", 30, type=int)
    return jsonify({"lines": _get_log_tail(n)})


@app.route("/api/export-csv")
def api_export_csv():
    """Download the full trade history as a CSV file."""
    portfolio = _load_portfolio()
    if not portfolio:
        return "No portfolio data found.", 404
    trades = portfolio.get("trade_log", [])
    rows = ["timestamp,symbol,side,price,amount,pnl,signals"]
    for t in trades:
        reasons  = t.get("reasons", {})
        buy_sigs  = reasons.get("buy_signals", [])
        sell_sigs = reasons.get("sell_signals", [])
        sig_str   = " | ".join(buy_sigs + sell_sigs).replace(",", ";")
        rows.append("{},{},{},{},{},{},{}".format(
            t.get("timestamp", ""),
            t.get("symbol", ""),
            t.get("side", ""),
            t.get("price", ""),
            t.get("amount", ""),
            t.get("pnl", ""),
            sig_str,
        ))
    return Response(
        "\n".join(rows),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=trade_history.csv"},
    )


@app.route("/api/chart-data/<path:symbol>")
def api_chart_data(symbol):
    """Return OHLCV candles + trade markers for LightweightCharts."""
    import warnings as _w
    try:
        from crypto_paper import fetch_ohlcv_tradingview
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            df = fetch_ohlcv_tradingview(symbol, n_bars_override=200)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    candles = []
    volumes = []
    for ts, row in df.iterrows():
        try:
            t = int(ts.timestamp())
        except Exception:
            t = int(ts) // 1000
        c = float(row["close"])
        o = float(row["open"])
        candles.append({
            "time":  t,
            "open":  round(o, 6),
            "high":  round(float(row["high"]), 6),
            "low":   round(float(row["low"]),  6),
            "close": round(c, 6),
        })
        volumes.append({
            "time":  t,
            "value": round(float(row["volume"]), 2),
            "color": "rgba(74,222,128,0.3)" if c >= o else "rgba(248,113,113,0.3)",
        })

    portfolio = _load_portfolio()
    markers = []
    for t in (portfolio or {}).get("trade_log", []):
        if t.get("symbol") != symbol:
            continue
        try:
            ts_str = t.get("timestamp", "")
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts_str)
            markers.append({
                "time":     int(dt.timestamp()),
                "position": "belowBar" if t["side"] == "BUY" else "aboveBar",
                "color":    "#4ade80" if t["side"] == "BUY" else "#f87171",
                "shape":    "arrowUp" if t["side"] == "BUY" else "arrowDown",
                "text":     "{} @${:.2f}".format(t["side"], t["price"]),
            })
        except Exception:
            pass

    return jsonify({
        "symbol":  symbol,
        "candles": candles,
        "volumes": volumes,
        "markers": sorted(markers, key=lambda x: x["time"]),
    })


@app.route("/api/risk")
def api_risk():
    """Return risk metrics + Fear&Greed + chain TVL + stablecoins + news."""
    try:
        from crypto_market_data import (
            compute_risk_metrics, fetch_fear_greed,
            fetch_chain_tvl, fetch_stablecoin_supply, fetch_news,
        )
        portfolio = _load_portfolio()
        if not portfolio:
            return jsonify({"error": "No portfolio data"}), 404
        prices  = _fetch_prices()
        metrics = compute_risk_metrics(portfolio, prices)
        return jsonify({
            "metrics":     metrics,
            "fear_greed":  fetch_fear_greed(),
            "chain_tvl":   fetch_chain_tvl(),
            "stablecoins": fetch_stablecoin_supply(),
            "news":        fetch_news(),
            "as_of":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/funding")
def api_funding():
    """Return funding rates and open interest for tracked symbols."""
    try:
        from crypto_market_data import fetch_funding_rates, fetch_open_interest
        return jsonify({
            "funding": fetch_funding_rates(),
            "oi":      fetch_open_interest(),
            "as_of":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser
    print("\n" + "=" * 52)
    print("  \U0001f916  Crypto Signal Bot Dashboard")
    print("  Opening http://localhost:5000 ...")
    print("  Press Ctrl+C to stop the dashboard.")
    print("=" * 52 + "\n")
    webbrowser.open("http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
