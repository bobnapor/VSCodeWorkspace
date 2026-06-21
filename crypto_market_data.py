# =============================================================================
# crypto_market_data.py — External Market & Risk Data Fetcher
# =============================================================================
# Fetches: funding rates, open interest, Fear & Greed, news, on-chain TVL.
# All endpoints are PUBLIC — no API keys required.
# Results are TTL-cached in memory to avoid rate-limiting external services.
# =============================================================================

import json
import logging
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import crypto_config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP + TTL cache
# ---------------------------------------------------------------------------

_cache = {}
_CACHE_TTL = {
    "funding":     60,
    "oi":          60,
    "fear_greed":  1800,
    "news":        300,
    "chain_tvl":   300,
    "stablecoins": 600,
}


def _get(url, params=None, timeout=8):
    """urllib GET → parsed JSON, or None on any failure."""
    try:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": "crypto-signal-bot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.debug("HTTP GET %s failed: %s", url, exc)
        return None


def _cached(key, fn):
    """Return cached result for `key`, or call `fn()` and cache it."""
    now = time.time()
    entry = _cache.get(key)
    ttl = _CACHE_TTL.get(key, 60)
    if entry and (now - entry["ts"]) < ttl:
        return entry["data"]
    data = fn()
    _cache[key] = {"data": data, "ts": now}
    return data


# ---------------------------------------------------------------------------
# Funding rates
# Binance Futures (BTC/ETH/SOL); Bybit linear (HYPE)
# ---------------------------------------------------------------------------

_BINANCE_FAPI = "https://fapi.binance.com"
_BYBIT_BASE   = "https://api.bybit.com"


def _classify_funding(rate):
    """Return BEARISH / NEUTRAL / BULLISH based on funding rate."""
    if rate > 0.0005:    # > +0.05%/8h → crowded longs, mean-rev risk
        return "BEARISH"
    if rate < -0.0001:   # negative → shorts paying → potential squeeze
        return "BULLISH"
    return "NEUTRAL"


def fetch_funding_rates():
    """
    Return {symbol: dict} with current funding rate for each tracked symbol.
    Keys: rate (float), rate_pct, rate_ann (annualized), next_funding,
          mark_price, index_price, signal ("BEARISH"/"NEUTRAL"/"BULLISH").
    """
    def _fetch():
        results = {}
        # Binance Futures
        for sym in cfg.SYMBOLS:
            if "HYPE" in sym:
                continue
            tv_sym = sym.replace("/", "")
            data = _get(
                _BINANCE_FAPI + "/fapi/v1/premiumIndex", {"symbol": tv_sym}
            )
            if not data:
                continue
            rate = float(data.get("lastFundingRate", 0))
            next_ts = data.get("nextFundingTime", 0)
            try:
                next_str = datetime.fromtimestamp(
                    next_ts / 1000, tz=timezone.utc
                ).strftime("%H:%M UTC")
            except Exception:
                next_str = "—"
            results[sym] = {
                "rate":         rate,
                "rate_pct":     "{:+.4f}%".format(rate * 100),
                "rate_ann":     "{:+.1f}%".format(rate * 100 * 3 * 365),
                "next_funding": next_str,
                "mark_price":   float(data.get("markPrice", 0)),
                "index_price":  float(data.get("indexPrice", 0)),
                "signal":       _classify_funding(rate),
            }

        # Bybit for HYPE
        if "HYPE/USDT" in cfg.SYMBOLS:
            data = _get(
                _BYBIT_BASE + "/v5/market/tickers",
                {"category": "linear", "symbol": "HYPEUSDT"},
            )
            lst = (data or {}).get("result", {}).get("list") or []
            if lst:
                t = lst[0]
                rate = float(t.get("fundingRate", 0))
                try:
                    next_str = datetime.fromtimestamp(
                        int(t.get("nextFundingTime", 0)) / 1000, tz=timezone.utc
                    ).strftime("%H:%M UTC")
                except Exception:
                    next_str = "—"
                results["HYPE/USDT"] = {
                    "rate":         rate,
                    "rate_pct":     "{:+.4f}%".format(rate * 100),
                    "rate_ann":     "{:+.1f}%".format(rate * 100 * 3 * 365),
                    "next_funding": next_str,
                    "mark_price":   float(t.get("markPrice", 0)),
                    "index_price":  float(t.get("indexPrice", 0)),
                    "signal":       _classify_funding(rate),
                }
        return results

    return _cached("funding", _fetch)


# ---------------------------------------------------------------------------
# Open Interest
# ---------------------------------------------------------------------------

def fetch_open_interest():
    """
    Return {symbol: dict} with current OI and 4h change for each tracked symbol.
    Keys: oi_coins, oi_usd, oi_fmt, change_pct, change_fmt, change_positive.
    """
    def _fetch():
        results = {}

        for sym in cfg.SYMBOLS:
            if "HYPE" in sym:
                continue
            tv_sym = sym.replace("/", "")
            data = _get(
                _BINANCE_FAPI + "/fapi/v1/openInterest", {"symbol": tv_sym}
            )
            if not data:
                continue
            oi = float(data.get("openInterest", 0))
            # Mark price for USD conversion
            pi = _get(
                _BINANCE_FAPI + "/fapi/v1/premiumIndex", {"symbol": tv_sym}
            )
            price = float((pi or {}).get("markPrice", 1)) or 1
            oi_usd = oi * price
            # 4-hour OI change via hourly history
            hist = _get(
                _BINANCE_FAPI + "/futures/data/openInterestHist",
                {"symbol": tv_sym, "period": "1h", "limit": 5},
            )
            change_pct = 0.0
            if hist and len(hist) >= 2:
                old_oi = float(hist[0].get("sumOpenInterest", 1)) or 1
                new_oi = float(hist[-1].get("sumOpenInterest", old_oi))
                change_pct = (new_oi - old_oi) / old_oi * 100
            results[sym] = {
                "oi_coins":       oi,
                "oi_usd":         oi_usd,
                "oi_fmt":         (
                    "${:.2f}B".format(oi_usd / 1e9)
                    if oi_usd >= 1e9 else
                    "${:.0f}M".format(oi_usd / 1e6)
                ),
                "change_pct":     change_pct,
                "change_fmt":     "{:+.1f}%".format(change_pct),
                "change_positive": change_pct >= 0,
            }

        # Bybit HYPE OI
        if "HYPE/USDT" in cfg.SYMBOLS:
            data = _get(
                _BYBIT_BASE + "/v5/market/tickers",
                {"category": "linear", "symbol": "HYPEUSDT"},
            )
            lst = (data or {}).get("result", {}).get("list") or []
            if lst:
                oi_val = float(lst[0].get("openInterestValue", 0))
                results["HYPE/USDT"] = {
                    "oi_coins":        oi_val,
                    "oi_usd":          oi_val,
                    "oi_fmt":          (
                        "${:.2f}B".format(oi_val / 1e9)
                        if oi_val >= 1e9 else
                        "${:.0f}M".format(oi_val / 1e6)
                    ),
                    "change_pct":      0.0,
                    "change_fmt":      "—",
                    "change_positive": True,
                }
        return results

    return _cached("oi", _fetch)


# ---------------------------------------------------------------------------
# Fear & Greed Index  (alternative.me — free, no key)
# ---------------------------------------------------------------------------

def fetch_fear_greed():
    """
    Return current Fear & Greed value + 7-day history.
    Value 0-25: Extreme Fear | 25-45: Fear | 45-55: Neutral |
    55-75: Greed | 75-100: Extreme Greed
    """
    def _fetch():
        data = _get("https://api.alternative.me/fng/", {"limit": 7})
        if not data or "data" not in data:
            return None
        items = data["data"]
        today = items[0]
        val = int(today["value"])
        if val >= 75:
            color = "#ef4444"
        elif val >= 55:
            color = "#f97316"
        elif val >= 45:
            color = "#eab308"
        elif val >= 25:
            color = "#84cc16"
        else:
            color = "#22c55e"
        return {
            "value": val,
            "label": today["value_classification"],
            "color": color,
            "history": [
                {
                    "value": int(x["value"]),
                    "label": x["value_classification"],
                    "date":  datetime.fromtimestamp(
                        int(x["timestamp"])
                    ).strftime("%m/%d"),
                }
                for x in reversed(items)
            ],
        }

    return _cached("fear_greed", _fetch)


# ---------------------------------------------------------------------------
# Crypto news  (CryptoCompare free tier — no key required for headlines)
# ---------------------------------------------------------------------------

def fetch_news(limit=12):
    """
    Return list of recent news items, filtered to tracked asset categories.
    Each item: title, source, url, published, tags.
    """
    def _fetch():
        cats = ",".join(s.split("/")[0] for s in cfg.SYMBOLS)
        data = _get(
            "https://min-api.cryptocompare.com/data/v2/news/",
            {"lang": "EN", "categories": cats, "sortOrder": "popular"},
        )
        if not data or data.get("Type") != 100:
            # Fallback: latest news regardless of category
            data = _get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                {"lang": "EN"},
            )
        if not data or "Data" not in data:
            return []
        items = []
        for n in (data.get("Data") or [])[:limit]:
            pub_ts = n.get("published_on", 0)
            pub = datetime.fromtimestamp(pub_ts).strftime("%m/%d %H:%M") if pub_ts else "—"
            items.append({
                "title":     n.get("title", ""),
                "source":    (n.get("source_info") or {}).get(
                    "name", n.get("source", "")
                ),
                "url":       n.get("url", "#"),
                "published": pub,
                "tags":      n.get("categories", ""),
            })
        return items

    return _cached("news", _fetch)


# ---------------------------------------------------------------------------
# On-chain: DeFiLlama chain TVL  (free, no key)
# ---------------------------------------------------------------------------

def fetch_chain_tvl():
    """
    Return top blockchain ecosystems by TVL with 1d / 7d change.
    Useful for understanding where capital is flowing across chains.
    """
    def _fetch():
        data = _get("https://api.llama.fi/chains")
        if not data:
            return []
        rows = []
        for c in sorted(data, key=lambda x: x.get("tvl", 0), reverse=True):
            if len(rows) >= 12:
                break
            tvl = c.get("tvl", 0) or 0
            if tvl < 1e6:
                continue
            d1 = c.get("change_1d") or 0.0
            d7 = c.get("change_7d") or 0.0
            rows.append({
                "name":    c.get("name", ""),
                "tvl":     tvl,
                "tvl_fmt": (
                    "${:.1f}B".format(tvl / 1e9)
                    if tvl >= 1e9 else
                    "${:.0f}M".format(tvl / 1e6)
                ),
                "d1":      d1,
                "d1_fmt":  "{:+.1f}%".format(d1),
                "d7":      d7,
                "d7_fmt":  "{:+.1f}%".format(d7),
                "d1_pos":  d1 >= 0,
            })
        return rows

    return _cached("chain_tvl", _fetch)


def fetch_stablecoin_supply():
    """
    Return total stablecoin market cap + top 5 stablecoins.
    Rising supply = capital entering crypto (bullish macro signal).
    Falling supply = capital leaving (bearish macro signal).
    """
    def _fetch():
        data = _get("https://stablecoins.llama.fi/stablecoins?includePrices=true")
        if not data or "peggedAssets" not in data:
            return None
        assets = data["peggedAssets"]
        total = sum(
            (a.get("circulating") or {}).get("peggedUSD", 0) for a in assets
        )
        top = []
        for a in sorted(
            assets,
            key=lambda x: (x.get("circulating") or {}).get("peggedUSD", 0),
            reverse=True,
        )[:5]:
            circ = (a.get("circulating") or {}).get("peggedUSD", 0)
            top.append({
                "symbol":      a.get("symbol", ""),
                "supply_fmt":  "${:.1f}B".format(circ / 1e9),
            })
        return {
            "total":     total,
            "total_fmt": "${:.1f}B".format(total / 1e9),
            "top":       top,
        }

    return _cached("stablecoins", _fetch)


# ---------------------------------------------------------------------------
# Portfolio risk metrics  (computed from local data, no external calls)
# ---------------------------------------------------------------------------

def compute_risk_metrics(portfolio, prices):
    """
    Compute trading risk metrics from the current portfolio state.
    Returns a dict with formatted values + a flags list.
    """
    cash      = portfolio.get("usdt_balance", 0)
    holdings  = portfolio.get("holdings", {})
    trade_log = portfolio.get("trade_log", [])
    starting  = cfg.PAPER_STARTING_BALANCE

    holdings_val = sum(
        pos["amount"] * prices.get(base, pos.get("avg_cost", 0))
        for base, pos in holdings.items()
    )
    total = cash + holdings_val

    # Per-asset concentration
    concentration = {}
    for base, pos in holdings.items():
        price = prices.get(base, pos.get("avg_cost", 0))
        val = pos["amount"] * price
        concentration[base] = val / total * 100 if total else 0.0

    max_conc       = max(concentration.values()) if concentration else 0.0
    max_conc_asset = max(concentration, key=concentration.get) if concentration else None

    # Max drawdown + current drawdown from trade history
    max_dd = 0.0
    current_dd = 0.0
    if trade_log:
        running_cash = starting
        running_hold = {}
        peak = starting
        for t in trade_log:
            base = t["symbol"].split("/")[0]
            if t["side"] == "BUY":
                running_cash -= t["amount"] * t["price"]
                running_hold[base] = running_hold.get(base, 0) + t["amount"]
            else:
                running_cash += t["amount"] * t["price"]
                running_hold[base] = max(
                    0, running_hold.get(base, 0) - t["amount"]
                )
            est = running_cash + sum(
                a * t["price"] for a in running_hold.values()
            )
            peak   = max(peak, est)
            dd     = (peak - est) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        current_dd = (peak - total) / peak * 100 if peak > 0 else 0

    # Win rate + profit factor
    sell_trades = [t for t in trade_log if t["side"] == "SELL" and "pnl" in t]
    wins   = [t for t in sell_trades if t["pnl"] > 0]
    losses = [t for t in sell_trades if t["pnl"] <= 0]

    win_rate     = len(wins) / len(sell_trades) * 100 if sell_trades else None
    avg_win      = sum(t["pnl"] for t in wins) / len(wins) if wins else None
    avg_loss     = sum(t["pnl"] for t in losses) / len(losses) if losses else None
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss   = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss else None

    pnl_pct      = (total - starting) / starting * 100 if starting else 0
    exposure_pct = holdings_val / total * 100 if total else 0

    # Risk flags
    flags = []
    if max_conc > 35:
        flags.append({
            "level": "danger",
            "text": "{} is {:.0f}% of portfolio — very high concentration".format(
                max_conc_asset, max_conc
            ),
        })
    elif max_conc > 20:
        flags.append({
            "level": "warn",
            "text": "{} is {:.0f}% of portfolio — elevated concentration".format(
                max_conc_asset, max_conc
            ),
        })
    if current_dd > 15:
        flags.append({
            "level": "danger",
            "text": "Current drawdown {:.1f}% from portfolio peak".format(current_dd),
        })
    elif current_dd > 7:
        flags.append({
            "level": "warn",
            "text": "Drawdown {:.1f}% from peak".format(current_dd),
        })
    if cash < 500:
        flags.append({
            "level": "warn",
            "text": "Low cash: ${:.0f} USDT — limited room for new entries".format(cash),
        })
    if win_rate is not None and len(sell_trades) >= 3 and win_rate < 40:
        flags.append({
            "level": "warn",
            "text": "Win rate {:.0f}% over {} closed trades".format(
                win_rate, len(sell_trades)
            ),
        })
    if not flags:
        flags.append({"level": "ok", "text": "No major risk flags detected"})

    return {
        "total":            total,
        "pnl_pct":          pnl_pct,
        "pnl_pct_fmt":      "{:+.2f}%".format(pnl_pct),
        "pnl_positive":     pnl_pct >= 0,
        "exposure_pct":     exposure_pct,
        "exposure_fmt":     "{:.0f}%".format(exposure_pct),
        "cash_pct":         cash / total * 100 if total else 100,
        "concentration":    [
            {"asset": k, "pct": v, "pct_fmt": "{:.1f}%".format(v)}
            for k, v in concentration.items()
        ],
        "max_conc":         max_conc,
        "max_dd":           max_dd,
        "max_dd_fmt":       "{:.1f}%".format(max_dd),
        "current_dd":       current_dd,
        "current_dd_fmt":   "{:.1f}%".format(current_dd),
        "win_rate_fmt":     "{:.0f}%".format(win_rate) if win_rate is not None else "—",
        "total_trades":     len(trade_log),
        "sell_trades":      len(sell_trades),
        "avg_win_fmt":      "${:+.2f}".format(avg_win)  if avg_win  is not None else "—",
        "avg_loss_fmt":     "${:+.2f}".format(avg_loss) if avg_loss is not None else "—",
        "profit_factor_fmt": "{:.2f}".format(profit_factor) if profit_factor else "—",
        "flags":            flags,
    }
