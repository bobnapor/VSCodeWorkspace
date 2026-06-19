# =============================================================================
# crypto_alerts.py — Alert System (Email + Telegram)
# =============================================================================

import logging
import smtplib
import urllib.request
import urllib.parse
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

import crypto_config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Email alerts
# ---------------------------------------------------------------------------

def _build_email_body(results):
    """Build plain-text and HTML email bodies."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actionable = [r for r in results if r.consensus is not None]

    # --- Plain text ---
    lines = [f"Crypto Signal Report — {now}", "=" * 50, ""]
    if actionable:
        lines.append(f"⚡ {len(actionable)} actionable signal(s) found:\n")
        for r in actionable:
            lines.append(r.summary())
            lines.append("")
    else:
        lines.append("No actionable signals at this time.\n")

    lines.append("-" * 50)
    lines.append("All symbols scanned:")
    for r in results:
        lines.append(r.summary())
        lines.append("")
    plain = "\n".join(lines)

    # --- HTML ---
    def consensus_badge(c):
        if c == "BUY":
            return (
                '<span style="color:green;font-weight:bold;">✅ BUY</span>'
            )
        if c == "SELL":
            return (
                '<span style="color:red;font-weight:bold;">🔴 SELL</span>'
            )
        return '<span style="color:gray;">— No Signal</span>'

    rows = ""
    for r in results:
        signal_detail = "<br>".join(
            f"&nbsp;&nbsp;<b>{k}</b>: {v or '—'}"
            for k, v in r.signals.items()
        )
        rows += (
            f"<tr>"
            f"<td><b>{r.symbol}</b></td>"
            f"<td>${r.price:,.4f}</td>"
            f"<td>{consensus_badge(r.consensus)}</td>"
            f"<td style='font-size:0.85em'>{signal_detail}</td>"
            f"</tr>"
        )

    html = f"""
<html><body>
<h2>🤖 Crypto Signal Report</h2>
<p><i>{now}</i></p>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-family:monospace">
  <tr style="background:#eee">
    <th>Symbol</th><th>Price</th>
    <th>Consensus</th><th>Indicators</th>
  </tr>
  {rows}
</table>
<hr>
<p style="font-size:0.8em;color:gray;">
  This is an automated signal alert. Not financial advice.
</p>
</body></html>
"""
    return plain, html


def send_email(results, subject_tag: str = "") -> bool:
    """Send signal report via Gmail SMTP. Returns True on success."""
    if not cfg.EMAIL_SENDER or not cfg.EMAIL_PASSWORD:
        logger.warning("Email credentials not configured. Skipping email.")
        return False

    actionable = [r for r in results if r.consensus is not None]
    tag = subject_tag or (
        f"{len(actionable)} signal(s)" if actionable else "No signals"
    )
    subject = f"[Crypto Bot] {tag} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    plain, html = _build_email_body(results)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.EMAIL_SENDER
    msg["To"] = ", ".join(cfg.EMAIL_RECIPIENTS)
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(cfg.EMAIL_SENDER, cfg.EMAIL_PASSWORD)
            server.sendmail(
                cfg.EMAIL_SENDER,
                cfg.EMAIL_RECIPIENTS,
                msg.as_string(),
            )
        logger.info("Email sent to %s", cfg.EMAIL_RECIPIENTS)
        return True
    except Exception as exc:
        logger.error("Failed to send email: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Telegram alerts
# ---------------------------------------------------------------------------

def send_telegram(results) -> bool:
    """Send signal summary via Telegram Bot API. Returns True on success."""
    if not cfg.TELEGRAM_BOT_TOKEN or not cfg.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured. Skipping.")
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"🤖 *Crypto Signal Report*\n_{now}_\n"]
    for r in results:
        if r.consensus == "BUY":
            icon = "✅"
        elif r.consensus == "SELL":
            icon = "🔴"
        else:
            icon = "⬜"
        lines.append(
            f"{icon} *{r.symbol}* @ `${r.price:,.4f}` — "
            f"{r.consensus or 'No Signal'}"
        )
        for name, sig in r.signals.items():
            lines.append(f"   • {name}: {sig or '—'}")
        lines.append("")

    text = "\n".join(lines)

    url = (
        f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/sendMessage"
    )
    data = urllib.parse.urlencode(
        {
            "chat_id": cfg.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }
    ).encode()

    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        if body.get("ok"):
            logger.info("Telegram message sent.")
            return True
        logger.error("Telegram API error: %s", body)
        return False
    except Exception as exc:
        logger.error("Failed to send Telegram message: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def send_alerts(results) -> None:
    """Send alerts via the channels specified in crypto_config.ALERT_MODE."""
    mode = cfg.ALERT_MODE.lower()
    if mode in ("email", "both"):
        send_email(results)
    if mode in ("telegram", "both"):
        send_telegram(results)
    if mode == "none":
        logger.info("Alert mode is 'none'. No alerts sent.")


# ---------------------------------------------------------------------------
# Heartbeat alert
# ---------------------------------------------------------------------------

def send_heartbeat_alert(summary_lines: list) -> None:
    """Send a 'bot is alive' heartbeat with a plain-text summary.
    Uses whatever channel(s) are configured in ALERT_MODE."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = "\n".join(
        [f"🤖 Crypto Bot Heartbeat — {now}", "=" * 42, ""]
        + summary_lines
        + ["", "— Crypto Signal Bot"]
    )
    mode = cfg.ALERT_MODE.lower()

    # Email
    if mode in ("email", "both"):
        if cfg.EMAIL_SENDER and "your_gmail" not in cfg.EMAIL_SENDER:
            subject = f"[Crypto Bot] ✅ Heartbeat — {now[:16]}"
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = cfg.EMAIL_SENDER
            msg["To"] = ", ".join(cfg.EMAIL_RECIPIENTS)
            html = f"<html><body><pre>{text}</pre></body></html>"
            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(cfg.EMAIL_SENDER, cfg.EMAIL_PASSWORD)
                    server.sendmail(
                        cfg.EMAIL_SENDER, cfg.EMAIL_RECIPIENTS, msg.as_string()
                    )
                logger.info("Heartbeat email sent.")
            except Exception as exc:
                logger.error("Heartbeat email failed: %s", exc)

    # Telegram
    if mode in ("telegram", "both"):
        if cfg.TELEGRAM_BOT_TOKEN and cfg.TELEGRAM_CHAT_ID:
            url = (
                f"https://api.telegram.org/bot"
                f"{cfg.TELEGRAM_BOT_TOKEN}/sendMessage"
            )
            import urllib.parse
            data = urllib.parse.urlencode(
                {"chat_id": cfg.TELEGRAM_CHAT_ID, "text": text}
            ).encode()
            try:
                import urllib.request
                urllib.request.urlopen(
                    urllib.request.Request(url, data=data), timeout=10
                )
                logger.info("Heartbeat Telegram message sent.")
            except Exception as exc:
                logger.error("Heartbeat Telegram failed: %s", exc)

    if mode == "none":
        logger.info("Heartbeat (no alert): %s", " | ".join(summary_lines))
