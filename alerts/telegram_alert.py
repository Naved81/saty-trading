"""
alerts/telegram_alert.py
========================
Sends trade signal alerts via Telegram.

SETUP
-----
1. Message @BotFather on Telegram → /newbot → copy bot token
2. Message @userinfobot → copy your chat ID
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""

import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_message(text: str, parse_mode: str = 'Markdown') -> bool:
    """
    Send a Telegram message to the configured chat.
    Returns True on success, False on failure.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id':    TELEGRAM_CHAT_ID,
        'text':       text,
        'parse_mode': parse_mode,
    }

    try:
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[!] Telegram send failed: {e}")
        return False


def send_signal(signal) -> bool:
    """Send a formatted Signal object as a Telegram alert."""
    return send_message(signal.alert_text())


def send_no_signal(tickers: list) -> bool:
    """Send a 'no signal today' notification."""
    text = (f"📊 *Saty Signal Check — No Trade*\n"
            f"Checked: {', '.join(tickers)}\n"
            f"No qualifying setups at 10:30 AM CT.")
    return send_message(text)


def test_connection() -> bool:
    """Send a test message to verify Telegram setup."""
    return send_message("✅ Saty Trading Bot connected successfully.")
