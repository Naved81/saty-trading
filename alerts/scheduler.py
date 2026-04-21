"""
alerts/scheduler.py
===================
Runs the signal checker every trading day at 10:30 AM CT.
Designed to run continuously on a server (Railway, VPS, etc.)

USAGE
-----
    python alerts/scheduler.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import schedule
from config import CT
from alerts.signal_checker import run_signal_check
from alerts.telegram_alert import send_message


def is_trading_day() -> bool:
    """Return True if today is a weekday (Mon–Fri)."""
    return datetime.now(CT).weekday() < 5


def job():
    """Scheduled job — runs at 10:30 AM CT on trading days."""
    if not is_trading_day():
        print(f"[{datetime.now(CT).strftime('%Y-%m-%d')}] Weekend — skipping.")
        return
    run_signal_check()


def main():
    print("Saty Trading Scheduler started.")
    print("Signal check scheduled: 10:30 AM CT, Mon–Fri\n")

    # Send startup message
    send_message("🚀 *Saty Trading Bot started*\nSignal check runs at 10:30 AM CT daily.")

    # Schedule the job
    schedule.every().day.at("10:30").do(job)

    print("Waiting for next scheduled run...")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == '__main__':
    main()
