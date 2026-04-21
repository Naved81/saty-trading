"""
alerts/signal_checker.py
========================
Runs the strategy on live Schwab data at 10:30 AM CT.
Called by scheduler.py — can also be run manually for testing.

USAGE
-----
    python alerts/signal_checker.py              # check all tickers
    python alerts/signal_checker.py --ticker QQQ # single ticker
    python alerts/signal_checker.py --dry-run    # check but don't send alert
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import DEFAULT_TICKERS, ET, ENTRY_TIME, RTH_START
from core.db import get_connection, load_bars
from core.sessions import filter_eth, filter_rth
from core.indicators import calc_spo, calc_compression, build_daily_reference
from strategies.saty_credit_spread import check_signal
from alerts.telegram_alert import send_signal, send_no_signal
from data.schwab_live import get_signal_data


def run_signal_check(tickers: list = None, dry_run: bool = False):
    """
    Main signal check — pulls live data, runs strategy, sends alerts.
    """
    tickers = tickers or DEFAULT_TICKERS
    now_et  = datetime.now(ET)
    print(f"\n[{now_et.strftime('%Y-%m-%d %H:%M ET')}] Running signal check...")
    print(f"Tickers: {', '.join(tickers)}\n")

    # Pull live data from Schwab
    live_data = get_signal_data(tickers)

    # Load historical bars from DB for indicator warm-up
    # We need enough history for the continuous EMAs to be stable
    con = get_connection()

    signals_fired = []

    for ticker in tickers:
        if ticker not in live_data:
            print(f"  [{ticker}] No live data available.")
            continue

        td = live_data[ticker]
        bars_today   = td['bars']
        prev_close   = td['prev_close']
        snap_entry   = td['snapshot_1130']
        snap_open    = td['snapshot_open']

        if snap_entry is None:
            print(f"  [{ticker}] No 11:30 ET bar yet.")
            continue

        if prev_close is None:
            print(f"  [{ticker}] No prev_close available.")
            continue

        # Load recent historical bars for indicator warm-up (~60 days)
        hist = load_bars(ticker, '1m_full', con=con)
        if hist.empty:
            print(f"  [{ticker}] No historical data in DB — skipping.")
            continue

        # Combine historical + today's live bars
        hist['date_ts'] = hist['timestamp'].dt.normalize()
        bars_today['date_ts'] = bars_today['timestamp'].dt.normalize()
        df_full = pd.concat([hist, bars_today], ignore_index=True)
        df_full = df_full.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        df_full['date_ts'] = df_full['timestamp'].dt.normalize()

        # Build indicators
        df_eth = filter_eth(df_full)
        df_rth = filter_rth(df_full)
        df_eth['date_ts'] = df_eth['timestamp'].dt.normalize()
        df_rth['date_ts'] = df_rth['timestamp'].dt.normalize()

        daily = build_daily_reference(df_full, df_rth)

        # Compute continuous SPO on full ETH history
        spo_df = calc_spo(df_eth)
        df_eth['spo']           = spo_df['spo']
        df_eth['pivot']         = spo_df['pivot']
        df_eth['iatr']          = spo_df['iatr']
        df_eth['in_compression']= calc_compression(df_eth, spo_df['pivot'], spo_df['iatr'])

        # Get today's row from daily reference
        today = bars_today['date_ts'].iloc[0]
        daily_row = daily[daily['date_ts'] == today]
        if daily_row.empty:
            print(f"  [{ticker}] No daily reference for today.")
            continue
        row = daily_row.iloc[0].copy()
        # Override prev_close with Schwab's reported value (more accurate)
        row['prev_close'] = float(prev_close)

        # Get indicator values at entry time
        entry_t = pd.Timestamp(ENTRY_TIME).time()
        eth_at_entry = df_eth[df_eth['timestamp'].dt.time == entry_t]
        if eth_at_entry.empty:
            print(f"  [{ticker}] No ETH bar at {ENTRY_TIME}.")
            continue

        spo_entry  = eth_at_entry['spo'].iloc[-1]
        comp_entry = bool(eth_at_entry['in_compression'].iloc[-1])
        price_entry= snap_entry['close']
        price_open = snap_open['close'] if snap_open is not None else np.nan

        # Check signal
        signal = check_signal(
            ticker=ticker,
            row=row,
            spo_at_entry=spo_entry,
            in_compression=comp_entry,
            price_at_entry=price_entry,
            price_at_open=price_open,
        )

        if signal:
            print(f"  [{ticker}] ✅ SIGNAL: {signal.direction}")
            print(f"             Short strike: ${signal.short_strike:.2f}")
            print(f"             Credit: ${signal.credit:.2f}")
            signals_fired.append(signal)
            if not dry_run:
                send_signal(signal)
        else:
            morning_pct = (price_entry - float(prev_close)) / row.get('prev_atr_rth', 1)
            print(f"  [{ticker}] — No signal  "
                  f"(morning: {morning_pct:+.3f}× ATR  "
                  f"SPO: {spo_entry:.1f}  "
                  f"comp: {comp_entry})")

    con.close()

    if not signals_fired and not dry_run:
        send_no_signal(tickers)

    print(f"\nDone. {len(signals_fired)} signal(s) fired.")
    return signals_fired


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker',  type=str)
    parser.add_argument('--dry-run', action='store_true',
                        help='Check signals but do not send Telegram alert')
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else None
    run_signal_check(tickers, dry_run=args.dry_run)
