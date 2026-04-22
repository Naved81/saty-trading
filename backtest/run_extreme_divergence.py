"""
backtest/run_extreme_divergence.py
====================================
Prints every date and time where the Saty Phase Oscillator is in the
EXTREME zone (> +100 or < -100) at a divergence pivot.

Logic:
  • Resample 1-min RTH bars to 3-min, 5-min, and 10-min
  • Compute SPO on each timeframe (continuous EMA — no daily resets)
  • Detect all four divergence types (bull / bear / hidden bull / hidden bear)
  • Keep only signals where |SPO at the pivot| > 100

This is an extreme reversal / continuation setup: the oscillator has
reached a historically overextended level while diverging from price,
suggesting strong mean-reversion potential.

USAGE
-----
    python backtest/run_extreme_divergence.py                     # all tickers
    python backtest/run_extreme_divergence.py --ticker SPY
    python backtest/run_extreme_divergence.py --start 2022-01-01
    python backtest/run_extreme_divergence.py --tf 3 5            # 3-min + 5-min only
    python backtest/run_extreme_divergence.py --output extreme.csv
    python backtest/run_extreme_divergence.py --catalog
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import (
    DIV_LB_LEFT, DIV_LB_RIGHT, DIV_RANGE_MIN, DIV_RANGE_MAX, DIV_TIMEFRAMES,
    WARMUP_DAYS,
)
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.sessions import filter_rth, resample_to
from core.indicators import calc_spo, calc_spo_divergence

EXTREME_THRESHOLD = 100.0   # |SPO| must exceed this at the pivot bar

DIV_LABEL = {
    'bull':        'Regular Bull  ',
    'bear':        'Regular Bear  ',
    'hidden_bull': 'Hidden Bull   ',
    'hidden_bear': 'Hidden Bear   ',
}

# ── Core scanner ──────────────────────────────────────────────────────────────

def scan_extreme_divergences(
    ticker: str,
    df_rth: pd.DataFrame,
    timeframes: list,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Scan all requested timeframes and return extreme-zone divergence events.

    Each row: ticker, timeframe_min, date, time, div_type,
              osc_value (at pivot), price_level (low or high at pivot)
    """
    all_events = []

    for tf_min in timeframes:
        # Resample RTH 1-min bars to target timeframe
        df = resample_to(df_rth, rule=f'{tf_min}min')
        if df.empty:
            continue

        # Date range filter
        if start:
            df = df[df['timestamp'] >= pd.Timestamp(start)]
        if end:
            df = df[df['timestamp'] <= pd.Timestamp(end)]

        warmup = max(WARMUP_DAYS, 40)
        if len(df) <= warmup:
            continue
        df = df.iloc[warmup:].reset_index(drop=True)

        # Compute SPO
        spo_df = calc_spo(df)
        osc    = spo_df['spo']

        # Extract raw pivot data so we can report the actual pivot bar's values
        osc_arr  = osc.values.astype(float)
        low_arr  = df['low'].values.astype(float)
        high_arr = df['high'].values.astype(float)
        ts_arr   = df['timestamp'].values
        n        = len(osc_arr)
        lb_l, lb_r = DIV_LB_LEFT, DIV_LB_RIGHT

        pl_list, ph_list = [], []
        for i in range(lb_l, n - lb_r):
            val = osc_arr[i]
            if np.isnan(val):
                continue
            left  = osc_arr[i - lb_l: i]
            right = osc_arr[i + 1: i + lb_r + 1]
            if np.any(np.isnan(left)) or np.any(np.isnan(right)):
                continue
            if np.all(left > val) and np.all(right > val):
                pl_list.append((i, i + lb_r))
            if np.all(left < val) and np.all(right < val):
                ph_list.append((i, i + lb_r))

        # Bull / hidden-bull from pivot lows
        for k in range(1, len(pl_list)):
            piv_p, conf_p = pl_list[k - 1]
            piv_c, conf_c = pl_list[k]
            if not (DIV_RANGE_MIN <= (conf_c - conf_p) <= DIV_RANGE_MAX):
                continue
            o_p, o_c = osc_arr[piv_p], osc_arr[piv_c]
            l_p, l_c = low_arr[piv_p],  low_arr[piv_c]
            if np.isnan(o_p) or np.isnan(o_c):
                continue

            # Extreme filter — oscillator at the current pivot must be extreme
            if abs(o_c) <= EXTREME_THRESHOLD:
                continue

            ts = pd.Timestamp(ts_arr[conf_c])
            base = dict(ticker=ticker, timeframe_min=tf_min,
                        date=ts.date(), time=ts.strftime('%H:%M'),
                        osc_value=round(o_c, 2), price_level=round(l_c, 2))

            if l_c < l_p and o_c > o_p:
                all_events.append({**base, 'div_type': 'bull'})
            if l_c > l_p and o_c < o_p:
                all_events.append({**base, 'div_type': 'hidden_bull'})

        # Bear / hidden-bear from pivot highs
        for k in range(1, len(ph_list)):
            piv_p, conf_p = ph_list[k - 1]
            piv_c, conf_c = ph_list[k]
            if not (DIV_RANGE_MIN <= (conf_c - conf_p) <= DIV_RANGE_MAX):
                continue
            o_p, o_c = osc_arr[piv_p], osc_arr[piv_c]
            h_p, h_c = high_arr[piv_p], high_arr[piv_c]
            if np.isnan(o_p) or np.isnan(o_c):
                continue

            if abs(o_c) <= EXTREME_THRESHOLD:
                continue

            ts = pd.Timestamp(ts_arr[conf_c])
            base = dict(ticker=ticker, timeframe_min=tf_min,
                        date=ts.date(), time=ts.strftime('%H:%M'),
                        osc_value=round(o_c, 2), price_level=round(h_c, 2))

            if h_c > h_p and o_c < o_p:
                all_events.append({**base, 'div_type': 'bear'})
            if h_c < h_p and o_c > o_p:
                all_events.append({**base, 'div_type': 'hidden_bear'})

    if not all_events:
        return pd.DataFrame()

    return (pd.DataFrame(all_events)
            .sort_values(['date', 'time', 'timeframe_min'])
            .reset_index(drop=True))


# ── Printer ───────────────────────────────────────────────────────────────────

def print_results(events: pd.DataFrame, ticker: str):
    if events.empty:
        print(f"  [{ticker}]  No extreme divergences found.\n")
        return

    total = len(events)
    print(f"\n  ┌{'─'*62}┐")
    print(f"  │  {ticker:<6s}  {str(events['date'].min())} → "
          f"{str(events['date'].max()):<16s}  "
          f"{total:4d} signals │")
    print(f"  └{'─'*62}┘")

    # Group by timeframe for readability
    for tf in sorted(events['timeframe_min'].unique()):
        sub = events[events['timeframe_min'] == tf].copy()
        tf_label = f"  ── {ticker}  {tf}min  ({len(sub)} signals) "
        print(f"\n{tf_label}{'─' * max(0, 64 - len(tf_label))}")
        print(f"  {'Date':<12} {'Time':<7} {'Type':<16} "
              f"{'SPO':>8}  {'Price':>9}")
        print(f"  {'─'*12} {'─'*6} {'─'*16} {'─'*8}  {'─'*9}")

        for _, row in sub.iterrows():
            direction = '▲' if row['div_type'] in ('bull', 'hidden_bull') else '▼'
            print(f"  {str(row['date']):<12} {row['time']:<7} "
                  f"{DIV_LABEL[row['div_type']]:<16} "
                  f"{row['osc_value']:>+8.2f}  "
                  f"{row['price_level']:>9.2f}  {direction}")

    # Summary by type
    print(f"\n  Summary for {ticker}:")
    for dtype, cnt in events['div_type'].value_counts().items():
        bar = '█' * int(cnt / total * 30)
        print(f"    {DIV_LABEL[dtype]}  {cnt:4d}  {bar}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SPO Extreme Divergence — print dates and times only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',  type=str,            help='Single ticker (default: all in DB)')
    parser.add_argument('--start',   type=str,            help='Start date YYYY-MM-DD')
    parser.add_argument('--end',     type=str,            help='End date YYYY-MM-DD')
    parser.add_argument('--tf',      type=int, nargs='+', help='Timeframes in minutes (default: 3 5 10)')
    parser.add_argument('--output',  type=str,            help='Optional CSV output path')
    parser.add_argument('--catalog', action='store_true', help='Show DB contents and exit')
    args = parser.parse_args()

    timeframes = args.tf if args.tf else DIV_TIMEFRAMES

    con = get_connection()

    if args.catalog:
        print(get_catalog(con).to_string(index=False))
        con.close()
        return

    tickers = ([args.ticker.upper()] if args.ticker
               else get_available_tickers('1m_full', con))
    if not tickers:
        print("[!] No tickers found. Run data/massive_pull.py first.")
        con.close()
        sys.exit(1)

    print()
    print("  SPO EXTREME DIVERGENCE SCAN")
    print(f"  Tickers    : {', '.join(tickers)}")
    print(f"  Timeframes : {timeframes} min")
    print(f"  Period     : {args.start or 'all'} → {args.end or 'latest'}")
    print(f"  Filter     : |SPO at pivot| > {EXTREME_THRESHOLD}")
    print(f"  Pivots     : lbL={DIV_LB_LEFT}  lbR={DIV_LB_RIGHT}  "
          f"range=[{DIV_RANGE_MIN},{DIV_RANGE_MAX}] bars")

    all_events = []
    for ticker in tickers:
        print(f"\n  Loading {ticker}...")
        df = load_bars(ticker, '1m_full', args.start, args.end, con)
        if df.empty:
            print("    [!] No data.")
            continue
        print(f"    {len(df):,} bars  |  "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        df_rth = filter_rth(df)
        events = scan_extreme_divergences(ticker, df_rth, timeframes,
                                          args.start, args.end)
        print_results(events, ticker)
        all_events.append(events)

    con.close()

    if not all_events:
        return

    combined = (pd.concat(all_events, ignore_index=True)
                .sort_values(['date', 'time'])
                .reset_index(drop=True))

    # Cross-ticker combined table
    if len(tickers) > 1:
        total = len(combined)
        print(f"\n\n  {'═'*64}")
        print(f"  COMBINED  —  {len(tickers)} tickers  |  {total} extreme divergences")
        print(f"  {'═'*64}")
        print(f"\n  {'Date':<12} {'Time':<7} {'Ticker':<8} "
              f"{'TF':>4}  {'Type':<16} {'SPO':>8}")
        print(f"  {'─'*12} {'─'*6} {'─'*7} {'─'*4}  {'─'*16} {'─'*8}")
        for _, row in combined.iterrows():
            direction = '▲' if row['div_type'] in ('bull', 'hidden_bull') else '▼'
            print(f"  {str(row['date']):<12} {row['time']:<7} "
                  f"{row['ticker']:<8} "
                  f"{row['timeframe_min']:>3}m  "
                  f"{DIV_LABEL[row['div_type']]:<16} "
                  f"{row['osc_value']:>+8.2f}  {direction}")

        print(f"\n  By timeframe:")
        for tf in sorted(combined['timeframe_min'].unique()):
            sub = combined[combined['timeframe_min'] == tf]
            by_t = sub['div_type'].value_counts()
            parts = '  '.join(f"{t.replace('_',' ')}={n}" for t, n in by_t.items())
            print(f"    {tf:2d}min  {len(sub):4d} signals   {parts}")

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\n  Saved: {args.output}  ({len(combined)} rows)")


if __name__ == '__main__':
    main()
