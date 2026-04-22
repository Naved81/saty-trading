"""
backtest/run_price_movement.py
================================
For each extreme-divergence signal (|SPO| > 100 AND divergence confirmed),
measure how much price moved in the expected direction in the N bars that follow.

Conditions:
  1. SPO is extreme at the pivot bar (|SPO| > 100, configurable via --threshold)
  2. A divergence is confirmed (bull / bear / hidden_bull / hidden_bear)

Expected direction per type:
  bull        → UP   (regular bull: price lower low, osc higher low)
  hidden_bull → UP   (hidden bull:  price higher low, osc lower low)
  bear        → DOWN (regular bear: price higher high, osc lower high)
  hidden_bear → DOWN (hidden bear:  price lower high, osc higher high)

Measured horizons: 5, 10, 15, 30 bars (signal timeframe) + all remaining bars
that day (EOD).  For each horizon:
  move_pts   — price change in the expected direction (+= good, -= bad)
  move_pct   — same as % of entry price
  mfe_pts    — max favorable excursion (best the price got for you)
  mae_pts    — max adverse excursion  (worst the price went against you)
  n_bars     — actual bars measured (may be < horizon if near EOD)

USAGE
-----
    python backtest/run_price_movement.py                       # all tickers
    python backtest/run_price_movement.py --ticker SPY
    python backtest/run_price_movement.py --tf 5 10             # 5-min + 10-min
    python backtest/run_price_movement.py --start 2022-01-01
    python backtest/run_price_movement.py --threshold 61.8      # relax extreme filter
    python backtest/run_price_movement.py --horizons 5 15 30    # custom horizons
    python backtest/run_price_movement.py --min-bars 5          # drop near-EOD signals
    python backtest/run_price_movement.py --output moves.csv
    python backtest/run_price_movement.py --no-plot
    python backtest/run_price_movement.py --catalog
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from config import (
    DIV_LB_LEFT, DIV_LB_RIGHT, DIV_RANGE_MIN, DIV_RANGE_MAX, DIV_TIMEFRAMES,
    WARMUP_DAYS, RTH_START, RTH_END,
)
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.sessions import filter_eth, resample_to
from core.indicators import calc_spo

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_EXTREME_THRESHOLD = 100.0
DEFAULT_HORIZONS = [5, 10, 15, 30]

DIV_LABEL = {
    'bull':        'Regular Bull',
    'bear':        'Regular Bear',
    'hidden_bull': 'Hidden Bull ',
    'hidden_bear': 'Hidden Bear ',
}

# +1 = price expected to rise; -1 = price expected to fall
EXPECTED_DIR = {
    'bull':        +1,
    'bear':        -1,
    'hidden_bull': +1,
    'hidden_bear': -1,
}

# ── Price movement measurement ────────────────────────────────────────────────

def measure_move(
    df_rth: pd.DataFrame,
    signal_idx: int,
    entry_price: float,
    expected_dir: int,
    horizons: list,
) -> dict:
    """
    From signal_idx in df_rth, look ahead and measure price movement.

    entry_price   — close of the confirmation bar (signal bar)
    expected_dir  — +1 (expect rise) or -1 (expect fall)
    horizons      — list of bar counts to measure at

    Returns a flat dict of metrics for all horizons plus EOD.
    """
    signal_date = df_rth.iloc[signal_idx]['timestamp'].date()

    # All future RTH bars on the same calendar day
    future_all = df_rth.iloc[signal_idx + 1:]
    future_day = future_all[future_all['timestamp'].dt.date == signal_date]

    result = {}

    for h in horizons:
        window = future_day.iloc[:h]
        n = len(window)

        if n == 0:
            result[f'move_{h}b']     = np.nan
            result[f'move_{h}b_pct'] = np.nan
            result[f'mfe_{h}b']      = np.nan
            result[f'mae_{h}b']      = np.nan
            result[f'n_bars_{h}b']   = 0
            continue

        close_h = window.iloc[-1]['close']
        highs   = window['high'].values
        lows    = window['low'].values

        if expected_dir == +1:
            move = close_h - entry_price
            mfe  = float(np.max(highs)) - entry_price
            mae  = entry_price - float(np.min(lows))
        else:
            move = entry_price - close_h
            mfe  = entry_price - float(np.min(lows))
            mae  = float(np.max(highs)) - entry_price

        result[f'move_{h}b']     = round(move, 4)
        result[f'move_{h}b_pct'] = round(move / entry_price * 100, 4)
        result[f'mfe_{h}b']      = round(mfe, 4)
        result[f'mae_{h}b']      = round(mae, 4)
        result[f'n_bars_{h}b']   = n

    # EOD — all remaining bars that day
    n_eod = len(future_day)
    if n_eod > 0:
        close_eod = future_day.iloc[-1]['close']
        highs_eod = future_day['high'].values
        lows_eod  = future_day['low'].values

        if expected_dir == +1:
            move_eod = close_eod - entry_price
            mfe_eod  = float(np.max(highs_eod)) - entry_price
            mae_eod  = entry_price - float(np.min(lows_eod))
        else:
            move_eod = entry_price - close_eod
            mfe_eod  = entry_price - float(np.min(lows_eod))
            mae_eod  = float(np.max(highs_eod)) - entry_price

        result['move_eod']     = round(move_eod, 4)
        result['move_eod_pct'] = round(move_eod / entry_price * 100, 4)
        result['mfe_eod']      = round(mfe_eod, 4)
        result['mae_eod']      = round(mae_eod, 4)
        result['n_bars_eod']   = n_eod
    else:
        result['move_eod']     = np.nan
        result['move_eod_pct'] = np.nan
        result['mfe_eod']      = np.nan
        result['mae_eod']      = np.nan
        result['n_bars_eod']   = 0

    return result


# ── Core scanner ──────────────────────────────────────────────────────────────

def scan_price_movement(
    ticker: str,
    df_full: pd.DataFrame,
    timeframes: list,
    horizons: list,
    extreme_threshold: float = DEFAULT_EXTREME_THRESHOLD,
    min_bars_ahead: int = 1,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Find every extreme-divergence signal and measure subsequent price movement.

    Returns one row per signal with signal metadata + movement metrics.
    """
    df_eth = filter_eth(df_full)
    all_rows = []

    for tf_min in timeframes:
        df_eth_rs = resample_to(df_eth, rule=f'{tf_min}min')
        if df_eth_rs.empty:
            continue

        # Warm up EMAs before measuring
        warmup = max(WARMUP_DAYS, 40)
        if len(df_eth_rs) <= warmup:
            continue
        df_eth_rs = df_eth_rs.iloc[warmup:].reset_index(drop=True)

        # Compute SPO on continuous ETH stream
        spo_df = calc_spo(df_eth_rs)
        df_eth_rs = df_eth_rs.copy()
        df_eth_rs['spo_val'] = spo_df['spo'].values

        # Apply date range after SPO (preserve EMA history)
        if start:
            df_eth_rs = df_eth_rs[df_eth_rs['timestamp'] >= pd.Timestamp(start)]
        if end:
            df_eth_rs = df_eth_rs[df_eth_rs['timestamp'] <= pd.Timestamp(end)]
        if df_eth_rs.empty:
            continue

        # Filter to RTH bars for divergence detection + measurement
        rth_mask = (
            (df_eth_rs['timestamp'].dt.time >= RTH_START) &
            (df_eth_rs['timestamp'].dt.time <= RTH_END)
        )
        df = df_eth_rs[rth_mask].reset_index(drop=True)
        if df.empty:
            continue

        osc_arr  = df['spo_val'].values.astype(float)
        low_arr  = df['low'].values.astype(float)
        high_arr = df['high'].values.astype(float)
        close_arr = df['close'].values.astype(float)
        ts_arr   = df['timestamp'].values
        n        = len(osc_arr)
        lb_l, lb_r = DIV_LB_LEFT, DIV_LB_RIGHT

        # Pivot detection (same algorithm as run_extreme_divergence.py)
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

        def add_signal(conf_c, div_type, price_level):
            """Record signal and measure forward price movement."""
            if conf_c >= n - 1:
                return  # no bars ahead at all

            ts = pd.Timestamp(ts_arr[conf_c])
            entry_price = float(close_arr[conf_c])
            exp_dir     = EXPECTED_DIR[div_type]

            movement = measure_move(df, conf_c, entry_price, exp_dir, horizons)

            # Drop signals with too few bars ahead to be meaningful
            eod_n = movement.get('n_bars_eod', 0)
            if eod_n < min_bars_ahead:
                return

            row = dict(
                ticker        = ticker,
                timeframe_min = tf_min,
                date          = str(ts.date()),
                time          = ts.strftime('%H:%M'),
                div_type      = div_type,
                expected_dir  = 'up' if exp_dir == +1 else 'down',
                spo_value     = round(float(osc_arr[conf_c - lb_r]), 2),
                entry_price   = round(entry_price, 4),
                price_level   = round(price_level, 4),
            )
            row.update(movement)
            all_rows.append(row)

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
            if abs(o_c) <= extreme_threshold:
                continue

            if l_c < l_p and o_c > o_p:
                add_signal(conf_c, 'bull', l_c)
            if l_c > l_p and o_c < o_p:
                add_signal(conf_c, 'hidden_bull', l_c)

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
            if abs(o_c) <= extreme_threshold:
                continue

            if h_c > h_p and o_c < o_p:
                add_signal(conf_c, 'bear', h_c)
            if h_c < h_p and o_c > o_p:
                add_signal(conf_c, 'hidden_bear', h_c)

    if not all_rows:
        return pd.DataFrame()

    return (pd.DataFrame(all_rows)
            .sort_values(['date', 'time', 'timeframe_min'])
            .reset_index(drop=True))


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame, horizons: list, ticker: str):
    if df.empty:
        print(f"  [{ticker}]  No signals found.\n")
        return

    h_labels = [f'{h}b' for h in horizons] + ['eod']
    move_cols = [f'move_{h}' for h in h_labels]

    total    = len(df)
    date_min = df['date'].min()
    date_max = df['date'].max()

    header = f"  {ticker:<6s}  {date_min} → {date_max}  {total:4d} signals"
    print(f"\n  ┌{'─'*68}┐")
    print(f"  │{header:<68s}│")
    print(f"  └{'─'*68}┘")

    for tf in sorted(df['timeframe_min'].unique()):
        df_tf = df[df['timeframe_min'] == tf]
        print(f"\n  ── {ticker}  {tf}min  ({len(df_tf)} signals) "
              + '─' * max(0, 60 - len(str(tf)) - len(ticker) - 18))

        for dtype in ['bull', 'bear', 'hidden_bull', 'hidden_bear']:
            sub = df_tf[df_tf['div_type'] == dtype]
            if sub.empty:
                continue

            n_sig = len(sub)
            label = DIV_LABEL[dtype]
            exp   = '▲ UP' if EXPECTED_DIR[dtype] == +1 else '▼ DOWN'
            print(f"\n    {label}  ({n_sig} signals)  expected: {exp}")
            print(f"    {'Horizon':<10} {'N':>4}  {'Avg Move':>10}  {'Win%':>6}  "
                  f"{'Avg MFE':>9}  {'Avg MAE':>9}  {'MFE/MAE':>8}")
            print(f"    {'─'*10} {'─'*4}  {'─'*10}  {'─'*6}  "
                  f"{'─'*9}  {'─'*9}  {'─'*8}")

            for col in move_cols:
                valid = sub[col].dropna()
                n_v   = len(valid)
                if n_v == 0:
                    continue

                suffix  = col[len('move_'):]          # '5b', '10b', ..., 'eod'
                mfe_col = f'mfe_{suffix}'
                mae_col = f'mae_{suffix}'

                avg_move = valid.mean()
                win_pct  = (valid > 0).mean() * 100
                avg_mfe  = sub[mfe_col].dropna().mean() if mfe_col in sub.columns else np.nan
                avg_mae  = sub[mae_col].dropna().mean() if mae_col in sub.columns else np.nan

                if suffix == 'eod':
                    h_label = 'EOD'
                else:
                    h_label = f'{suffix[:-1]} bars'   # '5b' → '5 bars'

                ratio_str = (f'{avg_mfe / avg_mae:.2f}x'
                             if (not np.isnan(avg_mae) and avg_mae > 0) else 'n/a')
                print(f"    {h_label:<10} {n_v:>4}  "
                      f"{avg_move:>+10.3f}  {win_pct:>5.1f}%  "
                      f"{avg_mfe:>+9.3f}  {avg_mae:>+9.3f}  "
                      f"{ratio_str:>8}")


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, horizons: list, output_path: str = None):
    if df.empty:
        return

    BG, PANEL = '#0d1117', '#161b22'
    WHITE, GRAY = '#e6edf3', '#8b949e'
    COLORS = {
        'bull':        '#00ff1e',
        'hidden_bull': '#00b9ff',
        'bear':        '#ff4444',
        'hidden_bear': '#ffcc00',
    }

    h_labels  = [f'{h}b' for h in horizons] + ['eod']
    move_cols = [f'move_{h}' for h in h_labels]
    x_labels  = [f'{h} bars' for h in horizons] + ['EOD']

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle('SPO Extreme Divergence — Price Movement After Signal',
                 color=WHITE, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           top=0.92, bottom=0.07, left=0.07, right=0.97)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)
        ax.axhline(0, color=GRAY, lw=0.6, ls='--', alpha=0.6)

    # ── Panel 1 (top-left): Avg move at each horizon by div_type ──
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, 'Avg price move in expected direction (pts)')
    x = np.arange(len(x_labels))
    w = 0.18
    for i, (dtype, label) in enumerate(DIV_LABEL.items()):
        sub = df[df['div_type'] == dtype]
        if sub.empty:
            continue
        avgs = [sub[col].dropna().mean() for col in move_cols]
        ax1.bar(x + i * w, avgs, width=w, color=COLORS[dtype],
                alpha=0.85, label=label.strip())
    ax1.set_xticks(x + w * 1.5)
    ax1.set_xticklabels(x_labels, color=GRAY, fontsize=8)
    ax1.set_ylabel('Points (+ = favorable)', color=GRAY, fontsize=9)
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE, loc='upper left')

    # ── Panel 2 (top-right): Win rate at each horizon ──
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, 'Win rate at each horizon (% signals moved in expected direction)')
    ax2.axhline(50, color=GRAY, lw=0.8, ls=':', alpha=0.5)
    for i, (dtype, label) in enumerate(DIV_LABEL.items()):
        sub = df[df['div_type'] == dtype]
        if sub.empty:
            continue
        win_rates = []
        for col in move_cols:
            valid = sub[col].dropna()
            win_rates.append((valid > 0).mean() * 100 if len(valid) > 0 else np.nan)
        ax2.plot(x_labels, win_rates, color=COLORS[dtype], lw=2,
                 marker='o', markersize=5, label=label.strip())
    ax2.set_ylabel('Win rate (%)', color=GRAY, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', labelrotation=20)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)

    # ── Panel 3 (bottom-left): EOD move distribution histograms ──
    ax3 = fig.add_subplot(gs[1, 0])
    style(ax3, 'EOD move distribution (pts, expected direction)')
    for dtype, label in DIV_LABEL.items():
        sub = df[df['div_type'] == dtype]['move_eod'].dropna()
        if sub.empty:
            continue
        ax3.hist(sub, bins=25, color=COLORS[dtype], alpha=0.5,
                 label=label.strip(), density=True)
    ax3.set_xlabel('Move in pts (+ = favorable, - = adverse)', color=GRAY, fontsize=9)
    ax3.set_ylabel('Density', color=GRAY, fontsize=9)
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)

    # ── Panel 4 (bottom-right): MFE vs MAE at EOD by div_type ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    ax4.tick_params(colors=GRAY, labelsize=9)
    for sp in ax4.spines.values():
        sp.set_color('#30363d')
    ax4.set_title('Avg MFE vs Avg MAE at EOD by div type',
                  color=WHITE, fontsize=10, fontweight='bold', pad=6)

    dtypes_present = []
    avg_mfes, avg_maes = [], []
    for dtype, label in DIV_LABEL.items():
        sub = df[df['div_type'] == dtype]
        mfe_m = sub['mfe_eod'].dropna().mean()
        mae_m = sub['mae_eod'].dropna().mean()
        if np.isnan(mfe_m) or np.isnan(mae_m):
            continue
        dtypes_present.append(label.strip())
        avg_mfes.append(mfe_m)
        avg_maes.append(mae_m)
        ax4.scatter(mae_m, mfe_m, color=COLORS[dtype], s=120, zorder=5,
                    label=f"{label.strip()}  (MFE {mfe_m:+.2f} / MAE {mae_m:+.2f})")

    if avg_mfes:
        lim = max(max(avg_mfes), max(avg_maes)) * 1.2
        ax4.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls='--', alpha=0.5,
                 label='1:1 MFE=MAE')
        ax4.set_xlim(left=0)
        ax4.set_ylim(bottom=0)
    ax4.set_xlabel('Avg MAE — adverse excursion (pts)', color=GRAY, fontsize=9)
    ax4.set_ylabel('Avg MFE — favorable excursion (pts)', color=GRAY, fontsize=9)
    ax4.legend(fontsize=7.5, facecolor=PANEL, labelcolor=WHITE)

    # Add ticker + signal count annotation
    tickers = df['ticker'].unique()
    ann = (f"Tickers: {', '.join(tickers)}  |  "
           f"{len(df)} signals  |  "
           f"|SPO| threshold applied at pivot bar")
    fig.text(0.5, 0.01, ann, ha='center', color=GRAY, fontsize=8)

    out = (output_path.replace('.csv', '_price_movement.png')
           if output_path else 'backtest_price_movement.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SPO Extreme Divergence — price movement after signal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',    type=str,              help='Single ticker (default: all in DB)')
    parser.add_argument('--start',     type=str,              help='Start date YYYY-MM-DD')
    parser.add_argument('--end',       type=str,              help='End date YYYY-MM-DD')
    parser.add_argument('--tf',        type=int, nargs='+',   help='Timeframes in minutes (default: 3 5 10)')
    parser.add_argument('--horizons',  type=int, nargs='+',   help='Bar horizons to measure (default: 5 10 15 30)')
    parser.add_argument('--threshold', type=float,            help=f'|SPO| extreme filter (default: {DEFAULT_EXTREME_THRESHOLD})')
    parser.add_argument('--min-bars',  type=int, default=1,   help='Minimum RTH bars remaining after signal (default: 1)')
    parser.add_argument('--output',    type=str,              help='Optional CSV output path')
    parser.add_argument('--no-plot',   action='store_true',   help='Skip chart generation')
    parser.add_argument('--catalog',   action='store_true',   help='Show DB contents and exit')
    args = parser.parse_args()

    timeframes = args.tf        if args.tf        else DIV_TIMEFRAMES
    horizons   = args.horizons  if args.horizons  else DEFAULT_HORIZONS
    threshold  = args.threshold if args.threshold else DEFAULT_EXTREME_THRESHOLD

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
    print("  SPO EXTREME DIVERGENCE — PRICE MOVEMENT BACKTEST")
    print(f"  Tickers    : {', '.join(tickers)}")
    print(f"  Timeframes : {timeframes} min")
    print(f"  Period     : {args.start or 'all'} → {args.end or 'latest'}")
    print(f"  Filter     : |SPO at pivot| > {threshold}")
    print(f"  Horizons   : {horizons} bars + EOD")
    print(f"  Pivots     : lbL={DIV_LB_LEFT}  lbR={DIV_LB_RIGHT}  "
          f"range=[{DIV_RANGE_MIN},{DIV_RANGE_MAX}] bars")
    print(f"  Min bars   : {args.min_bars} RTH bars remaining after signal")

    all_dfs = []
    for ticker in tickers:
        print(f"\n  Loading {ticker}...")
        df = load_bars(ticker, '1m_full', args.start, args.end, con)
        if df.empty:
            print("    [!] No data.")
            continue
        print(f"    {len(df):,} bars  |  "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        result = scan_price_movement(
            ticker, df, timeframes, horizons,
            extreme_threshold=threshold,
            min_bars_ahead=args.min_bars,
            start=args.start,
            end=args.end,
        )
        print_stats(result, horizons, ticker)
        all_dfs.append(result)

    con.close()

    if not all_dfs:
        return

    combined = (pd.concat(all_dfs, ignore_index=True)
                .sort_values(['date', 'time'])
                .reset_index(drop=True))

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\n  Saved: {args.output}  ({len(combined)} rows)")

    if not args.no_plot:
        plot_results(combined, horizons, args.output)

    # Cross-ticker summary
    if len(tickers) > 1:
        print(f"\n\n  {'═'*68}")
        print(f"  COMBINED — {len(tickers)} tickers  |  {len(combined)} signals")
        print(f"  {'═'*68}")
        for dtype, label in DIV_LABEL.items():
            sub = combined[combined['div_type'] == dtype]
            if sub.empty:
                continue
            eod_valid = sub['move_eod'].dropna()
            if eod_valid.empty:
                continue
            wr = (eod_valid > 0).mean() * 100
            avg = eod_valid.mean()
            print(f"    {label}  {len(sub):4d} signals  "
                  f"EOD avg {avg:+.3f} pts  win {wr:.1f}%")


if __name__ == '__main__':
    main()
