"""
backtest/run_spo_divergence.py
================================
SPO Divergence Scanner — finds every date/time where the Saty Phase
Oscillator produced a Regular or Hidden divergence on 3-min, 5-min,
or 10-min RTH bars.

This is a signal log, not a PnL backtest.  Each row in the output
tells you exactly when a divergence fired, on which timeframe, and
what type it was — so you can review price action at those moments.

USAGE
-----
    python backtest/run_spo_divergence.py                      # all tickers, 3/5/10 min
    python backtest/run_spo_divergence.py --ticker SPY
    python backtest/run_spo_divergence.py --ticker QQQ --start 2022-01-01
    python backtest/run_spo_divergence.py --tf 3 5            # only 3-min and 5-min
    python backtest/run_spo_divergence.py --output divs.csv
    python backtest/run_spo_divergence.py --catalog

OUTPUT COLUMNS
--------------
  ticker, timeframe_min, timestamp, date, time,
  div_type   : bull | bear | hidden_bull | hidden_bear
  osc_value  : SPO oscillator value at the PIVOT bar
  price_level: low (bull/hidden_bull) or high (bear/hidden_bear) at pivot
  osc_zone   : zone label at the pivot (extreme/dist/neutral/accum)
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
    WARMUP_DAYS,
)
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.sessions import filter_rth, resample_to
from core.indicators import calc_spo, calc_spo_divergence


# ── SPO zone label helper ─────────────────────────────────────────────────────

def _zone(v: float) -> str:
    """Map SPO value to readable zone name."""
    if np.isnan(v):       return 'n/a'
    if v >=  100:         return 'extreme_up'
    if v >=   61.8:       return 'distribution'
    if v >=   23.6:       return 'neutral_up'
    if v >   -23.6:       return 'neutral'
    if v >   -61.8:       return 'neutral_down'
    if v >  -100:         return 'accumulation'
    return 'extreme_down'


# ── Scan one ticker on one timeframe ─────────────────────────────────────────

def scan_timeframe(
    ticker: str,
    df_rth: pd.DataFrame,
    tf_min: int,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Resample RTH 1-min bars to tf_min, compute SPO + divergence, return events.

    Returns a DataFrame where each row is one divergence signal, with columns:
      ticker, timeframe_min, timestamp, date, time, div_type,
      osc_value, price_level, osc_zone
    """
    # Resample to target timeframe
    df = resample_to(df_rth, rule=f'{tf_min}min')
    if df.empty:
        return pd.DataFrame()

    # Date range filter
    if start:
        df = df[df['timestamp'] >= pd.Timestamp(start)]
    if end:
        df = df[df['timestamp'] <= pd.Timestamp(end)]

    # Skip warmup rows so indicators are stable
    warmup = max(WARMUP_DAYS, 40)
    if len(df) <= warmup:
        return pd.DataFrame()
    df = df.iloc[warmup:].reset_index(drop=True)

    # Compute SPO (continuous across the full resampled series)
    spo_df = calc_spo(df)
    osc    = spo_df['spo']

    # Compute divergence
    div_df = calc_spo_divergence(
        df, osc,
        lb_left=DIV_LB_LEFT,
        lb_right=DIV_LB_RIGHT,
        range_lower=DIV_RANGE_MIN,
        range_upper=DIV_RANGE_MAX,
    )

    # Also need the pivot bar values for reporting.
    # Re-run pivot detection to get (pivot_bar, confirm_bar) pairs with values.
    osc_arr  = osc.values.astype(float)
    low_arr  = df['low'].values.astype(float)
    high_arr = df['high'].values.astype(float)
    ts_arr   = df['timestamp'].values

    # Build lookup: confirm_bar → (pivot_bar, div_types list)
    # We'll pull osc/price at the pivot bar, not the confirmation bar.
    pl_list = []
    ph_list = []
    n = len(osc_arr)
    lb_l, lb_r = DIV_LB_LEFT, DIV_LB_RIGHT

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

    events = []

    # Bull / hidden-bull from pivot lows
    for k in range(1, len(pl_list)):
        piv_p, conf_p = pl_list[k - 1]
        piv_c, conf_c = pl_list[k]
        gap = conf_c - conf_p
        if not (DIV_RANGE_MIN <= gap <= DIV_RANGE_MAX):
            continue

        o_p, o_c = osc_arr[piv_p], osc_arr[piv_c]
        l_p, l_c = low_arr[piv_p], low_arr[piv_c]
        if np.isnan(o_p) or np.isnan(o_c):
            continue

        confirm_ts = pd.Timestamp(ts_arr[conf_c])

        if l_c < l_p and o_c > o_p:
            events.append({
                'ticker':       ticker,
                'timeframe_min': tf_min,
                'timestamp':    confirm_ts,
                'date':         confirm_ts.date(),
                'time':         confirm_ts.strftime('%H:%M'),
                'div_type':     'bull',
                'osc_value':    round(o_c, 2),
                'price_level':  round(l_c, 2),
                'osc_zone':     _zone(o_c),
            })

        if l_c > l_p and o_c < o_p:
            events.append({
                'ticker':       ticker,
                'timeframe_min': tf_min,
                'timestamp':    confirm_ts,
                'date':         confirm_ts.date(),
                'time':         confirm_ts.strftime('%H:%M'),
                'div_type':     'hidden_bull',
                'osc_value':    round(o_c, 2),
                'price_level':  round(l_c, 2),
                'osc_zone':     _zone(o_c),
            })

    # Bear / hidden-bear from pivot highs
    for k in range(1, len(ph_list)):
        piv_p, conf_p = ph_list[k - 1]
        piv_c, conf_c = ph_list[k]
        gap = conf_c - conf_p
        if not (DIV_RANGE_MIN <= gap <= DIV_RANGE_MAX):
            continue

        o_p, o_c = osc_arr[piv_p], osc_arr[piv_c]
        h_p, h_c = high_arr[piv_p], high_arr[piv_c]
        if np.isnan(o_p) or np.isnan(o_c):
            continue

        confirm_ts = pd.Timestamp(ts_arr[conf_c])

        if h_c > h_p and o_c < o_p:
            events.append({
                'ticker':       ticker,
                'timeframe_min': tf_min,
                'timestamp':    confirm_ts,
                'date':         confirm_ts.date(),
                'time':         confirm_ts.strftime('%H:%M'),
                'div_type':     'bear',
                'osc_value':    round(o_c, 2),
                'price_level':  round(h_c, 2),
                'osc_zone':     _zone(o_c),
            })

        if h_c < h_p and o_c > o_p:
            events.append({
                'ticker':       ticker,
                'timeframe_min': tf_min,
                'timestamp':    confirm_ts,
                'date':         confirm_ts.date(),
                'time':         confirm_ts.strftime('%H:%M'),
                'div_type':     'hidden_bear',
                'osc_value':    round(o_c, 2),
                'price_level':  round(h_c, 2),
                'osc_zone':     _zone(o_c),
            })

    return pd.DataFrame(events)


# ── Per-ticker scanner ────────────────────────────────────────────────────────

def scan_ticker(ticker: str, df_full: pd.DataFrame,
                timeframes: list, start: str = None, end: str = None) -> pd.DataFrame:
    """Scan all requested timeframes for one ticker. Returns all divergence events."""
    df_rth = filter_rth(df_full)

    all_events = []
    for tf in timeframes:
        events = scan_timeframe(ticker, df_rth, tf, start, end)
        if not events.empty:
            all_events.append(events)

    return pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(events: pd.DataFrame, ticker: str):
    if events.empty:
        print(f"  [{ticker}] No divergences found.")
        return

    total = len(events)
    print(f"\n  {'─'*62}")
    print(f"  {ticker}  |  "
          f"{str(events['date'].min())} → {str(events['date'].max())}")
    print(f"  {'─'*62}")
    print(f"  Total divergences: {total}")

    # By timeframe
    print(f"\n  By timeframe:")
    for tf in sorted(events['timeframe_min'].unique()):
        sub = events[events['timeframe_min'] == tf]
        by_type = sub['div_type'].value_counts()
        parts   = '  '.join(f"{t}={n}" for t, n in by_type.items())
        print(f"    {tf:2d}min  →  {len(sub):4d} signals   {parts}")

    # By type
    print(f"\n  By divergence type:")
    for dtype, count in events['div_type'].value_counts().items():
        pct = count / total * 100
        print(f"    {dtype:<15s}  {count:4d}  ({pct:.1f}%)")

    # By zone (where does the oscillator tend to be when divergences form?)
    print(f"\n  By SPO zone at pivot:")
    for zone, count in events['osc_zone'].value_counts().items():
        pct = count / total * 100
        print(f"    {zone:<20s}  {count:4d}  ({pct:.1f}%)")

    # Recent 10 signals
    print(f"\n  Most recent 10 divergences:")
    recent = events.sort_values('timestamp', ascending=False).head(10)
    for _, r in recent.iterrows():
        print(f"    {r['date']}  {r['time']}  "
              f"{r['timeframe_min']:2d}min  "
              f"{r['div_type']:<15s}  "
              f"SPO={r['osc_value']:+7.2f}  "
              f"zone={r['osc_zone']}")


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(all_events: pd.DataFrame, output_path: str = None):
    if all_events.empty:
        return

    BG, PANEL, GREEN, RED, YELLOW, BLUE, GRAY, WHITE = (
        '#0d1117', '#161b22', '#00ff1e', '#ff4444',
        '#ffcc00', '#00b9ff', '#8b949e', '#e6edf3'
    )

    TYPE_CLR = {
        'bull':        GREEN,
        'hidden_bull': '#00cc66',
        'bear':        RED,
        'hidden_bear': '#cc3333',
    }

    all_events = all_events.copy()
    all_events['year_month'] = pd.to_datetime(all_events['date']).dt.to_period('M')

    tickers    = sorted(all_events['ticker'].unique())
    timeframes = sorted(all_events['timeframe_min'].unique())
    div_types  = ['bull', 'bear', 'hidden_bull', 'hidden_bear']

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle('SPO Divergence Scanner — Signal Log',
                 color=WHITE, fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)

    # ── Row 1: signals per month (stacked by type, full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    monthly = (all_events.groupby(['year_month', 'div_type'])
               .size().unstack(fill_value=0)
               .reindex(columns=div_types, fill_value=0))
    x = np.arange(len(monthly))
    bottoms = np.zeros(len(monthly))
    for dt in div_types:
        if dt in monthly.columns:
            vals = monthly[dt].values
            ax1.bar(x, vals, bottom=bottoms, color=TYPE_CLR[dt],
                    alpha=0.8, label=dt.replace('_', ' ').title(), width=0.9)
            bottoms += vals
    ax1.set_xticks(x[::max(1, len(x) // 24)])
    ax1.set_xticklabels(
        [str(p) for p in monthly.index[::max(1, len(x) // 24)]],
        rotation=45, fontsize=7, color=GRAY)
    sax(ax1, 'Divergence signals per month (all tickers + timeframes)')
    ax1.legend(fontsize=9, facecolor=PANEL, labelcolor=WHITE, loc='upper right')
    ax1.set_ylabel('Count', color=GRAY)

    # ── Row 2 left: signals by ticker × timeframe heatmap ──
    ax2 = fig.add_subplot(gs[1, 0])
    pivot_tbl = (all_events.groupby(['ticker', 'timeframe_min'])
                 .size().unstack(fill_value=0))
    im = ax2.imshow(pivot_tbl.values, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(pivot_tbl.columns)))
    ax2.set_xticklabels([f"{c}min" for c in pivot_tbl.columns], color=GRAY)
    ax2.set_yticks(range(len(pivot_tbl.index)))
    ax2.set_yticklabels(pivot_tbl.index, color=GRAY)
    for i in range(pivot_tbl.shape[0]):
        for j in range(pivot_tbl.shape[1]):
            ax2.text(j, i, str(pivot_tbl.values[i, j]),
                     ha='center', va='center', fontsize=9, color='black')
    plt.colorbar(im, ax=ax2, label='Signal count').ax.tick_params(colors=GRAY, labelsize=7)
    sax(ax2, 'Signal count by ticker × timeframe')

    # ── Row 2 right: divergence type breakdown by timeframe ──
    ax3 = fig.add_subplot(gs[1, 1])
    tf_type = (all_events.groupby(['timeframe_min', 'div_type'])
               .size().unstack(fill_value=0)
               .reindex(columns=div_types, fill_value=0))
    x3 = np.arange(len(tf_type))
    w3 = 0.2
    for i, dt in enumerate(div_types):
        if dt in tf_type.columns:
            ax3.bar(x3 + i * w3, tf_type[dt], width=w3,
                    color=TYPE_CLR[dt], alpha=0.85, label=dt.replace('_', ' ').title())
    ax3.set_xticks(x3 + w3 * 1.5)
    ax3.set_xticklabels([f"{c}min" for c in tf_type.index], color=GRAY)
    sax(ax3, 'Divergence type by timeframe')
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    ax3.set_ylabel('Count', color=GRAY)

    # ── Row 3 left: intraday time distribution (when do divs fire?) ──
    ax4 = fig.add_subplot(gs[2, 0])
    all_events['hour'] = pd.to_datetime(
        all_events['time'], format='%H:%M').dt.hour
    time_dist = (all_events.groupby(['hour', 'div_type'])
                 .size().unstack(fill_value=0)
                 .reindex(columns=div_types, fill_value=0))
    x4 = np.arange(len(time_dist))
    b4 = np.zeros(len(time_dist))
    for dt in div_types:
        if dt in time_dist.columns:
            v = time_dist[dt].values
            ax4.bar(x4, v, bottom=b4, color=TYPE_CLR[dt], alpha=0.8,
                    label=dt.replace('_', ' ').title(), width=0.9)
            b4 += v
    ax4.set_xticks(x4)
    ax4.set_xticklabels([f"{h:02d}:00" for h in time_dist.index],
                        rotation=45, fontsize=8, color=GRAY)
    sax(ax4, 'Intraday time distribution of divergences (ET)')
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE)
    ax4.set_ylabel('Count', color=GRAY)

    # ── Row 3 right: SPO zone distribution ──
    ax5 = fig.add_subplot(gs[2, 1])
    zone_order = ['extreme_up', 'distribution', 'neutral_up', 'neutral',
                  'neutral_down', 'accumulation', 'extreme_down']
    zone_cnts  = all_events['osc_zone'].value_counts().reindex(zone_order, fill_value=0)
    zone_clrs  = [RED, '#ff6600', YELLOW, GRAY, YELLOW, '#00cc66', GREEN]
    bars = ax5.barh(zone_order, zone_cnts.values,
                    color=zone_clrs, alpha=0.8, height=0.6)
    for bar, val in zip(bars, zone_cnts.values):
        ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=8, color=WHITE)
    sax(ax5, 'SPO zone where divergences form (pivot bar)')
    ax5.set_xlabel('Count', color=GRAY)
    ax5.invert_yaxis()

    out = (output_path.replace('.csv', '_div_chart.png')
           if output_path else 'backtest_spo_divergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SPO Divergence Scanner — 3/5/10 min RTH bars',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',  type=str,            help='Single ticker (default: all in DB)')
    parser.add_argument('--start',   type=str,            help='Start date YYYY-MM-DD')
    parser.add_argument('--end',     type=str,            help='End date YYYY-MM-DD')
    parser.add_argument('--tf',      type=int, nargs='+', help='Timeframe(s) in minutes (default: 3 5 10)')
    parser.add_argument('--output',  type=str,            help='Save results to CSV path')
    parser.add_argument('--catalog', action='store_true', help='Show DB data coverage and exit')
    parser.add_argument('--no-plot', action='store_true', help='Skip chart')
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

    print("=" * 64)
    print("SPO Divergence Scanner")
    print(f"Tickers     : {', '.join(tickers)}")
    print(f"Timeframes  : {timeframes} min")
    print(f"Period      : {args.start or 'all'} → {args.end or 'latest'}")
    print(f"Pivot params: lbL={DIV_LB_LEFT}  lbR={DIV_LB_RIGHT}  "
          f"range=[{DIV_RANGE_MIN}, {DIV_RANGE_MAX}] bars")
    print("=" * 64)

    all_events = []
    for ticker in tickers:
        print(f"\nScanning {ticker}...")
        df = load_bars(ticker, '1m_full', args.start, args.end, con)
        if df.empty:
            print("  [!] No data.")
            continue
        print(f"  {len(df):,} 1-min bars  |  "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        events = scan_ticker(ticker, df, timeframes, args.start, args.end)
        print_stats(events, ticker)
        all_events.append(events)

    con.close()
    if not all_events:
        return

    combined = pd.concat(all_events, ignore_index=True)
    combined  = combined.sort_values('timestamp').reset_index(drop=True)

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\nDivergences saved: {args.output}  ({len(combined)} rows)")

    if not args.no_plot:
        plot_results(combined, args.output)

    # Cross-ticker combined summary
    if len(tickers) > 1:
        print(f"\n{'='*64}")
        print(f"COMBINED — {len(tickers)} tickers  |  {len(combined)} total divergences")
        for tf in sorted(combined['timeframe_min'].unique()):
            sub = combined[combined['timeframe_min'] == tf]
            by_t = sub['div_type'].value_counts()
            print(f"  {tf:2d}min  {len(sub):5d} signals  "
                  + '  '.join(f"{t}={n}" for t, n in by_t.items()))


if __name__ == '__main__':
    main()
