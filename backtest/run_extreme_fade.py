"""
backtest/run_extreme_fade.py
============================
Extreme SPO Divergence Fade — backtest on 10-minute candles.

Setup
-----
  • Resample 1-min ETH bars to 10-min on the fly (no DB pre-aggregation needed)
  • Compute SPO on the continuous 10-min ETH stream
  • Detect extreme divergences on RTH bars (|SPO at pivot| > 100)
  • Bull divergence at SPO < −100  → LONG  (fade the oversold extreme)
  • Bear divergence at SPO > +100  → SHORT (fade the overbought extreme)
  • Stop   : price pivot low (long) or price pivot high (short)
  • Target : 2× risk  (configurable via TARGET_R)
  • EOD    : exit at last RTH bar close if stop/target not hit

USAGE
-----
    python backtest/run_extreme_fade.py
    python backtest/run_extreme_fade.py --ticker SPY
    python backtest/run_extreme_fade.py --start 2022-01-01
    python backtest/run_extreme_fade.py --target-r 3
    python backtest/run_extreme_fade.py --output fade.csv
    python backtest/run_extreme_fade.py --catalog
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
    DIV_LB_LEFT, DIV_LB_RIGHT, DIV_RANGE_MIN, DIV_RANGE_MAX,
    WARMUP_DAYS, RTH_START, RTH_END,
)
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.sessions import filter_eth, resample_to
from core.indicators import calc_spo

EXTREME_THRESHOLD = 100.0
TIMEFRAME_MIN     = 10

DIV_LABEL = {
    'bull':        'Regular Bull',
    'bear':        'Regular Bear',
    'hidden_bull': 'Hidden Bull',
    'hidden_bear': 'Hidden Bear',
}


# ── Signal detection ──────────────────────────────────────────────────────────

def find_signals(ticker: str, df_full: pd.DataFrame,
                 start: str = None, end: str = None) -> tuple:
    """
    Resample 1-min ETH bars to 10-min, compute SPO on the full ETH stream,
    detect extreme divergences on RTH bars.

    Returns (signals_df, df_rth_10m).
    signals_df  — one row per signal with entry / stop / target / direction
    df_rth_10m  — the RTH 10-min DataFrame used for trade simulation
    """
    df_eth    = filter_eth(df_full)
    df_eth_rs = resample_to(df_eth, rule=f'{TIMEFRAME_MIN}min')
    if df_eth_rs.empty:
        return pd.DataFrame(), pd.DataFrame()

    warmup = max(WARMUP_DAYS, 40)
    if len(df_eth_rs) <= warmup:
        return pd.DataFrame(), pd.DataFrame()
    df_eth_rs = df_eth_rs.iloc[warmup:].reset_index(drop=True)

    # SPO on full ETH stream so EMAs are continuous
    spo_df    = calc_spo(df_eth_rs)
    df_eth_rs = df_eth_rs.copy()
    df_eth_rs['spo'] = spo_df['spo'].values

    # Date range AFTER SPO so history is preserved
    if start:
        df_eth_rs = df_eth_rs[df_eth_rs['timestamp'] >= pd.Timestamp(start)]
    if end:
        df_eth_rs = df_eth_rs[df_eth_rs['timestamp'] <= pd.Timestamp(end)]
    if df_eth_rs.empty:
        return pd.DataFrame(), pd.DataFrame()

    # RTH bars for signal detection and trade simulation
    rth_mask = (
        (df_eth_rs['timestamp'].dt.time >= RTH_START) &
        (df_eth_rs['timestamp'].dt.time <= RTH_END)
    )
    df_rth = df_eth_rs[rth_mask].reset_index(drop=True)
    if df_rth.empty:
        return pd.DataFrame(), pd.DataFrame()

    osc_arr   = df_rth['spo'].values.astype(float)
    low_arr   = df_rth['low'].values.astype(float)
    high_arr  = df_rth['high'].values.astype(float)
    close_arr = df_rth['close'].values.astype(float)
    ts_arr    = df_rth['timestamp'].values
    n         = len(osc_arr)
    lb_l, lb_r = DIV_LB_LEFT, DIV_LB_RIGHT

    # Pivot detection (same logic as run_extreme_divergence.py)
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

    signals = []

    # ── Bull divergences → LONG (SPO must be < −100, i.e. oversold) ──────────
    for k in range(1, len(pl_list)):
        piv_p, conf_p = pl_list[k - 1]
        piv_c, conf_c = pl_list[k]
        if not (DIV_RANGE_MIN <= (conf_c - conf_p) <= DIV_RANGE_MAX):
            continue
        o_p, o_c = osc_arr[piv_p], osc_arr[piv_c]
        l_p, l_c = low_arr[piv_p],  low_arr[piv_c]
        if np.isnan(o_p) or np.isnan(o_c):
            continue
        if o_c >= -EXTREME_THRESHOLD:
            continue  # only fade the oversold extreme

        entry = close_arr[conf_c]
        stop  = l_c            # price pivot low is the natural stop
        risk  = entry - stop
        if risk <= 0:
            continue

        div_type = None
        if l_c < l_p and o_c > o_p:
            div_type = 'bull'
        elif l_c > l_p and o_c < o_p:
            div_type = 'hidden_bull'
        if div_type is None:
            continue

        ts = pd.Timestamp(ts_arr[conf_c])
        signals.append({
            'ticker':    ticker,
            'timestamp': ts,
            'date':      ts.date(),
            'time':      ts.strftime('%H:%M'),
            'div_type':  div_type,
            'direction': 'long',
            'osc_value': round(o_c, 2),
            'entry':     round(entry, 2),
            'stop':      round(stop,  2),
            'risk':      round(risk,  2),
            'conf_bar':  conf_c,
        })

    # ── Bear divergences → SHORT (SPO must be > +100, i.e. overbought) ───────
    for k in range(1, len(ph_list)):
        piv_p, conf_p = ph_list[k - 1]
        piv_c, conf_c = ph_list[k]
        if not (DIV_RANGE_MIN <= (conf_c - conf_p) <= DIV_RANGE_MAX):
            continue
        o_p, o_c = osc_arr[piv_p], osc_arr[piv_c]
        h_p, h_c = high_arr[piv_p], high_arr[piv_c]
        if np.isnan(o_p) or np.isnan(o_c):
            continue
        if o_c <= EXTREME_THRESHOLD:
            continue  # only fade the overbought extreme

        entry = close_arr[conf_c]
        stop  = h_c            # price pivot high is the natural stop
        risk  = stop - entry
        if risk <= 0:
            continue

        div_type = None
        if h_c > h_p and o_c < o_p:
            div_type = 'bear'
        elif h_c < h_p and o_c > o_p:
            div_type = 'hidden_bear'
        if div_type is None:
            continue

        ts = pd.Timestamp(ts_arr[conf_c])
        signals.append({
            'ticker':    ticker,
            'timestamp': ts,
            'date':      ts.date(),
            'time':      ts.strftime('%H:%M'),
            'div_type':  div_type,
            'direction': 'short',
            'osc_value': round(o_c, 2),
            'entry':     round(entry, 2),
            'stop':      round(stop,  2),
            'risk':      round(risk,  2),
            'conf_bar':  conf_c,
        })

    return pd.DataFrame(signals) if signals else pd.DataFrame(), df_rth


# ── Trade simulator ───────────────────────────────────────────────────────────

def simulate_trade(sig: pd.Series, df_rth: pd.DataFrame,
                   target_r: float) -> dict:
    """
    Simulate entry → stop/target/EOD on 10-min RTH bars following the signal.

    Stop is checked on the bar's low (long) or high (short) — worst-case fill.
    Target is checked on the bar's high (long) or low (short).
    EOD exits at the last bar's close on the trade date.
    """
    conf_bar   = sig['conf_bar']
    entry      = sig['entry']
    stop       = sig['stop']
    risk       = sig['risk']
    direction  = sig['direction']
    trade_date = sig['date']

    target = (entry + target_r * risk) if direction == 'long' \
             else (entry - target_r * risk)

    # Bars after the entry bar, same day only
    after = df_rth.iloc[conf_bar + 1:].copy()
    after = after[after['timestamp'].dt.date == trade_date].reset_index(drop=True)

    exit_price  = None
    exit_reason = None

    for _, bar in after.iterrows():
        if direction == 'long':
            if bar['low'] <= stop:
                exit_price, exit_reason = stop, 'stop'
                break
            if bar['high'] >= target:
                exit_price, exit_reason = target, 'target'
                break
        else:
            if bar['high'] >= stop:
                exit_price, exit_reason = stop, 'stop'
                break
            if bar['low'] <= target:
                exit_price, exit_reason = target, 'target'
                break

    if exit_price is None:
        exit_price  = entry if after.empty else after.iloc[-1]['close']
        exit_reason = 'eod'

    pnl_pts = (exit_price - entry) if direction == 'long' \
              else (entry - exit_price)
    pnl_r   = pnl_pts / risk if risk > 0 else 0.0

    return {
        'target':      round(target,     2),
        'exit':        round(exit_price, 2),
        'exit_reason': exit_reason,
        'pnl_pts':     round(pnl_pts, 2),
        'pnl_r':       round(pnl_r,   3),
    }


# ── Per-ticker runner ─────────────────────────────────────────────────────────

def run_ticker(ticker: str, df_full: pd.DataFrame,
               start: str, end: str, target_r: float) -> pd.DataFrame:
    signals_df, df_rth = find_signals(ticker, df_full, start, end)
    if signals_df.empty:
        return pd.DataFrame()

    rows = []
    for _, sig in signals_df.iterrows():
        result = simulate_trade(sig, df_rth, target_r)
        rows.append({**sig.to_dict(), **result})

    return pd.DataFrame(rows)


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(results: pd.DataFrame, label: str, target_r: float):
    if results.empty:
        print(f"\n  [{label}]  No signals found.\n")
        return

    total  = len(results)
    wins   = (results['pnl_r'] > 0).sum()
    losses = (results['pnl_r'] <= 0).sum()
    wr     = wins / total * 100

    gw = results[results['pnl_r'] > 0]['pnl_r'].sum()
    gl = abs(results[results['pnl_r'] <= 0]['pnl_r'].sum())
    pf = gw / gl if gl > 0 else float('inf')

    avg_r = results['pnl_r'].mean()
    cum_r = results['pnl_r'].sum()

    print(f"\n  ┌{'─'*64}┐")
    print(f"  │  {label:<8}  "
          f"{str(results['date'].min())} → {str(results['date'].max()):<16}"
          f"  {total:3d} trades  │")
    print(f"  └{'─'*64}┘")
    print(f"\n  Win rate     : {wr:5.1f}%  ({wins}W / {losses}L)")
    print(f"  Profit factor: {pf:5.2f}")
    print(f"  Avg R/trade  : {avg_r:+.3f}R")
    print(f"  Total R      : {cum_r:+.2f}R")
    print(f"  Target       : {target_r}R   Stop: pivot high/low")

    print(f"\n  By direction:")
    for d in ['long', 'short']:
        sub = results[results['direction'] == d]
        if sub.empty:
            continue
        dwr = (sub['pnl_r'] > 0).sum() / len(sub) * 100
        davg = sub['pnl_r'].mean()
        print(f"    {d:<6}  {len(sub):3d} trades  "
              f"WR={dwr:.0f}%  avg={davg:+.3f}R")

    print(f"\n  By divergence type:")
    for dtype, sub in results.groupby('div_type'):
        dwr  = (sub['pnl_r'] > 0).sum() / len(sub) * 100
        davg = sub['pnl_r'].mean()
        print(f"    {DIV_LABEL.get(dtype, dtype):<15}  {len(sub):3d}  "
              f"WR={dwr:.0f}%  avg={davg:+.3f}R")

    print(f"\n  By exit reason:")
    for reason, sub in results.groupby('exit_reason'):
        print(f"    {reason:<8}  {len(sub):3d}  ({len(sub)/total*100:.0f}%)")

    results = results.copy()
    results['year'] = pd.to_datetime(results['date']).dt.year
    print(f"\n  By year:")
    print(f"  {'Year':<6} {'#':>4}  {'WR':>6}  {'Avg R':>7}  {'Cum R':>7}")
    print(f"  {'─'*6} {'─'*4}  {'─'*6}  {'─'*7}  {'─'*7}")
    for year, sub in results.groupby('year'):
        ywr  = (sub['pnl_r'] > 0).sum() / len(sub) * 100
        yavg = sub['pnl_r'].mean()
        ycum = sub['pnl_r'].sum()
        print(f"  {year:<6} {len(sub):>4}  {ywr:>5.0f}%  {yavg:>+7.3f}  {ycum:>+7.2f}")


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(results: pd.DataFrame, target_r: float,
                 output_path: str = None):
    if results.empty:
        return

    BG, PANEL = '#0d1117', '#161b22'
    GREEN, RED, YELLOW, CYAN, GRAY, WHITE = (
        '#00ff1e', '#ff4444', '#ffcc00', '#00b9ff', '#8b949e', '#e6edf3'
    )

    results = results.copy()
    results['cumR'] = results.sort_values('timestamp')['pnl_r'].cumsum()
    results['year'] = pd.to_datetime(results['date']).dt.year

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        f'Extreme SPO Divergence Fade  ·  {TIMEFRAME_MIN}-min  ·  '
        f'Target {target_r}R  ·  {len(results)} trades',
        color=WHITE, fontsize=13, fontweight='bold',
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35,
                           top=0.92, bottom=0.06, left=0.08, right=0.97)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)

    # ── Cumulative R (full width) ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    bar_clr = [GREEN if r > 0 else RED for r in results['pnl_r']]
    ax1.bar(range(len(results)), results['pnl_r'],
            color=bar_clr, alpha=0.45, width=0.85)
    ax1.plot(results['cumR'].values, color=WHITE, linewidth=1.5, zorder=5)
    ax1.axhline(0, color=GRAY, linewidth=0.8)
    sax(ax1, 'Cumulative R  (bars = per-trade, line = equity curve)')
    ax1.set_ylabel('R', color=GRAY)
    ax1.set_xlabel('Trade #', color=GRAY)

    # ── Annual R ──────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ann = results.groupby('year')['pnl_r'].sum()
    ax2.bar(ann.index.astype(str), ann.values,
            color=[GREEN if v > 0 else RED for v in ann.values], alpha=0.85)
    ax2.axhline(0, color=GRAY, linewidth=0.8)
    sax(ax2, 'Annual R')
    ax2.set_ylabel('R', color=GRAY)

    # ── R-multiple distribution ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(results['pnl_r'], bins=30, color=YELLOW, alpha=0.75, edgecolor=BG)
    ax3.axvline(0, color=RED, linewidth=1.2)
    ax3.axvline(results['pnl_r'].mean(), color=GREEN,
                linewidth=1.2, linestyle='--', label=f"avg={results['pnl_r'].mean():+.2f}R")
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    sax(ax3, 'R-multiple distribution')
    ax3.set_xlabel('R', color=GRAY)

    # ── Win rate by div type ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    type_order = ['bull', 'hidden_bull', 'bear', 'hidden_bear']
    type_stats = (
        results.groupby('div_type')
        .agg(wr=('pnl_r', lambda x: (x > 0).mean() * 100),
             cnt=('pnl_r', 'count'))
        .reindex(type_order).dropna()
    )
    bar_clrs = [GREEN, CYAN, RED, '#ff8800'][:len(type_stats)]
    bars = ax4.bar(
        [DIV_LABEL.get(t, t) for t in type_stats.index],
        type_stats['wr'], color=bar_clrs, alpha=0.85,
    )
    for bar, (_, row) in zip(bars, type_stats.iterrows()):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"n={int(row['cnt'])}", ha='center', fontsize=8, color=GRAY)
    ax4.axhline(50, color=GRAY, linewidth=0.8, linestyle='--')
    ax4.set_ylim(0, 108)
    ax4.tick_params(axis='x', labelsize=7)
    sax(ax4, 'Win rate by divergence type (%)')

    # ── Exit breakdown ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    exits    = results['exit_reason'].value_counts()
    clr_map  = {'target': GREEN, 'stop': RED, 'eod': YELLOW}
    ax5.bar(exits.index, exits.values,
            color=[clr_map.get(e, GRAY) for e in exits.index], alpha=0.85)
    for i, (label, val) in enumerate(exits.items()):
        ax5.text(i, val + 0.5, str(val), ha='center', fontsize=9, color=WHITE)
    sax(ax5, 'Exit breakdown  (green=target, red=stop, yellow=EOD)')
    ax5.set_ylabel('Count', color=GRAY)

    out = (output_path.replace('.csv', '_fade_chart.png')
           if output_path else 'backtest_extreme_fade.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extreme SPO Divergence Fade — 10-min backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',   type=str,            help='Single ticker (default: all)')
    parser.add_argument('--start',    type=str,            help='Start date YYYY-MM-DD')
    parser.add_argument('--end',      type=str,            help='End date YYYY-MM-DD')
    parser.add_argument('--target-r', type=float, default=2.0,
                        help='R-multiple target (default: 2.0)')
    parser.add_argument('--output',   type=str,            help='CSV output path')
    parser.add_argument('--catalog',  action='store_true', help='Show DB contents and exit')
    parser.add_argument('--no-plot',  action='store_true', help='Skip chart')
    args = parser.parse_args()

    target_r = args.target_r
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
    print("  EXTREME SPO DIVERGENCE FADE")
    print(f"  Timeframe  : {TIMEFRAME_MIN}-min  (resampled from 1-min on the fly)")
    print(f"  Tickers    : {', '.join(tickers)}")
    print(f"  Period     : {args.start or 'all'} → {args.end or 'latest'}")
    print(f"  Filter     : |SPO at pivot| > {EXTREME_THRESHOLD}")
    print(f"  Stop       : price pivot high/low")
    print(f"  Target     : {target_r}R")
    print(f"  Pivots     : lbL={DIV_LB_LEFT}  lbR={DIV_LB_RIGHT}  "
          f"range=[{DIV_RANGE_MIN},{DIV_RANGE_MAX}]")

    all_results = []
    for ticker in tickers:
        print(f"\n  Loading {ticker}...")
        df = load_bars(ticker, '1m_full', args.start, args.end, con)
        if df.empty:
            print("    [!] No data.")
            continue
        print(f"    {len(df):,} 1-min bars  |  "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        results = run_ticker(ticker, df, args.start, args.end, target_r)
        print_stats(results, ticker, target_r)
        if not results.empty:
            all_results.append(results)

    con.close()

    if not all_results:
        return

    combined = (pd.concat(all_results, ignore_index=True)
                .sort_values('timestamp')
                .reset_index(drop=True))

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\n  Saved: {args.output}  ({len(combined)} rows)")

    if not args.no_plot:
        plot_results(combined, target_r, args.output)

    if len(tickers) > 1:
        print("\n")
        print_stats(combined, 'ALL TICKERS', target_r)


if __name__ == '__main__':
    main()
