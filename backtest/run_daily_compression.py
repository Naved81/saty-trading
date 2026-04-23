"""
backtest/run_daily_compression.py
==================================
Saty's 0-5 Candle Daily Compression Break — full backtest.

Setup logic:
  1. Find 1–5 consecutive daily bars where the BB is inside the ATR channel
     (in_compression == True, as computed by calc_compression on daily bars).
  2. Lock the range: high = max(comp highs), low = min(comp lows).
  3. After compression ends, watch up to DAILY_COMP_MAX_WATCH bars for a
     breakout close outside the range in the ribbon's direction:
       • close > range_high AND bullish ribbon → LONG
       • close < range_low  AND bearish ribbon → SHORT
  4. Enter at the OPEN of the next daily bar.
  5. Exit when:
       • Target  : +DAILY_COMP_TARGET_R × risk   (default 2R)
       • Stop    : −1R (range high/low edge)
       • Max hold: DAILY_COMP_MAX_HOLD calendar days (default 30)

USAGE
-----
    python backtest/run_daily_compression.py                    # all tickers with 1d data
    python backtest/run_daily_compression.py --ticker AAPL
    python backtest/run_daily_compression.py --ticker AAPL MSFT NVDA
    python backtest/run_daily_compression.py --start 2020-01-01
    python backtest/run_daily_compression.py --output trades.csv
    python backtest/run_daily_compression.py --no-plot
    python backtest/run_daily_compression.py --catalog

OUTPUT
------
  Console : per-ticker stats + by-year breakdown + comp-candle breakdown
  Chart   : backtest_daily_compression.png  (or <output>_dc_chart.png)
  CSV     : optional, one row per trade
"""

import sys
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    WARMUP_DAYS,
    DAILY_COMP_MAX_CANDLES,
    DAILY_COMP_MIN_CANDLES,
    DAILY_COMP_MAX_WATCH,
    DAILY_COMP_MAX_HOLD,
    DAILY_COMP_TARGET_R,
)
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.indicators import calc_spo, calc_compression, calc_pivot_ribbon


# ── Indicator preparation ─────────────────────────────────────────────────────

def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SPO, compression, and ribbon columns to a daily bar DataFrame.
    Input df must have: timestamp, open, high, low, close, volume.
    """
    df = df.copy().reset_index(drop=True)
    spo_df = calc_spo(df)
    df['pivot']          = spo_df['pivot']
    df['iatr']           = spo_df['iatr']
    df['spo']            = spo_df['spo']
    df['in_compression'] = calc_compression(df, spo_df['pivot'], spo_df['iatr'])

    ribbon = calc_pivot_ribbon(df)
    df['bullish'] = ribbon['bullish']
    df['bearish'] = ribbon['bearish']
    return df


# ── Exit simulator ────────────────────────────────────────────────────────────

def simulate_exit(df: pd.DataFrame,
                  entry_idx: int,
                  direction: str,
                  entry_price: float,
                  entry_date: pd.Timestamp,
                  stop: float,
                  target: float) -> tuple:
    """
    Scan forward from entry_idx+1 for target / stop / max-hold exit.
    Returns (exit_price, exit_date, exit_reason, exit_idx).
    """
    n             = len(df)
    max_hold_end  = entry_date + pd.Timedelta(days=DAILY_COMP_MAX_HOLD)

    for k in range(entry_idx + 1, n):
        row = df.iloc[k]
        ts  = pd.Timestamp(row['timestamp'])

        if ts > max_hold_end:
            return row['open'], ts, 'maxhold', k

        if direction == 'long':
            if row['low'] <= stop:
                return stop, ts, 'stop', k
            if row['high'] >= target:
                return target, ts, 'target', k
        else:
            if row['high'] >= stop:
                return stop, ts, 'stop', k
            if row['low'] <= target:
                return target, ts, 'target', k

    # Ran out of data
    last = df.iloc[-1]
    return last['close'], pd.Timestamp(last['timestamp']), 'maxhold', n - 1


# ── Per-ticker backtest ───────────────────────────────────────────────────────

def run_ticker(ticker: str, df_raw: pd.DataFrame,
               start: str = None, end: str = None) -> pd.DataFrame:
    """
    Run the 0-5 candle daily compression break strategy on one ticker.
    Returns a DataFrame of trade results.
    """
    df = prepare_daily(df_raw)

    # Date range filter (applied after indicator warmup)
    if start:
        df = df[df['timestamp'] >= pd.Timestamp(start)].reset_index(drop=True)
    if end:
        df = df[df['timestamp'] <= pd.Timestamp(end)].reset_index(drop=True)

    # Skip warmup rows
    df = df.iloc[WARMUP_DAYS:].reset_index(drop=True)

    trades = []
    n      = len(df)
    i      = 0

    while i < n:
        row = df.iloc[i]

        # ── Wait for compression to start ──────────────────────────────────
        if not row['in_compression']:
            i += 1
            continue

        # ── Gather consecutive compression bars (max DAILY_COMP_MAX_CANDLES) ─
        comp_start = i
        while (i < n
               and df.iloc[i]['in_compression']
               and (i - comp_start) < DAILY_COMP_MAX_CANDLES):
            i += 1

        n_comp = i - comp_start
        if n_comp < DAILY_COMP_MIN_CANDLES:
            continue   # too few bars — skip (i already advanced)

        comp_slice = df.iloc[comp_start:i]
        range_high = float(comp_slice['high'].max())
        range_low  = float(comp_slice['low'].min())
        range_size = range_high - range_low

        if range_size <= 0:
            continue

        # ── Watch for breakout (up to DAILY_COMP_MAX_WATCH bars) ──────────
        watch_end      = min(i + DAILY_COMP_MAX_WATCH, n)
        signal_found   = False
        signal_bar_idx = -1

        for j in range(i, watch_end):
            wrow = df.iloc[j]

            long_break  = bool(wrow['bullish']) and wrow['close'] > range_high
            short_break = bool(wrow['bearish']) and wrow['close'] < range_low

            if not (long_break or short_break):
                continue

            # Entry is at the OPEN of the NEXT bar
            if j + 1 >= n:
                break

            entry_row   = df.iloc[j + 1]
            entry_price = float(entry_row['open'])
            entry_date  = pd.Timestamp(entry_row['timestamp'])
            direction   = 'long' if long_break else 'short'

            if direction == 'long':
                stop   = range_low
                risk   = entry_price - stop
            else:
                stop   = range_high
                risk   = stop - entry_price

            if risk <= 0:
                break

            target = (entry_price + DAILY_COMP_TARGET_R * risk
                      if direction == 'long'
                      else entry_price - DAILY_COMP_TARGET_R * risk)

            exit_price, exit_date, exit_reason, exit_idx = simulate_exit(
                df, j + 1, direction, entry_price, entry_date, stop, target
            )

            pnl   = (exit_price - entry_price if direction == 'long'
                     else entry_price - exit_price)
            r_mult = pnl / risk

            trades.append(dict(
                ticker       = ticker,
                date         = entry_date.strftime('%Y-%m-%d'),
                direction    = direction,
                n_comp       = n_comp,
                range_high   = round(range_high, 4),
                range_low    = round(range_low, 4),
                range_size   = round(range_size, 4),
                entry_price  = round(entry_price, 4),
                stop         = round(stop, 4),
                target       = round(target, 4),
                exit_price   = round(exit_price, 4),
                exit_date    = exit_date.strftime('%Y-%m-%d'),
                risk         = round(risk, 4),
                pnl_per_share = round(pnl, 4),
                r_multiple   = round(r_mult, 4),
                result       = 'WIN' if pnl > 0 else 'LOSS',
                exit_reason  = exit_reason,
            ))

            signal_found   = True
            signal_bar_idx = exit_idx
            break

        if signal_found:
            # Resume scanning after the trade exit bar
            i = signal_bar_idx + 1
        else:
            # No break found in watch window — advance past it
            i = watch_end

    return pd.DataFrame(trades)


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(tr: pd.DataFrame, ticker: str):
    if tr.empty:
        print(f"  [{ticker}] No trades.")
        return

    wins   = tr[tr['result'] == 'WIN']
    losses = tr[tr['result'] == 'LOSS']
    total  = len(tr)
    wr     = len(wins) / total * 100
    cum    = tr['pnl_per_share'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    pf_d   = abs(losses['pnl_per_share'].sum()) if len(losses) else 0
    pf     = wins['pnl_per_share'].sum() / pf_d if pf_d > 0 else float('inf')
    avg_r  = tr['r_multiple'].mean()

    by_exit = tr['exit_reason'].value_counts()
    by_dir  = tr['direction'].value_counts()

    print(f"\n  {'─' * 62}")
    print(f"  {ticker}  |  {tr['date'].min()} → {tr['date'].max()}")
    print(f"  {'─' * 62}")
    print(f"  Trades:        {total}  "
          f"(W:{len(wins)}  L:{len(losses)}  "
          f"long:{by_dir.get('long', 0)}  short:{by_dir.get('short', 0)})")
    print(f"  Win Rate:      {wr:.1f}%")
    print(f"  Profit Factor: {pf:.2f}x")
    print(f"  Avg R:         {avg_r:+.2f}R")
    print(f"  Total PnL:     ${tr['pnl_per_share'].sum():,.4f} per share")
    print(f"  Max Drawdown:  ${max_dd:,.4f} per share")
    if len(wins):
        print(f"  Avg Win:       ${wins['pnl_per_share'].mean():.4f} / "
              f"{wins['r_multiple'].mean():.2f}R")
    if len(losses):
        print(f"  Avg Loss:      ${losses['pnl_per_share'].mean():.4f} / "
              f"{losses['r_multiple'].mean():.2f}R")
    print(f"  Exits:         "
          f"target={by_exit.get('target', 0)}  "
          f"stop={by_exit.get('stop', 0)}  "
          f"maxhold={by_exit.get('maxhold', 0)}")

    # ── By year ────────────────────────────────────────────────────────────
    by_year = tr.groupby(tr['date'].str[:4]).agg(
        trades  = ('result', 'count'),
        wins    = ('result', lambda x: (x == 'WIN').sum()),
        losses  = ('result', lambda x: (x == 'LOSS').sum()),
        total_pnl = ('pnl_per_share', 'sum'),
        avg_r   = ('r_multiple', 'mean'),
    )
    print(f"\n  By year:")
    for yr, r in by_year.iterrows():
        wr_yr = int(r['wins']) / int(r['trades']) * 100
        print(f"    {yr}  trades:{int(r['trades']):3d}  "
              f"W:{int(r['wins'])}  L:{int(r['losses'])}  "
              f"WR:{wr_yr:.0f}%  "
              f"${r['total_pnl']:+.4f}/share  "
              f"avgR:{r['avg_r']:+.2f}")

    # ── By compression candle count ────────────────────────────────────────
    by_comp = tr.groupby('n_comp').agg(
        trades  = ('result', 'count'),
        wins    = ('result', lambda x: (x == 'WIN').sum()),
        total_pnl = ('pnl_per_share', 'sum'),
        avg_r   = ('r_multiple', 'mean'),
    )
    print(f"\n  By compression candles:")
    for nc, r in by_comp.iterrows():
        wr_nc = int(r['wins']) / int(r['trades']) * 100
        print(f"    {nc} candles  trades:{int(r['trades']):3d}  "
              f"WR:{wr_nc:.0f}%  "
              f"${r['total_pnl']:+.4f}/share  "
              f"avgR:{r['avg_r']:+.2f}")


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(all_trades: pd.DataFrame, output_path: str = None):
    if all_trades.empty:
        return

    BG, PANEL, GREEN, RED, BLUE, GOLD, GRAY, WHITE = (
        '#0d1117', '#161b22', '#00ff1e', '#ff4444',
        '#00b9ff', '#ffcc00', '#8b949e', '#e6edf3',
    )
    tickers = all_trades['ticker'].unique()
    clrs    = [GREEN, BLUE, GOLD, '#ff9600', '#cc66ff', RED]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle('Daily Compression Break (0–5 Candles) — Backtest Results',
                 color=WHITE, fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.07, right=0.97)

    def _style(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)

    # ── Panel 1: cumulative PnL (top, full width) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for i, t in enumerate(tickers):
        t_df = all_trades[all_trades['ticker'] == t].reset_index(drop=True)
        cum  = t_df['pnl_per_share'].cumsum()
        wr   = (t_df['result'] == 'WIN').mean() * 100
        ax1.plot(cum.values, color=clrs[i % len(clrs)], lw=1.8,
                 label=f"{t}  ({len(t_df)} trades  {wr:.0f}% WR  "
                       f"${cum.iloc[-1]:,.2f}/share)")
    ax1.axhline(0, color=GRAY, lw=0.5, ls='--')
    _style(ax1, 'Cumulative PnL per share ($)')
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE, loc='upper left')
    ax1.set_ylabel('$ per share', color=GRAY)

    # ── Panel 2: annual PnL by ticker ─────────────────────────────────────
    all_trades['year'] = pd.to_datetime(all_trades['date']).dt.year
    ax2 = fig.add_subplot(gs[1, 0])
    yr_pnl = (all_trades
              .groupby(['year', 'ticker'])['pnl_per_share']
              .sum()
              .unstack(fill_value=0))
    x = np.arange(len(yr_pnl))
    w = 0.8 / max(len(tickers), 1)
    for i, t in enumerate(tickers):
        if t in yr_pnl.columns:
            ax2.bar(x + i * w, yr_pnl[t], width=w,
                    color=clrs[i % len(clrs)], alpha=0.8, label=t)
    ax2.set_xticks(x + w * (len(tickers) - 1) / 2)
    ax2.set_xticklabels(yr_pnl.index, rotation=45, fontsize=8, color=GRAY)
    ax2.axhline(0, color=GRAY, lw=0.5)
    _style(ax2, 'Annual PnL by ticker ($/share)')
    ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE)
    ax2.set_ylabel('$ per share', color=GRAY)

    # ── Panel 3: R-multiple distribution ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(tickers):
        t_df = all_trades[all_trades['ticker'] == t]
        if not t_df.empty:
            ax3.hist(t_df['r_multiple'], bins=30,
                     color=clrs[i % len(clrs)], alpha=0.55, label=t, density=True)
    ax3.axvline(0,  color=RED,   lw=1.2, ls='--', alpha=0.8, label='breakeven')
    ax3.axvline(1,  color=GRAY,  lw=0.8, ls=':',  alpha=0.6, label='1R')
    ax3.axvline(2,  color=GREEN, lw=0.8, ls=':',  alpha=0.6, label=f'{DAILY_COMP_TARGET_R:.0f}R target')
    ax3.axvline(-1, color=RED,   lw=0.8, ls=':',  alpha=0.6, label='−1R stop')
    _style(ax3, 'R-multiple distribution')
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE)
    ax3.set_xlabel('R-multiples', color=GRAY)

    # ── Panel 4: win rate by compression candle count ─────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    combined_by_comp = (all_trades
                        .groupby('n_comp')
                        .agg(wr=('result', lambda x: (x == 'WIN').mean() * 100),
                             cnt=('result', 'count'))
                        .reset_index())
    bars = ax4.bar(combined_by_comp['n_comp'].astype(str),
                   combined_by_comp['wr'],
                   color=BLUE, alpha=0.8)
    ax4.axhline(50, color=GRAY, lw=0.8, ls='--', alpha=0.6, label='50%')
    for bar, cnt in zip(bars, combined_by_comp['cnt']):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f'n={cnt}', ha='center', va='bottom',
                 color=WHITE, fontsize=8)
    _style(ax4, 'Win Rate by compression candle count (%)')
    ax4.set_xlabel('Compression candles', color=GRAY)
    ax4.set_ylabel('Win Rate %', color=GRAY)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE)

    out = (output_path.replace('.csv', '_dc_chart.png')
           if output_path else 'backtest_daily_compression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='0-5 Candle Daily Compression Break — backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',  type=str, nargs='+',
                        help='Tickers to run (default: all with 1d data in DB)')
    parser.add_argument('--start',   type=str, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',     type=str, help='End date YYYY-MM-DD')
    parser.add_argument('--output',  type=str, help='Save trades to CSV path')
    parser.add_argument('--no-plot', action='store_true', help='Skip chart')
    parser.add_argument('--catalog', action='store_true',
                        help='Show DB data coverage and exit')
    args = parser.parse_args()

    con = get_connection()

    if args.catalog:
        print(get_catalog(con).to_string(index=False))
        con.close()
        return

    tickers = ([t.upper() for t in args.ticker] if args.ticker
               else get_available_tickers('1d', con))
    if not tickers:
        print("[!] No 1d data found. Run:  python data/pull_daily.py")
        con.close()
        sys.exit(1)

    print("=" * 64)
    print("0-5 Candle Daily Compression Break — Backtest")
    print(f"Tickers : {', '.join(tickers)}")
    print(f"Period  : {args.start or 'all'} → {args.end or 'latest'}")
    print(f"Setup   : {DAILY_COMP_MIN_CANDLES}–{DAILY_COMP_MAX_CANDLES} comp candles  |  "
          f"watch {DAILY_COMP_MAX_WATCH} bars  |  "
          f"target {DAILY_COMP_TARGET_R:.0f}R  |  "
          f"max hold {DAILY_COMP_MAX_HOLD}d")
    print("=" * 64)

    all_trades = []

    for ticker in tickers:
        print(f"\nLoading {ticker}...")
        df = load_bars(ticker, '1d', args.start, args.end, con)
        if df.empty:
            print("  [!] No data — run python data/pull_daily.py first.")
            continue
        print(f"  {len(df):,} bars  |  "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        trades = run_ticker(ticker, df, args.start, args.end)
        print_stats(trades, ticker)
        all_trades.append(trades)

    con.close()

    if not all_trades:
        return

    combined = pd.concat(all_trades, ignore_index=True)

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\nTrades saved: {args.output}")

    if not args.no_plot:
        plot_results(combined, args.output)

    if len(tickers) > 1:
        wins  = (combined['result'] == 'WIN').sum()
        total = len(combined)
        print(f"\n{'=' * 64}")
        print(f"COMBINED — {len(tickers)} tickers")
        print(f"  Trades : {total}  |  WR: {wins / total * 100:.1f}%")
        print(f"  Total  : ${combined['pnl_per_share'].sum():,.4f}/share")
        print(f"  Avg R  : {combined['r_multiple'].mean():+.2f}R")
        print(f"  Pf     : "
              f"{combined[combined['result']=='WIN']['pnl_per_share'].sum() / abs(combined[combined['result']=='LOSS']['pnl_per_share'].sum()):.2f}x"
              if (combined['result'] == 'LOSS').any() else "  Pf: ∞")


if __name__ == '__main__':
    main()
