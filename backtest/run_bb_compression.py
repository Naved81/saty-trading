"""
backtest/run_bb_compression.py
================================
BB Compression Breakout backtest runner.

Reads 1-min full-session bars from SQLite, computes continuous indicators
on the extended-hours stream, then scans each RTH day for compression
breakout signals. Simulates stop/target/EOD exits using 1-min data.

USAGE
-----
    python backtest/run_bb_compression.py                      # all tickers
    python backtest/run_bb_compression.py --ticker SPY         # single ticker
    python backtest/run_bb_compression.py --ticker QQQ --start 2022-01-01
    python backtest/run_bb_compression.py --output trades.csv  # save CSV
    python backtest/run_bb_compression.py --catalog            # show DB contents
    python backtest/run_bb_compression.py --no-plot            # skip chart

OUTPUT
------
  Console : per-ticker stats + by-year breakdown
  Chart   : backtest_bb_compression.png (or <output>_bb_chart.png)
  CSV     : optional, one row per trade
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

from config import WARMUP_DAYS, BB_COMPRESSION_LOOKBACK, BB_STOP_ATR_MULT
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.sessions import filter_eth, filter_rth
from core.indicators import calc_spo, calc_compression, build_daily_reference
from strategies.bb_compression_breakout import scan_day, calc_pnl


# ── Indicator builder (identical pattern to run.py) ──────────────────────────

def build_intraday_indicators(df_eth: pd.DataFrame) -> pd.DataFrame:
    """Add continuous SPO, pivot, iatr, and compression columns to ETH bars."""
    df     = df_eth.copy()
    spo_df = calc_spo(df)
    df['pivot']          = spo_df['pivot']
    df['iatr']           = spo_df['iatr']
    df['spo']            = spo_df['spo']
    df['in_compression'] = calc_compression(df, spo_df['pivot'], spo_df['iatr'])
    return df


# ── Per-ticker backtest ───────────────────────────────────────────────────────

def run_ticker(ticker: str, df_full: pd.DataFrame,
               start: str = None, end: str = None) -> pd.DataFrame:
    """
    Run the BB Compression Breakout strategy on one ticker's 1-min data.

    Returns a DataFrame of trade results (one row per trade).
    """
    df_full = df_full.copy()
    df_full['date_ts'] = df_full['timestamp'].dt.normalize()

    df_eth = filter_eth(df_full)
    df_rth = filter_rth(df_full)
    df_eth['date_ts'] = df_eth['timestamp'].dt.normalize()
    df_rth['date_ts'] = df_rth['timestamp'].dt.normalize()

    daily  = build_daily_reference(df_full, df_rth)
    df_eth = build_intraday_indicators(df_eth)

    # Date range filter on the daily reference
    if start:
        daily = daily[daily['date_ts'] >= pd.Timestamp(start)]
    if end:
        daily = daily[daily['date_ts'] <= pd.Timestamp(end)]

    # Skip warmup rows (indicators need history to stabilise)
    # Use the larger of WARMUP_DAYS or BB_COMPRESSION_LOOKBACK
    skip = max(WARMUP_DAYS, BB_COMPRESSION_LOOKBACK)
    daily = daily.iloc[skip:].reset_index(drop=True)

    trades = []

    for _, drow in daily.iterrows():
        date_ts  = drow['date_ts']
        prev_atr = drow.get('prev_atr_rth')

        if pd.isna(prev_atr) or prev_atr == 0:
            continue

        # ETH bars for this calendar day (includes premarket for lookback)
        df_eth_day = df_eth[df_eth['date_ts'] == date_ts]
        if df_eth_day.empty:
            continue

        signal = scan_day(
            ticker=ticker,
            trade_date=date_ts.date(),
            df_eth_day=df_eth_day,
            prev_atr_rth=prev_atr,
            daily_bullish=bool(drow.get('bullish', False)),
            daily_bearish=bool(drow.get('bearish', False)),
        )
        if signal is None:
            continue

        # RTH bars strictly after the entry bar — used for stop/target simulation
        entry_time = pd.Timestamp(signal.entry_time).time()
        df_rth_after = df_rth[
            (df_rth['date_ts'] == date_ts) &
            (df_rth['timestamp'].dt.time > entry_time)
        ]
        if df_rth_after.empty:
            continue

        trades.append(calc_pnl(signal, df_rth_after))

    return pd.DataFrame(trades)


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(tr: pd.DataFrame, ticker: str):
    if tr.empty:
        print(f"  [{ticker}] No trades.")
        return

    wins   = tr[tr['result'] == 'WIN']
    losses = tr[tr['result'] == 'LOSS']
    wr     = len(wins) / len(tr) * 100
    cum    = tr['pnl_per_share'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    pf_d   = abs(losses['pnl_per_share'].sum())
    pf     = wins['pnl_per_share'].sum() / pf_d if pf_d > 0 else float('inf')
    avg_r  = tr['r_multiple'].mean()

    by_exit    = tr['exit_reason'].value_counts()
    by_dir     = tr['direction'].value_counts()

    print(f"\n  {'─'*60}")
    print(f"  {ticker}  |  {tr['date'].min()} → {tr['date'].max()}")
    print(f"  {'─'*60}")
    print(f"  Trades:        {len(tr)}  "
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
          f"eod={by_exit.get('eod', 0)}")

    by_year = tr.groupby(tr['date'].str[:4]).agg(
        trades=('result', 'count'),
        wins=('result', lambda x: (x == 'WIN').sum()),
        losses=('result', lambda x: (x == 'LOSS').sum()),
        total_pnl=('pnl_per_share', 'sum'),
        avg_r=('r_multiple', 'mean'),
    )
    print(f"\n  By year:")
    for yr, r in by_year.iterrows():
        wr_yr = int(r['wins']) / int(r['trades']) * 100
        print(f"    {yr}  trades:{int(r['trades']):3d}  "
              f"W:{int(r['wins'])}  L:{int(r['losses'])}  "
              f"WR:{wr_yr:.0f}%  "
              f"${r['total_pnl']:+.4f}/share  "
              f"avgR:{r['avg_r']:+.2f}")


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(all_trades: pd.DataFrame, output_path: str = None):
    if all_trades.empty:
        return

    BG, PANEL, GREEN, RED, BLUE, GOLD, GRAY, WHITE = (
        '#0d1117', '#161b22', '#00ff1e', '#ff4444',
        '#00b9ff', '#ffcc00', '#8b949e', '#e6edf3'
    )
    tickers = all_trades['ticker'].unique()
    clrs    = [GREEN, BLUE, GOLD, '#ff9600', '#cc66ff', RED]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle('BB Compression Breakout — Backtest Results',
                 color=WHITE, fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                           top=0.92, bottom=0.07, left=0.07, right=0.97)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)

    # ── Panel 1: cumulative PnL per share (top, full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    for i, t in enumerate(tickers):
        t_df = all_trades[all_trades['ticker'] == t]
        cum  = t_df['pnl_per_share'].cumsum().reset_index(drop=True)
        wr   = (t_df['result'] == 'WIN').mean() * 100
        ax1.plot(cum, color=clrs[i % len(clrs)], lw=1.8,
                 label=f"{t}  ({len(t_df)} trades  {wr:.0f}% WR  "
                       f"${cum.iloc[-1]:,.2f}/share)")
    ax1.axhline(0, color=GRAY, lw=0.5, ls='--')
    sax(ax1, 'Cumulative PnL per share ($)')
    ax1.legend(fontsize=9, facecolor=PANEL, labelcolor=WHITE, loc='upper left')
    ax1.set_ylabel('$ per share', color=GRAY)

    # ── Panel 2: annual PnL by ticker ──
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
    sax(ax2, 'Annual PnL by ticker ($/share)')
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    ax2.set_ylabel('$ per share', color=GRAY)

    # ── Panel 3: R-multiple distribution ──
    ax3 = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(tickers):
        t_df = all_trades[all_trades['ticker'] == t]
        if not t_df.empty:
            ax3.hist(t_df['r_multiple'], bins=30,
                     color=clrs[i % len(clrs)], alpha=0.55, label=t, density=True)
    ax3.axvline(0,  color=RED,   lw=1.2, ls='--', alpha=0.8, label='breakeven')
    ax3.axvline(1,  color=GRAY,  lw=0.8, ls=':',  alpha=0.6, label='1R')
    ax3.axvline(2,  color=GREEN, lw=0.8, ls=':',  alpha=0.6, label='2R (target)')
    ax3.axvline(-1, color=RED,   lw=0.8, ls=':',  alpha=0.6, label='-1R (stop)')
    sax(ax3, 'R-multiple distribution')
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE)
    ax3.set_xlabel('R-multiples', color=GRAY)

    out = (output_path.replace('.csv', '_bb_chart.png')
           if output_path else 'backtest_bb_compression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BB Compression Breakout backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',  type=str,           help='Single ticker (default: all in DB)')
    parser.add_argument('--start',   type=str,           help='Start date YYYY-MM-DD')
    parser.add_argument('--end',     type=str,           help='End date YYYY-MM-DD')
    parser.add_argument('--output',  type=str,           help='Save trades to CSV path')
    parser.add_argument('--catalog', action='store_true', help='Show DB data coverage and exit')
    parser.add_argument('--no-plot', action='store_true', help='Skip chart generation')
    args = parser.parse_args()

    con = get_connection()

    if args.catalog:
        print(get_catalog(con).to_string(index=False))
        con.close()
        return

    tickers = ([args.ticker.upper()] if args.ticker
               else get_available_tickers('1m_full', con))
    if not tickers:
        print("[!] No tickers found in DB. Run data/massive_pull.py first.")
        con.close()
        sys.exit(1)

    print("=" * 62)
    print("BB Compression Breakout — Backtest")
    print(f"Tickers : {', '.join(tickers)}")
    print(f"Period  : {args.start or 'all'} → {args.end or 'latest'}")
    print(f"Stop    : {int(BB_STOP_ATR_MULT * 100)}% of prev daily RTH ATR  |  "
          f"Target: 2× stop  (2:1 R:R)")
    print("=" * 62)

    all_trades = []
    for ticker in tickers:
        print(f"\nLoading {ticker}...")
        df = load_bars(ticker, '1m_full', args.start, args.end, con)
        if df.empty:
            print("  [!] No data.")
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
        print(f"\n{'='*62}")
        print(f"COMBINED — {len(tickers)} tickers")
        print(f"  Trades : {total}  |  WR: {wins/total*100:.1f}%")
        print(f"  Total  : ${combined['pnl_per_share'].sum():,.4f}/share")
        print(f"  Avg R  : {combined['r_multiple'].mean():+.2f}R")


if __name__ == '__main__':
    main()
