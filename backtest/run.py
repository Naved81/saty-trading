"""
backtest/run.py
===============
Backtest runner — reads from SQLite DB, runs strategy, reports results.

USAGE
-----
    python backtest/run.py                        # all tickers in DB
    python backtest/run.py --ticker QQQ           # single ticker
    python backtest/run.py --start 2022-01-01     # date range
    python backtest/run.py --output trades.csv    # save trades
    python backtest/run.py --catalog              # show DB contents
    python backtest/run.py --no-plot              # skip chart
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

from config import ENTRY_TIME, RTH_START, SPREAD_WIDTH, PREMIUM, SPX_RATIO
from core.db import get_connection, load_bars, get_available_tickers, get_catalog
from core.sessions import filter_eth, filter_rth
from core.indicators import calc_spo, calc_compression, build_daily_reference
from strategies.saty_credit_spread import check_signal, calc_pnl


# ── Build intraday indicators ─────────────────────────────────────────────────

def build_intraday_indicators(df_eth: pd.DataFrame) -> pd.DataFrame:
    """Add SPO and compression columns to ETH bars. Continuous — no resets."""
    df = df_eth.copy()
    spo_df = calc_spo(df)
    df['pivot']          = spo_df['pivot']
    df['iatr']           = spo_df['iatr']
    df['spo']            = spo_df['spo']
    df['in_compression'] = calc_compression(df, spo_df['pivot'], spo_df['iatr'])
    return df


# ── Run strategy on one ticker ────────────────────────────────────────────────

def run_ticker(ticker: str, df_full: pd.DataFrame, start: str = None,
               end: str = None) -> pd.DataFrame:
    df_full = df_full.copy()
    df_full['date_ts'] = df_full['timestamp'].dt.normalize()

    df_eth = filter_eth(df_full)
    df_rth = filter_rth(df_full)
    df_eth['date_ts'] = df_eth['timestamp'].dt.normalize()
    df_rth['date_ts'] = df_rth['timestamp'].dt.normalize()

    daily  = build_daily_reference(df_full, df_rth)
    df_eth = build_intraday_indicators(df_eth)

    entry_t = pd.Timestamp(ENTRY_TIME).time()
    open_t  = RTH_START

    snap_entry = df_eth[df_eth['timestamp'].dt.time == entry_t][
        ['date_ts','close','spo','in_compression']
    ].rename(columns={'close':'price_entry','spo':'spo_entry',
                      'in_compression':'comp_entry'})
    snap_open = df_eth[df_eth['timestamp'].dt.time == open_t][
        ['date_ts','close']
    ].rename(columns={'close':'price_open'})

    daily = daily.merge(snap_entry, on='date_ts', how='left')
    daily = daily.merge(snap_open,  on='date_ts', how='left')
    daily = daily.dropna(subset=['prev_atr_rth','prev_atr_full','price_entry','prev_close'])

    trades = []
    for _, row in daily.iterrows():
        signal = check_signal(
            ticker=ticker,
            row=row,
            spo_at_entry=row['spo_entry'],
            in_compression=bool(row['comp_entry']),
            price_at_entry=row['price_entry'],
            price_at_open=row.get('price_open', np.nan),
        )
        if signal is None:
            continue

        eod = row['rth_close']
        if pd.isna(eod):
            continue

        trades.append(calc_pnl(signal, eod))

    return pd.DataFrame(trades)


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(tr: pd.DataFrame, ticker: str):
    if tr.empty:
        print(f"  [{ticker}] No trades."); return

    wins    = tr[tr['result']=='WIN']
    losses  = tr[tr['result']=='LOSS']
    partial = tr[tr['result']=='PARTIAL']
    wr      = len(wins)/len(tr)*100
    cum     = tr['pnl_per_contract'].cumsum()
    max_dd  = (cum - cum.cummax()).min()
    pf_d    = abs(losses['pnl_per_contract'].sum())
    pf      = wins['pnl_per_contract'].sum()/pf_d if pf_d>0 else float('inf')

    print(f"\n  {'─'*55}")
    print(f"  {ticker}  |  {tr['date'].min()} → {tr['date'].max()}")
    print(f"  {'─'*55}")
    print(f"  Trades:        {len(tr)}  (W:{len(wins)}  L:{len(losses)}  P:{len(partial)})")
    print(f"  Win Rate:      {wr:.1f}%")
    print(f"  Profit Factor: {pf:.2f}x")
    print(f"  Total PnL:     ${tr['pnl_per_contract'].sum():,.2f} per contract")
    print(f"  Max Drawdown:  ${max_dd:,.2f} per contract")
    print(f"  Avg Win:       ${wins['pnl_per_contract'].mean():.2f}" if len(wins) else "  Avg Win:       —")
    print(f"  Avg Loss:      ${losses['pnl_per_contract'].mean():.2f}" if len(losses) else "  Avg Loss:      —")
    print(f"  SPX Est Total: ${tr['spx_est_pnl'].sum():,.2f} per contract")

    by_year = tr.groupby(tr['date'].str[:4]).agg(
        trades=('result','count'),
        wins=('result', lambda x: (x=='WIN').sum()),
        losses=('result', lambda x: (x=='LOSS').sum()),
        total_pnl=('pnl_per_contract','sum')
    )
    print(f"\n  By year:")
    for yr, r in by_year.iterrows():
        wr_yr = int(r['wins'])/int(r['trades'])*100
        print(f"    {yr}  trades:{int(r['trades']):3d}  "
              f"W:{int(r['wins'])}  L:{int(r['losses'])}  "
              f"WR:{wr_yr:.0f}%  ${r['total_pnl']:,.0f}")


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(all_trades: pd.DataFrame, output_path: str = None):
    if all_trades.empty:
        return

    BG,PANEL,GREEN,RED,BLUE,GOLD,GRAY,WHITE = \
        '#0d1117','#161b22','#00ff1e','#ff4444','#00b9ff','#ffcc00','#8b949e','#e6edf3'
    tickers = all_trades['ticker'].unique()
    clrs    = [GREEN, BLUE, GOLD, '#ff9600', '#cc66ff', RED]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle('Saty ATR Credit Spread — Backtest Results',
                 color=WHITE, fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                           top=0.92, bottom=0.07, left=0.07, right=0.97)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)

    ax1 = fig.add_subplot(gs[0,:])
    for i, t in enumerate(tickers):
        t_df = all_trades[all_trades['ticker']==t]
        cum  = t_df['pnl_per_contract'].cumsum().reset_index(drop=True)
        wr   = (t_df['result']=='WIN').mean()*100
        ax1.plot(cum, color=clrs[i%len(clrs)], lw=1.8,
                 label=f"{t}  ({len(t_df)} trades, {wr:.0f}% WR, ${cum.iloc[-1]:,.0f})")
    ax1.axhline(0, color=GRAY, lw=0.5, ls='--')
    sax(ax1, 'Cumulative PnL per contract ($)')
    ax1.legend(fontsize=9, facecolor=PANEL, labelcolor=WHITE, loc='upper left')
    ax1.set_ylabel('$ per contract', color=GRAY)

    all_trades['year'] = pd.to_datetime(all_trades['date']).dt.year
    ax2 = fig.add_subplot(gs[1,0])
    yr_pnl = all_trades.groupby(['year','ticker'])['pnl_per_contract'].sum().unstack(fill_value=0)
    x = np.arange(len(yr_pnl))
    w = 0.8/len(tickers)
    for i, t in enumerate(tickers):
        if t in yr_pnl.columns:
            ax2.bar(x+i*w, yr_pnl[t], width=w, color=clrs[i%len(clrs)], alpha=0.8, label=t)
    ax2.set_xticks(x+w*(len(tickers)-1)/2)
    ax2.set_xticklabels(yr_pnl.index, rotation=45, fontsize=8, color=GRAY)
    ax2.axhline(0, color=GRAY, lw=0.5)
    sax(ax2, 'Annual PnL by ticker ($)')
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)

    ax3 = fig.add_subplot(gs[1,1])
    stats = all_trades.groupby('ticker').agg(
        wr=('result', lambda x: (x=='WIN').mean()*100),
        n=('result','count')
    )
    bars = ax3.bar(stats.index, stats['wr'],
                   color=[clrs[i] for i in range(len(stats))], alpha=0.85, width=0.5)
    ax3.axhline(50, color=RED, lw=1, ls='--', alpha=0.5)
    for b, (_, r) in zip(bars, stats.iterrows()):
        ax3.text(b.get_x()+b.get_width()/2, r['wr']+0.3,
                 f"{r['wr']:.1f}%\nn={int(r['n'])}", ha='center', fontsize=8.5, color=WHITE)
    sax(ax3, 'Win rate by ticker')
    ax3.set_ylim(0, 108)

    out = (output_path.replace('.csv','_chart.png')
           if output_path else 'backtest_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Saty ATR Credit Spread backtest')
    parser.add_argument('--ticker',  type=str)
    parser.add_argument('--start',   type=str)
    parser.add_argument('--end',     type=str)
    parser.add_argument('--output',  type=str)
    parser.add_argument('--catalog', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    con = get_connection()

    if args.catalog:
        print(get_catalog(con).to_string(index=False))
        con.close(); return

    tickers = ([args.ticker.upper()] if args.ticker
               else get_available_tickers('1m_full', con))
    if not tickers:
        print("[!] No tickers found. Run data/massive_pull.py first.")
        con.close(); sys.exit(1)

    print("=" * 60)
    print("Saty ATR Credit Spread — Backtest")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period:  {args.start or 'all'} → {args.end or 'latest'}")
    print("=" * 60)

    all_trades = []
    for ticker in tickers:
        print(f"\nLoading {ticker}...")
        df = load_bars(ticker, '1m_full', args.start, args.end, con)
        if df.empty:
            print(f"  [!] No data."); continue
        df['date_ts'] = df['timestamp'].dt.normalize()
        print(f"  {len(df):,} bars  |  "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        trades = run_ticker(ticker, df, args.start, args.end)
        print_stats(trades, ticker)
        all_trades.append(trades)

    con.close()
    if not all_trades: return

    combined = pd.concat(all_trades, ignore_index=True)
    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\nTrades saved: {args.output}")
    if not args.no_plot:
        plot_results(combined, args.output)

    if len(tickers) > 1:
        wins  = (combined['result']=='WIN').sum()
        total = len(combined)
        print(f"\n{'='*60}")
        print(f"COMBINED — {len(tickers)} tickers")
        print(f"  Trades:   {total}  |  WR: {wins/total*100:.1f}%")
        print(f"  Total:    ${combined['pnl_per_contract'].sum():,.2f}/contract")
        print(f"  SPX est:  ${combined['spx_est_pnl'].sum():,.2f}/contract")


if __name__ == '__main__':
    main()
