"""
backtest/run_schwab_trade_analysis.py
========================================
Compute Price MAE/MFE and Position MAE/MFE for a CSV of closed trades
(default: the output of data/pull_schwab_trades.py).

For each trade, fetches 1-min bars spanning entry->exit (DB -> Schwab ->
Polygon, see core/bar_source.py) and measures:
  price_mfe / price_mae       — best/worst price excursion from entry (pts)
  position_mfe / position_mae — same excursion scaled by trade quantity ($)
  mfe_r / mae_r               — same excursion in R-multiples, only if
                                 --risk-col points to a per-share risk column

Trades with no bar coverage (e.g. older than Schwab's ~48-day intraday
window, no DB history, and no Polygon key) get NaN MAE/MFE and are
reported separately rather than silently dropped.

USAGE
-----
    python backtest/run_schwab_trade_analysis.py
    python backtest/run_schwab_trade_analysis.py --trades my_trades.csv
    python backtest/run_schwab_trade_analysis.py --risk-col risk
    python backtest/run_schwab_trade_analysis.py --output results.csv --no-plot

Trades CSV must have columns: ticker, direction, quantity, entry_date,
entry_time, entry_price, exit_date, exit_time, exit_price
(this is exactly what data/pull_schwab_trades.py produces).
"""

import sys
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from config import DB_PATH, SCHWAB_CLIENT_ID, TOKEN_PATH
from core.bar_source import fetch_bars_for_window
from core.trade_metrics import compute_trade_metrics

REQUIRED_COLS = ['ticker', 'direction', 'quantity',
                 'entry_date', 'entry_time', 'entry_price',
                 'exit_date', 'exit_time', 'exit_price']


def _optional_db_connection():
    if not DB_PATH.exists():
        return None
    from core.db import get_connection
    return get_connection()


def _optional_schwab_client():
    if not (SCHWAB_CLIENT_ID and TOKEN_PATH.exists()):
        return None
    from data.schwab_client import get_client
    return get_client()


def analyze_trades(trades: pd.DataFrame, con=None, client=None, risk_col: str = None) -> pd.DataFrame:
    trades = trades.copy()
    trades['entry_ts'] = pd.to_datetime(trades['entry_date'] + ' ' + trades['entry_time'])
    trades['exit_ts']  = pd.to_datetime(trades['exit_date']  + ' ' + trades['exit_time'])

    metric_rows = []
    for _, t in trades.iterrows():
        bars = fetch_bars_for_window(t['ticker'], t['entry_ts'], t['exit_ts'], con=con, client=client)
        risk = float(t[risk_col]) if risk_col and pd.notna(t.get(risk_col)) else None
        metric_rows.append(compute_trade_metrics(
            bars, t['direction'], float(t['entry_price']), float(t['quantity']), risk))

    metrics = pd.DataFrame(metric_rows)
    return pd.concat([trades.drop(columns=['entry_ts', 'exit_ts']), metrics], axis=1)


def print_stats(df: pd.DataFrame):
    covered = df[df['n_bars'] > 0]
    missing = len(df) - len(covered)

    print(f"\n  {len(df)} trades  |  {len(covered)} with bar coverage"
          + (f"  |  {missing} skipped (no bar data)" if missing else ""))
    if covered.empty:
        return

    win_pct = (covered['pnl'] > 0).mean() * 100 if 'pnl' in covered else np.nan
    print(f"  Win rate            : {win_pct:.1f}%")
    print(f"  Avg Price MFE / MAE : {covered['price_mfe'].mean():+.4f} / "
          f"{covered['price_mae'].mean():+.4f}  "
          f"({covered['price_mfe'].mean() / covered['price_mae'].mean():.2f}x)"
          if covered['price_mae'].mean() else "")
    print(f"  Avg Position MFE/MAE: ${covered['position_mfe'].mean():+,.2f} / "
          f"${covered['position_mae'].mean():+,.2f}")
    if 'mfe_r' in covered.columns:
        print(f"  Avg MFE / MAE (R)   : {covered['mfe_r'].mean():+.2f}R / "
              f"{covered['mae_r'].mean():+.2f}R")


def plot_results(df: pd.DataFrame, output_path: str = None):
    covered = df[df['n_bars'] > 0]
    if covered.empty:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BG, PANEL, WHITE, GRAY = '#0d1117', '#161b22', '#e6edf3', '#8b949e'
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#30363d')

    wins  = covered[covered['pnl'] > 0] if 'pnl' in covered else covered
    losses = covered[covered['pnl'] <= 0] if 'pnl' in covered else covered.iloc[0:0]
    ax.scatter(wins['price_mae'], wins['price_mfe'], color='#00ff1e', alpha=0.7, label='Winners', s=40)
    ax.scatter(losses['price_mae'], losses['price_mfe'], color='#ff4444', alpha=0.7, label='Losers', s=40)

    lim = max(covered['price_mae'].max(), covered['price_mfe'].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls='--', alpha=0.5, label='1:1 MFE=MAE')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('Price MAE (pts)', color=GRAY, fontsize=9)
    ax.set_ylabel('Price MFE (pts)', color=GRAY, fontsize=9)
    ax.set_title('Per-trade MAE vs MFE', color=WHITE, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)

    out = (output_path.replace('.csv', '_mae_mfe.png') if output_path
           else 'backtest_schwab_mae_mfe.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Chart saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compute Price/Position MAE-MFE for a CSV of closed trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--trades', type=str, default='backtest/results/schwab_trades.csv',
                        help='Input trades CSV (default: data/pull_schwab_trades.py output)')
    parser.add_argument('--output', type=str, default='backtest/results/schwab_mae_mfe.csv')
    parser.add_argument('--risk-col', type=str, default=None,
                        help='Column in the trades CSV holding per-share risk, to add R-multiples')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"[!] {trades_path} not found. Run data/pull_schwab_trades.py first, "
              f"or point --trades at your own CSV.")
        sys.exit(1)

    trades = pd.read_csv(trades_path)
    missing_cols = [c for c in REQUIRED_COLS if c not in trades.columns]
    if missing_cols:
        print(f"[!] Trades CSV is missing required columns: {missing_cols}")
        sys.exit(1)

    con = _optional_db_connection()
    client = _optional_schwab_client()
    print(f"  Bar sources available: "
          f"{'DB ' if con else ''}{'Schwab ' if client else ''}"
          f"{'Polygon' if not con and not client else ''}".strip() or "none configured")

    result = analyze_trades(trades, con=con, client=client, risk_col=args.risk_col)

    if con:
        con.close()

    print_stats(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(result)} rows)")

    if not args.no_plot:
        plot_results(result, args.output)


if __name__ == '__main__':
    main()
