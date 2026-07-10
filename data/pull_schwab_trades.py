"""
data/pull_schwab_trades.py
============================
Pull filled TRADE transactions from your Schwab account(s) and reconstruct
them into round-trip trades (one row per matched open→close lot), ready
for MAE/MFE analysis in backtest/run_schwab_trade_analysis.py.

Schwab's transaction endpoint reports individual fills, not "trades" — a
position opened in two fills and closed in one fill is three transactions.
This script FIFO-matches opening fills to closing fills per symbol so each
output row is one complete open→close trade.

Equities/ETFs only — options legs are not parsed here (see the project's
Schwab-API-limits note for why: no historical intraday option pricing).

USAGE
-----
    python data/pull_schwab_trades.py                                # last 90 days, all accounts
    python data/pull_schwab_trades.py --start 2025-01-01 --end 2025-12-31
    python data/pull_schwab_trades.py --account 0                    # first linked account only
    python data/pull_schwab_trades.py --output backtest/results/schwab_trades.csv
    python data/pull_schwab_trades.py --dump-raw                     # print raw transaction JSON, no parsing

REQUIREMENTS
------------
    SCHWAB_CLIENT_ID / SCHWAB_SECRET / SCHWAB_REDIRECT_URI / SCHWAB_TOKEN_PATH
    set (see .env.example) and a Schwab API token already created — see
    data/schwab_client.py for the one-time login step.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import deque, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from data.schwab_client import get_client, get_account_hashes

DEFAULT_LOOKBACK_DAYS = 90
SKIP_ASSET_TYPES = {'CURRENCY'}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_raw_transactions(client, account_hash: str, start: str, end: str) -> list:
    """
    Pull TRADE transactions in <=90 day chunks (Schwab's practical query-range
    limit) and concatenate. start/end are 'YYYY-MM-DD'.
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    all_txns = []

    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(days=89), end_ts)
        resp = client.get_transactions(
            account_hash,
            start_date=chunk_start.to_pydatetime(),
            end_date=chunk_end.to_pydatetime(),
            transaction_types=['TRADE'],
        )
        if resp.status_code != 200:
            print(f"    [!] {chunk_start.date()} -> {chunk_end.date()}: "
                  f"HTTP {resp.status_code} — {resp.text[:200]}")
        else:
            txns = resp.json()
            print(f"    {chunk_start.date()} -> {chunk_end.date()}: {len(txns)} transactions")
            all_txns.extend(txns)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    return all_txns


# ── Parse raw transactions into individual fills ──────────────────────────────

def extract_fills(raw_txns: list) -> pd.DataFrame:
    """
    Flatten raw Schwab transaction JSON into one row per equity/ETF fill:
    ticker, timestamp, signed_qty (+buy / -sell), price, order_id.

    Schwab represents each fill's legs under 'transferItems' (falls back to
    the older 'transactionItem' singular key some account types still return).
    Non-equity legs (cash, currency) are skipped.
    """
    rows = []
    for txn in raw_txns:
        ts = pd.Timestamp(txn.get('time') or txn.get('tradeDate'))
        order_id = txn.get('orderId') or txn.get('activityId')

        items = txn.get('transferItems')
        if items is None and 'transactionItem' in txn:
            items = [txn['transactionItem']]
        if not items:
            continue

        for item in items:
            instrument = item.get('instrument') or {}
            asset_type = instrument.get('assetType')
            symbol = instrument.get('symbol')
            if not symbol or asset_type in SKIP_ASSET_TYPES:
                continue
            if asset_type not in (None, 'EQUITY', 'ETF', 'COLLECTIVE_INVESTMENT'):
                continue  # skip options/other instruments — see module docstring

            qty = item.get('amount')
            price = item.get('price')
            if qty is None or price is None:
                continue

            rows.append(dict(
                ticker=symbol,
                timestamp=ts,
                signed_qty=float(qty),
                price=float(price),
                order_id=order_id,
            ))

    if not rows:
        return pd.DataFrame(columns=['ticker', 'timestamp', 'signed_qty', 'price', 'order_id'])

    return (pd.DataFrame(rows)
            .sort_values('timestamp')
            .reset_index(drop=True))


# ── FIFO reconstruction ───────────────────────────────────────────────────────

def reconstruct_trades(fills: pd.DataFrame) -> pd.DataFrame:
    """
    FIFO-match opening fills to closing fills per ticker.

    A "lot" is an unmatched slice of a fill: (qty, price, timestamp, order_id),
    qty sign encodes long(+)/short(-). A fill on the same side as the open
    lots extends the position (pushes a new lot); a fill on the opposite side
    closes existing lots FIFO, emitting one trade row per matched slice. If a
    closing fill overshoots the open quantity, the remainder flips into a new
    lot on the other side.
    """
    open_lots = defaultdict(deque)   # ticker -> deque of dict(qty, price, ts, order_id)
    trades = []

    for _, fill in fills.iterrows():
        ticker = fill['ticker']
        remaining = fill['signed_qty']
        lots = open_lots[ticker]

        while remaining != 0 and lots and (lots[0]['qty'] > 0) != (remaining > 0):
            lot = lots[0]
            match_qty = min(abs(lot['qty']), abs(remaining))

            direction = 'long' if lot['qty'] > 0 else 'short'
            entry_price, exit_price = (lot['price'], fill['price'])
            pnl_per_share = ((exit_price - entry_price) if direction == 'long'
                              else (entry_price - exit_price))

            trades.append(dict(
                ticker=ticker,
                direction=direction,
                quantity=match_qty,
                entry_time=lot['ts'],
                entry_price=entry_price,
                exit_time=fill['timestamp'],
                exit_price=exit_price,
                pnl_per_share=round(pnl_per_share, 4),
                pnl=round(pnl_per_share * match_qty, 2),
                entry_order_id=lot['order_id'],
                exit_order_id=fill['order_id'],
            ))

            sign = 1 if lot['qty'] > 0 else -1
            lot['qty'] -= sign * match_qty
            remaining += sign * match_qty
            if lot['qty'] == 0:
                lots.popleft()

        if remaining != 0:
            lots.append(dict(qty=remaining, price=fill['price'],
                              ts=fill['timestamp'], order_id=fill['order_id']))

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades).sort_values('exit_time').reset_index(drop=True)
    df['entry_date'] = df['entry_time'].dt.strftime('%Y-%m-%d')
    df['entry_time'] = df['entry_time'].dt.strftime('%H:%M:%S')
    df['exit_date']  = df['exit_time'].dt.strftime('%Y-%m-%d')
    df['exit_time']  = df['exit_time'].dt.strftime('%H:%M:%S')

    cols = ['ticker', 'direction', 'quantity',
            'entry_date', 'entry_time', 'entry_price',
            'exit_date', 'exit_time', 'exit_price',
            'pnl_per_share', 'pnl', 'entry_order_id', 'exit_order_id']
    return df[cols]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pull Schwab trade history and reconstruct round-trip trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--start', type=str, help='Start date YYYY-MM-DD (default: 90 days ago)')
    parser.add_argument('--end',   type=str, help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--account', type=int, default=None,
                        help='Index of linked account to pull (default: all linked accounts)')
    parser.add_argument('--output', type=str, default='backtest/results/schwab_trades.csv')
    parser.add_argument('--dump-raw', action='store_true',
                        help='Print raw transaction JSON (first 5) and exit — use this to verify '
                             'the parsing in extract_fills() matches your account\'s schema')
    args = parser.parse_args()

    end   = args.end or pd.Timestamp.now().strftime('%Y-%m-%d')
    start = args.start or (pd.Timestamp.now() - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime('%Y-%m-%d')

    client = get_client()
    hashes = get_account_hashes(client)
    accounts = list(hashes.items())
    if args.account is not None:
        accounts = [accounts[args.account]]

    print(f"  Accounts   : {len(accounts)}")
    print(f"  Period     : {start} -> {end}")

    all_raw = []
    for acct_num, acct_hash in accounts:
        print(f"\n  Account ...{acct_num[-4:]}")
        all_raw.extend(fetch_raw_transactions(client, acct_hash, start, end))

    if args.dump_raw:
        print(f"\n  {len(all_raw)} raw transactions. First 5:\n")
        print(json.dumps(all_raw[:5], indent=2, default=str))
        return

    fills = extract_fills(all_raw)
    print(f"\n  Parsed {len(fills)} equity/ETF fills from {len(all_raw)} transactions.")
    if fills.empty:
        print("  No fills found — try --dump-raw to inspect the raw JSON schema.")
        return

    trades = reconstruct_trades(fills)
    print(f"  Reconstructed {len(trades)} round-trip trades.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(trades)} rows)")


if __name__ == '__main__':
    main()
