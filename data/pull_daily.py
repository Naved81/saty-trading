"""
data/pull_daily.py
==================
Download daily (1d) OHLCV bars from Polygon.io and store them in the
local SQLite database.  Supports incremental pulls — only fetches bars
newer than the last bar already in the database.

USAGE
-----
    python data/pull_daily.py                              # all TOP20 tickers
    python data/pull_daily.py --ticker AAPL                # single ticker
    python data/pull_daily.py --ticker AAPL MSFT NVDA      # explicit list
    python data/pull_daily.py --start 2020-01-01           # force re-pull from date
    python data/pull_daily.py --catalog                    # show DB contents and exit

REQUIREMENTS
------------
    Set the MASSIVE_API_KEY environment variable to your Polygon.io API key:
        export MASSIVE_API_KEY=your_polygon_api_key     (Linux/Mac)
        set    MASSIVE_API_KEY=your_polygon_api_key     (Windows CMD)
"""

import sys
import time
import argparse
import warnings
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

import pandas as pd

from config import (
    MASSIVE_API_KEY,
    TOP20_TICKERS,
    DEFAULT_START,
    MAX_RETRIES,
    RETRY_BASE_SEC,
    REQUEST_DELAY,
)
from core.db import get_connection, insert_bars, get_catalog

TIMEFRAME = '1d'


# ── Polygon fetch ─────────────────────────────────────────────────────────────

def fetch_daily_bars(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily adjusted OHLCV from Polygon.io for one ticker.
    Returns DataFrame: timestamp, open, high, low, close, volume.
    """
    from polygon import RESTClient

    client = RESTClient(api_key=MASSIVE_API_KEY)

    for attempt in range(MAX_RETRIES + 1):
        try:
            aggs = list(client.get_aggs(
                ticker,
                1,
                "day",
                from_=start_date,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=50000,
            ))
            break
        except Exception as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SEC * (2 ** attempt)
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} in {wait}s  ({exc})")
                time.sleep(wait)
            else:
                print(f"    [!] Failed after {MAX_RETRIES} retries: {exc}")
                return pd.DataFrame()

    if not aggs:
        return pd.DataFrame()

    rows = [
        {
            'timestamp': pd.Timestamp(a.timestamp, unit='ms').normalize(),
            'open':   float(a.open),
            'high':   float(a.high),
            'low':    float(a.low),
            'close':  float(a.close),
            'volume': float(a.volume),
        }
        for a in aggs
    ]
    return pd.DataFrame(rows)


# ── Per-ticker pull ───────────────────────────────────────────────────────────

def pull_ticker(ticker: str, start_override: str, con) -> int:
    """
    Pull daily bars for one ticker and upsert into the DB.
    Returns number of rows inserted.
    """
    # Determine pull-from date
    if start_override:
        pull_from = start_override
    else:
        row = con.execute(
            "SELECT last_bar FROM data_catalog WHERE ticker=? AND timeframe=?",
            (ticker, TIMEFRAME),
        ).fetchone()
        if row:
            last_bar  = pd.Timestamp(row[0])
            pull_from = (last_bar + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            pull_from = DEFAULT_START

    today = date.today().strftime('%Y-%m-%d')

    if pull_from > today:
        print(f"    Already up-to-date (last bar: {pull_from})")
        return 0

    print(f"    Fetching {ticker}:  {pull_from}  →  {today}")
    df = fetch_daily_bars(ticker, pull_from, today)

    if df.empty:
        print(f"    No data returned.")
        return 0

    n = insert_bars(ticker, TIMEFRAME, df, 'polygon', con)
    first = df['timestamp'].min().date()
    last  = df['timestamp'].max().date()
    print(f"    Inserted {n:,} bars  ({first} → {last})")

    time.sleep(REQUEST_DELAY)
    return n


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download daily OHLCV from Polygon.io',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ticker',  type=str, nargs='+',
                        help='Tickers to pull (default: TOP20 from config)')
    parser.add_argument('--start',   type=str,
                        help='Force re-pull from this date YYYY-MM-DD '
                             '(default: incremental from last DB bar)')
    parser.add_argument('--catalog', action='store_true',
                        help='Show DB data coverage and exit')
    args = parser.parse_args()

    if not MASSIVE_API_KEY:
        print("[!] MASSIVE_API_KEY environment variable is not set.")
        print("    export MASSIVE_API_KEY=your_polygon_api_key")
        sys.exit(1)

    con = get_connection()

    if args.catalog:
        df = get_catalog(con)
        if df.empty:
            print("  (no data in catalog yet)")
        else:
            print(df.to_string(index=False))
        con.close()
        return

    tickers = [t.upper() for t in args.ticker] if args.ticker else TOP20_TICKERS

    print("=" * 62)
    print("Polygon Daily Bar Download")
    print(f"Tickers  : {len(tickers)}")
    print(f"           {', '.join(tickers)}")
    print(f"Mode     : {'forced from ' + args.start if args.start else 'incremental'}")
    print("=" * 62)

    total = 0
    for ticker in tickers:
        print(f"\n{ticker}")
        total += pull_ticker(ticker, args.start, con)

    con.close()

    print(f"\n{'=' * 62}")
    print(f"Done — {total:,} bars inserted across {len(tickers)} tickers")
    print()
    print("Next step — run the daily compression backtest:")
    print("    python backtest/run_daily_compression.py")


if __name__ == '__main__':
    main()
