"""
core/db.py
==========
SQLite database helpers — connection, read, write, catalog.
All other modules import from here rather than touching sqlite3 directly.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection. Exits cleanly if DB missing."""
    if not DB_PATH.exists():
        print(f"[!] Database not found:\n    {DB_PATH}")
        print("    Run:  python db/db_init.py")
        sys.exit(1)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def load_bars(ticker: str,
              timeframe: str = '1m_full',
              start: str = None,
              end: str = None,
              con: sqlite3.Connection = None) -> pd.DataFrame:
    """
    Load OHLCV bars for a ticker/timeframe from the DB.

    Args:
        ticker    : e.g. 'SPY', 'QQQ'
        timeframe : '1m_full' | '1d' (matches data_catalog labels)
        start     : 'YYYY-MM-DD' or None
        end       : 'YYYY-MM-DD' or None
        con       : existing connection (created if None)
    """
    close_con = con is None
    if con is None:
        con = get_connection()

    query  = "SELECT timestamp,open,high,low,close,volume FROM ohlcv WHERE ticker=? AND timeframe=?"
    params = [ticker, timeframe]
    if start:
        query  += " AND timestamp >= ?"; params.append(start)
    if end:
        query  += " AND timestamp <= ?"; params.append(end + " 23:59:59")
    query += " ORDER BY timestamp"

    df = pd.read_sql_query(query, con, params=params, parse_dates=['timestamp'])

    if close_con:
        con.close()
    return df


def get_catalog(con: sqlite3.Connection = None) -> pd.DataFrame:
    """Return the data_catalog table as a DataFrame."""
    close_con = con is None
    if con is None:
        con = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM data_catalog ORDER BY ticker, timeframe", con)
    if close_con:
        con.close()
    return df


def get_available_tickers(timeframe: str = '1m_full',
                           con: sqlite3.Connection = None) -> list:
    """Return list of tickers that have data for a given timeframe."""
    close_con = con is None
    if con is None:
        con = get_connection()
    rows = con.execute(
        "SELECT ticker FROM data_catalog WHERE timeframe=? ORDER BY ticker",
        (timeframe,)
    ).fetchall()
    if close_con:
        con.close()
    return [r[0] for r in rows]


def insert_bars(ticker: str, timeframe: str,
                df: pd.DataFrame, source: str,
                con: sqlite3.Connection) -> int:
    """
    Upsert bars into ohlcv. Safe to re-run — INSERT OR REPLACE.
    Updates data_catalog automatically.
    Returns number of rows written.
    """
    if df.empty:
        return 0

    rows = [
        (ticker, timeframe, str(r['timestamp']),
         r['open'], r['high'], r['low'], r['close'], r['volume'])
        for _, r in df.iterrows()
    ]
    con.executemany("""
        INSERT OR REPLACE INTO ohlcv
            (ticker, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    con.execute("""
        INSERT INTO data_catalog (ticker,timeframe,first_bar,last_bar,bar_count,source,updated_at)
        VALUES (?,?,?,?,?,?,datetime('now'))
        ON CONFLICT(ticker,timeframe) DO UPDATE SET
            first_bar  = MIN(first_bar, excluded.first_bar),
            last_bar   = MAX(last_bar,  excluded.last_bar),
            bar_count  = (SELECT COUNT(*) FROM ohlcv WHERE ticker=? AND timeframe=?),
            source     = excluded.source,
            updated_at = datetime('now')
    """, (ticker, timeframe,
          str(df['timestamp'].min()), str(df['timestamp'].max()),
          len(df), source, ticker, timeframe))

    con.commit()
    return len(rows)
