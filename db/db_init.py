"""
db/db_init.py
=============
Creates the SQLite database and all required tables.
Run this once before anything else.

USAGE
-----
    python db/db_init.py

The database path is read from config.py (DB_PATH).
Override with the DB_PATH environment variable if needed.
"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # ── Main OHLCV table ──────────────────────────────────────────────────────
    # Primary key on (ticker, timeframe, timestamp) guarantees:
    #   - No duplicate bars
    #   - A 1m_full bar can NEVER be returned when querying for 1d
    #   - Safe to re-run (INSERT OR REPLACE)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker      TEXT    NOT NULL,
            timeframe   TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL,
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      REAL    NOT NULL,
            PRIMARY KEY (ticker, timeframe, timestamp)
        )
    """)

    # Fast range queries — most common access pattern is (ticker, timeframe, timestamp range)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
        ON ohlcv (ticker, timeframe, timestamp)
    """)

    # ── Data catalog ──────────────────────────────────────────────────────────
    # Tracks what's in the DB per ticker/timeframe.
    # Updated automatically after every pull — tells the pull script
    # exactly where to resume from without re-fetching existing data.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS data_catalog (
            ticker      TEXT    NOT NULL,
            timeframe   TEXT    NOT NULL,
            first_bar   TEXT,
            last_bar    TEXT,
            bar_count   INTEGER DEFAULT 0,
            source      TEXT,
            updated_at  TEXT,
            PRIMARY KEY (ticker, timeframe)
        )
    """)

    # ── Pull log ──────────────────────────────────────────────────────────────
    # Audit trail of every API call. Tracks successes, empties, and failures.
    # Failed chunks are visible here so re-runs know what to retry.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pull_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            timeframe       TEXT    NOT NULL,
            source          TEXT    NOT NULL,
            from_date       TEXT,
            to_date         TEXT,
            bars_fetched    INTEGER DEFAULT 0,
            bars_inserted   INTEGER DEFAULT 0,
            status          TEXT,
            error           TEXT,
            pulled_at       TEXT    NOT NULL
        )
    """)

    # ── Signal log ────────────────────────────────────────────────────────────
    # Records every signal the alert app fires.
    # Useful for tracking live performance vs backtest expectations.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            signal_date     TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            trend           TEXT,
            prev_close      REAL,
            price_at_entry  REAL,
            short_strike    REAL,
            long_strike     REAL,
            credit          REAL,
            spo_at_entry    REAL,
            morning_move    REAL,
            gap_pts         REAL,
            alerted_at      TEXT    NOT NULL
        )
    """)

    con.commit()
    con.close()

    print(f"Database initialized:")
    print(f"  {DB_PATH}")
    print()
    print("Tables created:")
    print("  ohlcv         — OHLCV bars, keyed on (ticker, timeframe, timestamp)")
    print("  data_catalog  — tracks coverage and last bar per ticker/timeframe")
    print("  pull_log      — audit trail of all API pulls")
    print("  signal_log    — live signal history for performance tracking")
    print()
    print("Next steps:")
    print("  python data/massive_pull.py          # pull historical data")
    print("  python backtest/run.py --catalog     # verify data loaded")
    print("  python backtest/run.py               # run backtest")


if __name__ == "__main__":
    init_db()
