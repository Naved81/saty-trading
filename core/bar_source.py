"""
core/bar_source.py
===================
Fetch 1-minute OHLC bars covering a specific trade's entry→exit window,
trying each source in order until one has coverage:

  1. Local SQLite DB (1m_full timeframe) — fastest, no API calls.
  2. Schwab price history — only ~48 calendar days of intraday history.
  3. Polygon.io minute aggregates (if MASSIVE_API_KEY is set) — for trades
     older than Schwab's intraday window.

Returns an empty DataFrame (with a printed warning) if no source covers
the requested window — callers should treat that trade's MAE/MFE as
unavailable rather than guessing.
"""

import time
import warnings

import pandas as pd

from config import MASSIVE_API_KEY, MAX_RETRIES, RETRY_BASE_SEC

SCHWAB_INTRADAY_LOOKBACK_DAYS = 45  # conservative; Schwab docs say ~48


def _from_db(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, con) -> pd.DataFrame:
    if con is None:
        return pd.DataFrame()
    from core.db import load_bars

    df = load_bars(ticker, '1m_full',
                    start=start_ts.strftime('%Y-%m-%d'),
                    end=end_ts.strftime('%Y-%m-%d'),
                    con=con)
    if df.empty:
        return df
    return df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)


def _from_schwab(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, client) -> pd.DataFrame:
    if client is None:
        return pd.DataFrame()

    days_ago = (pd.Timestamp.now(tz=start_ts.tz) - start_ts).days if start_ts.tzinfo else \
               (pd.Timestamp.now() - start_ts).days
    if days_ago > SCHWAB_INTRADAY_LOOKBACK_DAYS:
        return pd.DataFrame()

    resp = client.get_price_history_every_minute(
        ticker, start_datetime=start_ts.to_pydatetime(), end_datetime=end_ts.to_pydatetime())
    if resp.status_code != 200:
        return pd.DataFrame()

    candles = resp.json().get('candles', [])
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)


def _from_polygon(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if not MASSIVE_API_KEY:
        return pd.DataFrame()

    from polygon import RESTClient
    client = RESTClient(api_key=MASSIVE_API_KEY)

    for attempt in range(MAX_RETRIES + 1):
        try:
            aggs = list(client.get_aggs(
                ticker, 1, "minute",
                from_=start_ts.strftime('%Y-%m-%d'),
                to=end_ts.strftime('%Y-%m-%d'),
                adjusted=True, sort="asc", limit=50000,
            ))
            break
        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_SEC * (2 ** attempt))
            else:
                warnings.warn(f"Polygon fetch failed for {ticker}: {exc}")
                return pd.DataFrame()

    if not aggs:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp(a.timestamp, unit='ms'),
            'open': float(a.open), 'high': float(a.high),
            'low': float(a.low), 'close': float(a.close),
            'volume': float(a.volume),
        }
        for a in aggs
    ])
    return df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)


def fetch_bars_for_window(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                           con=None, client=None) -> pd.DataFrame:
    """
    Return 1-min bars spanning [start_ts, end_ts] for ticker, trying
    DB -> Schwab -> Polygon in order. Empty DataFrame if none have coverage.
    """
    df = _from_db(ticker, start_ts, end_ts, con)
    if not df.empty:
        return df

    df = _from_schwab(ticker, start_ts, end_ts, client)
    if not df.empty:
        return df

    df = _from_polygon(ticker, start_ts, end_ts)
    if not df.empty:
        return df

    warnings.warn(
        f"No intraday bar coverage for {ticker} {start_ts} -> {end_ts} "
        f"(checked DB, Schwab, Polygon)."
    )
    return pd.DataFrame()
