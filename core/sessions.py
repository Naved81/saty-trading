"""
core/sessions.py
================
Session filtering and resampling utilities.
Pure functions — no I/O, no side effects.
"""

import pandas as pd
from config import RTH_START, RTH_END, ETH_START


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Regular trading hours only: 09:30–16:00 ET."""
    return df[
        (df['timestamp'].dt.time >= RTH_START) &
        (df['timestamp'].dt.time <= RTH_END)
    ].copy()


def filter_eth(df: pd.DataFrame) -> pd.DataFrame:
    """Extended hours: 04:00 ET onwards (pre + RTH + post)."""
    return df[df['timestamp'].dt.time >= ETH_START].copy()


def filter_premarket(df: pd.DataFrame) -> pd.DataFrame:
    """Premarket only: 04:00–09:30 ET."""
    return df[
        (df['timestamp'].dt.time >= ETH_START) &
        (df['timestamp'].dt.time < RTH_START)
    ].copy()


def resample_to(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1-minute bars to any higher timeframe.
    e.g. rule='5min', '15min', '1h'

    Resamples across the full dataset — use filter_rth() or filter_eth()
    first if you want session-bounded aggregation.
    """
    out = (
        df.set_index('timestamp')
        .resample(rule)
        .agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum')
        )
        .dropna(subset=['open'])
        .reset_index()
    )
    return out
