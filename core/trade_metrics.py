"""
core/trade_metrics.py
======================
Price MAE/MFE and position MAE/MFE for a closed trade, given its
entry/exit and the bar-by-bar price path in between.

Definitions (direction-aware, all expressed as positive-is-better-or-worse
price distances from entry):
  price_mfe — best price reached in your favor during the trade
  price_mae — worst price reached against you during the trade
  position_mfe / position_mae — the same excursions scaled by quantity ($),
      and by risk-per-share into R-multiples if risk_per_share is supplied.

Pure functions only — no I/O, no side effects. Bar fetching lives in
core/bar_source.py.
"""

import numpy as np
import pandas as pd


def compute_excursions(bars: pd.DataFrame, direction: str, entry_price: float) -> dict:
    """
    From bars covering (entry, exit], compute price MAE/MFE for one trade.

    bars       — DataFrame with at least 'high','low' columns, one row per
                 bar between entry and exit (exclusive of the entry bar's
                 pre-entry price, inclusive of the exit bar).
    direction  — 'long' or 'short'
    entry_price — trade entry fill price

    Returns dict: price_mfe, price_mae, price_mfe_pct, price_mae_pct, n_bars.
    NaN values (with n_bars=0) if bars is empty.
    """
    if bars.empty:
        return dict(price_mfe=np.nan, price_mae=np.nan,
                     price_mfe_pct=np.nan, price_mae_pct=np.nan, n_bars=0)

    highs = bars['high'].values.astype(float)
    lows  = bars['low'].values.astype(float)

    if direction == 'long':
        mfe = float(np.max(highs)) - entry_price
        mae = entry_price - float(np.min(lows))
    elif direction == 'short':
        mfe = entry_price - float(np.min(lows))
        mae = float(np.max(highs)) - entry_price
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    return dict(
        price_mfe=round(mfe, 4),
        price_mae=round(mae, 4),
        price_mfe_pct=round(mfe / entry_price * 100, 4),
        price_mae_pct=round(mae / entry_price * 100, 4),
        n_bars=len(bars),
    )


def compute_position_metrics(price_mfe: float, price_mae: float,
                              quantity: float, risk_per_share: float = None) -> dict:
    """
    Scale price MAE/MFE into position terms.

    quantity        — shares/contracts held (unsigned)
    risk_per_share  — optional; if given, also returns R-multiples
                       (e.g. entry-to-stop distance for the trade).
    """
    out = dict(
        position_mfe=round(price_mfe * quantity, 2) if not np.isnan(price_mfe) else np.nan,
        position_mae=round(price_mae * quantity, 2) if not np.isnan(price_mae) else np.nan,
    )
    if risk_per_share:
        out['mfe_r'] = round(price_mfe / risk_per_share, 4) if not np.isnan(price_mfe) else np.nan
        out['mae_r'] = round(price_mae / risk_per_share, 4) if not np.isnan(price_mae) else np.nan
    return out


def compute_trade_metrics(bars: pd.DataFrame, direction: str, entry_price: float,
                           quantity: float, risk_per_share: float = None) -> dict:
    """Convenience wrapper: excursions + position scaling in one call."""
    result = compute_excursions(bars, direction, entry_price)
    result.update(compute_position_metrics(
        result['price_mfe'], result['price_mae'], quantity, risk_per_share))
    return result
