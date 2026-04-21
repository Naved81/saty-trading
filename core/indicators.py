"""
core/indicators.py
==================
Pure indicator functions — no I/O, no side effects.
Used by both backtest and live signal checker.

All EMAs/ATRs are CONTINUOUS — no per-day resets.
This matches how indicators appear on a live TradingView chart.

Key design:
  - Functions accept a DataFrame, return a Series or DataFrame
  - Caller decides which session to pass (ETH or RTH)
  - No global state
"""

import pandas as pd
import numpy as np


# ── ATR ───────────────────────────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Continuous ATR-n using Wilder's EMA smoothing.
    Pass full-session bars for true daily ATR,
    or RTH bars for intraday-only ATR.
    """
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()


# ── EMA ───────────────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """Continuous EMA. Never resets."""
    return series.ewm(span=span, adjust=False).mean()


# ── Pivot Ribbon ──────────────────────────────────────────────────────────────

def calc_pivot_ribbon(df: pd.DataFrame,
                      fast: int = 8,
                      pivot: int = 21,
                      slow: int = 34) -> pd.DataFrame:
    """
    Saty Pivot Ribbon — 8/21/34 EMA system on daily closes.
    Returns DataFrame with ema_fast, ema_pivot, ema_slow, bullish, bearish.

    Pass a daily DataFrame with a 'close' (or 'rth_close') column.
    """
    close = df.get('rth_close', df.get('close'))
    out = pd.DataFrame(index=df.index)
    out['ema_fast']  = calc_ema(close, fast)
    out['ema_pivot'] = calc_ema(close, pivot)
    out['ema_slow']  = calc_ema(close, slow)

    # Shift by 1 — trend based on prior day's ribbon
    pf = out['ema_fast'].shift(1)
    pp = out['ema_pivot'].shift(1)
    ps = out['ema_slow'].shift(1)
    pc = close.shift(1)

    out['bullish'] = (pc >= pf) & (pf >= pp) & (pp >= ps)
    out['bearish'] = (pc <= pf) & (pf <= pp) & (pp >= ps)
    return out


# ── Saty Phase Oscillator ─────────────────────────────────────────────────────

def calc_spo(df: pd.DataFrame,
             pivot_span: int = 21,
             atr_n: int = 14,
             smooth_span: int = 3) -> pd.DataFrame:
    """
    Saty Phase Oscillator — continuous, no session resets.

    Formula:
        pivot   = 21-EMA of close (continuous)
        iatr    = 14-ATR (continuous)
        raw_spo = ((close - pivot) / (3 * iatr)) * 100
        spo     = 3-EMA of raw_spo (continuous)

    Key zones: ±23.6 (neutral), ±61.8 (accumulation/distribution), ±100 (extreme)

    Returns DataFrame with: pivot, iatr, raw_spo, spo
    """
    out = pd.DataFrame(index=df.index)
    out['pivot'] = calc_ema(df['close'], pivot_span)
    out['iatr']  = calc_atr(df, atr_n)
    out['raw_spo'] = (
        (df['close'] - out['pivot']) /
        (3.0 * out['iatr'].replace(0, np.nan))
    ) * 100
    out['spo'] = calc_ema(out['raw_spo'], smooth_span)
    return out


# ── Bollinger Band Compression ────────────────────────────────────────────────

def calc_compression(df: pd.DataFrame,
                     pivot: pd.Series,
                     iatr: pd.Series,
                     stdev_n: int = 21,
                     bb_mult: float = 2.0,
                     atr_mult: float = 2.0) -> pd.Series:
    """
    Saty BB Compression signal.

    Compression = Bollinger Band width < ATR channel width
    When BBands are narrower than the ATR envelope, price is coiling.

    Returns boolean Series: True = in compression, False = expanding.
    """
    stdev    = df['close'].rolling(stdev_n, min_periods=5).std()
    above    = df['close'] >= pivot

    bband_up   = pivot + bb_mult  * stdev
    bband_down = pivot - bb_mult  * stdev
    atr_up     = pivot + atr_mult * iatr
    atr_down   = pivot - atr_mult * iatr

    comp_raw = np.where(
        above,
        bband_up   - atr_up,
        atr_down   - bband_down
    )
    return pd.Series(comp_raw <= 0, index=df.index, name='in_compression')


# ── Daily reference builder ───────────────────────────────────────────────────

def build_daily_reference(df_full: pd.DataFrame,
                           df_rth: pd.DataFrame) -> pd.DataFrame:
    """
    Build all per-day reference values needed by the strategy.

    Two ATR values — each serves a distinct role:

      atr_rth  (RTH ATR)  → morning move THRESHOLD
                             Stable across regimes — intraday volatility
                             doesn't expand as dramatically as overnight gaps

      atr_full (Full ATR) → STRIKE PLACEMENT
                             Captures true daily range including overnight gaps

    Args:
        df_full : full-session bars (all timestamps)
        df_rth  : RTH-only bars (09:30–16:00 ET)

    Both DataFrames must have a 'date_ts' column (timestamp.normalize()).
    Returns a daily DataFrame indexed by date_ts.
    """
    # Full-session daily OHLC
    daily = df_full.groupby('date_ts').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'),     close=('close','last')
    ).reset_index()

    # RTH daily OHLC
    daily_rth = df_rth.groupby('date_ts').agg(
        rth_open=('open','first'), rth_high=('high','max'),
        rth_low=('low','min'),     rth_close=('close','last')
    ).reset_index()

    # Full-session ATR — for strike placement
    daily['atr_full'] = calc_atr(daily)

    # RTH ATR — for morning move threshold
    daily_rth['atr_rth'] = calc_atr(
        daily_rth.rename(columns={
            'rth_high': 'high', 'rth_low': 'low', 'rth_close': 'close'
        })
    )

    daily = daily.merge(
        daily_rth[['date_ts', 'rth_close', 'atr_rth']],
        on='date_ts', how='left'
    )

    # Prior-day anchors
    daily['prev_close']    = daily['rth_close'].shift(1)   # prior 4 PM close
    daily['prev_atr_rth']  = daily['atr_rth'].shift(1)
    daily['prev_atr_full'] = daily['atr_full'].shift(1)

    # Strike levels — use full ATR
    from config import SHORT_PCT
    daily['short_put']  = daily['prev_close'] - SHORT_PCT * daily['prev_atr_full']
    daily['short_call'] = daily['prev_close'] + SHORT_PCT * daily['prev_atr_full']

    # Overnight gap
    daily['gap_pts']     = daily['open'] - daily['prev_close']
    daily['gap_pct_atr'] = daily['gap_pts'] / daily['prev_atr_full']

    # Daily ribbon
    ribbon = calc_pivot_ribbon(daily)
    daily  = pd.concat([daily, ribbon[['bullish','bearish']]], axis=1)

    return daily
