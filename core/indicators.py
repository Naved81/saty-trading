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
from numpy.lib.stride_tricks import sliding_window_view


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


# ── TTM Squeeze helpers ───────────────────────────────────────────────────────

def _rolling_linreg_end(series: pd.Series, length: int) -> pd.Series:
    """
    Vectorised rolling linear regression — returns the fitted value at the
    LAST point of each window (equivalent to ta.linreg(series, length, 0)
    in PineScript).

    Uses numpy sliding_window_view for speed; handles NaN rows gracefully.
    """
    arr = series.values.astype(float)
    n   = len(arr)
    if n < length:
        return pd.Series(np.nan, index=series.index)

    x     = np.arange(length, dtype=float)
    x_bar = (length - 1) / 2.0
    ss_xx = ((x - x_bar) ** 2).sum()

    windows  = sliding_window_view(arr, length)          # (n-length+1, length)
    nan_mask = np.any(np.isnan(windows), axis=1)         # rows with any NaN

    y_bar = np.where(nan_mask, np.nan, windows.mean(axis=1))

    delta_y = np.where(nan_mask[:, None], np.nan, windows - y_bar[:, None])
    ss_xy   = np.where(nan_mask, np.nan, ((x - x_bar) * delta_y).sum(axis=1))

    b    = ss_xy / ss_xx
    a    = y_bar - b * x_bar
    vals = a + b * (length - 1)

    out             = np.full(n, np.nan)
    out[length - 1:] = vals
    return pd.Series(out, index=series.index)


# ── TTM Squeeze ───────────────────────────────────────────────────────────────

def calc_ttm_squeeze(df: pd.DataFrame,
                     length: int = 20,
                     bb_mult: float = 2.0,
                     kc_mult_high: float = 1.0,
                     kc_mult_mid: float = 1.5,
                     kc_mult_low: float = 2.0) -> pd.DataFrame:
    """
    TTM Squeeze — Python translation of the Beardy Squeeze Pro PineScript.

    Squeeze state: Bollinger Bands compressed inside Keltner Channels.
    Three KC levels reveal squeeze intensity (high/mid/low).
    Momentum oscillator: linreg of (close − midrange/SMA anchor).

    Keltner Channel uses SMA of True Range (not Wilder ATR), matching
    the PineScript  devKC = ta.sma(ta.tr, length)  exactly.

    Args:
        df           : DataFrame with columns high, low, close
        length       : lookback for all components (default 20)
        bb_mult      : BB standard-deviation multiplier (default 2.0)
        kc_mult_high : inner KC multiplier  — orange dot (default 1.0)
        kc_mult_mid  : middle KC multiplier — red dot   (default 1.5)
        kc_mult_low  : outer KC multiplier  — black dot (default 2.0)

    Returns DataFrame with columns:
        bb_basis, bb_upper, bb_lower
        kc_basis, kc_upper/lower_high/mid/low
        no_sqz          : bool — BB extends outside outer KC  (green dot)
        squeeze_state   : 'none' | 'low' | 'mid' | 'high'
        in_squeeze      : bool — any squeeze level active (not no_sqz)
        momentum        : float — linreg oscillator
        mom_up          : bool — momentum rising vs prior bar
        squeeze_fired   : bool — transition squeeze → no squeeze
        squeeze_started : bool — transition no squeeze → squeeze
    """
    out = pd.DataFrame(index=df.index)

    # ── Bollinger Bands (SMA + sample stdev) ──
    sma            = df['close'].rolling(length, min_periods=length).mean()
    stdev          = df['close'].rolling(length, min_periods=length).std()
    out['bb_basis'] = sma
    out['bb_upper'] = sma + bb_mult * stdev
    out['bb_lower'] = sma - bb_mult * stdev

    # ── Keltner Channels (SMA of True Range — matches PineScript) ──
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    kc_dev = tr.rolling(length, min_periods=length).mean()

    out['kc_basis']      = sma   # shared with BB
    out['kc_upper_high'] = sma + kc_mult_high * kc_dev
    out['kc_lower_high'] = sma - kc_mult_high * kc_dev
    out['kc_upper_mid']  = sma + kc_mult_mid  * kc_dev
    out['kc_lower_mid']  = sma - kc_mult_mid  * kc_dev
    out['kc_upper_low']  = sma + kc_mult_low  * kc_dev
    out['kc_lower_low']  = sma - kc_mult_low  * kc_dev

    # ── Squeeze conditions (OR logic — exact PineScript translation) ──
    # no_sqz  : at least one BB band outside the outermost KC → expanding
    # high_sqz: at least one BB band inside the innermost KC  → tightest coil
    no_sqz   = ((out['bb_lower'] <  out['kc_lower_low']) |
                (out['bb_upper'] >  out['kc_upper_low']))
    low_sqz  = ((out['bb_lower'] >= out['kc_lower_low']) |
                (out['bb_upper'] <= out['kc_upper_low']))
    mid_sqz  = ((out['bb_lower'] >= out['kc_lower_mid']) |
                (out['bb_upper'] <= out['kc_upper_mid']))
    high_sqz = ((out['bb_lower'] >= out['kc_lower_high']) |
                (out['bb_upper'] <= out['kc_upper_high']))

    out['no_sqz']    = no_sqz
    out['in_squeeze'] = ~no_sqz

    # Dot colour priority: high > mid > low > none  (mirrors PineScript ternary)
    state = pd.Series('none', index=df.index)
    state[low_sqz]  = 'low'
    state[mid_sqz]  = 'mid'
    state[high_sqz] = 'high'
    out['squeeze_state'] = state

    # ── Momentum oscillator ──
    # mom = linreg(close − avg(avg(highest, lowest), sma), length, 0)
    highest  = df['high'].rolling(length, min_periods=length).max()
    lowest   = df['low'].rolling(length, min_periods=length).min()
    midrange = (highest + lowest) / 2.0
    anchor   = (midrange + sma) / 2.0
    delta    = df['close'] - anchor

    out['momentum'] = _rolling_linreg_end(delta, length)
    out['mom_up']   = out['momentum'] > out['momentum'].shift(1)

    # ── State transitions (for alerts and strategy entry) ──
    # shift(1) produces float NaN on the first row; fillna(True) → "was no-squeeze"
    # so the first bar never falsely fires as a squeeze_started event.
    prev_no_sqz            = no_sqz.shift(1).fillna(True).astype(bool)
    out['squeeze_fired']   = no_sqz  & ~prev_no_sqz   # squeeze releases
    out['squeeze_started'] = ~no_sqz &  prev_no_sqz   # squeeze begins

    return out
