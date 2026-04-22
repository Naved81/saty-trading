"""
strategies/ttm_squeeze_breakout.py
====================================
TTM Squeeze Breakout — signal generation and PnL simulation.

STRATEGY SUMMARY
----------------
The TTM Squeeze identifies periods of low volatility (price coiling inside
Keltner Channels) followed by an expansion burst — the "squeeze fire".
The momentum oscillator reveals the direction the energy releases.

ENTRY CONDITIONS
----------------
On the first qualifying 1-min bar during RTH (9:31–13:59 ET):
  1. Squeeze fires: no_sqz transitions False → True (expanding after coiling)
  2. At least TTM_SQUEEZE_MIN_BARS bars were in_squeeze before the fire
  3. Momentum confirms direction:
       Long  — momentum > 0 AND rising (aqua bars in PineScript)
       Short — momentum < 0 AND falling (red bars in PineScript)
  4. Daily 8/21/34 ribbon not fighting direction

TRADE STRUCTURE
---------------
  Entry  : close of the signal bar
  Stop   : entry ∓ TTM_STOP_ATR_MULT × prev_atr_rth
  Target : entry ± 2 × stop_distance  (2:1 reward/risk)
  Exit   : first of — stop hit, target hit, or EOD RTH close

MOMENTUM COLOUR KEY (for reference)
------------------------------------
  Aqua   : mom > 0 and rising   → best long entries
  Blue   : mom > 0 and falling  → long momentum waning
  Red    : mom < 0 and falling  → best short entries
  Yellow : mom < 0 and rising   → short momentum waning
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np

from config import (
    TTM_SQUEEZE_MIN_BARS, TTM_STOP_ATR_MULT, TTM_ENTRY_CUTOFF, RTH_START,
)


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class TTMSqueezeSignal:
    """
    One TTM Squeeze Breakout trade signal.
    Returned by scan_day() — consumed by calc_pnl() and future alert system.
    """
    ticker:           str
    trade_date:       date
    direction:        str    # 'long' | 'short'
    trend:            str    # 'bullish' | 'bearish' | 'neutral'
    entry_time:       str    # 'HH:MM' ET
    entry_price:      float
    stop_price:       float
    target_price:     float
    momentum:         float  # oscillator value at entry
    squeeze_bars:     int    # consecutive in_squeeze bars before fire
    squeeze_state:    str    # 'low' | 'mid' | 'high' — intensity before fire
    prev_atr_rth:     float

    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    def reward_per_share(self) -> float:
        return abs(self.target_price - self.entry_price)

    def to_dict(self) -> dict:
        return {
            'ticker':           self.ticker,
            'date':             str(self.trade_date),
            'direction':        self.direction,
            'trend':            self.trend,
            'entry_time':       self.entry_time,
            'entry_price':      round(self.entry_price, 2),
            'stop_price':       round(self.stop_price, 2),
            'target_price':     round(self.target_price, 2),
            'momentum':         round(self.momentum, 4),
            'squeeze_bars':     self.squeeze_bars,
            'squeeze_state':    self.squeeze_state,
            'prev_atr_rth':     round(self.prev_atr_rth, 2),
            'risk_per_share':   round(self.risk_per_share(), 4),
            'reward_per_share': round(self.reward_per_share(), 4),
        }


# ── Signal scanner ────────────────────────────────────────────────────────────

def scan_day(
    ticker: str,
    trade_date: date,
    df_eth_day: pd.DataFrame,
    prev_atr_rth: float,
    daily_bullish: bool,
    daily_bearish: bool,
) -> Optional[TTMSqueezeSignal]:
    """
    Scan one day's ETH bars (with pre-computed TTM Squeeze columns) for the
    first squeeze-fire signal within the RTH entry window.

    The lookback can include premarket bars, so a prolonged premarket coil
    that fires at the RTH open is captured correctly.

    Args:
        ticker        : symbol
        trade_date    : calendar date for this trading day
        df_eth_day    : all ETH bars for this day, sorted by timestamp,
                        with columns: timestamp, close, momentum, mom_up,
                        no_sqz, in_squeeze, squeeze_state, squeeze_fired
        prev_atr_rth  : prior day's RTH ATR — used for stop sizing
        daily_bullish : True if daily 8/21/34 ribbon is bullish
        daily_bearish : True if daily 8/21/34 ribbon is bearish

    Returns the first qualifying TTMSqueezeSignal, or None.
    """
    if pd.isna(prev_atr_rth) or prev_atr_rth == 0:
        return None

    rth_open = RTH_START
    cutoff   = pd.Timestamp(TTM_ENTRY_CUTOFF).time()

    stop_dist   = TTM_STOP_ATR_MULT * prev_atr_rth
    target_dist = 2.0 * stop_dist

    df = df_eth_day.reset_index(drop=True)

    for i, bar in df.iterrows():
        bar_time = bar['timestamp'].time()

        # Entry window: strictly after RTH open, before 2 PM ET
        if bar_time <= rth_open or bar_time >= cutoff:
            continue

        # Must be a squeeze fire bar
        if not bar.get('squeeze_fired', False):
            continue

        # Verify enough squeeze bars preceded this fire
        squeeze_count = int(df.loc[:i - 1, 'in_squeeze'].sum()
                            if i > 0 else 0)
        # Use a rolling tail: count consecutive in_squeeze bars ending at i-1
        consecutive = 0
        for j in range(i - 1, -1, -1):
            if df.at[j, 'in_squeeze']:
                consecutive += 1
            else:
                break
        if consecutive < TTM_SQUEEZE_MIN_BARS:
            continue

        mom    = bar['momentum']
        mom_up = bool(bar['mom_up'])
        price  = bar['close']

        if pd.isna(mom) or pd.isna(price):
            continue

        # Squeeze state just before fire (use prior bar's state)
        prior_state = df.at[i - 1, 'squeeze_state'] if i > 0 else 'none'

        # Long: momentum positive AND rising (aqua — best quality long)
        if mom > 0 and mom_up and not daily_bearish:
            return TTMSqueezeSignal(
                ticker=ticker,
                trade_date=trade_date,
                direction='long',
                trend='bullish' if daily_bullish else 'neutral',
                entry_time=bar['timestamp'].strftime('%H:%M'),
                entry_price=price,
                stop_price=round(price - stop_dist, 4),
                target_price=round(price + target_dist, 4),
                momentum=round(mom, 4),
                squeeze_bars=consecutive,
                squeeze_state=prior_state,
                prev_atr_rth=round(prev_atr_rth, 4),
            )

        # Short: momentum negative AND falling (red — best quality short)
        if mom < 0 and not mom_up and not daily_bullish:
            return TTMSqueezeSignal(
                ticker=ticker,
                trade_date=trade_date,
                direction='short',
                trend='bearish' if daily_bearish else 'neutral',
                entry_time=bar['timestamp'].strftime('%H:%M'),
                entry_price=price,
                stop_price=round(price + stop_dist, 4),
                target_price=round(price - target_dist, 4),
                momentum=round(mom, 4),
                squeeze_bars=consecutive,
                squeeze_state=prior_state,
                prev_atr_rth=round(prev_atr_rth, 4),
            )

    return None


# ── PnL simulator ─────────────────────────────────────────────────────────────

def calc_pnl(signal: TTMSqueezeSignal, df_rth_after: pd.DataFrame) -> dict:
    """
    Simulate the trade on 1-min RTH bars from just after entry until EOD.

    Stop is checked on bar low (long) / high (short) — worst-case intrabar.
    Target is checked on bar high (long) / low (short).
    If both could trigger in the same bar, stop takes priority (conservative).
    EOD exit uses the last bar's close.

    Args:
        signal        : the trade signal
        df_rth_after  : RTH bars strictly after the entry bar, sorted by time
    """
    entry  = signal.entry_price
    stop   = signal.stop_price
    target = signal.target_price

    exit_price  = entry
    exit_time   = 'EOD'
    exit_reason = 'eod'

    for _, bar in df_rth_after.iterrows():
        if signal.direction == 'long':
            if bar['low'] <= stop:
                exit_price, exit_time, exit_reason = (
                    stop, bar['timestamp'].strftime('%H:%M'), 'stop')
                break
            if bar['high'] >= target:
                exit_price, exit_time, exit_reason = (
                    target, bar['timestamp'].strftime('%H:%M'), 'target')
                break
        else:  # short
            if bar['high'] >= stop:
                exit_price, exit_time, exit_reason = (
                    stop, bar['timestamp'].strftime('%H:%M'), 'stop')
                break
            if bar['low'] <= target:
                exit_price, exit_time, exit_reason = (
                    target, bar['timestamp'].strftime('%H:%M'), 'target')
                break
    else:
        if not df_rth_after.empty:
            last       = df_rth_after.iloc[-1]
            exit_price = last['close']
            exit_time  = last['timestamp'].strftime('%H:%M')

    pnl  = (exit_price - entry) if signal.direction == 'long' else (entry - exit_price)
    risk = signal.risk_per_share()
    r    = round(pnl / risk, 2) if risk > 0 else 0.0

    result = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'FLAT')

    return {
        **signal.to_dict(),
        'exit_price':    round(exit_price, 2),
        'exit_time':     exit_time,
        'exit_reason':   exit_reason,
        'pnl_per_share': round(pnl, 4),
        'pnl_pct':       round(pnl / entry * 100, 4),
        'r_multiple':    r,
        'result':        result,
    }
