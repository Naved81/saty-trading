"""
strategies/bb_compression_breakout.py
======================================
BB Compression Breakout — signal generation and PnL simulation.

When Bollinger Bands compress inside the ATR channel (price coiling),
a breakout from that setup often produces a sustained directional move.
This is the inverse edge of the credit spread strategy, which actively
avoids compression.

ENTRY CONDITIONS
----------------
On the first qualifying 1-min bar during RTH (9:31–13:59 ET):
  1. Transition: previous bar in_compression=True, current bar=False
  2. At least BB_COMPRESSION_MIN_BARS of the last BB_COMPRESSION_LOOKBACK
     bars were in compression (confirms a real coil, not a fluke)
  3. Price is above pivot (long) or below pivot (short)
  4. |SPO| > BB_SPO_MIN aligned with direction
  5. Daily 8/21/34 ribbon not fighting direction

TRADE STRUCTURE
---------------
  Entry  : close of the signal bar
  Stop   : entry ∓ BB_STOP_ATR_MULT × prev_atr_rth
  Target : entry ± 2 × stop_distance  (2:1 reward/risk)
  Exit   : first of — stop hit, target hit, or EOD RTH close
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np

from config import (
    BB_COMPRESSION_LOOKBACK, BB_COMPRESSION_MIN_BARS,
    BB_SPO_MIN, BB_STOP_ATR_MULT, BB_ENTRY_CUTOFF, RTH_START,
)


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class BBBreakoutSignal:
    """
    Represents one BB Compression Breakout trade signal.
    Returned by scan_day() — used by both backtest and future live alert.
    """
    ticker:           str
    trade_date:       date
    direction:        str    # 'long' | 'short'
    trend:            str    # 'bullish' | 'bearish' | 'neutral'
    entry_time:       str    # 'HH:MM' ET
    entry_price:      float
    stop_price:       float
    target_price:     float
    pivot_at_entry:   float
    iatr_at_entry:    float
    spo_at_entry:     float
    compression_bars: int    # compressed bars counted in lookback window
    prev_atr_rth:     float  # daily RTH ATR used for stop sizing

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
            'pivot_at_entry':   round(self.pivot_at_entry, 2),
            'iatr_at_entry':    round(self.iatr_at_entry, 4),
            'spo_at_entry':     round(self.spo_at_entry, 2),
            'compression_bars': self.compression_bars,
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
) -> Optional[BBBreakoutSignal]:
    """
    Scan one day's ETH bars (with pre-computed indicators) for the first
    compression breakout signal within the RTH entry window.

    The lookback can include premarket bars so a multi-hour premarket
    compression that breaks out at the open is correctly captured.

    Args:
        ticker        : symbol
        trade_date    : calendar date for this trading day
        df_eth_day    : all ETH bars for this day, sorted by timestamp,
                        with columns: timestamp, close, pivot, iatr, spo,
                        in_compression
        prev_atr_rth  : previous day's RTH ATR (for stop sizing)
        daily_bullish : True if daily 8/21/34 ribbon is bullish
        daily_bearish : True if daily 8/21/34 ribbon is bearish

    Returns the first qualifying BBBreakoutSignal, or None.
    """
    if pd.isna(prev_atr_rth) or prev_atr_rth == 0:
        return None

    rth_open = RTH_START                             # 09:30
    cutoff   = pd.Timestamp(BB_ENTRY_CUTOFF).time()  # 14:00

    df = df_eth_day.reset_index(drop=True)
    n  = len(df)

    stop_dist   = BB_STOP_ATR_MULT * prev_atr_rth
    target_dist = 2.0 * stop_dist

    for i in range(BB_COMPRESSION_LOOKBACK, n):
        bar      = df.iloc[i]
        bar_time = bar['timestamp'].time()

        # Entry window: strictly after RTH open, before cutoff
        if bar_time <= rth_open or bar_time >= cutoff:
            continue

        # Must be the exact transition bar: compressed → not compressed
        if bar['in_compression']:
            continue
        if not df.iloc[i - 1]['in_compression']:
            continue  # already been out of compression — not a fresh break

        # Enough compression buildup in the lookback window
        window       = df.iloc[i - BB_COMPRESSION_LOOKBACK: i]
        n_compressed = int(window['in_compression'].sum())
        if n_compressed < BB_COMPRESSION_MIN_BARS:
            continue

        pivot = bar['pivot']
        iatr  = bar['iatr']
        spo   = bar['spo']
        price = bar['close']

        if pd.isna(pivot) or pd.isna(iatr) or iatr == 0 or pd.isna(spo):
            continue

        # Long: price above pivot, SPO bullish, ribbon not bearish
        if price > pivot and spo >= BB_SPO_MIN and not daily_bearish:
            return BBBreakoutSignal(
                ticker=ticker,
                trade_date=trade_date,
                direction='long',
                trend='bullish' if daily_bullish else 'neutral',
                entry_time=bar['timestamp'].strftime('%H:%M'),
                entry_price=price,
                stop_price=round(price - stop_dist, 4),
                target_price=round(price + target_dist, 4),
                pivot_at_entry=round(pivot, 4),
                iatr_at_entry=round(iatr, 4),
                spo_at_entry=round(spo, 2),
                compression_bars=n_compressed,
                prev_atr_rth=round(prev_atr_rth, 4),
            )

        # Short: price below pivot, SPO bearish, ribbon not bullish
        if price < pivot and spo <= -BB_SPO_MIN and not daily_bullish:
            return BBBreakoutSignal(
                ticker=ticker,
                trade_date=trade_date,
                direction='short',
                trend='bearish' if daily_bearish else 'neutral',
                entry_time=bar['timestamp'].strftime('%H:%M'),
                entry_price=price,
                stop_price=round(price + stop_dist, 4),
                target_price=round(price - target_dist, 4),
                pivot_at_entry=round(pivot, 4),
                iatr_at_entry=round(iatr, 4),
                spo_at_entry=round(spo, 2),
                compression_bars=n_compressed,
                prev_atr_rth=round(prev_atr_rth, 4),
            )

    return None


# ── PnL simulator ─────────────────────────────────────────────────────────────

def calc_pnl(signal: BBBreakoutSignal, df_rth_after: pd.DataFrame) -> dict:
    """
    Simulate the trade on 1-min RTH bars from just after entry until EOD.

    Stop is checked on bar low (long) / high (short) — worst-case intrabar fill.
    Target is checked on bar high (long) / low (short).
    If both could trigger in the same bar, stop takes priority (conservative).
    EOD exit uses the last bar's close.

    Args:
        signal        : the trade signal
        df_rth_after  : RTH bars strictly after the entry bar, sorted by time

    Returns a dict suitable for building a trades DataFrame.
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
                exit_price  = stop
                exit_time   = bar['timestamp'].strftime('%H:%M')
                exit_reason = 'stop'
                break
            if bar['high'] >= target:
                exit_price  = target
                exit_time   = bar['timestamp'].strftime('%H:%M')
                exit_reason = 'target'
                break
        else:  # short
            if bar['high'] >= stop:
                exit_price  = stop
                exit_time   = bar['timestamp'].strftime('%H:%M')
                exit_reason = 'stop'
                break
            if bar['low'] <= target:
                exit_price  = target
                exit_time   = bar['timestamp'].strftime('%H:%M')
                exit_reason = 'target'
                break
    else:
        # Loop completed without stop or target — exit at EOD close
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
