"""
strategies/saty_credit_spread.py
=================================
Saty ATR Credit Spread — signal generation.

This module is pure strategy logic. It knows nothing about:
  - Where data came from (DB, CSV, live API)
  - How alerts are sent
  - How backtests are run

It only answers one question: given this data, should we trade today?

STRATEGY SUMMARY
----------------
At 10:30 AM CT (11:30 ET), check five conditions:

  1. Morning move > 23.6% of RTH ATR from prior 4PM close
  2. Phase Oscillator aligned with direction
  3. No Bollinger Band compression
  4. Daily 8/21/34 ribbon not fighting direction
  5. Open-to-entry momentum not fading hard against direction

If all pass → sell put spread (up morning) or call spread (down morning)
at the ±38.2% full-session ATR level from prior close.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np

from config import (
    ENTRY_TIME, MORNING_THRESH, SPREAD_WIDTH, PREMIUM, MAX_LOSS,
    OPEN_FADE_THRESH, SPX_RATIO, RTH_START
)


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    Represents a trade signal for one ticker on one day.
    Returned by check_signal() — used by both backtest and alert app.
    """
    ticker:              str
    trade_date:          date
    direction:           str              # 'sell_put_spread' | 'sell_call_spread'
    trend:               str              # 'bullish' | 'bearish' | 'neutral'

    # Price context
    prev_close:          float
    price_at_open:       float
    price_at_entry:      float
    gap_pts:             float
    gap_pct_atr:         float

    # Indicator values
    daily_atr_rth:       float
    daily_atr_full:      float
    morning_move_pct_atr:float
    open_to_entry_pct_atr:float
    spo_at_entry:        float
    in_compression:      bool

    # Spread levels
    short_strike:        float
    long_strike:         float
    spread_width:        float = SPREAD_WIDTH
    credit:              float = PREMIUM
    max_loss:            float = MAX_LOSS
    spx_est_credit:      float = field(init=False)

    def __post_init__(self):
        self.spx_est_credit = round(self.credit * SPX_RATIO, 2)

    def to_dict(self) -> dict:
        return {
            'ticker':               self.ticker,
            'date':                 str(self.trade_date),
            'direction':            self.direction,
            'trend':                self.trend,
            'gap_pts':              round(self.gap_pts, 2),
            'gap_pct_atr':          round(self.gap_pct_atr, 3),
            'price_at_open':        round(self.price_at_open, 2),
            'price_at_entry':       round(self.price_at_entry, 2),
            'open_to_entry_pct_atr':round(self.open_to_entry_pct_atr, 3),
            'prev_close':           round(self.prev_close, 2),
            'daily_atr_rth':        round(self.daily_atr_rth, 2),
            'daily_atr_full':       round(self.daily_atr_full, 2),
            'morning_move_pct_atr': round(self.morning_move_pct_atr, 3),
            'spo_at_entry':         round(self.spo_at_entry, 2),
            'short_strike':         round(self.short_strike, 2),
            'long_strike':          round(self.long_strike, 2),
            'spread_width':         self.spread_width,
            'credit':               self.credit,
            'max_loss':             self.max_loss,
            'spx_est_credit':       self.spx_est_credit,
        }

    def alert_text(self) -> str:
        """Formatted Telegram alert message."""
        arrow  = '📈' if self.direction == 'sell_put_spread' else '📉'
        spread = 'PUT' if self.direction == 'sell_put_spread' else 'CALL'
        return (
            f"{arrow} *SATY SIGNAL — {self.ticker}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Direction:*  Sell {spread} Spread\n"
            f"*Trend:*      {self.trend.capitalize()}\n\n"
            f"*Strikes:*\n"
            f"  Short: ${self.short_strike:.2f}\n"
            f"  Long:  ${self.long_strike:.2f}\n"
            f"  Width: ${self.spread_width:.2f}\n\n"
            f"*Credit:*     ~${self.credit:.2f}  "
            f"(SPX est: ~${self.spx_est_credit:.2f})\n"
            f"*Max Loss:*   ${self.max_loss:.2f}\n\n"
            f"*Context:*\n"
            f"  Prev Close:    ${self.prev_close:.2f}\n"
            f"  Open:          ${self.price_at_open:.2f}\n"
            f"  Entry (11:30): ${self.price_at_entry:.2f}\n"
            f"  Gap:           {self.gap_pts:+.2f} pts "
            f"({self.gap_pct_atr:+.3f}× ATR)\n"
            f"  Morning move:  {self.morning_move_pct_atr:+.3f}× RTH ATR\n"
            f"  SPO:           {self.spo_at_entry:.1f}\n"
            f"  ATR (RTH):     ${self.daily_atr_rth:.2f}\n"
            f"  ATR (Full):    ${self.daily_atr_full:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Entry: 10:30 AM CT | Exit: EOD_"
        )


# ── Core signal check ─────────────────────────────────────────────────────────

def check_signal(ticker: str,
                 row: pd.Series,
                 spo_at_entry: float,
                 in_compression: bool,
                 price_at_entry: float,
                 price_at_open: float) -> Optional[Signal]:
    """
    Check all five entry conditions for one ticker on one day.

    Args:
        ticker          : ticker symbol
        row             : one row from the daily reference DataFrame
                          (must have: prev_close, prev_atr_rth, prev_atr_full,
                           short_put, short_call, bullish, bearish,
                           gap_pts, gap_pct_atr)
        spo_at_entry    : SPO value at 11:30 ET
        in_compression  : BB compression flag at 11:30 ET
        price_at_entry  : close price at 11:30 ET
        price_at_open   : close price at 09:30 ET open

    Returns:
        Signal if all conditions pass, None otherwise.
    """
    prev_c   = row['prev_close']
    atr_rth  = row['prev_atr_rth']
    atr_full = row['prev_atr_full']

    if pd.isna(prev_c) or pd.isna(atr_rth) or atr_rth == 0:
        return None

    # Morning move as % of RTH ATR — dynamic, regime-stable
    morning_pct = (price_at_entry - prev_c) / atr_rth

    # Open-to-entry momentum as % of RTH ATR
    open_to_entry = ((price_at_entry - price_at_open) / atr_rth
                     if not pd.isna(price_at_open) else 0.0)

    # Filter 3: no compression
    if in_compression:
        return None

    if morning_pct > MORNING_THRESH:
        # ── PUT SPREAD (morning up) ──
        # Filter 4: trend not bearish
        if row.get('bearish', False):
            return None
        # Filter 2: SPO positive
        if spo_at_entry < 0:
            return None
        # Filter 5: not fading from open
        if open_to_entry < -OPEN_FADE_THRESH:
            return None

        direction    = 'sell_put_spread'
        short_strike = row['short_put']
        long_strike  = short_strike - SPREAD_WIDTH

    elif morning_pct < -MORNING_THRESH:
        # ── CALL SPREAD (morning down) ──
        if row.get('bullish', False):
            return None
        if spo_at_entry > 0:
            return None
        if open_to_entry > OPEN_FADE_THRESH:
            return None

        direction    = 'sell_call_spread'
        short_strike = row['short_call']
        long_strike  = short_strike + SPREAD_WIDTH

    else:
        # Morning move insufficient
        return None

    trend = ('bullish' if row.get('bullish', False) else
             'bearish' if row.get('bearish', False) else
             'neutral')

    return Signal(
        ticker=ticker,
        trade_date=row['date_ts'].date() if hasattr(row['date_ts'], 'date') else row['date_ts'],
        direction=direction,
        trend=trend,
        prev_close=prev_c,
        price_at_open=price_at_open,
        price_at_entry=price_at_entry,
        gap_pts=row.get('gap_pts', 0.0),
        gap_pct_atr=row.get('gap_pct_atr', 0.0),
        daily_atr_rth=atr_rth,
        daily_atr_full=atr_full,
        morning_move_pct_atr=morning_pct,
        open_to_entry_pct_atr=open_to_entry,
        spo_at_entry=spo_at_entry,
        in_compression=in_compression,
        short_strike=short_strike,
        long_strike=long_strike,
    )


# ── PnL calculator (used by backtest) ────────────────────────────────────────

def calc_pnl(signal: Signal, eod_close: float) -> dict:
    """
    Calculate PnL for a signal given the EOD close.
    Returns a dict with pnl_per_share, pnl_per_contract, result.
    """
    if signal.direction == 'sell_put_spread':
        if eod_close >= signal.short_strike:
            pnl = PREMIUM
        elif eod_close <= signal.long_strike:
            pnl = -MAX_LOSS
        else:
            pnl = PREMIUM - (signal.short_strike - eod_close)
    else:
        if eod_close <= signal.short_strike:
            pnl = PREMIUM
        elif eod_close >= signal.long_strike:
            pnl = -MAX_LOSS
        else:
            pnl = PREMIUM - (eod_close - signal.short_strike)

    result = ('WIN'     if pnl >= PREMIUM * 0.95 else
              'LOSS'    if pnl < 0 else
              'PARTIAL')

    return {
        **signal.to_dict(),
        'eod_close':        round(eod_close, 2),
        'pnl_per_share':    round(pnl, 4),
        'pnl_per_contract': round(pnl * 100, 2),
        'spx_est_pnl':      round(pnl * SPX_RATIO * 100, 2),
        'result':           result,
    }
