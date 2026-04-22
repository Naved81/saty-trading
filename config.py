"""
config.py
=========
Central configuration for the saty-trading project.
All strategy parameters, paths, and constants live here.
Import this module anywhere — never hardcode values in other files.

Environment variables are read here too, so other modules
just import from config rather than calling os.environ directly.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DB_PATH    = Path(os.environ.get(
    'DB_PATH',
    r"C:\UnsyncedToCloudButKeep\Trading\Data\Technical\database\market_data.db"
))
TOKEN_PATH = Path(os.environ.get('SCHWAB_TOKEN_PATH', BASE_DIR / 'schwab_token.json'))

# ── Timezone ──────────────────────────────────────────────────────────────────

ET = ZoneInfo('America/New_York')
CT = ZoneInfo('America/Chicago')

# ── Session boundaries (ET) ───────────────────────────────────────────────────

import pandas as pd

ETH_START  = pd.Timestamp('04:00').time()   # premarket open
RTH_START  = pd.Timestamp('09:30').time()   # regular session open
RTH_END    = pd.Timestamp('16:00').time()   # regular session close
POST_END   = pd.Timestamp('20:00').time()   # afterhours close

# ── Tickers ───────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']

# ── Database timeframes ───────────────────────────────────────────────────────
# (label, API multiplier, API timespan, session)
DEFAULT_TIMEFRAMES = [
    ('1m_full', 1, 'minute', 'full'),  # full 24x5 — single source of truth
    ('1d',      1, 'day',    'rth'),   # daily bars for fast reference calcs
]

# ── Strategy parameters ───────────────────────────────────────────────────────

ENTRY_TIME      = '11:30'   # ET — signal check time (10:30 AM CT)
MORNING_THRESH  = 0.236     # morning move must exceed 23.6% of RTH ATR
SHORT_PCT       = 0.382     # short strike at ±38.2% of full-session ATR
SPREAD_WIDTH    = 2.0       # spread width in dollars
PREMIUM         = 0.70      # estimated credit received per spread
MAX_LOSS        = SPREAD_WIDTH - PREMIUM   # max loss per spread

WARMUP_DAYS     = 20        # minimum bars needed for indicator stability
SPX_RATIO       = 10.0      # approximate SPX/SPY price ratio for reference

# Open-to-entry fade filter:
# skip if price is reversing more than this % of RTH ATR since open
OPEN_FADE_THRESH = 0.10

# ── Schwab API ────────────────────────────────────────────────────────────────

SCHWAB_CLIENT_ID    = os.environ.get('SCHWAB_CLIENT_ID')
SCHWAB_SECRET       = os.environ.get('SCHWAB_SECRET')
SCHWAB_REDIRECT_URI = os.environ.get('SCHWAB_REDIRECT_URI', 'https://127.0.0.1')

# ── Massive (Polygon) API ─────────────────────────────────────────────────────

MASSIVE_API_KEY = os.environ.get('MASSIVE_API_KEY')

# Chunk sizes for data pulls
CHUNK_DAYS_FULL  = 365    # 1 year per chunk — full 24x5 session
CHUNK_DAYS_DAILY = 3650   # 10 years per chunk — daily bars
DEFAULT_START    = '2018-01-01'

# API retry settings
MAX_RETRIES    = 4
RETRY_BASE_SEC = 2
REQUEST_DELAY  = 0.25

# ── BB Compression Breakout parameters ───────────────────────────────────────

BB_COMPRESSION_LOOKBACK = 15    # 1-min bars to look back for compression history
BB_COMPRESSION_MIN_BARS = 7     # min compressed bars required in that window
BB_SPO_MIN              = 10.0  # minimum |SPO| to confirm breakout direction
BB_STOP_ATR_MULT        = 0.30  # stop distance = this × prev_atr_rth from entry
BB_ENTRY_CUTOFF         = '14:00'  # no new entries at or after 2:00 PM ET

# ── Telegram ──────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID')
