# saty-trading

Automated trading alert system built around the Saty Mahajan indicator framework.

## Strategy
Saty ATR Credit Spread — sells OTM put or call spreads at 10:30 AM CT on days 
where the morning has made a meaningful directional move, confirmed by the 
Phase Oscillator, Pivot Ribbon, and BB Compression indicators.

**Backtest results (SPY + QQQ + IWM + GLD, 2018–2026):**
- 1,173 trades | 96.7% win rate | $78,140 total per contract

## Project Structure

```
saty-trading/
├── core/                    # reusable building blocks
│   ├── db.py                # database connection and helpers
│   ├── indicators.py        # ATR, SPO, ribbon, compression
│   └── sessions.py          # RTH/ETH filters, resampling
├── data/                    # data acquisition
│   ├── massive_pull.py      # Massive API → SQLite
│   └── schwab_live.py       # Schwab API → live bars
├── strategies/              # strategy logic
│   └── saty_credit_spread.py
├── backtest/                # backtesting engine
│   └── run.py
├── alerts/                  # alert app
│   ├── signal_checker.py
│   ├── telegram_alert.py
│   └── scheduler.py
├── db/                      # database schema
│   └── db_init.py
└── config.py                # all configuration in one place
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in your credentials
python db/db_init.py        # create database
python data/massive_pull.py # pull historical data
```

## Usage

```bash
# Backtest
python backtest/run.py
python backtest/run.py --ticker QQQ --output trades.csv

# Live alerts (runs daily at 10:30 AM CT)
python alerts/scheduler.py
```

## Environment Variables

See `.env.example` for all required variables. Never commit `.env` to git.
