#!/bin/bash
# setup_repo.sh
# Run this once to initialize git and push to GitHub.
# Usage: bash setup_repo.sh your-github-username

GITHUB_USER=${1:-"your-github-username"}
REPO_NAME="saty-trading"

echo "Setting up git repository..."

git init
git add .
git commit -m "Initial commit — Saty ATR Credit Spread trading system

- core/: db, indicators, sessions (reusable building blocks)
- data/: Massive pull, Schwab live data
- strategies/: Saty credit spread signal logic
- backtest/: Full backtest runner
- alerts/: Signal checker, Telegram alerts, scheduler
- config.py: Centralized configuration"

git branch -M main
git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git
git push -u origin main

echo ""
echo "Done! Repository pushed to:"
echo "  https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in your credentials"
echo "  2. pip install -r requirements.txt"
echo "  3. python db/db_init.py"
echo "  4. python data/massive_pull.py"
echo "  5. python backtest/run.py --catalog"
