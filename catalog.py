"""Show all tickers and timeframes in the market data database."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.db import get_connection, get_catalog

con = get_connection()
print(get_catalog(con).to_string(index=False))
con.close()
