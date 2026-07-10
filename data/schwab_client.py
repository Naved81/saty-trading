"""
data/schwab_client.py
======================
Authenticated Schwab API client, shared by the trade-history puller and
(eventually) live-alert scripts.

This module never launches an interactive browser login — it only loads
the token already saved at SCHWAB_TOKEN_PATH (see config.py). If you
haven't authenticated yet, run a one-time interactive login locally
first (this needs a browser, so it can't run in a headless container):

    python -c "
    from schwab.auth import easy_client
    from config import SCHWAB_CLIENT_ID, SCHWAB_SECRET, SCHWAB_REDIRECT_URI, TOKEN_PATH
    easy_client(SCHWAB_CLIENT_ID, SCHWAB_SECRET, SCHWAB_REDIRECT_URI, str(TOKEN_PATH))
    "

That writes schwab_token.json, which this module (and schwab-py) will
transparently refresh on subsequent runs.
"""

import sys

from config import SCHWAB_CLIENT_ID, SCHWAB_SECRET, SCHWAB_REDIRECT_URI, TOKEN_PATH


def get_client():
    """Return an authenticated schwab.client.Client using the saved token."""
    if not (SCHWAB_CLIENT_ID and SCHWAB_SECRET):
        print("[!] SCHWAB_CLIENT_ID / SCHWAB_SECRET are not set.")
        print("    Set them in .env (see .env.example) and try again.")
        sys.exit(1)

    if not TOKEN_PATH.exists():
        print(f"[!] No Schwab token found at {TOKEN_PATH}.")
        print("    Run the one-time interactive login described in this file's")
        print("    docstring (data/schwab_client.py) from a machine with a browser.")
        sys.exit(1)

    from schwab.auth import easy_client
    return easy_client(
        SCHWAB_CLIENT_ID, SCHWAB_SECRET, SCHWAB_REDIRECT_URI, str(TOKEN_PATH),
        interactive=False,
    )


def get_account_hashes(client) -> dict:
    """Return {account_number: account_hash} for every linked account."""
    resp = client.get_account_numbers()
    resp.raise_for_status()
    return {a['accountNumber']: a['hashValue'] for a in resp.json()}
