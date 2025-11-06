"""
Interactive helper for requesting a 0x Permit2 quote. This script requires
ZEROX_API_KEY and optional TEST_* environment variables. It is not part of the
automated test suite; invoke manually:

    python tools/zerox_permit2_quote.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.env_loader import EnvLoader


def main() -> int:
    EnvLoader.load()

    raw_key = os.getenv("ZEROX_API_KEY", "")
    stripped_key = raw_key.strip().strip("'\"")
    print("[env] ZEROX_API_KEY repr(raw):", repr(raw_key))
    print("[env] ZEROX_API_KEY repr(stripped):", repr(stripped_key))
    print(
        "[env] len:",
        len(stripped_key),
        "head:",
        stripped_key[:4],
        "tail:",
        stripped_key[-4:] if stripped_key else "",
    )

    ans = input("Proceed with this API key? [Y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("wrong key (aborting)")
        return 1

    base_url = "https://api.0x.org"
    chain_id = os.getenv("TEST_CHAIN_ID", "42161")  # Arbitrum One
    sell_token = os.getenv("TEST_SELL_TOKEN", "0x45D9831d8751B2325f3DBf48db748723726e1C8c")  # EVA
    buy_token = os.getenv("TEST_BUY_TOKEN", "0x6985884c4392d348587b19cb9eaaf157f13271cd")  # ZRO
    sell_amount = os.getenv("TEST_SELL_AMOUNT", "246600000000000000")  # 0.2466
    taker = os.getenv("TAKER_ADDR", "0x291c854811e92906a658Fb94Aa511bF919f968ad")

    query = {
        "chainId": chain_id,
        "sellToken": sell_token,
        "buyToken": buy_token,
        "sellAmount": sell_amount,
        "taker": taker,
    }
    url = f"{base_url}/swap/permit2/quote?{urllib.parse.urlencode(query)}"
    headers = {
        "0x-api-key": stripped_key,
        "0x-version": "v2",
        "Accept": "application/json",
    }

    print("[req] GET", url)
    print(
        "[hdr] 0x-api-key len:",
        len(stripped_key),
        "head:",
        stripped_key[:4],
        "tail:",
        stripped_key[-4:] if stripped_key else "",
    )

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            print("[http]", response.status)
            try:
                print(json.dumps(json.loads(body), indent=2)[:2000])
            except Exception:
                print(body[:2000])
    except urllib.error.HTTPError as exc:  # pragma: no cover - manual tool
        body = exc.read().decode("utf-8", errors="ignore")
        print("[http]", exc.code)
        print(body[:2000])
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
