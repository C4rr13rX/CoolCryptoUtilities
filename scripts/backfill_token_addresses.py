"""Learn token contract addresses for the pools this bot already streams.

services/token_address_book.py is taught by the trending fetch, but the bot
streams a much wider universe than whatever is trending right now: on
2026-09-02 only five of the ten symbols with a recorded trade outcome could be
resolved from trending alone, so the rest still refused a live entry with
``reason=token_unresolved``.

We already hold the POOL address for those pairs, in
data/base_pair_provider_assignment.json, data/pair_index_<chain>.json and the
``trending.pair_address`` recorded on each discovered token. A pool is not a
token, but GeckoTerminal will name a pool's two tokens, thirty pools per call:

    /networks/{network}/pools/multi/{a1,a2,...}

so this walks the pools we know and writes down the token addresses they point
at. It only stores what the API reports -- nothing here guesses an address,
because the failure mode of guessing is sending real funds to the wrong
contract.

Usage::

    python scripts/backfill_token_addresses.py            # base, dry run
    python scripts/backfill_token_addresses.py --apply
    python scripts/backfill_token_addresses.py --chain base --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.token_address_book import (  # noqa: E402
    is_token_address,
    known_symbols,
    record_many,
)

GECKO_MULTI = "https://api.geckoterminal.com/api/v2/networks/{network}/pools/multi/{addrs}"
_GECKO_NETWORKS = {
    "base": "base",
    "ethereum": "eth",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon_pos",
    "bsc": "bsc",
}

#: GeckoTerminal's documented ceiling for the multi endpoint.
BATCH = 30
#: Free tier is ~30 calls/minute; stay comfortably under it.
SLEEP_SEC = 2.5

_ADDR = re.compile(r"^0x[0-9a-fA-F]{40}$")
DB = ROOT / "storage" / "trading_cache.db"


def known_pool_addresses(chain: str) -> Dict[str, str]:
    """Every 20-byte pool address we hold for ``chain``, mapped to its symbol.

    Uniswap v4 pool ids are 32 bytes and are deliberately excluded: they are
    not addresses and the endpoint cannot resolve them.
    """
    pools: Dict[str, str] = {}

    assign = ROOT / "data" / f"{chain}_pair_provider_assignment.json"
    if assign.exists():
        try:
            for addr, meta in (json.loads(assign.read_text("utf-8")).get("pairs") or {}).items():
                if _ADDR.match(addr):
                    pools.setdefault(addr.lower(), str((meta or {}).get("symbol") or ""))
        except (OSError, ValueError):
            pass

    index = ROOT / "data" / f"pair_index_{chain}.json"
    if index.exists():
        try:
            for addr, meta in (json.loads(index.read_text("utf-8")) or {}).items():
                if _ADDR.match(addr):
                    pools.setdefault(addr.lower(), str((meta or {}).get("symbol") or ""))
        except (OSError, ValueError):
            pass

    if DB.exists():
        try:
            conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5.0)
            try:
                rows = conn.execute(
                    "SELECT symbol, metadata FROM discovery_discoveredtoken WHERE chain=?",
                    (chain,),
                ).fetchall()
            finally:
                conn.close()
            for symbol, raw in rows:
                try:
                    meta = json.loads(raw or "{}")
                except ValueError:
                    continue
                addr = str((meta.get("trending") or {}).get("pair_address") or "")
                if _ADDR.match(addr):
                    pools.setdefault(addr.lower(), str(symbol or ""))
        except sqlite3.Error:
            pass

    return pools


def _tokens_from_pool(entry: dict) -> Dict[str, str]:
    """``{SYMBOL: address}`` for a pool, from its name and its relationships."""
    attrs = entry.get("attributes") or {}
    rel = entry.get("relationships") or {}
    # "Basecat / WETH 1%" -> ["Basecat", "WETH"]
    parts = [p.strip() for p in str(attrs.get("name") or "").split("/")]
    base_sym = parts[0] if parts else ""
    quote_sym = parts[1].split()[0] if len(parts) > 1 and parts[1] else ""
    out: Dict[str, str] = {}
    for sym, key in ((base_sym, "base_token"), (quote_sym, "quote_token")):
        tid = str(((rel.get(key) or {}).get("data") or {}).get("id") or "")
        addr = tid.rsplit("_", 1)[-1] if "_" in tid else tid
        if sym and is_token_address(addr):
            out[sym.upper()] = addr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="base")
    ap.add_argument("--apply", action="store_true", help="write to the address book")
    ap.add_argument("--limit", type=int, default=0, help="cap pools examined (0 = all)")
    args = ap.parse_args()

    chain = args.chain.lower()
    network = _GECKO_NETWORKS.get(chain)
    if not network:
        print(f"unsupported chain {chain!r}")
        return 2

    pools = known_pool_addresses(chain)
    addrs = sorted(pools)
    if args.limit:
        addrs = addrs[: args.limit]
    before = known_symbols(chain)
    print(f"{len(addrs)} known pool addresses on {chain}; "
          f"address book holds {len(before)} symbols")
    if not args.apply:
        print("dry run -- pass --apply to write")

    discovered: Dict[str, str] = {}
    failed = 0
    for i in range(0, len(addrs), BATCH):
        batch = addrs[i:i + BATCH]
        url = GECKO_MULTI.format(network=network, addrs=",".join(batch))
        try:
            resp = requests.get(
                url, timeout=25,
                headers={"User-Agent": "Mozilla/5.0 (compatible; R3V3N1R/1.0)",
                         "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
        except Exception as exc:  # noqa: BLE001
            failed += len(batch)
            print(f"  batch {i // BATCH + 1}: FAILED {exc}")
            time.sleep(SLEEP_SEC)
            continue
        for entry in data:
            discovered.update(_tokens_from_pool(entry))
        print(f"  batch {i // BATCH + 1}: {len(batch)} pools -> "
              f"{len(discovered)} distinct tokens so far")
        time.sleep(SLEEP_SEC)

    new = {s: a for s, a in discovered.items() if s not in before}
    print(f"\nresolved {len(discovered)} tokens, {len(new)} of them new"
          + (f"; {failed} pools could not be fetched" if failed else ""))
    for sym, addr in sorted(new.items()):
        print(f"  + {sym:14} {addr}")

    if args.apply and discovered:
        stored = record_many(chain, discovered, source="geckoterminal-pool-backfill")
        print(f"\nstored {stored} mappings; book now holds "
              f"{len(known_symbols(chain))} symbols on {chain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
