"""Contract addresses for the tokens this bot actually trades.

A live swap needs the ERC-20 contract address of both sides. The bot could
only get one from two places: the wallet's current holdings, and the static
core-token catalog -- eight symbols on base. The tradeable universe is the
feed's discovered symbols, so every live entry on anything else was refused
with ``reason=token_unresolved``. Measured 2026-09-02: of the ten symbols
with a recorded trade outcome, NINE had a base token outside the catalog.

The address was never missing, only discarded. GeckoTerminal returns it on
every trending pool::

    "relationships": {"base_token": {"data": {"id": "base_0xb200...f41d01"}}}

and services/discovery/trending_fetcher.py kept ``baseToken: {"symbol": ...}``
and dropped the id. Discovery stored the POOL address instead, which is not a
token and cannot be swapped -- and for a Uniswap v4 pool it is not even an
address, it is a 32-byte pool id.

So this is a learned book: discovery writes down the addresses it is already
being told, and the resolver reads them. It deliberately does NOT invent
addresses. An unknown symbol stays unknown and the trade stays blocked,
because the failure mode of guessing here is sending real funds to the wrong
contract.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from services.atomic_json import file_lock, read_json, write_json

#: Anchored to the repo root: the trading process runs from there while web
#: workers run from web/, and a relative path silently split this in two.
BOOK_PATH = Path(
    os.getenv("TOKEN_ADDRESS_BOOK_PATH")
    or (Path(__file__).resolve().parents[1] / "data" / "token_addresses.json")
)

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

#: Addresses that are the right SHAPE but are not ERC-20 contracts. Upstreams
#: use them as stand-ins for the chain's native coin: GeckoTerminal reported
#: base ETH as the zero address, and 0xEeee...EEeE is the other common
#: sentinel. Both would be accepted by a hex check and neither can be swapped;
#: the zero address in particular is where tokens go to be burned.
_NON_ERC20 = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
}


def is_token_address(value: Any) -> bool:
    """True for a 20-byte hex address of something that could be an ERC-20.

    Length is half the point: a Uniswap v4 pool id is 32 bytes and looks close
    enough to pass a ``startswith("0x")`` check, which is how a pool id reaches
    a swap call as if it were a token. The sentinel check is the other half --
    a native-coin placeholder is a correctly shaped address for a contract that
    does not exist.
    """
    raw = str(value or "").strip()
    if not _ADDRESS_RE.match(raw):
        return False
    return raw.lower() not in _NON_ERC20


def _normalise_symbol(symbol: str) -> str:
    """Upper-cased, pair suffix removed: ``TOAD-USDC`` and ``toad`` both key TOAD."""
    sym = str(symbol or "").strip().upper()
    if "-" in sym:
        sym = sym.split("-", 1)[0]
    return sym


def _normalise_chain(chain: str) -> str:
    return str(chain or "").strip().lower()


def _gecko_token_id_to_address(token_id: str) -> Optional[str]:
    """``"base_0xabc..."`` -> ``"0xabc..."``. Returns None unless it is an address."""
    raw = str(token_id or "").strip()
    if "_" in raw:
        raw = raw.rsplit("_", 1)[1]
    return raw if is_token_address(raw) else None


def record(chain: str, symbol: str, address: str, *, source: str = "discovery") -> bool:
    """Learn one symbol -> address mapping. False if it was not usable or not stored."""
    chain_l = _normalise_chain(chain)
    sym = _normalise_symbol(symbol)
    addr = str(address or "").strip()
    if not chain_l or not sym or not is_token_address(addr):
        return False
    with file_lock(BOOK_PATH):
        raw, ok = read_json(BOOK_PATH, default=None)
        if not ok:
            # Never blank a book of addresses because one read lost a race.
            return False
        book: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        chain_book = book.setdefault(chain_l, {})
        if not isinstance(chain_book, dict):
            chain_book = {}
            book[chain_l] = chain_book
        prior = chain_book.get(sym)
        if isinstance(prior, dict) and str(prior.get("address", "")).lower() == addr.lower():
            return True                       # already known, no rewrite
        chain_book[sym] = {"address": addr, "source": source, "ts": time.time()}
        return write_json(BOOK_PATH, book)


def record_many(chain: str, mapping: Dict[str, str], *, source: str = "discovery") -> int:
    """Learn several mappings under one lock. Returns how many were stored."""
    pairs = [
        (_normalise_symbol(s), str(a or "").strip())
        for s, a in (mapping or {}).items()
    ]
    pairs = [(s, a) for s, a in pairs if s and is_token_address(a)]
    chain_l = _normalise_chain(chain)
    if not chain_l or not pairs:
        return 0
    with file_lock(BOOK_PATH):
        raw, ok = read_json(BOOK_PATH, default=None)
        if not ok:
            return 0
        book: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        chain_book = book.setdefault(chain_l, {})
        if not isinstance(chain_book, dict):
            chain_book = {}
            book[chain_l] = chain_book
        now = time.time()
        stored = 0
        for sym, addr in pairs:
            prior = chain_book.get(sym)
            if isinstance(prior, dict) and str(prior.get("address", "")).lower() == addr.lower():
                continue
            chain_book[sym] = {"address": addr, "source": source, "ts": now}
            stored += 1
        if stored and not write_json(BOOK_PATH, book):
            return 0
        return stored


def record_from_gecko_ids(
    chain: str, base_symbol: str, base_id: str, quote_symbol: str, quote_id: str
) -> int:
    """Learn both sides of a GeckoTerminal pool from its ``relationships`` ids."""
    mapping: Dict[str, str] = {}
    for sym, tid in ((base_symbol, base_id), (quote_symbol, quote_id)):
        addr = _gecko_token_id_to_address(tid)
        if addr and _normalise_symbol(sym):
            mapping[_normalise_symbol(sym)] = addr
    return record_many(chain, mapping, source="geckoterminal")


def lookup(chain: str, symbol: str) -> Optional[str]:
    """The learned address for ``symbol`` on ``chain``, or None."""
    chain_l = _normalise_chain(chain)
    sym = _normalise_symbol(symbol)
    if not chain_l or not sym:
        return None
    raw, ok = read_json(BOOK_PATH, default=None)
    if not ok or not isinstance(raw, dict):
        return None
    entry = (raw.get(chain_l) or {}).get(sym)
    if isinstance(entry, dict):
        addr = entry.get("address")
    else:
        addr = entry                          # tolerate a plain symbol->address map
    return str(addr) if is_token_address(addr) else None


def known_symbols(chain: str) -> Dict[str, str]:
    """Every learned symbol -> address on ``chain``. Useful for reporting."""
    chain_l = _normalise_chain(chain)
    raw, ok = read_json(BOOK_PATH, default=None)
    if not ok or not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for sym, entry in (raw.get(chain_l) or {}).items():
        addr = entry.get("address") if isinstance(entry, dict) else entry
        if is_token_address(addr):
            out[str(sym)] = str(addr)
    return out


__all__ = [
    "BOOK_PATH",
    "is_token_address",
    "known_symbols",
    "lookup",
    "record",
    "record_from_gecko_ids",
    "record_many",
]
