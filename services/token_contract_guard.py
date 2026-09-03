"""Refuse token addresses that do not behave like tradeable ERC-20s.

Measured 2026-09-03: ``data/token_addresses.json`` held eight addresses of the
shape shared by BASECAT, BLUECHIP, NVDAC, BASEJUICE,
AAPL, GOOGLC, METAC and RAWR. ``eth_getCode`` returns ONE byte for every one
of them, so none is an ERC-20; but ``decimals()`` answered 18, so every check
downstream was satisfied. The bot spent 1.50 USDC entering BASECAT across two
swaps that settled on chain and can never be sold back, and the resulting
``no_fill_detected`` retries ended the only burst of rapid profitable trading
this system has produced.

A standing order asking an agent not to do this is a request. This is the
enforcement: an address cannot reach a swap without having been INTERROGATED
ON CHAIN and answered like a token.

Why the checks are behavioural rather than a blocklist
------------------------------------------------------
Blocking the eight known stubs would stop those eight. The ninth arrives
tomorrow from a discovery feed, under a different prefix, and the same money
is lost. Nothing here is hardcoded to an address, a symbol or a prefix: every
verdict comes from what the chain says about that specific address, right now.

An address must pass ALL of these, each a live call:

  1. it has deployed code, and enough of it to implement a token;
  2. ``decimals()`` answers, and the answer is in range (0-36);
  3. ``totalSupply()`` answers, and is greater than zero -- a token nobody
     holds cannot be sold;
  4. ``balanceOf(address)`` answers for a probe address, i.e. the one call a
     swap actually depends on is proven to work before any money moves.

The stubs fail 1 and 3. A contract that is real but not a token fails 2 or 4.
A token that is real but dead -- zero supply -- fails 3, and that is wanted:
it cannot be exited either.

Failure direction is chosen deliberately. An address that answers WRONG is
refused; an address we cannot REACH is allowed, and logged, because an RPC
outage is our problem rather than the token's and must not halt trading.
Refusals are cached permanently; allow-on-outage is never cached, so the next
call re-checks instead of inheriting an outage as a verdict.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Dict, Optional, Tuple

#: Below this there is not enough code to implement transfer(). Real ERC-20s
#: compile to hundreds of bytes; the measured stubs were 1. Configurable, but
#: it is only the first of four gates, not the whole test.
MIN_CONTRACT_CODE_BYTES = int(os.getenv("MIN_CONTRACT_CODE_BYTES", "64"))

#: Re-verify a previously-good token after this long, so a token that is
#: rugged or self-destructs mid-run stops being trusted on stale evidence.
REVERIFY_AFTER_SEC = float(os.getenv("TOKEN_GUARD_REVERIFY_SEC", "3600"))

#: An address the guard can safely call balanceOf on. Any address works; using
#: the zero address avoids depending on the bot's own wallet being funded.
_PROBE = "0x0000000000000000000000000000000000000000"

_RPCS: Dict[str, tuple] = {
    "base": (
        "https://base-rpc.publicnode.com",
        "https://base.llamarpc.com",
        "https://1rpc.io/base",
        "https://mainnet.base.org",
    ),
    "ethereum": ("https://ethereum-rpc.publicnode.com", "https://eth.llamarpc.com"),
    "arbitrum": ("https://arbitrum-one-rpc.publicnode.com",),
    "optimism": ("https://optimism-rpc.publicnode.com",),
    "polygon": ("https://polygon-bor-rpc.publicnode.com",),
}

# (chain, address) -> (verdict, checked_at, reason)
_verdicts: Dict[Tuple[str, str], Tuple[bool, float, str]] = {}
_lock = threading.Lock()

_SENTINELS = {
    "",
    "native",
    "0x0000000000000000000000000000000000000000",
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
}

# Function selectors, so no ABI dependency is needed.
_SEL_DECIMALS = "0x313ce567"
_SEL_TOTAL_SUPPLY = "0x18160ddd"
_SEL_BALANCE_OF = "0x70a08231"


def _rpc(chain: str, method: str, params: list) -> Tuple[Optional[object], bool]:
    """Return (result, reachable). reachable=False means no RPC answered."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 1,
                          "method": method, "params": params}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    reachable = False
    for url in _RPCS.get(str(chain or "").lower(), _RPCS["base"]):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = json.loads(resp.read()) or {}
            reachable = True
            if "result" in body:
                return body["result"], True
            # A JSON-RPC error is a real answer: the node reached the contract
            # and the call failed. Treat it as "answered, badly".
            return None, True
        except Exception:  # noqa: BLE001
            continue
    return None, reachable


def _as_int(hexstr: Optional[object]) -> Optional[int]:
    if not isinstance(hexstr, str) or not hexstr.startswith("0x") or hexstr == "0x":
        return None
    try:
        return int(hexstr, 16)
    except ValueError:
        return None


def _interrogate(chain: str, addr: str) -> Tuple[Optional[bool], str]:
    """Ask the chain whether this address behaves like a token.

    Returns (verdict, reason). verdict None means unreachable -- not a
    judgement about the token, so the caller must not cache it.
    """
    # 1. Deployed code.
    code, reachable = _rpc(chain, "eth_getCode", [addr, "latest"])
    if not reachable:
        return None, "rpc_unreachable"
    size = 0 if not isinstance(code, str) or code == "0x" else (len(code) - 2) // 2
    if size <= MIN_CONTRACT_CODE_BYTES:
        return False, "code_%d_bytes" % size

    # 2. decimals() answers, in a sane range.
    raw, reachable = _rpc(chain, "eth_call",
                          [{"to": addr, "data": _SEL_DECIMALS}, "latest"])
    if not reachable:
        return None, "rpc_unreachable"
    dec = _as_int(raw)
    if dec is None:
        return False, "decimals_unanswered"
    if dec > 36:
        return False, "decimals_%d_out_of_range" % dec

    # 3. totalSupply() answers and is non-zero. A supply of zero cannot be
    #    bought into and cannot be sold out of.
    raw, reachable = _rpc(chain, "eth_call",
                          [{"to": addr, "data": _SEL_TOTAL_SUPPLY}, "latest"])
    if not reachable:
        return None, "rpc_unreachable"
    supply = _as_int(raw)
    if supply is None:
        return False, "total_supply_unanswered"
    if supply <= 0:
        return False, "total_supply_zero"

    # 4. balanceOf() answers -- the one call every swap depends on.
    data = _SEL_BALANCE_OF + "0" * 24 + _PROBE[2:]
    raw, reachable = _rpc(chain, "eth_call",
                          [{"to": addr, "data": data}, "latest"])
    if not reachable:
        return None, "rpc_unreachable"
    if _as_int(raw) is None:
        return False, "balance_of_unanswered"

    return True, "ok_code_%d_dec_%d" % (size, dec)


def verify(chain: str, address: Optional[str]) -> Tuple[bool, str]:
    """(allowed, reason). Refuses on a wrong answer, allows on an outage."""
    addr = str(address or "").strip()
    if addr.lower() in _SENTINELS:
        return True, "native"
    if not addr.startswith("0x") or len(addr) != 42:
        return False, "not_an_address"

    key = (str(chain or "").lower(), addr.lower())
    now = time.time()
    with _lock:
        cached = _verdicts.get(key)
    if cached is not None:
        ok, at, reason = cached
        # A refusal stands. An approval is re-checked periodically so a token
        # that dies mid-run is not trusted forever on one old measurement.
        if not ok or (now - at) < REVERIFY_AFTER_SEC:
            return ok, reason

    verdict, reason = _interrogate(chain, addr)

    if verdict is None:
        _log("token-guard",
             "could not verify %s on %s (%s); allowing rather than halting trading"
             % (addr, chain, reason), "warning")
        return True, reason           # deliberately not cached

    with _lock:
        _verdicts[key] = (verdict, now, reason)

    if not verdict:
        _log("token-guard",
             "REFUSED %s on %s: %s -- does not behave like a tradeable ERC-20"
             % (addr, chain, reason), "error")
    return verdict, reason


def is_real_contract(chain: str, address: Optional[str]) -> bool:
    """Boolean form of :func:`verify`, for call sites that only branch."""
    ok, _ = verify(chain, address)
    return ok


def _log(channel: str, message: str, severity: str = "info") -> None:
    try:
        from services.logging_utils import log_message

        log_message(channel, message, severity=severity)
    except Exception:  # noqa: BLE001
        pass


def clear_cache() -> None:
    """Drop remembered verdicts. For tests, and after editing the book."""
    with _lock:
        _verdicts.clear()
