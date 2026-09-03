"""Base must keep an endpoint that will actually answer for a receipt.

Once a fill is read from the transaction receipt (services/fill_receipt.py),
an unreadable receipt is not a cosmetic gap: it drops the bot back onto the
wallet-balance delta, the measurement that recorded our first two real trades
as ``no_fill_detected``.

Probed 2026-09-03 against the receipt of a real swap,
0xd4c2d4df7886a80f4113314d518772e80113a68dd64c3321f2e873aec7c9c196:

    base-rpc.publicnode.com  403  "Archive requests require a personal token"
    base.publicnode.com      403  same
    base.llamarpc.com        521  Cloudflare, no origin
    1rpc.io/base             200  "You've reached the usage limit"
    base.drpc.org            408  "Request timeout on the free plan"
    mainnet.base.org         200  receipt returned in 0.5s
    base.meowrpc.com         200  receipt returned in 0.8s

Every configured endpoint failed and both omitted ones worked. Base was also
the only chain in the table with no official public endpoint, while ethereum
and arbitrum both carry theirs.

These tests do not hit the network -- a test that depends on a free public RPC
being up is a test that fails for reasons unrelated to this repo. They pin the
configuration decision instead, and ``SwapService`` reads this same list.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router_wallet import CHAINS

OFFICIAL = "https://mainnet.base.org"
MEASURED_BROKEN = (
    "https://base-rpc.publicnode.com",
    "https://base.publicnode.com",
    "https://base.llamarpc.com",
    "https://1rpc.io/base",
    "https://base.drpc.org",
)


def _base_rpcs() -> list[str]:
    return [u for u in CHAINS["base"]["rpcs"] if u]


def test_base_carries_its_official_public_endpoint():
    assert OFFICIAL in _base_rpcs()


def test_an_endpoint_that_answered_is_tried_before_the_ones_that_did_not():
    """SwapService.fetch_receipt walks this list in order, not by health.

    ``UltraSwapBridge._w3`` ranks by measured latency, but the receipt poll
    iterates the raw list, so order here decides how long a fill waits.
    """
    rpcs = _base_rpcs()
    first_broken = min(
        (rpcs.index(u) for u in MEASURED_BROKEN if u in rpcs), default=len(rpcs)
    )
    assert rpcs.index(OFFICIAL) < first_broken


def test_the_endpoints_that_failed_are_kept_as_fallbacks():
    """Not removed: a free endpoint that is rate-limited today answers
    tomorrow, and more places to ask is strictly better once the order is
    right. This test exists so a future cleanup does not read the comment
    above as a licence to delete them."""
    rpcs = _base_rpcs()
    assert sum(1 for u in MEASURED_BROKEN if u in rpcs) >= 3
    assert len(rpcs) >= 7


def test_every_traded_chain_has_more_than_one_endpoint():
    """One endpoint is a single point of failure for reading our own trades."""
    for chain in ("base", "ethereum", "arbitrum", "optimism"):
        urls = [u for u in CHAINS[chain]["rpcs"] if u]
        assert len(urls) >= 2, chain
        assert len(set(urls)) == len(urls), f"{chain} lists a duplicate endpoint"
