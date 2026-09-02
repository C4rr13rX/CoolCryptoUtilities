"""The money path must survive a flaky RPC and leave evidence behind.

Measured 2026-09-02 against wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad
on Base. Three things were true at once:

  * trading_cache.db held 62,599 trading_ops rows and ZERO 66-character
    transaction hashes, so no in-repo record could prove a trade happened.
  * A real swap -- 0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5,
    block 50785332, status 1, -0.05 USDC / +0.0000208 WETH -- executed while
    SwapService printed "[ERR] All routes failed." and fell through to Camelot
    and Sushi.
  * The cause was a 403 from the rate-limited public RPC arriving *after* the
    broadcast. ``_send`` caught it, returned ("0x", False), and discarded the
    hash.

On Base that fall-through was invisible because Camelot and Sushi are not
configured there. On Arbitrum, where all three routes resolve, the same
sequence sends the swap three times.

These tests pin the two halves of that: a broadcast is never retried on
another route, and the hash always comes back to the caller.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.swap_service import SwapOutcome, SwapService

_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_WETH = "0x4200000000000000000000000000000000000006"
_REAL_HASH = "0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5"


class _FakeAcct:
    address = "0x291c854811e92906a658Fb94Aa511bF919f968ad"


class _FakeEth:
    chain_id = 8453
    default_account = None


class _FakeW3:
    eth = _FakeEth()


class _FakeBridge:
    """Signs nothing; just enough surface for SwapService to run."""

    acct = _FakeAcct()

    def __init__(self, *, broadcast_hash: str = _REAL_HASH, raise_on_send: bool = False):
        self.broadcast_hash = broadcast_hash
        self.raise_on_send = raise_on_send
        self.sends = 0

    def _w3(self, chain):
        return _FakeW3()

    def erc20_decimals(self, chain, token):
        return 6 if token.lower() == _USDC.lower() else 18

    def erc20_allowance(self, chain, token, owner, spender):
        return 2 ** 256 - 1  # already approved; keep the test off the approve path

    def send_prebuilt_tx(self, chain, to, data, *, value=0, gas=None, fee_scope=None, fee_urgency=None):
        self.sends += 1
        if self.raise_on_send:
            raise RuntimeError("connection reset before broadcast")
        return self.broadcast_hash


class _Provider:
    """A quote provider that always produces a routable tx."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def quote_and_build(self, *a, **kw):
        self.log.append(self.name)
        return {
            "aggregator": self.name,
            "allowanceTarget": _WETH,
            "tx": {"to": _WETH, "data": "0x", "value": 0, "gas": 150000},
            "buyAmount": "20855309348927",
            "fee": 100,
        }


def _service(bridge, log):
    svc = SwapService.__new__(SwapService)
    svc.bridge = bridge
    svc.zx = None
    svc.uni = _Provider("uniswap", log)
    svc.camelot = _Provider("camelot", log)
    svc.sushi = _Provider("sushi", log)
    return svc


@pytest.fixture(autouse=True)
def _no_route_only(monkeypatch):
    monkeypatch.delenv("ROUTE_ONLY", raising=False)
    monkeypatch.delenv("SWAP_ENABLE_0X", raising=False)


def test_unreadable_receipt_does_not_spend_the_money_again(monkeypatch):
    """The exact production failure: broadcast succeeds, receipt read 403s.

    Before the fix this returned ("0x", False) and the caller tried Camelot
    and then Sushi -- three swaps for one decision on any chain where those
    routes resolve.
    """
    quoted: list[str] = []
    bridge = _FakeBridge()
    svc = _service(bridge, quoted)

    # Every RPC refuses to answer, exactly as the rate-limited endpoint did.
    monkeypatch.setattr(svc, "_confirm_receipt", lambda *a, **kw: None)

    outcome = svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert bridge.sends == 1, f"broadcast once, got {bridge.sends} transactions"
    assert quoted == ["uniswap"], f"no route may follow a broadcast, got {quoted}"
    assert outcome.broadcast is True
    assert outcome.tx_hash == _REAL_HASH, "a broadcast hash must never be discarded"
    assert outcome.confirmed is None, "unknown is not the same as failed"
    assert outcome.ok is False, "an unconfirmed swap must not claim success"


def test_successful_swap_returns_its_hash():
    quoted: list[str] = []
    bridge = _FakeBridge()
    svc = _service(bridge, quoted)
    svc._confirm_receipt = lambda *a, **kw: True

    outcome = svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert outcome.ok is True
    assert outcome.broadcast is True
    assert outcome.confirmed is True
    assert outcome.route == "UniswapV3"
    assert len(outcome.tx_hash) == 66 and outcome.tx_hash.startswith("0x"), (
        "callers audit for 66-character hashes; anything else is unverifiable"
    )
    assert bridge.sends == 1


def test_reverted_swap_is_not_retried_on_another_route():
    """A revert still spent gas and consumed the nonce. Do not re-send."""
    quoted: list[str] = []
    bridge = _FakeBridge()
    svc = _service(bridge, quoted)
    svc._confirm_receipt = lambda *a, **kw: False

    outcome = svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert bridge.sends == 1
    assert quoted == ["uniswap"]
    assert outcome.ok is False
    assert outcome.broadcast is True
    assert outcome.confirmed is False
    assert outcome.tx_hash == _REAL_HASH


def test_failure_before_broadcast_still_falls_through_to_other_routes():
    """The fall-through is only wrong *after* money moves; keep it otherwise."""
    quoted: list[str] = []
    bridge = _FakeBridge(raise_on_send=True)
    svc = _service(bridge, quoted)

    outcome = svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert quoted == ["uniswap", "camelot", "sushi"], (
        f"nothing was broadcast, so every route should be tried: {quoted}"
    )
    assert outcome.ok is False
    assert outcome.broadcast is False
    assert outcome.tx_hash == ""
    assert "all_routes_failed" in outcome.reason
