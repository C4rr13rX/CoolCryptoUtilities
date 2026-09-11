"""One rate-limited endpoint must not be the end of a live exit.

Measured 2026-09-05 on the live book. The AERO-USDC position had been held
66 minutes -- past MAX_HOLD_FORCE_SECONDS=2700 -- and the forced exit ran at
12:38:13. It produced this, in full:

    [live-swap] exit sizing for AERO-USDC from chain: holds 2.87900902954275 ...
    [approve] allowance read failed: HTTPError('429 Client Error:
              Too Many Requests for url: https://mainnet.base.org/')
    [ERR] approval failed
    [UniswapV3] failed, trying Camelot...
    [CamelotV2] CamelotV2: no router configured for base.
    [CamelotV2] failed, trying SushiV2...
    [SushiV2] SushiV2 unsupported on base
    [ERR] All routes failed.

One ``eth_call`` got a 429 and the position did not sell. The same nine lines
appear 12 times in the current log and have killed live exits (CBETH, AERO)
and live entries (CBBTC) alike; on Base there is no second route to fall
through to, so the first refusal is the last word.

Two facts make it pure waste. The allowance it could not read was 2^256-1 --
the router was ALREADY approved, so the swap needed no approval at all. And
probing all seven configured Base endpoints at the time of the failure, three
answered that identical call in ~0.5s (mainnet.base.org,
base-rpc.publicnode.com, base.publicnode.com) while four are permanently
broken for eth_call (meowrpc "method not supported", llamarpc 521, 1rpc plan
limit, drpc 408). A working endpoint was always one retry away.

The cause was that ``_w3`` scores an endpoint only on whether it CONNECTS.
``is_connected()`` keeps returning True for a rate-limited RPC, so the cached
client was never evicted and RpcHealthTracker never heard about a failure that
happened after the handshake.

These tests pin the call-level failover, and pin that it does not retry an
answer the chain actually gave.
"""
from __future__ import annotations

import pytest

import router_wallet
from router_wallet import UltraSwapBridge


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HTTPError(Exception):
    """Shaped like requests.HTTPError: carries .response.status_code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _FakeResponse(status_code)


class ContractLogicError(Exception):
    """Same name web3 raises for a revert -- classification is by type name."""


class _Bridge(UltraSwapBridge):
    """A bridge with the RPC plumbing under test and nothing else.

    ``UltraSwapBridge.__init__`` needs wallet secrets and live endpoints, so
    this builds only the state ``_call_with_rpc_failover`` touches.
    """

    def __init__(self, urls):
        self._urls = list(urls)
        self._rpc_clients = {}
        self._rpc_latency = {}
        self._rpc_health = router_wallet.RpcHealthTracker()
        self._handed_out = []

    def _w3(self, chain: str):
        # Stand in for the real ranker: hand out the first URL that has not
        # been demoted, which is what health ranking does after a failure.
        failed = {u for u, lat in self._rpc_latency.items() if lat == float("inf")}
        for url in self._urls:
            if url not in failed:
                self._rpc_clients[chain] = (f"w3::{url}", url)
                self._handed_out.append(url)
                return f"w3::{url}"
        raise RuntimeError("RPC not reachable for base")


BASE_URLS = ["https://mainnet.base.org", "https://base-rpc.publicnode.com",
             "https://base.publicnode.com"]


@pytest.fixture(autouse=True)
def _base_chain(monkeypatch):
    monkeypatch.setitem(router_wallet.CHAINS, "base",
                        {"id": 8453, "poa": True, "rpcs": list(BASE_URLS)})


def test_a_429_moves_to_the_next_endpoint_instead_of_failing_the_swap():
    """The exact AERO failure: first endpoint 429s, a healthy one answers."""
    bridge = _Bridge(BASE_URLS)
    calls = []

    def _read(w3):
        calls.append(w3)
        if len(calls) == 1:
            raise _HTTPError(
                "429 Client Error: Too Many Requests for url: "
                "https://mainnet.base.org/", 429)
        return 2 ** 256 - 1

    got = bridge._call_with_rpc_failover("base", _read, what="allowance(AERO)")

    assert got == 2 ** 256 - 1, "the allowance the chain actually holds"
    assert len(calls) == 2, "it must ask a second endpoint, not give up"
    assert bridge._handed_out[0] != bridge._handed_out[1], (
        "the retry must land on a DIFFERENT endpoint; retrying the same "
        "rate-limited URL is the bug wearing a loop"
    )


def test_the_rate_limited_endpoint_is_demoted_and_its_client_evicted():
    """A cached client that 429s must not be handed to the next caller."""
    bridge = _Bridge(BASE_URLS)
    seen = []

    def _read(w3):
        seen.append(w3)
        if len(seen) == 1:
            raise _HTTPError("429 Too Many Requests", 429)
        return 7

    bridge._call_with_rpc_failover("base", _read, what="allowance(AERO)")

    bad = BASE_URLS[0]
    assert bridge._rpc_latency[bad] == float("inf"), "the 429 must demote it"
    assert bridge._rpc_health.score(bad) > bridge._rpc_health.score(BASE_URLS[1]), (
        "the endpoint that refused the call must rank below one that answered"
    )


def test_a_revert_is_an_answer_and_is_never_retried_elsewhere():
    """Asking a second node about a revert gets the same revert, slower.

    Worse, retrying would hide a real contract failure behind a wall of
    endpoint churn. Only transport failures are transient.
    """
    bridge = _Bridge(BASE_URLS)
    calls = []

    def _read(w3):
        calls.append(w3)
        raise ContractLogicError("execution reverted")

    with pytest.raises(ContractLogicError):
        bridge._call_with_rpc_failover("base", _read, what="allowance(AERO)")

    assert len(calls) == 1, "a revert must be raised on the first answer"


def test_every_endpoint_failing_still_raises_the_last_error():
    """Failover adds attempts; it must not swallow a genuine outage.

    Callers already handle a raised failure. This only removes the case where
    a healthy alternative was never asked.
    """
    bridge = _Bridge(BASE_URLS)
    calls = []

    def _read(w3):
        calls.append(w3)
        raise _HTTPError("429 Too Many Requests", 429)

    with pytest.raises(_HTTPError):
        bridge._call_with_rpc_failover("base", _read, what="allowance(AERO)")

    assert len(calls) == len(BASE_URLS), (
        "every configured endpoint gets exactly one attempt"
    )


@pytest.mark.parametrize("exc, transient", [
    (_HTTPError("429 Too Many Requests", 429), True),
    (_HTTPError("503 Service Unavailable", 503), True),
    (_HTTPError("408 Request Timeout", 408), True),
    (_HTTPError("521 origin down", 521), True),
    (Exception("Read timed out"), True),
    (Exception("{'code': -32001, 'message': \"You've reached the usage limit\"}"), True),
    (Exception("{'code': -32000, 'message': 'The method eth_call is not supported.'}"), True),
    (ContractLogicError("execution reverted"), False),
])
def test_transient_classification_matches_what_base_actually_returns(exc, transient):
    """Every string here was observed from a real Base endpoint."""
    bridge = _Bridge(BASE_URLS)
    assert bridge._rpc_error_is_transient(exc) is transient
