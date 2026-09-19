"""base's route order must not advertise a venue that only exists on Arbitrum.

Measured by the operator 2026-09-18 and confirmed here in pass 479: on base,
``SushiV2`` answers ``"SushiV2 unsupported on base"`` and ``CamelotV2`` answers
``"no router configured for base"``, because ``SUSHI_V2_ROUTER`` carries only an
arbitrum key and ``CAMELOT_V2_ROUTER_BASE`` is unset. Both return that error
dict *before* constructing a web3 handle, so the wasted work is 4.5us per swap
and not, as the backlog item guessed, a timeout -- that part of the claim was
measured and killed rather than repeated.

What survives is worse than the cost: ``default_route_order()`` listed all three
venues on every chain, so a base swap printed ``routes=uniswap->camelot->sushi``
and an operator reading that log line saw two fallbacks that do not exist. This
repo has already been bitten twice by a route list with nothing behind it (see
base-has-no-dex-fallback and base-rpc-list-had-no-working-endpoint). The list
must be honest about its length, and it must say WHY a route is missing.

Against the pre-pass-479 implementation every test here fails: ``chain=`` was
not a parameter, so each call raised
``TypeError: default_route_order() got an unexpected keyword argument 'chain'``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.swap_service import default_route_order, route_availability  # noqa: E402

#: Venues whose router table in this repo carries an arbitrum entry and no base
#: entry. Named explicitly rather than derived, so that configuring one of them
#: on base has to come here and delete a line -- which is the moment somebody
#: re-reads this test.
ARBITRUM_ONLY_ON_BASE = ("sushi",)


def test_base_route_order_holds_no_arbitrum_only_venue(monkeypatch):
    """The one chain we trade must advertise only routes that can answer."""
    # Do not inherit an operator's .env: this asserts the shipped default.
    monkeypatch.delenv("CAMELOT_V2_ROUTER_BASE", raising=False)
    monkeypatch.setenv("CAMELOT_V2_ROUTER_BASE", "")

    order = default_route_order(chain="base")

    assert order == ["uniswap"], (
        f"base advertises {order}; only uniswap can answer there. "
        "A route in this list is a promise that it will be tried."
    )
    for venue in ARBITRUM_ONLY_ON_BASE:
        assert venue not in order, f"{venue} is arbitrum-only in this repo"


def test_arbitrum_keeps_all_three_routes():
    """The filter must be chain-specific, not a blanket deletion.

    Arbitrum is where camelot and sushi genuinely resolve. If this goes red the
    change removed real redundancy instead of imaginary redundancy.
    """
    assert default_route_order(chain="arbitrum") == ["uniswap", "camelot", "sushi"]


def test_no_chain_given_keeps_the_chain_agnostic_order():
    """Callers that do not name a chain keep the old answer.

    ``tests/test_swap_never_requires_0x.py`` asserts this exact list, and the
    0x-is-opt-in property it guards is unrelated to chain reachability.
    """
    assert default_route_order() == ["uniswap", "camelot", "sushi"]


def test_an_unset_router_env_is_a_named_reason_not_an_empty_result(monkeypatch):
    """CAMELOT_V2_ROUTER_BASE being unset must NAME itself.

    The acceptance criterion this defends: a route that is configured-but-empty
    has to be distinguishable from one the venue cannot serve at all, because
    the first is fixable from .env and the second needs code. Collapsing both
    into "no quote available" is what made base look redundant for months.
    """
    monkeypatch.setenv("CAMELOT_V2_ROUTER_BASE", "")

    why = route_availability("base")

    assert why["uniswap"] == "", "uniswap answers on base"
    assert why["camelot"].startswith("unconfigured:"), why["camelot"]
    assert "CAMELOT_V2_ROUTER_BASE" in why["camelot"], (
        "the reason must name the variable an operator would set"
    )
    assert why["sushi"].startswith("unsupported_on_chain:"), why["sushi"]
    # The two codes must differ -- that IS the distinction being asserted.
    assert why["camelot"].split(":")[0] != why["sushi"].split(":")[0]


def test_setting_the_camelot_router_puts_camelot_back_on_base(monkeypatch):
    """The reason is live, not a hard-coded table.

    If an operator sets the router, the route must reappear without a code
    change -- otherwise 'unconfigured' would be a lie about a closed door.
    """
    monkeypatch.setenv(
        "CAMELOT_V2_ROUTER_BASE", "0xC873FECbd354f5A56E00E710B90EF4201db2448d"
    )

    assert route_availability("base")["camelot"] == ""
    assert default_route_order(chain="base") == ["uniswap", "camelot"]


def test_an_unknown_chain_reports_every_route_as_unreachable():
    """A chain nothing is configured for must not silently return uniswap."""
    why = route_availability("fantom")
    assert why["uniswap"].startswith("unsupported_on_chain:"), why["uniswap"]
    assert default_route_order(chain="fantom") == []


@pytest.mark.parametrize("chain", ["BASE", " base ", "Base"])
def test_chain_name_is_normalised(chain):
    """Callers pass whatever the ledger stored; casing must not open a route."""
    assert default_route_order(chain=chain) == default_route_order(chain="base")
