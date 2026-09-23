"""fb660668 -- an UNPRICED symbol must be refused by a STATED default.

THE DEFECT. `trading/selector.py`'s admission path asked a candidate for USD
accounting, a volume score, a live price, a streaming feed and OHLCV, and never
once asked whether the symbol could be COST-judged. A symbol whose base token
has no address in `data/token_addresses.json` cannot be quoted, so no cost
census can price its round trip -- and "no cost" fell through as "cheap
enough". BASECAT-USDC is what that cost: 37 closed round trips for -1.9138
before the OUTCOME gate (`services.symbol_edge_gate`) had enough losses to
speak.

MEASURED PASS 677 over all 265 closed rows in `trade_outcomes`: priceable 25
symbols / 157 rows / net +4.3848; unpriced 33 symbols / 108 rows / net -2.2674.

WHY THIS FAILS AGAINST THE PRE-FIX BEHAVIOUR. Before the fix there is no
`_unpriced_symbol_refused`, no `UNPRICED_SYMBOL_ADMISSION` and no refusal, so
`test_the_stated_default_exists_and_is_refuse` fails at import/attribute and
`test_an_unpriced_symbol_is_refused` fails on the fall-through returning False
(admitted). The carve-out tests are what stop the fix from being a universe
deletion.
"""

from __future__ import annotations

import pytest

from trading import selector


def test_the_stated_default_exists_and_is_refuse():
    """A reader must be able to find the line that decides it."""
    assert selector.UNPRICED_SYMBOL_ADMISSION == "refuse"
    assert selector.UNPRICED_SYMBOL_ADDRESS_BOOK.name == "token_addresses.json"


def test_an_unpriced_symbol_is_refused(monkeypatch):
    """The defect itself: a base token with no address is REFUSED, not admitted."""
    monkeypatch.setattr(
        selector, "_address_book_tokens", lambda chain: frozenset({"WETH", "USDC"})
    )
    # BASECAT is one of the nine the 20260917 cost census could not price.
    assert selector._is_priceable("BASECAT-USDC", "base") is False
    assert selector._unpriced_symbol_refused("BASECAT-USDC", "base") is True


def test_a_priced_symbol_is_still_admitted(monkeypatch):
    monkeypatch.setattr(
        selector, "_address_book_tokens", lambda chain: frozenset({"WETH", "USDC"})
    )
    assert selector._unpriced_symbol_refused("WETH-USDC", "base") is False


def test_the_synonym_carve_out_keeps_our_deepest_pair(monkeypatch):
    """The book holds WETH, not ETH. A naive test refuses ETH-USDC."""
    monkeypatch.setattr(
        selector, "_address_book_tokens", lambda chain: frozenset({"WETH", "USDC"})
    )
    assert selector._unpriced_symbol_refused("ETH-USDC", "base") is False


def test_an_unreadable_address_book_fails_open(monkeypatch):
    """A selector that empties the universe because a file moved is worse."""
    monkeypatch.setattr(selector, "_address_book_tokens", lambda chain: frozenset())
    assert selector._unpriced_symbol_refused("BASECAT-USDC", "base") is False


def test_the_escape_hatch_restores_the_old_behaviour(monkeypatch):
    monkeypatch.setattr(
        selector, "_address_book_tokens", lambda chain: frozenset({"WETH", "USDC"})
    )
    monkeypatch.setattr(selector, "UNPRICED_SYMBOL_ADMISSION", "allow")
    assert selector._unpriced_symbol_refused("BASECAT-USDC", "base") is False


def test_the_helper_carries_the_held_exemption():
    """A held position refused a bot is a position nothing can sell."""
    import inspect

    source = inspect.getsource(selector.select_pairs)
    assert "held: bool = False" in source
    assert "if not held and _unpriced_symbol_refused(" in source
    # And the bulk scan, which never calls the helper, is gated too.
    assert source.count("_unpriced_symbol_refused(") >= 2


def test_the_nine_unpriceable_symbols_are_refused_against_the_real_book():
    """Against the REAL data/token_addresses.json, not a fixture."""
    refused = [
        sym
        for sym in (
            "BASECAT-USDC",
            "CBXRP-USDC",
            "CBADA-USDC",
            "OMARCHY-USDC",
            "CBZEC-USDC",
            "PUMP-USDC",
            "CBMEGA-USDC",
            "MOG-USDC",
        )
        if selector._unpriced_symbol_refused(sym, "base")
    ]
    if not selector._address_book_tokens("base"):
        pytest.skip("address book unreadable here; the fail-open path is tested above")
    assert len(refused) == 8, f"expected all 8 refused, got {refused}"
    # BSTONK is the ninth of the census's nine and it IS in the book today --
    # "unquotable by the census" and "absent from the address book" are not the
    # same set, and this asserts the difference rather than assuming it.
    assert selector._unpriced_symbol_refused("BSTONK-USDC", "base") is False
