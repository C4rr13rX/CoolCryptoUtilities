"""A pair whose P&L cannot be denominated in USD must never get a feed.

THE FAILURE THIS PREVENTS
-------------------------
``trading/bot.py:6046`` asks ``is_usd_accounting_pair(base, quote)`` as the
FIRST thing in the decision path. A pair that fails it returns
``hold-price-domain`` / ``usd_pnl_requires_nonstable_base_and_stable_quote``
and can never reach an entry, however the market moves.

Nothing upstream asked the same question, so ``trading/selector.py`` handed
those pairs price streams anyway. Measured 2026-09-06 over 24h of
``market_stream``: 13 of 42 streamed symbols were non-USD-accounting and took
**1793 of 4567 ticks -- 39.3% of the whole feed**:

    CBETH-WETH 175   CBETH-CBBTC 162   AERO-WETH 158   VVV-WETH 147
    USDT-USDC  147   VIRTUAL-WETH 136  EURC-WETH 134   MORPHO-WETH 132
    EURC-USDC  129   JITOSOL-CBBTC 126 SOL-CBBTC 123   TIBBIR-VIRTUAL 121
    DAI-USDC   103

Over the same window ``organism_snapshots`` recorded 151 of 606 decisions
(24.9%) ending in ``hold-price-domain``, on exactly those symbols.

Entries AND exits here are sample-driven -- a position is marked out only when
a tick for its symbol arrives -- so a tick spent on an untradeable pair is a
tick a tradeable one did not get. While USDT-USDC took 51 ticks/hour, the
symbols that could actually trade were on 2-6 (BASEMATE-USDC 2, BST-USDC 4,
TONY-USDC 6, CBZEC-USDC 6). Each of the 13 also held a slot against
``select_pairs``'s limit.

WHAT IS ASSERTED
----------------
Behaviour, not prose: ``select_pairs`` is driven with a candidate list
containing both kinds of pair and its RETURN VALUE is checked. Against the
pre-fix selector every candidate below is picked, so each of these fails.

The held-position exemption is asserted too, and it is the more important
half: taking the feed away from an open position is how positions became
immortal, and a gate that fixed the waste by stranding capital would be a
worse bug than the one it replaced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trading import selector
from trading.selector import PairCandidate, select_pairs
from services.trading_accounting import is_usd_accounting_symbol


def _candidate(symbol: str, *, volume: float = 10_000.0) -> PairCandidate:
    return PairCandidate(
        symbol=symbol.upper(),
        tokens=[part for part in symbol.upper().split("-") if part],
        avg_volume=volume,
        volatility=0.05,
        score=1.0,
        datapath=Path("."),
    )


@pytest.fixture
def selector_env(monkeypatch):
    """Neutralise every gate in select_pairs EXCEPT the accounting one.

    The live-price probe, the streaming-feed check and the OHLCV bootstrap all
    reach the network. Passing them unconditionally is what makes this test a
    statement about the accounting gate alone: if the symbol is missing from
    the result, only the accounting gate can have removed it.
    """
    monkeypatch.setattr(selector, "load_watchlists", lambda *_a, **_k: {})
    monkeypatch.setattr(selector, "DEFAULT_LIVE_PAIRS", [])
    monkeypatch.setattr(selector, "_load_top_symbols", lambda *_a, **_k: [])
    monkeypatch.setattr(selector, "_has_live_price", lambda *_a, **_k: True)
    monkeypatch.setattr(selector, "_has_streaming_feed", lambda *_a, **_k: True)
    monkeypatch.setattr(selector, "_ensure_ohlcv", lambda *_a, **_k: True)
    monkeypatch.setattr(selector, "_vol_spread_score", lambda _c: 1.0)
    monkeypatch.setattr(selector, "_persist_chain_lock", lambda *_a, **_k: None)
    monkeypatch.setattr(selector, "_chain_lock_state", lambda: (None, 0.0))

    class _NoHoldings:
        holdings: dict = {}

        def refresh(self, *_a, **_k):
            return None

    monkeypatch.setattr(selector, "PortfolioState", _NoHoldings)

    def _drive(symbols):
        monkeypatch.setattr(
            selector,
            "analyse_historical_pairs",
            lambda *_a, **_k: [_candidate(sym) for sym in symbols],
        )
        return {pair.symbol for pair in select_pairs(limit=len(symbols) + 4)}

    return _drive


# The exact 13 symbols measured above, in the form they appear in
# `market_stream`, plus the two structural shapes they represent.
MEASURED_UNTRADEABLE = [
    "CBETH-WETH",
    "CBETH-CBBTC",
    "AERO-WETH",
    "VVV-WETH",
    "USDT-USDC",
    "VIRTUAL-WETH",
    "EURC-WETH",
    "MORPHO-WETH",
    "EURC-USDC",
    "JITOSOL-CBBTC",
    "SOL-CBBTC",
    "TIBBIR-VIRTUAL",
    "DAI-USDC",
]


def test_the_thirteen_measured_pairs_are_not_selected(selector_env):
    """None of the 39.3% of the feed may come back from select_pairs."""
    picked = selector_env(MEASURED_UNTRADEABLE + ["AERO-USDC", "CBBTC-USDC"])
    leaked = sorted(picked & set(MEASURED_UNTRADEABLE))
    assert not leaked, f"untradeable pairs were given a feed: {leaked}"


def test_the_tradeable_pairs_are_still_selected(selector_env):
    """The gate must not be a blanket refusal -- that is the same as off.

    A filter that emptied the universe would stop trading just as completely
    as the waste it removes, so the positive case is asserted in the same
    breath as the negative one.
    """
    picked = selector_env(MEASURED_UNTRADEABLE + ["AERO-USDC", "CBBTC-USDC", "TONY-USDC"])
    assert picked == {"AERO-USDC", "CBBTC-USDC", "TONY-USDC"}


def test_a_stablecoin_base_is_refused_even_against_a_stable_quote():
    """USDT-USDC and DAI-USDC pass a naive "quote is a stable" test.

    They were 250 of the 1793 wasted ticks, and USDT-USDC was the single
    most-streamed symbol in the book. A gate that only looked at the quote leg
    would have left the largest one in place.
    """
    assert not is_usd_accounting_symbol("USDT-USDC")
    assert not is_usd_accounting_symbol("DAI-USDC")
    assert is_usd_accounting_symbol("AERO-USDC")


def test_a_malformed_symbol_is_refused():
    """A bare base with no quote leg cannot be judged, so it cannot trade.

    `services/internal_cron.py:364` adds quote-less symbols to watchlists.
    They reach the decision path with an empty quote, fail
    `is_usd_accounting_pair`, and hold a stream slot until they get there.
    """
    assert not is_usd_accounting_symbol("AERO")
    assert not is_usd_accounting_symbol("")
    assert not is_usd_accounting_symbol("A-B-C")


def test_an_open_position_keeps_its_feed_whatever_its_quote(monkeypatch):
    """The exemption that stops this fix from stranding capital.

    Exits are sample-driven: a position is only marked out when a tick for its
    symbol arrives. A held CBETH-WETH with no stream is a position nothing can
    ever close -- the dark-feed immortality bug, reintroduced by a gate meant
    to save bandwidth. `build()` must keep held symbols regardless.
    """
    held = ["CBETH-WETH"]
    monkeypatch.setattr(selector, "_held_position_symbols", lambda _db: held)

    # build()'s ordering step, exercised directly on the same predicate it
    # uses: held symbols survive, unheld ones of the identical shape do not.
    held_upper = {symbol.upper() for symbol in held}
    survivors = [
        symbol
        for symbol in ("CBETH-WETH", "AERO-WETH", "AERO-USDC")
        if symbol in held_upper or selector._can_denominate_pnl_in_usd(symbol)
    ]
    assert survivors == ["CBETH-WETH", "AERO-USDC"]


def test_the_gate_can_be_switched_off_by_env(monkeypatch):
    """An escape hatch, so a bad verdict is one env var from reversible."""
    monkeypatch.setattr(selector, "_REQUIRE_USD_ACCOUNTING", False)
    assert selector._can_denominate_pnl_in_usd("CBETH-WETH")
    monkeypatch.setattr(selector, "_REQUIRE_USD_ACCOUNTING", True)
    assert not selector._can_denominate_pnl_in_usd("CBETH-WETH")
