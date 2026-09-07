"""A candidate slot must not be spent on a symbol the book already holds.

THE STARVATION IT FIXES, measured 2026-09-07 over 6h from ``trading_ops``.
``atf_static`` is the only executor in the ledger that can spend real money --
``atf_static_scout`` is ``graduation_blocked`` because it hardcodes
``wallet="ghost"`` and has no live branch at all -- and it was refused 57
times:

    entry-refused-duplicate           25   <- held by atf_static itself
    entry-refused-slot-busy           17   <- held by another strategy
    entry-refused-symbol-edge         11   } the 15 that _drop_already_refused
    entry-refused-symbol-motion        3   } already pre-filters
    entry-refused-stop-survivability   1   }

**42 of 57 (74%) were "the symbol is already held"**, 40 of them on AERO-USDC
alone, against a position atf_static had itself been holding for up to 3169s.
``_drop_already_refused`` was written for exactly this argument -- "a slot
spent re-proposing a standing refusal is a slot an eligible symbol did not
get" -- and covered only the rarest third of the census.

It is worth fixing because that census IS the live blocker. Re-arming needs 20
ghost round trips gathered since the demotion; atf_static had 8 in 31.3h
(0.26/h, ledger exactly matching trading_ops at 3 wins / +0.5811), so on the
measured rate the live lane cannot open for another ~47h while three quarters
of its candidate slots go to symbols it cannot enter.

Unlike a gate verdict, "already held" is read off the position book and the
entry site refuses it with CERTAINTY -- so the pre-filter changes no decision,
it only stops the pipeline paying a 0x quote probe for a refusal already on
file, on a feed that is already rate-limited.

The two carve-outs below are the whole risk of this change, and each has a
test here because each has cost this repo a live trade before:

  * a LIVE-APPROVED strategy must drop NOTHING (its entry may be live, and a
    live entry deliberately displaces a ghost position -- including its own,
    the ghost->live upgrade at bot.py:6807, which was link 6 and cost 7 of 9
    live-capable symbols on 2026-09-02);
  * a held symbol must keep its place in the STREAM watchlist, because every
    exit rule hangs off a market sample and a held position with no feed is
    never closed at all.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.atf_static_strategy as atf


STRATEGY = "atf_static"


class _Book:
    """Just enough of TradingDatabase for the book read."""

    def __init__(self, positions):
        self._positions = positions

    def load_state(self):
        return {"ghost_trading": {"positions": self._positions}}


def _position(strategy_id, *, mode="ghost", age=60.0):
    return {
        "strategy_id": strategy_id,
        "mode": mode,
        "entry_ts": time.time() - age,
    }


@pytest.fixture()
def book(monkeypatch):
    """Point the module at a position book we control, with nobody approved."""

    def _load(positions, *, approved=()):
        monkeypatch.setattr(atf, "get_db", lambda: _Book(positions))

        class _Ledger:
            def approved_ids(self):
                return list(approved)

        import trading.strategies.ledger as ledger_module

        monkeypatch.setattr(ledger_module, "StrategyLedger", _Ledger)
        return atf._certainly_refused_as_held(quote_token="USDC")

    return _load


def test_a_symbol_the_same_strategy_holds_is_skipped(book):
    """25 of the 57 refusals. The entry site calls this one duplicate."""
    assert book({"AERO-USDC": _position(STRATEGY)}) == {"AERO-USDC"}


def test_a_symbol_another_strategy_holds_is_skipped(book):
    """17 of the 57. The entry site calls this one slot-busy."""
    held = book({"AERO-USDC": _position("donchian_breakout@1d")})
    assert held == {"AERO-USDC"}


def test_a_live_approved_strategy_drops_nothing(book):
    """Link 6. A live entry DISPLACES a ghost position -- including its own.

    Pre-filtering these away would mean a strategy that finally graduates can
    never take the live entry for any symbol its own ghost lane happens to be
    holding. That is the failure this whole change exists to reach, so it must
    not be the failure the change causes.
    """
    positions = {
        "AERO-USDC": _position(STRATEGY),
        "CBZEC-USDC": _position("donchian_breakout@1d"),
    }
    assert book(positions, approved=[STRATEGY]) == set()


def test_a_position_past_max_hold_is_still_offered(book, monkeypatch):
    """The stale-slot escape hatch.

    Exits are sample-driven, so a symbol whose feed goes quiet is never closed
    and eviction is the ONLY thing that frees it. bot.py protects a position
    only while it can still exit normally, and this must not be stricter.
    """
    monkeypatch.setenv("MAX_HOLD_SECONDS", "3600")
    positions = {"AERO-USDC": _position("donchian_breakout@1d", age=7200.0)}
    assert book(positions) == set()


def test_a_position_with_no_strategy_is_still_offered(book):
    """It books as "unclassified" and counts toward no graduation, so bot.py
    does not refuse a live-attributable entry for it. Neither may we."""
    assert book({"AERO-USDC": _position("")}) == set()


def test_another_strategys_live_position_is_still_offered(book):
    """bot.py's slot-busy refusal requires the held position to be a ghost."""
    positions = {"AERO-USDC": _position("donchian_breakout@1d", mode="live")}
    assert book(positions) == set()


def test_an_unreadable_book_narrows_nothing(monkeypatch):
    """Fail OPEN. A book that cannot be read leaves the funnel as it was --
    the one thing worse than a wasted candidate slot is no candidate at all."""

    def _boom():
        raise RuntimeError("state blob is corrupt")

    monkeypatch.setattr(atf, "get_db", _boom)
    assert atf._certainly_refused_as_held(quote_token="USDC") == set()


def test_the_held_symbol_still_reaches_the_stream_watchlist():
    """A held symbol may lose its candidate slot but NEVER its price feed.

    Every exit rule -- stop, target, timed exit, confidence drop -- is reached
    only from a market sample, so a held position whose symbol drops out of
    the stream watchlist is a position nothing can ever close. That is how
    HIGH-USDC sat in the book for 11.4 days. The skip is placed after
    ``pairs.append`` for this reason, and this test is what holds it there.
    """
    source = Path(atf.__file__).read_text(encoding="utf-8")
    body = source[source.index("for idx, candidate in enumerate(candidates"):]
    body = body[: body.index("if pairs:")]

    appended = body.index("pairs.append(pair_symbol)")
    skipped = body.index("if pair_symbol in held_pairs:")
    assert appended < skipped, (
        "the held-symbol skip must come AFTER pairs.append, or a held "
        "position loses the feed that is the only thing able to close it"
    )
