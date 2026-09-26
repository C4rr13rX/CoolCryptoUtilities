"""A symbol became live-tradeable by having no price feed. Pin all three states.

``trading.strategies.ledger._live_tradeable`` decides which ghost round trips
count toward a licence to spend real money, and until commit 81c6e67 it was an
AND of TWO fail-open cache lookups: ``services.symbol_edge_gate`` and
``services.stop_survivability_gate``. Both answer "allowed" for a symbol they
have never heard of, and a symbol whose price feed DIES stops being heard of --
so a dead feed was a route INTO the spendable population. Measured over all
1230 ``ghost_entry`` fills (Echo, pass 704), 66.0% of the 754-row tradeable book
was dead-feed across 34 of 57 symbols.

THE REASON THIS FILE EXISTS IS THE THIRD STATE, NOT THE SECOND. The obvious fix
-- "no recent tick means untradeable" -- strips EVERY symbol during the
whole-process stalls this build has (``trading_ops`` and ``market_stream`` both
stopped at 05:05:20 on 2026-09-24, 42 minutes after a restart; 97 min/day of
dark ``market_stream`` while ``trading_ops`` writes normally). Across all 490
symbols that ever ticked, p50 last-tick age is 12.9 days, so a fail-CLOSED on
absence refuses nearly the whole universe. ``_feed_state`` therefore asks
whether the tape is alive AT ALL before convicting one symbol, and the outage
answer is ABSTAIN, not refuse.

Every test here builds its own tiny ``market_stream`` in a temp file and points
``TRADING_DB_PATH`` at it, so nothing reads the 39GB production store and
nothing depends on today's feed.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import trading.strategies.ledger as ledger


def _make_store(path: Path, ticks) -> None:
    """A ``market_stream`` with just enough columns for the freshness query."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE market_stream (id INTEGER PRIMARY KEY, ts REAL, "
            "chain TEXT, symbol TEXT, price REAL, volume REAL, raw TEXT)"
        )
        conn.executemany(
            "INSERT INTO market_stream (ts, chain, symbol, price, volume, raw) "
            "VALUES (?, 'base', ?, 1.0, 0.0, '')",
            [(ts, sym) for sym, ts in ticks],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def feed(tmp_path, monkeypatch):
    """Point the ledger's freshness arm at a store this test owns.

    The module caches for ``LEDGER_FEED_CACHE_SEC`` and the cache is global, so
    it is cleared around every case -- a stale entry from a sibling test would
    make this file pass for the wrong reason.
    """
    def build(ticks):
        db = tmp_path / f"feed_{len(ledger._feed_symbol_cache)}_{time.time_ns()}.db"
        _make_store(db, ticks)
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        ledger._feed_symbol_cache.clear()
        ledger._feed_global_newest = 0.0
        ledger._feed_checked_at = 0.0
        return db

    yield build
    ledger._feed_symbol_cache.clear()
    ledger._feed_global_newest = 0.0
    ledger._feed_checked_at = 0.0


def _bound() -> float:
    from services.frozen_tape_guard import MAX_TICK_AGE_SEC

    return float(MAX_TICK_AGE_SEC)


def test_the_bound_is_not_a_third_clock(feed):
    """The freshness arm reuses a constant that already exists, at 86400s.

    This repo has already paid for three incompatible WETH tolls and two
    disagreeing evidence clocks. The arm imports
    ``frozen_tape_guard.MAX_TICK_AGE_SEC`` -- which already bounds exactly this
    quantity, the age of a symbol's newest ``market_stream`` tick -- rather than
    naming a number. 86400s is also ``GHOST_MAX_STALE_SEC``'s default at
    trading/pipeline.py:4938, so the two clocks on this path agree.
    """
    assert _bound() == pytest.approx(86400.0)


def test_a_live_symbol_on_a_live_tape_is_live(feed):
    now = time.time()
    feed([("AERO-USDC", now - 30.0), ("WETH-USDC", now - 60.0)])
    assert ledger._feed_state("AERO-USDC", now=now) == "live"
    assert ledger._feed_state("WETH-USDC", now=now) == "live"


def test_a_dark_symbol_on_a_live_tape_is_dark(feed):
    """The defect itself: BSTONK-USDC, 17 days silent while the tape runs.

    AERO is in the same store one minute old, so the tape is demonstrably
    alive and the verdict on BSTONK is a verdict, not an outage.
    """
    now = time.time()
    feed([("AERO-USDC", now - 60.0), ("BSTONK-USDC", now - 17.5 * 86400)])
    assert ledger._feed_state("AERO-USDC", now=now) == "live"
    assert ledger._feed_state("BSTONK-USDC", now=now) == "dark"


def test_a_symbol_that_never_ticked_is_dark_not_unknown(feed):
    """``refusal_reason('TOTALLY-FAKE-USDC')`` returns None -- absence is not a
    verdict in either gate, which is how absence became a licence."""
    now = time.time()
    feed([("AERO-USDC", now - 60.0)])
    assert ledger._feed_state("TOTALLY-FAKE-USDC", now=now) == "dark"


def test_a_whole_tape_outage_abstains_and_does_not_convict_anybody(feed):
    """THE CRITERION MOST LIKELY TO BE SILENTLY BROKEN BY A LATER REFACTOR.

    Every symbol is stale, which is the signature of a stopped feed rather than
    of 553 dead symbols. The arm must return ``unknown`` for ALL of them --
    including the one that would be the freshest -- so that a 97-minute feed
    gap cannot delete the whole evidence population.
    """
    now = time.time()
    stale = now - 9.0 * 86400
    feed([("AERO-USDC", stale), ("WETH-USDC", stale - 3600), ("BSTONK-USDC", stale - 86400)])
    for sym in ("AERO-USDC", "WETH-USDC", "BSTONK-USDC"):
        assert ledger._feed_state(sym, now=now) == "unknown", sym


def test_an_empty_tape_abstains(feed):
    """An empty ``market_stream`` is an unbuilt store, never 553 dead symbols.

    This is the state the test suite itself runs in since conftest redirects
    ``TRADING_DB_PATH`` (commit 7e14559), so this case also documents why the
    freshness arm is inert under pytest.
    """
    now = time.time()
    feed([])
    assert ledger._feed_state("AERO-USDC", now=now) == "unknown"


def test_a_missing_store_abstains(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "does-not-exist.db"))
    ledger._feed_symbol_cache.clear()
    ledger._feed_global_newest = 0.0
    ledger._feed_checked_at = 0.0
    try:
        assert ledger._feed_state("AERO-USDC") == "unknown"
    finally:
        ledger._feed_symbol_cache.clear()
        ledger._feed_global_newest = 0.0
        ledger._feed_checked_at = 0.0


def test_the_arm_only_ever_tightens(feed, monkeypatch):
    """A dark symbol is refused, and an abstain leaves the old answer alone.

    The two gates are stubbed to ALLOW, so any difference here is the new arm
    and nothing else. This is the direction check the item demands: the fix
    never widens a ban set and never recovers a refused row.
    """
    monkeypatch.setattr(
        "trading.pipeline.stop_is_unenforceable", lambda *_a, **_k: False, raising=False
    )
    monkeypatch.setattr(
        "services.symbol_edge_gate.refusal_reason", lambda *_a, **_k: None, raising=False
    )
    now = time.time()

    feed([("AERO-USDC", now - 60.0), ("BSTONK-USDC", now - 17.5 * 86400)])
    assert ledger._live_tradeable("AERO-USDC") is True
    assert ledger._live_tradeable("BSTONK-USDC") is False

    # Same two symbols, tape stopped: the arm abstains and both revert to the
    # gates' own verdict, which is what kept the outage from stripping the book.
    feed([("AERO-USDC", now - 9.0 * 86400), ("BSTONK-USDC", now - 17.5 * 86400)])
    assert ledger._live_tradeable("AERO-USDC") is True
    assert ledger._live_tradeable("BSTONK-USDC") is True
