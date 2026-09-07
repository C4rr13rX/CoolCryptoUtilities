"""A symbol the price feed cannot carry must not be offered as a candidate.

THE FAILURE THIS PREVENTS
-------------------------
Measured 2026-09-07 over the two hours to 05:10, from ``trading_ops``:

    48 ghost candidates emitted, 2 ghost entries
    VIRTUAL-USDC   17 candidates   last tick 31.0h ago
    AIXBT-USDC      4 candidates   last tick 30.9h ago

22 of the 48 (45.8%) were offered on symbols the stream had not priced in over
a day. They cannot be entered -- every entry rule reads a market sample -- but
they never reach a gate either, so they leave no refusal row. They die
silently, above the gates, having spent a candidate slot, a 0x quote probe, a
``ghost_candidate`` row and a bus action each.

The filter is a LIVENESS filter, not a ban on anything unproven. A symbol that
has never been subscribed is kept: putting it on the watchlist is how a new
pool gets its first tick, and refusing it here would switch new-pool discovery
off. Only a symbol that is already subscribed AND still silent is dropped --
it has been asked and did not answer.

And a held pair is never dropped whatever the feed says, because dropping a
candidate takes its stream watchlist entry with it, and a position with no
feed is never closed.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import pytest

from services import atf_static_strategy as atf


@dataclass
class _Cand:
    symbol: str
    address: str = "0xdead"


@pytest.fixture()
def feed(tmp_path, monkeypatch):
    """A market_stream carrying LIVE-USDC now and STALE-USDC a day ago."""
    now = time.time()
    db_path = tmp_path / "storage" / "trading_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE market_stream (ts REAL, symbol TEXT, price REAL)")
    conn.executemany(
        "INSERT INTO market_stream (ts, symbol, price) VALUES (?, ?, ?)",
        [
            (now - 30.0, "LIVE-USDC", 1.0),
            (now - 86400.0, "STALE-USDC", 1.0),
            (now - 86400.0, "HELD-USDC", 1.0),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(atf, "REPO_ROOT", tmp_path)
    return now


def _watchlist(monkeypatch, stream):
    monkeypatch.setattr(
        atf, "load_watchlists", lambda *a, **k: {"stream": list(stream), "ghost": [], "live": []}
    )


def test_a_subscribed_symbol_with_a_day_old_tick_is_dropped(feed, monkeypatch):
    """The exact production case: on the watchlist, silent for 24 hours."""
    _watchlist(monkeypatch, ["LIVE-USDC", "STALE-USDC"])
    kept = atf._drop_unstreamable(
        [_Cand("LIVE"), _Cand("STALE")], quote_token="USDC", now=feed
    )
    assert [c.symbol for c in kept] == ["LIVE"]


def test_a_symbol_never_subscribed_still_gets_its_first_look(feed, monkeypatch):
    """New-pool discovery must survive the filter.

    NEWPOOL has no tick and no watchlist entry. It has never been asked, so
    silence is not evidence about it -- being offered is how it gets asked.
    """
    _watchlist(monkeypatch, ["LIVE-USDC", "STALE-USDC"])
    kept = atf._drop_unstreamable(
        [_Cand("NEWPOOL"), _Cand("STALE")], quote_token="USDC", now=feed
    )
    assert [c.symbol for c in kept] == ["NEWPOOL"]


def test_a_held_pair_keeps_its_slot_however_stale_the_feed(feed, monkeypatch):
    """Dropping it would take the stream entry the exit path needs."""
    _watchlist(monkeypatch, ["HELD-USDC", "STALE-USDC"])
    kept = atf._drop_unstreamable(
        [_Cand("HELD"), _Cand("STALE")],
        quote_token="USDC",
        protected={"HELD-USDC"},
        now=feed,
    )
    assert [c.symbol for c in kept] == ["HELD"]


def test_an_unreadable_feed_drops_nothing(tmp_path, monkeypatch):
    """Fail OPEN. No market_stream at all is not evidence that a symbol is dead."""
    monkeypatch.setattr(atf, "REPO_ROOT", tmp_path)
    _watchlist(monkeypatch, ["STALE-USDC"])
    candidates = [_Cand("STALE")]
    assert atf._drop_unstreamable(candidates, quote_token="USDC") == candidates


def test_an_unreadable_watchlist_drops_nothing(feed, monkeypatch):
    """Without the watchlist there is no record of who was already asked."""
    def _boom(*_a, **_k):
        raise RuntimeError("watchlist store down")

    monkeypatch.setattr(atf, "load_watchlists", _boom)
    candidates = [_Cand("STALE")]
    assert atf._drop_unstreamable(candidates, quote_token="USDC", now=feed) == candidates


def test_the_filter_runs_before_the_standing_refusal_filter(monkeypatch):
    """Order matters, and the call site must not be reordered by accident.

    A symbol with no feed has no closed round trips, so ``symbol_edge_gate``
    has nothing to ban it on -- asking the gates first lets it straight
    through. The liveness question has to be asked first or it is not asked at
    all.
    """
    import inspect

    source = inspect.getsource(atf.build_static_strategy_signals)
    live = source.index("_drop_unstreamable(")
    refused = source.index("_drop_already_refused(")
    assert live < refused
