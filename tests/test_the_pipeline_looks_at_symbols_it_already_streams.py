"""``services/streamed_symbol_candidates.py`` proposes symbols we already trust.

THE STARVATION IT FIXES, measured 2026-09-06 an hour after production came
back up: **79% of ghost candidates were offered on symbols the gates refuse**,
and eleven of the thirteen symbols passing every gate were never offered a
candidate at all. The pipeline managed two entries in an hour while the feed
carried 774 ticks per ten minutes across 25 symbols.

The cause is structural, not a bug: ``select_candidates`` builds its list from
DexScreener and Gecko NEW POOLS, so a symbol we have streamed for a week --
whose edge, motion and stop-survivability we have already measured -- can
never become a candidate, because it is not new.

This module proposes; the gates dispose. Every test here that matters is
about keeping that boundary: a proposer that quietly became a decider would
be a fourth gate nobody reviewed.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_feed(path: Path, series: dict, *, spacing: float = 30.0) -> None:
    """A market_stream table shaped like the real one.

    Timestamps are recent: the module only reads the last LOOKBACK_SEC, so a
    fixture stamped 1970 produces an empty read and the selector correctly
    returns nothing -- which looks exactly like a broken selector.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE market_stream ("
        " id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, price REAL)"
    )
    base = time.time() - 86400
    for symbol, prices in series.items():
        for index, price in enumerate(prices):
            conn.execute(
                "INSERT INTO market_stream (ts, symbol, price) VALUES (?,?,?)",
                (base + index * spacing, symbol, price),
            )
    conn.commit()
    conn.close()


@pytest.fixture()
def selector(tmp_path):
    """The selector pointed at a feed we control, cache cleared."""
    import services.streamed_symbol_candidates as module
    importlib.reload(module)
    db_path = tmp_path / "feed.db"

    def _load(series, **overrides):
        if db_path.exists():
            db_path.unlink()
        _make_feed(db_path, series, spacing=overrides.get("spacing", 30.0))
        module.DB_PATH = db_path
        module.MIN_TICKS = overrides.get("min_ticks", 10)
        module.MIN_WINDOWS = overrides.get("min_windows", 2)
        module.MIN_CLEAR_RATE = overrides.get("min_rate", 0.03)
        module.FRESH_SEC = overrides.get("fresh_sec", 10 * 86400)
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _mover(n: int = 60, start: float = 100.0):
    """A series that repeatedly rises well past a 0.65% round trip."""
    prices = []
    for index in range(n):
        # Sawtooth: +3% over 10 ticks, then reset.
        prices.append(start * (1.0 + 0.003 * (index % 10)))
    return prices


def _flat(n: int = 60, start: float = 100.0):
    """A series that never moves enough to pay for a trade."""
    return [start * (1.0 + 0.0001 * ((index % 3) - 1)) for index in range(n)]


def test_it_offers_a_symbol_that_clears_its_round_trip(selector):
    module = selector({"MOVER-USDC": _mover()})
    symbols = [c.symbol for c in module.streamed_candidates()]
    assert "MOVER-USDC" in symbols


def test_it_does_not_offer_a_symbol_that_cannot_pay(selector):
    """THE HALF THAT MATTERS.

    Offering a symbol that cannot clear its cost wastes a slot in the offer
    list, which is precisely the starvation this module exists to fix. A
    proposer that proposes everything has proposed nothing.
    """
    module = selector({"FLAT-USDC": _flat()})
    assert module.streamed_candidates() == []


def test_both_kinds_present_only_the_mover_is_offered(selector):
    module = selector({"MOVER-USDC": _mover(), "FLAT-USDC": _flat()})
    symbols = [c.symbol for c in module.streamed_candidates()]
    assert symbols == ["MOVER-USDC"]


def test_candidates_are_ranked_by_how_often_they_clear_cost(selector):
    """The caller takes the top N, so the ordering IS the product."""
    module = selector({
        "OFTEN-USDC": _mover(),
        "RARELY-USDC": _flat(n=60)[:50] + [100.0, 104.0] * 5,
    })
    rates = [c.clear_rate for c in module.streamed_candidates()]
    assert rates == sorted(rates, reverse=True), (
        "a lower-yielding symbol must never be offered ahead of a better one")


def test_a_stale_symbol_is_not_offered(selector):
    """A symbol that stopped ticking cannot be entered OR exited.

    Offering one would reproduce the dark-feed immortal position: an entry
    that no exit rule can ever reach, holding its slot forever.
    """
    module = selector({"MOVER-USDC": _mover()}, fresh_sec=60)
    # The fixture's newest tick is ~23h old, well past a 60s freshness bar.
    assert module.streamed_candidates() == []


def test_too_few_windows_is_not_a_rate(selector):
    """Three windows that happened to clear is an anecdote, not a rate."""
    module = selector({"THIN-USDC": _mover(n=20)}, min_windows=50)
    assert module.streamed_candidates() == []


def test_the_high_is_measured_from_the_window_open(selector):
    """Not from the window's low.

    An entry happens at the price available when the decision is made. A move
    measured from a low nobody could have bought at is selection bias, and it
    would make a symbol that fell then recovered look like a winner.
    """
    # Falls 5%, then recovers exactly to the open: profitable from the LOW,
    # break-even from the OPEN. Must NOT be offered.
    series = []
    for _ in range(6):
        series.extend([100.0, 97.0, 95.0, 97.0, 100.0])
    module = selector({"VSHAPE-USDC": series})
    assert module.streamed_candidates() == [], (
        "a round trip back to the open is not a profit; measuring from the "
        "low would manufacture one")


def test_it_fails_empty_when_the_feed_cannot_be_read(tmp_path):
    """A candidate source that raises would stop the trading cycle.

    Having no suggestions is a normal state -- a quiet market, a cold feed --
    not an error.
    """
    import services.streamed_symbol_candidates as module
    importlib.reload(module)
    module.DB_PATH = tmp_path / "does-not-exist.db"
    module._cache.clear()
    module._cache_built_at = 0.0
    assert module.streamed_candidates() == []
    assert module.as_dicts() == []


def test_it_proposes_but_never_decides(selector):
    """The boundary this module must not cross.

    Nothing here consults or bypasses a gate. Every symbol it returns still
    passes through symbol_edge_gate, symbol_motion_gate,
    stop_survivability_gate and strategy_edge_gate. Verified by source: a
    proposer that imported a gate would be deciding, and a proposer that
    imported an EXECUTION path would be trading.
    """
    import ast

    source = (ROOT / "services" / "streamed_symbol_candidates.py").read_text(
        encoding="utf-8")
    # Parsed, not grepped. The module docstring NAMES the gates it defers to
    # -- that is documentation, and a substring check cannot tell it apart
    # from a call. Only a real import can make this module decide anything.
    imported: set = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for forbidden in ("services.symbol_edge_gate",
                      "services.symbol_motion_gate",
                      "services.stop_survivability_gate",
                      "services.strategy_edge_gate",
                      "services.swap_service",
                      "db"):
        assert forbidden not in imported, (
            f"{forbidden} imported by a candidate PROPOSER means it is "
            f"deciding or trading, not proposing")


def test_the_dict_shape_carries_its_evidence(selector):
    """A candidate without its rationale is unreadable in a log six weeks on."""
    module = selector({"MOVER-USDC": _mover()})
    rows = module.as_dicts()
    assert rows
    row = rows[0]
    for key in ("symbol", "clear_rate", "windows", "ticks", "rationale", "source"):
        assert key in row
    assert "cleared" in row["rationale"]
