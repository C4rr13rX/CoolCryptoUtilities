"""``stop_survivability_gate`` must not score a multi-day move as one tick.

THE FAILURE, measured 2026-09-07. ``entry-refused-stop-survivability`` was
100% of refusals (7-8/h) while the aggregate live gate was wide open
(``scripts/live_gate_map.py``: live_mode ready, block_reason none, $6.00
recommended against $18.19 deployable). The gate refused all 46 symbols it had
enough feed to judge; the other 163 it abstained on for thinness. Nothing could
enter.

``_tick_jumps`` selected ``price`` and never read ``ts``, so consecutive ROWS
were consecutive TICKS however far apart they were in time. On a feed running
at 8-19 ticks/10m that has gone dark for hours -- our own outages, most since
fixed -- the gaps reach 31 hours:

    symbol          max gap between stored rows    p99 "single-tick" jump
    AAVE-USDC          111156s  (30.9 h)                  4.81%
    VIRTUAL-USDC       115584s  (32.1 h)                  4.50%
    LFG-USDC           183319s  (50.9 h)                 44.21%

Restricted to rows actually adjacent in time (<=120s), the same symbols read
0.25%, 0.52% and 2.68% against a 4.00% ceiling. The gate was charging a 2%
stop a multi-day return and concluding the stop was decoration.

This is the conflation trading/pipeline.py:4988 records against ``sparse`` --
the AGE of a measurement scored as a fault in the thing measured. A market that
gaps and a feed that went down last Tuesday are different problems.

These tests carry the asymmetric burden the rest of this gate's tests carry:
proving the fix reopens the lane matters less than proving it still slams shut
on a symbol whose price really does jump between two ticks a second apart.
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


def _make_timed_feed(path: Path, series: dict) -> None:
    """A market_stream table where the fixture chooses the TIMESTAMPS.

    ``series`` maps symbol -> list of (offset_sec, price), oldest first, with
    offsets measured back from now. The existing fixture in
    test_a_stop_must_bind_on_its_own_feed.py stamps rows one second apart,
    which is exactly the case this bug could not distinguish -- so the gap has
    to be expressible here or the regression cannot be written down.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE market_stream ("
        " id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, price REAL)"
    )
    now = time.time()
    for symbol, points in series.items():
        for offset, price in points:
            conn.execute(
                "INSERT INTO market_stream (ts, symbol, price) VALUES (?,?,?)",
                (now - offset, symbol, price),
            )
    conn.commit()
    conn.close()


@pytest.fixture()
def gate(tmp_path):
    """The gate pointed at a feed we control, cache cleared."""
    import services.stop_survivability_gate as module
    importlib.reload(module)
    db_path = tmp_path / "feed.db"

    def _load(series, **overrides):
        if db_path.exists():
            db_path.unlink()
        _make_timed_feed(db_path, series)
        module.DB_PATH = db_path
        module.MIN_TICKS = overrides.get("min_ticks", 10)
        module.MIN_BREACHES = overrides.get("min_breaches", 2)
        module.STOP_PCT = overrides.get("stop_pct", 0.02)
        module.MAX_JUMP_RATIO = overrides.get("max_ratio", 2.0)
        module.MAX_TICK_GAP_SEC = overrides.get("max_gap", 120.0)
        module.NEVER_BAN = overrides.get("never_ban", set())
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _calm_points(n: int, start_offset: float, step: float, start: float = 100.0):
    """``n`` ticks ``step`` seconds apart, moving ~0.1% -- a 2% stop holds."""
    return [
        (start_offset - i * step, start * (1.0 + 0.001 * ((i % 5) - 2)))
        for i in range(n)
    ]


def test_a_jump_across_a_dark_feed_does_not_ban_the_symbol(gate):
    """AAVE's shape: calm while the feed runs, one big step across the outage.

    40 calm ticks a second apart, the feed goes dark for 31 hours, then 40 more
    calm ticks at a price 40% higher. The only pair that breaches the 4.00%
    ceiling is the one spanning the outage, and it is not a tick -- no stop was
    ever asked to hold across it, because there was nothing to evaluate it on.

    Against the old behaviour this p99 is the 40% step and the symbol is banned.
    """
    dark_gap = 31 * 3600.0
    before = _calm_points(40, start_offset=dark_gap + 200.0, step=1.0, start=100.0)
    after = _calm_points(40, start_offset=120.0, step=1.0, start=140.0)
    module = gate({"AAVE-USDC": before + after})

    assert module.refusal_reason("AAVE-USDC") is None, (
        "a 40% move across a 31-hour hole in the feed is not a single-tick jump"
    )


def test_the_same_jump_between_two_adjacent_ticks_still_bans(gate):
    """THE LOAD-BEARING HALF -- the guard must not have been switched off.

    Identical price series to the test above. The ONLY difference is that the
    40% step happens between two ticks one second apart instead of across a
    31-hour outage. That is a real gap-through: a 2% stop is evaluated on the
    far side and the position is already 40% down.
    """
    before = _calm_points(40, start_offset=260.0, step=1.0, start=100.0)
    after = _calm_points(40, start_offset=220.0, step=1.0, start=140.0)
    module = gate({"AAVE-USDC": before + after})

    reason = module.refusal_reason("AAVE-USDC")
    assert reason, "a 40% jump between two ticks one second apart must be refused"
    assert "stop cannot bind" in reason


def test_moonbase_is_still_refused(gate):
    """The symbol this module was written about, at its measured shape.

    MOONBASE-USDC's denomination flip is a 1000x move between ticks seconds
    apart, not across an outage, so the gap cap cannot reach it. Measured
    against the live feed after this change it still reads p99 27.78% and is
    still refused, alongside BSTONK-USDC 5.03% and OMARCHY-USDC 17.55%.
    """
    points = _calm_points(60, start_offset=120.0, step=2.0, start=100.0)
    points.insert(30, (points[29][0] - 1.0, 100_000.0))  # the flip, 1s later
    module = gate({"MOONBASE-USDC": points})

    reason = module.refusal_reason("MOONBASE-USDC")
    assert reason, "a 1000x single-tick jump must still be refused"
    assert "stop cannot bind" in reason


def test_long_gap_breaches_cannot_ban_through_the_thin_feed_door(gate):
    """The filter has to apply to the BREACH COUNT, not only the percentile.

    Below MIN_TICKS the gate still bans on MIN_BREACHES directly observed jumps
    past the ceiling -- deliberately, because watching a price gap through
    twice is evidence a percentile does not need. If that counter kept scoring
    long-gap pairs, every symbol banned by the old bug would come straight back
    through the thin-feed door and the fix would be cosmetic.

    Six ticks, each separated by six hours, each 50% apart: six breaches by the
    old reckoning, zero adjacent pairs by the new one.
    """
    module = gate(
        {"SPARSE-USDC": [(6 * 3600.0 * (6 - i), 100.0 * (1.5 ** i)) for i in range(6)]},
        min_ticks=200,
        min_breaches=2,
    )

    assert module.refusal_reason("SPARSE-USDC") is None, (
        "breaches observed only across multi-hour gaps are not observed breaches"
    )


def test_the_cap_can_be_switched_off(gate):
    """STOP_SURVIVE_MAX_TICK_GAP_SEC=0 restores the unbounded behaviour.

    An escape hatch that is testable, so a future reader who believes the gaps
    are real risk can turn adjacency off with one env var and see the old
    verdicts rather than editing the percentile.
    """
    dark_gap = 31 * 3600.0
    before = _calm_points(40, start_offset=dark_gap + 200.0, step=1.0, start=100.0)
    after = _calm_points(40, start_offset=120.0, step=1.0, start=140.0)
    series = {"AAVE-USDC": before + after}

    assert gate(series, max_gap=120.0).refusal_reason("AAVE-USDC") is None
    assert gate(series, max_gap=0.0).refusal_reason("AAVE-USDC")


def test_a_calm_dense_feed_is_still_allowed(gate):
    """AERO-USDC, the symbol that actually trades. Unchanged by any of this."""
    module = gate({"AERO-USDC": _calm_points(120, start_offset=240.0, step=2.0)})
    assert module.refusal_reason("AERO-USDC") is None
