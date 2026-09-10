"""A large move across a quiet feed is not a jump a stop failed to bind on.

THE QUESTION THIS ANSWERS, measured 2026-09-10 (item 66b46f42). A 7-day census
of ``market_stream`` counted ADJACENT-ROW jumps and reported BASECAT-USDC with a
p99 of 5.559% and 31 jumps above 5%, against the gate's 4.00% ceiling (2% stop
x 2.0) -- while ``refusal_reason('BASECAT-USDC')`` returned None. The gate was
suspected of reading too short a window.

IT IS NOT A WINDOW DISAGREEMENT. ``WINDOW_SEC`` is 604800.0 -- the gate reads
the SAME seven days. The entire disagreement is ``MAX_TICK_GAP_SEC``: the census
counted consecutive ROWS, the gate counts rows adjacent IN TIME.

Re-measured over the same 7 days, splitting each symbol's >5% jumps by the gap
they span:

    symbol         pairs   p99 by row   p99 <=120s   >5% jumps   of those <=120s
    BASECAT-USDC    1809       5.272%       2.646%          23          0
    BSTONK-USDC     1455      12.318%       4.788%          79         10
    AERO-USDC       3668       0.862%       0.426%           1          0

BASECAT's 23 large jumps span a MINIMUM of 281 seconds and a median of 2135;
inside 120 seconds its largest move in the whole week is 4.472%. AERO's single
large jump spans 22 hours. BSTONK's do not need the gap: ten of them land
between ticks as little as 17 seconds apart, which is why its capped p99 stays
at 4.788% and the gate refuses it.

The 120s measure is the right one, and the reason is not a preference. A stop is
only enforceable on a tick that ARRIVES: a 5% move accumulated over 35 minutes
of feed silence is not a move the stop failed to bind on, it is a move nobody
was there to evaluate. Charging it to the symbol is the AGE of a measurement
scored as a fault in the thing measured -- the conflation
test_a_stop_gate_cannot_call_a_31_hour_gap_one_tick.py was written about.

And the discriminator is load-bearing rather than cosmetic: it is what separates
BSTONK -- the symbol that actually booked six 17-25% gross ghost rows -- from
BASECAT, which booked one. The two feeds are indistinguishable by row-adjacent
p99 ordering alone at the ceiling; only time-adjacency splits them.

WHAT THESE TESTS ADD over the 31-hour-gap file. That file proves the mechanism
on ONE big step. These prove it survives the shape actually measured: MANY large
jumps, on a symbol dense enough to be judged on a percentile rather than on the
breach counter, where the only thing distinguishing the banned symbol from the
allowed one is which side of the gap cap its jumps fall on.
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

#: The measured median gap spanned by BASECAT's 23 large jumps. Well above the
#: 120s cap and well below the multi-hour outages the other file uses -- the
#: point being that the cap does not need an OUTAGE to bite, only a market that
#: went quiet for half an hour.
QUIET_GAP_SEC = 2135.0

#: The measured median gap between BASECAT's rows, rounded up.
DENSE_STEP_SEC = 4.0


def _make_timed_feed(path: Path, series: dict) -> None:
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
    import services.stop_survivability_gate as module
    importlib.reload(module)
    db_path = tmp_path / "feed.db"

    def _load(series, **overrides):
        if db_path.exists():
            db_path.unlink()
        _make_timed_feed(db_path, series)
        module.DB_PATH = db_path
        # Above MIN_TICKS deliberately: these symbols must be judged on the
        # percentile, not on the thin-feed breach counter, or the test would be
        # exercising a different branch than the one BASECAT lands on.
        module.MIN_TICKS = overrides.get("min_ticks", 50)
        module.MIN_BREACHES = overrides.get("min_breaches", 2)
        module.STOP_PCT = overrides.get("stop_pct", 0.02)
        module.MAX_JUMP_RATIO = overrides.get("max_ratio", 2.0)
        module.MAX_TICK_GAP_SEC = overrides.get("max_gap", 120.0)
        module.NEVER_BAN = overrides.get("never_ban", set())
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _gappy_series(segments: int = 8, per_segment: int = 40,
                  segment_step: float = 0.08, jitter: float = 0.001):
    """BASECAT's shape: dense calm trading in blocks, big steps BETWEEN blocks.

    ``segments`` blocks of ``per_segment`` ticks ``DENSE_STEP_SEC`` apart. Inside
    a block the price wanders ~0.1%, which a 2% stop holds through easily. Each
    block opens ``segment_step`` (8%) away from where the last one closed, across
    ``QUIET_GAP_SEC`` of silence.

    By ROW adjacency this feed shows ``segments - 1`` jumps of 8% -- twice the
    4.00% ceiling -- and its p99 is one of them. By TIME adjacency it shows none.
    """
    points = []
    offset = QUIET_GAP_SEC * segments + per_segment * DENSE_STEP_SEC
    price = 100.0
    for _segment in range(segments):
        for i in range(per_segment):
            points.append((offset, price * (1.0 + jitter * ((i % 5) - 2))))
            offset -= DENSE_STEP_SEC
        price *= 1.0 + segment_step
        offset -= QUIET_GAP_SEC
    return points


def test_many_jumps_across_quiet_stretches_do_not_ban_a_dense_symbol(gate):
    """BASECAT-USDC's measured shape: 23 big jumps, none of them between ticks.

    This is the case the one-step test cannot reach. The symbol is well above
    MIN_TICKS, so it is judged on a percentile; by row-adjacency SEVEN pairs sit
    at 8% and drag the p99 to twice the ceiling. Every one of them spans 35
    minutes of a feed that was not printing.

    A stop is a promise about a price you can OBSERVE crossing the level. Nothing
    crossed anything here -- there was no tick to evaluate it on.
    """
    module = gate({"BASECAT-USDC": _gappy_series()})

    assert module.refusal_reason("BASECAT-USDC") is None, (
        "large moves spanning 2135s of silence are not single-tick jumps and "
        "cannot ban a symbol whose every ADJACENT tick moves ~0.1%"
    )


def test_that_same_feed_read_by_row_adjacency_bans_it(gate):
    """THE PRE-FIX BEHAVIOUR, so the test above is proving something.

    Byte-identical price series. The only change is MAX_TICK_GAP_SEC=0, the
    documented escape hatch restoring unbounded row-adjacency -- which is exactly
    what the 7-day census that filed this item computed. It bans BASECAT.

    If this assertion ever fails, the test above has stopped discriminating and
    is passing because the gate bans nothing.
    """
    series = {"BASECAT-USDC": _gappy_series()}

    reason = gate(series, max_gap=0.0).refusal_reason("BASECAT-USDC")
    assert reason, (
        "row-adjacency must still see the 8% steps -- otherwise the test above "
        "proves nothing about which measure is doing the work"
    )
    assert "stop cannot bind" in reason


def test_a_jump_between_two_adjacent_ticks_still_bans_the_same_shaped_feed(gate):
    """BSTONK-USDC's measured shape, and the half that must not be switched off.

    The SAME gappy blocks as BASECAT -- same quiet stretches, same 8% steps
    across them -- plus what BSTONK actually has and BASECAT does not: jumps
    inside the cap. Ten of BSTONK's 79 large jumps land between ticks as little
    as 17 seconds apart, which is a real gap-through: the stop is evaluated on
    the far side with the position already 6% down.

    So the discriminator is not "gappy feeds are forgiven". It is "the gap is not
    the jump". Take the identical feed, move four of the jumps inside 20 seconds,
    and the gate must refuse.
    """
    points = _gappy_series()
    # Four 6% gap-throughs at 20s spacing -- inside the 120s cap, so these ARE
    # jumps a stop was asked to survive. Placed late in the series where the
    # timestamps are small, keeping them adjacent to their neighbours.
    tail_offset = points[-1][0]
    price = points[-1][1]
    for step in range(1, 5):
        price *= 1.06
        points.append((tail_offset - 20.0 * step, price))

    module = gate({"BSTONK-USDC": points})

    reason = module.refusal_reason("BSTONK-USDC")
    assert reason, (
        "a 6% move between two ticks 20 seconds apart is a gap-through a 2% "
        "stop cannot bind on, however quiet the rest of the feed was"
    )
    assert "stop cannot bind" in reason


def test_the_two_feeds_are_separated_only_by_the_gap_cap(gate):
    """The claim stated as one assertion: same shape, opposite verdicts.

    BASECAT and BSTONK differ by four pairs out of ~320. Row-adjacency calls both
    of them unenforceable; time-adjacency splits them, and it splits them the way
    the ghost book did -- BSTONK is the symbol that booked six 17-25% gross rows,
    BASECAT booked one.
    """
    calm = _gappy_series()
    jumpy = _gappy_series()
    tail_offset, price = jumpy[-1]
    for step in range(1, 5):
        price *= 1.06
        jumpy.append((tail_offset - 20.0 * step, price))

    assert gate({"S-USDC": calm}).refusal_reason("S-USDC") is None
    assert gate({"S-USDC": jumpy}).refusal_reason("S-USDC")

    # ...and by row-adjacency they are indistinguishable: both refused.
    assert gate({"S-USDC": calm}, max_gap=0.0).refusal_reason("S-USDC")
    assert gate({"S-USDC": jumpy}, max_gap=0.0).refusal_reason("S-USDC")
