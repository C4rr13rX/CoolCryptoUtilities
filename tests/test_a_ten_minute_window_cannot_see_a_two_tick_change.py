"""The feed-rate detection floor must be computed from disjoint, full bins.

Item [cee842c9]'s last criterion asks for the feed rate before and after over
"a full 10-minute window each".  `scripts/feed_rate_power.py` answers whether
one such window can resolve the 1-2 ticks/10min the model category was
measured to cost, and the answer -- 100.8 ticks, a factor of ~50 short --
rests entirely on the spread of the binned counts.

Two ways that spread goes wrong silently, and both inflate it, which would
make the criterion look *more* impossible than it is:

  * overlapping bins, which correlate neighbours and understate the spread;
  * a short trailing bin, which holds a fraction of a window's ticks and is
    indistinguishable from a stall.

So these pin the binning itself rather than the arithmetic on top of it.  A
test that only checked `windows_needed` would pass against either bug.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from scripts.feed_rate_power import BIN_SEC, bin_counts, summarise, windows_needed


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE market_stream (ts REAL, symbol TEXT)")
    return c


def _seed(c: sqlite3.Connection, offsets: list[float], now: float) -> None:
    """Insert one row per offset, expressed as seconds BEFORE `now`."""
    c.executemany("INSERT INTO market_stream (ts, symbol) VALUES (?, 'X-USDC')",
                  [(now - o,) for o in offsets])


def test_bins_are_disjoint_so_one_tick_is_counted_once(conn):
    now = 1_000_000.0
    # One tick in each of three successive 10-minute bins.  If the bins
    # overlapped, some tick would be counted twice and the total would exceed 3.
    _seed(conn, [100.0, 700.0, 1300.0], now)
    counts, _, _ = bin_counts(conn, hours=1.0, now=now)
    assert sum(counts) == 3
    assert counts.count(1) == 3


def test_the_partial_trailing_bin_is_dropped_not_counted_short(conn):
    now = 1_000_000.0
    # 1.5 hours of lookback over a 600s bin is 9 whole bins; the leftover half
    # bin must not appear at all.  A short bin would show up as a low count and
    # widen the spread for a reason that has nothing to do with the feed.
    counts, start, end = bin_counts(conn, hours=1.5, now=now)
    assert len(counts) == 9
    assert end - start == pytest.approx(9 * BIN_SEC)


def test_rows_outside_the_lookback_are_not_counted(conn):
    now = 1_000_000.0
    # 3700s back is outside a 1h lookback (6 bins, 3600s); 100s back is inside.
    _seed(conn, [100.0, 3700.0, 99_999.0], now)
    counts, _, _ = bin_counts(conn, hours=1.0, now=now)
    assert sum(counts) == 1


def test_the_most_recent_bin_ends_at_now(conn):
    now = 1_000_000.0
    # loop_status.py frames its single window as [now-600, now).  A tick one
    # second old must land in the LAST bin, not be lost off the end.
    _seed(conn, [1.0], now)
    counts, _, end = bin_counts(conn, hours=1.0, now=now)
    assert end == now
    assert counts[-1] == 1
    assert sum(counts) == 1


def test_an_empty_ten_minute_window_is_a_zero_not_a_missing_bin(conn):
    now = 1_000_000.0
    # A stalled feed must produce a zero-count bin.  If empty bins were simply
    # absent the mean would be computed over the surviving bins only and a
    # 10-minute outage would read as a healthy feed.
    _seed(conn, [100.0, 3100.0], now)
    counts, _, _ = bin_counts(conn, hours=1.0, now=now)
    assert len(counts) == 6
    assert counts.count(0) == 4
    assert sum(counts) == 2


def test_the_detection_floor_is_the_two_sample_normal_bound():
    # The published 100.8 ticks at N=1 comes from 1.96 * sd * sqrt(2/N) at
    # sd=36.37.  Pin the identity so a "simplification" to a one-sample bound
    # (which is sqrt(2) smaller and would make the criterion look reachable)
    # cannot pass.
    sd = 36.37
    counts = [50, 50 + int(round(sd * 2))] * 8  # spread chosen only to be non-zero
    floors = summarise(counts)["min_detectable_diff_ticks"]
    got_sd = summarise(counts)["sd_ticks_per_10min"]
    assert floors["1"] == pytest.approx(1.96 * got_sd * math.sqrt(2.0), abs=0.1)
    assert floors["6"] == pytest.approx(1.96 * got_sd * math.sqrt(2.0 / 6), abs=0.1)
    # and N windows a side must shrink it, never grow it
    assert floors["36"] < floors["12"] < floors["6"] < floors["3"] < floors["1"]


def test_two_ticks_needs_thousands_of_windows_at_the_measured_spread():
    # The finding, as an assertion: at the sd actually measured over 24h, a
    # 2 tick/10min effect is not a ten-minute measurement.
    assert windows_needed(36.37, 2.0) == 2541
    # and one window a side cannot call it
    assert 1.96 * 36.37 * math.sqrt(2.0) > 2.0 * 20


def test_windows_needed_refuses_a_zero_effect():
    # Asking how many windows resolve "no change" has no answer; it must not
    # return 0 and be read as "one window is enough".
    assert windows_needed(36.37, 0.0) is None
    assert windows_needed(0.0, 2.0) is None
