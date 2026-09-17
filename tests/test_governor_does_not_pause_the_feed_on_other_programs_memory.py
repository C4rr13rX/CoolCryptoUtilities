"""The governor must not shed OUR work to relieve memory we do not hold.

MEASURED, 2026-09-16 (sprint item [2c1cde59], re-measured before writing):

    SYSTEM   74.5% used, 8.10 GB available of 31.78 GB
    OUR TREE 1,164 MB across 9 processes (pid 32632 `main.py --action start`
             is 563 MB, pid 28148 `run_waitress.py` is 481 MB)

Our share of that box is 3.6%. Other programs hold 22.5 GB. And yet:

    should_throttle(NORMAL) is `cpu >= 0.75 or mem >= 1.0 - _RESERVE_MEM`
                            = `... or mem >= 0.70`,  and mem is 0.745.

so the governor throttles us on memory owned by other programs. The cost is
not abstract: trading/ghost_limits.py:38 reads should_throttle() and HALVES
the ghost pair budget (`base_limit // 2`) when it is True, and the ghost lane
is the first failing link on the path to a paid live trade.

The bug is not unfairness, it is INEFFECTIVENESS. The governor sheds our work
on the premise that doing so relieves the pressure. Holding 3.6% of the box,
shedding us frees 3.6% and the box stays full -- we go dark and nothing is
fixed. Shedding only works when the memory is ours.

So the rule these tests pin:

  * memory pressure we did NOT cause, with absolute headroom left, must not
    pause or throttle us;
  * memory pressure we DID cause keeps the guard at FULL force;
  * a genuinely exhausted box still sheds, because below the fatal floor the
    next allocation fails no matter whose fault it is.

CPU is deliberately untouched by all of this: a busy CPU is real contention
regardless of who owns the processes.
"""

from __future__ import annotations

import pytest

from services.resource_governor import (
    Priority,
    ResourceGovernor,
    _CRITICAL_MEM,
    _EMERGENCY_MEM,
    _RESERVE_MEM,
)


def _governor(*, mem_frac: float, our_rss_mb: float, avail_mb: float,
              total_mb: float = 31.78 * 1024, cpu_pct: float = 5.0) -> ResourceGovernor:
    """A governor pinned to one measured snapshot, with no sampler running.

    ``ResourceGovernor.__init__`` does not start the sampler thread (``start()``
    does), so setting the smoothed fields directly is stable.
    """
    g = ResourceGovernor()
    with g._lock:
        g._ema_cpu = cpu_pct
        g._ema_mem = mem_frac
        g._mem = mem_frac
        g._cpu = cpu_pct
        g._our_rss_mb = our_rss_mb
        g._mem_available_mb = avail_mb
        g._total_mem_mb = total_mb
        g._other_programs_mem_mb = max(0.0, (total_mb - avail_mb) - our_rss_mb)
        g._max_rss_mb = max(2048.0, total_mb * 0.50)
    return g


# ---------------------------------------------------------------------------
# The measured case: the pressure is external and there is headroom left.
# ---------------------------------------------------------------------------

def test_todays_measured_box_does_not_throttle_us():
    """74.5% used, 8.10 GB free, we hold 1,164 MB. Nothing of ours may shed."""
    g = _governor(mem_frac=0.745, our_rss_mb=1164.0, avail_mb=8.10 * 1024)

    assert g._other_programs_mem_mb > 20_000, "sanity: other programs hold >20 GB"
    assert 0.745 >= 1.0 - _RESERVE_MEM, "sanity: this is over the old throttle line"

    assert not g.should_throttle(Priority.NORMAL)
    assert not g.should_pause(Priority.NORMAL)
    assert not g.should_pause(Priority.HIGH)
    assert not g.should_pause(Priority.CRITICAL)


def test_the_feed_survives_a_box_another_program_has_filled():
    """The item's own scenario: HIGH/CRITICAL work must not pause on it.

    A 92% box with 2.4 GB still free and 143 MB held by us is external
    pressure with headroom. Pausing the feed here frees 143 MB of a 29 GB
    problem and buys nothing but darkness.
    """
    g = _governor(mem_frac=0.92, our_rss_mb=143.0, avail_mb=2.4 * 1024)

    assert 0.92 >= _EMERGENCY_MEM, "sanity: this is over the emergency line"
    assert not g.should_pause(Priority.HIGH)
    assert not g.should_pause(Priority.CRITICAL)
    assert not g.should_throttle(Priority.NORMAL)


# ---------------------------------------------------------------------------
# The guard keeps FULL force against our own tree.
# ---------------------------------------------------------------------------

def test_our_own_tree_still_sheds_at_full_force():
    """When WE are the hog the shed works, so it must still happen."""
    total = 31.78 * 1024
    g = _governor(mem_frac=0.92, our_rss_mb=total * 0.50, avail_mb=2.4 * 1024,
                  total_mb=total)

    assert g.should_pause(Priority.HIGH), "our tree holds half the box"
    assert g.should_pause(Priority.NORMAL)
    assert g.should_throttle(Priority.NORMAL)


def test_a_genuinely_exhausted_box_still_sheds():
    """Below the fatal floor the next allocation fails whoever caused it."""
    g = _governor(mem_frac=0.99, our_rss_mb=143.0, avail_mb=80.0)

    assert g.should_pause(Priority.HIGH)
    assert g.should_throttle(Priority.NORMAL)


def test_no_sample_yet_keeps_the_old_behaviour():
    """A governor whose sampler has not run has no attribution evidence.

    ``avail_mb == 0`` means "unknown", not "exhausted", and the safe reading
    of unknown is the conservative one: behave exactly as before.
    """
    g = _governor(mem_frac=0.95, our_rss_mb=0.0, avail_mb=0.0)

    assert g.should_pause(Priority.HIGH)
    assert g.should_pause(Priority.NORMAL)


# ---------------------------------------------------------------------------
# CPU is not attributable this way and must be left alone.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("priority", [Priority.NORMAL, Priority.LOW])
def test_cpu_pressure_is_untouched_by_the_memory_discriminator(priority):
    """A busy CPU is real contention no matter who owns the processes."""
    g = _governor(mem_frac=0.20, our_rss_mb=1164.0, avail_mb=8.10 * 1024,
                  cpu_pct=99.0)

    assert g.should_pause(priority), "CPU pressure must still pause"
    assert g.should_throttle(priority)


def test_our_memory_still_pauses_normal_work_at_the_critical_line():
    """The CRITICAL_MEM path stays live for memory that is genuinely ours."""
    total = 31.78 * 1024
    g = _governor(mem_frac=_CRITICAL_MEM, our_rss_mb=total * 0.50,
                  avail_mb=4.0 * 1024, total_mb=total)

    assert g.should_pause(Priority.NORMAL)
