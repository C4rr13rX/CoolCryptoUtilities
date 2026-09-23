"""DRB-USDC streamed ONE price 71 times over 7 days and every statistic on it was garbage.

The defect (backlog edb6da74, measured p673): ``mean|move|`` over DRB-USDC came
to EXACTLY 0.0bp because the tape was one constant, 166h stale. Pass 480 fed
that zero into h* and published 5.842 -- "unpayable at any skill" -- which was a
statement about a dead feed, and it sat priority-1 in the backlog for 193
passes. Nothing refused to compute on it.

These tests pin the three behaviours the fix has to have, and each FAILS against
the pre-fix tree, where ``services/frozen_tape_guard`` did not exist at all and
``symbol_motion_gate`` scored a frozen symbol as merely quiet and ALLOWED it.

No database, no clock, no live feed: every series here is built in the test.
"""

from __future__ import annotations

import pytest

from services import frozen_tape_guard as guard


NOW = 1_700_000_000.0
HOUR = 3600.0


def _series(prices, *, newest_age_h=0.5, spacing_h=1.0, now=NOW):
    """(ts, price) with the LAST point ``newest_age_h`` hours before ``now``."""
    n = len(prices)
    end = now - newest_age_h * HOUR
    return [(end - (n - 1 - i) * spacing_h * HOUR, p) for i, p in enumerate(prices)]


def test_a_constant_tape_is_refused_even_when_it_is_still_publishing():
    """DRB's actual defect, minus the staleness: 71 ticks of one price.

    This is the case a staleness-only guard misses -- Kite p673 counted 18 live
    symbols publishing a dead price. The feed is 30 minutes old and useless.
    """
    frozen = _series([0.0001663] * 71, newest_age_h=0.5, now=NOW)
    reason = guard.verdict(frozen, symbol="DRB-USDC", now=NOW)
    assert reason is not None, "a constant tape must be refused"
    assert "FROZEN" in reason
    assert "1 distinct price" in reason


def test_a_stale_tape_is_refused_even_though_its_prices_move():
    """The other half: prices that vary, from a feed that stopped days ago.

    Variance cannot catch this one -- the series is perfectly well-behaved. Only
    the age of the newest tick says the number is a memory.
    """
    stale = _series([1.0, 1.01, 0.99, 1.02, 1.03, 0.98, 1.00, 1.04],
                    newest_age_h=166.1, now=NOW)
    reason = guard.verdict(stale, symbol="TOLLY-USDC", now=NOW)
    assert reason is not None, "a tape 166h old must be refused"
    assert "STALE" in reason
    assert "166.1h" in reason


def test_a_live_moving_tape_is_allowed():
    """The gate must narrow the universe, never empty it."""
    healthy = _series([1.0, 1.01, 0.99, 1.02, 1.03, 0.98, 1.00, 1.04, 1.05],
                      newest_age_h=0.2, now=NOW)
    assert guard.verdict(healthy, symbol="WETHX-USDC", now=NOW) is None


def test_a_pegged_asset_is_constant_by_design_and_is_not_called_frozen():
    """Criterion 4: refusing USDC-style pegs as 'frozen' would be a new defect.

    A peg is separated by a LIST LOOKUP, never by a statistic -- a peg and a
    corpse are both flat, and no amount of variance maths tells them apart.
    """
    peg = _series([1.0] * 200, newest_age_h=0.1, now=NOW)
    assert guard.is_pegged("USDT-USDC") is True
    assert guard.verdict(peg, symbol="USDT-USDC", now=NOW) is None
    # ...and the identical series on a non-peg IS refused. Same numbers, same
    # variance, opposite verdict: the discriminator is membership.
    assert guard.verdict(peg, symbol="DRB-USDC", now=NOW) is not None


def test_a_pegged_asset_whose_feed_stopped_is_still_refused():
    """'Not moving' is fine for a peg. 'Not updating' is never fine.

    A stablecoin whose feed died 30 hours ago is not telling you it is still
    worth a dollar -- it is telling you nothing, and pricing a leg off it is
    pricing off a memory. Staleness is checked BEFORE the peg exemption.
    """
    dead_peg = _series([1.0] * 200, newest_age_h=30.0, now=NOW)
    reason = guard.verdict(dead_peg, symbol="USDT-USDC", now=NOW)
    assert reason is not None, "a peg exemption must not survive a dead feed"
    assert "STALE" in reason


def test_a_newly_listed_symbol_with_three_ticks_is_not_called_frozen():
    """Too few ticks is not frozen.

    A token listed an hour ago has one distinct price trivially. Refusing it
    would ban exactly the new movers this pipeline exists to catch.
    """
    newborn = _series([0.004] * 3, newest_age_h=0.1, spacing_h=0.2, now=NOW)
    assert guard.verdict(newborn, symbol="NEWTOKEN-USDC", now=NOW) is None


def test_an_empty_feed_is_allowed_because_no_evidence_is_not_evidence_of_death():
    """Fails open, matching symbol_motion_gate. A dark pipeline is worse."""
    assert guard.verdict([], symbol="ANY-USDC", now=NOW) is None
    assert guard.verdict([(NOW, 0.0)], symbol="ANY-USDC", now=NOW) is None


@pytest.mark.parametrize("symbol", ["USDC-USDC", "DAI-USDC", "USDBC-USDC", "WETH-USDC"])
def test_the_never_ban_list_is_shared_with_the_motion_gate(symbol):
    """The two gates must agree on what a stable leg is, or they fight."""
    from services import symbol_motion_gate

    assert guard.PEG_ASSETS == symbol_motion_gate.NEVER_BAN
    assert guard.is_pegged(symbol) is True


def test_the_motion_gate_refuses_a_frozen_symbol_it_used_to_score_as_merely_quiet():
    """The integration half, and the one that fails hardest pre-fix.

    ``symbol_motion_gate`` judges a symbol on its PAY RATE, and a frozen tape
    has a pay rate of exactly 0.0 -- but it also has fewer than MIN_WINDOWS
    windows, so the gate's own "not enough feed to judge" rule ALLOWED it. A
    constant series read as a quiet symbol. It no longer does.
    """
    from services import symbol_motion_gate as gate

    frozen = _series([0.0001663] * 71, newest_age_h=0.5, spacing_h=2.0, now=NOW)
    healthy = _series([1.0 + 0.05 * (i % 3) for i in range(71)],
                      newest_age_h=0.5, spacing_h=2.0, now=NOW)

    # Pre-fix, both of these were scored on pay rate alone and both were
    # allowed for want of windows.
    rate, windows = gate.pay_rate(frozen, window_sec=900.0, cost=0.0065)
    assert rate == 0.0
    assert windows < gate.MIN_WINDOWS

    assert guard.verdict(frozen, symbol="DRB-USDC", now=NOW) is not None
    assert guard.verdict(healthy, symbol="AERO-USDC", now=NOW) is None

    original = gate._load_feed
    gate._load_feed = lambda now: {"DRB-USDC": frozen, "AERO-USDC": healthy}
    try:
        gate._cache.clear()
        gate._rebuild(NOW)
        reason = gate._cache.get("DRB-USDC")
        assert reason is not None, "the motion gate must now refuse a frozen tape"
        assert "FROZEN" in reason[1]
        assert gate._cache.get("AERO-USDC") is None, "and must not refuse a live one"
    finally:
        gate._load_feed = original
        gate._cache.clear()
        gate._cache_built_at = 0.0
