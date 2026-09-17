"""A declining strategy may name the guard that refused, and by how much.

WHY THIS EXISTS. Measured over 168h of ``trading_ops`` at pass 443: 3,644
arbitration contests recorded 101,199 strategy-skips across 72 strategies, and
100,370 of them (99.2%) were the bare literal ``"no_signal"``, which names
nothing. The 829 that said ``"min_samples 33<40"`` named the guard AND the
distance to it. Every bespoke census in this repo -- the dip census, the
momentum census, the cost-cohort split -- reconstructs by hand what the running
system already decided a hundred thousand times a week and threw away.

The seam is ``StrategyRegistry.evaluate_all``: when ``strat.evaluate()`` returns
falsy, the reason dies inside ``evaluate``. It now reads an optional
``last_decline`` off the strategy.

THE TWO THINGS THAT MUST HOLD, and they pull in opposite directions:

  1. A strategy that sets NOTHING is reported exactly as before. This is what
     keeps the change additive across 72 strategies that have not been
     instrumented -- if this fails, the population's skip labels all move at
     once and every stored comparison breaks.
  2. A strategy that DOES set one has it reported verbatim, and a reason left
     by an EARLIER tick is never reported for a later one. A stale label is
     worse than no label: it names a guard that did not decline this time.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.base import StrategyRegistry  # noqa: E402


class _Stub:
    """Minimal strategy: declines unless told to fire, names a reason if given."""

    min_samples = 0

    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id
        self.fire = False
        self.set_reason: object = None
        self.seen_on_entry: list = []

    def enabled(self) -> bool:
        return True

    def evaluate(self, state, ctx):  # noqa: ANN001
        # Record what the registry handed us BEFORE we touch it, so the test
        # can prove the previous tick's reason was cleared rather than trusting
        # that it was.
        self.seen_on_entry.append(getattr(self, "last_decline", "MISSING"))
        if self.set_reason is not None:
            self.last_decline = self.set_reason
        if self.fire:
            return {"strategy_id": self.strategy_id, "action": "enter"}
        return None


class _State:
    samples = [1, 2, 3]


def _registry(*stubs):
    reg = StrategyRegistry.__new__(StrategyRegistry)
    reg._strategies = {s.strategy_id: s for s in stubs}
    return reg


def test_a_strategy_that_names_nothing_still_reports_no_signal():
    """The 71 uninstrumented strategies must not move. This is the whole
    safety of the change: opt-in, or the population's labels all shift."""
    quiet = _Stub("quiet")
    reg = _registry(quiet)

    assert reg.evaluate_all(_State(), None) == []
    assert reg.last_skips == {"quiet": "no_signal"}


def test_a_named_decline_is_reported_verbatim():
    """The margin is the point -- it must survive to the stored blob intact."""
    named = _Stub("named")
    named.set_reason = "dip_too_shallow 0.31%<0.50%"
    reg = _registry(named)

    assert reg.evaluate_all(_State(), None) == []
    assert reg.last_skips == {"named": "dip_too_shallow 0.31%<0.50%"}


def test_a_stale_reason_is_never_reported_for_a_later_tick():
    """Decline naming a guard, then fire, then decline setting nothing.

    The third tick must report the bare "no_signal" and NOT the first tick's
    guard. Reporting a stale guard is the failure mode that would make the
    whole histogram lie, and it is silent.
    """
    s = _Stub("flip")
    reg = _registry(s)

    s.set_reason = "r30_not_positive -0.0012<=0"
    reg.evaluate_all(_State(), None)
    assert reg.last_skips == {"flip": "r30_not_positive -0.0012<=0"}

    # Tick 2: it fires, so there is no skip at all.
    s.fire = True
    s.set_reason = None
    assert len(reg.evaluate_all(_State(), None)) == 1
    assert reg.last_skips == {}

    # Tick 3: declines again, names nothing. Must NOT inherit tick 1's guard.
    s.fire = False
    reg.evaluate_all(_State(), None)
    assert reg.last_skips == {"flip": "no_signal"}

    # And prove the clearing is what did it, rather than a lucky overwrite.
    assert s.seen_on_entry == [None, None, None]


def test_a_falsy_reason_does_not_become_a_skip_label():
    """An empty string is not a diagnosis. It must fall back, not report ''."""
    for junk in ("", 0, False):
        s = _Stub("junk")
        s.set_reason = junk
        reg = _registry(s)
        reg.evaluate_all(_State(), None)
        assert reg.last_skips == {"junk": "no_signal"}, junk


def test_candidates_are_unchanged_when_a_strategy_fires():
    """Nothing on the candidate path moves: same candidate, no skip recorded."""
    a, b = _Stub("a"), _Stub("b")
    a.fire = True
    b.set_reason = "vol_fade 0.4<1.0"
    reg = _registry(a, b)

    cands = reg.evaluate_all(_State(), None)
    assert [c["strategy_id"] for c in cands] == ["a"]
    assert reg.last_skips == {"b": "vol_fade 0.4<1.0"}


def test_a_read_only_last_decline_cannot_kill_the_tick():
    """Clearing must never raise into the tick path. A strategy using
    __slots__ or a property has no writable attribute, and one strategy is
    never allowed to stop the others."""

    class _ReadOnly(_Stub):
        @property
        def last_decline(self):  # noqa: D401
            return "immutable_guard 1<2"

    ro = _ReadOnly("ro")
    other = _Stub("other")
    reg = _registry(ro, other)

    assert reg.evaluate_all(_State(), None) == []
    assert reg.last_skips == {"ro": "immutable_guard 1<2", "other": "no_signal"}
