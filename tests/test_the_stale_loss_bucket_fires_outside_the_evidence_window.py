"""The scout's only loss-realising exit fired exactly at the evidence horizon.

MEASURED 2026-09-07, over every ``ghost-exit`` row in ``trading_ops``, filtered
to live-tradeable symbols and split on
``trading.strategies.ledger._exceeds_evidence_horizon``:

    rows dropped as out-of-horizon    3     0 wins    3 losses    net -0.0706
    exit reasons                      stale_underwater x3
    ages                              4.07h, 4.09h, 4.10h   (horizon 4.00h)

Every live-tradeable round trip the horizon filter has ever dropped is a loss,
and every one of them is the ``stale_underwater`` bucket.

That is not a coincidence and it is not sampling noise, it is two constants
colliding. ``services/atf_static_strategy`` reaches ``stale_underwater`` at
``max(2 * max_hold_sec, ATF_STATIC_MAX_UNDERWATER_SEC)`` = 14400s at the
shipped defaults, and ``_max_evidence_hold_sec()`` returns
``MAX_HOLD_SECONDS * STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE`` = 3600 * 4 = 14400s.
The same number. Because the scout only evaluates positions when a tick
arrives, the exit always landed a few minutes the wrong side of the bound.

And ``stale_underwater`` is the ONLY reason this module can reach carrying a
loss that never hit its stop: ``target_hit`` requires clearing the round trip
and ``max_hold`` requires ``profit > cost_rate``, so both are winners by
construction. So the graduation book kept every winner and dropped every slow
loser. Downstream, ``atf_static_scout``'s live-tradeable record read
62 trades / 62 wins / 0 losses / +0.2708 and cleared the 20-trip, 55%,
net-positive bar outright -- the only strategy in the population that did --
on a book whose losses could not physically appear in it.

These tests pin the bound to the horizon it is graded against. Against the old
``max(...)`` they fail: the bound equalled the horizon, so a round trip closed
on it did not count as evidence once any tick lag was added.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

from services.atf_static_strategy import _stale_underwater_sec
from trading.strategies import ledger as ledger_mod


@contextmanager
def _env(**pairs):
    """Set env vars for the body, restoring exactly what was there before."""
    saved = {k: os.environ.get(k) for k in pairs}
    try:
        for k, v in pairs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


#: Longest gap ever observed between the stale bound elapsing and the tick that
#: actually closed the position: the three recorded rows fired at 4.07h, 4.09h
#: and 4.10h against a 4.00h bound, so 6 minutes. Tested against ten times that
#: so a slow feed cannot silently re-open the hole.
_OBSERVED_TICK_LAG_SEC = 360.0


class StaleLossBucketFiresInsideTheEvidenceWindow(unittest.TestCase):
    def test_stale_bound_is_strictly_inside_the_evidence_horizon(self):
        """The bound must not equal the horizon. That equality WAS the bug."""
        with _env(MAX_HOLD_SECONDS=3600, STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE=None,
                  ATF_STATIC_MAX_UNDERWATER_SEC=None,
                  ATF_STATIC_STALE_HORIZON_MARGIN=None):
            horizon = ledger_mod._max_evidence_hold_sec()
            bound = _stale_underwater_sec(3600.0)
        self.assertEqual(horizon, 14400.0, "shipped default horizon changed")
        self.assertLess(
            bound,
            horizon,
            "the only exit that realises a slow loss fires at %.0fs and the "
            "evidence horizon rejects at %.0fs -- every such loss is dropped "
            "from the graduation book" % (bound, horizon),
        )

    def test_a_position_closed_on_the_bound_counts_as_evidence(self):
        """Closing exactly on the bound must produce a countable round trip."""
        with _env(MAX_HOLD_SECONDS=3600, STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE=None,
                  ATF_STATIC_MAX_UNDERWATER_SEC=None,
                  ATF_STATIC_STALE_HORIZON_MARGIN=None):
            bound = _stale_underwater_sec(3600.0)
            self.assertFalse(
                ledger_mod._exceeds_evidence_horizon(bound),
                "a loss realised at the stale bound is refused as evidence",
            )

    def test_the_bound_survives_the_tick_lag_that_was_actually_observed(self):
        """The three real rows landed 4-6 min late. Leave room for ten times that."""
        with _env(MAX_HOLD_SECONDS=3600, STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE=None,
                  ATF_STATIC_MAX_UNDERWATER_SEC=None,
                  ATF_STATIC_STALE_HORIZON_MARGIN=None):
            bound = _stale_underwater_sec(3600.0)
            for lag in (0.0, 60.0, _OBSERVED_TICK_LAG_SEC, _OBSERVED_TICK_LAG_SEC * 10):
                with self.subTest(tick_lag_sec=lag):
                    self.assertFalse(
                        ledger_mod._exceeds_evidence_horizon(bound + lag),
                        "a stale exit %.0fs after the bound stops counting" % lag,
                    )

    def test_the_bound_tracks_the_horizon_rather_than_a_fixed_number(self):
        """Retune MAX_HOLD_SECONDS and the bound must follow, not drift."""
        for max_hold, mult in ((600, 4.0), (1800, 3.0), (3600, 4.0), (7200, 2.0)):
            with self.subTest(max_hold=max_hold, multiple=mult):
                with _env(MAX_HOLD_SECONDS=max_hold,
                          STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE=mult,
                          ATF_STATIC_MAX_UNDERWATER_SEC=None,
                          ATF_STATIC_STALE_HORIZON_MARGIN=None):
                    # _MAX_HOLD_MULTIPLE is read at import; drive the horizon
                    # through the same module constant the ledger reads.
                    saved = ledger_mod._MAX_HOLD_MULTIPLE
                    ledger_mod._MAX_HOLD_MULTIPLE = mult
                    try:
                        horizon = ledger_mod._max_evidence_hold_sec()
                        bound = _stale_underwater_sec(float(max_hold))
                        self.assertLess(bound, horizon)
                        self.assertFalse(ledger_mod._exceeds_evidence_horizon(bound))
                    finally:
                        ledger_mod._MAX_HOLD_MULTIPLE = saved

    def test_the_backstop_never_fires_before_the_hold_timer(self):
        """Clamping must not crystallise the loss the hold timer refused to take."""
        with _env(MAX_HOLD_SECONDS=3600, ATF_STATIC_MAX_UNDERWATER_SEC=None,
                  ATF_STATIC_STALE_HORIZON_MARGIN=None):
            saved = ledger_mod._MAX_HOLD_MULTIPLE
            # A horizon barely longer than one hold: the clamp would otherwise
            # pull the backstop in front of the timer it backs up.
            ledger_mod._MAX_HOLD_MULTIPLE = 1.1
            try:
                self.assertGreaterEqual(_stale_underwater_sec(3600.0), 3600.0)
            finally:
                ledger_mod._MAX_HOLD_MULTIPLE = saved

    def test_a_tighter_underwater_setting_is_still_honoured(self):
        """The clamp only lowers the bound. It must never raise a tight one."""
        with _env(MAX_HOLD_SECONDS=3600, ATF_STATIC_MAX_UNDERWATER_SEC=5400,
                  ATF_STATIC_STALE_HORIZON_MARGIN=None):
            self.assertEqual(_stale_underwater_sec(1800.0), 5400.0)

    def test_a_nonsense_margin_falls_back_rather_than_disabling_the_clamp(self):
        """A mis-set fraction must not restore the bug."""
        for bad in ("0", "1", "-0.5", "12", "banana", ""):
            with self.subTest(margin=bad):
                with _env(MAX_HOLD_SECONDS=3600,
                          ATF_STATIC_MAX_UNDERWATER_SEC=None,
                          ATF_STATIC_STALE_HORIZON_MARGIN=bad):
                    bound = _stale_underwater_sec(3600.0)
                    horizon = ledger_mod._max_evidence_hold_sec()
                    self.assertLess(bound, horizon)

    def test_stale_underwater_is_the_only_loss_bearing_exit_reason(self):
        """The premise. If another reason can carry a slow loss, retune this test.

        ``max_hold`` is gated on ``profit > cost_rate`` and ``target_hit`` on
        clearing the round trip, so neither can label a loser. Read the source
        rather than trusting the comment above it -- this repo has shipped a
        test that asserted on a word appearing only in a comment.
        """
        import inspect

        import services.atf_static_strategy as mod

        src = inspect.getsource(mod)
        head, _, tail = src.partition('reason = "max_hold"')
        self.assertTrue(tail, "the max_hold branch moved; re-derive this test")
        # The guard immediately preceding the assignment must compare the
        # return against the round-trip cost, not against zero.
        guard = head.rsplit("if ", 1)[1]
        self.assertIn("cost_rate", guard,
                      "max_hold no longer requires clearing the round trip, so "
                      "it can now book a loss and this file's premise is stale")


if __name__ == "__main__":
    unittest.main()
