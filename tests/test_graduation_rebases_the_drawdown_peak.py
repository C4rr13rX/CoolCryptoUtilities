"""A fresh licence is not judged against the peak of a revoked one.

Link 5 (GRADUATION) read "no strategy approved for live" on 2026-09-04 while
``atf_static`` -- the only strategy that has ever spent real money on this
account -- was net POSITIVE on live money: +0.14230439571137737 over 9 live
round trips.

The ledger recorded how it happened. Approval and revocation landed in the same
``record()`` call, 27 microseconds apart on the wall clock:

    graduated_ts  1788529132.2898452
    demoted_ts    1788529132.2898726
    demote_reason "live drawdown: +0.1423 from peak +0.2221"

The give-back brake measures against ``dd_ref``, "the peak reached under the
CURRENT licence", and falls back to ``peak_profit`` -- "the most this strategy
has EVER been up" -- when that field is absent. ``_maybe_rearm_locked`` re-based
``dd_ref`` when it handed out a licence. ``_evaluate_graduation_locked`` did
not. So a strategy graduating with any historical peak above its current total
was demoted by its own history the instant it was approved, and re-arming then
demanded 20 fresh ghost trades -- days of ghosting on a book that yields ~3
exits an hour across every strategy combined.

No threshold reaches this. At the configured 25% the bar was +0.1666 against a
current +0.1423; lowering it only moves which strategies the trap catches.

These tests use the real figures from data/strategy_ledger.json as of
2026-09-04T13:38:52Z.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from trading.strategies.ledger import StrategyLedger

#: The real atf_static live record at the moment of the instant re-demotion.
LIVE_TRADES = 9
LIVE_PROFIT = 0.14230439571137737
LIVE_PEAK = 0.2221068671625748

#: Its ghost book, which clears the graduation bar (22/36 = 61% >= 55%).
GHOST_TRADES = 36
GHOST_WINS = 22
GHOST_PROFIT = 0.9127350117620626


def _entry() -> dict:
    """The ledger entry as it stood just before the 13:38:52 graduation."""
    return {
        "ghost": {
            "trades": GHOST_TRADES,
            "wins": GHOST_WINS,
            "losses": 3,
            "total_profit": GHOST_PROFIT,
            "peak_profit": 0.9924689923694343,
            "max_drawdown": 0.08501068553523006,
            "consecutive_losses": 3,
            "conf_ema": 0.17195,
            "last_ts": 1788538852.240907,
        },
        "live": {
            "trades": LIVE_TRADES,
            "wins": 2,
            "losses": 7,
            "total_profit": LIVE_PROFIT,
            "peak_profit": LIVE_PEAK,
            "max_drawdown": 0.07980247145119743,
            "consecutive_losses": 0,
            "conf_ema": 0.37290670858355,
            "last_ts": 1788529132.2898176,
            # NOTE: no `dd_ref`. That absence is the whole defect.
        },
        "live_approved": False,
        "demotions": 5,
        "demote_reason": None,
        "graduation_blocked": False,
        "graduation_blocked_reason": None,
    }


class GraduationRebasesTheDrawdownPeak(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ledger.json"
        self.path.write_text(json.dumps({"atf_static": _entry()}), encoding="utf-8")

        self._saved = {}
        for key, value in {
            "STRATEGY_GRADUATION_MIN_TRADES": "20",
            "STRATEGY_GRADUATION_MIN_WINRATE": "0.55",
            "STRATEGY_GRADUATION_MIN_PROFIT": "0.0",
            "STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES": "8",
            "STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN": "0.25",
            "STRATEGY_DEMOTE_MIN_LIVE_TRADES": "12",
            "STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "2",
        }.items():
            self._saved[key] = os.environ.get(key)
            os.environ[key] = value
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _ledger(self) -> StrategyLedger:
        return StrategyLedger(path=self.path)

    # ------------------------------------------------------------------

    def test_the_bar_that_convicted_it_was_the_old_peak(self) -> None:
        """Pin the arithmetic, so a threshold change cannot silently re-arm it."""
        bar = LIVE_PEAK * (1.0 - 0.25)
        self.assertAlmostEqual(bar, 0.1665801503719311, places=12)
        self.assertLess(LIVE_PROFIT, bar)          # the brake fires on peak_profit
        self.assertGreater(LIVE_PROFIT, 0.0)       # ...while it is UP on real money

    def test_graduation_seeds_dd_ref_from_the_current_total(self) -> None:
        led = self._ledger()
        ent = led._entry("atf_static")
        self.assertIsNone(ent["live"].get("dd_ref"), "precondition: no dd_ref")

        with led._lock:
            led._evaluate_graduation_locked("atf_static")

        self.assertTrue(ent["live_approved"])
        self.assertAlmostEqual(ent["live"]["dd_ref"], LIVE_PROFIT, places=15)
        # peak_profit means "the most this has ever been up" and must survive.
        self.assertAlmostEqual(ent["live"]["peak_profit"], LIVE_PEAK, places=15)

    def test_a_graduated_strategy_is_not_demoted_in_the_same_call(self) -> None:
        """The regression itself: approve, then run the brake, still approved."""
        led = self._ledger()
        ent = led._entry("atf_static")
        with led._lock:
            led._evaluate_graduation_locked("atf_static")
            led._evaluate_demotion_locked("atf_static")

        self.assertTrue(
            ent["live_approved"],
            "graduated then demoted in one call on reason=%r" % ent.get("demote_reason"),
        )
        self.assertIsNone(ent.get("demote_reason"))
        self.assertEqual(ent["demotions"], 5, "no new demotion was recorded")

    def test_record_of_a_live_outcome_leaves_it_approved(self) -> None:
        """Through the public API, which is what production actually calls."""
        led = self._ledger()
        led.record("atf_static", mode="live", profit=0.004, symbol="BSTONK-USDC")

        ent = led._entry("atf_static")
        self.assertTrue(ent["live_approved"], ent.get("demote_reason"))
        self.assertIn("atf_static", led.approved_ids())
        # dd_ref tracks the running max under this licence, from the re-base.
        self.assertAlmostEqual(
            ent["live"]["dd_ref"], LIVE_PROFIT + 0.004, places=12
        )

    def test_the_brake_still_fires_on_a_give_back_under_this_licence(self) -> None:
        """Re-basing must not disarm the brake -- only re-zero its reference."""
        led = self._ledger()
        led.record("atf_static", mode="live", profit=0.004, symbol="BSTONK-USDC")
        self.assertTrue(led._entry("atf_static")["live_approved"])

        # dd_ref is now +0.146304 and the 25% bar is +0.109728. Note the brake,
        # not the streak, is what must fire here: the consecutive-loss rule
        # cannot demote a strategy that is still net positive, by design.
        led.record("atf_static", mode="live", profit=-0.02, symbol="BSTONK-USDC")
        ent = led._entry("atf_static")
        bar = (LIVE_PROFIT + 0.004) * 0.75
        self.assertAlmostEqual(ent["live"]["total_profit"], LIVE_PROFIT - 0.016, places=12)
        self.assertGreater(ent["live"]["total_profit"], bar)
        self.assertTrue(ent["live_approved"], "+0.126304 is still above +0.109728")

        led.record("atf_static", mode="live", profit=-0.02, symbol="BSTONK-USDC")
        ent = led._entry("atf_static")
        self.assertAlmostEqual(ent["live"]["total_profit"], LIVE_PROFIT - 0.036, places=12)
        self.assertLess(ent["live"]["total_profit"], bar)
        self.assertFalse(ent["live_approved"], "+0.106304 is below +0.109728")
        self.assertIn("live drawdown", str(ent.get("demote_reason")))

    def test_a_never_live_strategy_is_unaffected(self) -> None:
        """peak_profit 0 -> the brake needs peak > 0, so nothing changes."""
        ent = _entry()
        ent["live"] = {
            "trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0,
            "peak_profit": 0.0, "max_drawdown": 0.0, "consecutive_losses": 0,
            "conf_ema": 0.0, "last_ts": 0.0,
        }
        self.path.write_text(json.dumps({"fresh": ent}), encoding="utf-8")

        led = self._ledger()
        with led._lock:
            led._evaluate_graduation_locked("fresh")
            led._evaluate_demotion_locked("fresh")
        row = led._entry("fresh")
        self.assertTrue(row["live_approved"])
        self.assertEqual(row["live"]["dd_ref"], 0.0)

    def test_rearm_and_graduation_grant_the_same_licence(self) -> None:
        """Both ways in go through one helper, so they cannot drift again."""
        ent = _entry()
        ent["demote_reason"] = "live drawdown: +0.1423 from peak +0.2221"
        ent["ghost_at_demotion"] = {"trades": 10, "wins": 5, "total_profit": 0.1}
        ent["ghost"] = {
            "trades": 40, "wins": 27, "losses": 13, "total_profit": 0.9,
            "peak_profit": 0.99, "max_drawdown": 0.085,
            "consecutive_losses": 0, "conf_ema": 0.17, "last_ts": 1788538852.0,
        }
        self.path.write_text(json.dumps({"atf_static": ent}), encoding="utf-8")

        led = self._ledger()
        row = led._entry("atf_static")
        with led._lock:
            led._maybe_rearm_locked("atf_static")   # 30 fresh ghost, 22 wins (73%)

        self.assertTrue(row["live_approved"], "re-arm did not fire")
        self.assertIsNotNone(row.get("rearmed_ts"))
        self.assertAlmostEqual(row["live"]["dd_ref"], LIVE_PROFIT, places=15)

        with led._lock:
            led._evaluate_demotion_locked("atf_static")
        self.assertTrue(row["live_approved"], "re-armed then instantly demoted")


if __name__ == "__main__":
    unittest.main()
