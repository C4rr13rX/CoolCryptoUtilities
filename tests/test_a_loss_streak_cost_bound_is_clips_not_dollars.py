"""The streak cost bound is a fraction of a clip; reading it as dollars is 6x tighter.

``GHOST_MAX_LOSS_STREAK_COST`` decides whether a run of losses counts as a
breach. Its calibration note in ``trading/pipeline.py`` is the evidence for
which unit it is in:

    "Measured 2026-08-27 on corroborated atf_static trades: the worst 7-loss
     streak cost -0.067 in total (~$0.13 on a $2 clip)"

-0.067 is the number that came out of the book and $0.13 is its translation
into dollars, so ``profit`` was a FRACTION when 0.25 was chosen. The bound
meant "a quarter of one clip, cumulative" -- the same unit as the 0.08 stop it
sits beside. ``profit`` later became USD (services/roundtrip_cost.py) and the
comparison was never re-based, so 0.25 came to mean $0.25: on the $6 live clip
that is 4.2% of one clip, exhausted by THREE ordinary losses.

MEASURED 2026-09-06 on the 5-day book, priced at the $6 live clip:

    strategy               n   net USD  worst costly streak    cost
    atf_static_scout     105    +9.990          8            $0.669
    obv_accumulation@1w   10    -0.216          9            $0.312
    atf_static            46    +4.681          2            $0.793

Every book with a run longer than three reported its full length, so the cost
test had stopped discriminating and the guard had silently reverted to the bare
occurrence counter it was written to replace. The pooled verdict was
``effective_loss_streak = 9`` against a bar of 5 -- a veto worth $0.31 spread
over nine trades, against $18.19 of deployable capital -- and it was the last
thing holding the live lane shut.

At the calibrated meaning the bound is 0.25 x $6.00 = $1.50, about three
consecutive full stop-outs on an 8% stop. The guard keeps its teeth: on the
same book ``effective_loss_streak`` is 4, not 0, because one book does cross
$1.50.

This is the same shape as the tail-risk bug already fixed in this file -- a
guardrail calibrated while ``profit`` was fractional, still being compared
after ``profit`` became USD.

Against the pre-fix code (``streak_cost_guard = float(os.getenv(
"GHOST_MAX_LOSS_STREAK_COST", "0.25"))``) every test in BoundTest fails.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.metrics import TradePerformance  # noqa: E402
from trading.pipeline import TrainingPipeline  # noqa: E402

CLIP = 6.0


class _StubMetrics:
    def __init__(self, trades: List[TradePerformance]) -> None:
        self._trades = trades

    def ghost_trade_snapshot(self, **_kwargs: Any) -> List[TradePerformance]:
        return list(self._trades)

    def aggregate_trade_metrics(self, trades: List[TradePerformance]) -> Dict[str, float]:
        profits = [float(t.profit) for t in trades]
        if not profits:
            return {"win_rate": 0.0, "avg_profit": 0.0, "profit_factor": 0.0}
        wins = [p for p in profits if p > 0]
        gross_loss = abs(sum(p for p in profits if p <= 0))
        return {
            "win_rate": len(wins) / len(profits),
            "avg_profit": sum(profits) / len(profits),
            "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else 999.0,
        }


def _row(return_pct: float, ts: float) -> TradePerformance:
    """One round trip whose recorded prices imply ``return_pct``.

    Built from prices rather than from a profit figure so that
    price_book_at_live_clip -- which is what denominates the book the streak is
    counted over -- produces the value under test.
    """
    return TradePerformance(
        symbol="CBBTC-USDC",
        entry_ts=ts - 300.0,
        exit_ts=ts,
        profit=0.0,
        expected_delta=0.0,
        realized_delta=return_pct,
        reason="target",
        route=[],
        strategy_id="atf_static",
        return_pct=return_pct,
    )


def _book(
    now: float, losing_run: int, loss_return: float, win_return: float = 0.050
) -> List[TradePerformance]:
    """A profitable book carrying one run of ``losing_run`` losses.

    ``win_return`` is raised in the stop-out case so that the run is the ONLY
    term the gate fails on: six -8% losses against +5% wins also drag the
    payoff ratio under its 2.0 bar, and the verdict would then read
    ``no_tradeable_edge`` -- a true statement about a different guard.
    """
    rows: List[TradePerformance] = []
    ts = now - 7200.0
    for _ in range(12):
        ts += 30.0
        rows.append(_row(win_return, ts))
    for _ in range(losing_run):
        ts += 30.0
        rows.append(_row(loss_return, ts))
    for _ in range(12):
        ts += 30.0
        rows.append(_row(win_return, ts))
    return rows


class BoundTest(unittest.TestCase):
    ENV = {
        "LIVE_MIN_CLIP_USD": str(CLIP),
        "GHOST_MAX_LOSS_STREAK_COST": "0.25",
        "GHOST_MAX_LOSS_STREAK": "5",
        "GHOST_VALIDATION_LOOKBACK_SEC": "172800",
        "GHOST_MAX_STALE_SEC": "172800",
    }

    def _verdict(
        self, losing_run: int, loss_return: float, win_return: float = 0.050, **env: str
    ) -> Dict[str, Any]:
        import time as _time

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.metrics = _StubMetrics(
            _book(_time.time(), losing_run, loss_return, win_return)
        )
        pipeline.min_ghost_trades = 5
        pipeline.min_ghost_win_rate = 0.0
        pipeline.min_realized_margin = 0.0
        pipeline.focus_lookback_sec = 172800
        with mock.patch.dict(os.environ, {**self.ENV, **env}, clear=False):
            return pipeline._ghost_validation()

    def test_the_bound_is_reported_in_the_clip_it_is_compared_against(self):
        verdict = self._verdict(2, -0.010)
        self.assertAlmostEqual(
            float(verdict["loss_streak_cost_guardrail"]), 0.25 * CLIP, places=9
        )

    def test_a_cheap_run_of_eight_is_not_a_breach(self):
        """atf_static_scout's real shape: 8 losses costing $0.669 on a +$9.99 book."""
        verdict = self._verdict(8, -0.010)
        self.assertLess(
            float(verdict["max_loss_streak_cost"]), 0.25 * CLIP,
            "premise: eight 1% losses on a $6 clip cost well under a quarter clip",
        )
        self.assertEqual(int(verdict["max_loss_streak"]), 8)
        self.assertEqual(
            int(verdict["effective_loss_streak"]), 0,
            "a run that never reaches the bound is not a costly streak",
        )

    def test_a_run_of_genuine_stop_outs_still_breaches(self):
        """The guard keeps its teeth: ~three full 8% stop-outs exceed a quarter clip."""
        verdict = self._verdict(6, -0.080, win_return=0.250)
        self.assertGreater(float(verdict["max_loss_streak_cost"]), 0.25 * CLIP)
        self.assertGreaterEqual(int(verdict["effective_loss_streak"]), 5)
        self.assertFalse(verdict["tradeable_edge_block"], "the run must be the only fault")
        self.assertFalse(verdict["ready"])
        self.assertEqual(verdict["reason"], "loss_streak")

    def test_the_bound_tracks_the_clip_it_is_measured_at(self):
        """A clip change must not silently retighten the guard again."""
        at_six = self._verdict(2, -0.010)
        at_two = self._verdict(2, -0.010, LIVE_MIN_CLIP_USD="2.0")
        self.assertAlmostEqual(float(at_six["loss_streak_cost_guardrail"]), 0.25 * 6.0, places=9)
        self.assertAlmostEqual(float(at_two["loss_streak_cost_guardrail"]), 0.25 * 2.0, places=9)

    def test_an_explicit_dollar_bound_still_wins(self):
        verdict = self._verdict(2, -0.010, GHOST_MAX_LOSS_STREAK_COST_USD="0.25")
        self.assertAlmostEqual(float(verdict["loss_streak_cost_guardrail"]), 0.25, places=9)

    def test_zero_still_means_off(self):
        """With the cost test disabled the raw occurrence count is reported."""
        verdict = self._verdict(8, -0.010, GHOST_MAX_LOSS_STREAK_COST="0")
        self.assertEqual(float(verdict["loss_streak_cost_guardrail"]), 0.0)
        self.assertEqual(int(verdict["effective_loss_streak"]), 8)


if __name__ == "__main__":
    unittest.main()
