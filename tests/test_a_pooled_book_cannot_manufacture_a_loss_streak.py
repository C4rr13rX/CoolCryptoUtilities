"""A loss streak belongs to one decision-maker, not to 36 of them concatenated.

``effective_loss_streak`` was computed over the POOLED ghost book -- every
strategy's trades in one chronological list. Measured 2026-09-06 on the 48h book
the live gate reads (242 paired round trips, 36 strategies), that reported a
12-trade costly streak against a bar of 5, and the "streak" was six independent
strategies each losing once or twice inside the same five hours:

    bus_schedule x4, atf_static_scout x4, obv_accumulation@5h,
    obv_accumulation@1w, obv_accumulation@12h, rsi_reversal@1d

across seven symbols. No strategy ran 12 losses; the worst single one ran 8, and
30 of the 36 ran 2 or fewer. The bar of 5 is calibrated on ONE strategy's own
sequence, so comparing it against a concatenation makes the left-hand side grow
with the number of strategies trading concurrently -- the guard tightens as the
fleet grows, for reasons that have nothing to do with risk.

This is the shape the bug had: interleave enough independent strategies and a
market-wide down-hour becomes a "losing run" for a trader that does not exist.
"""

from __future__ import annotations

import os
import time
import unittest
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

from trading.metrics import MetricsCollector, TradePerformance
from trading.pipeline import TrainingPipeline


def _trade(symbol: str, strategy: str, profit: float, ts: float, ret: float) -> TradePerformance:
    return TradePerformance(
        symbol=symbol,
        entry_ts=ts - 60.0,
        exit_ts=ts,
        profit=profit,
        expected_delta=0.0,
        realized_delta=0.0,
        reason="test",
        route=[],
        strategy_id=strategy,
        return_pct=ret,
    )


class _FakeMetrics:
    def __init__(self, trades: Sequence[TradePerformance]) -> None:
        self._trades = list(trades)

    def ghost_trade_snapshot(self, **kwargs: Any) -> List[TradePerformance]:
        wanted = kwargs.get("strategy_id")
        if wanted is None:
            return list(self._trades)
        return [t for t in self._trades if t.strategy_id == str(wanted).strip()]

    def aggregate_trade_metrics(self, trades: Sequence[TradePerformance]) -> Dict[str, float]:
        return MetricsCollector.aggregate_trade_metrics(self, trades)  # type: ignore[arg-type]


def _pipeline(trades: Sequence[TradePerformance]) -> TrainingPipeline:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.metrics = _FakeMetrics(trades)
    pipeline.min_ghost_trades = 5
    pipeline.min_ghost_win_rate = 0.0
    pipeline.min_realized_margin = 0.0
    pipeline.focus_lookback_sec = 172800
    return pipeline


class PooledLossStreakTest(unittest.TestCase):
    """Six strategies losing twice each is not a twelve-loss streak."""

    def _interleaved_book(self) -> List[TradePerformance]:
        """12 consecutive pooled losses, but only 2 per strategy.

        Each strategy loses $0.05 twice -- $0.10, comfortably inside the $0.25
        cost bound. Pooled, the same twelve trades cost $0.60 and cross it.
        """
        now = time.time()
        strategies = [
            "bus_schedule",
            "atf_static_scout",
            "obv_accumulation@5h",
            "obv_accumulation@1w",
            "obv_accumulation@12h",
            "rsi_reversal@1d",
        ]
        book: List[TradePerformance] = []
        # A win apiece first, so every strategy's streak starts clean.
        ts = now - 4000.0
        for strat in strategies:
            book.append(_trade("AERO-USDC", strat, +0.40, ts, +0.01))
            ts += 10.0
        # Now the interleaved losing hour: round-robin, twice through.
        for _ in range(2):
            for strat in strategies:
                book.append(_trade("AERO-USDC", strat, -0.05, ts, -0.002))
                ts += 10.0
        return book

    def test_interleaved_strategies_do_not_make_a_pooled_streak(self):
        book = self._interleaved_book()
        # Precondition: pooled, these really are 12 losses in a row.
        tail = [t for t in book if t.profit <= 0]
        self.assertEqual(len(tail), 12)
        self.assertEqual(book[-12:], tail, "the last 12 pooled trades must all be losses")
        self.assertGreater(
            sum(abs(t.profit) for t in tail), 0.25,
            "pooled the run must breach the cost bound, or the test proves nothing",
        )

        with mock.patch.dict(os.environ, {"GHOST_MAX_LOSS_STREAK_COST": "0.25"}, clear=False):
            verdict = _pipeline(book)._ghost_validation()

        # Per strategy nobody ran more than two, and two cheap losses are not a
        # breach -- so the effective streak is zero, not twelve.
        self.assertEqual(
            verdict["max_loss_streak"], 2,
            "the longest run any ONE strategy had was 2; got %r" % (verdict["max_loss_streak"],),
        )
        self.assertEqual(
            verdict["effective_loss_streak"], 0,
            "no strategy ran a costly streak; pooled accounting reported 12",
        )
        self.assertNotEqual(verdict.get("reason"), "loss_streak")

    def test_a_real_single_strategy_streak_still_breaches(self):
        """The guard keeps its teeth: one strategy losing real money in a row."""
        now = time.time()
        book = [_trade("AERO-USDC", "atf_static", +0.40, now - 5000.0, +0.01)]
        ts = now - 4000.0
        for _ in range(6):
            book.append(_trade("AERO-USDC", "atf_static", -0.10, ts, -0.01))
            ts += 10.0
        # Other strategies trading profitably alongside must not dilute it.
        for i in range(6):
            book.append(_trade("CBBTC-USDC", "rsi_reversal", +0.30, ts, +0.01))
            ts += 10.0

        with mock.patch.dict(os.environ, {"GHOST_MAX_LOSS_STREAK_COST": "0.25"}, clear=False):
            verdict = _pipeline(book)._ghost_validation()

        self.assertEqual(verdict["max_loss_streak"], 6)
        self.assertGreaterEqual(
            verdict["effective_loss_streak"], 3,
            "a $0.60 six-loss run by ONE strategy is a genuine breach",
        )

    def test_unattributed_trades_are_their_own_book(self):
        """Trades with no strategy must not be folded into a strategy's record."""
        now = time.time()
        book = [
            _trade("AERO-USDC", "atf_static", +0.50, now - 5000.0, +0.01),
            _trade("AERO-USDC", "", -0.20, now - 4000.0, -0.01),
            _trade("AERO-USDC", "atf_static", -0.20, now - 3900.0, -0.01),
            _trade("AERO-USDC", "", -0.20, now - 3800.0, -0.01),
        ]
        with mock.patch.dict(os.environ, {"GHOST_MAX_LOSS_STREAK_COST": "0.25"}, clear=False):
            verdict = _pipeline(book)._ghost_validation()
        # The two unattributed losses are consecutive within their own book
        # ($0.40, a breach); atf_static's single $0.20 loss is not.
        self.assertEqual(verdict["max_loss_streak"], 2)
        self.assertEqual(verdict["effective_loss_streak"], 2)


if __name__ == "__main__":
    unittest.main()
