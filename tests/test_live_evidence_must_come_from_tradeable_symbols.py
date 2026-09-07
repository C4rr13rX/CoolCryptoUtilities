"""Profit earned on symbols the entry gate refuses cannot authorise live money.

Every statistic the live gate reads was computed over the WHOLE ghost book,
including symbols ``stop_survivability_gate`` refuses -- symbols whose p99
single-tick jump exceeds the stop, so there is no stop worth the name to place.
A ghost trade on such a symbol rehearses nothing the live lane can repeat.

Measured 2026-09-06 on the 48h book (242 paired round trips):

    BSTONK-USDC    34 trades  net +6.56649   REFUSED
    BPAD-USDC       6 trades  net +2.12736   REFUSED
    BASECAT-USDC   32 trades  net +1.93025   REFUSED
    MOONBASE-USDC   7 trades  net +0.73003   REFUSED
    -----------------------------------------------------
    refused symbols            net +11.35993 over  82 trades
    tradeable symbols          net  +0.15181 over 160 trades

98.7% of the book's net profit was on symbols that can no longer be entered. The
pooled book reported profit_factor 2.567, payoff 2.698 and expectancy
+$0.04757/trade -- clearing every expectancy-path threshold -- while the same
measures over the tradeable subset were profit_factor 1.070, payoff 1.784 and
+$0.00095/trade against a $0.02317 round trip.

``single_symbol_dependence`` does not catch it: that jackknife drops the ONE top
symbol, and net-ex-BSTONK is +4.9453 because BASECAT, BPAD and MOONBASE carry
it -- all three equally unenterable.
"""

from __future__ import annotations

import os
import time
import unittest
from typing import Any, Dict, List, Sequence
from unittest import mock

from trading.metrics import MetricsCollector, TradePerformance
from trading.pipeline import TrainingPipeline

REFUSED = {"BSTONK-USDC", "BASECAT-USDC"}


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
    # High enough that the win-rate path and fast-track both refuse, so the
    # POSITIVE-EXPECTANCY path is the only route to live -- the one this guards.
    pipeline.min_ghost_win_rate = 0.95
    pipeline.min_realized_margin = 0.0
    pipeline.focus_lookback_sec = 172800
    return pipeline


def _book() -> List[TradePerformance]:
    """A book whose edge lives entirely on two unenterable symbols.

    Refused symbols: 12 wins of +$0.50 against 3 losses of -$0.05, twice over.
    Tradeable symbols: 10 wins of +$0.02 against 10 losses of -$0.02 -- dead
    flat, payoff 1.0, which is what the real tradeable subset looks like.
    """
    now = time.time()
    book: List[TradePerformance] = []
    ts = now - 40000.0
    for symbol in ("BSTONK-USDC", "BASECAT-USDC"):
        for i in range(15):
            profit, ret = (+0.50, +0.08) if i % 5 != 4 else (-0.05, -0.01)
            book.append(_trade(symbol, "refused_lane", profit, ts, ret))
            ts += 60.0
    for symbol in ("AERO-USDC", "CBBTC-USDC"):
        for i in range(20):
            profit, ret = (+0.02, +0.003) if i % 2 == 0 else (-0.02, -0.003)
            book.append(_trade(symbol, "tradeable_lane", profit, ts, ret))
            ts += 60.0
    return book


class TradeableEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch(
            "trading.pipeline.stop_is_unenforceable",
            side_effect=lambda symbol: str(symbol).upper() in REFUSED,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.book = _book()

    def _verdict(self, **env: str) -> Dict[str, Any]:
        base = {"GHOST_MAX_LOSS_STREAK_COST": "0.25", "GHOST_TRADEABLE_MIN_TRADES": "30"}
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=False):
            return _pipeline(self.book)._ghost_validation()

    def test_the_pooled_book_looks_profitable(self):
        """Precondition: without the guard this book WOULD have gone live."""
        verdict = self._verdict(GHOST_REQUIRE_TRADEABLE_EDGE="0")
        self.assertGreater(verdict["total_net_profit"], 0.0)
        self.assertGreaterEqual(verdict["profit_factor"], 1.5)
        self.assertGreaterEqual(verdict["payoff_ratio"], 2.0)
        self.assertGreater(verdict["net_expectancy"], 0.0)
        self.assertTrue(
            verdict["ready"],
            "with the guard off the expectancy path must clear -- otherwise this "
            "test is not measuring the guard. reason=%r" % (verdict.get("reason"),),
        )
        self.assertEqual(verdict["reason"], "positive_expectancy")

    def test_profit_from_refused_symbols_does_not_authorise_live(self):
        verdict = self._verdict()
        self.assertFalse(
            verdict["ready"],
            "98%% of this book's profit is on symbols the entry gate refuses",
        )
        self.assertEqual(verdict["reason"], "no_tradeable_edge")
        self.assertTrue(verdict["tradeable_edge_block"])

    def test_the_verdict_reports_the_tradeable_book_it_judged(self):
        verdict = self._verdict()
        self.assertEqual(verdict["tradeable_samples"], 40)
        self.assertAlmostEqual(verdict["tradeable_net_profit"], 0.0, places=6)
        self.assertAlmostEqual(verdict["tradeable_payoff_ratio"], 1.0, places=6)
        self.assertLess(verdict["tradeable_payoff_ratio"], 2.0)
        # The pooled numbers must still be reported, and must differ -- that
        # divergence is the whole finding.
        self.assertGreater(verdict["payoff_ratio"], verdict["tradeable_payoff_ratio"])
        self.assertGreater(verdict["total_net_profit"], verdict["tradeable_net_profit"])

    def test_a_book_with_real_tradeable_edge_still_goes_live(self):
        """The guard must not shut the lane on a genuinely tradeable book."""
        now = time.time()
        book: List[TradePerformance] = []
        ts = now - 40000.0
        for symbol in ("AERO-USDC", "CBBTC-USDC"):
            for i in range(30):
                # 2 wins of +$0.30 per loss of -$0.05: payoff 6.0, pf 12.0.
                profit, ret = (+0.30, +0.05) if i % 3 != 2 else (-0.05, -0.01)
                book.append(_trade(symbol, "tradeable_lane", profit, ts, ret))
                ts += 60.0
        self.book = book
        verdict = self._verdict()
        self.assertFalse(verdict["tradeable_edge_block"])
        self.assertTrue(verdict["ready"], "reason=%r" % (verdict.get("reason"),))
        self.assertEqual(verdict["reason"], "positive_expectancy")
        self.assertEqual(verdict["tradeable_samples"], 60)

    def test_a_subset_too_thin_to_judge_does_not_block(self):
        """Refusing on an unmeasured subset would shut the lane on no evidence."""
        verdict = self._verdict(GHOST_TRADEABLE_MIN_TRADES="500")
        self.assertFalse(verdict["tradeable_edge_block"])
        self.assertTrue(verdict["ready"])


if __name__ == "__main__":
    unittest.main()
