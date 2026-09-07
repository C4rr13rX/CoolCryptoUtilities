"""The live gate must not ratio dollars recorded at 1276x different sizes.

MEASURED 2026-09-06 on the 5-day ghost book (246 paired round trips).

``_ghost_validation`` decides live trading from ``profit`` -- expectancy,
profit factor, payoff ratio, total net, the sign that makes a win a win, and
the cost bound on a losing streak are all sums or ratios over that column. A
ghost trade is a SIMULATION, so its notional is whatever the lane that wrote it
chose. Each row's implied notional is ``profit / return_pct``, and across
``atf_static``'s 34 tradeable trades it spans:

    min $0.031   p25 $0.272   median $1.000   p75 $4.863   max $39.573

so the book is a mixture of games and every ratio over it is decided by which
rows happened to be recorded large:

    atf_static, tradeable subset      as-is (mixed)   at the $6 live clip
    expectancy USD/trade                  -0.00143            +0.01140
    net USD                               -0.04850            +0.38760
    profit factor                            0.850               1.537

    pooled tradeable subset (153)     as-is (mixed)   at the $6 live clip
    expectancy USD/trade                  +0.00245            +0.04473
    profit factor                            1.239               2.792
    payoff ratio                             2.145               4.448

The mixed book reported ``no_tradeable_edge`` -- profit factor 1.239 against a
1.5 bar -- and held the live lane shut on 0 live trades. A live entry is raised
to ``ghost_clip_usd()`` every time, so the size the evidence is priced at and
the size the decision executes at must be the same number.

The re-pricing charges a FULL round trip to every row, so it cannot be a way of
flattering the book: a trade whose gross return does not cover the fee becomes
a loss it was not before. ``test_repricing_charges_every_row_its_round_trip``
pins that direction.

Against the pre-fix code (``trades`` used raw from ``ghost_trade_snapshot``)
every test in ReplayTest and RepricingTest fails.
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

from services.roundtrip_cost import roundtrip_cost_usd  # noqa: E402
from trading.metrics import TradePerformance  # noqa: E402
from trading.pipeline import TrainingPipeline, price_book_at_live_clip  # noqa: E402

CLIP = 6.0


def _trade(symbol: str, return_pct: float, notional: float, ts: float) -> TradePerformance:
    """One simulated round trip, recorded at ``notional`` the way a lane wrote it."""
    return TradePerformance(
        symbol=symbol,
        entry_ts=ts - 600.0,
        exit_ts=ts,
        profit=return_pct * notional - roundtrip_cost_usd(notional),
        expected_delta=0.0,
        realized_delta=return_pct,
        reason="target",
        route=[],
        strategy_id="atf_static",
        return_pct=return_pct,
    )


class RepricingTest(unittest.TestCase):
    """What price_book_at_live_clip does to a row, in isolation."""

    def test_two_identical_returns_at_different_sizes_become_one_number(self):
        """The bug in one line: same trade, two clips, two different 'edges'."""
        small = _trade("AAA-USDC", 0.02, 0.03, 1000.0)
        large = _trade("AAA-USDC", 0.02, 39.57, 1000.0)
        self.assertNotAlmostEqual(small.profit, large.profit, places=4)
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": str(CLIP)}, clear=False):
            priced = price_book_at_live_clip([small, large])
        self.assertAlmostEqual(priced[0].profit, priced[1].profit, places=12)
        self.assertAlmostEqual(
            priced[0].profit, 0.02 * CLIP - roundtrip_cost_usd(CLIP), places=12
        )

    def test_repricing_charges_every_row_its_round_trip(self):
        """It is not a whitewash: a flat trade becomes the loss it always was."""
        flat = _trade("AAA-USDC", 0.0005, 39.57, 1000.0)
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": str(CLIP)}, clear=False):
            priced = price_book_at_live_clip([flat])[0]
        self.assertLess(priced.profit, 0.0, "0.05% does not cover a 0.39% round trip")

    def test_a_row_with_no_recorded_return_is_left_alone(self):
        """Re-pricing without a price would be inventing one; shortening is worse."""
        blind = _trade("AAA-USDC", 0.02, 1.0, 1000.0)
        blind = TradePerformance(**{**blind.__dict__, "return_pct": None})
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": str(CLIP)}, clear=False):
            priced = price_book_at_live_clip([blind])
        self.assertEqual(len(priced), 1)
        self.assertAlmostEqual(priced[0].profit, blind.profit, places=12)

    def test_repricing_is_idempotent(self):
        """Scout rows arrive already converted; converting twice must not double-charge."""
        row = _trade("AAA-USDC", 0.02, 0.03, 1000.0)
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": str(CLIP)}, clear=False):
            once = price_book_at_live_clip([row])
            twice = price_book_at_live_clip(once)
        self.assertAlmostEqual(once[0].profit, twice[0].profit, places=12)


class _StubMetrics:
    """Enough MetricsCollector for _ghost_validation, over a fixed book."""

    def __init__(self, trades: List[TradePerformance]) -> None:
        self._trades = trades

    def ghost_trade_snapshot(self, **_kwargs: Any) -> List[TradePerformance]:
        return list(self._trades)

    def aggregate_trade_metrics(self, trades: List[TradePerformance]) -> Dict[str, float]:
        profits = [float(t.profit) for t in trades]
        if not profits:
            return {"win_rate": 0.0, "avg_profit": 0.0, "profit_factor": 0.0}
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        gross_loss = abs(sum(losses))
        return {
            "win_rate": len(wins) / len(profits),
            "avg_profit": sum(profits) / len(profits),
            "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else 999.0,
        }


def _book(now: float) -> List[TradePerformance]:
    """A book whose edge is real but whose recorded sizes disagree.

    Winners are recorded at the measured MEDIAN notional ($1.000) and losers at
    the measured p75 ($4.863) -- the shape found on atf_static, where the
    mixed-clip profit factor was 0.850 while the same trades at one clip were
    1.537. Both halves stay genuinely signed in the mixed book, so the verdict
    turns on the ratio rather than on a row that changed sign.

    Every symbol is CBBTC-USDC so the tradeable subset is the whole book and
    stop_survivability cannot narrow what is being measured.
    """
    rows: List[TradePerformance] = []
    ts = now - 3600.0
    for i in range(40):
        ts += 30.0
        if i % 2 == 0:
            rows.append(_trade("CBBTC-USDC", 0.030, 1.000, ts))   # a win, recorded at the median
        else:
            rows.append(_trade("CBBTC-USDC", -0.004, 4.863, ts))  # a loss, recorded at the p75
    return rows


class ReplayTest(unittest.TestCase):
    """The verdict the live gate reaches on that book."""

    ENV = {
        "LIVE_MIN_CLIP_USD": str(CLIP),
        "GHOST_REQUIRE_TRADEABLE_EDGE": "1",
        "GHOST_TRADEABLE_MIN_TRADES": "30",
        "GHOST_EXPECTANCY_MIN_PROFIT_FACTOR": "1.5",
        "GHOST_EXPECTANCY_MIN_PAYOFF": "2.0",
        "GHOST_VALIDATION_LOOKBACK_SEC": "172800",
        "GHOST_MAX_STALE_SEC": "172800",
    }

    def _verdict(self) -> Dict[str, Any]:
        import time as _time

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.metrics = _StubMetrics(_book(_time.time()))
        pipeline.min_ghost_trades = 25
        pipeline.min_ghost_win_rate = 0.0
        pipeline.min_realized_margin = 0.0
        pipeline.focus_lookback_sec = 172800
        with mock.patch.dict(os.environ, self.ENV, clear=False):
            return pipeline._ghost_validation()

    def test_the_mixed_clip_book_would_have_read_as_a_losing_one(self):
        """Guards the premise: as written, these rows really do ratio to < 1."""
        import time as _time

        profits = [t.profit for t in _book(_time.time())]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        self.assertEqual(len(wins), 20, "premise: the winners must still be wins")
        self.assertLess(
            sum(wins) / abs(sum(losses)), 1.5,
            "premise: mixed clips sink the ratio below the bar the same trades "
            "clear at one clip",
        )

    def test_a_real_edge_is_not_refused_as_no_tradeable_edge(self):
        verdict = self._verdict()
        self.assertNotEqual(
            verdict.get("reason"),
            "no_tradeable_edge",
            "a +3.0%/-0.4% book at one clip has an edge; only the recorded "
            "sizes said otherwise",
        )

    def test_the_tradeable_ratios_are_computed_at_the_live_clip(self):
        verdict = self._verdict()
        self.assertGreaterEqual(float(verdict["tradeable_profit_factor"]), 1.5)
        self.assertGreater(float(verdict["tradeable_expectancy"]), 0.0)
        self.assertAlmostEqual(
            float(verdict["tradeable_expectancy"]),
            (0.5 * (0.030 * CLIP - roundtrip_cost_usd(CLIP))
             + 0.5 * (-0.004 * CLIP - roundtrip_cost_usd(CLIP))),
            places=9,
        )

    def test_the_pooled_ratios_are_computed_at_the_live_clip_too(self):
        """Both books or neither -- a half-converted verdict is its own bug."""
        verdict = self._verdict()
        self.assertGreater(float(verdict["total_net_profit"]), 0.0)
        self.assertGreaterEqual(float(verdict["profit_factor"]), 1.5)


if __name__ == "__main__":
    unittest.main()
