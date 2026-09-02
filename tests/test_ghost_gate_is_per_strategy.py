"""One strategy's losses must not convict another strategy's record.

Live execution is already per-strategy: STRATEGY_GRADUATION_ENFORCED means only
a graduated strategy spends real money, and StrategyLedger tracks each one
separately. But `_ghost_validation` pooled every strategy's trades into a single
book, and `_build_transition_plan` gated live money on that pooled verdict. So a
strategy could be graduated, live-approved and net positive, and still be
blocked by trades it did not make.

Measured 2026-09-02 over the 48h ghost book (56 paired round trips):

    atf_static          38 trades   35W/3L   net +0.5192
    rsi_reversal@5h     13 trades    0W/13L  net -1.9118
    money_button         1 trade     1W/0L   net +0.0053
    (four others)         4 trades              net -0.0092

rsi_reversal@5h's thirteen losses are thirteen near-simultaneous BASECAT-USDC
positions, opened between 0.0316 and 0.0319 and closed between 0.0303 and
0.0304 -- one 4.7% move in one token, counted thirteen times, at a clip roughly
8x atf_static's. Pooled, they take the book to -1.4722 and every aggregate with
it: profit factor 0.304, payoff 0.156, net expectancy -0.0328. The gate reported
`negative_margin` and set block_reason=ghost_validation_block.

Scoping the gate does not weaken it -- every guardrail still applies, to the
record of the strategy being judged. These tests pin both halves: that a
profitable strategy is judged on its own book, AND that scoping did not become a
way for a bad record to slip through.
"""

from __future__ import annotations

import unittest
from unittest import mock

from trading.metrics import MetricsCollector, TradePerformance
from trading.pipeline import TrainingPipeline


def _trade(profit: float, *, symbol: str = "AAA-USDC", strategy_id: str = "s") -> TradePerformance:
    return TradePerformance(
        symbol=symbol,
        entry_ts=0.0,
        exit_ts=60.0,
        profit=profit,
        expected_delta=0.0,
        realized_delta=0.0,
        reason="target_hit",
        route=[],
        strategy_id=strategy_id,
    )


class _StubPipeline:
    """A TrainingPipeline stand-in that serves a fixed book.

    `_ghost_validation` is called unbound so no database, model or filesystem
    is touched; only the book and the env matter to what it computes.
    """

    def __init__(self, trades):
        self._trades = list(trades)
        self.metrics = mock.Mock()
        self.metrics.ghost_trade_snapshot.side_effect = self._snapshot
        self.metrics.aggregate_trade_metrics = MetricsCollector.aggregate_trade_metrics.__get__(
            self.metrics, type(self.metrics)
        )
        self.focus_lookback_sec = 172800
        self.min_ghost_trades = 5
        self.min_ghost_win_rate = 0.55
        self.min_realized_margin = 0.0

    def _snapshot(self, *, limit=500, lookback_sec=None, strategy_id=None):
        if strategy_id is None:
            return list(self._trades)
        return [t for t in self._trades if t.strategy_id == str(strategy_id).strip()]


# Staleness is measured against wall-clock; these fixtures use ts=0, so the
# guard is disabled for the gate tests. Every OTHER guardrail stays live.
_ENV = {
    "GHOST_MAX_STALE_SEC": "0",
    "GHOST_REQUIRE_MULTI_SYMBOL_EDGE": "1",
    "GHOST_MAX_SYMBOL_DOMINANCE": "0.82",
}


def _validate(pipeline, strategy_id=None, **env):
    with mock.patch.dict("os.environ", {**_ENV, **env}, clear=False):
        return TrainingPipeline._ghost_validation(pipeline, strategy_id)


class GhostGateIsPerStrategyTest(unittest.TestCase):
    def _production_shaped_book(self):
        """The 2026-09-02 shape: a winner and a loser sharing one book."""
        trades = []
        # atf_static: 35W/3L across five symbols, net +0.5192, and profitable
        # WITHOUT its best symbol so it survives the jackknife.
        for i in range(15):
            trades.append(_trade(0.0338, symbol="BASECAT-USDC", strategy_id="atf_static"))
        trades.append(_trade(-0.0961, symbol="BASECAT-USDC", strategy_id="atf_static"))
        for i in range(11):
            trades.append(_trade(0.0034, symbol="CBBTC-USDC", strategy_id="atf_static"))
        trades.append(_trade(-0.0066, symbol="CBBTC-USDC", strategy_id="atf_static"))
        for i in range(4):
            trades.append(_trade(0.0028, symbol="AERO-USDC", strategy_id="atf_static"))
        for i in range(4):
            trades.append(_trade(0.0010, symbol="CBXRP-USDC", strategy_id="atf_static"))
        trades.append(_trade(-0.0080, symbol="CBXRP-USDC", strategy_id="atf_static"))
        trades.append(_trade(0.00002, symbol="CBETH-USDC", strategy_id="atf_static"))
        # rsi_reversal@5h: 13 losses on one token, one move counted 13 times.
        for i in range(13):
            trades.append(_trade(-0.147, symbol="BASECAT-USDC", strategy_id="rsi_reversal@5h"))
        return trades

    def test_pooled_book_is_dragged_negative_by_the_losing_strategy(self):
        """The bug: pooled, the whole book fails."""
        p = _StubPipeline(self._production_shaped_book())
        verdict = _validate(p)
        self.assertFalse(verdict["ready"], "pooled book should not be ready")
        self.assertLess(
            verdict["total_net_profit"], 0.0,
            "13 losses at -0.147 must outweigh 35 wins at ~+0.02",
        )

    def test_profitable_strategy_is_judged_on_its_own_trades(self):
        """The fix: atf_static's own book passes every guardrail."""
        p = _StubPipeline(self._production_shaped_book())
        verdict = _validate(p, "atf_static")
        self.assertEqual(verdict["strategy_id"], "atf_static")
        self.assertEqual(verdict["samples"], 38)
        self.assertGreater(verdict["total_net_profit"], 0.0)
        self.assertTrue(
            verdict["ready"],
            "atf_static is graduated and net positive on 38 of its own trades; "
            "reason=%r" % verdict["reason"],
        )

    def test_losing_strategy_still_fails_on_its_own_trades(self):
        """Scoping must not launder a bad record."""
        p = _StubPipeline(self._production_shaped_book())
        verdict = _validate(p, "rsi_reversal@5h")
        self.assertFalse(
            verdict["ready"],
            "0 wins in 13 trades must never be ready, scoped or pooled",
        )
        self.assertEqual(verdict["wins"], 0)

    def test_scoping_does_not_invent_trades(self):
        """A strategy with no trades gets a cold-start verdict, not evidence."""
        p = _StubPipeline(self._production_shaped_book())
        verdict = _validate(p, "never_traded")
        self.assertEqual(verdict["samples"], 0)
        self.assertEqual(verdict["total_net_profit"], 0.0)


class JackknifeGuardTest(unittest.TestCase):
    """The edge must survive removing the symbol that produced it.

    `symbol_dominance` counts TRADES. atf_static's real 48h book spread 38
    trades over 5 symbols for a dominance of 0.42 -- inside the 0.82 guard --
    while BASECAT-USDC alone produced +0.5071 of the +0.5192 net, or 97.7%.
    Trade-count concentration reported a diversified book; the P&L was one
    token. So the book must stay net positive with its largest single profit
    contributor removed.
    """

    def _one_token_wonder(self):
        """Many symbols, but every dollar came from one -- and the rest lose.

        Deliberately built to clear every OTHER guardrail, so the only thing
        that can refuse it is the jackknife: win rate 0.60, profit factor 1.5,
        avg profit positive, ES95 0.05 against a 0.10 tail guard, loss rate
        0.40, trade dominance 0.60 against the 0.82 guard, and losses
        interleaved so no streak becomes costly. Net +0.50 overall; net -1.00
        once LUCKY-USDC is removed.
        """
        wins = [_trade(0.05, symbol="LUCKY-USDC", strategy_id="s") for _ in range(30)]
        losses = [_trade(-0.05, symbol="BBB-USDC", strategy_id="s") for _ in range(20)]
        trades = []
        while wins or losses:
            for _ in range(3):
                if wins:
                    trades.append(wins.pop())
            for _ in range(2):
                if losses:
                    trades.append(losses.pop())
        return trades

    def test_single_symbol_profit_is_refused(self):
        p = _StubPipeline(self._one_token_wonder())
        verdict = _validate(p, "s")
        self.assertGreater(verdict["total_net_profit"], 0.0, "book is net positive overall")
        self.assertLessEqual(
            verdict["symbol_dominance"], 0.82,
            "trade-count dominance passes -- that is the hole being closed",
        )
        self.assertLess(verdict["net_profit_ex_top_symbol"], 0.0)
        self.assertTrue(verdict["single_symbol_dependence"])
        self.assertFalse(verdict["ready"])
        self.assertEqual(verdict["reason"], "single_symbol_dependence")

    def test_edge_present_on_more_than_one_symbol_passes(self):
        """A genuinely broad book is untouched by the guard."""
        trades = [_trade(0.02, symbol="AAA-USDC", strategy_id="s") for _ in range(10)]
        trades += [_trade(0.02, symbol="BBB-USDC", strategy_id="s") for _ in range(10)]
        trades += [_trade(-0.01, symbol="CCC-USDC", strategy_id="s") for _ in range(4)]
        p = _StubPipeline(trades)
        verdict = _validate(p, "s")
        self.assertFalse(verdict["single_symbol_dependence"])
        self.assertGreater(verdict["net_profit_ex_top_symbol"], 0.0)
        self.assertTrue(verdict["ready"], "reason=%r" % verdict["reason"])

    def test_guard_can_be_disabled_but_defaults_on(self):
        p = _StubPipeline(self._one_token_wonder())
        off = _validate(p, "s", GHOST_REQUIRE_MULTI_SYMBOL_EDGE="0")
        self.assertFalse(off["single_symbol_dependence"])


class LiveGateCandidateTest(unittest.TestCase):
    """`_ghost_validation_for_live` asks about whoever would actually trade."""

    def test_falls_back_to_pooled_when_graduation_is_not_enforced(self):
        p = _StubPipeline([])
        with mock.patch.dict(
            "os.environ", {"STRATEGY_GRADUATION_ENFORCED": "0"}, clear=False
        ):
            self.assertEqual(TrainingPipeline._live_gate_candidates(p), [])

    def test_untraded_approved_strategy_is_not_treated_as_ready(self):
        """Cold start is an allowance to begin collecting, never evidence.

        Pooled, cold start means the system has no trades at all. Per-strategy
        it would mean every untried strategy reads "ready", so the selector
        requires an EARNED verdict.
        """
        book = [_trade(0.02, symbol="AAA-USDC", strategy_id="proven") for _ in range(10)]
        book += [_trade(0.02, symbol="BBB-USDC", strategy_id="proven") for _ in range(10)]
        p = _StubPipeline(book)
        p._live_gate_candidates = lambda: ["untraded_newcomer"]
        p._ghost_validation = lambda sid=None: TrainingPipeline._ghost_validation(p, sid)
        with mock.patch.dict("os.environ", {**_ENV}, clear=False):
            verdict = TrainingPipeline._ghost_validation_for_live(p)
        self.assertEqual(verdict["samples"], 0)
        self.assertFalse(
            verdict["ready"] and verdict["reason"] not in ("", "fast_track", "positive_expectancy"),
            "a strategy with zero trades must not be selected as live-ready",
        )
        self.assertEqual(verdict["total_net_profit"], 0.0)

    def test_picks_the_approved_strategy_with_the_strongest_earned_record(self):
        book = [_trade(0.02, symbol="AAA-USDC", strategy_id="proven") for _ in range(10)]
        book += [_trade(0.02, symbol="BBB-USDC", strategy_id="proven") for _ in range(10)]
        book += [_trade(-0.05, symbol="AAA-USDC", strategy_id="broken") for _ in range(10)]
        p = _StubPipeline(book)
        p._live_gate_candidates = lambda: ["broken", "proven"]
        p._ghost_validation = lambda sid=None: TrainingPipeline._ghost_validation(p, sid)
        with mock.patch.dict("os.environ", {**_ENV}, clear=False):
            verdict = TrainingPipeline._ghost_validation_for_live(p)
        self.assertEqual(verdict["strategy_id"], "proven")
        self.assertTrue(verdict["ready"])

    def test_reports_a_named_refusal_when_nobody_qualifies(self):
        book = [_trade(-0.05, symbol="AAA-USDC", strategy_id="broken") for _ in range(10)]
        p = _StubPipeline(book)
        p._live_gate_candidates = lambda: ["broken"]
        p._ghost_validation = lambda sid=None: TrainingPipeline._ghost_validation(p, sid)
        with mock.patch.dict("os.environ", {**_ENV}, clear=False):
            verdict = TrainingPipeline._ghost_validation_for_live(p)
        self.assertFalse(verdict["ready"])
        self.assertEqual(
            verdict["strategy_id"], "broken",
            "a block should name the strategy it is blocking",
        )


if __name__ == "__main__":
    unittest.main()
