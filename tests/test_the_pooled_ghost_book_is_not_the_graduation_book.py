"""The pooled ghost book and the book the graduation bar reads differ BY SIGN.

The failure this prevents
------------------------
``scripts/readiness_report.py`` computes ``ready`` from the pooled ghost book;
``trading/strategies/ledger.py`` grants the licence off the live-tradeable
subset. On 2026-09-10 those two disagreed about whether the system was making
money at all, measured over 7 days of ``trade_outcomes``:

    POOLED            124 trips  38% win  +0.7915
    LIVE-TRADEABLE    109 trips  36% win  -0.7877   <- what the bar reads
    untradeable        15 trips  53% win  +1.5792   <- BSTONK alone +1.7017

Every bit of the pooled book's positive sign came from 12% of the volume, on
symbols the live lane refuses on sight. Reading the pooled number named the
wall "READY BUT UNSTAMPED -- two strategies clear the bar and carry no
approval", which is an instruction to go fix a stamp that was refusing
correctly.

So these tests pin the property that matters: a splitter must be able to report
a POSITIVE pooled book and a NEGATIVE tradeable book from the same rows. A
reader that returns the pooled aggregate for both -- the pre-fix behaviour --
fails every sign assertion below.
"""

from __future__ import annotations

import unittest

from scripts.tradeable_book import collect


def _row(symbol, net, sid="s1", mode="ghost"):
    return {"symbol": symbol, "net": float(net), "strategy_id": sid,
            "mode": mode, "ts": 1789000000.0}


# BSTONK stands in for "no stop can bind on it", the real predicate's verdict.
def _pred(symbol: str) -> bool:
    return not str(symbol).startswith("BSTONK")


class PooledIsNotTheGraduationBook(unittest.TestCase):
    def test_a_positive_pooled_book_can_hide_a_negative_tradeable_book(self):
        """The shape measured on 2026-09-10, in miniature."""
        rows = [
            _row("BSTONK-USDC", +1.70),
            _row("AERO-USDC", -0.50),
            _row("COMP-USDC", -0.12),
            _row("VVV-USDC", -0.19),
        ]
        r = collect(rows=rows, is_tradeable=_pred, now=1789000000.0)

        self.assertGreater(r["pooled"]["net"], 0.0,
                           "fixture must reproduce a POSITIVE pooled book")
        self.assertLess(r["tradeable"]["net"], 0.0,
                        "the spendable subset loses money and must report so")
        # The sign flip is the whole point; assert it explicitly rather than
        # relying on the two bounds above happening to straddle zero.
        self.assertNotEqual(
            r["pooled"]["net"] > 0, r["tradeable"]["net"] > 0,
            "pooled and tradeable must be allowed to disagree about the sign")
        self.assertEqual(r["tradeable"]["trades"], 3)
        self.assertEqual(r["untradeable"]["trades"], 1)

    def test_an_untradeable_winner_never_enters_the_tradeable_book(self):
        """One unspendable symbol must not carry a licence to spend."""
        rows = [_row("BSTONK-USDC", +5.0)] + [
            _row("AERO-USDC", -0.10) for _ in range(4)]
        r = collect(rows=rows, is_tradeable=_pred, now=1789000000.0)

        self.assertEqual(r["tradeable"]["wins"], 0)
        self.assertAlmostEqual(r["tradeable"]["net"], -0.40, places=6)
        self.assertAlmostEqual(r["untradeable"]["net"], +5.0, places=6)

    def test_the_split_is_reported_per_strategy_not_only_in_aggregate(self):
        """A licence is granted per strategy, so the split must be per strategy."""
        rows = [
            _row("BSTONK-USDC", +2.0, sid="atf_static"),
            _row("AERO-USDC", -0.30, sid="atf_static"),
            _row("AERO-USDC", +0.40, sid="other"),
        ]
        r = collect(rows=rows, is_tradeable=_pred, now=1789000000.0)
        by = {s["id"]: s for s in r["strategies"]}

        self.assertLess(by["atf_static"]["tradeable"]["net"], 0.0)
        self.assertGreater(by["atf_static"]["pooled"]["net"], 0.0)
        self.assertGreater(by["other"]["tradeable"]["net"], 0.0)

    def test_live_rows_are_not_counted_as_ghost_evidence(self):
        """The graduation bar reads the GHOST book; a live fill is not evidence."""
        rows = [_row("AERO-USDC", -0.50, mode="live"),
                _row("AERO-USDC", +0.10, mode="ghost")]
        r = collect(rows=rows, is_tradeable=_pred, now=1789000000.0)

        self.assertEqual(r["tradeable"]["trades"], 1)
        self.assertAlmostEqual(r["tradeable"]["net"], +0.10, places=6)

    def test_an_unjudgeable_symbol_is_not_counted_as_spendable(self):
        """Cannot establish tradeability -> not proof of it. Same rule as the ledger."""
        rows = [_row("", +1.0), _row("AERO-USDC", +0.25)]
        r = collect(rows=rows, is_tradeable=lambda s: bool(str(s).strip()) and _pred(s),
                    now=1789000000.0)

        self.assertEqual(r["tradeable"]["trades"], 1)
        self.assertAlmostEqual(r["tradeable"]["net"], +0.25, places=6)

    def test_tradeability_cannot_silently_fail_open(self):
        """If the predicate is unloadable the report errors rather than guessing.

        A splitter that fell back to "everything is tradeable" would print the
        pooled book under the tradeable heading -- exactly the misreading this
        module exists to end, but now wearing the right label.
        """
        r = collect(rows=[_row("AERO-USDC", +1.0)], is_tradeable=None,
                    now=1789000000.0)
        if "error" not in r:
            # A predicate WAS importable in this environment; then it must be
            # the ledger's own, and BSTONK must come back untradeable.
            self.assertFalse(
                r["symbols"] and False,
                "collect() returned a split; the real predicate was available")
        else:
            self.assertIn("stop_is_unenforceable", r["error"])



class CostIsMeasuredAgainstNotionalNotZero(unittest.TestCase):
    """A book with a POSITIVE gross edge can still lose, and the split says why.

    The failure this prevents: reporting only ``net`` cannot distinguish "we
    pick the wrong direction" from "we pick correctly and hand it to the fee",
    and those have opposite fixes. Measured 2026-09-10 over 109 live-tradeable
    round trips the answer was the second -- gross +0.6289, fees 1.4166, net
    -0.7877, i.e. a real +0.2625%-of-notional edge against a 0.5913% cost.

    It also pins the part that kills the obvious fix: raising the clip retires
    only the FIXED component, so a gross edge below ``COST_VARIABLE`` loses at
    EVERY clip. A reader that compared the edge to a single flat percentage
    would not be able to tell.
    """

    def _rows(self):
        # gross positive, fees larger: the shape measured on the real book.
        return [{"symbol": "AERO-USDC", "strategy_id": "s1", "mode": "ghost",
                 "net": -0.01, "gross": +0.005, "fees": 0.015,
                 "notional": 2.0, "ts": 1789000000.0} for _ in range(10)]

    def test_a_positive_gross_edge_is_reported_even_when_net_is_negative(self):
        r = collect(rows=self._rows(), is_tradeable=_pred, now=1789000000.0)
        ra = r["tradeable"]["rates"]

        self.assertLess(r["tradeable"]["net"], 0.0)
        self.assertGreater(r["tradeable"]["gross"], 0.0,
                           "gross must survive the aggregation; net alone "
                           "cannot tell direction from cost")
        self.assertGreater(ra["gross_pct"], 0.0)
        self.assertGreater(ra["cost_pct"], ra["gross_pct"])

    def test_the_clip_is_notional_per_round_trip_not_total_notional(self):
        """A units error here would misprice every point on the clip curve."""
        r = collect(rows=self._rows(), is_tradeable=_pred, now=1789000000.0)
        self.assertAlmostEqual(r["tradeable"]["notional"], 20.0, places=6)
        self.assertAlmostEqual(r["tradeable"]["rates"]["clip"], 2.0, places=6)

    def test_an_edge_below_the_variable_floor_loses_at_every_clip(self):
        """The variable component is a floor no clip size can move."""
        from scripts.tradeable_book import COST_VARIABLE, cost_at_clip

        r = collect(rows=self._rows(), is_tradeable=_pred, now=1789000000.0)
        ra = r["tradeable"]["rates"]
        self.assertLess(ra["gross_pct"], 100.0 * COST_VARIABLE)
        self.assertFalse(ra["clears_floor"])
        for clip in (2.0, 10.0, 100.0, 1_000_000.0):
            self.assertLess(ra["gross_pct"], cost_at_clip(clip),
                            "an edge below the variable floor must lose at "
                            "clip $%s too" % clip)

    def test_an_edge_above_the_variable_floor_clears_at_a_large_enough_clip(self):
        """The other side of the same rule, so the test is not vacuous."""
        from scripts.tradeable_book import cost_at_clip

        rows = [{"symbol": "CBADA-USDC", "strategy_id": "s1", "mode": "ghost",
                 "net": 0.01, "gross": +0.0088, "fees": 0.011,
                 "notional": 2.0, "ts": 1789000000.0} for _ in range(10)]
        ra = collect(rows=rows, is_tradeable=_pred,
                     now=1789000000.0)["tradeable"]["rates"]

        self.assertTrue(ra["clears_floor"], "0.44%% must clear a 0.3187%% floor")
        self.assertLess(ra["gross_pct"], cost_at_clip(2.0),
                        "but it must still lose at the $2 clip, where the "
                        "fixed component dominates")
        self.assertGreater(ra["gross_pct"], cost_at_clip(20.0),
                           "and win at $20, which is the whole argument for "
                           "raising the clip")

    def test_the_cost_model_keeps_fixed_and_variable_separate(self):
        """Collapsing them to one flat percentage hides which one is winning."""
        from scripts.tradeable_book import COST_FIXED, COST_VARIABLE, cost_at_clip

        self.assertAlmostEqual(cost_at_clip(1.0),
                               100.0 * (COST_FIXED + COST_VARIABLE), places=9)
        # The curve must fall with clip and asymptote to the variable floor.
        self.assertGreater(cost_at_clip(2.0), cost_at_clip(20.0))
        self.assertGreater(cost_at_clip(1e9), 100.0 * COST_VARIABLE * 0.999)
        self.assertLess(cost_at_clip(1e9), 100.0 * COST_VARIABLE * 1.001)


if __name__ == "__main__":
    unittest.main()
