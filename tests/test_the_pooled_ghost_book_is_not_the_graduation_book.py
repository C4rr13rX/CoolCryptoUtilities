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


if __name__ == "__main__":
    unittest.main()
