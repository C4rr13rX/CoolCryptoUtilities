"""A ghost-exit standing on a real on-chain buy is not ghost evidence.

Measured 2026-09-04. Four ``trading_ops`` rows carry ``status='ghost-exit'``,
``wallet='ghost'``, ``fill_source='simulated'``, an EMPTY ``tx_hash`` -- and a
66-character ``entry_tx_hash``. They were written on 09-03 between 17:40 and
20:38 UTC, before ``live_position_cannot_exit_in_simulation`` closed the writer
(23593e2, 182bd8a); no row written since carries one.

Each is a position bought with real money and then "closed" by marking it
against the feed price. No swap was broadcast, so the tokens stayed in the
wallet and the recorded profit is for a sale that never happened. The largest:

    entry 0xcd6fb05c92af5077f9be707727c1d57e0ac1dfedd54d9f87e860376b96ea560b
    bought 360.264243225392976659 BSTONK for 0.750000 USDC at 17:02:15Z,
    booked -0.142865 at 19:12:15Z with tx_hash="",

and ``balanceOf`` on base still returns exactly 360264243225392976659 -- not
one wei was sold.

Scored as ghost trades these put atf_static's ES95 at 0.10213 against a 0.10
guardrail, which is ``ghost_validation_block`` -- the live gate refusing real
money over a loss that never occurred. Excluding them, the same 26 genuine
ghost round trips read ES95 0.01723, PF 4.726, net +0.6210.

The same four were annulled in ``trade_outcomes`` by
scripts/annul_unsettled_live_exits.py. The live gate does not read that table;
it reads ``trading_ops`` through ``ghost_trade_snapshot``, so the correction had
reached only one of the two books. This pins the reader-side half.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.metrics import MetricsCollector, distribution_report


REAL_ENTRY_TX = "0xcd6fb05c92af5077f9be707727c1d57e0ac1dfedd54d9f87e860376b96ea560b"


class _DB:
    def __init__(self, rows):
        self._rows = rows

    def fetch_trades(self, **_kwargs):
        # Mirrors the real ORDER BY ts DESC.
        return sorted(self._rows, key=lambda r: -float(r["ts"]))


def _exit_row(ts, symbol, entry_price, exit_price, profit, *, entry_tx=""):
    details = {
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "profit": profit,
        "entry_ts": ts - 100.0,
        "exit_ts": ts,
        "exit_reason": "stop_loss",
        "strategy_id": "atf_static",
        "fill_source": "simulated" if entry_tx else "ghost",
        "tx_hash": "",
    }
    if entry_tx:
        details["entry_tx_hash"] = entry_tx
    return {"ts": ts, "status": "ghost-exit", "wallet": "ghost",
            "symbol": symbol, "details": details}


class RealBuyIsNotGhostEvidence(unittest.TestCase):
    def test_exit_backed_by_a_real_entry_tx_is_refused(self):
        rows = [
            _exit_row(100.0, "AAA-USDC", 1.0, 1.02, +0.02),
            _exit_row(200.0, "BSTONK-USDC", 0.0020818052696136586,
                      0.0016987795506045683, -0.142865471,
                      entry_tx=REAL_ENTRY_TX),
        ]
        trades = MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)
        symbols = [t.symbol for t in trades]
        self.assertEqual(symbols, ["AAA-USDC"])
        self.assertNotIn("BSTONK-USDC", symbols)

    def test_a_genuine_ghost_exit_is_still_counted(self):
        """The rule must not quietly shrink the honest book."""
        rows = [_exit_row(100.0, "AAA-USDC", 1.0, 0.95, -0.05)]
        trades = MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].profit, -0.05)

    def test_the_rule_is_profit_blind(self):
        """It drops a WIN on the same terms as a loss.

        The four real rows are one win (+0.000267) and three losses. A rule
        that only removed losers would be fitting the gate to the outcome.
        """
        rows = [
            _exit_row(100.0, "CBBTC-USDC", 1.0, 1.01, +0.000267,
                      entry_tx=REAL_ENTRY_TX),
            _exit_row(200.0, "CBETH-USDC", 1.0, 0.99, -0.003011,
                      entry_tx=REAL_ENTRY_TX),
        ]
        trades = MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)
        self.assertEqual(trades, [])

    def test_a_short_or_absent_hash_is_not_treated_as_a_real_buy(self):
        """Only a full 66-character hash is proof money moved.

        A truncated or placeholder value must not silently delete honest ghost
        evidence -- the failure direction here is to KEEP the trade.
        """
        for bogus in ("", "0x", "0xcd6fb05c92af5077f9be7077", "none"):
            rows = [_exit_row(100.0, "AAA-USDC", 1.0, 0.9, -0.1,
                              entry_tx=bogus)]
            trades = MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)
            self.assertEqual(len(trades), 1, "dropped on entry_tx=%r" % bogus)

    def test_the_bstonk_fiction_is_what_moved_the_tail_over_its_guard(self):
        """Reproduces the measured gate flip from the real numbers.

        The 25 honest atf_static returns plus the BSTONK fiction score above
        the 0.10 guardrail; the honest 25 alone score far below it.
        """
        honest = [-0.0203, -0.0142, -0.0077, -0.0025, -0.0020, -0.0012,
                  -0.0004] + [0.004] * 18
        with_fiction = honest + [-0.18399]
        es_honest = abs(
            distribution_report(honest)["expected_shortfall_95"])
        es_fiction = abs(
            distribution_report(with_fiction)["expected_shortfall_95"])
        self.assertGreater(es_fiction, 0.10, "the fiction must breach the guard")
        self.assertLess(es_honest, 0.10, "the honest book must clear it")


if __name__ == "__main__":
    unittest.main()
