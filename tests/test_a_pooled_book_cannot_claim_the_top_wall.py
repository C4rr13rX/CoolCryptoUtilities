"""A structurally blocked strategy outranks the real wall only on POOLED volume.

``classify_wall`` names the ONE thing standing between the ledger and a live
strategy, in the ledger's own order of precedence, and STRUCTURALLY BLOCKED
sits at the top of it. That branch selected its candidate on ``pooled_trades``
-- the whole ghost book, including round trips in symbols the live lane
refuses -- which is the one number this loop has already ruled is not evidence.
The EVIDENCE branch immediately below it says so in as many words: "pooled
ghost volume is not progress".

MEASURED 2026-09-10 on ``atf_static_scout``, the strategy that branch had named
as the system's top wall for at least six consecutive passes:

    ledger ghost book        237 trades, 186 wins, +6.4498
    ledger TRADEABLE subset    4 trades,   1 win,  -0.1097
    rows in trade_outcomes     0            all-time, out of 210 closed rows
    trading_ops naming it    374            over 7 days, so it IS running

Two things follow. The book that outranked every other wall is four tradeable
round trips that LOSE money. And it has no receipts at all: the implausibility
filter, the take-profit clamp and the tradeable predicate in
scripts/tradeable_book.py all read ``trade_outcomes``, and not one of them can
see a single scout round trip, so its +6.4498 cannot be de-contaminated, audited
or reproduced by any instrument in this repo.

Ranking that above a measurable wall costs a pass every time it happens: the
header is the first thing each pass reads, and the standing instruction is to
work the wall it names.

This does NOT weaken the structural bar. ``graduation_blocked`` still stops the
scout from graduating, it is still listed under STRUCTURALLY BLOCKED further
down the report, and a blocked strategy that has genuinely earned TRADEABLE
evidence still takes the top of the order -- ``test_a_blocked_strategy_with_real
_tradeable_evidence_still_wins`` pins exactly that.
"""

from __future__ import annotations

import unittest

from scripts.graduation_status import classify_wall


def _strategy(sid: str, *, ghost_trades: int, pooled_trades: int,
              blocked: bool = False) -> dict:
    return {
        "id": sid,
        "ghost_trades": ghost_trades,
        "pooled_trades": pooled_trades,
        "structurally_blocked": blocked,
        "live_approved": False,
    }


#: The real figures, so a change to either number is visible in the diff.
SCOUT_POOLED = 237
SCOUT_TRADEABLE = 4
MIN_TRADES = 20


class APooledBookCannotClaimTheTopWall(unittest.TestCase):
    def _wall(self, ranked, blocked) -> str:
        return classify_wall(
            approved=(),
            ready_not_approved=(),
            ranked=ranked,
            min_trades=MIN_TRADES,
            blocked=blocked,
            demoted=(),
        )

    def test_the_scouts_unmeasurable_pooled_book_is_not_the_top_wall(self) -> None:
        """The regression itself. Fails against the pooled_trades selector."""
        ranked = [
            _strategy("rsi_reversal", ghost_trades=6, pooled_trades=19),
            _strategy("atf_static_scout", ghost_trades=SCOUT_TRADEABLE,
                      pooled_trades=SCOUT_POOLED, blocked=True),
        ]
        blocked = [{"id": "atf_static_scout", "why": "ghost-only executor"}]

        wall = self._wall(ranked, blocked)

        self.assertNotIn("STRUCTURALLY BLOCKED", wall)
        self.assertIn("EVIDENCE (TRADEABLE)", wall)
        self.assertIn("rsi_reversal", wall, "it names the strategy actually closest")

    def test_a_blocked_strategy_with_real_tradeable_evidence_still_wins(self) -> None:
        """The bar is not removed -- 20 TRADEABLE trips still take the top."""
        ranked = [
            _strategy("rsi_reversal", ghost_trades=6, pooled_trades=19),
            _strategy("atf_static_scout", ghost_trades=MIN_TRADES,
                      pooled_trades=SCOUT_POOLED, blocked=True),
        ]
        blocked = [{"id": "atf_static_scout", "why": "ghost-only executor"}]

        wall = self._wall(ranked, blocked)

        self.assertIn("STRUCTURALLY BLOCKED", wall)
        self.assertIn("atf_static_scout", wall)

    def test_one_trip_below_the_bar_is_still_not_evidence(self) -> None:
        """No off-by-one: the selector is >= min_trades, on the same bar."""
        ranked = [
            _strategy("rsi_reversal", ghost_trades=6, pooled_trades=19),
            _strategy("atf_static_scout", ghost_trades=MIN_TRADES - 1,
                      pooled_trades=SCOUT_POOLED, blocked=True),
        ]
        blocked = [{"id": "atf_static_scout", "why": "ghost-only executor"}]

        self.assertNotIn("STRUCTURALLY BLOCKED", self._wall(ranked, blocked))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
