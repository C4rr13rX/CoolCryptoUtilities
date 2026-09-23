"""``reconstruct()`` counted round trips the live lane would have refused.

``services/tradeable_evidence.reconstruct`` says in its own docstring that it
applies "the ledger's own three filters, in the order ``record()`` applies
them, so a row counted here is a row the ledger would have counted", and
``services/live_gate_map.py:94`` builds on that claim in capitals: "THE
POPULATION IS THE ONE GRADUATION USES."

Neither was true. ``record()`` asks ``_live_tradeable(symbol, sid)`` --
trading/strategies/ledger.py:798, and its comment says why: "Omitting it would
keep the pooled ban and silently drop the pair ban, which is where
atf_static/AERO-USDC lives." ``reconstruct()`` asked ``_live_tradeable(symbol)``
with no strategy at all, so ``symbol_edge_gate.refusal_reason`` was called with
``strategy_id=None`` and only the POOLED half of the ban could ever fire.

The two halves genuinely disagree, and they disagree on the pair the live lane
is aimed at most. Driving the real gate on 2026-09-23 while measuring this:

    SYMBOL EDGE GATE: refusing atf_static on AERO-USDC  -- 17 round trips at
        mean return -0.470% vs 0.772% cost (t=-4.05)
    SYMBOL EDGE GATE: refusing atf_static on CBBTC-USDC -- 6 round trips at
        mean return -0.193% vs 0.655% cost (t=-2.90)

Both are PAIR refusals -- AERO-USDC and CBBTC-USDC are allowed pooled. Every
atf_static round trip on them was being counted as proof that real money could
have been placed there, by the one module whose entire job is to answer that
question.

The effect is one-directional, which is what makes it dangerous rather than
merely wrong: dropping a ban can only ADMIT rows, never refuse them, so every
count reconstruct produced was inflated in the direction of "this strategy has
earned a licence".

These tests drive the REAL ``_live_tradeable``. Stubbing it out -- which the
older reconstruct tests do, reasonably, for their own subject -- is precisely
what would hide this bug, because the predicate under test is the argument list
that reaches it.
"""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from services import tradeable_evidence

#: The pair the ledger's own comment names as the one that motivated passing
#: the strategy id. Allowed pooled, refused for this executor.
BANNED_PAIR = ("AERO-USDC", "atf_static")


def _make_db(path: Path, rows) -> None:
    """A ``trading_ops`` table holding exactly the given ghost exits."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trading_ops (id INTEGER PRIMARY KEY, ts REAL, wallet TEXT, "
        "chain TEXT, symbol TEXT, action TEXT, status TEXT, details TEXT)"
    )
    for i, row in enumerate(rows):
        conn.execute(
            "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                float(i),
                "ghost",
                "base",
                row.get("symbol", ""),
                "exit",
                "ghost-exit",
                json.dumps(row),
            ),
        )
    conn.commit()
    conn.close()


def _exit(strategy_id, symbol, profit, *, held_sec=60.0):
    return {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "profit": profit,
        # Stamped so ``_profit_usd`` trusts the number as written rather than
        # reconverting it from prices this fixture does not carry. The unit is
        # not what these tests are about.
        "profit_unit": "usd",
        "age_sec": held_sec,
        "exit_reason": "target_hit",
    }


def _pair_only_gate(symbol, strategy_id=None):
    """A gate that refuses ONE (symbol, strategy) pair and nothing pooled.

    This is the shape that exposes the defect. A caller that passes the
    strategy is refused; a caller that omits it is waved through, because with
    ``strategy_id=None`` there is no pair to condemn. The real gate behaves
    the same way by construction -- ``refusal_reason``'s docstring says a call
    site that omits the id "can only ever be more permissive, never less".
    """
    sym, sid = BANNED_PAIR
    if str(symbol) == sym and str(strategy_id or "") == sid:
        return f"{sid}: 17 closed round trips at mean return -0.470% vs 0.772% cost"
    return None


class ReconstructAppliesThePairBan(unittest.TestCase):
    """A row the live lane would have refused is not evidence that it would have paid."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.db = self.tmp / "ops.db"

        # Patch BELOW `_live_tradeable`, never `_live_tradeable` itself: the
        # argument list that reaches the gate is the subject of these tests.
        patches = [
            mock.patch.object(tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)),
            mock.patch("trading.pipeline.stop_is_unenforceable", return_value=False),
            mock.patch("services.symbol_edge_gate.refusal_reason", _pair_only_gate),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_a_pair_refused_round_trip_is_not_counted_as_tradeable(self):
        """Three refused round trips and two allowed ones must reconstruct as two.

        Against the shipped code this counts FIVE: the pair ban never fires
        because the strategy never reaches the gate, so the three AERO-USDC
        rows are admitted as round trips "the live lane could have placed" on
        a pair it refuses outright.
        """
        sym, sid = BANNED_PAIR
        rows = [_exit(sid, sym, 0.25) for _ in range(3)]
        rows += [_exit(sid, "TOAD-USDC", -0.01) for _ in range(2)]
        _make_db(self.db, rows)

        found = tradeable_evidence.reconstruct(self.db)

        ev = found[sid]
        self.assertEqual(ev.exits, 5, "all five rows must be SEEN")
        self.assertEqual(ev.trades, 2, "only the two allowed-pair rows are evidence")
        self.assertEqual(ev.dropped_untradeable, 3)

    def test_the_refused_rows_do_not_carry_the_books_sign(self):
        """The direction of the error: dropping the ban inflates the verdict.

        The refused rows are the winners here, which is not a contrivance --
        it is the measured shape. 68% of reconstruct's entire positive sign on
        the real database (+9.93 of +14.60) was twelve rows in BSTONK-USDC and
        BPAD-USDC. Counting refused rows does not blur a book, it flips it.
        """
        sym, sid = BANNED_PAIR
        rows = [_exit(sid, sym, 0.25) for _ in range(3)]
        rows += [_exit(sid, "TOAD-USDC", -0.01) for _ in range(2)]
        _make_db(self.db, rows)

        ev = tradeable_evidence.reconstruct(self.db)[sid]

        self.assertAlmostEqual(ev.net, -0.02, places=9)
        self.assertEqual(ev.wins, 0)
        self.assertEqual(ev.losses, 2)

    def test_the_pooled_ban_still_refuses_every_strategy(self):
        """Adding the strategy must not cost the pooled half of the ban.

        A pooled refusal names no executor, so it has to refuse one that is
        passed as surely as one that is not. Without this, "fixing" the pair
        ban by threading an id through could quietly narrow the pooled ban to
        nothing and no test would notice.
        """
        def _pooled_gate(symbol, strategy_id=None):
            return "pooled: the book condemns it" if str(symbol) == "COMP-USDC" else None

        with mock.patch("services.symbol_edge_gate.refusal_reason", _pooled_gate):
            rows = [_exit("rsi_reversal", "COMP-USDC", 0.4) for _ in range(4)]
            rows += [_exit("rsi_reversal", "TOAD-USDC", -0.01)]
            _make_db(self.db, rows)

            ev = tradeable_evidence.reconstruct(self.db)["rsi_reversal"]

        self.assertEqual(ev.trades, 1)
        self.assertEqual(ev.dropped_untradeable, 4)

    def test_two_strategies_on_one_symbol_get_two_answers(self):
        """The memo must not answer for the wrong executor.

        ``reconstruct`` memoises the predicate because it is expensive. Keyed
        on the symbol alone that cache is now actively wrong: whichever
        strategy reaches AERO-USDC first decides it for every other strategy
        in the file. This is the regression a naive fix introduces, so it is
        pinned rather than left to chance.
        """
        sym, banned_sid = BANNED_PAIR
        rows = [_exit(banned_sid, sym, 0.25) for _ in range(3)]
        rows += [_exit("money_button", sym, 0.10) for _ in range(2)]
        _make_db(self.db, rows)

        found = tradeable_evidence.reconstruct(self.db)

        self.assertEqual(found[banned_sid].trades, 0, "refused for this executor")
        self.assertEqual(found["money_button"].trades, 2, "allowed for that one")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
