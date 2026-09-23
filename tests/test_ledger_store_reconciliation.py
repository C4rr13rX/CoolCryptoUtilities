"""The graduation bar reads a STORED COUNTER, and nothing reconciles it.

``scripts/graduation_status.py:219`` asks ``ledger._tradeable_of(ghost)``, and
``trading/strategies/ledger.py`` implements that as ``stats.get("tradeable")``
-- a sub-dict accumulated as trades close. It is never recomputed from
``trade_outcomes``. So the number that decides whether a strategy may spend
real money is a second copy of a book, and the repo's own docstring at
``scripts/tradeable_book.py:73`` records what happens to second copies: they
drift, and both sides go on printing.

Measured at HEAD 2026-09-23, ledger ``ghost.tradeable`` summed per strategy
against ``trade_outcomes`` closed rows filtered through the SAME predicate the
bar delegates to (``ledger._live_tradeable``, imported -- never reimplemented):

    LEDGER pooled   116 trades   30 wins   -2.4145   <- what graduation reads
    STORE  pooled   182 trades   60 wins   +6.1425
    gap            +66 trades             +8.5570

THE HEADLINE GAP IS MOSTLY AN ARTIFACT AND THIS FILE REFUSES TO REPEAT IT.
A store row with no ``strategy_id`` is bucketed under the literal string
``"unclassified"``, and ``unclassified`` is ALSO a real ledger strategy id with
its own record. The two collide: LEDGER 5/0/-0.0326 against STORE 60/30/+5.6245,
which is +5.66 of the +8.56. Excluding that one collided bucket, the honest
named-strategy gap is **+2.90 over +11 trades** -- which reproduces the +2.99
measured independently on pass 681. ``test_unclassified_bucket_is_a_name_collision``
pins the collision so nobody quotes the pooled figure as drift again.

WHY THIS IS A TEST AND NOT A REPAIR. Pass 681 established, and this file
enforces, that the two books index DIFFERENT POPULATIONS: a ghost-only executor
has no live branch, so its ledger record can exist against a store that never
received it. ``atf_static_scout`` is the case that matters -- 29 tradeable
ledger trades at 65% win, the record the briefing ranks CLOSEST TO THE BAR. A
recompute-from-store run on the assumption that the store is truth would delete
it. That is the accident these tests exist to make loud, and it is why there is
no assertion here that the two books agree: they do not, and asserting that they
should would invite exactly the repair that destroys evidence.

The direction has MOVED since pass 681 and both sides of the move are recorded
here so the next pass starts from the measurement rather than the note:

  * Pass 681 found "any strategy_id containing 'scout': NONE" and closed the
    drift hypothesis on it. At HEAD the scout HAS six store rows
    (6/4/-0.0228). The conclusion drawn from that census is stale.
  * Pass 681 measured 9 strategies behind / 10 ahead and called the
    disagreement bidirectional. At HEAD it is 21 behind / 6 ahead / 6 equal /
    4 store-zero. Twelve of the 21 hold a ledger record of ZERO trades against
    1-4 store rows, so most of the one-directionality is an unwritten counter,
    not an arithmetic disagreement.

These tests are characterisation tests. They are green at HEAD, they do not
assert an aspiration, and each one fails on a specific FUTURE event worth
stopping for.
"""

from __future__ import annotations

import json
import os
import sqlite3
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "strategy_ledger.json"
DB_PATH = Path(os.getenv("TRADING_DB_PATH") or (ROOT / "storage" / "trading_cache.db"))

#: The store bucket that is NOT a strategy. Rows predating the 2026-09-03 start
#: of strategy attribution carry no ``strategy_id`` and land here; the ledger's
#: own ``unclassified`` entry is a different thing wearing the same name.
COLLIDED_BUCKET = "unclassified"

#: Named strategies holding a ledger tradeable record against ZERO store rows.
#:
#: THE FIRST VERSION OF THIS COMMENT SAID THE STORE NEVER RECEIVED THEM. THAT
#: WAS WRONG AND THE CORRECTION IS THE POINT OF THIS FILE. The store holds every
#: one of these rows. ``_live_tradeable`` removes them, and the arm doing the
#: removing is ``services.symbol_edge_gate.refusal_reason`` -- which computes its
#: verdict FROM the closed round trips it is filtering. Split over all 284 closed
#: rows at HEAD 2026-09-23:
#:
#:      refused by the STOP gate            2
#:      refused by the SYMBOL EDGE GATE   100   <- 35.2%, and it is in-sample
#:      tradeable                         182
#:
#: Per row, and every count matches the ledger EXACTLY -- not one missing row:
#:
#:      rsi_reversal@5d       DRB-USDC   edge-refused 3   (ledger tradeable 3)
#:      bollinger_squeeze     OP-USDC    edge-refused 1
#:      bollinger_squeeze     VVV-USDC   edge-refused 1   (ledger tradeable 2)
#:      donchian_breakout     OP-USDC    edge-refused 1   (ledger tradeable 1)
#:      donchian_breakout@3d  OP-USDC    edge-refused 1   (ledger tradeable 1)
#:
#: So these are not a dead write path and not a second undeclared ghost-only
#: executor. The ledger is the predicate AT WRITE TIME; this recomputation is the
#: predicate AT READ TIME; the gate has since learned to refuse those symbols.
#: The set may shrink or grow as the gate learns, which is why the assertion
#: below names the edge gate in its failure message rather than blaming a writer.
KNOWN_STORE_ZERO = {
    "rsi_reversal@5d",          # ledger 3/0/-0.0760, all 3 edge-refused
    "bollinger_squeeze",        # ledger 2/1/-0.0307, both edge-refused
    "donchian_breakout",        # ledger 1/0/-0.0268, edge-refused
    "donchian_breakout@3d",     # ledger 1/0/-0.0142, edge-refused
}


def _tradeable_predicate():
    """The ledger's own tradeability test. Imported, never reimplemented.

    ``scripts/tradeable_book.py:73`` documents why at length: a second copy of
    this predicate drifted from the ledger's and the report went on printing
    symbols graduation had stopped counting. Returning ``None`` rather than a
    fail-open lambda is the same deliberate choice made there -- a run that
    silently counted every symbol as tradeable would manufacture the very
    disagreement this file measures.
    """
    try:
        from trading.strategies.ledger import _live_tradeable
    except Exception:  # noqa: BLE001
        return None

    def _ok(symbol: str, strategy_id: str) -> bool:
        try:
            return bool(_live_tradeable(symbol, strategy_id))
        except TypeError:
            return bool(_live_tradeable(symbol))
        except Exception:  # noqa: BLE001
            return False

    return _ok


def _ledger_book() -> Dict[str, Tuple[int, int, float]]:
    """Per strategy, the ``ghost.tradeable`` sub-dict the graduation bar reads."""
    raw = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Tuple[int, int, float]] = {}
    for sid, ent in (raw or {}).items():
        ghost = (ent or {}).get("ghost")
        sub = ghost.get("tradeable") if isinstance(ghost, dict) else None
        if not isinstance(sub, dict):
            continue
        out[str(sid)] = (
            int(sub.get("trades", 0) or 0),
            int(sub.get("wins", 0) or 0),
            float(sub.get("total_profit", 0.0) or 0.0),
        )
    return out


def _store_book(pred) -> Dict[str, Tuple[int, int, float]]:
    """Per strategy, the same book recomputed from ``trade_outcomes``.

    ``annulled`` rows are dropped for the reason ``scripts/tradeable_book.py``
    gives: an annulled outcome is one the system has already decided did not
    happen, and counting it puts a retracted trade back into the evidence.
    """
    con = sqlite3.connect("file:%s?mode=ro" % DB_PATH.as_posix(), uri=True)
    con.row_factory = sqlite3.Row
    out: Dict[str, Dict[str, Any]] = {}
    try:
        cur = con.execute(
            "SELECT symbol, status, net_profit, details FROM trade_outcomes"
        )
        for r in cur:
            if str(r["status"] or "").lower() != "closed":
                continue
            try:
                det = json.loads(r["details"] or "{}")
            except Exception:  # noqa: BLE001
                det = {}
            if not isinstance(det, dict):
                det = {}
            if str(det.get("annulled") or "").lower() in ("1", "true", "yes"):
                continue
            sid = str(det.get("strategy_id") or "") or COLLIDED_BUCKET
            sym = str(r["symbol"] or "")
            if not pred(sym, sid):
                continue
            acc = out.setdefault(sid, {"t": 0, "w": 0, "p": 0.0})
            net = float(r["net_profit"] or 0.0)
            acc["t"] += 1
            acc["w"] += 1 if net > 0 else 0
            acc["p"] += net
    finally:
        con.close()
    return {k: (v["t"], v["w"], v["p"]) for k, v in out.items()}


class LedgerStoreReconciliation(unittest.TestCase):
    """Reconciliation facts that must stay true, or stop the pass if they move."""

    @classmethod
    def setUpClass(cls) -> None:
        if not LEDGER_PATH.exists():
            raise unittest.SkipTest("no strategy ledger at %s" % LEDGER_PATH)
        if not DB_PATH.exists() or DB_PATH.stat().st_size < 1024:
            # data/trading.db and web/db.sqlite3 are 0-byte stubs that return
            # nothing silently. Skipping beats passing on an empty store.
            raise unittest.SkipTest("no populated trade store at %s" % DB_PATH)
        pred = _tradeable_predicate()
        if pred is None:
            raise unittest.SkipTest("cannot import ledger._live_tradeable")
        cls.ledger = _ledger_book()
        cls.store = _store_book(pred)

    def test_the_bar_reads_a_stored_counter_not_a_query(self) -> None:
        """``_tradeable_of`` returns a sub-dict. It does not query anything.

        This is the whole defect in one assertion. If a future pass makes the
        bar recompute, this test fails and forces whoever did it to confront
        the population question below rather than discovering it afterwards.
        """
        from trading.strategies import ledger as led

        import inspect

        src = inspect.getsource(led._tradeable_of)
        self.assertIn("get(\"tradeable\")", src.replace("'", '"'))
        self.assertNotIn("trade_outcomes", src)

    def test_no_recompute_path_may_ignore_the_ghost_only_exemption(self) -> None:
        """If the ledger ever reads the store, it must know who is exempt.

        Pass 681's warning, encoded: a recompute-from-store on the assumption
        that the store is truth would delete ``atf_static_scout``'s 29-trade,
        65%-win record -- the strongest record in the graduation table -- because
        the scout is a ghost-only executor whose rows the store largely never
        received. Today ``ledger.py`` reads ``trade_outcomes`` nowhere on the
        counter path; the day it does, ``_ghost_only_ids`` must be in the same
        module or this fires.
        """
        src = (ROOT / "trading" / "strategies" / "ledger.py").read_text(
            encoding="utf-8", errors="replace"
        )
        if "trade_outcomes" in src:
            self.assertIn(
                "_ghost_only_ids",
                src,
                "ledger.py now reads trade_outcomes but never consults the "
                "ghost-only exemption; a blind recompute deletes evidence",
            )

    def test_unclassified_bucket_is_a_name_collision_not_drift(self) -> None:
        """The pooled gap is mostly one bucket meaning two things.

        A store row with no ``strategy_id`` is bucketed as ``unclassified``;
        the ledger's ``unclassified`` is a real strategy. Measured at HEAD the
        store side of that bucket is an order of magnitude larger than the
        ledger side, and it carries +5.62 of the +6.14 pooled store profit.
        Any reconciliation that compares these two is comparing names, not
        books.
        """
        led_t = self.ledger.get(COLLIDED_BUCKET, (0, 0, 0.0))[0]
        sto_t = self.store.get(COLLIDED_BUCKET, (0, 0, 0.0))[0]
        self.assertGreater(
            sto_t,
            led_t * 3,
            "the unclassified collision has closed (store %d vs ledger %d); "
            "re-derive the honest gap before quoting either book" % (sto_t, led_t),
        )

    def test_named_gap_is_reported_without_the_collided_bucket(self) -> None:
        """The honest drift is the NAMED gap, and it is not zero.

        +2.90 over +11 trades at HEAD, against +2.99 measured independently on
        pass 681. Asserting it is non-zero is asserting the defect is still
        open: if it ever closes, this fails and the item can be marked done on
        a measurement rather than on a narrative.
        """
        named = [k for k in set(self.ledger) | set(self.store) if k != COLLIDED_BUCKET]
        lp = sum(self.ledger.get(k, (0, 0, 0.0))[2] for k in named)
        sp = sum(self.store.get(k, (0, 0, 0.0))[2] for k in named)
        self.assertGreater(
            abs(sp - lp),
            0.01,
            "ledger and store now agree on named strategies (%.4f vs %.4f); "
            "[28131071] may be closeable -- verify before assuming" % (lp, sp),
        )

    def test_store_zero_strategies_are_declared_and_do_not_grow(self) -> None:
        """A ledger record against an empty store must be a KNOWN fact.

        Every such strategy is a record a recompute would delete. The set may
        shrink -- a strategy whose rows arrive stops being store-zero and that
        is progress. A NEW one means a write path died or an undeclared
        ghost-only executor appeared, and either is worth a pass.
        """
        try:
            from trading.strategies.ledger import _ghost_only_ids

            declared = set(_ghost_only_ids())
        except Exception:  # noqa: BLE001
            declared = set()
        allowed = declared | KNOWN_STORE_ZERO | {COLLIDED_BUCKET}
        actual = {
            sid
            for sid, (t, _w, _p) in self.ledger.items()
            if t > 0 and self.store.get(sid, (0, 0, 0.0))[0] == 0
        }
        self.assertFalse(
            actual - allowed,
            "strategies hold ledger evidence this recomputation cannot see: %s "
            "-- CHECK services.symbol_edge_gate.refusal_reason FIRST. Every one "
            "of the four known cases is an edge-gate refusal, not a missing "
            "write, and the gate's verdict moves as it learns. Only after "
            "ruling the gate out is this a dead writer."
            % sorted(actual - allowed),
        )

    def test_the_store_book_is_an_in_sample_selection(self) -> None:
        """The predicate is 98% one gate, and that gate reads this same book.

        ``_live_tradeable`` has two arms. The stop arm removes 2 of 284 closed
        rows; ``services.symbol_edge_gate.refusal_reason`` removes 100 -- 35.2%.
        That gate decides by measuring the closed round trips it is filtering
        ("refusing atf_static on AERO-USDC -- 17 closed round trips at mean
        return -0.470% vs 0.772% cost, t=-4.05"), so the store-side "tradeable
        book" is a function of WHEN it is asked, not a quantity. Two agents
        measured 182 and 177 minutes apart on pass 687 for exactly this reason.

        This asserts the shape, not the value: the edge arm must dominate the
        stop arm. If that ever inverts, the in-sample problem has moved and
        every conclusion drawn from a "store book" number needs re-deriving.
        """
        try:
            from services.symbol_edge_gate import refusal_reason
            from trading.pipeline import stop_is_unenforceable
        except Exception:  # noqa: BLE001
            self.skipTest("cannot import both arms of the tradeable predicate")
        con = sqlite3.connect("file:%s?mode=ro" % DB_PATH.as_posix(), uri=True)
        con.row_factory = sqlite3.Row
        stop_refused = edge_refused = 0
        try:
            for r in con.execute(
                "SELECT symbol, status, details FROM trade_outcomes"
            ):
                if str(r["status"] or "").lower() != "closed":
                    continue
                try:
                    det = json.loads(r["details"] or "{}")
                except Exception:  # noqa: BLE001
                    det = {}
                if not isinstance(det, dict):
                    det = {}
                sym = str(r["symbol"] or "")
                sid = str(det.get("strategy_id") or "") or COLLIDED_BUCKET
                try:
                    if bool(stop_is_unenforceable(sym)):
                        stop_refused += 1
                        continue
                except Exception:  # noqa: BLE001
                    stop_refused += 1
                    continue
                try:
                    if refusal_reason(sym, sid) is not None:
                        edge_refused += 1
                except Exception:  # noqa: BLE001
                    pass
        finally:
            con.close()
        self.assertGreater(
            edge_refused,
            stop_refused,
            "the in-sample edge gate no longer dominates the stop gate "
            "(edge %d vs stop %d); re-derive every store-book number that "
            "assumed it did" % (edge_refused, stop_refused),
        )

    def test_the_store_books_positive_sign_is_one_row(self) -> None:
        """Its SUM is positive and its MEDIAN ROW LOSES MONEY. Quote the median.

        Measured at HEAD 2026-09-23 over the 182 tradeable closed rows:

            net        +6.1425
            MEDIAN     -0.009095      <- the typical row loses
            top 1 row  +3.4700  =  56% of the entire sum
            top 5 rows +8.9786  = 146% of the entire sum
            net without the single top row  +2.6725

        The top two are the SAME DAY, 2026-08-26, both carrying no strategy_id:
        AAVE-USDC +3.4700 and AERO-USDC +3.2066, together +6.68 against a book
        total of +6.14. The other 180 rows sum negative.

        This is the shape this repo has shipped three times -- "98% of gross was
        7 rows", "six rows carry 236% of the ghost book and four are one day",
        "the ghost AERO positive is one tape-uncorroborated row". A sum can be
        carried by one row; a median cannot, which is why this asserts on the
        median and not the sum.

        It fails the day the median turns positive -- which is GOOD NEWS and
        still worth stopping for, because it would mean the book's central
        tendency changed rather than its tail.
        """
        con = sqlite3.connect("file:%s?mode=ro" % DB_PATH.as_posix(), uri=True)
        con.row_factory = sqlite3.Row
        nets = []
        try:
            for r in con.execute(
                "SELECT symbol, status, net_profit, details FROM trade_outcomes"
            ):
                if str(r["status"] or "").lower() != "closed":
                    continue
                try:
                    det = json.loads(r["details"] or "{}")
                except Exception:  # noqa: BLE001
                    det = {}
                if not isinstance(det, dict):
                    det = {}
                sid = str(det.get("strategy_id") or "") or COLLIDED_BUCKET
                if not _tradeable_predicate()(str(r["symbol"] or ""), sid):
                    continue
                nets.append(float(r["net_profit"] or 0.0))
        finally:
            con.close()
        self.assertGreater(len(nets), 50, "too few rows to judge concentration")
        nets.sort()
        median = nets[len(nets) // 2]
        total = sum(nets)
        top = nets[-1]
        self.assertLess(
            median,
            0.0,
            "the tradeable book's median row is now POSITIVE (%.6f over %d "
            "rows, sum %+.4f) -- re-derive before quoting, the central "
            "tendency has moved" % (median, len(nets), total),
        )
        if total > 0:
            self.assertGreater(
                top / total,
                0.25,
                "the book's positive sum is no longer carried by its top row "
                "(top %+.4f of %+.4f) -- the concentration this file warns "
                "about has changed and the warning needs re-measuring"
                % (top, total),
            )

    def test_the_scout_is_in_the_store_now(self) -> None:
        """Pass 681's census -- "any id containing 'scout': NONE" -- is stale.

        It is the fact criterion 1 of [28131071] was closed on. Six closed
        tradeable store rows carry the literal id at HEAD. The scout is still
        structurally blocked from live and its ledger record is still 23 trades
        larger than the store's, so the exemption stands; what does NOT stand
        is "the store has nothing under this name", and a future pass must not
        re-derive a conclusion from it.
        """
        scout = self.store.get("atf_static_scout", (0, 0, 0.0))
        self.assertGreater(
            scout[0],
            0,
            "the store has no atf_static_scout rows again -- pass 681's census "
            "may have been right and this file's premise needs re-measuring",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
