"""Recover every ghost take-profit row's OWN target and measure the overshoot.

WHY THIS EXISTS. 5504769 stopped a ghost limit exit booking the tick that
tripped it (``trading.bot.limit_exit_fill_price``: a limit order fills at the
limit, not at wherever the next sample landed). That fix is forward-only, so
[c4f16946] asked for the already-written rows to be annulled out of
``trade_outcomes`` AND ``data/strategy_ledger.json`` together -- the ledger
being the book ``_evaluate_graduation_locked`` and ``_maybe_rearm_locked``
actually read.

THE ROW SET THAT ITEM NAMED IS WRONG, AND THIS SCRIPT IS HOW THAT WAS FOUND.
Every previous attempt identified a "gap fill" by its RETURN (exit/entry),
because the outcome row does not store the target it was aiming at. A return
is the wrong instrument: a +25% round trip is a gap fill only if its target
was near the entry, and is an ordinary limit fill if the strategy set a +23%
target in the first place.

The target IS recoverable, exactly. ``trade_outcomes.trade_id`` ends in the
position hash, and the ``action='enter'`` row in ``trading_ops`` for that same
hash carries ``details.target_price``. Joining on the hash -- not on a price
tolerance, which silently matches the wrong entry when a symbol is re-entered
at a similar price -- gives each closed row the number it was actually aiming
at. Measured 2026-09-10 over all 28 closed ghost take-profit rows, 26 of which
have a recoverable target:

    exit/target > 1.10          5 rows   AERO 2.486, AAVE 2.611,
                                         BASELINE 1.466, BASECAT 1.105,
                                         rsi_reversal@5h BSTONK 1.203
    atf_static's two "gap fills" 1.019 and 1.074  -- INSIDE 1.10x

So the +1.0201 that [c4f16946] calls "66% of atf_static's book fabricated" is
two rows whose fills sat 1.9% and 7.4% past their own limits. Their large
RETURNS (+25.35%, +17.28%) come from targets set 23.0% and 9.2% above entry,
which is the strategy choosing a wide target, not the feed skipping past a
narrow one. Annulling them on the return-based premise would strike real
evidence and would leave the two rows that ARE 2.5x their target (AERO, AAVE,
both unattributed) standing.

WHAT THIS SCRIPT WILL AND WILL NOT DO. It reports. ``--apply`` is deliberately
refused: the corrected row set is not the one [c4f16946]'s acceptance criteria
name, so applying it would write ledger numbers nobody has agreed to, and a
correction that reaches only one book is a failure this repo has already
shipped. Re-file the row set from this output first, then teach --apply the
set it agreed to.

Two tolerances are printed, and they are different questions:
  * ``target * (1 + fee_rate)`` -- the bound the FORWARD fix enforces. A fill
    inside one leg's fee is ordinary slippage against the limit.
  * ``1.10x target`` -- the bound the forward fix's own docstring used to size
    the damage, and the one the operator verified the book against.

Run:  python -X utf8 scripts/annul_ghost_gap_fills.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"

#: Exit reasons that close against a stored upward target. ``take_profit_limit``
#: and ``target_hit`` are trading.bot.LIMIT_EXIT_REASONS; ``time_take_profit``
#: is the same shape (it also trips on price >= target) and is included so the
#: census cannot miss a row by reason name alone.
TAKE_PROFIT_REASONS = ("take_profit_limit", "target_hit", "time_take_profit")

#: One leg's variable cost, from receipts (0.3187% of notional). A fill this
#: far past the limit is slippage, not a sampling gap.
DEFAULT_FEE_RATE = 0.003187

#: The bound trading.bot.limit_exit_fill_price's docstring used to size the
#: damage, and the one the operator verified the whole book against.
GAP_FILL_RATIO = 1.10


def _position_hash(trade_id: Any) -> str:
    """The hash that identifies one position across both tables."""
    return str(trade_id or "").split(":")[-1]


def target_for_trade(conn: sqlite3.Connection, trade_id: Any) -> Optional[float]:
    """The target the position was actually aiming at, or None.

    Joined on the position hash rather than on a price tolerance. A price
    match looks right and is not: BSTONK-USDC alone is re-entered dozens of
    times at similar prices, so a tolerance match returns some OTHER entry's
    target and the resulting exit/target ratio is meaningless.
    """
    found: Optional[float] = None
    hsh = _position_hash(trade_id)
    if not hsh:
        return None
    for row in conn.execute(
        "SELECT details FROM trading_ops WHERE action='enter' AND details LIKE ? ORDER BY ts",
        ("%" + hsh + "%",),
    ):
        try:
            det = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            continue
        try:
            target = float(det.get("target_price") or 0.0)
        except (TypeError, ValueError):
            continue
        if target > 0.0:
            found = target
    return found


def strategy_for_trade(conn: sqlite3.Connection, trade_id: Any) -> Optional[str]:
    """Who owns this round trip, from the ops rather than the outcome row.

    ``trade_outcomes.details.strategy_id`` is absent on 96 of 214 rows, and
    every previous census read that field alone and called the row
    unattributed. The SAME position hash reaches ops rows that DO carry
    ``strategy_id``: BASELINE +57.94% is rsi_reversal's and BASECAT +17.31% is
    atf_static's, both of which read as owned by nobody in the outcome table.

    Returning None here therefore means genuinely unowned -- no row in either
    table names a strategy -- which is a much stronger statement than the
    outcome row being blank, and it is the one that decides whether annulling
    a row can move any strategy's ledger at all.
    """
    hsh = _position_hash(trade_id)
    if not hsh:
        return None
    for row in conn.execute(
        "SELECT details FROM trading_ops WHERE details LIKE ? ORDER BY ts",
        ("%" + hsh + "%",),
    ):
        try:
            det = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            continue
        sid = det.get("strategy_id")
        if sid:
            return str(sid)
    return None


def classify(entry: float, exit_price: float, target: Optional[float],
             *, fee_rate: float = DEFAULT_FEE_RATE) -> Dict[str, Any]:
    """How far past its own limit did this row book, and by which bound.

    ``over_fee`` is the forward fix's rule; ``gap_fill`` is the 1.10x bound.
    A row with no recoverable target is neither -- it is UNKNOWN, and saying
    so is the point. Guessing a target from the return is the mistake this
    whole script exists to correct.
    """
    out: Dict[str, Any] = {
        "target": target, "ratio": None, "over_fee": False, "gap_fill": False,
        "return_pct": None, "target_pct": None, "known": target is not None,
    }
    try:
        entry = float(entry)
        exit_price = float(exit_price)
    except (TypeError, ValueError):
        return out
    if entry > 0.0 and exit_price > 0.0:
        out["return_pct"] = (exit_price / entry - 1.0) * 100.0
    if target is None or not (target > 0.0) or not (exit_price > 0.0):
        return out
    out["ratio"] = exit_price / target
    out["over_fee"] = exit_price > target * (1.0 + max(0.0, fee_rate))
    out["gap_fill"] = out["ratio"] > GAP_FILL_RATIO
    if entry > 0.0:
        out["target_pct"] = (target / entry - 1.0) * 100.0
    return out


def census(conn: sqlite3.Connection, *, fee_rate: float = DEFAULT_FEE_RATE) -> List[Dict[str, Any]]:
    """Every closed ghost take-profit round trip, with its own target attached."""
    conn.row_factory = sqlite3.Row
    rows: List[Dict[str, Any]] = []
    for row in conn.execute(
        "SELECT * FROM trade_outcomes WHERE status='closed' AND wallet='ghost' ORDER BY ts"
    ):
        try:
            det = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            det = {}
        reason = str(det.get("reason", "")).split(":")[0]
        if reason not in TAKE_PROFIT_REASONS:
            continue
        target = target_for_trade(conn, row["trade_id"])
        info = classify(row["entry_price"], row["exit_price"], target, fee_rate=fee_rate)
        owner = det.get("strategy_id") or strategy_for_trade(conn, row["trade_id"])
        info.update({
            "symbol": row["symbol"],
            "strategy_id": owner,
            "owned": owner is not None,
            "reason": reason,
            "net_profit": float(row["net_profit"] or 0.0),
            "outcome_id": row["outcome_id"],
            "ts": float(row["ts"] or 0.0),
        })
        rows.append(info)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="refused -- see the module docstring")
    ap.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    rows = census(conn, fee_rate=args.fee_rate)

    print(f"database : {DB_PATH}")
    print(f"closed ghost take-profit round trips: {len(rows)}   "
          f"target recovered for {sum(1 for r in rows if r['known'])}\n")
    header = "%-14s %-21s %9s %9s %9s %10s  %s" % (
        "symbol", "strategy", "return%", "target%", "exit/tgt", "net", "verdict")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (x["ratio"] or 0.0), reverse=True):
        if not r["known"]:
            verdict = "UNKNOWN -- entry op carries no target_price"
        elif r["gap_fill"]:
            verdict = "GAP FILL (>1.10x its own target)"
        elif r["over_fee"]:
            verdict = "past the limit by more than one leg's fee"
        else:
            verdict = "filled at its limit"
        print("%-14s %-21s %9s %9s %9s %+10.5f  %s" % (
            r["symbol"], str(r["strategy_id"]),
            "%+.2f" % r["return_pct"] if r["return_pct"] is not None else "-",
            "%+.2f" % r["target_pct"] if r["target_pct"] is not None else "-",
            "%.3f" % r["ratio"] if r["ratio"] is not None else "-",
            r["net_profit"], verdict))

    gap = [r for r in rows if r["gap_fill"]]
    fee = [r for r in rows if r["over_fee"] and not r["gap_fill"]]
    unknown = [r for r in rows if not r["known"]]
    print(f"\nabove 1.10x their own target : {len(gap)} rows, net {sum(r['net_profit'] for r in gap):+.5f}")
    print(f"past the fee bound only      : {len(fee)} rows, net {sum(r['net_profit'] for r in fee):+.5f}")
    print(f"target not recoverable       : {len(unknown)} rows, net {sum(r['net_profit'] for r in unknown):+.5f}")

    print("\nby strategy, rows above 1.10x their own target:")
    per: Dict[str, List[Dict[str, Any]]] = {}
    for r in gap:
        per.setdefault(str(r["strategy_id"]), []).append(r)
    for sid, rs in sorted(per.items()):
        print("  %-22s %d rows  net %+.5f" % (sid, len(rs), sum(x["net_profit"] for x in rs)))

    # The number that decides whether annulling anything can move graduation.
    # A row no strategy owns is in the POOLED book and in no strategy's ledger,
    # so striking it changes the headline and moves no strategy toward the bar.
    unowned = [r for r in gap if not r.get("owned")]
    if unowned:
        print(
            "\n  UNOWNED (no strategy_id in trade_outcomes OR trading_ops): "
            "%d rows, net %+.5f" % (len(unowned), sum(r["net_profit"] for r in unowned))
        )
        for r in unowned:
            print("    %-14s %.3fx target  net %+.5f" % (r["symbol"], r["ratio"], r["net_profit"]))
        print("  These are in the pooled book and in NO strategy's ledger: annulling")
        print("  them moves the headline and moves no strategy toward the bar.")

    if args.apply:
        print(
            "\nREFUSING --apply. [c4f16946]'s acceptance criteria name a row set "
            "chosen by RETURN (atf_static +1.0201 of 'fabricated' fills). Joined "
            "to their own targets those two rows are 1.019x and 1.074x -- inside "
            "the 1.10x bound -- and the rows that ARE above it are different ones. "
            "Re-file the set from this output before any book is rewritten: a "
            "correction that reaches only one book is a failure this repo has "
            "already shipped.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
