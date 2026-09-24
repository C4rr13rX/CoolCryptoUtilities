"""Tie the live scoreboard to the chain, row by row, or say which rows it cannot.

THE LOOP'S ONE NUMBER is cumulative realised live net P/L, and the standing
rule is that a profit which cannot be tied to a settled on-chain transaction
hash did not happen. Until this script existed, nothing performed that tie.

Pass 695 (Sage, auditor) spent four wrong turns finding the key, so they are
written down here to stop the next agent repeating them:

  * ``trade_outcomes`` has NO tx/hash column. A sweep of every table in the
    ~35GB store for a column named tx/hash returns ``discovery_swapprobe``
    (92 probe rows, not the trading book) plus three EMPTY tables:
    ``branddozer_gaterun``, ``transfers``, ``payouts_payout``.
  * The live hashes DO exist -- in ``trading_ops.details`` under actions
    enter / exit / swap / repair, 47 of them across 90 live rows.
  * Joining the two tables on ``(symbol, ts)`` is WRONG. Measured at +/-180s
    it matched 15 of 18 rows and COLLIDED: two AERO outcomes 501 seconds
    apart both matched 0x52135c2073523f601b827f. Widening to +/-600s rescued
    the three orphans (one hash sits at +185s, five seconds outside the tight
    window) but then drew 2-6 candidates per row. There is no window that is
    simultaneously one-to-one and complete. Do not use a time join.
  * THE KEY IS ``trade_outcomes.trade_id``. It appears verbatim inside
    ``trading_ops.details``. It is exact, and on the book as of p695 it ties
    18 of 18 live rows with zero blanks.

One nuance the tie exposes, and the reason this script prints positions as
well as rows: ``outcome_id`` carries an ``:exit:N`` suffix, and three of the
15 live positions scaled out in two legs (exit:1 then exit:2 -- same entry
price, smaller quantity, different exit price, both closed). So "18 live
trades" is 18 EXITS over 15 POSITIONS. Summing the legs is correct; counting
them as separate trades in a win rate or a payoff is not obviously so. Say
which denominator you used.

Usage:
    python scripts/live_chain_audit.py                 # live book
    python scripts/live_chain_audit.py --wallet ghost  # any wallet
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# Keys under which trading_ops.details carries a settled transaction hash.
# ``repair`` rows use "tx"; enter/exit/swap use "tx_hash".
_HASH_KEYS = ("tx_hash", "tx", "exit_tx_hash", "entry_tx_hash")

# Above this many distinct trade_ids, one scan of trading_ops beats one
# indexless LIKE scan per id. See _hashes_by_trade_id.
_SCAN_ONCE_ABOVE = 32


def _ascii(text: Any) -> str:
    """Windows consoles here are cp1252 and the ghost book carries symbols
    that are not encodable in it -- a UnicodeEncodeError on a print would
    lose the whole audit over a ticker name."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _hash_from(details: str) -> str:
    try:
        blob = json.loads(details) if details else {}
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(blob, dict):
        return ""
    for key in _HASH_KEYS:
        value = blob.get(key)
        if isinstance(value, str) and value.startswith("0x"):
            return value
    return ""


def _hashes_by_trade_id(conn: sqlite3.Connection, trade_ids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Exact substring join: trade_id appears verbatim in trading_ops.details.

    NOT a fuzzy match -- the id is a uuid-suffixed string, so a hit is a hit.

    Two directions, because the cheap one flips with the size of the book.
    One ``LIKE`` per id is a full scan of ~146k ops rows per id: fine for the
    18-row live book, but the 272-row ghost book exceeded 120s and had to be
    killed (p695). Above the threshold, scan the ops table ONCE and test each
    row against the id set in memory instead.
    """
    ids = [t for t in dict.fromkeys(trade_ids) if t]
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    if not ids:
        return out

    if len(ids) <= _SCAN_ONCE_ABOVE:
        for tid in ids:
            for action, details in conn.execute(
                "SELECT action, details FROM trading_ops WHERE details LIKE ?",
                ("%" + tid + "%",),
            ):
                tx = _hash_from(details)
                if tx:
                    out[tid].append((str(action), tx))
        return out

    for action, details in conn.execute("SELECT action, details FROM trading_ops"):
        if not details:
            continue
        tx = _hash_from(details)
        if not tx:
            continue
        for tid in ids:
            if tid in details:
                out[tid].append((str(action), tx))
    return out


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--wallet", default="live")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print("no such db: %s" % args.db)
        return 2

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT outcome_id, trade_id, ts, symbol, quantity, net_profit, status
        FROM trade_outcomes WHERE wallet = ? ORDER BY ts
        """,
        (str(args.wallet),),
    ).fetchall()

    if not rows:
        # wallet is a TEXT column: wallet=1 returns nothing, silently.
        print("no %r rows in trade_outcomes -- note wallet is a STRING column" % args.wallet)
        return 1

    ties = _hashes_by_trade_id(conn, [r[1] for r in rows])

    print("=" * 78)
    print("LIVE CHAIN AUDIT -- every outcome row tied to a settled tx, or named as untied")
    print("  db     : %s" % args.db)
    print("  wallet : %s" % args.wallet)
    print("=" * 78)

    tied = 0
    untied_net = 0.0
    untied: List[Tuple[str, float]] = []
    seen_once: Dict[str, int] = defaultdict(int)

    for outcome_id, trade_id, _ts, symbol, _qty, net, status in rows:
        net = float(net or 0.0)
        uniq = sorted({tx for _a, tx in ties.get(trade_id or "", [])})
        for tx in uniq:
            seen_once[tx] += 1
        if uniq:
            tied += 1
            shown = ", ".join(tx[:18] for tx in uniq)
        else:
            untied.append((str(symbol), net))
            untied_net += net
            shown = "-- NO SETTLED HASH --"
        leg = str(outcome_id).rsplit(":", 2)[-2:]
        leg = ":".join(leg) if len(leg) == 2 else ""
        print("  %-14s %-8s net=%+.6f  txs=%d  %s" % (_ascii(symbol), leg, net, len(uniq), shown))

    total = sum(float(r[5] or 0.0) for r in rows)
    positions = len({r[1] for r in rows})

    print("-" * 78)
    print("  exits (outcome rows)      : %d" % len(rows))
    print("  positions (trade_id)      : %d   <- three p695 rows are exit:2 legs of a scale-out" % positions)
    print("  TIED to a settled tx      : %d of %d" % (tied, len(rows)))
    print("  UNTIED rows               : %d, carrying net %+.6f" % (len(untied), untied_net))
    for symbol, net in untied:
        print("      untied: %-14s %+.6f" % (_ascii(symbol), net))
    print("  cumulative realised net   : %+.6f   <- THE SCOREBOARD" % total)
    print("  distinct settled hashes   : %d" % len(seen_once))
    shared = sum(1 for _tx, n in seen_once.items() if n > 1)
    print("  hashes against >1 row     : %d  (expected: a scale-out shares its ENTRY hash)" % shared)
    print("=" * 78)
    if untied:
        if str(args.wallet) != "live":
            # A simulated book has nothing to settle. Untied is the EXPECTED
            # state here, and calling it a failure would train people to
            # ignore this script's exit code on the one wallet it matters for.
            print("VERDICT: %d of %d %r row(s) untied -- EXPECTED, a simulated book has no chain."
                  % (len(untied), len(rows), str(args.wallet)))
            print("         The %d tied row(s) are worth a look: a ghost row should not"
                  % (len(rows) - len(untied)))
            print("         carry a settled hash at all.")
            return 0
        print("VERDICT: %d row(s) are NOT chain-backed. That P/L did not happen until they are."
              % len(untied))
        return 1
    print("VERDICT: every row is chain-backed. The scoreboard is auditable.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
