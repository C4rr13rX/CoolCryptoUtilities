#!/usr/bin/env python3
"""Per-row forensics on the live book: clip, P/L, and a SETTLED chain hash for each row.

The scoreboard for this repo is one number -- cumulative realised net P/L over live
round trips that settled on chain. This script is the thing that proves each of its
rows actually happened, because a P/L that cannot be tied to a settled transaction
hash did not happen.

Three things it does, and each is a separate acceptance criterion on [42a65c7b]:

  1. TIE. Join each closed live `trade_outcomes` row to its entry and exit tx hash.
     The hashes are NOT in any column of the P/L tables -- they live in
     `trading_ops.details` as JSON, which is why several earlier censuses reported
     "zero of 60 rows carry a hash" and were accurate about the columns they read
     while being wrong about the store.

  2. SETTLE. Hit `eth_getTransactionReceipt` for every hash and record status and
     block. A hash in our own book is still only our own book asserting itself; the
     receipt is the independent source.

  3. SPLIT. Report the live rows separately from the ghost rows. Two repricing
     scripts load all of `trade_outcomes` and silently mix the live rows into a
     "ghost" book, which is a contamination this repo has been bitten by before.

Read-only throughout: it opens the database for reading and makes only receipt
lookups against the chain. It places no trade and writes nothing but its report.

    python scripts/live_book_forensics.py              # tie + split, no chain calls
    python scripts/live_book_forensics.py --verify     # ...and settle every hash
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# The public Base endpoint is the fallback the production path itself falls to when
# the Alchemy key is rate-limited, so verifying through it measures the same chain
# production sees. Receipt lookups are cheap enough that the free tier serves them.
BASE_RPCS = [
    "https://mainnet.base.org",
    "https://base.llamarpc.com",
]


def scoreboard_rows(con: sqlite3.Connection) -> list[dict]:
    """The closed live round trips -- exactly the rows that sum to the scoreboard."""
    cur = con.cursor()
    cur.execute(
        "SELECT outcome_id, trade_id, ts, symbol, quantity, entry_price, exit_price, "
        "       gross_profit, fee_cost, net_profit "
        "FROM trade_outcomes "
        "WHERE status='closed' AND json_extract(details,'$.mode')='live' "
        "ORDER BY ts"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def hash_index(con: sqlite3.Connection) -> tuple[dict, dict]:
    """Index every tx hash in `trading_ops` by the outcome and trade it belongs to.

    Returns (by_outcome_id, by_trade_id); each maps an id to a list of
    (status, tx_hash, entry_tx_hash) triples.
    """
    cur = con.cursor()
    cur.execute(
        "SELECT status, details FROM trading_ops "
        "WHERE json_extract(details,'$.tx_hash') LIKE '0x%' "
        "   OR json_extract(details,'$.entry_tx_hash') LIKE '0x%'"
    )
    by_outcome: dict[str, list] = {}
    by_trade: dict[str, list] = {}
    for status, raw in cur.fetchall():
        try:
            det = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            continue
        triple = (status or "", det.get("tx_hash"), det.get("entry_tx_hash"))
        oid, tid = det.get("outcome_id"), det.get("trade_id")
        if oid is not None:
            by_outcome.setdefault(str(oid), []).append(triple)
        if tid is not None:
            by_trade.setdefault(str(tid), []).append(triple)
    return by_outcome, by_trade


def tie_hashes(row: dict, by_outcome: dict, by_trade: dict) -> tuple[str | None, str | None]:
    """Recover (entry_hash, exit_hash) for one scoreboard row.

    The exit leg is keyed by outcome_id and the entry leg by trade_id, so both
    indexes have to be consulted -- joining on one alone loses half the book.
    """
    entry = exit_ = None
    for status, tx, entry_tx in by_outcome.get(str(row["outcome_id"]), []) + by_trade.get(
        str(row["trade_id"]), []
    ):
        if entry_tx and not entry:
            entry = entry_tx
        if tx and "exit" in status and not exit_:
            exit_ = tx
        if tx and "entry" in status and not entry:
            entry = tx
    return entry, exit_


def notional(row: dict) -> float | None:
    """Traded notional of the entry leg -- the clip. This is what decides the bp."""
    qty, px = row.get("quantity"), row.get("entry_price")
    if qty is None or px is None:
        return None
    return abs(float(qty) * float(px))


def connect_chain():
    """First responsive Base endpoint, or None if the chain is unreachable."""
    try:
        from web3 import Web3
    except ImportError:
        return None
    for url in BASE_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 15}))
            if w3.is_connected():
                _ = int(w3.eth.chain_id)
                return w3
        except Exception:
            continue
    return None


def settle(w3, tx_hash: str) -> dict:
    """Fetch one receipt. A missing receipt is a real answer, not an error."""
    try:
        rcpt = w3.eth.get_transaction_receipt(tx_hash)
    except Exception as exc:
        return {"found": False, "error": type(exc).__name__}
    return {
        "found": True,
        "status": int(rcpt.get("status", -1)),
        "block": int(rcpt.get("blockNumber", -1)),
    }


def _plausibility_banner(con: sqlite3.Connection) -> list[str]:
    """How many rows in the mode split the ledger REFUSES, sized in dollars."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from services import outcome_plausibility as op
    except Exception as exc:  # noqa: BLE001
        return ["!! UNCENSORED and the contamination is UNMEASURED: "
                f"services.outcome_plausibility did not import ({exc})."]
    cur = con.cursor()
    cur.execute("SELECT symbol, quantity, entry_price, exit_price, "
                "gross_profit, net_profit, details FROM trade_outcomes")
    cols = [d[0] for d in cur.description]
    rows = []
    for r in cur.fetchall():
        row = dict(zip(cols, r))
        try:
            meta = json.loads(row.get("details") or "{}")
        except (TypeError, ValueError):
            meta = {}
        row["strategy_id"] = meta.get("strategy_id") or ""
        rows.append(row)
    return op.banner(rows, censored=False)


def ghost_live_split(con: sqlite3.Connection) -> dict:
    """How much of `trade_outcomes` is live, so a ghost census can exclude it."""
    cur = con.cursor()
    cur.execute(
        "SELECT COALESCE(json_extract(details,'$.mode'),'(none)') AS mode, "
        "       COUNT(*), SUM(net_profit) "
        "FROM trade_outcomes GROUP BY mode ORDER BY COUNT(*) DESC"
    )
    return {m: {"rows": n, "net": p} for m, n, p in cur.fetchall()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="fetch a receipt for every hash")
    ap.add_argument("--json", metavar="PATH", help="write the full per-row report here")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = scoreboard_rows(con)
    by_outcome, by_trade = hash_index(con)
    split = ghost_live_split(con)
    censor_lines = _plausibility_banner(con)
    con.close()

    total_net = sum(float(r["net_profit"] or 0) for r in rows)
    print(f"SCOREBOARD  {len(rows)} closed live rows, net {total_net:.12f}")
    print(f"store       {DB}")
    print()

    w3 = connect_chain() if args.verify else None
    if args.verify and w3 is None:
        print("!! no Base RPC reachable -- reporting the tie only, NOT settlement")

    report, clips = [], []
    tied = settled = reverted = missing = 0
    for row in rows:
        entry, exit_ = tie_hashes(row, by_outcome, by_trade)
        clip = notional(row)
        if clip:
            clips.append(clip)
        rec = {
            "outcome_id": row["outcome_id"],
            "symbol": row["symbol"],
            "clip_usd": clip,
            "gross": row["gross_profit"],
            "fee_cost": row["fee_cost"],
            "net": row["net_profit"],
            "bp": (float(row["net_profit"]) / clip * 10_000) if clip else None,
            "entry_tx": entry,
            "exit_tx": exit_,
        }
        if entry and exit_:
            tied += 1
        if w3 is not None:
            for leg, tx in (("entry", entry), ("exit", exit_)):
                if not tx:
                    continue
                res = settle(w3, tx)
                rec[f"{leg}_receipt"] = res
                if not res["found"]:
                    missing += 1
                elif res["status"] == 1:
                    settled += 1
                else:
                    reverted += 1
                # The free endpoint rate-limits; spacing costs seconds and buys
                # the difference between a real answer and a throttled one.
                time.sleep(0.25)
        report.append(rec)

    print(f"{'symbol':<14}{'clip $':>9}{'net $':>11}{'bp':>9}  entry/exit hash")
    for rec in report:
        clip = f"{rec['clip_usd']:.4f}" if rec["clip_usd"] else "?"
        bp = f"{rec['bp']:.1f}" if rec["bp"] is not None else "?"
        mark = "OK" if rec["entry_tx"] and rec["exit_tx"] else "UNTIED"
        if w3 is not None:
            legs = [rec.get("entry_receipt"), rec.get("exit_receipt")]
            found = [r for r in legs if r and r.get("found")]
            if len(found) == len([r for r in legs if r]):
                mark = "SETTLED" if all(r["status"] == 1 for r in found) else "REVERTED"
            elif found:
                mark = "PARTIAL"
        print(f"{rec['symbol']:<14}{clip:>9}{float(rec['net']):>11.5f}{bp:>9}  {mark}")

    clips.sort()
    print()
    print(f"TIED        {tied}/{len(rows)} rows carry BOTH an entry and an exit hash")
    if w3 is not None:
        print(f"SETTLED     {settled} legs status=1, {reverted} reverted, {missing} not on chain")
    if clips:
        mid = clips[len(clips) // 2]
        print(f"CLIP        median ${mid:.4f}  min ${clips[0]:.4f}  max ${clips[-1]:.4f}")
        print(f"            total traded notional ${sum(clips):.4f}")
        print(f"            net is {total_net / sum(clips) * 10_000:.1f}bp of traded notional")

    print()
    print("trade_outcomes by mode -- a ghost census must exclude the live rows:")
    for mode, agg in split.items():
        print(f"  {mode:<10} {agg['rows']:>4} rows   net {agg['net'] or 0:+.5f}")
    # This split sums net_profit over EVERY row, ghost included, and
    # `trade_outcomes` is a PRE-GUARD source: the ledger refuses a row at
    # record() time but the row is still written. The scoreboard above is
    # live-only and chain-tied, so it is unaffected; the ghost line is not.
    # Uncensored on purpose -- a mode census that dropped rows would
    # understate the table it exists to describe -- with the size printed.
    # See [a9bbe626].
    for line in censor_lines:
        print(f"  {line}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": report, "split": split, "net": total_net}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
