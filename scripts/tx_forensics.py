"""Answer "did this money get booked?" from the STORE, not from the log.

WHY THIS EXISTS. Sprint item [df638247] was filed, estimated at 5 points, and
dealt to an agent on the strength of this line in ``logs/system.log``:

    SETTLED BUY 0x0dfedf3a...0959 for CBBTC-USDC CANNOT BE MEASURED (no_fill);
    refusing to guess its fill, and refusing new entries on this symbol until
    it can be read -- the money is already gone.

Every factual claim in that item was false, and ``trading_ops`` said so in one
query each:

  * all four named transactions HAVE booking rows (``live-entry``,
    executed=True, size ~9.27e-06, quote_spent 0.75). The money had a basis.
  * CBBTC-USDC was never entry-locked. Its newest rows are ordinary
    ``entry-arbitration`` / ``entry-refused-symbol-motion`` refusals, and no
    row anywhere cites an unreconciled settled buy.
  * all 7,367 "cannot read fills from receipts" lines in that log name
    ``_Swapper`` (6,428), ``_OldSwapper`` (938) or ``_DeadSwapper`` (1) --
    classes that exist ONLY under tests/. Production's ``SwapService`` has
    ``read_fill`` (services/swap_service.py:151), so the ``no_fill`` branch
    that produced the whole item is unreachable outside a test double.

logs/system.log is written by the test suite, by CLI readers and by production,
with nothing in the line marking which. A claim derived from it is not evidence
about production until the store agrees. That mistake is cheap to make and cost
a full sprint item, so this makes the check cheap to avoid:

    python -X utf8 scripts/tx_forensics.py --audit
    python -X utf8 scripts/tx_forensics.py --symbol CBBTC-USDC
    python -X utf8 scripts/tx_forensics.py --tx 0x0dfedf3a...0959
    python -X utf8 scripts/tx_forensics.py --attribute "cannot read fills"

``--audit`` is the one that generalises: it walks EVERY symbol and reports each
settled live entry that has no booking row, which is the measurable form of
"real money with no booked basis" for the whole store at once.

Read-only. Opens the store with mode=ro and never writes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DB = os.environ.get("TRADING_DB_PATH") or os.path.join(
    "storage", "trading_cache.db"
)
DEFAULT_LOG = os.path.join("logs", "system.log")

#: Classes that only ever exist inside tests/. A log line naming one of these
#: was written by the suite, not by production, and says nothing about the
#: money. Extend this list rather than re-deriving it.
TEST_DOUBLE_RE = re.compile(r"\b_(?:[A-Z][A-Za-z0-9]*)?(?:Swapper|Bridge|Db|DB)\b")

SETTLED = "live-swap-settled"
BOOKED = "live-entry"


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        sys.exit(f"no store at {db_path!r} -- set TRADING_DB_PATH")
    size = os.path.getsize(db_path)
    if size < 1_000_000:
        # data/trading.db, trading_bot.db and web/db.sqlite3 are 0-byte stubs
        # that answer every query with nothing. Failing loudly here beats
        # reporting "no unbooked money" from an empty file.
        sys.exit(
            f"{db_path} is {size} bytes -- that is a stub, not the trading "
            f"store. The real one is storage/trading_cache.db (~25GB)."
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _details(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        parsed = json.loads(row["details"] or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ts(value: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
    except Exception:
        return "?"


def _rows(conn: sqlite3.Connection, where: str, params: Iterable[Any]) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        f"SELECT ts, symbol, status, details FROM trading_ops WHERE {where} ORDER BY ts",
        list(params),
    )
    return cur.fetchall()


def _booked_hashes(conn: sqlite3.Connection, symbol: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """tx_hash -> booking details, for every executed live-entry row."""
    where = "status=?"
    params: List[Any] = [BOOKED]
    if symbol:
        where += " AND symbol=?"
        params.append(symbol)
    out: Dict[str, Dict[str, Any]] = {}
    for row in _rows(conn, where, params):
        det = _details(row)
        tx = str(det.get("tx_hash") or "").lower()
        # ``executed`` is the booking row's own flag. A row that records a
        # FAILED entry carries the hash too, and counting it as booked would
        # hide exactly the money this tool is looking for.
        if tx and det.get("executed") is True:
            out[tx] = {"ts": row["ts"], "symbol": row["symbol"], **det}
    return out


def _settled_entries(
    conn: sqlite3.Connection, symbol: Optional[str]
) -> List[Tuple[sqlite3.Row, Dict[str, Any]]]:
    """Every confirmed settled live_entry swap, oldest first."""
    where = "status=?"
    params: List[Any] = [SETTLED]
    if symbol:
        where += " AND symbol=?"
        params.append(symbol)
    out: List[Tuple[sqlite3.Row, Dict[str, Any]]] = []
    for row in _rows(conn, where, params):
        det = _details(row)
        if str(det.get("purpose") or "") != "live_entry":
            continue
        if det.get("confirmed") is not True or not det.get("ok"):
            continue
        if not str(det.get("tx_hash") or ""):
            continue
        out.append((row, det))
    return out


def audit(conn: sqlite3.Connection, symbol: Optional[str] = None) -> int:
    """Settled live entries with no booking row, across the whole store.

    This is the measurable form of "real money left the wallet with no booked
    basis". Returns the count so a caller can gate on it.
    """
    booked = _booked_hashes(conn, symbol)
    settled = _settled_entries(conn, symbol)
    orphans = [(r, d) for r, d in settled if str(d.get("tx_hash", "")).lower() not in booked]

    scope = symbol or "ALL SYMBOLS"
    print(f"=== UNBOOKED SETTLED LIVE ENTRIES -- {scope} ===")
    print(f"settled live entries : {len(settled)}")
    print(f"booked (executed)    : {len(booked)} live-entry rows carry a hash")
    print(f"UNBOOKED             : {len(orphans)}")
    if not orphans:
        print("\nno settled live entry is missing a booking row.")
        return 0
    print()
    by_symbol: Dict[str, int] = {}
    for row, det in orphans:
        sym = str(row["symbol"])
        by_symbol[sym] = by_symbol.get(sym, 0) + 1
        print(
            f"  {_ts(row['ts'])}  {sym:<14} {det.get('tx_hash')}"
            f"\n      sell={det.get('sell')} buy={det.get('buy')}"
        )
    print("\nby symbol: " + ", ".join(f"{k}={v}" for k, v in sorted(by_symbol.items())))
    return len(orphans)


def show_symbol(conn: sqlite3.Connection, symbol: str) -> None:
    """The full settled-vs-booked picture for one symbol, plus its lock state."""
    print(f"=== {symbol} ===\n")
    rows = _rows(conn, "symbol=? AND status IN (?,?)", [symbol, SETTLED, BOOKED])
    if not rows:
        print("no settled or booking rows for this symbol.")
    booked = _booked_hashes(conn, symbol)
    print("settled swaps (newest last):")
    for row in rows:
        det = _details(row)
        if row["status"] != SETTLED:
            continue
        tx = str(det.get("tx_hash") or "")
        purpose = str(det.get("purpose") or "?")
        mark = ""
        if purpose == "live_entry":
            mark = "  <-- BOOKED" if tx.lower() in booked else "  <-- *** UNBOOKED ***"
        print(f"  {_ts(row['ts'])}  {purpose:<11} {tx[:24]}{mark}")

    print("\nbooking rows:")
    for tx, det in sorted(booked.items(), key=lambda kv: kv[1]["ts"]):
        print(
            f"  {_ts(det['ts'])}  {tx[:24]}  size={det.get('size')} "
            f"spent={det.get('quote_spent')}"
        )

    # The entry lock the refusal message threatens lives in
    # ``_unreconciled_settled_buy``, which is in-memory. Its observable shadow
    # is a refusal row that names it -- so its ABSENCE is the evidence that the
    # symbol is not locked, whatever the log says.
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM trading_ops WHERE symbol=? AND "
        "(status LIKE '%unreconcil%' OR details LIKE '%unreconciled%')",
        [symbol],
    )
    locked = cur.fetchone()[0]
    cur.execute(
        "SELECT ts, status FROM trading_ops WHERE symbol=? ORDER BY ts DESC LIMIT 5",
        [symbol],
    )
    recent = cur.fetchall()
    print(f"\nrows citing an unreconciled settled buy: {locked}")
    print("newest activity (if these are ordinary refusals, it is not locked):")
    for row in recent:
        print(f"  {_ts(row['ts'])}  {row['status']}")


def show_tx(conn: sqlite3.Connection, tx: str) -> None:
    """Every row in the store mentioning one transaction hash."""
    needle = f"%{tx.lower().removeprefix('0x')}%"
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, symbol, status, details FROM trading_ops "
        "WHERE lower(details) LIKE ? ORDER BY ts",
        [needle],
    )
    rows = cur.fetchall()
    print(f"=== {tx} ===")
    print(f"{len(rows)} row(s) in trading_ops\n")
    if not rows:
        print("NOT IN THE STORE. If a log line names it, that line did not")
        print("come from production -- run --attribute on the message.")
        return
    for row in rows:
        det = _details(row)
        extra = {
            k: det[k]
            for k in ("purpose", "executed", "confirmed", "ok", "size", "quote_spent")
            if k in det
        }
        print(f"  {_ts(row['ts'])}  {row['symbol']:<14} {row['status']:<26} {extra}")


def attribute(pattern: str, log_path: str) -> None:
    """Did production write these log lines, or did the test suite?

    A line naming a test double was written by pytest. This is the check that
    [df638247] skipped.
    """
    if not os.path.exists(log_path):
        sys.exit(f"no log at {log_path!r}")
    rx = re.compile(pattern, re.IGNORECASE)
    total = 0
    doubles: Dict[str, int] = {}
    samples: List[str] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not rx.search(line):
                continue
            total += 1
            hit = TEST_DOUBLE_RE.findall(line)
            if hit:
                for name in hit:
                    doubles[name] = doubles.get(name, 0) + 1
            elif len(samples) < 5:
                samples.append(line.rstrip())
    tagged = sum(doubles.values())
    print(f"=== lines matching {pattern!r} in {log_path} ===")
    print(f"total           : {total}")
    print(f"name a TEST DOUBLE: {tagged}")
    if total:
        print(f"                  ({100.0 * tagged / total:.1f}% -- these are pytest output)")
    for name, count in sorted(doubles.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<16} {count}")
    print(f"unattributed    : {total - tagged}")
    if samples:
        print("\nlines NOT naming a test double (check these against the store):")
        for line in samples:
            print(f"  {line[:200]}")
    elif total:
        print("\nEVERY matching line names a test double. This log pattern is")
        print("test output and is not evidence about production.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check a money claim against trading_ops instead of the log."
    )
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--audit", action="store_true", help="every unbooked settled live entry")
    ap.add_argument("--symbol", help="settled vs booked, and lock state, for one symbol")
    ap.add_argument("--tx", help="every store row mentioning one transaction hash")
    ap.add_argument("--attribute", help="regex: did production write these log lines?")
    args = ap.parse_args(argv)

    if args.attribute:
        attribute(args.attribute, args.log)
        return 0
    if not (args.audit or args.symbol or args.tx):
        ap.print_help()
        return 2

    conn = _connect(args.db)
    try:
        if args.audit:
            audit(conn, args.symbol)
        elif args.symbol:
            show_symbol(conn, args.symbol)
        if args.tx:
            if args.audit or args.symbol:
                print()
            show_tx(conn, args.tx)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
