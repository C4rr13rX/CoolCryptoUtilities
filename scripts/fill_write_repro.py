#!/usr/bin/env python
"""Offline reproduction of the fill-write exception -- item [11866e31].

The premise of 5b51114a is that ``db.py`` record_trade_fill's INSERT into
trade_fills is what raises, and that finding out costs a 13-minute production
restart.  It does not.  The statement is tiny, the table has no index, no
trigger and no CHECK, and the whole thing runs offline against a scratch file
in under a second.

What this probe separates, which the item conflates:

  (A) Python argument marshalling.  db.py:929-936 evaluates float() on four
      arguments and json.dumps() on a fifth, INSIDE the ``with self._conn:``
      block but BEFORE sqlite is handed anything.  float(None) and
      json.dumps(Decimal) both raise TypeError there, with a transaction open,
      so the rollback makes it surface AT the fill write while never having
      reached the database.

  (B) The INSERT itself -- a genuine sqlite fault.

  (C) trading/bot.py:12729 record_fill calls self._save_state() immediately
      after the write.  An exception from THAT is attributed to the fill by
      every caller that wraps the pair in one try.

Run:  python -X utf8 scripts/fill_write_repro.py

Writes only to a scratch directory.  Never opens storage/trading_cache.db for
write, never copies it (the copy does not fit: 26.08 GiB of store against
14.57 GiB of free disk, measured pass 485).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import traceback
from decimal import Decimal

# The production schema, read verbatim from sqlite_master on
# storage/trading_cache.db, pass 485.  Kept as a literal so the probe does not
# need the 26 GiB file to run.
TRADE_FILLS_SCHEMA = """
CREATE TABLE trade_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    chain TEXT,
    symbol TEXT,
    expected_amount REAL,
    executed_amount REAL,
    expected_price REAL,
    executed_price REAL,
    details TEXT
)
"""

# db.py:923-937, verbatim.
INSERT_SQL = """
INSERT INTO trade_fills(ts, chain, symbol, expected_amount, executed_amount, expected_price, executed_price, details)
VALUES(strftime('%s','now'), ?, ?, ?, ?, ?, ?, ?)
"""


class _Probe:
    """Mirrors db.py record_trade_fill exactly, including where float() runs."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record_trade_fill(
        self,
        *,
        chain,
        symbol,
        expected_amount,
        executed_amount,
        expected_price,
        executed_price,
        details=None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                INSERT_SQL,
                (
                    chain,
                    symbol,
                    float(expected_amount),
                    float(executed_amount),
                    float(expected_price),
                    float(executed_price),
                    json.dumps(details or {}),
                ),
            )


class _AttributeDict(dict):
    """Stand-in for web3's AttributeDict, which json.dumps handles fine, and
    for the receipt objects it nests, which it does not."""


class _Receipt:
    """A non-serialisable object of the shape a swap receipt has."""

    def __init__(self) -> None:
        self.tx_hash = b"\x01\x02\x03"


def _run(name: str, fn) -> dict:
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the exception IS the result
        return {
            "case": name,
            "raised": True,
            "type": type(exc).__name__,
            "msg": str(exc)[:200],
            "where": _raising_line(exc),
        }
    return {"case": name, "raised": False, "type": None, "msg": None, "where": None}


def _raising_line(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return "?"
    last = tb[-1]
    return "%s:%d in %s" % (os.path.basename(last.filename), last.lineno, last.name)


def main() -> int:
    scratch = os.path.join(tempfile.gettempdir(), "fill_write_repro")
    os.makedirs(scratch, exist_ok=True)
    path = os.path.join(scratch, "scratch_fills.db")
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(TRADE_FILLS_SCHEMA)
    conn.commit()
    probe = _Probe(conn)

    good = dict(
        chain="base",
        symbol="AERO-USDC",
        expected_amount=1.0,
        executed_amount=0.9987,
        expected_price=0.7412,
        executed_price=0.7405,
        details={"mode": "live", "slippage_bps": 9.4},
    )

    cases = [
        ("well_formed", lambda: probe.record_trade_fill(**good)),
        (
            "none_executed_amount",
            lambda: probe.record_trade_fill(**{**good, "executed_amount": None}),
        ),
        (
            "none_executed_price",
            lambda: probe.record_trade_fill(**{**good, "executed_price": None}),
        ),
        (
            "decimal_amount",
            lambda: probe.record_trade_fill(
                **{**good, "executed_amount": Decimal("0.9987")}
            ),
        ),
        (
            "str_amount",
            lambda: probe.record_trade_fill(**{**good, "executed_amount": "0.9987"}),
        ),
        (
            "details_has_decimal",
            lambda: probe.record_trade_fill(
                **{**good, "details": {"mode": "live", "gas": Decimal("0.0031")}}
            ),
        ),
        (
            "details_has_bytes",
            lambda: probe.record_trade_fill(
                **{**good, "details": {"tx_hash": b"\xde\xad\xbe\xef"}}
            ),
        ),
        (
            "details_has_receipt_object",
            lambda: probe.record_trade_fill(
                **{**good, "details": {"receipt": _Receipt()}}
            ),
        ),
        (
            "details_is_attribute_dict",
            lambda: probe.record_trade_fill(
                **{**good, "details": _AttributeDict(mode="live", n=1)}
            ),
        ),
        ("nan_price", lambda: probe.record_trade_fill(**{**good, "executed_price": float("nan")})),
        ("symbol_is_none", lambda: probe.record_trade_fill(**{**good, "symbol": None})),
    ]

    results = [_run(n, f) for n, f in cases]

    print("=" * 74)
    print("OFFLINE FILL-WRITE REPRO -- item [11866e31]")
    print("scratch db: %s" % path)
    print("=" * 74)
    width = max(len(r["case"]) for r in results)
    for r in results:
        if r["raised"]:
            print(
                "  RAISED  %-*s  %-14s %s"
                % (width, r["case"], r["type"], r["where"])
            )
            print("          %-*s  %s" % (width, "", r["msg"]))
        else:
            print("  clean   %-*s" % (width, r["case"]))

    rows = conn.execute("SELECT COUNT(*) FROM trade_fills").fetchone()[0]
    print("-" * 74)
    print("rows committed to scratch: %d of %d attempts" % (rows, len(cases)))

    # The one structural check that does NOT need a copy of the 26 GiB store:
    # does the scratch table, built from the production schema, survive its own
    # integrity check after these writes?
    ic = conn.execute("PRAGMA integrity_check").fetchone()[0]
    print("scratch integrity_check: %s" % ic)

    wf = next(r for r in results if r["case"] == "well_formed")
    raised = [r for r in results if r["raised"]]
    print("=" * 74)
    if wf["raised"]:
        print("VERDICT: the statement itself is wrong. Hypothesis A is DEAD.")
    elif raised:
        marshalling = [r for r in raised if r["where"].startswith("fill_write_repro")]
        print(
            "VERDICT: well-formed input inserts CLEAN. %d of %d malformed inputs "
            "raise, and %d of those raise in Python argument marshalling BEFORE "
            "sqlite is handed anything." % (len(raised), len(cases) - 1, len(marshalling))
        )
        print("         Hypothesis A SUPPORTED. The INSERT is exonerated.")
        print("         Failing inputs, in the order a caller would hit them:")
        for r in raised:
            print("           %-28s -> %s: %s" % (r["case"], r["type"], r["msg"][:80]))
    else:
        print(
            "VERDICT: NOTHING raises, including deliberately malformed input. "
            "The statement cannot be the source. Look at the caller."
        )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
