"""Audit the donchian DEEP cohort against the executor's own book.

WHY THIS EXISTS (pass 674, item 13254b56).

The one positive result in this repo is the donchian_breakout DEEP cohort out
of ``scripts/strategy_replay_table.py``: n=44, 65.9% win, mean net +0.136%,
mean GROSS +0.246%. Because +0.246% gross survives only if the WETH/ETH round
trip is cheaper than 24.60bp, every pass since 495 has treated this as a COST
question and has queued a live WETH round trip to settle it.

It is not a cost question. This script goes to the rows the EXECUTOR actually
wrote and prices each one against its own notional, and the deep cohort's
gross is NEGATIVE before any cost row is consulted. A negative gross is not
rescued by a cheaper cost table, so the 24.60bp break-even cannot decide it.

WHAT IT PRINTS, and every number is taken from the store, never assumed:

  * one line per deep row: notional, gross in bp, fee charged in bp, net in bp
  * the cohort mean gross, and the count that came out gross-negative
  * the implied fixed-gas share at that row's clip, which is the whole reason
    two rows are charged 176bp and two are charged 46bp for the same token

READ THE ``fee_bp`` COLUMN BEFORE YOU BELIEVE ANY COHORT MEAN THIS PRINTS.
The book spans a COST CUTOVER. Rows written before 2026-09-04 were charged a
FLAT 65.00bp from ``.env`` ``GHOST_EXPECTANCY_FEE_RATE=0.0065`` (in the code
2026-08-27, last touched 2026-09-03 by ``8e638f9``); rows written after were
charged the size-dependent model in ``services/roundtrip_cost.py``, first
called 2026-09-04 in ``b5722f8`` and wired broadly 2026-09-06 in ``fa5ff9a``.
Measured pass 674: 112 of 265 rows are on the flat path (ages 18.9-27.7 days)
and 153 on the size-dependent one (ages 0.8-19.6 days); the two ranges touch
only at the boundary, and the boundary is the commit date. Nothing is broken
TODAY -- every recent row is on the correct path -- but 42% of the book is
priced under the old rule, so any cohort mean spanning 2026-09-04 mixes two
cost eras. The flat path is the permissive one: at a $0.109 notional it charges
$0.00071 against $0.004047 of gas actually burned, which is why ``gas_share``
prints above 100% on the dust rows. That is exactly where the donchian THIN
cohort's +173.11bp headline comes from -- BSTONK +1786.62bp and TYBG +651.82bp
at notionals between $0.109 and $2.00, all pre-cutover. The ``gas_share``
column exists to make those rows impossible to miss.

THE TWO SCHEMA TRAPS, written down because both cost a probe:

  ``trade_outcomes`` has NO ``strategy_id`` column and NO ``mode`` column.
  Both live inside the ``details`` JSON blob. A query naming either as a
  column dies with ``sqlite3.OperationalError: no such column``.

  ``trading_ops`` has no ``timestamp`` column either; it is ``ts``.

READ-ONLY. Opens the store with ``mode=ro`` and writes nothing.

    python -X utf8 scripts/deep_cohort_gross_audit.py
    python -X utf8 scripts/deep_cohort_gross_audit.py --strategy rsi_reversal
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

#: The deep names. This is the cohort the replay table calls DEEP and the only
#: one whose cost row (11bp, n=0 live) has ever been the deciding term.
DEEP_BASES = frozenset({"WETH", "ETH"})

#: services/roundtrip_cost.py DEFAULT_FIXED_USD -- the gas leg, in dollars, per
#: round trip. Quoted here only to SHOW its share of each row's charged fee;
#: nothing below depends on it being right.
FIXED_GAS_USD = 0.004047


def _store_path() -> str:
    return os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def _rows(conn: sqlite3.Connection, prefix: str) -> list[dict]:
    out: list[dict] = []
    sql = (
        "select symbol, entry_price, quantity, gross_profit, fee_cost, "
        "net_profit, status, details from trade_outcomes"
    )
    for symbol, entry, qty, gross, fee, net, status, details in conn.execute(sql):
        try:
            blob = json.loads(details or "{}")
        except (TypeError, ValueError):
            blob = {}
        sid = str(blob.get("strategy_id") or blob.get("strategy") or "")
        if not sid.startswith(prefix):
            continue
        notional = float(entry or 0.0) * float(qty or 0.0)
        out.append(
            {
                "symbol": symbol or "",
                "sid": sid,
                "mode": blob.get("mode"),
                "status": status,
                "notional": notional,
                "gross": float(gross or 0.0),
                "fee": float(fee or 0.0),
                "net": float(net or 0.0),
                "deep": (symbol or "").split("-")[0].upper() in DEEP_BASES,
            }
        )
    return out


def _bp(value: float, notional: float) -> float:
    """Value in dollars as bp of the row's OWN notional.

    A row whose notional is zero cannot be expressed as a rate at all; it is
    reported as nan rather than silently dropped, because a dropped row is how
    a cohort mean gets quietly improved.
    """
    return 1e4 * value / notional if notional else float("nan")


def _report(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{label}: no rows")
        return
    print(f"\n{label}  n={len(rows)}")
    header = (
        f"  {'symbol':11s} {'strategy':26s} {'notional':>9s} "
        f"{'gross_bp':>9s} {'fee_bp':>8s} {'net_bp':>9s} {'gas_share':>10s}"
    )
    print(header)
    gross_bps: list[float] = []
    for row in sorted(rows, key=lambda r: r["notional"]):
        gbp = _bp(row["gross"], row["notional"])
        fbp = _bp(row["fee"], row["notional"])
        nbp = _bp(row["net"], row["notional"])
        share = 100.0 * FIXED_GAS_USD / row["fee"] if row["fee"] else float("nan")
        print(
            f"  {row['symbol']:11s} {row['sid']:26s} {row['notional']:9.3f} "
            f"{gbp:9.2f} {fbp:8.2f} {nbp:9.2f} {share:9.1f}%"
        )
        if gbp == gbp:  # not nan
            gross_bps.append(gbp)
    if not gross_bps:
        return
    mean_gross = sum(gross_bps) / len(gross_bps)
    negative = sum(1 for g in gross_bps if g < 0)
    wins = sum(1 for r in rows if r["net"] > 0)
    print(
        f"  MEAN GROSS {mean_gross:+.2f}bp over n={len(gross_bps)}; "
        f"{negative} of {len(gross_bps)} are gross-NEGATIVE; "
        f"net wins {wins}/{len(rows)}"
    )
    if mean_gross < 0:
        print(
            "  -> gross is NEGATIVE before any cost row is consulted. No value "
            "of the WETH cost row makes this cohort positive; the 24.60bp "
            "break-even is MOOT on these rows."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strategy",
        default="donchian",
        help="strategy_id prefix to audit (default: donchian)",
    )
    args = ap.parse_args()

    path = _store_path()
    if not os.path.exists(path):
        print(f"store not found: {path}")
        return 2
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = _rows(conn, args.strategy)
    finally:
        conn.close()

    if not rows:
        print(f"no trade_outcomes rows with strategy_id prefix {args.strategy!r}")
        return 1

    print(f"store: {path}   strategy prefix: {args.strategy!r}   rows: {len(rows)}")
    modes = sorted({str(r["mode"]) for r in rows})
    print(f"modes present: {modes}  (a cohort that is all-ghost has no live evidence)")
    _report("DEEP (WETH/ETH)", [r for r in rows if r["deep"]])
    _report("THIN (everything else)", [r for r in rows if not r["deep"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
