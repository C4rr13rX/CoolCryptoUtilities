#!/usr/bin/env python
"""Ghost-entry conversion census -- Bay, pass 693.

WHAT QUESTION THIS ANSWERS
--------------------------
scripts/graduation_status.py ladder step 3 GHOST prints [FAIL] as
"no ghost activity in 1h". That rung tells you nothing about WHY, and it
flaps: at the observed 24h mean of ~1 ghost entry per hour a Poisson 1h
window reads zero about a third of the time on a perfectly healthy lane.

This instrument separates the three things that can make step 3 fail, and
it is the normalisation that matters -- ghost entries per QUOTE-OK
CANDIDATE, not per hour:

  1. THROUGHPUT   the arbitration loop slowed down, so fewer candidates
                  are offered. Conversion per candidate is unchanged.
  2. CONVERSION   candidates keep arriving and clearing the quote step,
                  but a smaller share of them become entries. Something
                  downstream of the quote is refusing more.
  3. SAMPLING     nothing is wrong; the 1h window just missed.

Arm 1 and arm 3 are the null. Only arm 2 is a defect, and you cannot see
it in a per-hour count because throughput and conversion move together.

HYPOTHESIS AND COMPLEMENT (pre-registered, pass 693)
  HYPOTHESIS: the quote_ok -> ghost-entry step is broken; the silence
    survives normalisation by candidate count.
  COMPLEMENT: throughput alone fell, and per-candidate conversion is
    flat; the silence is arithmetic, not a defect.
  KILL TEST: take a healthy reference window, compute its conversion
    rate, multiply by the candidate count inside the silence, and ask
    Poisson how surprising zero entries is. If P(0) is large the
    hypothesis dies.

Read-only. Touches nothing but the trading store.

  python scripts/ghost_conversion_census.py
  python scripts/ghost_conversion_census.py --hours 48 --bucket 2
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import time

# The statuses trading_ops writes on the ghost entry path. These are the
# real spellings, read off the store -- `action` is enter/exit/hold and the
# lane lives in `status`, which is why an `op_type` query returns nothing.
CANDIDATE = "ghost_candidate_quote_ok"
ENTERED = "ghost-entry"
ARBITRATION = "entry-arbitration"

DEFAULT_DB = "storage/trading_cache.db"


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(
            f"trading store not found at {path!r}.\n"
            "data/trading.db, trading_bot.db and web/db.sqlite3 are 0-byte "
            "stubs that return nothing silently -- set TRADING_DB_PATH."
        )
    if os.path.getsize(path) < 1_000_000:
        raise SystemExit(
            f"{path} is {os.path.getsize(path)} bytes -- that is a stub, not "
            "the ~22GB trading store. Refusing to report zeros off it."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _count(conn, status, lo, hi):
    return conn.execute(
        "select count(*) from trading_ops where status=? and ts>=? and ts<?",
        (status, lo, hi),
    ).fetchone()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("TRADING_DB_PATH", DEFAULT_DB))
    ap.add_argument("--hours", type=float, default=24.0, help="census span")
    ap.add_argument("--bucket", type=float, default=1.0, help="bucket size, hours")
    ap.add_argument(
        "--ref",
        type=float,
        nargs=2,
        metavar=("OLDER_H", "NEWER_H"),
        default=(12.0, 5.0),
        help="healthy reference window, in hours-ago (older newer)",
    )
    args = ap.parse_args()

    conn = _connect(args.db)
    now = time.time()

    last = conn.execute(
        "select max(ts) from trading_ops where status=?", (ENTERED,)
    ).fetchone()[0]
    if last is None:
        print("no ghost-entry row has EVER been written. Nothing to normalise.")
        return 1

    gap_h = (now - last) / 3600.0
    print("=" * 74)
    print("GHOST-ENTRY CONVERSION CENSUS -- is step 3 GHOST a defect or arithmetic?")
    print("=" * 74)
    print(f"  store                {args.db}")
    print(f"  last {ENTERED:<16} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last))}")
    print(f"  silence             {gap_h:.2f} h")

    # ---- per-bucket conversion --------------------------------------
    print()
    print(f"  conversion by {args.bucket:g}h bucket (h ago, most recent first)")
    print("    h ago   quote_ok  ghost-entry   conv%")
    n_buckets = int(math.ceil(args.hours / args.bucket))
    series = []
    for b in range(n_buckets):
        lo, hi = now - (b + 1) * args.bucket * 3600, now - b * args.bucket * 3600
        q = _count(conn, CANDIDATE, lo, hi)
        g = _count(conn, ENTERED, lo, hi)
        conv = (100.0 * g / q) if q else None
        series.append((b, q, g, conv))
        shown = f"{conv:6.1f}" if conv is not None else "    --"
        print(f"    {b * args.bucket:5.0f}   {q:8d}  {g:11d}  {shown}")

    # ---- the kill test ----------------------------------------------
    older, newer = args.ref
    r_lo, r_hi = now - older * 3600, now - newer * 3600
    q_ref = _count(conn, CANDIDATE, r_lo, r_hi)
    g_ref = _count(conn, ENTERED, r_lo, r_hi)
    print()
    print(f"  REFERENCE WINDOW {older:g}h..{newer:g}h ago ({older - newer:g}h)")
    if not q_ref:
        print("    zero candidates in the reference window -- pick another --ref.")
        return 1
    conv_ref = g_ref / q_ref
    print(f"    quote_ok {q_ref}   ghost-entry {g_ref}   conversion {100 * conv_ref:.2f}%")

    # Candidates inside the silence. Open interval: the boundary row IS
    # the last entry, so counting from `last` inclusive would credit the
    # silence with the entry that ended the previous one.
    q_gap = _count(conn, CANDIDATE, last + 1e-6, now)
    g_gap = _count(conn, ENTERED, last + 1e-6, now)
    expected = q_gap * conv_ref
    print()
    print(f"  SILENCE WINDOW ({gap_h:.2f}h, open at the last entry)")
    print(f"    quote_ok {q_gap}   ghost-entry {g_gap}")
    print(f"    expected at reference conversion = {expected:.2f}")
    if expected > 0:
        p0 = math.exp(-expected)
        print(f"    P(observe 0 | Poisson mean {expected:.2f}) = {100 * p0:.2f}%")

    # ---- throughput, so arm 1 is visible next to arm 2 --------------
    print()
    print("  THROUGHPUT for contrast (arbitration rows per bucket)")
    print("    h ago   arbitration")
    for b in range(n_buckets):
        lo, hi = now - (b + 1) * args.bucket * 3600, now - b * args.bucket * 3600
        print(f"    {b * args.bucket:5.0f}   {_count(conn, ARBITRATION, lo, hi):11d}")

    # ---- verdict -----------------------------------------------------
    print()
    print("  VERDICT")
    if expected <= 0:
        print("    no candidates cleared the quote inside the silence. This is")
        print("    ARM 1 (throughput) or ARM 3 (sampling) -- the conversion step")
        print("    was never offered anything, so it cannot be blamed.")
    else:
        p0 = math.exp(-expected)
        if p0 > 0.05:
            print(f"    P(0) = {100 * p0:.2f}% is not surprising. The COMPLEMENT stands:")
            print("    throughput fell, per-candidate conversion did not. Step 3")
            print("    GHOST is arithmetic here, NOT a defect. Do not go hunting.")
        else:
            print(f"    P(0) = {100 * p0:.2f}%. The silence SURVIVES normalisation by")
            print(f"    candidate count: {q_gap} candidates cleared the quote and")
            print("    produced none. The defect is DOWNSTREAM OF THE QUOTE, and it")
            print("    is not a throughput artifact and not a sampling artifact.")
    print()
    print("  CAVEAT, stated because this number invites over-reading: a Poisson")
    print("  model assumes the reference conversion rate is constant. The bucket")
    print("  table above is how you check that assumption -- if conversion is")
    print("  DECAYING rather than cliff-edged, the reference window is itself")
    print("  part of the decline and the true expected count is higher than")
    print("  printed, which makes P(0) an UPPER bound on the surprise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
