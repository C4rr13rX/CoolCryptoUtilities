#!/usr/bin/env python3
"""Split ``trading_ops`` ghost rows by the executor that WROTE them, and price
the scout's candidate->entry conversion as a Bernoulli trial per candidate.

WHY THIS EXISTS
---------------
Two different executors write ``status='ghost-entry'`` into the same table and
nothing in the schema tells them apart:

  * the ATF scout -- ``services/atf_static_strategy.py:1247`` -- which writes
    ``ghost-entry`` DIRECTLY and, by its own comment at line 1148, "never
    passes through trading/bot.py";
  * the classical path -- ``trading/bot.py:9794`` -- reached through the CDCL
    solver, carrying a ``bus_plan``.

Measured 2026-09-22 over 30 days: 1714 ``ghost-entry`` rows, of which 521 are
the scout (``details.source == 'c0d3rv2_atf_static'``) and 1193 are classical
(a ``bus_plan`` key, no ``source``). Meanwhile ALL 1938
``ghost_candidate_quote_ok`` rows in 7 days are the scout's and not one of
them carries a ``bus_plan``.

That last fact is the point of this script. ``ghost_candidate_quote_ok`` is a
SCOUT-ONLY status, so pooling it against the pooled ``ghost-entry`` count
mixes a scout numerator's denominator with a mostly-classical numerator. Read
pooled, the funnel says "N quote-OK candidates died before an entry" when many
of those entries were never downstream of any quote_ok row. Several items in
this repo's backlog are argued from that pooled ratio.

WHAT IT PRINTS
--------------
  1. a daily table splitting ``ghost-entry`` into SCOUT and CLASSICAL, beside
     the scout's own ``ghost_candidate_quote_ok`` supply;
  2. the scout's conversion rate over a baseline window, and an exact binomial
     tail for a recent window against it -- one trial per candidate, which is
     the right unit. Counting per DAY instead invites the burst artifact that
     ``ladder-step-3-ghost-is-a-threshold-artifact`` already warns about: the
     median inter-arrival gap between ghost entries is 0.01h because they
     arrive in clumps, so any window chosen around a clump is circular.

A CAVEAT THE OUTPUT REPEATS, because it is easy to over-read: if the recent
window is chosen because it is a run of zeros, the tail probability is
suggestive and not a hypothesis test. Pass ``--recent-days`` deliberately and
say which you picked.

USAGE
    python -X utf8 scripts/ghost_entry_producer_split.py
    python -X utf8 scripts/ghost_entry_producer_split.py --days 21 --recent-days 3
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

SCOUT_SOURCE = "c0d3rv2_atf_static"
ENTRY_STATUS = "ghost-entry"
CANDIDATE_STATUS = "ghost_candidate_quote_ok"


def db_path() -> str:
    return os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def _is_scout(details: str | None) -> bool:
    """True if this row was written by the ATF scout.

    The scout stamps ``source``; the classical path stamps none and carries a
    ``bus_plan`` instead. A row with neither marker is counted as classical,
    which is the conservative direction here -- it can only DEFLATE the scout's
    measured conversion, never inflate it.
    """
    try:
        payload = json.loads(details or "{}")
    except (TypeError, ValueError):
        return False
    return payload.get("source") == SCOUT_SOURCE


def split_entries(cur, lo: float, hi: float) -> tuple[int, int]:
    cur.execute(
        "SELECT details FROM trading_ops WHERE status=? AND ts>=? AND ts<?",
        (ENTRY_STATUS, lo, hi),
    )
    scout = classical = 0
    for (details,) in cur.fetchall():
        if _is_scout(details):
            scout += 1
        else:
            classical += 1
    return scout, classical


def candidates(cur, lo: float, hi: float) -> int:
    cur.execute(
        "SELECT COUNT(*) FROM trading_ops WHERE status=? AND ts>=? AND ts<?",
        (CANDIDATE_STATUS, lo, hi),
    )
    return int(cur.fetchone()[0])


def binom_tail_zero(p: float, n: int) -> float:
    """P(exactly zero successes | p, n). Exact, and the only tail we need --
    the interesting case is always a silent window."""
    return (1.0 - p) ** n if 0.0 <= p <= 1.0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=14, help="table length in days")
    ap.add_argument(
        "--recent-days",
        type=int,
        default=3,
        help="window tested against the baseline; the rest of --days is the baseline",
    )
    args = ap.parse_args()

    if args.recent_days >= args.days:
        print("--recent-days must be smaller than --days; there would be no baseline.")
        return 2

    path = db_path()
    if not os.path.exists(path):
        print(f"no trading store at {path} -- set TRADING_DB_PATH")
        return 2

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    now = time.time()

    print(f"trading store: {path}")
    print(f"scout rows are details.source == {SCOUT_SOURCE!r}; all others counted classical\n")
    print("days_ago   SCOUT-entry   CLASSICAL-entry   scout_quote_ok")

    recent_c = recent_e = base_c = base_e = 0
    for d in range(args.days):
        lo, hi = now - (d + 1) * 86400, now - d * 86400
        scout, classical = split_entries(cur, lo, hi)
        cands = candidates(cur, lo, hi)
        print(f"   -{d:<3d}     {scout:7d}        {classical:9d}      {cands:10d}")
        if d < args.recent_days:
            recent_e += scout
            recent_c += cands
        else:
            base_e += scout
            base_c += cands

    print()
    if base_c == 0:
        print("baseline window has no scout candidates -- nothing to compare against.")
        return 0

    rate = base_e / base_c
    print(
        f"SCOUT baseline (days {args.recent_days}-{args.days - 1}): "
        f"{base_e} entries on {base_c} quote-OK candidates = {rate * 100:.3f}%"
    )
    print(f"SCOUT recent   (days 0-{args.recent_days - 1}): {recent_e} entries on {recent_c} candidates")
    print(f"expected at the baseline rate: {recent_c * rate:.2f}")
    if recent_e == 0 and recent_c:
        tail = binom_tail_zero(rate, recent_c)
        print(f"P(exactly zero | p={rate:.5f}, n={recent_c}) = {tail:.4g}")
        print(
            "  READ THIS CAREFULLY: if you chose --recent-days because it was a run of\n"
            "  zeros, that number is suggestive, not a test. Say which window you picked\n"
            "  and why when you quote it."
        )

    # The classical path is reported beside it so nobody reads a scout-only
    # silence as a system-wide one -- that mistake costs a pass.
    lo_recent = now - args.recent_days * 86400
    _, cl_recent = split_entries(cur, lo_recent, now)
    _, cl_base = split_entries(cur, now - args.days * 86400, lo_recent)
    base_days = args.days - args.recent_days
    print(
        f"\nCLASSICAL path for contrast: {cl_recent / args.recent_days:.2f} entries/day recent "
        f"vs {cl_base / base_days:.2f}/day baseline "
        f"({cl_recent} over {args.recent_days}d, {cl_base} over {base_days}d)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
