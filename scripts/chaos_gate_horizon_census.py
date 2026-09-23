#!/usr/bin/env python3
"""Prove -- or refute -- that the chaos gate reads a strategy's LOOKBACK
SUFFIX as its forecast horizon.

THE CLAIM
---------
``entry-refused-lattice`` rows carry a detail like::

    chaos: CAKE-USDC: information decays after 5.4 min but the signal looks
    10080.0 min ahead -- 1880.6x past where prediction is possible

10080 minutes is exactly one week, and the strategy was ``supertrend_follow@1w``.
The suspicion is that "how far ahead the signal looks" is the strategy id's
horizon label parsed into minutes, not a forecast horizon at all.

THE TEST, AND WHY IT IS DECISIVE
--------------------------------
Group the extracted "X min ahead" figure by the strategy's suffix and print
the DISTINCT values of X in each group. A genuine forecast horizon varies with
the signal, so a real one gives MANY distinct values per group. A parsed label
gives exactly ONE, equal to the suffix.

Measured 2026-09-22 over 7 days, 6893 rows, 100% of them ``chaos:``::

    @1w   n=1749  X = {10080.0}      10080 min = 7 days
    @5d   n=1584  X = {7200.0}        7200 min = 5 days
    @3d   n=972   X = {4320.0}        4320 min = 3 days
    @1d   n=555   X = {1440.0}        1440 min = 1 day
    @12h  n=1107  X = {720.0}          720 min = 12 hours
    @5h   n=684   X = {300.0}          300 min = 5 hours

One value each, and each is its own label.

THE COMPLEMENT, which this script also prints rather than leaving to the
reader: maybe those strategies really do forecast a week ahead and the gate is
refusing them CORRECTLY. Two things in the same output kill it.

  * the NO-SUFFIX group (n=196) shows X in {20.0, 45.0, 60.0} -- real, VARYING
    horizons, at a median impossibility ratio of 6.2x against 3453.0x for
    ``@1w``. The field CAN carry a true horizon; it degenerates to a constant
    only when there is a suffix to misread.
  * ``donchian_breakout@1w`` denotes a one-week Donchian CHANNEL, a lookback.
    ``.env:169 MAX_HOLD_FORCE_SECONDS=2700`` caps holds at 45 minutes, so a
    10080-minute forecast horizon is meaningless on its face.

The impossibility ratio is monotone in the label and in nothing else, which is
the signature of a label being read where a measurement belongs.

Confirms the memory item ``chaos-gate-refuses-a-lookback-label-as-a-forecast``,
which held the assertion but not this measurement. Backlog item c813f5fe.

This script is READ-ONLY and needs no restart. A FIX does -- production serves
stale code until restarted.

USAGE
    python -X utf8 scripts/chaos_gate_horizon_census.py
    python -X utf8 scripts/chaos_gate_horizon_census.py --days 14
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import statistics
import time

STATUS = "entry-refused-lattice"
AHEAD_RE = re.compile(r"signal looks ([\d.]+) min ahead")
RATIO_RE = re.compile(r"([\d.]+)x past")

# Minutes each suffix denotes, so the output can check itself rather than
# asking the reader to do the arithmetic.
SUFFIX_MINUTES = {
    "@1w": 7 * 24 * 60,
    "@5d": 5 * 24 * 60,
    "@3d": 3 * 24 * 60,
    "@1d": 24 * 60,
    "@12h": 12 * 60,
    "@5h": 5 * 60,
}
ORDER = ["@1w", "@5d", "@3d", "@1d", "@12h", "@5h"]


def db_path() -> str:
    return os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def suffix_of(strategy_id: str) -> str:
    return "@" + strategy_id.split("@", 1)[1] if "@" in strategy_id else "NO-SUFFIX"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    path = db_path()
    if not os.path.exists(path):
        print(f"no trading store at {path} -- set TRADING_DB_PATH")
        return 2

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cutoff = time.time() - args.days * 86400
    # NOTE: the column is `ts` (a float), not `timestamp`. A date-string WHERE
    # here returns zero rows silently.
    cur.execute("SELECT details FROM trading_ops WHERE status=? AND ts>?", (STATUS, cutoff))
    rows = cur.fetchall()

    ahead: dict[str, set[float]] = collections.defaultdict(set)
    ratios: dict[str, list[float]] = collections.defaultdict(list)
    counts: collections.Counter = collections.Counter()
    chaos = 0

    for (details,) in rows:
        try:
            payload = json.loads(details or "{}")
        except (TypeError, ValueError):
            continue
        detail = payload.get("detail") or ""
        if not detail.startswith("chaos:"):
            continue
        chaos += 1
        sfx = suffix_of(payload.get("strategy_id") or "")
        counts[sfx] += 1
        if (m := AHEAD_RE.search(detail)) is not None:
            ahead[sfx].add(float(m.group(1)))
        if (r := RATIO_RE.search(detail)) is not None:
            ratios[sfx].append(float(r.group(1)))

    total = len(rows)
    if not total:
        print(f"no {STATUS} rows in {args.days}d -- nothing to census.")
        return 0

    print(f"{STATUS} rows in {args.days}d: {total}")
    print(f"  of which detail starts with 'chaos:': {chaos} ({100 * chaos / total:.1f}%)")
    print(f"  refusal rate: {chaos / args.days:.0f}/day\n")

    print(f"{'suffix':10s} {'n':>5s} {'median_ratio':>13s}  DISTINCT 'min ahead' VALUES   VERDICT")
    suffixed = 0
    for sfx in ORDER + ["NO-SUFFIX"]:
        if sfx not in counts:
            continue
        values = sorted(ahead[sfx])
        med = statistics.median(ratios[sfx]) if ratios[sfx] else float("nan")
        expected = SUFFIX_MINUTES.get(sfx)
        if expected is not None:
            suffixed += counts[sfx]
            if values == [float(expected)]:
                verdict = f"IS THE LABEL ({expected} min)"
            else:
                verdict = "varies -- NOT the label"
        else:
            verdict = "varies (real horizons)" if len(values) > 1 else "single value"
        shown = values[:4] + (["..."] if len(values) > 4 else [])
        print(f"{sfx:10s} {counts[sfx]:5d} {med:13.1f}  {str(shown):28s}  {verdict}")

    print()
    if chaos:
        print(f"suffixed strategies carry {suffixed} of {chaos} chaos refusals ({100 * suffixed / chaos:.1f}%)")
    degenerate = [s for s in ORDER if s in counts and sorted(ahead[s]) == [float(SUFFIX_MINUTES[s])]]
    if len(degenerate) >= 2:
        print(
            f"VERDICT: {len(degenerate)} of {len([s for s in ORDER if s in counts])} suffix groups collapse to a\n"
            "  SINGLE 'min ahead' value equal to their own label. A forecast horizon varies\n"
            "  with the signal; a parsed label does not. The gate is reading the LOOKBACK.\n"
            "  Complement check: the NO-SUFFIX group above should show SEVERAL values at a\n"
            "  far lower ratio -- if it does, the field can carry a true horizon and only\n"
            "  degenerates where a suffix exists to misread."
        )
    else:
        print(
            "VERDICT: suffix groups do NOT collapse to their label. The claim behind item\n"
            "  c813f5fe does not reproduce on this window -- say so rather than requoting it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
