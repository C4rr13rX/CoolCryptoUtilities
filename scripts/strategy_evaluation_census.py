#!/usr/bin/env python
"""Are the other strategies STARVED, or do they DECLINE? -- Echo, pass 486.

Item [583ba889] read as a fairness complaint: atf_static takes 179 of ~186
decision cycles, so the other strategies never get a look in. If that were
true, unstarving them would yield a dozen measurable signal candidates -- and
after pass 486 closed atf_static from both ends (entry +2.69bp against a
27.77bp toll, all 28 exit cells negative), a new signal is the only thing left
that can move the scoreboard. So the branch matters.

  H1 STARVED:   the others are rarely offered a decision. Scheduler item.
  H0 DECLINING: they are offered decisions and refuse. Then their low proposal
                count is their own selectivity, a scheduler fix helps nothing,
                and the roster question is a finding rather than a change.

THE SOURCE, and finding it is half the job. Declines are 99.2% uninstrumented
-- 71 of 72 strategies emit the bare literal no_signal -- so an evaluation
count CANNOT come from skipped{}. It also cannot come from the refusal-status
rows, none of which name a strategy. But trading_ops rows with action='hold'
DO carry strategy_id in their details JSON, for more distinct strategies than
ever propose. That is the evaluation count, and it is already on disk.

  evaluations(strategy) = enter rows + hold rows carrying that strategy_id
  enter%                = proposals / evaluations

A strategy with many evaluations and few proposals is SELECTIVE. A strategy
with few evaluations is STARVED. The two are distinguishable in one table.

READ-ONLY. Opens the store with mode=ro. Changes nothing.

Run:  python -X utf8 scripts/strategy_evaluation_census.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections import Counter

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
LOOKBACK_DAYS = 7.0
RANKABLE = 30  # entries needed before a per-strategy skill CI means anything


def main() -> int:
    conn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    since = time.time() - LOOKBACK_DAYS * 86400.0

    ent, hold = Counter(), Counter()
    rows_seen, rows_attr = 0, 0
    for r in conn.execute(
        "SELECT action, details FROM trading_ops WHERE ts > ? AND action IN ('enter','hold')",
        (since,),
    ):
        rows_seen += 1
        try:
            j = json.loads(r["details"] or "{}")
        except Exception:
            continue
        sid = j.get("strategy_id")
        if not sid:
            continue
        rows_attr += 1
        (ent if r["action"] == "enter" else hold)[sid] += 1

    names = set(ent) | set(hold)
    print("=" * 72)
    print("STRATEGY EVALUATION CENSUS -- item [583ba889], Echo pass 486")
    print("lookback %.0fd | enter+hold rows %d | attributed %d | strategies %d"
          % (LOOKBACK_DAYS, rows_seen, rows_attr, len(names)))
    print("=" * 72)
    print("%-28s %8s %8s %9s %8s" % ("strategy", "enter", "hold", "evals", "enter%"))
    print("-" * 72)
    for s in sorted(names, key=lambda s: -(ent[s] + hold[s]))[:22]:
        t = ent[s] + hold[s]
        print("%-28s %8d %8d %9d %7.1f%%"
              % (s[:28], ent[s], hold[s], t, 100.0 * ent[s] / t if t else 0.0))
    if len(names) > 22:
        print("... %d more" % (len(names) - 22))

    top = max(names, key=lambda s: ent[s] + hold[s]) if names else None
    others = [s for s in names if s != top]
    print("-" * 72)
    print("%-28s evaluations %d" % (top, ent[top] + hold[top]))
    print("all %d OTHER strategies combined  evaluations %d"
          % (len(others), sum(ent[s] + hold[s] for s in others)))
    print("strategies with >=%d evaluations : %d"
          % (RANKABLE, sum(1 for s in names if ent[s] + hold[s] >= RANKABLE)))
    print("strategies with >=%d ENTRIES     : %d"
          % (RANKABLE, sum(1 for s in names if ent[s] >= RANKABLE)))

    other_evals = sum(ent[s] + hold[s] for s in others)
    print()
    if top and other_evals >= (ent[top] + hold[top]):
        print("VERDICT: H0, DECLINING. The other strategies receive MORE evaluations")
        print("         in aggregate than the dominant one. They are not starved --")
        print("         they are evaluated and refuse. A scheduler fix helps nothing.")
        print("         Their low proposal count is their own selectivity.")
    else:
        print("VERDICT: H1, STARVED. The dominant strategy takes more evaluations than")
        print("         all others combined. A scheduler change is on the critical path.")
    if top and (ent[top] + hold[top]):
        pct = 100.0 * ent[top] / (ent[top] + hold[top])
        if pct >= 95.0:
            print()
            print("AND NOTE WHAT %s IS: it enters on %.1f%% of the ticks it sees" % (top, pct))
            print("(%d holds in %.0f days). That is a near-unconditional BUYER, not a"
                  % (hold[top], LOOKBACK_DAYS))
            print("selective signal -- which is why its edge over the tape is tape-level,")
            print("why its expected_return is a constant, and why the entry gate carries")
            print("no information. Those are ONE finding, not four.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
