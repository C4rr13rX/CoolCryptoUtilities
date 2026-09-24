#!/usr/bin/env python
"""Item 6285ff87 criterion 2 -- prove the RESTARTED process is running the NEW CODE.

A running process is not proof.  Production pid 20452 was alive, healthy and
writing 18856 trading_ops rows a day while serving code from 2026-09-23
06:23:01 that did not contain four of its own shipped fixes.  Three consecutive
passes measured against it and none of them noticed.

So this checks CODE MARKERS, not liveness.  Each marker is a row shape that
only post-fix code can write, and each had exactly ZERO rows across all of
history before the 2026-09-24 04:23:00 restart -- which is what makes a first
appearance conclusive rather than suggestive.

  3fa3478  _confidence_abstains  -> writes a "confidence_abstains" key onto
           every entry-conjunction block.  Absent from all 3298
           entry-refused-no-candidates rows ever written.
  365beac  the arbitration lookback-label fix -> emits
           "entry-deferred-label-horizon".  0 rows all-time.

The before/after rates are read from data/production_restart_baseline.json,
captured while the OLD process was still running.  Do not re-derive the "before"
side with a fresh query: any window reaching back past the restart mixes two
different builds, and the boot itself leaves a hole in the tape that depresses
every rate computed across it.

    python scripts/restart_verify.py

Exit code is 0 when at least one marker has fired, 2 while both are still
silent -- so it can be used as a wait condition.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

BASELINE = "data/production_restart_baseline.json"

MARKERS = (
    (
        "3fa3478 confidence abstain",
        "SELECT COUNT(*) FROM trading_ops WHERE details LIKE '%confidence_abstains%'",
        "entry-refused-no-candidates blocks stop being vetoed by a dead head",
    ),
    (
        "365beac label horizon",
        "SELECT COUNT(*) FROM trading_ops WHERE status LIKE '%label-horizon%'",
        "the arbitrator stops handing a tick to a lookback-labelled candidate",
    ),
)

AFTER_RATES = (
    ("trading_ops", "SELECT COUNT(*) FROM trading_ops WHERE ts>=?", "trading_ops_60min_before", 3600),
    ("entry-refused-no-candidates",
     "SELECT COUNT(*) FROM trading_ops WHERE ts>=? AND status='entry-refused-no-candidates'",
     "entry_refused_no_candidates_24h", 86400),
    ("ghost-entry", "SELECT COUNT(*) FROM trading_ops WHERE ts>=? AND status='ghost-entry'",
     "ghost_entry_24h", 86400),
    ("ghost-exit", "SELECT COUNT(*) FROM trading_ops WHERE ts>=? AND status='ghost-exit'",
     "ghost_exit_24h", 86400),
    ("trade_fills", "SELECT COUNT(*) FROM trade_fills WHERE ts>=?", "trade_fills_24h", 86400),
    ("trade_outcomes", "SELECT COUNT(*) FROM trade_outcomes WHERE ts>=?", "trade_outcomes_24h", 86400),
)


def main() -> int:
    if not os.path.exists(BASELINE):
        print("MISSING %s -- the 'before' side cannot be reconstructed. Do not "
              "substitute a fresh query: it would span the restart." % BASELINE)
        return 3
    base = json.loads(open(BASELINE, encoding="utf-8").read())
    restart_ts = float(base["captured_ts"])
    conn = sqlite3.connect(os.getenv("TRADING_DB_PATH", "storage/trading_cache.db"))
    now = time.time()
    elapsed = (now - restart_ts) / 60.0

    print("=" * 76)
    print("6285ff87 criterion 2 -- is the NEW CODE actually loaded?")
    print("  restart baseline captured %s  (%.1f min ago)" % (base["captured_at"], elapsed))
    print("  baseline HEAD %s   old pid %s booted %s"
          % (base.get("head_commit"), base.get("old_pid"), base.get("old_pid_start")))
    print("=" * 76)

    fired = 0
    print("\nCODE MARKERS -- each was 0 rows ALL-TIME before the restart:")
    for label, sql, means in MARKERS:
        n = conn.execute(sql).fetchone()[0]
        state = "FIRED" if n else "silent"
        print("  [%-6s] %-28s n=%-6d  %s" % (state, label, n, means))
        fired += 1 if n else 0

    print("\nACTIVITY SINCE THE RESTART:")
    n_ops = conn.execute("SELECT COUNT(*) FROM trading_ops WHERE ts>=?", (restart_ts,)).fetchone()[0]
    n_str = conn.execute("SELECT COUNT(*) FROM market_stream WHERE ts>=?", (restart_ts,)).fetchone()[0]
    print("  trading_ops   %6d" % n_ops)
    print("  market_stream %6d" % n_str)
    if not n_ops:
        print("\n  STILL BOOTING (or dead). Imports take ~13 min; check that the")
        print("  process exists and its working set is growing before assuming a hang:")
        print("    Get-CimInstance Win32_Process | Where-Object "
              "{ $_.CommandLine -like '*start_production*' }")
        print("\n  VERDICT: UNPROVEN -- no rows written yet. Not a failure, not a pass.")
        return 2

    print("\nBEFORE vs AFTER (before = the json, never a fresh query):")
    for label, sql, key, window in AFTER_RATES:
        before = base.get(key)
        after = conn.execute(sql, (now - window,)).fetchone()[0]
        note = ""
        if elapsed < window / 60.0:
            note = "  <- window %.0fh > %.0f min since restart: still contains pre-restart rows" % (
                window / 3600.0, elapsed)
        print("  %-28s before %-8s after %-8s%s" % (label, before, after, note))

    print("\n  VERDICT: %s" % (
        "NEW CODE CONFIRMED -- %d of %d markers fired" % (fired, len(MARKERS)) if fired
        else "WRITING BUT UNPROVEN -- rows are landing and NEITHER marker has fired. "
             "Either the paths that emit them were not reached yet, or the restart "
             "did not pick up the commits. Do not record a pass on this."))
    return 0 if fired else 2


if __name__ == "__main__":
    raise SystemExit(main())
