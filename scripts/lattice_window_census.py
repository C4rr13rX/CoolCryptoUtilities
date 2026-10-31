#!/usr/bin/env python3
"""[bfe06f90] Is the lattice's 64-sample threshold REACHABLE?

THE QUESTION. `trading/bot.py:2402` refuses to run the chaos layer when
`len(prices) < 64`, stamping `short_window:{len(prices)}`. Over 7d that exit is
30 of 45 readable ghost entries (66.7%), at the discrete values 20, 21 and 60.

    H           the threshold sits above anything the call site can ever
                produce, so `short_window` is a CONSTANT problem.
    COMPLEMENT  the buffer faithfully mirrors a genuinely thin tape, and disk
                would show the same counts -- a FEED problem no constant fixes.

These have opposite fixes. This script tells them apart and does not edit the
trading path: the disposition on [0d7f5a6f] is LEAVE THE GUARD.

WHY THE ANSWER IS STRUCTURAL AND NOT STATISTICAL. The prices the guard counts
come from an in-process ring buffer, not from the 35GB store:

    bot.py:603   self._brain_window = max(self.window_size,
                     max((h for _, h in self.swarm.horizon_defs), default=...))
    bot.py:607   self._buffer = deque(maxlen=self._brain_window)
    bot.py:2400  history = list(self._buffer)[-400:]
    bot.py:2402  if len(prices) < 64:  ->  stamp short_window, return None

`len(deque(maxlen=N))` cannot exceed N, so the `[-400:]` slice is dead whenever
N < 400 and the count the guard tests is bounded by `_brain_window`. With
BOT_WINDOW_SIZE defaulting to 20 (bot.py:536) the bound is below 64 and the
comparison can never be satisfied -- on any symbol, at any tape density, for as
long as the process runs.

`self._buffer` is also SHARED, not per-symbol. The per-symbol deques are
`self._price_history` at bot.py:608, and the lattice never opens them.

This script reports, from first-hand sources only:
  1. the stamped short_window values actually observed, from trading_ops;
  2. the on-disk tape available to the SAME symbol over the SAME window, from
     market_stream -- the arm that decides H against its complement;
  3. the structural bound, printed beside the threshold.

Run:  python scripts/lattice_window_census.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
THRESHOLD = 64  # trading/bot.py:2402, hardcoded, no env var
WINDOW_DAYS = 7


def _lattice_exit(raw: str | None):
    """Read the stamp from TOP-LEVEL details.lattice_exit.

    NOT from `bus_plan`. 1199 rows over 30d carry a bus_plan dict and zero of
    them contain lattice_exit, so reading that path returns a clean, plausible
    zero that looks exactly like an instrument that stopped emitting.
    """
    if not raw:
        return None
    try:
        det = json.loads(raw)
    except Exception:  # noqa: BLE001 - a malformed row must not kill the census
        return None
    if not isinstance(det, dict):
        return None
    return det.get("lattice_exit")


def main() -> int:
    if not os.path.exists(DB):
        print("no store at %s" % DB)
        return 1
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    q = con.execute

    # trading_ops.ts is a UNIX EPOCH FLOAT, not a datetime string. Comparing it
    # to datetime('now','-7 days') silently returns zero rows, and sqlite's
    # now() is UTC while this box is UTC-4, so it is wrong twice.
    now = q("SELECT MAX(ts) FROM trading_ops").fetchone()[0]
    cutoff = now - WINDOW_DAYS * 86400

    rows = q(
        "SELECT ts, symbol, details FROM trading_ops "
        "WHERE action='enter' AND status='ghost-entry' AND ts>=? ORDER BY ts",
        (cutoff,),
    ).fetchall()

    stamped = []
    for ts, sym, raw in rows:
        ex = _lattice_exit(raw)
        if ex:
            stamped.append((ts, sym, str(ex)))

    short = [(ts, sym, ex) for ts, sym, ex in stamped if ex.startswith("short_window:")]

    print("=" * 74)
    print("[bfe06f90] LATTICE WINDOW CENSUS -- is the 64-sample threshold reachable?")
    print("=" * 74)
    print("store            : %s" % DB)
    print("window           : %d days, %d ghost-entry rows, %d stamped"
          % (WINDOW_DAYS, len(rows), len(stamped)))
    print("threshold        : len(prices) < %d  (trading/bot.py:2402, hardcoded)" % THRESHOLD)
    print()

    print("-- 1. STAMPED EXITS, from trading_ops.details.lattice_exit ------------")
    for ex, n in Counter(e for _, _, e in stamped).most_common():
        print("   %4d  %s" % (n, ex))
    print()

    # The observed short_window integers are the count the guard actually had.
    vals = sorted({int(e.split(":", 1)[1]) for _, _, e in short if e.split(":", 1)[1].isdigit()})
    print("-- 2. THE COUNTS THE GUARD ACTUALLY HAD ------------------------------")
    print("   distinct observed len(prices) : %s" % (vals or "none"))
    print("   max observed                  : %s" % (max(vals) if vals else "n/a"))
    print("   threshold                     : %d" % THRESHOLD)
    if vals and max(vals) < THRESHOLD:
        print("   -> every observed count is BELOW the threshold. Never satisfied.")
    print()

    # Per-symbol: if each symbol always reports the same integer, the number is
    # a fixed capacity being reported, not a measurement of available tape.
    per = defaultdict(set)
    for _, sym, ex in short:
        part = ex.split(":", 1)[1]
        if part.isdigit():
            per[sym].add(int(part))
    print("-- 3. IS THE COUNT A PROPERTY OF THE SYMBOL OR OF THE BUFFER? --------")
    fixed = sum(1 for s in per if len(per[s]) == 1)
    print("   symbols with a stamped short_window : %d" % len(per))
    print("   symbols reporting ONE fixed value   : %d" % fixed)
    for sym in sorted(per)[:12]:
        print("      %-16s %s" % (sym, sorted(per[sym])))
    print()

    # THE ARM THAT DECIDES IT. For each stamped short_window row, count the tape
    # on DISK for that symbol over the window the buffer is supposed to cover.
    # If disk holds >= 64 where the buffer held 20, the fail-open is a
    # WRONG-SOURCE bug and the 64 is innocent.
    print("-- 4. KILL TEST: the tape ON DISK for the same symbol and window -----")
    print("   %-16s %8s %10s %10s" % ("symbol", "buffer", "disk 1h", "disk 24h"))
    verdict_disk_richer = 0
    checked = 0
    for ts, sym, ex in short[-12:]:
        part = ex.split(":", 1)[1]
        if not part.isdigit():
            continue
        buf = int(part)
        d1 = q(
            "SELECT COUNT(*) FROM market_stream WHERE symbol=? AND ts>=? AND ts<=?",
            (sym, ts - 3600, ts),
        ).fetchone()[0]
        d24 = q(
            "SELECT COUNT(*) FROM market_stream WHERE symbol=? AND ts>=? AND ts<=?",
            (sym, ts - 86400, ts),
        ).fetchone()[0]
        checked += 1
        if d24 >= THRESHOLD:
            verdict_disk_richer += 1
        print("   %-16s %8d %10d %10d" % (sym, buf, d1, d24))
    print()
    print("   rows where DISK held >= %d in 24h : %d of %d"
          % (THRESHOLD, verdict_disk_richer, checked))
    print()

    print("-- 5. THE STRUCTURAL BOUND -------------------------------------------")
    print("   bot.py:607   self._buffer = deque(maxlen=self._brain_window)")
    print("   bot.py:603   _brain_window = max(window_size, max horizon_defs)")
    print("   bot.py:536   window_size default = int(os.getenv('BOT_WINDOW_SIZE','20'))")
    print("   bot.py:2400  history = list(self._buffer)[-400:]   <- slice is dead")
    print("   bot.py:2402  if len(prices) < %d: stamp and return None" % THRESHOLD)
    print()
    print("   len(deque(maxlen=N)) <= N. The guard tests a count bounded by")
    print("   _brain_window, so whenever _brain_window < %d the comparison is" % THRESHOLD)
    print("   UNSATISFIABLE regardless of how much tape exists on disk.")
    print()
    print("   self._buffer is SHARED across symbols (bot.py:607, a single deque).")
    print("   The per-symbol deques are self._price_history (bot.py:608) and the")
    print("   lattice never opens them.")
    print()

    print("=" * 74)
    if vals and max(vals) < THRESHOLD and verdict_disk_richer == checked and checked:
        print("VERDICT: H HOLDS. The threshold is unreachable at the call site.")
        print("  Every observed count is a BUFFER CAPACITY, not a tape measurement,")
        print("  and disk held >= %d for every row checked. The complement -- 'the" % THRESHOLD)
        print("  tape is genuinely thin' -- is refuted: the tape is there and the")
        print("  guard does not read it.")
    else:
        print("VERDICT: NOT ESTABLISHED by this run -- read the arms above.")
    print()
    print("RECOMMENDATION (no edit made; any trading-path change is a separate")
    print("item Echo approves): the defect is the SOURCE, not the constant. The")
    print("guard should read the per-symbol history it already has on disk, or")
    print("self._price_history[symbol], rather than the shared brain buffer.")
    print("Raising _brain_window to 64 would also satisfy the comparison but")
    print("would feed the estimator a cross-symbol mixture, which is worse: a")
    print("shared buffer's last 64 samples span many symbols.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
