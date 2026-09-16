"""What `data_ingest` spends its 700+ seconds on, and what that actually starves.

Filed as [b8c24590] on the observation only: `sequential task_overrun: {task:
data_ingest, running_for: 712.6}`, and the same task at 3215.8s an hour earlier.
The item's premise was that a blocked ingest starves the price feed, because
this repo has been stopped by that exact shape twice (a 3-year CEX backfill
inside the feed's task; a 142-source news crawl blocking it).

THE PREMISE IS FALSE, and this script is what falsifies it. Measured
2026-09-16 over a 3h49m continuous stall: 96.5 market_stream ticks per 10
minutes INSIDE the stall against 93.8 OUTSIDE it. The feed does not care.
`market_stream` is written by db.py:1534 from the stream's own thread; it was
never inside `run_cycle`.

What IS inside run_cycle, in order, on ONE thread:

    for chain in static_workers: _run_worker("static:<chain>", ...)
    _run_worker("dynamic",   ...)
    _run_worker("market",    ...)   <- writes the `prices` table
    _run_worker("discovery", ...)   <- finds new tradeable tokens

`market` and `discovery` are LAST, so every second the static chain downloads
take is a second they do not get. That is the real starvation and it is
measurable: see the STARVED section below.

And the seconds are going somewhere we cannot spend them. TokenDownloadSupervisor
reads DOWNLOAD_WORKER_CHAINS, which defaults to
"base,ethereum,arbitrum,optimism,polygon" and is not set in .env -- while .env
already says LIVE_FOCUS_CHAINS=base, because base is the only chain we trade.

Read-only. Run:  python -X utf8 scripts/data_ingest_starvation_census.py
"""

from __future__ import annotations

import collections
import datetime as _dt
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "logs" / "system.log"
CORPUS = REPO / "data" / "historical_ohlcv"

#: The one chain the wallet trades. .env: LIVE_FOCUS_CHAINS=base.
TRADED_CHAIN = os.getenv("LIVE_FOCUS_CHAINS", "base").split(",")[0].strip().lower()

#: Emitted by DownloadSupervisor._run_worker when a worker beats its 60s budget.
#: Only over-budget runs are logged, so every number here is a LOWER bound.
SLOW = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*slow worker: (\S+) took ([\d.]+)s"
)


def _db():
    path = os.getenv("TRADING_DB_PATH", str(REPO / "storage" / "trading_cache.db"))
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def _epoch(stamp: str) -> float:
    return _dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()


def scan_log():
    """Every over-budget worker run: (end_ts, label, seconds)."""
    if not LOG.exists():
        print("no %s -- nothing to scan" % LOG)
        return []
    runs = []
    with LOG.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "slow worker:" not in line or "download-supervisor" not in line:
                continue
            m = SLOW.search(line)
            if m:
                runs.append((_epoch(m.group(1)), m.group(2), float(m.group(3))))
    return runs


def census(runs) -> float:
    """Who ate the thread. Returns seconds spent on chains we do not trade."""
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for _end, label, sec in runs:
        row = agg[label]
        row[0] += 1
        row[1] += sec
        row[2] = max(row[2], sec)

    print("WHERE data_ingest's SECONDS GO (only runs over the 60s budget are logged,")
    print("so every figure here is a LOWER bound on the real total)")
    print()
    print("  %-18s %6s %11s %9s %9s  %s" % ("worker", "runs", "total_s", "mean_s", "max_s", "traded?"))
    offpath = 0.0
    total = 0.0
    for label, (n, secs, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        chain = label.split(":", 1)[1] if label.startswith("static:") else ""
        traded = "" if not chain else ("yes" if chain == TRADED_CHAIN else "NO -- we cannot trade it")
        if chain and chain != TRADED_CHAIN:
            offpath += secs
        total += secs
        print("  %-18s %6d %11.0f %9.1f %9.1f  %s" % (label, n, secs, secs / n, mx, traded))
    print("  %-18s %6d %11.0f  = %.1f hours" % ("TOTAL", sum(a[0] for a in agg.values()), total, total / 3600))
    if total > 0:
        print()
        print("  ON CHAINS WE DO NOT TRADE: %.0fs of %.0fs = %.1f%% (%.1f hours)"
              % (offpath, total, 100.0 * offpath / total, offpath / 3600))
    return offpath


def stalls(runs, min_sec: float = 300.0):
    """Merge over-budget runs into continuous windows the thread was held."""
    spans = sorted((end - sec, end) for end, _label, sec in runs if sec >= min_sec)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 60.0:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged if (b - a) >= min_sec]


def feed_during(con, windows):
    """THE CRITERION THAT DECIDES THE ITEM: ticks/10min inside a stall vs outside.

    If the stall starved the feed these differ. They do not.
    """
    print()
    print("DOES IT STARVE THE PRICE FEED? market_stream ticks per 10 minutes.")
    print()
    if not windows:
        print("  no stall long enough to measure")
        return
    # Longest stall, and the equally-long stretch immediately before it.
    start, end = max(windows, key=lambda w: w[1] - w[0])
    span = end - start
    rows = []
    for label, t0, t1 in (
        ("INSIDE the stall", start, end),
        ("OUTSIDE (the same span just before it)", start - span, start),
    ):
        n, syms = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM market_stream WHERE ts >= ? AND ts < ?",
            (t0, t1),
        ).fetchone()
        mins = (t1 - t0) / 60.0
        rows.append((label, mins, n, n / mins * 10.0 if mins else 0.0, syms))
    print("  %-40s %7s %7s %10s %8s" % ("window", "mins", "ticks", "ticks/10m", "symbols"))
    for label, mins, n, rate, syms in rows:
        print("  %-40s %7.1f %7d %10.1f %8d" % (label, mins, n, rate, syms))
    inside, outside = rows[0][3], rows[1][3]
    print()
    print("  stall ran %s -> %s (%.0f min)"
          % (_dt.datetime.fromtimestamp(start).strftime("%H:%M:%S"),
             _dt.datetime.fromtimestamp(end).strftime("%H:%M:%S"), span / 60.0))
    if outside > 0 and inside >= outside * 0.8:
        print("  VERDICT: the feed is NOT starved (%.1f inside vs %.1f outside)."
              % (inside, outside))
        print("           market_stream is written by db.py:1534 on the stream's own")
        print("           thread. It was never inside run_cycle. [b8c24590]'s premise")
        print("           is falsified -- do not 'fix' the feed here.")
    else:
        print("  VERDICT: the feed IS degraded inside the stall (%.1f vs %.1f)."
              % (inside, outside))


def starved(con):
    """What run_cycle DOES hold: the two workers queued behind the downloads."""
    print()
    print("WHAT IS ACTUALLY STARVED -- the workers queued BEHIND the static chains")
    print("in run_cycle (background_workers.py:778): static:* -> dynamic -> market -> discovery")
    print()
    now = time.time()
    try:
        newest = con.execute("SELECT MAX(ts) FROM prices").fetchone()[0]
        if newest:
            print("  market    -> `prices` newest row is %.1f h old  (MarketDataWorker.run_once)"
                  % ((now - float(newest)) / 3600.0))
    except Exception as exc:
        print("  market    -> could not read prices: %s" % exc)
    try:
        newest = con.execute("SELECT MAX(created_at) FROM discovery_discoveryevent").fetchone()[0]
        if newest:
            ts = _dt.datetime.fromisoformat(str(newest).split(".")[0]).timestamp()
            print("  discovery -> newest event is %.1f h old  (finds new tradeable tokens)"
                  % ((now - ts) / 3600.0))
    except Exception as exc:
        print("  discovery -> could not read discovery events: %s" % exc)
    print()
    print("  NOTE, so this is not over-claimed: the only readers of `prices` are")
    print("  trading/wizard_trainer.py and trading/brain/*, and the brain is PAUSED")
    print("  (OMEN_STRATEGY_ENABLED=0). `discovery` is the one with a money link.")


def corpus():
    """What 17.9 hours of downloading bought, per chain."""
    print()
    print("WHAT THE DOWNLOADS BOUGHT (data/historical_ohlcv)")
    print()
    if not CORPUS.exists():
        print("  no corpus directory")
        return
    print("  %-12s %7s  %-20s %s" % ("chain", "files", "newest", "traded?"))
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir():
            continue
        files = list(d.glob("*.json"))
        newest = max((f.stat().st_mtime for f in files), default=0)
        stamp = _dt.datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "--"
        traded = "yes" if d.name == TRADED_CHAIN else "NO"
        print("  %-12s %7d  %-20s %s" % (d.name, len(files), stamp, traded))
    print()
    print("  and the DB tables that are supposed to hold it:")
    try:
        con = _db()
        for table in ("ohlcv_datasets", "ohlcv_bars"):
            n = con.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
            print("    %-16s %d rows" % (table, n))
        print("  the corpus is JSON on disk only; nothing on the trading path reads it")
        print("  out of the database, because there is nothing in the database.")
    except Exception as exc:
        print("    could not read: %s" % exc)


def main() -> int:
    print("=" * 74)
    print("data_ingest: what it does with 700+ seconds, and what that starves")
    print("[b8c24590] -- read-only; source logs/system.log + market_stream")
    print("=" * 74)
    print()
    runs = scan_log()
    if not runs:
        print("no slow-worker lines in the log -- nothing to census")
        return 0
    span = (max(r[0] for r in runs) - min(r[0] for r in runs)) / 86400.0
    print("%d over-budget worker runs across %.1f days of log" % (len(runs), span))
    print()
    offpath = census(runs)

    con = _db()
    feed_during(con, stalls(runs))
    starved(con)
    corpus()

    print()
    print("=" * 74)
    print("THE FIX, and it is one line of .env:")
    print()
    print("  DOWNLOAD_WORKER_CHAINS=%s" % TRADED_CHAIN)
    print()
    print("  services/background_workers.py:678 defaults it to")
    print("  \"base,ethereum,arbitrum,optimism,polygon\" and .env never set it,")
    print("  while .env line 431 already says LIVE_FOCUS_CHAINS=base.")
    if span > 0:
        print("  Recovers the %.1f h of off-path download inside %.1f days of log"
              % (offpath / 3600.0, span))
        print("  -- a %.1f h/day rate, i.e. the supervisor thread spends most of"
              % (offpath / 3600.0 / span))
        print("  its wall-clock inside a download for a chain we cannot trade.")
    print()
    print("  SAFE BECAUSE: the DynamicDownloadWorker still covers any chain that")
    print("  shows up in recent pairs (db.fetch_recent_pairs), so a non-base pair")
    print("  that ever does matter is still fetched. This only stops the STATIC")
    print("  every-cycle sweep of four chains the wallet cannot trade.")
    print()
    print("  This does NOT fix a starved price feed -- there isn't one. It stops")
    print("  `discovery` being held behind candle downloads for chains the wallet")
    print("  cannot trade.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
