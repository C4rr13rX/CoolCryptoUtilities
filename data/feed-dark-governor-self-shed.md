# The feed is dark because the governor pauses it on memory it does not hold

Gale, pass 123 (2026-09-15). Measured, not theorised. Every number below has the
command that produced it.

## The headline

The price feed has been dark for **57.6 hours**, and the cause is not the
sequential queue. `ResourceGovernor` gates on **system-wide** memory and, at
>=90%, pauses work at *every* priority including HIGH/CRITICAL (the feed) and
kills our own children every 20 seconds. The machine is at 99% because of
memory production does not hold: production's own RSS is **143 MB**. The guard
therefore fires forever and cannot be relieved by anything it is able to do.

This is the "a gate that blocks everything is a bug, not safety" shape.

## The feed

    python -c "import sqlite3,time; c=sqlite3.connect('file:storage/trading_cache.db?mode=ro',uri=True); \
      print(c.execute('SELECT MAX(ts),COUNT(*) FROM market_stream').fetchone())"

    newest market_stream row : 2026-09-13 10:45:47   (57.6 h before 2026-09-15 20:22)
    rows                     : 89,207
    ticks in last 10 min     : 0

`data/news/free_news.parquet` and `ethical_news.parquet` last wrote
2026-09-13 11:00:19 — within 15 minutes of the last tick. Everything production
writes stopped the same morning. There is no market-data parquet: the only two
parquet files in the tree are the news ones, so `market_stream` is still the
feed's store and 0 ticks means 0 ticks, not a storage migration.

## The process

    powershell "Get-CimInstance Win32_Process | ... start_production"
    Get-Process -Id 588 | Select CPU, WS

    PID 588   started 2026-09-09 18:34:13   CPU 414,573 s (115 CPU-hours)   RSS 143 MB
    PID 19740 started 2026-09-09 18:34:13   CPU 0.03 s                      RSS  20 MB  (launcher)

It is **not hung** — it is burning ~80% of a core sustained. It is spinning in
the shed loop. It is also running **pre-fix code**: it started six days before
any commit made today, and production serves stale code until restarted.

## The machine

    GlobalMemoryStatusEx via ctypes

    MEMORY LOAD   : 99%
    TOTAL         : 31.78 GB
    AVAILABLE     : 0.24 GB
    PAGEFILE FREE : 1.75 GB

Sum of RSS over every process psutil could read: **7.32 GB**. That is a FLOOR,
not a mystery — processes whose `memory_info()` raised AccessDenied were
skipped, and Windows holds compressed (MemCompression, 885 MB), nonpaged-pool
and driver memory outside any working set. What is certain is the ratio that
matters: production holds 143 MB of the 31.5 GB in use, i.e. **0.45%**.

## The mechanism, line by line

`services/resource_governor.py`:

    203-204  vm = psutil.virtual_memory(); mem = vm.percent / 100.0   <- SYSTEM-WIDE
    254      other_mem = max(0.0, (total_mb - avail_mb) - our_rss)    <- already knows!
    283      if mem >= _CRITICAL_MEM (0.82): GC + pause LOW and NORMAL
    304      if mem >= _EMERGENCY_MEM (0.90) and 20s since last: _emergency_shed_load()
    381      kill any child using > 150 MB
    492-494  should_pause(HIGH or CRITICAL) -> return mem >= _EMERGENCY_MEM

Line 494 is the one that takes the feed down. The feed runs at HIGH/CRITICAL,
and at 99% system memory `should_pause` returns True for it, permanently.

Line 254 is the fix, already computed and never consulted: the governor
**already measures** how much memory other programs hold. With our RSS at 143 MB
and 0.24 GB available, its own `other_mem` is ~31.4 GB. It knows the pressure is
not ours and reacts as though it were.

The loop cannot converge. `_emergency_shed_load` frees, at absolute most, our
143 MB. The number it is reacting to is ~31.4 GB of somebody else's memory. So
it sheds, re-reads 99%, sheds again, 20 seconds later, for 57.6 hours — which is
exactly what `logs/console.log` shows for hundreds of consecutive lines:

    CRITICAL memory (99.1%) - GC collected 0 objects, pausing low-priority work
    EMERGENCY memory shed (99.1%, our RSS=2496MB) - clearing caches and killing workers
    ... (repeating, RSS bouncing 125 MB - 4453 MB as workers are killed and respawn)

## Why this supersedes item cee842c9

cee842c9 blamed `dataset_warmup` (model category) for holding the sequential
queue. Pass 120 already refuted that with an attribution: the model category
holds 93.36 s/10min but loop blocks are **not** enriched inside its join windows
(17.9% vs 15.6% baseline over 6 h = 1.15x; 14.8% vs 15.6% over 4 days, i.e.
below background), so it explains at most 8.7 and realistically 1-2 of the 39
missing ticks/10min.

This finding explains the rest, and it explains the 58 s event-loop block in the
original report far better than a 120 s task does: a box at 99% memory with
0.24 GB free is page-thrashing, and a thrashing box blocks an event loop for
58 s of a 10 s budget without any task holding a queue.

Criterion 3 of cee842c9 reads "**IF** the fix is to move or bound the model
tasks, the feed rate is measured BEFORE and AFTER". The antecedent is false —
the fix is not to move or bound the model tasks — so the criterion cannot be
satisfied as written and I am not redefining it. It is recorded here instead.

## The fix this points to (NOT yet applied)

The guard is correct in the case it was written for — our tree eating the box —
and must keep full force there. It is wrong only when the pressure is external.
The discriminator already exists on line 254. Proposed, for whoever takes it:

  * When our own tree is within its RSS budget (`our_rss < _max_rss_mb`), the
    pressure is by definition not ours. In that case do NOT pause HIGH/CRITICAL
    and do NOT run the child-kill: neither frees memory we hold, and both stop
    the feed.
  * Keep GC and keep pausing LOW/NORMAL — those are cheap, harmless, and do
    genuinely reduce our footprint.
  * Log the external case loudly and distinctly, so "the box is full" never
    again looks identical in the log to "we are full".

Two things must be true before the feed can be trusted again, and they are
independent: the governor must stop self-harming, **and** something must give
back the ~31 GB. Fixing only the governor leaves production running on a box
with 0.24 GB free.

## What I did not do

I did not restart production and I did not kill anything. Three other agents
share this box and this worktree, a restart costs a 13-minute boot, and with
0.24 GB free a restart could fail to come back at all. That is an operator call.
