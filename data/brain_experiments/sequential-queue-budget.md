# The sequential queue budget: which category makes the feed wait

Gale, pass 120, 2026-09-11. Item [cee842c9].

## The question

`SequentialScheduler` runs production's tasks one at a time and `_execute`
joins the worker **on the scheduler's own thread**, so every second a task
spends before it returns or times out is a second the tasks behind it -- the
price feed among them -- do not get. The item's hypothesis was that
`dataset_warmup`, a 120-second model task sitting in the same queue as
`data_ingest`, is what dropped the tick rate to 17/10min.

The item said to measure before changing anything. I measured, and **the
hypothesis does not survive.**

## Corpus

`logs/system.log`, the `sequential task_*` events, 2026-08-27 09:12:37 to
2026-09-11 07:14:32 (14,370 events). Headline window is the last 6 hours:
2026-09-11 01:14:38 -> 07:14:32, 35.99 ten-minute windows. Loop-block events
(`market-stream ... event loop was blocked Ns`) span 2026-09-07 00:26:47 ->
2026-09-11 07:14:05, 120,733 events.

Command: `python -X utf8 scripts/seq_queue_budget.py --hours 6`

Every number below is a **floor**. A task that finishes inside its timeout logs
nothing, so only timeouts are visible from history. That is why the fix also
adds live per-category accounting to `SequentialScheduler.status()` -- so the
next pass can read the complete number instead of a floor.

## Result 1 -- seconds per 10 minutes each category holds the queue

| category     | s/10min | share of queue | source |
|--------------|--------:|---------------:|--------|
| model        |   93.36 |         15.6%  | dataset_warmup, 28 timeouts |
| feed         |   17.78 |          3.0%  | data_ingest 15.00, news_enrichment 2.78 |
| trade        |    3.75 |          0.6%  | scheduler_refresh 2.50, ghost_metrics 1.25 |
| housekeeping |    0.83 |          0.1%  | telemetry_flush |

So the model category really does hold 15.6% of the queue, and all of it is
`dataset_warmup`. Over 6 hours it had 36 opportunities to run (600s interval)
and **timed out on 28 of them** -- it essentially never completes.

## Result 2 -- the attribution, and why the hypothesis fails

If the model task's join is what starves the feed, event-loop blocks should
cluster inside its 120-second join windows. They do not.

| window                          | loop blocks inside | baseline coverage | enrichment |
|---------------------------------|-------------------:|------------------:|-----------:|
| last 6h                         |   1,115 of 6,228 = 17.9% | 15.6% | **1.15x** |
| full 4-day span                 |  17,854 of 120,733 = 14.8% | 15.6% | **0.95x** |

Over four days the blocks are *below* the background rate inside those windows.
The event loop is blocked essentially all the time -- 120,733 block events at a
**mean of 61.7 seconds each** -- not specifically during model work.

**Attribution of the tick shortfall.** Observed 17 ticks/10min against 56
earlier the same day: 39 missing. Under the most generous assumption -- every
second the queue is held costs ticks at the full rate -- the model category's
15.6% share explains at most **8.7 of the 39 missing ticks (22%)**. The
correlation above says the true figure is far lower: dataset_warmup windows are
enriched for loop blocks by 1.15x in the recent window and 0.95x over four days,
so the excess attributable to the model category is roughly **2 percentage
points, on the order of 1-2 of the 39 missing ticks.**

## Result 3 -- what is actually blocking

`data_ingest`, the feed task itself.

- **346 overrun events in 6 hours.** An overrun means the previous run is still
  alive and the scheduler declined to start a second copy. Zero queue cost --
  and the feed task is not running at all for that whole time.
- **Longest abandoned run: 6,879 seconds (114 minutes).**
- **Longest single event-loop block measured: 7,393.1 seconds.** The two match.

The scheduler is right to refuse to stack copies. What is missing is that
nothing notices the feed task has been dead for two hours: `status()` did not
expose overruns or a stuck timer at all.

## The change shipped

`Task.detach` -- start the worker and return instead of joining it. Safe
because **the timeout never killed anything**: the worker is a daemon thread,
abandoned rather than killed, so `timeout_sec` only ever decided how long the
scheduler waited before admitting it was not going to finish. `is_running()`
already refuses to start a second copy. Detaching changes *when* the task is
abandoned, not *whether* a second copy can start.

`dataset_warmup` and `candidate_training` are registered with `detach=True`.

Before/after with the real task shape and the measured durations, pressure
pinned so the noisy CPU sampler cannot confound it:

|          | cycle wall | at production scale | model held | feed task ran |
|----------|-----------:|--------------------:|-----------:|---------------|
| BEFORE (joined)   | 8.05s | 322s | 7.50s of 8.05s (93%) | yes |
| AFTER (detached)  | 0.19s |   7s | 0.000s               | yes |

**42x less queue time per cycle, and the feed task still runs.**

## What this is NOT

This is not the fix for the tick rate, and saying it were would be the fake
edge this repo keeps paying for. By my own correlation it is worth **1-2 ticks
per 10 minutes out of 39 missing**. It removes 15.6% of the queue that was
being spent for no benefit whatsoever, and it makes the comment above those
task registrations ("never at the cost of trading") true for the first time.

## Criterion not met, and why

The item asks for the live feed rate over a full 10-minute window before and
after. That cannot be done in this pass and I am not redefining it: production
serves stale code until restarted, its boot takes ~13 minutes, and two
10-minute measurement windows plus a boot is over 30 minutes on a 30-minute
budget. The before/after above is the queue-held measurement in a harness, at
the real task shape. **The live 10-minute before/after is owed and is filed as
part of the follow-up item.**

## Next

Filed: `data_ingest` is abandoned for up to 6,879 seconds, which is the number
that actually matters here and is ~4x larger than everything in the table
above. That is the next item, not another tune of a model timeout.
