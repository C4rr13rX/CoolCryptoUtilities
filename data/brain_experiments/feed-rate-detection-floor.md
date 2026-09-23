# A 10-minute feed-rate window cannot see a 2-tick change: the detection floor is 100.8 ticks

Gale, pass 121. Item [cee842c9], last open criterion.

## The question

[cee842c9]'s fourth-listed criterion reads: *"If the fix is to move or bound
the model tasks, the feed rate is measured BEFORE and AFTER over a full
10-minute window each."* The fix shipped in `bcabb78` does move them —
`dataset_warmup` and `candidate_training` are now `Task.detach` — so the
criterion applies. Pass 120 recorded it as unmet for a mechanical reason:
production serves stale code until restarted, boots in ~13 minutes, and a
before window plus a boot plus an after window does not fit a 30-minute pass.

That reason was true and it was also the wrong reason. The criterion is not
blocked by the boot time. It is blocked by arithmetic, and nobody with any
amount of time can satisfy it as written.

## Corpus and method

`storage/trading_cache.db`, table `market_stream`, opened read-only in SQLite
`immutable=1` mode so the running production process is never blocked. The
statistic is counted exactly the way `scripts/loop_status.py:125` counts it —
`SELECT COUNT(*) FROM market_stream WHERE ts>=?` over a 600-second window —
but over every **disjoint** 10-minute bin in the lookback instead of the one
window ending now.

Script: `scripts/feed_rate_power.py`. Nothing is fitted and nothing is
modelled; the numbers are the empirical distribution of the criterion's own
statistic.

Window: the 24 hours ending 2026-09-11 ~08:00, 144 disjoint bins. Production
(PID 19740) has been up since 2026-09-09 18:34, so this entire window lies
inside one uninterrupted production lifetime.

## The numbers

| statistic | ticks / 10 min |
|---|---|
| mean | 49.85 |
| standard deviation | 36.37 |
| coefficient of variation | 0.73 |
| min / p05 / median / p95 / max | 0 / 0.0 / 46.0 / 104.7 / 321 |
| bins with zero ticks | 13 of 144 (9.0%) |

Smallest difference in mean ticks/10 min that N windows **a side** can call at
95% confidence, by `1.96 · sd · sqrt(2/N)`:

| N windows a side | minimum detectable difference |
|---|---|
| 1 | **100.8** |
| 3 | 58.2 |
| 6 | 41.2 |
| 12 | 29.1 |
| 36 | 16.8 |

## What that means for the criterion

The effect under test is **1–2 ticks / 10 min**. That is not a guess: it is
this item's own second criterion, measured in `bcabb78` by
`scripts/seq_queue_budget.py`. The model category holds 93.36 s per 10 minutes
of the sequential queue (15.6%), but event-loop blocks are *not* enriched
inside those join windows — 17.9% inside against a 15.6% baseline over 6h
(1.15×), and 14.8% against 15.6% over four days, which is *below* background.
So the model category explains at most 8.7 of the 39 missing ticks under the
most generous assumption and roughly 1–2 on the correlation.

One 10-minute window a side resolves **100.8** ticks. The effect is **2**.
The criterion is underpowered by a factor of about 50, and the measurement it
asks for would return a number whose sign is decided by noise.

To resolve 2 ticks/10 min at 95% confidence takes **2,541 windows a side —
423.5 hours, 17.6 days, each side**. Even the most generous 8.7-tick reading
needs 135 windows a side, 22.4 hours a side, which is a two-day experiment and
still not the "full 10-minute window each" the criterion names.

This is why the criterion cannot be satisfied by restarting production and
watching for ten minutes, and why doing so would have produced a confident
number that meant nothing.

## One thing I checked and did not claim

Over 96 hours the picture looks far worse — mean 31.6, sd 42.91, and 223 of
576 bins (38.7%) with zero ticks. I am not reporting that as feed darkness.
`market_stream`'s oldest row is 385.5 hours old, so nothing is pruned and the
zeros are real gaps, but the longest zero run is **131 consecutive bins
(21.8 hours)** and it sits 72–96 hours back — before production started on
09-09 18:34. That is downtime, not a stall. Zero-bin share by quarter, oldest
first: 129/144, 72/144, 9/144, 13/144. Only the last two quarters lie inside
the current production lifetime, and they are the 24h window quoted above.

The 9.0% of zero bins *inside* production's uptime is real and is not the
model queue. It is the abandoned-`data_ingest` stall filed as [3550bc30] and
instrumented in `6ab23b3`.

## What the criterion should have asked for

The fix's guarantee is deterministic, not statistical: a detached task holds
**zero** seconds of the sequential queue, by construction, and that is what
`tests/test_sequential_scheduler.py::ModelTaskMustNotHoldTheQueueTest` asserts
in five tests that all fail against the pre-detach scheduler. The queue-held
seconds per category — the thing the fix actually changes — are measurable
before and after from the scheduler's own `task_timeout` events without a
restart and without a 17.6-day wait.

The feed-rate criterion was written on the premise that the model category was
the cause of the tick shortfall. This item's own criterion 2 falsified that
premise. A before/after feed-rate measurement is the right test of a cause
that turned out not to be the cause.

## Reproduce

    python -X utf8 scripts/feed_rate_power.py --hours 24 --json-out data/feed_rate_power_24h.json
    python -X utf8 scripts/feed_rate_power.py --hours 96 --json-out data/feed_rate_power_96h.json
