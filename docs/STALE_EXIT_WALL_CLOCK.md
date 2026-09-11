# `stale_exit_secs` now runs on a wall clock — [3f7a0677]

Iris, pass 119, 2026-09-11.

## The defect

`stale_exit_secs` is 900 seconds. It is a **wall-clock promise enforced on a
tick-driven schedule**: `_handle_sample` is the only caller of
`_interpret_predictions`, it passes ONE sample for ONE symbol, and it does so
only after two gates that have nothing to do with exits —

* the model-window gate (`len(self._buffer) < self.window_size`), which a bot
  added by `reconcile_pairs` for a HELD symbol — added precisely so the position
  can be closed — must fill before any exit rule runs at all;
* the duplicate-signature return, which drops a repeated `(symbol, ts)`, so a
  symbol whose publisher restamps one timestamp never evaluates an exit however
  long it keeps ticking.

Measured over the 14 days to 2026-09-11, `scripts/hold_time_edge.py`, on the 78
round trips the live lane could have placed (1 unplaceable, excluded):

| hold time | n | gross $ | loss share | win % | ticks/min |
|---|---|---|---|---|---|
| under 5 min | 9 | +0.2766 | 0.0% | 66.7 | 0.71 |
| 5–15 min | 5 | −0.0698 | 6.3% | 0.0 | 0.36 |
| 15–60 min | 29 | +0.1137 | 0.0% | 44.8 | 0.55 |
| **1–4 hours** | **30** | **−0.2107** | **19.0%** | 46.7 | 0.27 |
| over 4 hours | 5 | +0.0855 | 0.0% | 40.0 | 0.14 |

86% of trips outlive the 900s clock and ticks/min falls monotonically with hold
time. The trips that overstay are the ones the **feed stopped watching**, so the
positions that most need rule 4 are the ones least able to reach it.

## The fix

`TradingBot._close_stale_positions_on_the_clock`, called from the same place the
dark-feed sweeps are called from (above every early return in `_handle_sample`,
so it runs on ANY symbol's tick) and **before** them, because the two rules tile
one space: a position whose mark is still fresh enough to book honestly should be
CLOSED with its outcome recorded, and only one too dark to mark out should be
abandoned with nothing recorded. That ordering is what stops the sweep throwing
away the observation graduation is starved of.

It adds **no exit rule, lowers no bar and shortens no clock.** It runs the
EXISTING chain with a NEUTRAL model read against a price somebody actually
observed. Neutral is the point: with `model_neutral` true the two opinion rules
cannot fire, so the only things that can close a position are its own facts — its
target, its stop, and the clock it has already outlived.

Bounded, deliberately:

* **marks expire.** `GHOST_STALE_SWEEP_PRICE_MAX_AGE_SEC`, default 900s (the p90
  inter-tick gap is 406s). Past it the sweep does nothing and the dark-feed
  sweep owns the position — marking out against an hour-old price is the
  fabricated outcome `StrategyLedger._is_implausible` exists to reject, and
  AERO-USDC once booked +161% exactly that way.
* **live positions are untouched.** The live lane already has its own wall-clock
  treatment in `_exit_dark_live_positions`, which asks the CHAIN for a price
  rather than trusting the feed. A live exit must go through a real swap.
* **one bot closes one position.** `self.positions` is the merged book of every
  bot and every bot runs the sweep, so a module-level in-flight claim keyed
  `symbol:trade_id` stops two bots booking two outcomes for one round trip.
  Graduation would count that evidence twice.
* **the sweep and the chain read ONE clock** (`_stale_exit_secs`). Two env reads
  would let the sweep offer a position the chain then refuses, on every tick,
  forever — the "two copies of one rule that drifted apart" shape four separate
  defects in this repo have had.
* **no entry time is no clock.** Same choice rule 4 makes for an unknown cost
  basis: the max-hold eviction owns it rather than an invented hold time.

## Before and after, with n

`scripts/hold_time_edge.py --days 14` now replays the sweep against the ticks
each trip actually received (`wall_clock_replay`). **n = 74** timed trips:

| | before | after |
|---|---|---|
| median hold | 58.8 min | **15.5 min** |
| outlive clock+30s | **86%** | **4%** |
| n within that | 10 | 71 |

62 of 74 trips would have been closed by the sweep.

The threshold carries the sweep's 30s period on **both** columns. Against a bare
900s, every swept trip closes at 930s and still reads as "outlived the clock" —
the share printed 86% before and 86% after while the median fell from 58.8 to
15.5 minutes. 30 seconds of grid is not a position outliving its clock, and
hiding that would have flattered the OLD behaviour, not the new one.

**This is a counterfactual, not a post-fix measurement, and it is labelled as one
everywhere it prints.** A real post-fix number needs production to run the new
code over a fresh window. Nothing here substitutes for that.

## The ≤15 min bucket after the fix — n, and why it is not ranked

Today's `≤15 min` bucket is **n = 10**, which is under this loop's own n≥30 floor,
so **it is not ranked and no edge is claimed from it.** The replay says 71 of 74
trips would land inside the clock, which would clear n≥30 — but that bucket
cannot be priced from this book: a trip closed at 930s instead of 58 minutes
books a different gross, and re-marking it would be inventing the outcome rather
than measuring it. The per-trade net against buy-every-bar in an UP and a DOWN
window is therefore the **next** pass's measurement, off the post-fix book.

## The selection effect, stated because the motivating number carries it

The hold-time split is over **REALISED** trips. A trip may have closed fast
BECAUSE it hit its target — `hold_time_edge.py`'s own docstring says so. Closing
more trips on time does **not** prove the fast ones were good. What this fix does
is move the population so the question can be asked at n≥30; **the post-fix book
is the test of the +0.4588%/notional claim, not this document.**

## What proves each part

```
python -X utf8 -m pytest tests/test_a_stale_position_closes_with_no_further_tick_on_its_symbol.py -q
    8 passed   (the wall clock, the price bound, live immunity, the double-book
                guard, the one-clock rule, the tick/price pairing, no-entry-ts)
python -X utf8 -m pytest tests/test_a_stale_loser_is_measured_not_forecast.py \
    tests/test_a_bearish_model_cannot_outrank_the_stale_clock.py \
    tests/test_a_position_that_went_nowhere_still_exits.py \
    tests/test_an_opinion_cannot_spend_a_round_trip_the_move_never_earned.py \
    tests/test_the_hold_time_report_counts_positions_not_log_rows.py -q
    37 passed with the new file   (the exit chain is unchanged)
python -X utf8 scripts/hold_time_edge.py --days 14
    the before/after table above
```

Production serves stale code until restarted, so none of this is in the live
lane until the 14 python processes are recycled.
