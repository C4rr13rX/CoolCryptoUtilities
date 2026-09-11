# The entry size conjunct tested the right shape in the wrong units

**[4d3310e7], pass 113, Iris. 2026-09-10.**

## What this file is

`709c505` added a move-SIZE conjunct to the model-long entry test and was
reopened because **criterion 3 was not met**: the admitted count over the same
cycle window was **9 before and 9 after**. This file says why, with the
measurement, and what changed.

It does not redefine criterion 3, and it does not lower any threshold.

## The finding: `delta` is 751x the tape, so a cost floor cannot refuse it

`delta` is `price_mu`, the model's forward expected return. The shipped
conjunct compares it to `c(N) = 0.003187 + 0.004047/N`, the measured round-trip
rate. Both are nominally dimensionless fractions, and the code comment said so.
That was asserted, never measured.

Measured over decision cycles in `organism_snapshots` that carry both a
prediction and a forward `market_stream` tick 15 minutes later
(n = 9,667, horizon 900s, tolerance 600s):

| quantity | median | ratio to the realised move |
|---|---:|---:|
| realised absolute 15-minute move | 0.1181% | 1.00 |
| `delta` (= `price_mu`) | 90.4323% | **751x** |
| `volatility_rel` | 0.1137% | **0.96** |

`c(N)` on this corpus is 0.3862% at the $6.00 live clip and 0.8723% at the
median recorded `entry_fee_rate`. **A floor in that range cannot refuse a
quantity whose median magnitude is 90%.** The delta conjunct is algebraically
stricter — `c(N) > 0`, so it implies the `delta >= 0.0` it replaced — and
empirically inert. That is exactly what 9-before/9-after was reporting, and it
is a units failure at a boundary, not a threshold that needs raising.

## The fix: a different QUANTITY, not a different threshold

`volatility_rel` is computed at `trading/bot.py:3544` as the standard deviation
of the per-tick **fractional** change over the symbol's last ≤20 ticks. It is:

* **dimensionless**, like `c(N)`, so the comparison is unit-clean;
* built only from data available **at entry**, and it estimates a move that has
  not happened — a forward-looking estimate of SIZE, not the realised move;
* **calibrated on this feed at 0.96**, measured, not assumed.

The calibration is worth stating precisely because a derivation would have got
it wrong. This feed's median inter-tick gap is 39.5s, so 15 minutes is ~22.8
ticks and a square-root-of-time argument predicts ~4.8x per-tick sigma. The
measured answer is 1.0x. The ticks are not independent. The number used is the
measured one.

The threshold is unchanged: the same `min_expected_move = entry_fees *
ENTRY_MIN_MOVE_COST_MULT`, `entry_fees` being `roundtrip_cost_rate(notional)`
for the notional this entry is about to spend. **No new constant is introduced
and no existing one is lowered.**

A cycle with no `volatility_rel` is **refused**. A symbol whose expected move
cannot be estimated has not been shown to clear its cost, and defaulting an
unmeasurable size to "big enough" is the loosening this item exists to prevent.

## Criterion 3, before and after

Command:

```
python scripts/entry_move_size_census.py --limit 40000 --mult 1.0 --horizon 900
```

Window: the newest 12,000 `organism_snapshots` decision cycles, 100.5 hours,
99 symbols, replayed against `market_stream` at a 15-minute forward horizon
with a 10-minute tolerance. Same window for every row.

```
ADMITTED BEFORE (delta >= 0.0)                :  9
ADMITTED AFTER  (delta >= 1 x round trip)     :  9   ( 0.0% fewer)   <- the shipped conjunct, inert
ADMITTED AFTER  (+ vol_rel >= 1 x round trip) :  1   (88.9% fewer)   <- criterion 3 met
```

**The count falls, 9 to 1.** The marginal denominator is 9 cycles, so the
smallest fall this window can resolve is 1 cycle = 11.1 percentage points; an
8-cycle fall is well clear of that resolution.

Each conjunct on its own, over all 12,000 cycles — this is why the denominator
is 9 and not larger, and it corroborates Gale's finding independently on this
window:

```
direction_prob >= enter_threshold            562 of 12000   ( 4.683%)
exit_conf      >= enter_threshold             36 of 12000   ( 0.300%)   <- the binding term
net_margin     >= min_margin_gate           3432 of 12000   (28.600%)
net_margin_after_fees >= MIN_NET_MARGIN     3442 of 12000   (28.683%)
expected_profit_units >= SMALL_PROFIT_FLOOR 3382 of 12000   (28.183%)
```

And the same three tests in isolation, which is where the strictness is legible
without exit_conf in the way:

```
delta   >= 0.0        : 3741 of 12000  (31.2%)
delta   >= 1x cost    : 3460 of 12000  (28.8%)   <- refuses 7.5% of what the sign test admits
vol_rel >= 1x cost    : 1587 of 12000  (13.2%)   <- refuses 57.6% of what the sign test admits
```

**It is not a gate that refuses everything.** On its own the volatility floor
still admits 13.2% of all cycles — one cycle in eight. What it removes is the
86.8% where this feed's own measured volatility says the price is not going to
move far enough to pay the round trip.

### Criterion 4, scored honestly, and it licenses no verdict on return

```
before          : n=    8  mean NET -0.4981%  median NET -0.3923%  win 0.0%
after x1        : n=    8  mean NET -0.4981%  median NET -0.3923%  win 0.0%
after x1 +vol   : n=    1  mean NET -0.4111%  median NET -0.4111%  win 0.0%
every cycle     : n=10196                     median NET -0.8583%  win 5.7%
```

**n=1. That is one number wearing a percentage sign, and no edge is claimed
from it.** A per-trade mean on a single trade has no standard error. The
admitted subset is small because `exit_conf` admits 0.300% of cycles, not
because of this conjunct, and the fix for that is not in this item.

Two things the table does say honestly. Every arm is negative, including
buy-every-tick at a median -0.8583%. And the buy-every-tick **mean** gross of
+1588.9542% is contamination, not a return — a handful of implausible forward
ticks dominate it, which is why the median is quoted throughout.

The `|move|` line from the same run, which is the number this whole item exists
for: median absolute 15-minute move **0.1136%** on n=10,196, and only **23.9%**
of ticks clear the 0.3861% round trip. So on 76.1% of ticks a perfectly correct
direction call still loses money.



## What this does not claim

This is a cost-floor arithmetic fix. It does not claim an edge, and the
admitted subset is too small to carry one. The scored per-trade numbers below
are reported with their `n` and are not an edge claim at any `n` this window
provides.

It also does not fix `price_mu`. `delta` being 751x the tape is a model-head
defect, filed separately as `[1b0fd55f]`; this conjunct now survives it rather
than depending on it. The directive entry path still bypasses this conjunction
entirely (`[7231f8ac]`).
