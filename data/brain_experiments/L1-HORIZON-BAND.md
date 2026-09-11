# L1 across the 30-240 minute band, and why 4 windows in 5 call nothing

Pass 118, Cove. Item [c6196eb2]. No node was contacted; production's :8090 was
not touched. Harness `scripts/omen_l1_horizon_sweep.py`, tests
`tests/test_an_l1_negative_names_the_horizon_it_measured.py`.

## 0. What was already known, and is not re-litigated here

Pass 114 (45a2959) scored the L1 motif -> trough rule over 102 base corpora at
a 3600s median cadence, train 350 / purge 12 / test 60, bands fitted on TRAIN
and frozen: **DOWN +0.10pp on 194 trades, UP -1.97pp on 6**. That is a clean,
well-powered negative and nothing below disputes it. Two things it did not
settle, which are what this pass measured:

1. **Scope.** On a 3600s cadence `--horizon 12` is *twelve hours*. The negative
   ruled out a twelve-hour rule and ruled out nothing between thirty minutes
   and four hours -- the band where [15cc71d4] measured the cost floor and the
   hit rate crossing (the share of ticks whose realised move outruns the
   0.3592% live cost floor is 17.8% at 5 minutes, 53.2% at 60, 80.3% at 240).
2. **Abstention.** 83 of those 102 windows called zero trades, so the headline
   was a 19-window estimate wearing a 102-window label.

## 1. Corpora, and the horizon a cadence cannot express

| set | chain | median bar spacing | corpora scored | unreadable |
|---|---|---|---|---|
| A | base | 3600s | 102 | 3 |
| B | base | 300s | 27 | 62 |

Both sets use train 350 / purge = horizon / test 60, bands fitted on TRAIN and
frozen, hysteresis 0.50, `min_lift` 1.3, `min_support` 20, round-trip cost
0.0065. One process per set.

**Thirty minutes cannot be asked of set A.** A 3600s cadence is one bar per
hour, so 30 minutes is half a bar. It rounds to one bar and the run happily
produces numbers -- numbers that are byte-identical to the 60-minute arm. A
report that then wrote `horizon_minutes: 30` would have labelled a sixty-minute
measurement as a thirty-minute one, which is a stale number no later reader
could detect. Every arm here records the **resolved** minutes beside the asked
ones and carries `sub_bar: true` when the two disagree by more than a tenth;
`test_a_sub_bar_horizon_is_not_reported_as_the_minutes_that_were_asked` pins
it. The genuine sub-hour question is therefore asked of **set B**, where 30
minutes is 6 bars, and set B is reported as its own set and never pooled with
A.

## 2. Held-out net per trade, by horizon

Per-trade net is the realised forward return over the horizon minus the
round-trip cost, pooled over every called trade. The baseline is buy-every-bar
**in the windows the rule actually fired in** -- pooling it over declined
windows would score the rule against a market it never traded, and in a sweep
that declines four windows in five that is most of the sample
(`test_the_baseline_is_taken_only_in_windows_the_rule_actually_traded`). The
standard error is over TRADES, not over window means, and the difference's
error adds the two in quadrature, which is conservative because the called
trades are a subset of the baseline bars.

### Set A -- 102 corpora at 3600s

| horizon | bars | window | corpora | fired | per-trade | baseline | edge | z |
|---|---|---|---|---|---|---|---|---|
| 30m* | 1 | UP | 40 | 7 | -0.4733% | -0.6053% | +0.13pp | 0.80 |
| 30m* | 1 | DOWN | 62 | 11 | -0.7198% | -0.7337% | +0.01pp | 0.27 |
| 60m | 1 | UP | 40 | 7 | -0.4733% | -0.6053% | +0.13pp | 0.80 |
| 60m | 1 | DOWN | 62 | 11 | -0.7198% | -0.7337% | +0.01pp | 0.27 |
| 120m | 2 | UP | 37 | 6 | -0.3541% | -0.5719% | +0.22pp | 1.68 |
| 120m | 2 | DOWN | 65 | 15 | -0.8540% | -0.8344% | -0.02pp | -0.27 |
| 240m | 4 | UP | 36 | 9 | +0.1259% | -0.5004% | +0.63pp | 2.29 |
| 240m | 4 | DOWN | 66 | 13 | -1.1507% | -1.0662% | -0.08pp | -0.68 |
| 720m | 12 | UP | 32 | 6 | +0.1591% | -0.0535% | +0.21pp | 0.40 |
| 720m | 12 | DOWN | 70 | 21 | -1.7470% | -1.7918% | +0.04pp | 0.36 |

`* 30m is sub-bar here and its row is byte-identical to 60m, which is what
being sub-bar means.` Abstention on set A: 84 of 102 at 60m, 81 at 120m, 80 at
240m, 75 at 720m -- between 74% and 82%, at every horizon in the band.

The 720m row is not a reproduction of pass 114's +0.10pp/194 trades and must
not be quoted as one. Two things differ: `data/historical_ohlcv` is
live-appended, so `bars[-window:]` is a later slice than pass 114 scored, and
the baseline here is pooled over test BARS in firing windows where pass 114
weighted each window's baseline by its own trade count. Both land inside one
standard error of zero, which is the only claim either run supports.

### Set B -- 27 corpora at 300s

| horizon | bars | window | corpora | fired | per-trade | baseline | edge | z |
|---|---|---|---|---|---|---|---|---|
| 30m | 6 | UP | 16 | 0 | -- | -- | no trades | -- |
| 30m | 6 | DOWN | 11 | 0 | -- | -- | no trades | -- |
| 60m | 12 | UP | 16 | 0 | -- | -- | no trades | -- |
| 60m | 12 | DOWN | 11 | 0 | -- | -- | no trades | -- |
| 120m | 24 | UP | 15 | 0 | -- | -- | no trades | -- |
| 120m | 24 | DOWN | 12 | 1 | -1.1985% | -1.3118% | +0.11pp | 0.79 |
| 240m | 48 | UP | 16 | 0 | -- | -- | no trades | -- |
| 240m | 48 | DOWN | 11 | 0 | -- | -- | no trades | -- |

Abstention on set B: 27 of 27 windows at 30m, 60m and 240m; 26 of 27 at 120m.
**The entire 300s set produced two trades in four horizons.** A window that
called nothing has no per-trade net and is reported as `None`, never as 0.0 --
0.0 against a negative baseline reads as a positive edge, and an all-abstaining
sweep would otherwise report the rule beating the market by exactly the
market's own loss (`test_a_window_that_called_nothing_has_no_per_trade_net`).

### Verdict, horizon by horizon: at or below baseline everywhere

**No horizon in the band passes the both-windows rule, and that includes the
one that looks best.** The only arm that clears two standard errors is 240m UP
at +0.63pp, z=2.29, per-trade +0.1259% -- an actual profit after the 0.65%
round-trip cost. It must not be read as an edge, for the reason this loop
already has a standing rule about: **it is one window class**. The same rule at
the same horizon in a DOWN window is -0.08pp, z=-0.68, and a long-only rule
scored only where the market rose is measuring the market. Nine firing windows
produced that number. 120m UP (+0.22pp, z=1.68) is the same shape, smaller.

Every other arm is inside one standard error of its own baseline: 60m +0.13pp
UP / +0.01pp DOWN, 120m DOWN -0.02pp, 240m DOWN -0.08pp, 720m +0.21pp UP /
+0.04pp DOWN. **The 30-240 minute band is an honest negative.** Pass 114's
twelve-hour negative was not a horizon artefact, and the cost-floor/hit-rate
crossing that [15cc71d4] located between 60 and 240 minutes does not rescue
this rule, because the rule's problem is not that it trades too short.

## 3. Why four windows in five call nothing: it is a SUPPORT famine

This is the part worth more than the edge estimate, and it is not what the
item's byproduct hypothesis expected. The abstention is not the rule being
selective about *markets*. It is the L1 encoder cutting the train window into
more motifs than 350 bars can support.

Measured on the train slice only, over the same two sets:

| set | horizon | median L1 vocabulary on TRAIN | median motifs clearing `min_support` 20 | median share of train bars covered by a supported motif | corpora with any motif at lift >= 1.3 |
|---|---|---|---|---|---|
| B (300s) | 30m / 6 bars | 61 motifs | **0** | **0.00** | 2 of 27 |
| A (3600s) | 60m / 1 bar | 49 motifs | 2 | 0.18 | 40 of 102 |

Roughly 350 labelled train bars spread over 49-61 distinct motifs is about six
bars per motif. The *median* corpus on the 300s set has **no motif at all**
that reaches twenty train samples, so there is nothing to fit before there is
anything to find; on the 3600s set the median corpus can fit two motifs
covering 18% of its train window. A rule fitted on 18% of its evidence
declining 81% of its windows is not a narrow predictor, it is an under-powered
one, and the fix is upstream of the horizon: either a coarser L1 encoder, a
longer train window, or a lower support floor -- each of which is a different
experiment and none of which this pass ran.

Command that produced the table: the diagnostic block quoted in
`data/attempts-revenir.md` for pass 118, which rebuilds the same train frames
with the same bands and margin and reports `label_skew` coverage.

## 4. Do the firing windows differ before they are scored?

Five properties, all computed from the TRAIN slice and nothing after it
(`test_window_properties_never_look_past_the_train_stop` poisons every bar
after the train stop and asserts no property moves): realised volatility,
up-rate, bar count, mean absolute return, train drift. AUC 0.500 means the
property says nothing about whether the window will fire.

### Set A, every horizon (AUC of the property over fired vs abstained windows)

| horizon | fired | abstained | realised vol | up-rate | bar count | mean abs return | train drift |
|---|---|---|---|---|---|---|---|
| 60m | 18 | 84 | 0.505 | 0.561 | 0.500 | 0.526 | 0.441 |
| 120m | 21 | 81 | 0.525 | 0.602 | 0.500 | 0.539 | 0.535 |
| 240m | 22 | 80 | 0.601 | 0.669 | 0.500 | 0.605 | 0.602 |
| 720m | 27 | 75 | 0.585 | 0.575 | 0.500 | 0.594 | 0.498 |

Bar count is exactly 0.500 at every horizon: the sweep slices a fixed window
off the end of each corpus, so a longer corpus gives the rule nothing extra and
the length cannot predict firing. The other four sit between 0.44 and 0.67,
which is the range you get from 102 samples and five properties looked at four
times.

### The held-out check

| horizon | cut fitted on the DISCOVERY half (51 corpora) | Youden J | selected on the HELD-OUT half | fire rate inside | outside | overall | lift |
|---|---|---|---|---|---|---|---|
| 60m | up-rate >= 0.4875 | 0.339 | 31 of 51 | 19% | 10% | 16% | **1.23x** |
| 120m | train drift >= 0.00912 | 0.397 | 34 of 51 | 12% | 24% | 16% | **0.75x** |
| 240m | train drift >= 0.00689 | 0.415 | 33 of 51 | 12% | 22% | 16% | **0.77x** |
| 720m | up-rate >= 0.5292 | 0.357 | 10 of 51 | 20% | 22% | 22% | **0.93x** |

**No property separates them.** Every cut looked strong on the half that chose
it -- Youden J between 0.34 and 0.42 -- and three of the four INVERTED on the
half they had never seen: the 120m and 240m cuts selected windows that fired
*less* often than the ones they rejected. The one arm above 1.0, 60m at 1.23x,
is 6 firing windows out of 31 selected against 8 of 51 overall; one window
either way moves it across 1.0. A selector that only works on the sample that
found it is not a selector, and this is what that looks like when you check it
honestly.

The selector was fitted on a DISCOVERY half of the corpora (every other one in
sorted order) by Youden's J in both directions, and read only on the HELD-OUT
half, because a threshold read on the sample that chose it always separates
that sample.

**The selector question is closed.** Section 3 says why it was always going to
be: a window fires when its train slice happens to contain a motif repeated
twenty times, and that is a property of the encoder's granularity meeting a
particular corpus, not of the market regime the window is in.

## 5. What this closes and what it opens

Closed: the L1 motif rule has no held-out edge anywhere in the 30-240 minute
band, on either cadence, and the twelve-hour negative from pass 114 was not a
horizon artefact. Closed: the abstention is a support famine in the encoder,
not a regime property of the market, so there is no free selector to be had
from volatility, up-rate or bar count.

Open, and each is a different experiment: a coarser L1 encoder with fewer
motifs per train window; a train window long enough to support the vocabulary
the current encoder produces; and the same sweep at `min_support` low enough
that the median corpus can fit anything at all -- which would raise the trade
count but must be measured with the same frozen-band discipline, because a
support floor of 5 will fit noise.
