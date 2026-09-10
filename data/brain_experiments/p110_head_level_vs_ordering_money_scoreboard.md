# Pass 110 — the trading head scored as a MONEY rule, level vs ordering

Iris, 2026-09-10. Item [93a905e5]. Every number below is from
`scripts/head_vs_realised_census.py`, which is committed with this report.

    python -X utf8 scripts/head_vs_realised_census.py --hours 26 --horizon-min 15
    python -X utf8 scripts/head_vs_realised_census.py --hours 12 --split-hours 6 --horizon-min 60

## CORPUS AND WINDOWS

- Source: `storage/trading_cache.db`. Predictions from `organism_snapshots`
  (`payload.prediction.direction_prob`), realised prices independently from
  `market_stream`. Base and forward price both come from the stream, never
  mixed with the snapshot's own `sample.price` — a cross-source ratio would
  manufacture a return that never happened (`feed-denomination-contamination`).
- `(dp==0.5, nm==0.0)` rows are dropped as bot.py's no-model sentinel.
- 26h lookback, split at −12h. 2104–2772 scored rows per era per horizon.
  Horizons 5 / 15 / 30 / 60 minutes.
- Cost floor: the measured receipt cost, 0.3187% of notional + 0.004047 fixed,
  amortised at a $5 clip → **0.3996% round trip, 0.1998% one leg**.

## 1. THE LEVEL — no horizon beats the majority baseline at any horizon

Scored against "always call the more common realised direction", never 0.5.

| horizon | era | scored | baseline | head | edge | z | significant |
|---|---|---|---|---|---|---|---|
| 5min | pre | 2351 | 0.4365 | 0.4390 | +0.0024 | +0.24 | no |
| 5min | post | 2768 | 0.5202 | 0.5181 | −0.0021 | −0.23 | no |
| 15min | pre | 2285 | 0.4402 | 0.4319 | −0.0083 | −0.80 | no |
| 15min | post | 2725 | 0.5400 | 0.5453 | +0.0053 | +0.56 | no |
| 30min | pre | 2248 | 0.4366 | 0.4141 | −0.0225 | **−2.16** | no |
| 30min | post | 2685 | 0.5490 | 0.5482 | −0.0007 | −0.08 | no |
| 60min | pre | 2104 | 0.4616 | 0.4316 | −0.0300 | **−2.78** | no |
| 60min | post | 2358 | 0.5556 | 0.5560 | +0.0004 | +0.04 | no |

**ANSWER TO THE ITEM'S CRITERION: NO horizon beats the majority baseline
significantly on LEVEL.** The largest edge anywhere is +0.0053 at z=+0.56,
which is half a standard error. At 30 and 60 minutes pre-collapse the head is
significantly *below* baseline. There is no healthy state to recalibrate back
to, and calibration cannot help in any case: it is a monotone remap, so it
moves the level and preserves the order. **The head needs different features,
not recalibration.**

The head has also stopped calling up at all. `dp_p50` post-collapse is 0.0388
and `dp_max` at 60min is 0.5346, so "head says UP" fires on **1 of 2358 rows**
at 60min and 65 of 2726 at 15min.

## 2. THE ORDERING — real but its SIGN IS NOT STABLE across adjacent windows

| horizon | pre AUC | post AUC |
|---|---|---|
| 5min | 0.4145 ± 0.0117 **INVERTED** | 0.4978 ± 0.0110 chance |
| 15min | 0.4191 ± 0.0119 **INVERTED** | 0.5427 ± 0.0111 above chance |
| 30min | 0.4263 ± 0.0120 **INVERTED** | 0.5266 ± 0.0112 above chance |
| 60min | 0.4781 ± 0.0126 chance | 0.5467 ± 0.0119 above chance |

Split the post-collapse window itself into two 6h halves at 60min:
AUC **0.4383 ± 0.0169** then **0.5217 ± 0.0182**. Both bounds exclude the
other's point estimate.

**So an AUC number is not evidence about a rule.** The ordering carries
information whose direction flips between adjacent windows, which is worse for
a rule-builder than no information: a rank threshold fitted on either half is
backwards in the other.

## 3. THE MONEY SCOREBOARD — operator direction of 15:07, and the sell half

Precision counts a call as right **only if the move cleared the cost floor**;
a directional hit smaller than the round trip is a loss. Baselines are the
do-it-every-bar money rules in the same rows.

| horizon | era | rule | n | precision | net/trade | vs base |
|---|---|---|---|---|---|---|
| 15min | pre | head UP (buy) | 931 | 10.85% | −0.4017% | +1.36pp |
| 15min | pre | **top decile (buy)** | 268 | 17.54% | −0.4362% | **+8.05pp** |
| 15min | pre | buy EVERY bar | 2676 | 9.49% | −0.3795% | — |
| 15min | pre | head DOWN (exit) | 1745 | 13.35% | −0.2319% | −2.83pp |
| 15min | pre | exit EVERY bar | 2676 | 16.18% | −0.2200% | — |
| 30min | pre | **top decile (buy)** | 260 | 26.54% | −0.5853% | **+11.02pp** |
| 30min | pre | buy EVERY bar | 2604 | 15.51% | −0.3976% | — |
| 60min | pre | **top decile (buy)** | 244 | 31.15% | −0.5446% | **+9.30pp** |
| 60min | pre | buy EVERY bar | 2435 | 21.85% | −0.4147% | — |
| 60min | post | head DOWN (exit) | 2357 | 41.62% | −0.2033% | +0.02pp |
| 60min | post | exit EVERY bar | 2358 | 41.60% | −0.2033% | — |

**THE ORDERING IS VOLATILITY SELECTION, NOT DIRECTION.** The top-decile rule
raises the share of bars clearing the floor *upward* by 8 to 11 points — the
largest effect in the whole census — and its net per trade is *worse* than
buying blind every time. At 15min pre-collapse its gross is −0.0358% against
the baseline's +0.0191%. It selects bigger moves in both directions, so it
lifts the payers and the tail together. A scoreboard reporting precision
without net/trade would have published that as an edge.

**THE SELL-HIGH HALF FOR THIS HEAD.** `omen_experiment.py:552` is long-only so
a crest scores zero there. Cove closed that gap for the OMEN brain's crest
label in the same pass (commit eac666b, crest +15.5pp in a DOWN window); this
is the separate and independent question for the TRADING head's
`direction_prob`, scored against forward returns as an exit on a held
position and costed at one leg. Head-says-DOWN beats
exit-every-bar by **+0.02 to +0.58pp** post-collapse and is **−1.17 to
−5.85pp** pre-collapse. The exit call is worth nothing above exiting blind.
So for this head the sell side has no more skill than the buy side -- which is
the opposite of what Cove found for the omen brain's crest, and the two are
different instruments on different labels, not a contradiction.

## 4. THE ONE POSITIVE CELL, AND WHY IT IS NOT AN EDGE

Pooled over the whole post-collapse window, top-decile buys at 60min read
**+0.3242%/trade at 44.92% precision against 23.88% for buying every bar
(+21.04pp), n=236** — the only positive net/trade in 40 rule-cells, and in a
down-leaning tape (44.4% up), so not the long-only-in-an-up-window trap.

Split that same window into its two 6h halves and it dissolves:

| half | tape | top decile | buy EVERY bar |
|---|---|---|---|
| −12h..−6h | 30.8% up (DOWN) | **0.00%** prec, **−1.8354%**/trade, n=131 | 13.61%, −0.9109% |
| −6h..0h | 61.3% up (UP) | 40.00% prec, −0.0746%/trade, n=105 | 36.66%, **+0.2432%** |

The rule is negative in **both** halves. In the down half it hit **0 of 131**
and lost twice what blind buying lost. In the up half blind buying beat it.
The pooled +0.3242% was an artifact of averaging a hard down 6h with a strong
up 6h.

And note what the only genuinely positive number in this report is:
**buy-every-bar, +0.2432%/trade, in the up window.** That is beta, not edge,
and it is exactly the flattery the standing instructions warn about.

## 5. THE CAP THAT BOUNDS ANY REPAIRED HEAD

Only **30.5%** of 15-minute ticks move further than the 0.3187% proportional
cost alone. Buy-every-bar precision rises with horizon — 6.37% at 5min, 9.49%
at 15, 15.51% at 30, 21.85% at 60 — which independently reproduces the horizon
finding in **[8b1846d8]** off a different instrument. Clip is not the lever
(**[4d3310e7]**): at a $5 clip the fixed leg is 0.0809% of a 0.3996% floor, so
25× the capital removes about a fifth of the floor and none of the 0.3187%.

## VERDICT

The head's **LEVEL** carries no information at any horizon tested and is
significantly negative at two of them; it needs different features. The head's
**ORDERING** has weak measurable skill whose sign flips between adjacent
windows, and what it selects is volatility rather than direction — so a rank
threshold raises precision and still loses more money per trade. **No rule
built on this head has a positive net per trade in any window that
replicates.** Reported at or below baseline, honestly, which is the normal
outcome here.
