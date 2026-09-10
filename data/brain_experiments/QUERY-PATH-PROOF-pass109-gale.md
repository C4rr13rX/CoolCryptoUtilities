# The query path fires, and the pass-108 relation null was a broken comparison

Gale, pass 109, 2026-09-10. Operator's ordered step 2: *prove a query-set
change moves at least one held-out prediction.* It does. It also proves, by
running rather than by reading, that the pass-108 relation result measured
nothing.

## Setup

- Corpora: `data/brain_experiments/p108_aero_down.json` and
  `p108_aero_up.json`, 900 bars each, 3600s cadence, AERO-USDC.
- Windows, identical in both: train bars `[345, 695)`, test bars
  `[707, 887)`, horizon 12 bars (720 min), 12-bar purge between them.
- Held-out size: **180 predictions per cell**, 180 admitted of 180.
- Node: fresh `127.0.0.1:8091`, identity
  `brains/market_predictor_v3_assoc.identity.toml`, a **fresh brain dir per
  window** (`brain-data-p109-gale`, `brain-data-p109-gale-up`). Production on
  `:8090` untouched.
- Round trip cost 0.6500%, omen threshold 0.9750%.
- **Cadence is identical in both corpora — 3600s — so the UP/DOWN comparison
  here is apples-to-apples.** `--horizon 12` resolves to 720 minutes in both,
  and the runs print it. Iris's warning that `--horizon` is in BARS over a
  mixed-cadence corpus is right in general and does not bite this comparison;
  check the printed `= N min` before comparing any two runs.
- **Held-out label balance, per Iris's request that every report carry it.**
  The majority class differs between windows and that is the whole point of
  running both:

  | window | held-out label mix (180 bars) | majority |
  |---|---|---|
  | DOWN | `slide` 99, `murk` 44, `trough` 25, `crest` 10, `climb` 2 | `slide` 55.0% |
  | UP | `climb` 106, `slide` 22, `trough` 20, `murk` 19, `crest` 13 | `climb` 58.9% |

  Training was `--balance`d to 311 (DOWN) and 303 (UP) samples; the held-out
  windows are left at their natural imbalance, which is what makes the
  majority-class baseline meaningful.
- Every pair of cells within a window is **back-to-back on one fabric**: the
  second cell is `--skip-train`, so it re-measures the *same* trained fabric
  with a different query set. Nothing but the query set differs.

## The result: 2 windows x 3 arms

| window | arm — query set that FIRED | held-out exact | majority class | buy omens | per-trade | every-bar buy |
|---|---|---|---|---|---|---|
| DOWN | flat, default `temporal, geometry, cross` | 30.0% | 55.0% | 41 | -3.1565% | -2.8336% |
| DOWN | flat, measured `+ flow` | 30.0% | 55.0% | 28 | -3.2422% | -2.8336% |
| DOWN | **relations, measured `+ flow + rel_move_vol + rel_shape_flow + rel_trend_noise`** | **31.7%** | 55.0% | 42 | **-2.5468%** | -2.8336% |
| UP | flat, default `temporal, geometry, cross` | 20.0% | 58.9% | 6 | -1.6991% | +3.6841% |
| UP | flat, measured `+ flow` | 26.7% | 58.9% | 4 | +1.6656% | +3.6841% |
| UP | **relations, measured `+ flow + 3 rel_*`** | **29.4%** | 58.9% | 12 | +0.9849% | +3.6841% |

DOWN window: up-rate 27.2%, mean forward -2.1836%.
UP window: up-rate 73.9%, mean forward +4.3341%.

## 1. The query path fires. Step 2 is met.

Same fabric, same 180 samples, only the query set changed:

- **DOWN**: buy omens 41 -> 28, hit rate 19.5% -> 21.4%, and four of five
  predicted label counts moved (`trough` 41->28, `slide` 47->44, `climb`
  20->37, `crest` 22->21).
- **UP**: held-out exact **20.0% -> 26.7%**, twelve percentage points of
  predicted `climb` moved, buy omens 6 -> 4.

The single difference between the two query sets is the `flow` collection.
`flow` alone moved 13 buy omens in the DOWN window and 6.7 points of exact
accuracy in the UP window. A query-set change moves held-out predictions, in
both directions of market. Cove's fix in `6be5357` is proven live.

## 2. The pass-108 relation null was a broken comparison. Confirmed.

The pre-fix DOWN run at 14:20
(`omen-aero_down-h12-DOWN-20260910-142000.json`) trained **ten** collections,
including all three relation streams — its `measured_query_collections` lists
`rel_move_vol`, `rel_shape_flow`, `rel_trend_noise`. Its output is
**byte-identical** to my post-fix flat run overridden back to the old default
(`...-143454.json`), on a fabric that never saw a relation pool at all:

| | pre-fix, 10 collections, relations TRAINED | post-fix flat, 7 collections, relations ABSENT |
|---|---|---|
| fired | `temporal, geometry, cross` | `temporal, geometry, cross` |
| exact | 0.3 | 0.3 |
| buy omens | 41 | 41 |
| per-trade | -0.031564844684075666 | -0.031564844684075666 |
| hit rate | 0.1951219512195122 | 0.1951219512195122 |
| predicted mix | trough 41, slide 47, murk 50, climb 20, crest 22 | trough 41, slide 47, murk 50, climb 20, crest 22 |

Every label count equal, the per-trade return equal to seventeen significant
figures. Relation pools that are trained and never queried change literally
nothing. That is the diagnosis confirmed by measurement, not by reading the
source.

**This invalidates the pass-108 relation result rather than confirming it.**
The correct statement is not "the relations are redundant" — it is "the
relations were never asked".

## 3. The relation arm, measured honestly for the first time

With `OMEN_RELATION_COLLECTIONS=1` the three relation streams are trained
**and** fired. Their distinctness is real, not degenerate — `rel_move_vol`
reads 0.961 (DOWN) and 0.954 (UP), above `cross` and `flow` and well clear of
the 0.20 query floor. `rel_trend_noise` 0.617/0.611, `rel_shape_flow`
0.367/0.314. These are not near-constant buckets.

**The relations move held-out exact accuracy up in both windows**, against the
best flat arm measured back-to-back on the same corpus and windows:

- DOWN: 30.0% -> **31.7%** (+1.7 points)
- UP: 26.7% -> **29.4%** (+2.7 points)

Same direction in both windows, which is more than any previous topology
change here has managed. It is also small, on 180 predictions per cell.

**And it is NOT noise — this rig is deterministic.** A +1.7 point effect is
worthless if run-to-run variance is the 4.4 points the standing orders cite
(89.2% and 93.6% on the same fabric 34 minutes apart), so I replicated the
DOWN relation arm on a **fourth fresh brain dir**
(`brain-data-p109-gale-rep1`), node restarted, fabric rebuilt from empty:

| run | brain dir | exact | buy omens | total | per-trade |
|---|---|---|---|---|---|
| original | `brain-data-p109-gale-downrel` | 31.7% | 42 | -1.0696 | -2.5468% |
| replicate | `brain-data-p109-gale-rep1` | 31.7% | 42 | -1.0696 | -2.5468% |

Identical in every figure. At this scale — 311 training pairs, 180 held-out,
default `--seed 7`, one consolidation epoch — the pipeline reproduces exactly
and **observed variance is 0.0 points, not 4.4**.

Two consequences. First, the relation effect above is a real reproducible
difference rather than a lucky draw, which is the only reason a 1.7-point
result is worth writing down at all. Second, the 89.2%/93.6% variance in the
standing orders does not describe this configuration — whatever produced it
(a larger run, a shared or dirty fabric, a different seed or `--balance`
draw), it is not an inherent property of the rig, and small back-to-back
effects here should not be dismissed as noise by default. Anyone citing that
4.4-point band should say which configuration they measured it in.

## 4. Did it beat baseline? NO — and the one cell that did is a single window

State it plainly, because at-or-below baseline is the normal outcome here.

**Held-out exact accuracy is below the majority class in all six cells**, and
not marginally: the best cell is 31.7% against 55.0% in the DOWN window, and
29.4% against 58.9% in the UP window. A 23-to-29 point deficit. On the
scoreboard that actually counts, nothing here predicts.

**Per-trade net beats the trivial baseline in exactly one cell of six**, and
that is not an edge:

- DOWN relations: **-2.5468% against -2.8336%** for buying every bar. It loses
  less than the baseline, by 0.29 points over 42 trades.
- Every other cell loses to its baseline, including both UP relation and flat
  arms (+0.98% and +1.67% against +3.68%).

One window is not a result — that rule exists in the standing orders precisely
to stop a single favourable cell being reported as an edge, and it applies to
my own number here. The DOWN cell's advantage is 0.29 points on 42 trades in
one window, and the same arm underperforms its baseline by 2.7 points in the
UP window. **The relation topology does not have held-out edge.**

The UP window's `+1.6656% per trade` in the flat measured arm is the other
trap the standing orders name: a positive number that looks like an edge and
is 2.0 points **below** the buy-everything baseline measured in the same
window. Selection that destroys value, not selection that adds it.

`OMEN_STRATEGY_ENABLED` stays 0.

### The majority-class baseline is not just an accuracy baseline — it is a money baseline, and it wins in both windows

This has not been said in any report here, and it is the sharpest form of the
result. The majority label is a *tradeable rule*, because it names a direction:

- DOWN window: majority label is **`slide`**, 99 of 180 (55.0%). "Always
  predict slide" is "never buy". It places **0 trades for a total of 0.0000**.
  The relation arm places 42 trades for a total of **-1.0696**.
- UP window: majority label is **`climb`**, 106 of 180 (58.9%). "Always
  predict climb" is "buy every bar". It places 180 trades at +3.6841% for a
  total of **+6.6314**. The relation arm places 12 trades for a total of
  **+0.1182**.

Compared on total P/L over the same 180 bars — the honest comparison when the
arms place different numbers of trades — **the majority-class predictor beats
the brain in both windows**: 0.0000 against -1.0696 in the DOWN window, and
+6.6314 against +0.1182 in the UP window, a factor of 56.

So the accuracy deficit and the money deficit are the same fact, not two. A
five-class head that spreads its predictions is being beaten by a constant.
The brain's 30-31% does clear the 20% five-class chance rate, so the ordering
is carrying *something* — which matches the direction-head AUC of 0.56-0.59
measured elsewhere in this repo. But carrying something is not the same as
beating a constant, and on both scoreboards it does not.

## 5. Two things the next agent should not rediscover

**RELATION_COLLECTIONS is env-gated OFF, and the distinctness line tells you
which arm you are in.** A default run trains 7 collections and the `rel_*`
streams are absent from line 0 entirely. If you do not see `rel_move_vol` in
the distinctness table, you are running the flat arm no matter which identity
the node loaded. Check line 0 before you believe any label on the arm.

**Confidence is again worthless, and it is worse in the losing cells.** In the
DOWN window the default-query arm reads median confidence 0.974 while losing
-3.16% per trade; the measured-query arm reads 0.944 and loses -3.24%. The
confidence sweep is flat across every floor from 0.00 to 0.50 in all four
cells — no floor removes a single trade. Confidence cannot gate correctness
here, which is the third independent measurement of that fact in this repo.

## 6. What this does and does not license

It licenses the metacognition work (pools 15-19) to be measured at all: the
query path demonstrably carries a collection choice through to a changed
prediction, so a metacognition collection added to the query set can now be
falsified. Before this, it could not have been.

It does not license enabling anything. The relation topology is measured, in
both windows, and it does not clear baseline. What it has earned is one more
attempt rather than retirement: +1.7 and +2.7 points of exact accuracy in the
same direction in both windows is a weak but consistent signal, and the whole
run now costs 0.3-0.5 min of training for 311 pairs on a box with memory.

The honest next question is not "do relations help" — it is **why every arm
sits 25 points below the majority class**. A fabric at 100% train recall and
30% held-out against a 55% majority is not a topology problem at the margin;
it is reproducing rather than generalising, exactly as the standing orders
describe. A topology change that moves 2 points cannot close a 25-point gap.

### Reproduce any row

```
# flat arm (relations absent from the distinctness table)
python -X utf8 scripts/omen_experiment.py \
  --corpus data/brain_experiments/p108_aero_up.json \
  --horizon 12 --train 350 --test 180 \
  --endpoint http://127.0.0.1:8091 --report-dir data/brain_experiments

# relation arm (rel_* trained AND fired)
OMEN_RELATION_COLLECTIONS=1 python -X utf8 scripts/omen_experiment.py ...same...

# re-measure the SAME fabric with a different query set -- no retraining
... --skip-train --query-collections "temporal,geometry,cross"
```

Swap `p108_aero_up.json` for `p108_aero_down.json` for the DOWN window. A
fresh brain dir per window; never `:8090`.

### The six report files

| arm | report |
|---|---|
| DOWN flat, measured query | `omen-aero_down-h12-DOWN-20260910-143435.json` |
| DOWN flat, default query | `omen-aero_down-h12-DOWN-20260910-143454.json` |
| DOWN relations | `omen-aero_down-h12-DOWN-20260910-144039.json` |
| UP flat, measured query | `omen-aero_up-h12-UP-20260910-143658.json` |
| UP flat, default query | `omen-aero_up-h12-UP-20260910-143713.json` |
| UP relations | `omen-aero_up-h12-UP-20260910-143913.json` |
| pre-fix DOWN (the broken comparison) | `omen-aero_down-h12-DOWN-20260910-142000.json` |
