# The curvature is a null too — acceleration-into-move does not exist on this tape

**Route:** the operator's reflexivity hypothesis. Other systems trade on the
expected reactions of others, so the causal trigger may be boolean while the
response path is higher-order. First-order price-action predicates are measured
exhausted here (continuation under 1bp at every horizon; 2% dips INVERT out of
sample), so the surviving question is whether structure lives in the CURVATURE
of the response rather than in its level.

**Answer: NO.** Across K=18 cells (6 |first-derivative| buckets x 3 statistics)
on 8,788 fires over the 9 pairs the round-trip quote census can actually price,
**not one cell clears its own noise bar**. The largest curvature point estimate,
+11.45bp, sits under its own 15.2bp bar *and* under the 24bp cost of the
cheapest pool we can fill.

Run it: `python -X utf8 scripts/feature_forge_second_order.py` (~35s incl. the
1000-trial permutation null). Commit `7b32a8a`.

---

## 1. The fillable set is READ, never assumed

`data/round_trip_quote_census_20260917T190210Z.json` (git `769cc58`), its
`quoted` block: **9 of 18 symbols**. The other 9 are UNQUOTABLE and the census
charges them nothing — there is no default, so they are simply not in the
universe. The quote denomination (USDC) is parsed out of the census's own
`what_is_measured` string, because it prices DIRECT USDC<->token pools only.

| pair | ticks | med gap s | p90 gap s | span h | regular% | zero-return% | blackouts >16m | newest tick age h | round trip bp |
|---|---|---|---|---|---|---|---|---|---|
| AAVE-USDC | 511 | 118 | 1126 | 480.5 | 72.4% | 1.8% | 58 | 28.1 | 60.3 |
| AERO-USDC | 8244 | 31 | 291 | 530.4 | 62.6% | 15.8% | 207 | 0.0 | 10.1 |
| CBBTC-USDC | 4318 | 17 | 278 | 500.6 | 62.1% | 11.7% | 115 | 36.2 | 14.5 |
| CBETH-USDC | 4458 | 18 | 308 | 513.5 | 62.3% | **49.2%** | 131 | 23.4 | 67.1 |
| COMP-USDC | 4438 | 14 | 252 | 506.7 | 63.6% | **61.6%** | 124 | 24.1 | 3301.5 |
| DRB-USDC | 2388 | 73 | 371 | 325.2 | 72.4% | 11.1% | 73 | 39.4 | 1507.6 |
| VIRTUAL-USDC | 1393 | 109 | 534 | 484.6 | 71.6% | 1.9% | 83 | 0.0 | 67.6 |
| VVV-USDC | 3024 | 60 | 364 | 255.6 | 71.0% | 14.1% | 87 | 39.7 | 212.5 |
| WETH-USDC | 1717 | 97 | 543 | 525.6 | 70.5% | 0.1% | 81 | 0.0 | 4.1 |

**Two coverage facts that are findings in their own right.**

1. **Six of the nine "fillable" pairs have a DEAD tape.** AAVE, CBBTC, CBETH,
   COMP, DRB and VVV have newest ticks **23–40 hours old**, against a table whose
   newest row is minutes old. Only AERO, VIRTUAL and WETH are current. Quotable
   is not the same as streamed — this is the mirror image of
   `predictability-and-fillability-are-anticorrelated`, where the deep pools
   existed and we weren't streaming them.
2. **Half of some pairs' ticks are re-prints.** COMP 61.6% and CBETH 49.2% of
   consecutive ticks carry an IDENTICAL price. A second derivative over a
   re-print is structurally zero, which is why the bucket medians below sit at
   -0.001 to -0.2bp while the means are dominated by a thin tail.

Denomination filter (structural log-gap clustering, same test as
`scripts/money_button_edge_test.py`) dropped 175 ticks, 0.57% of the subset —
almost all COMP, whose raw sigma reads 816bp against 13bp for CBBTC.

## 2. The cost floor the effect has to clear

- Cheapest measured round trip in the fillable set: **WETH-USDC at 4.1bp**
  (pool cost only — a floor, not a realisable cost).
- Measured gas at the $2 median clip: **+20.2bp**. Live measured round-trip
  slippage **13.5bp** that no quote carries at all.
- So **~24bp in the cheapest pool we can fill**, and **3,322bp on COMP-USDC**,
  which the census still calls quotable.

## 3. Computability — 51.9% REPRODUCED, not corrected

Re-running the published definition unchanged (runs of three consecutive ticks
whose gaps are each within 2x that symbol's median, over the whole table):

> **54,846 usable triples across 361 symbols = 51.9% of 105,608 rows.**

Published figure was 51,807 / 353 symbols / 51.9%. **REPRODUCED to 0.0pp.** The
table has grown from ~99.8k rows to 105,608 since pass 432 and the triple count
grew with it; the fraction did not move.

**But that is a whole-table number over 361 symbols we mostly cannot fill.** On
the fillable subset it is much smaller, and per feature:

| feature | n | fraction of the 30,491 fillable ticks | mean | median |
|---|---|---|---|---|
| acceleration (bp per median-gap²) | 7,479 | **24.5%** | 0.5013 | 0.0000 |
| curvature (dimensionless κ, σ/median-gap coords) | 7,479 | **24.5%** | -0.0071 | 0.0000 |
| convexity_post_move (bp per median-gap²) | 438 | **1.4%** | -3.0943 | -0.3122 |
| spacing_asymmetry (up-gap − down-gap)/mean | 18,573 | **60.9%** | 0.0122 | 0.0190 |
| post_move_decay (log-slope per median-gap) | 438 | **1.4%** | 0.0464 | 0.0439 |

So: **51.9% is right for the tape and wrong for the trade.** The second-order
feature class is computable on a quarter of the fillable ticks, and the
*post-move* features — the ones that actually encode "how does the crowd respond
to a move" — exist on **1.4%**, n=438. That is the real ceiling on this
hypothesis and it was not visible in the whole-table number.

**One knob had to be added.** The published triple definition bounds a gap only
from ABOVE. That is fine for counting, but the time-normalised second derivative
divides by `h1*h2*(h1+h2)`, and with no lower bound its maximum is **2.0e10** bp
per median-gap² — a division artifact, not a market. Requiring gaps to be within
[0.25, 2.0]x the median brings the maximum to **1.3e3** and costs about half the
triples. The bp-space second *difference* needs no division and is measured
without that bound.

## 4. THE REFLEXIVITY TEST

Fires: 8,788 on 9 pairs. A fire is a non-zero move r0 arriving over a comparable
gap, followed by two further ticks at comparable gaps. **The bucketing move r0 is
deliberately DISJOINT from the statistic** — `d2 = sign(r0)*(r2-r1)` uses only
the two *subsequent* returns. Centring the second derivative on the move instead
would put r0 inside its own statistic, and conditioning on large |r0| would then
manufacture positive signed acceleration out of nothing. Any version of this test
that does not do this is measuring its own selection.

SEs are cluster-robust on (symbol, 6h block), because consecutive fires overlap
by up to two ticks.

### Signed second difference, sign(r0)*(r2−r1), bp — >0 = curving INTO the move

| \|r0\| bucket bp | n | mean bp | SE clust | **bar** | median bp | blocks | verdict |
|---|---|---|---|---|---|---|---|
| 0.00–0.06 | 1757 | +0.422 | 0.592 | 1.696 | -0.001 | 106 | **null** |
| 0.06–0.20 | 1758 | -0.188 | 0.315 | 0.904 | -0.004 | 125 | **null** |
| 0.20–0.63 | 1757 | +0.083 | 0.507 | 1.452 | -0.014 | 147 | **null** |
| 0.63–3.01 | 1758 | +0.219 | 0.426 | 1.222 | -0.102 | 166 | **null** |
| 3.01–17.45 | 1318 | +0.992 | 0.652 | 1.869 | -0.211 | 158 | **null** |
| 17.45–inf | 440 | **+11.452** | 5.313 | **15.225** | +2.285 | 94 | **null** |

### Signed second derivative, time-normalised, bp per median-gap²

| \|r0\| bucket bp | n | mean | SE clust | **bar** | median | blocks | verdict |
|---|---|---|---|---|---|---|---|
| 0.00–0.06 | 682 | +0.327 | 0.798 | 2.287 | -0.001 | 76 | **null** |
| 0.06–0.20 | 835 | +0.118 | 1.007 | 2.887 | -0.026 | 108 | **null** |
| 0.20–0.63 | 929 | -0.251 | 1.065 | 3.051 | -0.104 | 130 | **null** |
| 0.63–3.01 | 1083 | -0.587 | 1.751 | 5.017 | -0.339 | 148 | **null** |
| 3.01–17.45 | 861 | +2.922 | 1.959 | 5.613 | -0.000 | 138 | **null** |
| 17.45–inf | 321 | +22.843 | 9.725 | 27.870 | +2.566 | 74 | **null** |

### Signed first-order continuation, sign(r0)*r1, bp — the level, for contrast

| \|r0\| bucket bp | n | mean bp | SE clust | **bar** | median bp | verdict |
|---|---|---|---|---|---|---|
| 0.00–0.06 | 1757 | -0.142 | 0.300 | 0.825 | +0.017 | **null** |
| 0.06–0.20 | 1758 | +0.341 | 0.209 | 0.573 | +0.094 | **null** |
| 0.20–0.63 | 1757 | +0.265 | 0.360 | 0.988 | +0.284 | **null** |
| 0.63–3.01 | 1758 | +0.397 | 0.345 | 0.949 | +0.764 | **null** |
| 3.01–17.45 | 1318 | +0.463 | 0.448 | 1.230 | +1.689 | **null** |
| 17.45–inf | 440 | **-10.618** | 4.186 | 11.500 | -3.638 | **null** (clears the analytic bar, fails the measured one) |

**ACCELERATION-INTO-MOVE DOES NOT EXIST ON THIS TAPE.** Every bucket's second
derivative is inside its own noise bar, in both the bp and the time-normalised
formulation, at every move size from 0.06bp to unbounded.

## 5. The multiple-comparisons bill, paid AND priced

K is counted mechanically from the cells the script actually renders, not
asserted: **K = 18**, so the analytic bar is `SE * sqrt(2 ln 18) = SE * 2.404`.

Then the same family-wise quantity was **measured** rather than assumed: flip the
sign of whole (symbol, 6h) blocks — which preserves every magnitude, every bucket
boundary and the within-block dependence, and destroys only the directional link
— and take `max |mean|/SE` over all 18 cells. 1000 trials:

> median **1.87**, 95th percentile **2.75** (2.67–2.75 across seeds 7/11/23).

**The analytic constant 2.404 is OPTIMISTIC here**, by about 14%. The script uses
`max(analytic, permutation)` and says so. This matters: at the analytic bar one
cell survives (first-order reversion at |r0|>17.45bp, -10.6bp vs bar 10.06); at
the measured bar nothing does. Anyone quoting `SE*sqrt(2 ln K)` on this tape
without checking it is quoting a bar ~14% too low.

## 6. The two things that would have been a false positive

**(a) The 17.45bp+ bucket's +11.45bp.** It is inside its bar, but it is also
*not robust*: the same cell's first-order partner, -10.618bp, has **75% of it
carried by DRB-USDC alone** (76 of 440 fires). DRB's round trip is **1,508bp**.
The one number in this study large enough to be tempting is concentrated in the
second-most-expensive pair the census will still quote — the
predictability/fillability anti-correlation arriving again, this time *inside*
the fillable set.

**(b) The spike filter, which is the trap.** 899 of 8,788 fires (10.2%) are
single-tick spike-reversals — the very next tick undoes ≥50% of the move in the
opposite direction, i.e. a bad print and its correction. They carry mean d2 =
**+19.16bp** against **-1.26bp** for the rest, and excluding them flips three
middle buckets to a "significant" NEGATIVE (deceleration) result.

**That flip is the filter, not the tape.** The spike test selects on r1, and r1
is *inside* the statistic (`d2 = s*r2 - s*r1`), so dropping the most negative
`s*r1` drags every bucket mean down mechanically. The script reports the
diagnostic, refuses to count it in K, and does not use it. The unbiased
robustness checks are the median column (which agrees: all near zero) and the
leave-one-symbol-out jackknife.

## 7. What this closes and what it does not

**CLOSED:** second-order price-action structure on the fillable subset of this
feed, at the tick-triple horizon. Curvature is exhausted in the same way the
level was. Do not re-commission "acceleration", "convexity", "curvature" or
"post-move decay" as *price-only* features on this tape without new data.

**NOT CLOSED, and it is the same wall every route hits:**

- **Six of nine fillable pairs have a tape that stopped 23–40h ago.** The
  hypothesis was tested largely on history, not on a live feed. Fixing what we
  stream is upstream of every feature question.
- **The post-move features have n=438, 1.4% of ticks.** Reflexivity is a
  statement about the *response to a move*, and this feed supplies 438 measurable
  responses across 9 symbols and ~500 hours. Even a real effect of a few bp
  cannot be proven at that n against a 24bp cost floor — the bill is the SAMPLE,
  not the search.
- **Price is not the only observable, and the others are not recorded.**
  `market_stream.volume` is **zero or null on 30,665 of the 30,666 fillable ticks
  — 100.0%**. Order flow, pool reserves and mempool data are not in the table at
  all. Reflexivity is a claim about *participants reacting to participants*; if
  it is here, it is far more likely to be visible in flow than in the price path,
  and we record no flow. **Instrumenting volume/flow on the three pairs that are
  still streaming is a strictly better next route than any further price-only
  search**, because the price-only search now has two exhaustion results (level
  and curvature) and no remaining degrees of freedom worth spending.
