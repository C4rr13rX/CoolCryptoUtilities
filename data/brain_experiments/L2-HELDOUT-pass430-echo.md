# The shipped L2, scored held-out for the first time, FAILS

Pass 430, Echo (senior). Item [e57fe7b0]. Measurement only — no production
code, no threshold changed, no gate touched. `OMEN_STRATEGY_ENABLED` stays 0.

## One command per window

```
python -X utf8 scripts/omen_layer_probe.py \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --horizon 12 --heldout --relative-bands --train 2400 --test 600 \
  --pick-window down --json-out data/brain_experiments/p430-l2-heldout-down.json
# ...and --pick-window up for the other window.
```

## Why this arm could be spent when p108's could not

`heldout_edge` scores the TAIL of whatever bar list it is handed. That is why
every previous L2 arm was a single-window read, and on the p108 corpora the
largest L2 group was 17 of 539 train bars — below the n>=20 support floor, so
the probe refused the arm and the shipped L2 had **no held-out number
anywhere**.

On `0004_AERO-USDC.json` (21926 bars, same encoder) the support is there:
**20 L2_transitions groups at n>=20** in the DOWN train split, 10 in the UP.
The arm is spendable, and it was spent.

## The window rule, fixed before any held-out number existed

`block_census()` in `scripts/omen_layer_probe.py`. Cut the corpus into
consecutive non-overlapping blocks of `train + test + horizon + lookback + 1`
bars from index 0; keep whole blocks; **DROP THE FINAL BLOCK**; rank on
`log(last close / first close)`; UP = max, DOWN = min. The probe prints the
whole census so the choice is auditable rather than asserted:

```
WINDOW CENSUS -- 5 non-tail blocks of 3181 bars (block 5 IS the tail, excluded)
  block  0  bars [     0,  3181)  log return +2.1692   <== UP
  block  1  bars [  3181,  6362)  log return +0.4129
  block  2  bars [  6362,  9543)  log return -0.7820   <== DOWN
  block  3  bars [  9543, 12724)  log return +0.4902
  block  4  bars [ 12724, 15905)  log return -0.3292
```

Both windows carry 2400 train samples and 588 test samples with a 12-bar purge.
Neither is the corpus tail. "Every brain number scored ONE down window" is the
failure this rule exists to stop repeating.

## The result — both windows, sample size per cell, net of ROUND_TRIP_COST 0.65%

| window | arm | per-trade net | n called | baseline (buy-every-bar) | n base | EDGE | sd | SE | t |
|---|---|---|---|---|---|---|---|---|---|
| UP block 0 | **L2_transitions (SHIPPED)** | −3.6607% | 29 | −1.2002% | 588 | **−2.4605%** | 2.2179% | 0.4119% | **−5.97** |
| UP block 0 | L2_sequence (rejected control) | — | 87 | — | 588 | +0.2641% | 2.2179% | 0.2378% | +1.11 |
| DOWN block 2 | **L2_transitions (SHIPPED)** | −0.5838% | 27 | −1.2290% | 588 | **+0.6452%** | 5.0554% | 0.9729% | **+0.66** |
| DOWN block 2 | L2_sequence (rejected control) | — | 95 | — | 588 | +0.3151% | 5.0554% | 0.5187% | +0.61 |

95% CI on the shipped arm: UP **[−3.268%, −1.653%]**, DOWN [−1.262%, +2.552%].

## What it says

1. **The one resolvable number is a significant LOSS.** The UP window's
   −2.4605% per trade is t = −5.97 with a CI that does not come within 1.65pp
   of zero. This is not "inside the noise" — it is the shipped L2 being
   reliably worse than buying every bar, by more than three round-trip costs.

2. **The positive number is noise.** DOWN's +0.6452% has SE 0.9729% and t =
   0.66. Its CI straddles zero. Quoting it alone would be the single-window
   read this item was written to prevent — and note it is the *sign-flip
   partner* of the significant loss, not independent support.

3. **The rejected scheme beats the shipped one.** [fa75fa1a] rejected
   `L2_sequence` on DISTINCTNESS — a statement about vocabulary size, not about
   prediction. On the only held-out evidence that has ever existed, the
   rejected scheme is the better of the two: margin of shipped over rejected is
   **−2.7245%** per trade in UP and +0.3301% in DOWN. The distinctness criterion
   picked the worse predictor.

4. **The dispersion is 2.2–5.1% per 12-bar trade, not the ~2% assumed.** At the
   DOWN window's 5.0554% sd and its 4.6% call rate, resolving a 0.65pp edge at
   2 SE needs **236 trades = ~5,136 test bars**. A 588-bar window cannot do it.
   Any future single-window L2 arm on this corpus is unresolvable by
   construction, and should not be run expecting an answer.

## Verdict

**FINISHED NEGATIVE. Not retried, and it should not be retried.** The item says
a result at or below baseline is a finished measurement. It is below baseline in
the window where the measurement has power, and unresolvable in the window where
it does not. Neither arm abstained, so neither is UNMEASURABLE — this is a real,
spent, negative result.

The shipped L2 does not earn its consolidation and query per sample. This is
consistent with, and independent of, `wizard-brain-untrained` and
`brain-first-trained-result-below-baseline`: the layer is not merely untrained,
its motif->trough map is anti-predictive out of sample in the window that can
see.

## What this does NOT say

It does not say the brain is worth another pass. It says the opposite: the
strongest remaining argument for the L2 was "it has never been fairly scored."
It has now been fairly scored, on a corpus with real support, in two windows
picked by a published rule, against its own rejected control. It lost.
`OMEN_STRATEGY_ENABLED` stays 0 and the brain stays paused.
