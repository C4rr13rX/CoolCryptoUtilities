# price_mu is a one-bar log return, and the head is 490x its own label

Pass 117, Iris, 2026-09-11. Item [1b0fd55f].

## The question

`trading/bot.py` reads the model's `price_mu` head as `delta`, a forward
expected return, and [4d3310e7] put `delta >= roundtrip_cost_rate(notional)`
into the entry conjunction. Pass 112 measured median |delta| 93.16% against a
median realised 15-minute move of 0.1240%. Two explanations were live: the
head forecasts a far longer horizon than 15 minutes, or its target is not a
return at all.

## Corpus and windows

* Label corpus: `data/historical_ohlcv`, 629 files. 120 sampled at seed 7 →
  2,277,175 one-bar log returns.
* Bar cadence, same sample: **3600s on 95 files**, 600s on 6, 300s on 6, the
  rest one-offs. The corpus is HOURLY.
* Served model: `models/active_model.keras`, built 2026-09-11 01:57,
  `PriceVolScaleNorm` PRESENT in the graph.
* Live cycles: newest 5,000 `organism_snapshots` cycles, 20.2h, 28 symbols.

## 1. What the target is

`trading/data_loader.py` (loop at ~1110):

    current_price = closes[end - 1]
    future_price  = closes[end]          # next_idx = end, line 1008
    mu = log(future_price) - log(current_price)

It is a **return**, and the lookahead is **exactly one bar = one hour**. Not
15 minutes, and nothing longer. The "longer horizon" story is dead by
construction.

Label distribution over 2,277,175 samples:

| statistic | value |
|---|---|
| median \|mu\| | 0.003617 |
| p99 \|mu\| | 0.034602 |
| max \|mu\| | 0.907800 |
| frac(mu > 0) | 0.4720 |
| frac \|mu\| >= 0.2065 | 0.0036% (83) |
| frac \|mu\| >= 0.93 | 0 |

So the "clean window" served value of -0.2065 sits at the 99.9964th percentile
of the label, and the production p50 of -1.2076 is **beyond the largest label
the corpus can produce**. `data_loader.py`'s claim that -0.1655..-0.2428 was
"comfortably inside the band the head was trained on" was false; corrected in
this pass.

## 2. The head against its own training bars

New arm, `scripts/model_window_probe.py --train-samples 40`. Forty windows
drawn straight from `data/historical_ohlcv` — the bars the head was fitted on —
with every auxiliary input identical to the LIVE section, so the only thing
that differs between the two sections is the price/volume window.

    TRUE  : median |mu|  0.002858   p99 0.028920   max 0.031322
    PRED  : median |mu|  1.399697   p99 3.056572   max 3.110560
    RATIO : 489.8x
    SIGN  : agrees on direction 62.5% of 40   (n too small to mean anything)

**The head is uncalibrated on clean, in-corpus windows.** It does not need a
contaminated window and it does not need a long horizon.

Two collateral findings from the same run:

* Contamination moves price_mu the WRONG WAY as often as not. Clean
  JACKET-USDC -0.6491; one foreign ETH row -0.1715; ETH/JACKET interleaved
  -0.0419. Only the x100 row inflates it (-1.0071). Dirtying a window is not
  what produces the large magnitudes.
* **The artifact regressed.** The LEVEL probe (a fixed +0.2%/step ramp) read
  price_mu -0.000620 on 2026-09-10 and reads **-1.160425** today, identical to
  six decimals across eight orders of magnitude at both dates. The scale-free
  transform still works; the head moved by three orders of magnitude between
  the two artifacts.

## 3. The census at the true horizon

`scripts/entry_move_size_census.py --limit 5000`, back-to-back on one cycle
set, horizon the only variable:

| horizon | median \|delta\| | median realised | ratio | overstates |
|---|---|---|---|---|
| 15 min (as filed) | 28.33% | 0.1158% | 244.6x | 98.8% of 4,446 |
| **60 min (the head's actual horizon)** | **27.05%** | **0.2896%** | **93.4x** | **98.3% of 3,851** |

Knowing the true horizon cuts the exaggeration from 244.6x to 93.4x and
changes nothing that matters.

## 4. What the cost conjunct does now

Over the newest 20.2h:

    delta >= 0.0        alone : 3211 of 5000 (64.2%)
    delta >= 1x cost    alone : 3182 of 5000 (63.6%)   -- 29 cycles, 0.9%
    vol_rel >= 1x cost  alone :  593 of 5000 (11.9%)

Against the 484h figure the item was filed on (19,518 → 5,451, 72.1% fewer),
the term now removes **0.9%**. The marginal denominator — cycles where every
non-delta conjunct passes — is **2 of 5000**, so the census cannot resolve
better than 50 percentage points there and no tuning of the threshold is
measurable.

**VERDICT: delta is not comparable to a round trip.** A head whose median
forecast is 27% sails over any 0.39-0.86% cost floor by construction, so the
conjunct is not a cost test — it is a sign test with extra arithmetic. It
should be revisited at `trading/bot.py`, not tuned. `vol_rel` is the term in
that census with the right units: median 0.1309% against a realised 0.2896%,
ratio 0.45, and it overstates on only 30.6% of cycles.

## Commands that reproduce this

    D:/Projects/CoolCryptoUtilities/.venv/Scripts/python.exe -X utf8 \
        scripts/model_window_probe.py --train-samples 40 --symbols 4
    python -X utf8 scripts/entry_move_size_census.py --horizon 3600 --limit 5000
    python -X utf8 scripts/entry_move_size_census.py --horizon 900  --limit 5000
    python -X utf8 -m pytest \
        tests/test_price_mu_is_a_one_bar_log_return_not_a_long_horizon_forecast.py -q

## What I would try next

The head is broken at the source, not at the seam, so the next question is the
training run rather than the serving path: `price_mu` is supervised twice —
weakly through `price_gaussian`/`gaussian_nll_loss` (where a free `log_var`
from the same `Dense(2)` can inflate and kill the gradient on `mu`) and
strongly through `net_margin` MSE at loss_weight 1.0. A weight-1.0 MSE against
a label of median 0.0036 cannot leave the head at 1.40 if it converged, so the
run is either not converging or the artifact was not produced by this loss.
Compare the training history of the 2026-09-11 01:57 artifact against the
2026-09-10 one, which was 1,800x better on the identical LEVEL probe.
