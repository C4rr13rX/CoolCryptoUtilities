# SHAPE RELATION ACROSS THE CORPUS POPULATION

Pass 120, Cove. Cove's part of item `[5c3b2189]` (owner Jet). This
widens the single-corpus result in `SHAPE-RELATION-pass120-cove.md`
from one tape to many. Still node-free, still an OFFLINE UPPER BOUND:
a descriptor -> majority-label lookup is the best a substrate could do
with this stream alone, so a stream that fails here fails on a fabric.

## 1. Protocol

| | |
|---|---|
| corpora | `data/historical_ohlcv/base/*.json`, first 12 by name |
| horizon | 720 min, converted per corpus cadence |
| train | 1200 bars, ending a full horizon before the earlier test window |
| held-out | 400 bars per window class |
| window classes | the most positive and most negative of the last 10 disjoint blocks, by mean forward return BEFORE cost |
| granularity | chosen on each corpus's OWN train half, sweep `2x2,3x2,4x2,2x3,3x3,4x3,6x3,4x5,8x5` |
| min_support | 20 |

Choosing the extreme blocks is deliberate: it is the hardest honest
pair, and a rule that works in only one direction cannot hide in it.

## 2. Per corpus

| corpus | scheme | train n | UP exact | UP baseline | UP lift | DOWN exact | DOWN baseline | DOWN lift |
|---|---|---|---|---|---|---|---|---|
| `0004_AERO-USDC.json` | 3x2 | 1200 | 0.2675 | 0.2725 | **-0.50pp** | 0.2525 | 0.3400 | **-8.75pp** |
| `0005_AERO-USDC.json` | 3x2 | 1200 | 0.2550 | 0.3275 | **-7.25pp** | 0.3875 | 0.3875 | **0.00pp** |
| `0006_AERO-USDC.json` | 3x2 | 1200 | 0.2750 | 0.3775 | **-10.25pp** | 0.2200 | 0.3650 | **-14.50pp** |
| `0006_cbBTC-USDC.json` | 3x2 | 1200 | 0.2575 | 0.5450 | **-28.75pp** | 0.6225 | 0.6225 | **0.00pp** |
| `0007_AERO-USDC.json` | 3x2 | 1200 | 0.2500 | 0.3100 | **-6.00pp** | 0.2750 | 0.3375 | **-6.25pp** |
| `0007_VELVET-USDC.json` | 3x2 | 1200 | 0.2200 | 0.4225 | **-20.25pp** | 0.2975 | 0.5650 | **-26.75pp** |
| `0008_CBBTC-USDC.json` | 3x2 | 1200 | 0.2575 | 0.5625 | **-30.50pp** | 0.8575 | 0.8575 | **0.00pp** |
| `0009_JITOSOL-CBBTC.json` | 3x2 | 1200 | 0.5450 | 0.5450 | **0.00pp** | 0.1450 | 0.3900 | **-24.50pp** |

Skipped, and why -- a corpus that cannot supply both window
classes is not folded in:

* `0004_CBBTC-USDC.json` -- unreadable: placeholder corpus -- no bar objects in the file
* `0005_CBBTC-WETH.json` -- unreadable: placeholder corpus -- no bar objects in the file
* `0007_EURC-USDC.json` -- no UP/DOWN pair in 1653 bars
* `0009_HYPE-BASEDHYPE.json` -- unreadable: placeholder corpus -- no bar objects in the file

## 3. Population result

* corpora scored: **8**
* positive in BOTH window classes: **0**
* positive in one window class only (a FAIL): 0
* mean lift: **-12.94pp** UP, **-10.09pp** DOWN

**NEGATIVE ACROSS THE POPULATION -- 0 of 8 corpora positive in both window classes, mean lift -12.94pp UP and -10.09pp DOWN**

## 4. What this does NOT establish

* Nothing was trained. The node was not touched; production on
  `:8090` was not involved.
* A lookup is an upper bound for THIS STREAM ALONE. It does not
  predict what a fabric does with the stream beside ten other pools.
* The window classes are the extremes of each corpus's own tail, so
  they are harder than average windows, not representative ones.
