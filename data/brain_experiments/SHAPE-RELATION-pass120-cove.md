# SHAPE RELATION -- the 'same shape' descriptor, measured node-free

Pass 120, Cove. Cove's part of item `[5c3b2189]` (owner Jet).

This is NOT a fabric measurement and does not claim an edge. It is the
offline upper bound on what a pool carrying this stream could learn:
a descriptor -> majority-label lookup fitted on the train window and
read on two held-out windows. A substrate given this stream and nothing
else cannot beat its own frequency table out of sample, so a lookup
that fails here fails on the node too, for less than a minute of CPU.

## 1. Corpus and windows

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json` |
| bars | 21926 at 3600s cadence |
| horizon | 720 min = 12 bars |
| train window | `[19301, 20501)` |
| UP held-out | `[20513, 20913)` |
| DOWN held-out | `[21313, 21713)` |
| descriptor | 3 segments x 2 bands, chosen on the TRAIN half (sweep in §2) |
| min_support | 20 |

Windows are the pass-114 two-window protocol so these numbers sit
beside a measured fabric baseline on the same corpus and the same bars.
Window CLASS is not asserted -- it is read from mean forward return
before cost, printed in §4.

## 2. Granularity sweep -- TRAIN half only

The held-out windows are not read here. The winner is the scheme
covering the most of the train window at min_support, among schemes
whose most common descriptor holds at most 50% of the window and that
have at least four descriptors; ties break toward more descriptors.

| scheme | distinct | median/descriptor | top share | supported | covered | eligible |
|---|---|---|---|---|---|---|
| 2x2 | 2 | 600.0 | 0.5508 | 2 | 1.0000 | no  |
| 3x2 | 6 | 223.5 | 0.2625 | 6 | 1.0000 | yes **<-- chosen** |
| 4x2 | 14 | 68.5 | 0.2783 | 10 | 0.9717 | yes  |
| 2x3 | 4 | 359.0 | 0.4000 | 3 | 0.9983 | yes  |
| 3x3 | 18 | 32.0 | 0.2975 | 12 | 0.9458 | yes  |
| 4x3 | 36 | 19.0 | 0.1242 | 17 | 0.8508 | yes  |
| 6x3 | 110 | 6.0 | 0.0950 | 14 | 0.4667 | yes  |
| 4x5 | 90 | 8.0 | 0.0675 | 21 | 0.6108 | yes  |
| 8x5 | 359 | 3.0 | 0.0317 | 1 | 0.0317 | yes  |

## 3. Gate 1 -- invariance under the label-safe mutations

| mutation | n | path changed | descriptor SAME | label flips |
|---|---|---|---|---|
| `deep_jitter` | 150 | 1.0000 | **0.9867** | 0 |
| `deep_flatten` | 150 | 1.0000 | **0.5333** | 0 |
| `deep_dilate` | 150 | 1.0000 | **0.3600** | 0 |

`path changed` is the share of anchors where the mutation really moved
the price path -- without it a high SAME rate would mean nothing. Label
flips must read 0: the admitted mutations are deep-prefix only and
cannot touch entry, future or position-in-range.

## 4. Gates 2 and 3 -- non-degeneracy and support, TRAIN half only

| | |
|---|---|
| labelled anchors | 1200 |
| distinct descriptors | 6 |
| median anchors per descriptor | 223.5 |
| share held by the most common | 0.2625 |
| descriptors clearing min_support=20 | 6 |
| share of the train window they cover | 1.0000 |
| lookup entries fitted | 6 |

## 5. Gate 4 -- held-out lift, both window classes

| window | n | answered | abstention | exact | se | baseline (answered) | lift | mean forward |
|---|---|---|---|---|---|---|---|---|
| UP | 400 | 400 | 0.0000 | **0.1800** | 0.0192 | 0.3200 | **-14.00pp** | 0.011568 |
| DOWN | 400 | 400 | 0.0000 | **0.2575** | 0.0219 | 0.2975 | **-4.00pp** | -0.004170 |

The baseline is the majority class OF THE ANSWERED SUBSET, not of the
whole window: an arm that abstains is not entitled to credit for the
bars it declined, and comparing an abstaining arm against the
unconditional baseline is how abstention gets mistaken for skill.

## 6. Verdict

**NEGATIVE in both windows -- the stream carries no held-out lift**

## 7. What this does NOT establish

* It is not a fabric number. Nothing was trained; the node was not
  touched and production on `:8090` was not involved.
* A lookup table is an UPPER bound on this stream alone, not a
  prediction of what the fabric does with it alongside ten other pools.
* One corpus. The pass-114 binding limitation is unchanged.
* `PoolKind::Internal` remains inert (0 matches across `crates/`; the
  only behavioural `PoolKind` match is `Action` at `brain.rs:7425`), so
  if this stream ever ships it ships as a client-computed
  `SensoryInput` relation pool, the pass-106 pattern.
