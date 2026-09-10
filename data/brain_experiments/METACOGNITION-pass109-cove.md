# Metacognition and temporal pools: wired, read, and BELOW baseline in both windows

Pass 109, Cove, 2026-09-10.

## The short version

The five metacognition/temporal pools (15-19) are now streamed and are
genuinely read by the query path. That is proven. **They do not produce an
edge.** Held-out accuracy is below the majority-class baseline in an UP window
and far below it in a DOWN window, and in the DOWN window the brain's buy
omens lost *more* per trade than buying every bar indiscriminately.

This is the normal outcome here and it is a finished pass. What it buys is a
mechanism that now demonstrably works and one specific, measured reason the
temporal idea did not land — see §4.

## 1. Corpus, windows, node

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21926 bars, 3600s cadence |
| pairs | 1 (AERO-USDC) |
| horizon | 12 bars = 720 min; omen threshold 0.9750% (round trip 0.6500%) |
| node | fresh `brain-data-meta-p109-2win` on `:8092`/`:8093`, `market_predictor_v4_meta` identity (19 pools) |
| train window | bars [841, 1441), **PINNED** with `--train-end 1441`, 600 samples, ONE epoch |
| UP window | bars [1453, 1573) — up-rate 56.7%, mean forward **+1.3793%** |
| DOWN window | bars [1813, 1933) — up-rate 14.2%, mean forward **-2.6721%** |

Both windows were measured **on one fabric**: the UP run trained, the DOWN run
used `--skip-train`. Train recall reads 99.5% identically in both, which is the
check that the second run really did re-measure the same fabric rather than a
new one. Windows were **picked from `--list-windows`**, not hoped for.

## 2. The headline numbers

| | UP window | DOWN window |
|---|---|---|
| held-out exact | **25.8%** (120/120 admitted) | **19.2%** (120/120 admitted) |
| majority-class baseline | 26.7% | 65.0% |
| verdict vs baseline | **below** | **far below** |
| buy omens | 15 | 50 |
| omens that paid | 53.3% | 6.0% |
| per-trade net | +1.4610% | **-3.7937%** |
| every-bar buy | +0.7293% | -3.3221% |
| verdict vs buy-everything | above | **below** |

**Read the UP column honestly.** +1.4610% against +0.7293% looks like an edge
and is not: a long-only rule flatters itself in an up window, which is the
exact error that produced a fake 78% and a fake +0.9067% in this repo before.
The DOWN column is the same rule's bill. Taken together the brain has no edge.

The confidence sweep is flat in both windows — every threshold from 0.00 to
0.50 admits the identical trade set. Confidence is not a gate here, again.

## 3. What IS proven: the query path fires the pool

Two mechanism results, measured with `scripts/omen_query_path_probe.py`, which
trains one fabric and predicts the SAME held-out samples under two query sets.
A negative control (query set against itself) must move zero, and does — so the
node is deterministic and any diff is attributable to the query set.

| arm | control (A vs A) | treatment | streams fired |
|---|---|---|---|
| relation pools 12/13/14 | 0/120 | **79/120 moved** | 3 -> 6 |
| temporal pools 17/18 | 0/120 | **77/120 moved** | 3 -> 5 |

And `taught 600/600` with the meta pools on: enabling them against the v4 node
does **not** trigger the silent-MISS failure the env gate guards against.

This retires the pass-108 conclusion that the relation pools are redundant.
That arm came back byte-for-byte identical over 180 predictions because
`omen_experiment.py` computed the measured query set, printed it, and passed
`None` — so both arms fired the hard-coded default against fabrics whose extra
pools were bound and never read. Fixed in `6be5357`. **It measured nothing and
needs re-running, not recording.**

## 4. The measured reason the temporal idea did not land

The dilution law, run on this corpus, ranks every collection:

```
temporal=1.000  geometry=0.990  cross=0.493  temporal_sequence=0.323
flow=0.305  volatility=0.175  temporal_scale=0.045  horizon=0.002
instrument=0.002  self_outcome=0.002  self_agreement=0.002  self_error_run=0.002
```

Three things fall out, and the operator's guess was two-thirds right:

* **Pool 17 `temporal_sequence` earns its place — 0.323, above the 0.20 bar.**
  The measured query set selected it without being told to:
  `(geometry, temporal, flow, cross, temporal_sequence)`. Order is a
  discriminating stream. That part of the design works.
* **Pool 18 `temporal_scale` is too coarse to survive the filter — 0.045.**
  This is the sharp finding. Multi-scale structure is the mechanism that was
  supposed to let the brain see a regime, and its frame
  (`scl s3=u s12=u s48=d agree=split`) has so few distinct values that the
  dilution law excludes it from the query. **The one pool aimed at regime
  never fires.** That is a frame-resolution problem, not a topology problem,
  and it is fixable: the scales collapse to a 3-token direction each.
* **Pools 15/16/19 read 0.002 — constant, as expected.** Nothing feeds them
  settled predictions yet, so they emit their `na` sentinels. They are bound
  and uninformative. This is not a null result about metacognition; it is a
  missing feeder.

## 5. Why the DOWN window failed, in the brain's own output

Predicted mix in the DOWN window: `trough=50, murk=28, slide=23, crest=13,
climb=6`. It called `trough` — the buy signal — 50 times into a window where
14.2% of bars rose. It is long-biased and cannot see the regime it is in,
which is precisely the defect pools 17/18 were designed to remove, and pool 18
is the one the query never fires.

## 6. What is next, in order

1. **Raise pool 18's resolution so it survives the dilution filter.** Carry
   magnitude buckets per scale, not a direction token, and re-measure
   distinctness before training anything. Target the 0.103-0.260 empty band.
2. **Build the resolved-prediction feeder for pools 15/16/19.** They are wired;
   `build_collections` takes an optional `history=` and refuses unresolved
   rows. Nothing records settled predictions in that shape yet. This is the
   real remaining work on the metacognition side.
3. **Re-run the relation arm on top of `6be5357`.** Its null result measured
   nothing.

## 7. Commands that prove each claim

```bash
# mechanism: the query path fires the pool (exits nonzero if it does not)
OMEN_META_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8092 \
python -X utf8 scripts/omen_query_path_probe.py \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --train 600 --test 120 \
  --query-a temporal,geometry,cross \
  --query-b temporal,geometry,cross,temporal_sequence,temporal_scale

# the two windows, one fabric
OMEN_META_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8093 \
python -X utf8 scripts/omen_experiment.py --corpus <corpus> \
  --train 600 --test 120 --train-end 1441 --test-end 1573
OMEN_META_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8093 \
python -X utf8 scripts/omen_experiment.py --corpus <corpus> \
  --train 600 --test 120 --train-end 1441 --test-end 1933 --skip-train

# the wiring invariant, in all four flag combinations
python -X utf8 -m pytest \
  tests/test_metacognition_collections_are_off_until_the_node_has_the_pools.py -q
```

Raw reports: `omen-AERO-USDC-h12-UP-20260910-144800.json`,
`omen-AERO-USDC-h12-DOWN-20260910-144819.json`,
`QUERY-PATH-PROBE-pass109-cove.json`,
`QUERY-PATH-PROBE-meta-pass109-cove.json`.

## 8. Limits of this measurement, stated rather than buried

One symbol, 600 training samples, 120 bars per window. That is small, and it is
small deliberately: the question this pass had budget for was *does the
mechanism work*, and that question is answered. The skill number is a first
reading on a thin sample, not a verdict on the topology — but it is a reading
taken in both directions, and it is below baseline in both.

---

## 9. ADDENDUM, same pass: pool 18 sharpened and re-measured

The frame-resolution fix from §6.1 was made and the two windows re-run on a
fresh fabric (`brain-data-meta-p109-scale`, `:8094`), same corpus, same pinned
train window, same two windows.

`temporal_scale` distinctness **0.045 -> 0.303**, and the measured query set
now selects it on its own merit:
`(geometry, temporal, flow, cross, temporal_sequence, temporal_scale)`.

| | UP before | UP after | DOWN before | DOWN after |
|---|---|---|---|---|
| held-out exact | 25.8% | **30.8%** | 19.2% | 19.2% |
| majority baseline | 26.7% | 26.7% | 65.0% | 65.0% |
| vs baseline | below | **above** | far below | far below |
| buy omens | 15 | 14 | 50 | 50 |
| omens that paid | 53.3% | 71.4% | 6.0% | 12.0% |
| per-trade net | +1.4610% | +3.3406% | -3.7937% | -3.5850% |
| every-bar buy | +0.7293% | +0.7293% | -3.3221% | -3.3221% |

**The verdict does not change: there is no edge.** Firing pool 18 lifted the UP
window above its baseline and left the DOWN window exactly where it was --
19.2% against a 65.0% majority, with 50 buy omens still losing more per trade
(-3.5850%) than buying every bar (-3.3221%).

That asymmetry is the finding. A change that helps only in the up window is the
signature this repo has been fooled by twice, and it is why the two-window rule
exists. The brain still calls `trough` 50 times into a window where 14.2% of
bars rise. Sharpening the regime pool made it a better *long* predictor, not a
regime-aware one.

What it did buy, and it is worth keeping: the mechanical reason pool 18 could
never contribute is gone, so the next attempt starts from a pool that fires.
