# The shape-relation pool: measured, negative, and the reason is the dilution law

Pass 120, Jet. Item [5c3b2189]. **The result is below baseline in both windows
and is reported as that, not as progress.**

## 0. The item's named mechanism does not exist

[5c3b2189] asked for `PoolKind::Internal` "carrying the base-frame-to-mutation
relation as a first-class bindable thing", citing `identity.rs` and
`pool.rs:683`. Re-verified this pass against `D:/Projects/W1z4rDV1510n`:

```
grep -rn "PoolKind::Internal" crates/        -> 0 matches
grep -rn "matches!(.*\.kind" crates/ | grep PoolKind
  -> crates/brain/src/brain.rs:7425:  matches!(ps.kind, PoolKind::Action)
```

The variant is declared and matched nowhere; the only behavioural match on a
pool kind anywhere in the engine is `Action`. Cove recorded this in
`docs/BRAIN_POOL_TOPOLOGY.md` in pass 106 and it is still true. Declaring a
pool `Internal` is a naming convention. **A relation becomes bindable by being
SENT, not by being declared**, so the arm below sends a client-computed
relation into a `SensoryInput` pool.

No new identity file and no new node build were needed: pool 7
(`news_entities`) is declared in the shipped `market_predictor_v2.identity.toml`,
is fully knobbed (`recent_atoms_window` 65536, `max_concept_member_count` 24,
`decay_rate` 0.00002, `prune_floor` 0.001), and has never been fed by any
client. This spends it.

## 1. The corpus, the windows, the fabrics

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json`, symbol AERO-USDC |
| horizon | 12 bars, resolved by the node as `720 min = 12 bars of 3600s` |
| train | bars [2300, 2600), 298 base samples -> 596 pairs with `deep_jitter`, 0 poisoned dropped |
| UP window | bars [3568, 3968), 400 samples, up-rate 63.5%, mean forward +2.2452% |
| DOWN window | bars [3168, 3568), 400 samples, up-rate 32.0%, mean forward -1.4398% |
| fabrics | `:8092` (no shape pool) and `:8093` (shape pool), both censused `neurons=0 concepts=0` before training |

Both windows on each arm are read off the SAME fabric via `--skip-train`, which
is what makes them comparable given this node's run-to-run variance.

## 2. The key was gated node-free before any node time was spent

A relation key has to do two opposite things: COLLIDE across a base frame and
its label-safe mutations (or it buys nothing), and SEPARATE genuinely different
shapes (or it is a constant). `omen_shape_mutations.py shapekey` measures both,
over 400 anchors:

| resolution | distinct keys | distinctness | keys with support | anchors in a supported key | collides under `deep_jitter` | under `deep_dilate` |
|---|---|---|---|---|---|---|
| k8 (8 points, 5 bands) | 307 | **76.8%** | 64 | 39.2% | 73.0% | 0.2% |
| k4 (4 points, 3 bands) | 44 | **11.0%** | 42 | **99.5%** | **92.8%** | 6.8% |

Two things were decided here rather than in the fabric:

* **The fine key is the pass-117 failure repeating.** At 76.8% distinctness it
  is still unique-per-instant for three anchors in four. A key nothing else
  shares teaches nothing. `OMEN_SHAPE_RESOLUTION` defaults to `k4` for this
  reason and the census is quoted in the code comment.
* **`deep_dilate` is not a "same shape" mutation and cannot be taught as one.**
  It collides at 0.2%/6.8%. Reading `mut_deep_dilate` explains it: read
  positions clamp at 0, so the head of the prefix becomes a flat run and the
  result is a CROP of the tail, not the same shape at a different tempo. No
  full-prefix canonical key can hold a crop on its base's key. The arm below
  therefore mutates with `deep_jitter` only.

## 3. The measured result

Majority-class baselines: **UP 0.4225, DOWN 0.4300**.

| arm | UP held-out exact | DOWN held-out exact | train_recall |
|---|---|---|---|
| base+jitter, no shape pool | 0.3550 | 0.2625 | 1.0000 |
| shape pool trained, NOT in the query | 0.3550 | 0.2625 | 1.0000 |
| shape pool trained AND in the query | **0.2175** | **0.2050** | 1.0000 |

Recall-generalisation gap: +0.6450 / +0.7375 (baseline) -> +0.7825 / +0.7950
(queried). The gap WIDENED, and `train_recall` is 1.0000 in every cell — a
recall rise with a falling held-out is the failure this item defined, and that
is what happened.

**Every arm is below its majority baseline in both windows.** The shape pool
did not rescue the shape-mutation arm; queried, it cost **-0.1375 (UP)** and
**-0.0575 (DOWN)**.

## 4. Two findings worth more than the arm

**(a) A new pool is trained but NOT queried by default, and the failure is
silent.** The middle row above is byte-identical to the baseline on all four
metrics across 400 test samples. The frame is built (`shp k4=rqpp`) and
streamed to pool 7 — `build_collections` returns 8 collections — but
`OmenBrain.predict` streams `PREDICT_COLLECTIONS`, a fixed
`("temporal", "geometry", "cross")`, so the new pool never reaches the decode.
Anyone adding a pool and reading an unchanged number will conclude the pool did
nothing, when in fact it was never asked. `RELATION_COLLECTIONS` (12/13/14) and
`META_COLLECTIONS` (15-19) have exactly this exposure. Turning it on needs no
code change: `OMEN_PREDICT_COLLECTIONS=temporal,geometry,cross,shape_class`.

**(b) The support requirement and the dilution law are in direct conflict for
this key, and no resolution of it satisfies both.** The dilution law's empty
band is 0.103 to 0.260 distinctness. k4 sits at **0.110 — inside the band** —
and that is precisely why it has support. k8 clears the band at 0.768 and has
almost no support. The -0.1375 UP drop is what the dilution law predicts for
querying a stream in the empty band, and it is the second independent
confirmation of that law rather than a surprise. A shape key that is both
supported and sharp would need a different construction, not a different
band count.

## 5. What this does NOT claim

These are 400-bar windows on ONE symbol. The arm is not evidence about other
symbols, and no live knob was touched: `OMEN_SHAPE_COLLECTION` defaults to `0`
and production's frames are unchanged.

## 6. Commands that reproduce it

```
python -X utf8 scripts/omen_shape_mutations.py shapekey \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --horizon 12 \
  --start 841 --stop 3968 --samples 400 \
  --mutate deep_jitter --mutate deep_dilate

# baseline arm, fresh fabric on :8092
python -X utf8 scripts/omen_shape_mutations.py arm \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --horizon 12 \
  --train 300 --test 400 --train-end 2600 --test-end 3968 \
  --mutate deep_jitter --endpoint http://127.0.0.1:8092
# ... and --test-end 3568 --skip-train for the DOWN window

# shape arm, fresh fabric on :8093, pool in the QUERY
OMEN_SHAPE_COLLECTION=1 OMEN_SHAPE_RESOLUTION=k4 \
OMEN_PREDICT_COLLECTIONS=temporal,geometry,cross,shape_class \
python -X utf8 scripts/omen_shape_mutations.py arm ... --endpoint http://127.0.0.1:8093
```
