# Powering the shape-mutation arm: 400 held-out bars in an UP and a DOWN window

Pass 117, Cove, 2026-09-11. Item [f4c0975a].

## Why this run exists

Pass 112 measured base against base+`deep_jitter`+`deep_dilate` and read
UP held-out 0.2083 -> 0.2333 and DOWN 0.1500 -> 0.1833. Both moves were
measured on **120 test bars**, so +0.0250 is THREE BARS and +0.0333 is FOUR.
A move that size on that n licenses no verdict. This run re-asks the same
question with the test window at **400 bars in both directions**, where a
2.5pp move is ten bars.

## Setup

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21,926 hourly bars, AERO-USDC |
| horizon | 12 bars, passed explicitly (`--horizon 12`); the node resolved it as `720 min = 12 bars of 3600s` |
| train window | `[2300, 2600)`, 298 base samples |
| held-out UP | `[3568, 3968)`, **400 bars**, up-rate 63.5%, mean forward +2.2452% |
| held-out DOWN | `[3168, 3568)`, **400 bars**, up-rate 32.0%, mean forward -1.4398% |
| disjointness | the earliest test anchor reads back 168 bars to 3000, which is 400 bars clear of the training window; no test frame contains a trained bar |
| base fabric | fresh node `127.0.0.1:8092`, brain dir `brain-data-p117b-cove-base`, `pool_count 12`, censused `neurons=0 concepts=0 tick=0` before a byte was taught |
| mutated fabric | fresh node `127.0.0.1:8093`, brain dir `brain-data-p117b-cove-mut`, `pool_count 12`, censused `neurons=0 concepts=0 tick=0` before a byte was taught |

Both windows were read off ONE fabric per arm: the UP run trains, the DOWN run
re-measures the same fabric with `--skip-train`. The two arms were trained
back-to-back, in this pass, on the two fresh fabrics above.

The windows were chosen node-free, by ranking every 400-bar window in the
corpus that clears the training window by mean forward return at h=12. The
chosen pair is adjacent -- the DOWN window ends where the UP window begins --
so the same era supplies both directions.

## RESULT — POWERED, THE PASS-112 DIRECTION REVERSES. Mutations COST 18 bars in each window.

| | base (298 pairs) | base + `deep_jitter` + `deep_dilate` (894 pairs) | move |
|---|---|---|---|
| train_recall | 1.0000 | 0.9900 | −0.0100 |
| **UP** held-out exact | 0.3400 | **0.2950** | **−0.0450 (18 bars)** |
| UP majority baseline | 0.4225 | 0.4225 | — |
| **UP gap** | +0.6600 | **+0.6950** | **+0.0350 — WIDER** |
| **DOWN** held-out exact | 0.2825 | **0.2375** | **−0.0450 (18 bars)** |
| DOWN majority baseline | 0.4300 | 0.4300 | — |
| **DOWN gap** | +0.7175 | **+0.7525** | **+0.0350 — WIDER** |
| admitted / total | 400 / 400 | 400 / 400 | — |
| `poisoned_dropped` | 0 | 0 | — |

**Pass 112's result does not survive being powered, and it does not merely
fail to replicate — it inverts.** At 120 test bars the mutated arm read
+0.0250 (UP) and +0.0333 (DOWN) ABOVE base, which was three and four bars.
At 400 test bars it reads **−0.0450 in both windows, which is eighteen bars
in each**, and it is the same sign and the same magnitude in an UP window and
a DOWN window. The item's headline number, the recall-generalisation gap,
moved the WRONG WAY in both: +0.6600 -> +0.6950 and +0.7175 -> +0.7525.

The 0.0100 of recall the mutated arm gave up is one answered sample in a
hundred, and it does not explain a 0.0450 held-out fall; the gap widened
because held-out FELL, not because recall rose.

**Both arms are below their majority baselines in both windows.** 0.3400 and
0.2950 against 0.4225; 0.2825 and 0.2375 against 0.4300. Calling the majority
class beats every cell here, and the mutated arm is the furthest from it. That
is the result, and it is not progress of any kind.

`poisoned_dropped` is **0** in all four cells: 596 mutated pairs were generated
and not one moved its label, so the RESULT-1 arithmetic from pass 112 holds at
this size too. The mutations are clean. They are simply not useful — at this
training size, on this corpus, `deep_jitter` and `deep_dilate` make the fabric
generalise **worse**.

### The money lines, which agree

| | base UP | mut UP | base DOWN | mut DOWN |
|---|---|---|---|---|
| buy omens | 45 | 31 | 103 | 119 |
| trough precision | 0.3556 | 0.3548 | 0.1165 | 0.1345 |
| net per trade | +1.8984% | +1.7834% | −3.0918% | −2.7617% |
| every-bar net per trade | +1.5952% | +1.5952% | −2.0898% | −2.0898% |

In the UP window both arms beat every-bar buying by about 0.2-0.3pp per trade,
which is what a long-only rule does in an up window and is not evidence of
anything. In the DOWN window both arms are WORSE than buying every bar
(−3.09% and −2.76% against −2.09%), i.e. the buy omen is actively selecting
the worse bars. The mutated arm is nearer every-bar in DOWN only because it
fires more often.

Crest precision, which looked strong at tiny n in pass 112 (1.0000 on 9 and 10
sell omens), reads **0.5366 / 0.4528 in UP and 0.5417 / 0.4815 in DOWN** on
24-53 omens. The pass-112 crest number was small-n noise and this retires it.

## What this closes

The item asked for a verdict at power, and there is one: **`deep_jitter` +
`deep_dilate` at 298 base samples do not narrow the recall-generalisation
gap. They widen it by 0.0350 in both directions.** No further arm of this
shape is worth node time without changing something other than the sample
count — the mutation set, the training size, or the substrate's ability to
use a shape at all.

## What was cut, and why it is named rather than hidden

Two things in the item's description were not delivered:

1. **Train 2000 was not reached; the arms ran at train 298.** The item sized a
   powered arm off "the node measured 24 pairs/s". That number is wrong by
   about 5x in the regime that matters. Measured here, on a FRESH fabric the
   node trains at 18.9 pairs/s, and **the rate decays as the fabric fills**:
   the 894-pair mutated arm read 17.7, then 12.6, then 9.9, then 8.2 pairs/s
   at the 200/400/600/800 marks. The first attempt at train 2000 (2,000 vs
   6,000 vs 8,000 pairs on three nodes) was measured at ~4 pairs/s aggregate
   and projected to 18-25 minutes of training for the mutated arm alone. It
   was killed rather than left half-trained, and the arms were re-run at the
   size that fits, **with the test window held at 400 bars in both directions
   — the power this item is actually about.**
2. **`deep_flatten` was not measured.** The three-mutation arm (8,000 pairs)
   was launched and killed at the same time and for the same reason. It is
   not reported at all rather than reported half-trained.

## Commands

```
python "<scratchpad>/window_census.py" 2600 2000      # window choice, node-free

python -X utf8 scripts/omen_shape_mutations.py arm \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --horizon 12 \
  --train 300 --test 400 --train-end 2600 --test-end 3968 \
  --endpoint http://127.0.0.1:8092 \
  --report data/brain_experiments/p117_cove_arm_base_UP.json
python -X utf8 scripts/omen_shape_mutations.py arm ... --test-end 3568 \
  --endpoint http://127.0.0.1:8092 --skip-train \
  --report data/brain_experiments/p117_cove_arm_base_DOWN.json
python -X utf8 scripts/omen_shape_mutations.py arm ... --test-end 3968 \
  --mutate deep_jitter --mutate deep_dilate --endpoint http://127.0.0.1:8093 \
  --report data/brain_experiments/p117_cove_arm_mut2_UP.json
python -X utf8 scripts/omen_shape_mutations.py arm ... --test-end 3568 \
  --mutate deep_jitter --mutate deep_dilate --endpoint http://127.0.0.1:8093 \
  --skip-train --report data/brain_experiments/p117_cove_arm_mut2_DOWN.json
```

## THE TRAP THAT COST THE FIRST RUN, AND IS NOW FIXED

`--endpoint 127.0.0.1:8092` did **not** mean `:8092`. `urlparse` on a
scheme-less string parses the whole thing as a PATH — `hostname` is None and
`port` is None — so `trading/omen_brain.py` and `trading/brain_bridge.py` both
fell back to their own default port and connected there in silence. The arm
printed `FAIL: node has no /brain/predict/multi -- stale binary or port` while
`:8092` answered that exact route with a 200; it was talking to `:8091`, where
nothing was listening. Had a node been up on `:8091` it would have **trained
that one instead** and said nothing — and the fabric census resolves the same
way, so the report would have been self-consistent and about the wrong brain.
On `brain_bridge` the default is `8090`, which is **production**.

Fixed with one shared resolver, `trading.brain_bridge.resolve_node_endpoint`,
which honours a named port in any spelling (`127.0.0.1:8092`, `:8092`, `8092`,
`http://...`) and falls back only when no port is named. Test:
`tests/test_a_scheme_less_endpoint_is_not_silently_another_node.py`, 8 passed.
The old two lines resolve `127.0.0.1:8092` to port 8091, so it fails against
the old behaviour.

## NEXT

The direction is closed at this size; the open question is whether the
substrate can use a shape AT ALL, which pair-volume arms cannot answer. Two
candidates, in order:
1. Run the same two arms at train 2000 **sized honestly** — 2,000 base against
   6,000 mutated is ~25 minutes of node time at the decaying rate measured
   here, so it needs a pass of its own with nothing else in it.
2. Stop adding pairs and change the RELATION: an `Internal` pool carrying
   "same shape as" between the base frame and its mutation, rather than
   teaching both frames flat into the same sensory pools. Eleven flat
   SensoryInput pools cannot represent "these two are the same shape", which
   is the thing the mutations are trying to teach.

