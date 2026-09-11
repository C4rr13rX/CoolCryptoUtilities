# Shape mutations: which ones can be taught without moving the label

Pass 112, Gale, 2026-09-10. Item [a2449616].

**Corpus** `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21,926 bars,
AERO-USDC, hourly, horizon 12 bars.
**Windows** two, 120 sampled anchors each, chosen to match the windows the
pass-110 baseline arms trained on:
- DOWN-side training window, bars `[841, 1441)`
- UP-side training window, bars `[21091, 21691)`

**Commands**

```
python scripts/omen_shape_mutations.py census \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --start 841 --stop 1441 --samples 120 \
  --report data/brain_experiments/p112_gale_shape_mutation_census_DOWNtrain.json
python scripts/omen_shape_mutations.py census \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --start 21091 --stop 21691 --samples 120 \
  --report data/brain_experiments/p112_gale_shape_mutation_census_UPtrain.json
python -m pytest tests/test_shape_mutations_preserve_the_label.py -q   # 9 passed
```

Both exit 0. The script exits nonzero if an admitted mutation ever flips a
label or is swallowed by the encoder's bands, so it is an acceptance test
rather than a transcript.

---

## RESULT 1 — the poison question is settled by arithmetic, not by argument

`label_omen` reads exactly three things: the entry close, the close
`horizon` bars later, and where entry sits between the min and max of the last
`RANGE_WINDOW` closes. `RANGE_WINDOW` is **24** and `LOOKBACK_BARS` is **168**.

So a mutation confined to bars `[anchor-168, anchor-24)` **cannot move the
label**. Not "is unlikely to" — cannot. It touches none of the three inputs.
That is the whole admission criterion, and it is derived from the labeller at
run time (`DEEP_EDGE == RANGE_WINDOW`, enforced by a test) rather than
hard-coded, so it moves if the labeller moves.

## RESULT 2 — three mutations admitted, measured on both windows

| mutation | flip | no-op | x-label | slots moved | why the label survives |
|---|---|---|---|---|---|
| `deep_jitter` | 0.0% / 0.0% | 1.7% / 0.8% | 0.0% / 0.0% | 34.3% / 32.1% | deep-only |
| `deep_flatten` | 0.0% / 0.0% | 4.2% / 1.7% | 0.0% / 0.0% | 45.8% / 46.9% | deep-only |
| `deep_dilate` | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 51.2% / 51.1% | deep-only |

(DOWN-side / UP-side.)

All three clear three gates at once, which is what makes them usable:

1. **Label flip 0.0%** — the pair is not poison.
2. **No-op rate 0–4.2%** — the mutation survives the encoder's bands, so the
   pair is a genuinely NEW frame rather than the same pair taught twice. A
   mutation the bands swallow buys nothing and this is where most candidate
   mutations would have died.
3. **Cross-label collision 0.0%** — no mutated frame lands on a key already
   carrying a different label, so none of them lowers the recall ceiling.

`deep_flatten` is the item's "truncated" variant expressed at a fixed frame
length: `build_collections` refuses a short frame on purpose, because a short
frame is a different byte string and padding would quietly create a second atom
for the same situation. `deep_dilate` is "same shape, different tempo".

## RESULT 3 — amplitude scaling is DROPPED, and the number says why

The item asked for amplitude scaling to be justified or dropped. It is dropped,
and the interesting part is that the obvious test **passes**:

- **label flip 0.0%.** Position-in-range is a ratio, so scaling deviations
  about the anchor leaves it invariant. The label does not move.

That is exactly the trap. The label's *first* test is
`abs(forward) >= threshold`, an absolute move against an absolute round-trip
cost — and amplitude is the one quantity that separates `murk` from `trough`.
The census shows which slots it moves, on 120 of 120 samples in both windows:

    amplitude   geometry: 0/120     temporal: 120/120
                volatility: 120/120  cross: 120/120

Geometry never moves (it is built from ratios). Every magnitude slot moves,
every time. Training that pair on an unchanged label teaches the fabric to
ignore the only quantity the cost threshold is read on. **Dropped.**

## RESULT 4 — inversion is DROPPED at 70.0% / 60.0% label flip

Mirroring the window about the anchor turns `at_low` into `at_high`, i.e.
`trough` into `crest`. Keeping the label is a lie. Flipping it instead assumes
this tape is up/down symmetric, which nothing measured here supports. Both
dropped mutations stay implemented and `admitted=False`, so the refusal carries
a number and a test rather than a memory.

---

## RESULT 5 — THE ARM RAN. The gap narrowed in BOTH windows, and it is 3 bars.

The box freed up late in the pass (6,774 MB against the 4,096 MB floor), so the
arms ran after all. Two fresh fabrics, `brain-data-p112-gale-base2` (:8092) and
`brain-data-p112-gale-mut2` (:8093), each censused at `neurons=0` before a byte
was taught. Both windows read off ONE fabric per arm via `--skip-train`.

Train `[1141, 1441)`, 300 base samples. Held-out UP `[1453, 1573)`, DOWN
`[1813, 1933)` — the same windows the pass-110 baselines used.

| | base (300 pairs) | base + deep_jitter + deep_dilate (900 pairs) | move |
|---|---|---|---|
| train_recall | 1.0000 | 1.0000 | 0 |
| **UP** held-out exact | 0.2083 | **0.2333** | +0.0250 |
| UP majority baseline | 0.2667 | 0.2667 | — |
| **UP gap** | +0.7917 | **+0.7667** | **−0.0250** |
| **DOWN** held-out exact | 0.1500 | **0.1833** | +0.0333 |
| DOWN majority baseline | 0.6500 | 0.6500 | — |
| **DOWN gap** | +0.8500 | **+0.8167** | **−0.0333** |

**The gap narrowed in both windows, and it narrowed the right way** — recall
sat pinned at 1.0000 in both arms, so the whole move came from held-out rising,
not from recall falling. That is the direction the item asks for, and it is
consistent across an UP window and a DOWN window.

**And it is not an edge, and it is not powered.** Two things must be said
before anyone builds on this:

1. **Both arms are BELOW their majority baselines in both windows.** 0.2333
   against 0.2667, and 0.1833 against 0.6500. Narrowing the memorisation gap
   did not produce a brain that beats calling the majority class. An
   at-or-below-baseline result is the normal outcome here and this is one.
2. **+0.0250 on 120 bars is 3 bars. +0.0333 is 4 bars.** By the same power
   arithmetic the operator applied to the self-pool verdict at 20:03, a move
   this size on this n licenses no verdict. It is not evidence that mutations
   help; it is a direction worth powering. The money lines are worse still —
   5 and 7 buy omens in the UP cells, which the arm itself refuses to let stand
   (`buy_power_sufficient: false`).

### The negative control: it is NOT pair volume

The mutated arm differed from base in two ways, not one — the mutations AND the
pair count (300 → 900). So a third fabric (`brain-data-p112-gale-rep3`, :8094,
`neurons=0`) was trained on the SAME 300 base pairs taught three times:
identical pair count, zero new information.

| held-out exact | base (300) | **repeat ×3 (900)** | base+2 mutations (900) |
|---|---|---|---|
| UP (majority 0.2667) | 0.2083 | **0.2167** | 0.2333 |
| DOWN (majority 0.6500) | 0.1500 | **0.1500** | 0.1833 |

Repetition bought **zero bars in DOWN and one bar in UP**. The mutated arm beat
the repetition control in both windows. So whatever the +0.0250 / +0.0333 is,
it is not explained by training on three times as many pairs — which was the
obvious alternative explanation and is now closed. It is still 3 and 4 bars,
and both arms are still below their majority baselines; the control makes the
direction believable, not the size.

The one number that is not fragile: **`poisoned_dropped` is 0 across all four
cells.** 600 mutated pairs were generated and not one moved its label, which
is the arithmetic from RESULT 1 confirmed on live training data.

Worth a note for whoever powers this: **crest precision was 1.0000 on 9 and 10
sell omens in the DOWN window** in both arms. Tiny n, but it is the sell-high
half that the long-only scoreboard measures nowhere, and it is the second pass
running in which crest has looked stronger than trough.

**Commands**

```
arm --train 300 --test 120 --train-end 1441 --test-end 1573 --endpoint :8092
arm --train 300 --test 120 --train-end 1441 --test-end 1933 --endpoint :8092 --skip-train
arm --train 300 --test 120 --train-end 1441 --test-end 1573 --mutate deep_jitter --mutate deep_dilate --endpoint :8093
arm --train 300 --test 120 --train-end 1441 --test-end 1933 --mutate deep_jitter --mutate deep_dilate --endpoint :8093 --skip-train
```

Reports: `p112_gale_arm_{base,mut}_{UP,DOWN}.json`.

## THE BLOCKER THAT COST THE FIRST HALF OF THE PASS

**The held-out arm did not run. The box is ~2,000 MB short of the node's
consolidation floor.** This is the blocker, stated as a number:

    fresh node, :8092, clean (total_neurons 0), /health OK
    backpressure_probe -> backpressure=True, available_mb 3879, floor_mb 4096

I killed the pass-111 node on :8091 (2,278 MB) expecting that to clear it.
**It did not.** `available_mb` then read 3708, then 2044 — it fell by 1,835 MB
while I freed 2,278. So `available_mb` is not tracking node memory; the box is
oversubscribed by the other three agents' censuses running concurrently, and
freeing nodes does not touch it. That was my call and it was wrong; both of my
nodes (:8092, :8093) are killed so the memory goes back to the running work.

A run started under backpressure binds nothing and takes up to 120 s per
sample — a slow no-op leaving a half-trained directory for the next pass to
discard. So it was not started.

**The arm itself is written, sized and proven short of the node.** `--dry-run`
on the real corpus:

```
python scripts/omen_shape_mutations.py arm \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --train 600 --test 120 --train-end 1441 --test-end 1573 \
  --mutate deep_jitter --mutate deep_dilate --dry-run
-> 600 base samples -> 1800 pairs (2 mutations, 0 poisoned dropped)
-> test [1453, 1573), window is UP (up-rate 56.7%, mean forward +1.3793%)
```

Its headline is `recall_generalisation_gap = train_recall - heldout_exact`,
not recall, per the item. Recall is scored over BASE pairs only — reproducing a
mutation is not what we want and scoring it would inflate the one number the
item forbids leading on. Crest precision is scored against forward returns so
the sell-high half is measured at all. Following the operator's 20:03 note on
power, the arm now refuses to let its money line stand alone: below 28 buy
omens it prints that the per-trade net **licenses no verdict** and that the
powered comparison is the held-out accuracy over the full test window.

## NEXT

Run the two arms back-to-back when the box has 4,096 MB:

```
# fresh dir A
arm --train-end 1441 --test-end 1573                         # base, UP
arm --train-end 1441 --test-end <DOWN> --skip-train          # base, DOWN
# fresh dir B
arm --mutate deep_jitter --mutate deep_dilate --train-end 1441 --test-end 1573
arm --mutate deep_jitter --mutate deep_dilate --train-end 1441 --test-end <DOWN> --skip-train
```

Expected cost: 600 pairs and 1,800 pairs. At the 5.6 pairs/s baseline that is
1.8 min and 5.4 min of training. `deep_flatten` is left out of the first arm
deliberately — one change at a time, and it is the mutation with the highest
no-op rate of the three.
