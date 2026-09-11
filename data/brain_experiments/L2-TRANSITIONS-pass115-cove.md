# L2 is possible after all: TRANSITIONS over a sticky L1, 0.2633 DOWN / 0.2000 UP

**Pass 115, Cove, 2026-09-11. Item [fa75fa1a]. NO NODE WAS CONTACTED.**
Production :8090 and :8091 untouched. Every number here is node-free encoder
arithmetic, which is the whole point: the guard is decided before an arm is
spent, not after.

## Corpus and exact configuration

| | |
|---|---|
| corpora | `data/brain_experiments/p108_aero_down.json`, `p108_aero_up.json` |
| pair | AERO-USDC on base, one pair, horizon 12 bars |
| samples | 600 per corpus, both corpora and every cell in ONE process |
| L1 streams | geometry, temporal, flow, volatility, cross (5 of 5 live) |
| L2 window | 12 bars |
| identifier ceiling | 0.30 |
| support floor | n >= 20, same as `omen_layer_probe.label_skew` |

Reproduce with:

```
python -X utf8 scripts/omen_l2_scheme_probe.py \
    --corpus data/brain_experiments/p108_aero_down.json \
    --corpus data/brain_experiments/p108_aero_up.json
```

Exit 0, and it exits 1 if no scheme clears the ceiling at the shipped banding.

## RESULT 1 -- the criterion, and it PASSES in both windows

`L2_transitions` at `L2_TRANSITION_STEPS = 2` over a sticky L1 at
`L1_HYSTERESIS_MARGIN = 0.50`:

| corpus | distinctness | ceiling | verdict |
|---|---|---|---|
| p108_aero_down | **0.2633** | 0.30 | PASS |
| p108_aero_up | **0.2000** | 0.30 | PASS |

This reproduces Jet's pass-113 unblock number to four decimal places, in one
process, which is why the pass-111 block is formally dead rather than merely
outvoted.

## RESULT 2 -- the rejected scheme, with ITS number rather than a dismissal

`L2_runlength` (motif plus a bucketed dwell) at the identical setting:

| corpus | run-length | transitions |
|---|---|---|
| p108_aero_down | **0.7117** FAIL | 0.2633 PASS |
| p108_aero_up | **0.5983** FAIL | 0.2000 PASS |

It fails at every margin and step count swept (margins 0.00 / 0.50 / 0.75 x
steps 2 / 3 / 4; the best cell it ever reaches is 0.5633). The mechanism is
mechanical and not a tuning accident: the dwell bucket is an extra symbol at
every position, so it WIDENS the alphabet in a layer whose entire problem is
that its alphabet is too wide. Rejected on its own number.

The old fixed-length path `L2_sequence` is also reported and also fails:
0.5317 DOWN / 0.4083 UP even at the shipped banding, against 0.8233 / 0.8250
under plain banding. `MOTIF_SEQUENCE_STEPS = 3` is dead as an L2 value and the
constant now says so in its own comment.

## RESULT 3 -- step count, measured rather than inherited

Transitions over the sticky L1, worst of the two corpora:

| steps | DOWN | UP | verdict |
|---|---|---|---|
| 2 | 0.2633 | 0.2000 | **PASSES BOTH** |
| 3 | 0.3667 | 0.3167 | fails both |
| 4 | 0.4133 | 0.3817 | fails both |

2 clears by 0.037 in the worse window and 3 misses by 0.067, so this is not
the 0.8% margin the stale `MOTIF_SEQUENCE_STEPS` sweep had. Do not raise it to
carry "more history": a transition path already spans as many BARS as its
window allows, because a regime holding for forty bars costs it one symbol.
Length of memory is the window, not the step count.

## RESULT 4 -- the finding that is worth more than the guard

The guard can be cleared the wrong way, by collapsing toward a constant. So the
probe now reports supported groups (n >= 20) beside distinctness, and this is
where the pass actually moved something. Jet recorded that under plain relative
banding L2 has NO label-skew group reaching n=20 in either window -- meaning
every L2 number measured to date measured nothing. **Reproduced exactly**, and
then removed:

| | plain banding (margin 0.00) | sticky L1 (margin 0.50) |
|---|---|---|
| DOWN, supported groups | **0**, covering 0.0% | **3**, covering 10.5% |
| UP, supported groups | **0**, covering 0.0% | **5**, covering 21.9% |

L2 is measurable for the first time. The UP groups, with the corpus trough
rate at 14.6%:

| n | trough rate | lift | top label | purity | frame |
|---|---|---|---|---|---|
| 26 | 61.5% | **4.21x** | trough | 62% | `co2t path=lllhm\|llhhm` |
| 21 | 28.6% | 1.95x | slide | 33% | `co2t path=llhhm\|lllhm` |
| 28 | 17.9% | 1.22x | slide | 36% | `co2t path=llhhm\|llhlm` |
| 22 | 0.0% | 0.00x | crest | 50% | `co2t path=hhlhh\|hhhhh` |
| 32 | 0.0% | 0.00x | climb | 59% | `co2t path=mmhlm\|mmllm` |

That 4.21x is above L1's own best in the same window (2.46x), which is the
first time any layer above L1 has beaten it on this corpus.

**The first two rows are the same two symbols in opposite order, and they are
different situations: 61.5% trough against 28.6%.** That is criterion 3
satisfied on the real corpus rather than only on a fixture -- order survived
the repeat-dropping and it carries outcome information, which is the entire
reason the layer exists.

### What this is NOT, said plainly

This is an IN-SAMPLE support and skew measurement over one pair and one
600-bar window per direction. It is not a held-out edge and must not be quoted
as one. n=26 and n=21 are small; by the standard the operator applied to
[5ec44914] these cells are underpowered for a per-trade return claim, and no
such claim is made here. DOWN is materially weaker than UP -- its best
supported group is 1.47x and one group sits at 0.00x -- so the honest read is
that the L2 signal at this support floor is an UP-window observation with a
DOWN-window that has not yet shown one.

What it licenses is exactly one thing: **a node arm on L2_transitions is now
worth spending, and it was not before.** That is what criterion 4 asks.

## THE DEFECT THIS PASS FIXED, which is why the item was blocked

`scripts/omen_l2_scheme_probe.py` built its table with per-bar
`cooccurrence_motif` -- plain relative banding, margin 0 -- and its EXIT CODE
read that table. The live path ships a sticky L1. So the probe printed
"NEITHER scheme clears 0.30, do NOT spend a node arm" about an encoder the
system does not use, pass 111 recorded that as "no order-carrying scheme can
meet criterion 1", and a true statement about the wrong configuration blocked
the item for four passes. Every number in that block was correct.

The gate now judges `--gate-margin`, defaulting to the shipped
`L1_HYSTERESIS_MARGIN`, and always prints margin 0 beside it as the control.

Pinned by `tests/test_the_l2_gate_must_judge_the_banding_the_live_path_ships.py`
(7 tests). The two mechanism tests were PROVEN to fail against the defect by
stubbing `sticky_motifs` to ignore its margin, rather than merely observed to
pass.

## Code that moved

* `trading/omen_layers.py` -- `transition_motif` promoted out of the probe
  (the live path cannot import a probe), `L2_TRANSITION_STEPS = 2` and
  `L1_HYSTERESIS_MARGIN = 0.5` with their measured tables in the comments,
  `IDENTIFIER_CEILING` stated once so no probe can disagree about what passing
  means, and `_compact_motif` shared so every scheme is compared over an
  identical alphabet.
* `scripts/omen_l2_scheme_probe.py` -- gate reads the shipped banding; control
  and treatment both printed; supported-group counts and best lift in the same
  table as distinctness; `build_parser` split out so the default that decides
  the exit code is testable without a corpus.

## NEXT, and it is a stale-number risk

`scripts/omen_layer_probe.py` still computes `L2_sequence` -- the REJECTED
scheme -- as its L2, at `MOTIF_SEQUENCE_STEPS`. Its L2 column is therefore
measuring the dead layer. It is not a false number (the scheme genuinely
fails), but any held-out L2 arm run through that probe today would score the
loser. Wiring `transition_motif` in there is the next change, and it must be
done before any L2 held-out arm is attempted.
