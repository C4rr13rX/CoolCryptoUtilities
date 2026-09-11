# Neither L2 scheme works, and the sweep says why: there is nothing to collapse

Gale, pass 111, 2026-09-10. Item `[fa75fa1a]`. No node was used and none should
be. Corpora: `p108_aero_down.json` and `p108_aero_up.json`, 600 samples each,
AERO-USDC h12, relative banding, **both computed in one process** so nothing is
cross-run.

Command:

    python -X utf8 scripts/omen_l2_scheme_probe.py \
      --corpus data/brain_experiments/p108_aero_down.json \
      --corpus data/brain_experiments/p108_aero_up.json \
      --json-out data/brain_experiments/L2-SCHEME-p111-gale.json
    # exits 1

## RESULT 1 — both proposed schemes FAIL at the default step count

Identifier ceiling 0.30. L1 is inside the guard in both windows.

| key | DOWN | UP | verdict |
|---|---|---|---|
| `L1_cooccurrence` | 0.1983 (vocab 119) | 0.1383 (vocab 83) | — |
| `L2_sequence` (current) | 0.8233 (494) | 0.8250 (495) | FAIL |
| `L2_transitions` | 0.7067 (424) | 0.6767 (406) | FAIL |
| `L2_runlength` | 0.9217 (553) | 0.9450 (567) | FAIL |

Transitions improve on the current path (0.82 → 0.71 DOWN) and are still in the
same class. **Run-length at steps=3 is WORSE than the thing it replaces**
(0.9217 vs 0.8233 DOWN, 0.9450 vs 0.8250 UP): the dwell bucket is an extra
symbol per position, so it widens the alphabet faster than dropping repeats
narrows it.

Neither is left unmeasured. Both are reported with their numbers, as the
acceptance criterion asks.

## RESULT 2 — the step sweep is the one that explains it

Same fabricless run, transitions and run-length at steps 1, 2, 3, 4, 6, window
12 bars:

| steps | transitions DOWN / UP | run-length DOWN / UP |
|---|---|---|
| 1 | **0.1983 / 0.1383** | **0.2983 / 0.2200** |
| 2 | 0.6017 / 0.4917 | 0.8267 / 0.7667 |
| 3 | 0.7067 / 0.6767 | 0.9217 / 0.9450 |
| 4 | 0.7400 / 0.7250 | 0.9383 / 0.9650 |
| 6 | 0.7717 / 0.7550 | 0.9633 / 0.9767 |

Read the first row against the second. **The jump from one symbol to two is the
whole failure.** Transitions at steps=1 are numerically identical to L1
(0.1983/0.1383) because with repeats dropped, one symbol IS the current motif —
it is a copy, not an abstraction. Add a second symbol and it lands at
0.60/0.49, twice the ceiling, before any third symbol exists.

**The mechanism, and it is not the scheme.** L1 changes on **73.1% of bars
(DOWN) and 74.0% (UP)**. Dropping repeats can only remove the ~27% that ARE
repeats, so the last two *changed* motifs are essentially the last two bars'
motifs. Any ordered pair drawn from a 119-symbol alphabet that reshuffles
three bars in four is near-unique by construction. **The repeat-collapsing
family cannot work here because there are almost no repeats to collapse.**

## THE VERDICT, and criterion 1 is NOT met — I am not redefining it

The acceptance criterion asks for an L2 at or under 0.30 in both windows.
**No order-carrying configuration of either scheme reaches it.** The only cell
in the whole sweep that clears the ceiling in both corpora is **run-length at
steps=1 — 0.2983 DOWN, 0.2200 UP** — and steps=1 is a single symbol, so it
carries no order at all and fails the third criterion instead. Criteria 2 and 4
are met (both schemes named with their numbers; distinctness measured before
any node run, and the probe exits 1 so no arm can be spent). Criterion 3 is met
as a property of the code — `transition_motif` and `run_length_motif` map
A→B→C and C→B→A to different frames, proved in
`tests/test_an_l2_scheme_must_keep_the_order_it_exists_to_carry.py` — but it
cannot be met *simultaneously* with criterion 1.

## WHAT THIS SAYS TO DO NEXT, and it is upstream, not downstream

The binding constraint is that **L1 is near-i.i.d. bar to bar**. No layer built
on top of it can be both ordered and coarse, because its input names a new
symbol three bars in four. So the next change belongs to L1, not L2:

1. **Make L1 persist.** Hysteresis on the relative bands — a stream must cross
   a band boundary by a margin before its token flips — would drop the change
   rate directly, and the change rate is measurable in one run with no node.
   Every point of change rate removed is what buys L2 its second symbol.
2. **`run_length_motif` at steps=1 is a real L1.5 and it clears the guard**
   (0.2983 / 0.2200, vocab 179 / 132 against L1's 119 / 83). It is *not* L2 and
   must not be labelled one, but "the current motif and how long it has been
   held" is a legitimate frame that L1 does not carry today, and it is the
   cheapest thing on this list to wire.

Do NOT resolve this by reverting to the sign-banded encoder. Under sign banding
the change rate was 43.6% and L2 looked better only because L1 was blind — that
trade was already measured and refused (7d2a74e).

Reproduce every table in this report with one command (847cd50) -- margin 0 is
always included, so the control and its treatments are one measurement:

    python -X utf8 scripts/omen_l2_scheme_probe.py       --corpus data/brain_experiments/p108_aero_down.json       --corpus data/brain_experiments/p108_aero_up.json       --margin 0.5 --steps 2 --steps 3

That run also fills in the run-length column the first sweep never printed:
under hysteresis at margin 0.50 run-length is 0.7117 DOWN and 0.5983 UP against
transitions at 0.2633 and 0.2000. The dwell bucket costs more alphabet than the
stickiness saves, at EVERY margin measured -- so transitions is the scheme, and
run-length is rejected with its number rather than left unmeasured.

---

# ADDENDUM, same pass: the upstream lever works, and criterion 1 is REACHABLE

Hysteresis on the relative bands — a slot holds its band until the score is
pushed `margin × (hi − lo)` past the boundary — measured on both corpora in one
process, 600 samples each, `sticky_motifs` in `scripts/omen_l2_scheme_probe.py`:

| margin | L1 change rate DOWN / UP | L1 distinct (vocab) DOWN / UP | L2_transitions steps=2 | steps=3 |
|---|---|---|---|---|
| 0.00 | 73.1% / 74.0% | 0.1983 (119) / 0.1383 (83) | 0.6017 / 0.4917 FAIL | 0.7067 / 0.6767 |
| 0.25 | 57.9% / 57.4% | 0.1933 (116) / 0.1300 (78) | 0.5083 / 0.4050 FAIL | 0.5817 / 0.5183 |
| **0.50** | **37.6% / 38.2%** | 0.0817 (49) / 0.0683 (41) | **0.2633 / 0.2000 PASS** | 0.3667 / 0.3167 |
| 1.00 | 18.9% / 17.9% | 0.0633 (38) / 0.0300 (18) | 0.1483 / 0.1017 PASS | 0.2167 / 0.1817 |

**An order-carrying L2 under 0.30 in BOTH windows exists: transitions at
steps=2 over an L1 banded with margin 0.50 — 0.2633 DOWN, 0.2000 UP.** It
needed an L1 change, not another L2 scheme, exactly as the change-rate
mechanism predicted.

**The cost, stated because distinctness cannot see it.** L1 coarsens hard:
0.1983 → 0.0817 DOWN (vocabulary 119 → 49) and 0.1383 → 0.0683 UP (83 → 41).
Whether that coarser L1 still carries **label skew** is NOT measured here and
is the gate before any node arm — `omen_layer_probe`'s skew test answers it and
costs no node time. A layer that abstracts perfectly and predicts nothing is
still worthless, and margin 1.00's UP vocabulary of 18 is well into that risk.

**A seam the test caught, and it is why the arm is trustworthy.**
`test_hysteresis_at_zero_margin_reproduces_plain_relative_banding` failed on the
first draft: `relative_bands` omits a stream whose terciles collapse and
`cooccurrence_motif` falls back to absolute sign banding for it, while
`sticky_motifs` was emitting `na`. margin=0 would not have been the same encoder
as the comparison arm, so the whole table would have been a two-change
measurement. Fixed; all five streams carry bands on these corpora so the numbers
above are unaffected, and the test now pins it.

**Next, in order:** (1) label skew on the margin-0.50 L1 in both windows —
node-free, decisive, and it can still kill this; (2) only if skew survives,
one node arm on L0 vs L0+L1+L2 with the operator's scoreboard (trough per-trade
net, trough precision, crest precision, exact accuracy demoted).

---

# ADDENDUM 2, same pass: the skew gate PASSES, and it says something sharper than the distinctness table

`label_skew` (from `scripts/omen_layer_probe`, imported not edited), trough as
the target, `min_support=20`, same 600 samples per corpus, same process:

| corpus | margin | key | groups n≥20 | coverage | max lift |
|---|---|---|---|---|---|
| DOWN | 0.00 | L1 | 6 | 30.8% | 2.87 |
| DOWN | 0.00 | **L2** | **0** | **0.0%** | — |
| DOWN | 0.50 | L1 | 10 | 57.8% | 2.37 |
| DOWN | 0.50 | L2 | 3 | 10.5% | 1.47 |
| UP | 0.00 | L1 | 8 | 43.7% | 3.42 |
| UP | 0.00 | **L2** | **0** | **0.0%** | — |
| UP | 0.50 | L1 | 8 | 68.0% | 2.46 |
| UP | 0.50 | L2 | 5 | 21.9% | 4.21 |

**The row that matters is the one with a zero in it.** Without hysteresis, L2
has *no group at all* reaching 20 samples in either window — not a weak signal,
no measurable signal, because no L2 frame is shared by twenty bars. That is the
identifier problem restated in the only currency that counts, and it is why
every L2 held-out number measured so far was measuring nothing.

**The feared cost did not materialise.** Coarsening L1 was supposed to trade
information for abstraction. It nearly **doubled L1's usable coverage** —
30.8% → 57.8% DOWN and 43.7% → 68.0% UP — for a modest fall in peak lift
(2.87 → 2.37, 3.42 → 2.46). More bars land in a group big enough to say
anything, which is worth more than a taller lift over 30% of the corpus.

**What this is NOT.** These lifts are IN-SAMPLE: the bands are fitted on the
same corpus they are scored on, so they are a green light to spend one node
arm, not an edge. And L2's DOWN side is thin and weak — 3 groups, 10.5%
coverage, lift 1.47 — against a much stronger UP side (5 groups, 21.9%, 4.21).
A layer that works in one window and not the other is the exact failure the
both-windows rule exists to catch, so **L2 must not be promoted on the UP
number.** L1 at margin 0.50 is the part that holds up in both.

**Revised next step:** the arm worth spending is **L0 vs L0+L1(margin 0.50)**,
with bands fitted on train and reused on held-out. L2 rides along to be
measured, not to be claimed.
