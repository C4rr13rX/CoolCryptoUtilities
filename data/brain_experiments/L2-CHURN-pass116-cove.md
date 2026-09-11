# L2 carries dwell in ONE symbol, not one per position — [fa75fa1a] criterion 3

**Pass 116, Cove. Node-free. No node was contacted; :8090 and :8091 untouched.**

## What was wrong

Jet reopened [fa75fa1a] on criterion 3 alone and the reopen was correct. The
pass-115 encoder's L2 frame was a change-order path and nothing else, and at
the **shipped** `L2_TRANSITION_STEPS = 2` that frame cannot say how many times
the alphabet changed. Two sequences with the identical multiset —

    persistent   A A A A B B B B     one change
    alternating  A B A B A B A B     seven changes

— collapse to the same two kept symbols and produced **one identical frame**,
`co2t path=hlmhl|lhlmh`. Dwell and alternation were the same situation by
construction. The test suite could not see it because every call in
`tests/test_an_l2_scheme_must_keep_the_order_it_exists_to_carry.py` passed
`steps=3` explicitly and never the module default, and one test actively
**pinned the discarding of dwell as correct**
(`transition_motif([A]*8+[B]) == transition_motif([A,B])`).

## The fix, and why it is not run-length

Run-length is the obvious way to carry dwell and it was already measured and
rejected: it attaches a bucket to **every kept symbol**, so it multiplies the
alphabet once per position and reads **0.7117 DOWN / 0.5983 UP** against the
0.30 ceiling — worse than the fixed path it was meant to replace.

`churn_band` attaches **one** bucket to the whole frame: the window's
`changes / adjacencies`, banded. The alphabet then grows by a bounded factor of
at most `len(cuts)+1` however many symbols the path keeps, and in practice by
far less because churn and path are correlated. A rate rather than a count, so
the symbol means the same thing at any window length.

    co2t path=hlmhl|lhlmh chn=lo     persistent
    co2t path=hlmhl|lhlmh chn=mid    alternating

## Corpus, windows, method

`data/brain_experiments/p108_aero_down.json` and `p108_aero_up.json`,
AERO-USDC on base, horizon 12, **600 samples each**, window 12,
`L2_TRANSITION_STEPS = 2`, sticky L1 at `L1_HYSTERESIS_MARGIN = 0.50`.
Both corpora, both bandings and **every churn candidate computed in ONE
process**, all candidates priced off the same build of each corpus. Relative
bands are fitted on each corpus's own frames; margin 0.00 is carried as the
control row.

    python -X utf8 scripts/omen_l2_scheme_probe.py \
        --corpus data/brain_experiments/p108_aero_down.json \
        --corpus data/brain_experiments/p108_aero_up.json

## Result 1 — what dwell costs (the cut-point sweep)

Worst of the two windows, path + churn, shipped banding, ceiling 0.30:

    cuts            worst distinct   buckets   verdict
    (none)          0.2633           1         the pass-115 frame, NO dwell
    (0.15,)         0.2800           2         PASSES  <- shipped
    (0.20,)         0.2967           2         passes by 0.0033, too thin
    (0.30,)         0.3183           2         fails
    (0.40,)         0.3300           2         fails
    (0.50,)         0.3350           2         fails
    (0.20, 0.55)    0.3433           3         fails
    (0.25, 0.55)    0.3433           3         fails

**Dwell costs 0.0167 of distinctness and there is room for exactly one cut.**
Three buckets is too dear in the DOWN window. `L2_CHURN_CUTS = (0.15,)` is
shipped: at window 12 it says *held* when at most one change occurred across
all eleven adjacencies, which is the property criterion 3 asks for, and it is
the cheapest banding that has it.

## Result 2 — criterion 1 still holds, per corpus

Shipped banding, margin 0.50, window 12, steps 2:

    corpus              L1       L2_seq   L2_path   L2_SHIPPED   L2_runlen
    p108_aero_down     0.0817   0.5317   0.2633    0.2800 PASS   0.7117 FAIL
    p108_aero_up       0.0683   0.4083   0.2000    0.2267 PASS   0.5983 FAIL

`L2_path` is the pass-115 frame with the churn symbol stripped, derived from
the shipped frame rather than reimplemented so the two cannot drift apart.

Control, plain relative banding (margin 0.00): the shipped scheme reads
0.6533 DOWN / 0.5467 UP and fails both, which is the same story as every other
scheme — **the sticky L1 underneath is what makes any L2 possible**, not the
L2 scheme.

## Result 3 — criterion 3, at the shipped default

    python -X utf8 -c "from trading.omen_layers import transition_motif as t; \
      A='co1 geo=hi tmp=lo flo=mid vol=hi crs=lo'; \
      B='co1 geo=lo tmp=hi flo=lo vol=mid crs=hi'; \
      print(t([A]*4+[B]*4)); print(t([A,B]*4))"

    co2t path=hlmhl|lhlmh chn=lo
    co2t path=hlmhl|lhlmh chn=mid

Different frames, same multiset, **no `steps=` argument** — the shipped
default. Pinned by `test_a_held_regime_and_an_alternating_one_are_different_frames`,
which fails against the old encoder.

## Result 4 — WHAT IT COST IN SUPPORT, and this is the uncomfortable half

The churn symbol splits groups as well as separating regimes. Supported groups
are `n >= 20` label-skew groups:

    corpus         frame          groups   covered   best lift
    p108_aero_down L2_path             3     10.5%     1.47x
    p108_aero_down L2_SHIPPED          1      3.4%     (below support)
    p108_aero_up   L2_path             5     21.9%     4.21x
    p108_aero_up   L2_SHIPPED          4     15.5%     1.71x

**So the layer now carries the property it exists for and has LESS supported
mass than the frame that did not.** UP's best lift falls 4.21x to 1.71x against
a 14.6% base trough rate, and DOWN drops below the support floor in every
group but one. That is a real cost and it is stated rather than buried: the
frame that scored best in pass 115 is the frame that could not tell a held
regime from a churning one, and both facts are true at once.

Which of the two a node arm should query is **not settled by this pass** and
this encoder does not answer it. Both are measurable in one process from the
probe. The honest reading is that pass 115's 4.21x was measured on a frame
that was conflating two situations, so it was never the clean number it looked
like.

## What this licenses, and what it does not

It licenses the L2 encoder being **correct**: it clears the identifier guard in
both windows and it carries dwell at the setting the live path ships. It does
**not** license an edge claim. Every number here is in-sample, on one pair, one
window per direction, and `n=20`-ish groups are underpowered for a return claim
by the standard the operator applied to [5ec44914]. No held-out number is
claimed and none was measured.

## Reproduce

    python -X utf8 scripts/omen_l2_scheme_probe.py \
        --corpus data/brain_experiments/p108_aero_down.json \
        --corpus data/brain_experiments/p108_aero_up.json   # exit 0
    python -X utf8 -m pytest \
        tests/test_an_l2_scheme_must_keep_the_order_it_exists_to_carry.py \
        tests/test_the_l2_gate_must_judge_the_banding_the_live_path_ships.py -q
