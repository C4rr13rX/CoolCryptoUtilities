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
the symbol's MEANING does not shift with the window — though see Result 5,
which measures that its COST does, and corrects an earlier claim in this report
that the rate made it window-free.

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

## Result 5 — the cut does NOT travel across window lengths, and this argues against my own constant

Added after the fact, because the obvious next question about a constant chosen
on one setting is whether it survives the others. Same sweep, same corpora, same
process, at three window lengths (worst of both windows):

    window   one change reads   (0.10,)   (0.15,) SHIPPED
     8       0.1429             0.2683    0.3100  FAILS the ceiling
    12       0.0909             0.2800    0.2800  passes
    20       0.0526             0.2667    0.2700  passes

**A rate does not make the cut window-free.** The reason is quantisation, not
anything subtle: the rate can only take the values `k/(n-1)`, so which side of
the cut *one change* falls on is a function of the window. At 12 and 20 bars one
change is held; at 8 bars it is 0.1429, lands above 0.15, the held band
collapses to "no change at all", the buckets split differently and the frame
goes over the ceiling.

And the window-8 row is **not** an argument for lowering the cut to 0.10: at a
window of 8 that puts one change above the held band, so a persistent regime
stops being distinguishable from an alternating one — the property the symbol
exists for. The two constraints pull opposite ways, and 12 is where both hold.

So `L2_CHURN_CUTS = (0.15,)` is correct **at the shipped window of 12** and is
not a free constant. Pinned by
`test_one_change_is_held_at_the_window_the_cut_was_measured_under`, which fails
if either the cut or the window moves without a re-sweep. This interaction is
invisible in any synthetic that happens to use a length where the two agree,
which is why it is a test and not a comment.

    for W in 8 12 20; do python -X utf8 scripts/omen_l2_scheme_probe.py \
        --corpus data/brain_experiments/p108_aero_down.json \
        --corpus data/brain_experiments/p108_aero_up.json --window $W; done

## Still unmeasured, stated so nobody reads this as broader than it is

Every number here is AERO-USDC on base. `p108_aero_up/down` are the only bar
corpora in `data/brain_experiments/`, so **the churn cut has not been tested on
a second pair** and could be fitted to this one. That is the first thing to do
to this encoder, and it is node-free.

## Result 6 — A SECOND PAIR, AND L2 FAILS THE GUARD ON IT ENTIRELY

Measured in the same pass, because the "still unmeasured" section above was one
command away from being measured. Built `p116_arbweth_up.json` and
`p116_arbweth_down.json` from `data/historical_ohlcv/arbitrum/0039_ARB-WETH.json`
(40,504 bars) by sliding a 900-bar block and taking the most-up and most-down
block by close-to-close return: **UP +13.49%, DOWN -11.22%**, 600 samples each,
shipped banding, window 12, steps 2.

    corpus              L1       L2_path   L2_SHIPPED   L1 change rate
    p116_arbweth_down  0.1950   0.4650    0.4700 FAIL   62.4%
    p116_arbweth_up    0.1150   0.4467    0.4533 FAIL   62.3%

    (AERO, for comparison: L2 0.2800 / 0.2267 PASS, change rate 37.6% / 38.2%)

**L2 is an identifier on ARB-WETH, with or without the churn symbol.** The
churn symbol is close to free here (+0.0050) precisely because the path is
already near-unique — there is nothing left for it to split.

**THE CAUSE IS L1, NOT L2, AND IT IS THE SAME MECHANISM AS BEFORE.** The sticky
L1 takes AERO's change rate to 37.6%/38.2% and a change-order path over that is
fine; on ARB-WETH the same `L1_HYSTERESIS_MARGIN = 0.50` only reaches
62.3%/62.4%, and a path over an alphabet that changes on nearly two bars in
three is near-unique by construction — exactly the finding that opened
[fa75fa1a] in the first place.

So `L1_HYSTERESIS_MARGIN = 0.50` is **fitted to AERO-USDC**. It is not a
property of the feed, and the L2 guard passing is not a property of the encoder
— it is a property of AERO. The margin is what needs re-sweeping per pair (or
replacing with something that targets a change RATE rather than a fixed band
fraction), and no node arm on L2 should be spent until it holds on more than
one pair.

**One caveat on the DOWN window, stated because it makes half the table
unreadable rather than merely weak:** `p116_arbweth_down` has a **0.0% trough
base rate** — the omen label never fires in that window — so every group, lift
and coverage number for that corpus is meaningless. The DISTINCTNESS numbers
are unaffected (they do not use the label) and they are what this section
claims. A window with no positive label needs replacing before any skew or lift
is quoted from it.

    python -X utf8 scripts/omen_l2_scheme_probe.py \
        --corpus data/brain_experiments/p116_arbweth_down.json \
        --corpus data/brain_experiments/p116_arbweth_up.json \
        --symbol ARB-WETH --chain arbitrum
