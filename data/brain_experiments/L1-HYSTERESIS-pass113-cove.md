# L1 HYSTERESIS -- pass 113, Cove, item [2a53f971]

**Date:** 2026-09-10
**Corpus:** `data/brain_experiments/p108_aero_down.json`, `p108_aero_up.json` --
ONE symbol (AERO), 2 windows, 12-bar horizon.
**Node contacted:** NONE. Production :8090 untouched. Every number below is
node-free and both corpora were computed in ONE process.
**Machine-readable:** `data/brain_experiments/L1-HYSTERESIS-pass113-cove.json`
**Reproduce:**
`python -X utf8 scripts/omen_layer_probe.py --corpus data/brain_experiments/p108_aero_up.json --horizon 12 --relative-bands --heldout --hysteresis 0.5 --test 60`

The `--test 60` matters: the CLI defaults to 180. At the default the UP window
is 168 labelled samples and **still calls 0 trades**, printing the probe's own
`UNMEASURABLE, NOT NEGATIVE` verdict -- so the zero below is not an artefact of
the short window I chose. The tables use `--test 60` so they sit beside the
pass-110/111/112 numbers, which all used 60.

---

## THE HEADLINE, AND IT IS A REFUSAL TO SPEND A NODE ARM

Hysteresis does everything Gale measured it would do to the ENCODER, and the
UP window still calls **ZERO held-out trades at every margin from 0.00 to
1.00**. So criterion 3 of [2a53f971] -- one node arm, L0 vs L0+L1 at margin
0.50, in BOTH windows -- is **NOT MET AND SHOULD NOT BE RUN YET**. The item's
own gating rule is that the node-free precursor is decisive; it was run, and it
says the UP side of that arm cannot produce a number for any node to improve on.

## 1. WHAT HYSTERESIS DOES TO THE ENCODER (train window, 350 samples)

| corpus | margin | L1 vocab | L1 distinct | change rate | groups n>=20 | of those, lift>=1.3 | coverage |
|--------|--------|----------|-------------|-------------|--------------|---------------------|----------|
| DOWN   | 0.00   | 86       | 0.2457      | 75.4%       | 2            | 1                   | 12.6%    |
| DOWN   | 0.25   | 68       | 0.1943      | 57.3%       | 2            | 1                   |  6.3%    |
| DOWN   | **0.50** | 51     | 0.1457      | **32.1%**   | 5            | **3**               | **25.4%** |
| DOWN   | 0.75   | 26       | 0.0743      | 18.6%       | 6            | 2                   | 27.1%    |
| DOWN   | 1.00   | 14       | 0.0400      | 15.2%       | 5            | 1                   | 28.0%    |
| UP     | 0.00   | 79       | 0.2257      | 68.2%       | 3            | 1                   |  6.3%    |
| UP     | 0.25   | 50       | 0.1429      | 44.7%       | 3            | 1                   |  6.0%    |
| UP     | **0.50** | 23     | 0.0657      | **31.5%**   | 6            | **3**               | **28.9%** |
| UP     | 0.75   | 21       | 0.0600      | 28.6%       | 6            | 3                   | 30.9%    |
| UP     | 1.00   | 10       | 0.0286      | 12.9%       | 7            | 3                   | 39.4%    |

**JET'S ONE NUMBER, ASKED FOR BEFORE ANY NODE TIME: how many UP motifs reach
n>=20 with lift>=1.3 at margin 0.50? THREE.** Up from one at margin 0.00, and
the coverage they carry goes 6.3% -> 28.9%. It is not still zero, and it is not
still one. The same sweep on DOWN goes 1 -> 3 groups and 12.6% -> 25.4%.

Direction reproduces Gale's pass-111 measurement on a different window split
(mine is the held-out arm's TRAIN window, 350 samples; Gale's was 600 whole-
corpus samples), and the magnitudes differ accordingly: change rate 75.4% ->
32.1% DOWN here against 73.1% -> 37.6% there. Nothing is cross-run: both
margins and both corpora were computed in one process, on one fabric-free code
path.

## 2. THE HELD-OUT ARM, AND WHY IT DOES NOT LICENSE A NODE RUN

Train 350 bars, purge 12, test 60 (48 labelled). Motif->trough map fitted on
TRAIN ONLY and applied frozen. Cut points fitted on TRAIN ONLY and reused
unchanged -- refitting on the test window would leak its distribution into the
frame, which is the error class this repo already paid for.

| corpus | margin | trades called | per-trade net | buy-every-bar baseline |
|--------|--------|---------------|---------------|------------------------|
| DOWN   | 0.00   | 9             | +0.4818%      | -1.9901%               |
| DOWN   | 0.50   | 12            | +0.1269%      | -1.9901%               |
| DOWN   | 0.75   | 6             | -0.2185%      | -1.9901%               |
| DOWN   | 1.00   | 0             | n/a           | -1.9901%               |
| UP     | any    | **0**         | **n/a**       | +3.7706%               |

**THE UP WINDOW CALLS NOTHING AT ANY MARGIN.** Diagnosed rather than assumed --
the mechanism is not that the test window's motifs are unseen:

| corpus | margin | test bars | test vocab | seen in train | in an n>=20 group | BUYABLE (lift>=1.3) |
|--------|--------|-----------|------------|---------------|-------------------|---------------------|
| DOWN   | 0.00   | 48        | 15         | 42 (88%)      |  9 (19%)          |  9 (19%)            |
| DOWN   | 0.50   | 48        |  9         | 48 (100%)     | 14 (29%)          | 12 (25%)            |
| UP     | 0.00   | 48        | 14         | 23 (48%)      | 12 (25%)          |  **0 (0%)**         |
| UP     | 0.50   | 48        |  7         | 19 (40%)      | 13 (27%)          |  **0 (0%)**         |

UP's held-out bars DO land in supported groups -- 27% of them at margin 0.50 --
and **none of those groups is trough-rich**. The UP train window's buy-low
motifs simply do not recur in the UP test window. Hysteresis moves that from
25% supported to 27% supported and from 0 buyable to 0 buyable. This is a
different failure from the one hysteresis was built to fix, and no margin
addresses it.

The one thing hysteresis clearly buys on the held-out side is DOWN's train
coverage of the test window: 88% -> 100% of test bars carry a motif the train
window has seen. A frozen map cannot score a frame it has never met, so that is
real and it is the mechanism to keep.

## 3. WHAT I AM NOT CLAIMING

DOWN reads POSITIVE at margins 0.00 and 0.50, against a buy-every-bar baseline
of -1.9901%. **That is not an edge and must not be quoted as one**, for two
reasons, both of which this loop has already paid for:

1. **ONE WINDOW IS NOT A RESULT.** The UP side produced no trades, so the
   both-windows rule is not merely failed -- it is untestable on this corpus.
   A long-only rule flatters itself in a down window that it sat out.
2. **NINE AND TWELVE TRADES HAVE NO STANDARD ERROR.** The operator's own power
   note on [5ec44914] is the standard here: at this feed's 2-3% per-trade
   dispersion, detecting 1.0pp needs ~63 trades per arm. Twelve is underpowered
   by 5x. +0.1269% on 12 trades is a direction, not a number.

## 4. WHAT THIS SAYS ABOUT THE STALE PASS-110 NEGATIVE

`LAYER-L1-pass110-cove.md` RESULT 3 read -0.2128% DOWN and -0.5901% UP through
the 3-of-5 encoder. Under the fixed encoder plus hysteresis the DOWN number is
+0.1269% and the UP number does not exist. **The pass-110 negative is still not
re-established, and it is still not refuted.** The honest status of "does L1
carry buy-low information" is UNMEASURED in an up window, on this corpus, at
this horizon -- not negative.

## 5. WHAT TO DO NEXT, IN ORDER

1. **Do not spend the [2a53f971] node arm on this corpus.** Its UP side calls
   zero trades before any node is involved, so the arm would return one window
   and a blank.
2. **Get an UP window whose buy-low motifs recur.** Either more UP corpora (the
   40-corpus census I ran last pass names which windows exist) or a longer UP
   test window -- but see `data/brain_experiments/` pass-112: the DOWN window
   cannot be lengthened, so the arms must be sized separately, not symmetrically.
3. **[fa75fa1a] is now cheap to settle.** L1 change rate at margin 0.50 is
   32.1%/31.5%, and Gale measured L2_transitions at steps=2 clearing the 0.30
   ceiling (0.2633/0.2000) at exactly this margin. The encoder half of that item
   is now in `trading/omen_layers.sticky_motifs` rather than probe-local, so
   whoever takes it wires the scheme rather than re-deriving the banding.

## 6. THE SAME QUESTION AT 102x THE SAMPLE -- AND IT IS A NEGATIVE

Section 2 said "this corpus cannot be scored". The instruction here is to go
and get more samples rather than write that down as a finding, so I did.

**Corpus:** every `data/historical_ohlcv/base/*.json` with >=600 bars whose
MEDIAN BAR SPACING IS 3600s -- 102 corpora. The cadence filter is deliberate
and comes from [4d0b539b]: the corpus spans 166s to 345600s bars, so
`--horizon 12` means 12 hours on these 102 and something else entirely
elsewhere. Comparing across mixed cadences would be comparing 33 minutes with
48 days under one flag.

**Method:** last 482 bars of each corpus; train 350, purge 12, test 60;
relative bands fitted on TRAIN and frozen; hysteresis margin 0.50; motif->trough
map fitted on TRAIN with n>=20 and lift>=1.3. One process, no node contacted,
production :8090 untouched. 102 of 102 produced a number; no errors, no skips.
Machine-readable: `data/brain_experiments/SCORABLE-WINDOWS-pass113-cove.json`.

| window class | corpora | that CALL >=1 trade | trades | per-trade net | buy-every-bar | EDGE |
|--------------|---------|---------------------|--------|---------------|---------------|------|
| UP (baseline > 0)  | 11 | 2  | 6   | -1.3176% | +0.6541% | **-1.97pp** |
| DOWN (baseline <= 0) | 91 | 17 | 194 | -1.5575% | -1.6617% | **+0.10pp** |
| both               | 102 | 19 | 200 | -1.5503% | -1.5922% | +0.04pp |

**THE VERDICT: NO HELD-OUT EDGE.** The DOWN side's +0.10pp over 194 trades is
not a result -- at this feed's 2% per-trade dispersion the standard error over
194 trades is about 0.14pp, so +0.10pp is under one standard error of zero. The
UP side is negative and has 6 trades, which is no number at all. **The rule
fails the both-windows test, and it fails it with 10x the trades any previous
arm here had.**

**THE SECOND FINDING, AND IT IS THE BIGGER ONE: 83 OF 102 WINDOWS CALL NOTHING.**
Section 2 read the AERO UP zero as a property of that corpus. It is not -- an
81% abstention rate is what this rule DOES. Abstention is free and that is not
automatically bad, but it means the 200 trades above are drawn from 19 windows,
so the edge estimate is a 19-window estimate wearing a 102-window label.

**WHAT THIS SETTLES.** The pass-110 held-out negative is now RE-ESTABLISHED,
under the FIXED encoder and with hysteresis, at far higher power than the run
that produced it: L1 co-occurrence motifs alone carry no buy-low edge over
buy-every-bar on 1-hour base-chain bars at a 12-bar horizon. Section 4 of this
report said that question was unmeasured in an up window; it is measured now,
and the answer is no. Anyone citing section 2's DOWN +0.1269% must cite this
section with it.

**WHAT IT DOES NOT SETTLE.** This is the motif rule scored DIRECTLY, with no
fabric. It says there is little for a node to find in L1 alone at this horizon
and cadence; it does not test L1 AS A POOL alongside L0, which is what
[2a53f971]'s node arm was for, and it says nothing about L2 or about the
metacognition pools.

## SHIPPED THIS PASS

* `trading/omen_layers.py` -- `sticky_motifs()` moved out of the probe into the
  layer module, so the live path can use the same encoder the probes measure.
* `scripts/omen_l2_scheme_probe.py` -- local copy deleted, re-exports the shared
  one. Two copies is how a margin=0 fallback drifts apart.
* `scripts/omen_layer_probe.py` -- `--hysteresis` flag; the window is encoded in
  ONE call because stickiness is a fact about a sequence, not a bar; a data hole
  no longer resets the held band.
* `tests/test_hysteresis_margin_zero_is_byte_identical.py` -- 7 tests. The
  margin=0 byte-identity is pinned WITH the collapsed-tercile fallback, which
  the existing test never exercised. Proven to FAIL against the wrong encoder
  (emit `na` for an omitted stream) rather than merely passing.
