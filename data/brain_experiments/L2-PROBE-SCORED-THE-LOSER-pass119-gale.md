# The L2 held-out arm was pointed at the rejected scheme, and the winner has no support to spend an arm on

Gale, pass 119. Item [a4ba2028]. Node-free throughout: no fabric, no training
run, no :8091.

## THE DEFECT

`scripts/omen_layer_probe.py` is the instrument that does `label_skew` and
`heldout_edge` -- it is what any held-out L2 arm would be scored through. Until
this pass it computed exactly one L2 column, `L2_sequence`, at
`MOTIF_SEQUENCE_STEPS`, and there was no `L2_transitions` column at all.

`trading/omen_layers` ships `transition_motif` as L2 and REJECTS
`sequence_motif` ([fa75fa1a], done at dd0f0e1). So the probe's exit code judged
the loser, and an L2 arm run through it would have measured the loser under the
winner's name.

Two smaller instances of the same class, fixed in the same pass:

* `--hysteresis` defaulted to **0.0**. L2's frames are a function of the L1
  alphabet underneath them, so margin 0 judges an encoder the live path does
  not ship -- the configuration that produced the pass-111 block. It now
  defaults to the shipped `L1_HYSTERESIS_MARGIN` (0.50), and the margin is
  printed on every run and written onto every artifact.
* `heldout_edge` hard-wired `"L1_cooccurrence"` in five places, so a caller
  could not score any other layer even if it had one.

## THE CORPUS AND THE WINDOWS

`data/brain_experiments/p108_aero_{down,up}.json`, 900 bars each, AERO-USDC,
horizon 12 bars, `--relative-bands`, hysteresis 0.50. Both windows in every
table. Distinctness is `distinct frames / samples`, so it depends on the sample
count; each row below says which.

## (1) THE COLUMN, AND IT REPRODUCES THE SHIPPED CONSTANTS EXACTLY

At 600 samples per corpus -- the setting `L2_CHURN_CUTS` and
`L2_TRANSITION_STEPS` were fitted at:

| layer | DOWN | UP | ceiling |
|---|---|---|---|
| L1 co-occurrence | 0.0817 | 0.0683 | -- |
| **L2 transitions (SHIPPED)** | **0.2800** | **0.2267** | 0.30 PASS |
| L2 sequence (REJECTED control) | 0.5317 | 0.4083 | 0.30 FAIL |

0.2800 / 0.2267 are the numbers `L2_CHURN_CUTS`' own sweep table records for
the shipped cut, and 0.5317 / 0.4083 are the numbers [a4ba2028] quotes for the
rejected scheme. The new column is not a re-derivation that happens to agree --
it calls `trading.omen_layers.transition_motif`, pinned by a test.

At the full corpus (719 samples) both still pass: 0.2907 DOWN / 0.2114 UP,
against a control at 0.5299 / 0.3713.

## (2) WHAT THE DEFAULT CHANGE COSTS AND BUYS

Same corpora, same split, the ONLY difference is `--hysteresis`:

| | DOWN L2 | UP L2 | DOWN L1 held-out edge | UP L1 held-out edge |
|---|---|---|---|---|
| margin 0.00 (old default) | 0.5605 FAIL | 0.4965 FAIL | +1.0378%/trade, n=56 | no trade called |
| margin 0.50 (shipped, new default) | 0.2907 PASS | 0.2114 PASS | +1.0179%/trade, n=81 | no trade called |

The default was deciding whether L2 is measurable at all and was barely
touching L1. That is the whole argument for changing it.

## (3) THE HELD-OUT L2 ARM WAS REFUSED, NOT FAILED

The item's third criterion: run an L2 arm ONLY if `L2_transitions` still has at
least one n>=20 group in the TRAIN window. It does not, in either window.

Default split (train 350, test 168, 12-bar purge):

| window | L2 train frames | largest group | groups n>=20 | verdict |
|---|---|---|---|---|
| DOWN | 108 over 350 | 16 | 0 | ARM NOT SPENT |
| UP | 110 over 350 | 19 | 0 | ARM NOT SPENT |

Then ONE widening, chosen before any number was seen and with no tuning
freedom in it: the **maximum** train region the corpus allows without touching
the test window (train 539 = bars [168, 707]).

| window | L2 train frames | largest group | groups n>=20 | test bars called | verdict |
|---|---|---|---|---|---|
| DOWN | 171 over 539 | 17 | 0 | -- | ARM NOT SPENT |
| UP | 130 over 539 | 30 | 2 | **0 of 168** | spent, UNMEASURABLE |

The UP arm cleared support, fitted one buyable frame
(`co2t path=llhhm|llhlm chn=mid`, train trough lift >= 1.3), and that frame
never occurred in the held-out window. Zero called trades is an abstention, not
an edge: the `-3.8377%` the arithmetic would print is the buy-every-bar
baseline with a minus sign, and the probe flags it `UNMEASURABLE, NOT
NEGATIVE` rather than letting it be quoted.

The sell-high half at the same split, reported for completeness with its n:
UP crest called 1 bar (fall precision 100.0% on n=1, base rate 28.0%) -- far
below any floor worth a sentence.

**So the honest answer to "what is the shipped L2's held-out edge" is: it has
none to measure yet on this corpus, and the reason is a SUPPORT FAMINE, not a
market result.** The layer abstracts (criterion 1 of [fa75fa1a]) and it is too
finely grained to carry a motif->trough map at 900 bars: 130-171 distinct
frames over 350-539 labelled samples, largest group 16-30.

This is the same shape as `l1-abstention-is-a-support-famine` one layer up, and
the DOWN side being the weaker one is what the item predicted (best in-sample
lift 1.47x).

## (4) FOR CONTEXT, THE L1 ARM AT THE SAME SPLIT

L1 is measurable and the DOWN window is positive: **+1.0179% per trade over 81
trades** (48.2% of the window) against a buy-every-bar baseline of -2.9288%,
round trip cost 0.6500%, trough precision 24.7% against a 14.3% base rate. The
UP window calls zero bars, so it is unmeasurable there -- one window is not
evidence, and a rule that only fires in a falling market has not been shown to
generalise.

## (5) THE SUPPORT FAMINE IS A CORPUS-LENGTH PROBLEM, AND IT DISSOLVES

Measured after the tables above, on `data/historical_ohlcv/base/0004_AERO-USDC.json`
(21926 bars, same encoder: horizon 12, relative bands fitted on each window,
hysteresis 0.50). This is a SUPPORT CENSUS, not an edge measurement -- it makes
no claim about the market and therefore needs no UP/DOWN split.

| train samples | L1 vocab | L1 groups n>=20 | L2 vocab | L2 largest group | L2 groups n>=20 |
|---|---|---|---|---|---|
| 350 | 59 | 3 | 121 | 28 | 1 |
| 1000 | 104 | 12 | 281 | 24 | 1 |
| 2000 | 160 | 27 | 486 | 87 | 6 |
| 4000 | 180 | 57 | 796 | 140 | 23 |

L2's vocabulary grows roughly linearly with samples while its largest group
grows FASTER -- 28 -> 140 for a 11.4x increase in samples -- so the frame is
not an identifier that dilutes forever; it is a real grouping that the 900-bar
p108 corpora simply cannot fill. At 4000 train samples there are 23 supported
L2 groups, against 0 at the 350-539 the p108 windows allow.

**So the next L2 arm is worth spending, and it should be spent on a long
corpus rather than on p108.** It still needs an UP window and a DOWN window
chosen honestly -- this table establishes only that the arm will have something
to fit, which is the thing that was missing.

## WHAT TO DO NEXT, AND WHAT NOT TO

* **Do not lower `--min-support`.** Manufacturing a group of 12 to have a
  number is how this repo's fake edges were made. The probe now refuses the arm
  and says so.
* The lever is **more samples per frame**, and section (5) shows a longer
  corpus supplies them: 23 supported L2 groups at 4000 train samples against 0
  at 539. Coarsening L2 instead (hysteresis 0.75 took it to 0.2267 / 0.1633 in
  [fa75fa1a]'s table) is the fallback, and it costs L1 vocabulary, so try the
  corpus first.
* **A node arm on L2 is now worth spending, but not on p108.** Pick an UP and a
  DOWN window out of a 20k-bar corpus with at least ~2000 train samples, and
  run the node-free held-out arm in this probe first -- it costs seconds and a
  negative there means there is nothing for a fabric to find.

## COMMANDS THAT PROVE EACH PART

```
python -X utf8 scripts/omen_layer_probe.py \
    --corpus data/brain_experiments/p108_aero_down.json \
    --horizon 12 --relative-bands --stop 768            # table (1), DOWN
python -X utf8 scripts/omen_layer_probe.py \
    --corpus data/brain_experiments/p108_aero_up.json \
    --horizon 12 --relative-bands --stop 768            # table (1), UP
python -X utf8 scripts/omen_layer_probe.py \
    --corpus data/brain_experiments/p108_aero_down.json \
    --horizon 12 --relative-bands --heldout --train 539 # table (3), DOWN
python -X utf8 scripts/omen_layer_probe.py \
    --corpus data/brain_experiments/p108_aero_up.json \
    --horizon 12 --relative-bands --heldout --train 539 # table (3), UP
python -X utf8 -m pytest \
    tests/test_the_l2_probe_scores_the_shipped_scheme_not_the_rejected_one.py -q
```

The p108 corpora are local artefacts and are not in the repository, so the
tests are written against `data/historical_ohlcv/base` and skip cleanly where
that is absent.
