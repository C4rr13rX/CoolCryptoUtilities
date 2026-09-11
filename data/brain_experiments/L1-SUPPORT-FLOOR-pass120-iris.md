# The L1 abstention is a SUPPORT famine — change train length, and only train length

Item [89693706], pass 120, Iris. **Node-free**: encoder arithmetic over stored
bars, so it is not blocked by the RAM floor [beec23cc]. Production's fabric on
:8090 was never contacted.

## The measurement this answers

Cove, pass 118 (`bd609fb`), on the train slice only, at the same bands and margin
the scoring path uses: over 102 base corpora at 3600s the median corpus has 49
distinct L1 motifs across ~350 labelled train bars and only **two** of them reach
`min_support=20`, covering **18%** of the train window. Over 27 corpora at 300s
the median has 61 motifs and **zero** reach support, covering 0%. That is why
74–82% of windows call nothing at every horizon from 60 to 720 minutes, and why
the whole 300s set produced two trades across four horizons.

The rule is not being selective about markets. It has nothing to be selective
with.

## The lever, and why this one

Criterion 1 allows exactly one of encoder granularity, train length, or
`min_support` to change, because changing two tells you nothing about either.

**Train length**, for two reasons. First, it is the one with a measurement
already behind it: Gale's census at `50cef36` found 0 supported groups at 539
train samples becoming 23 at 4000, on the same encoder. Second, the other two
levers both buy support by making the thing being counted weaker — a coarser
encoder merges groups, a lower floor admits smaller ones — so a rise in supported
groups would be arithmetic rather than evidence. More train bars raise support
without touching what a group *is*.

Nothing else moves. Same encoder, relative banding, `min_support=20`,
`min_lift=1.3`, and the **shipped** `L1_HYSTERESIS_MARGIN = 0.50` — deliberately
*not* the `solve_hysteresis_margin` shipped an hour earlier in this same pass for
[746c2ece], because using it here would make this a two-change measurement.

Critically, **both arms are scored on the same held-out window.** `heldout_edge`
anchors the test window to the tail of the corpus and grows the train window
backwards from it, so the short arm's 350 bars are a suffix of the long arm's
2000 and the test bars are bit-for-bit identical between them.

Command:

    python -X utf8 scripts/omen_l1_support_floor.py --horizon 6 --horizon 12 \
        --test 250 \
        --json-out data/brain_experiments/L1-SUPPORT-FLOOR-pass120-iris.json

Corpus: all 102 `data/historical_ohlcv/base` files whose median bar spacing is
3600s and which hold enough bars for the long arm — the same set Cove's census
was taken on. Test window 250 bars.

## Standard errors: read the label

`heldout_edge` returns a per-corpus mean net, not the individual trade returns,
so the dispersion available is **between corpora** and the column is named
`stderr` for the corpus-level net across corpora, not a per-trade standard error.
That is arguably the more honest denominator — trades inside one corpus are
heavily correlated, so a per-trade SE would be optimistically small — but it must
be labelled, and this repo has already quoted a +0.10pp difference over 194
trades as if it were a result.

Per-corpus nets are combined **trade-weighted**, so a corpus that called two
trades does not count as much as one that called ninety.

## Two kinds of silence, counted separately

- **UNSPENT** — no train group cleared `min_support`, so the rule had nothing to
  fit a motif→trough map *on*. Its "edge" would be the baseline with a minus
  sign. `heldout_edge` returns these as `unspent` and they are counted apart from
  abstention, because folding them together is exactly the error that made a
  support census read as a fact about the market.
- **ABSTAINED** — the rule had supported groups but none of them cleared
  `min_lift` on a test bar, so it called nothing.

## What of criterion 4 this run does and does not cover

Criterion 4 asks for the abstention rate **at every horizon** alongside the edge.
This run sweeps **two** — 6 and 12 bars, which at 3600s cadence is 6 and 12 hours
— and not the full 60-to-720-minute range the original census covered. The reason
is compute, stated rather than hidden: `build_collections` runs at ~1239
builds/s on this box and each additional horizon costs another full encode of
both arms across all 102 corpora, roughly 3½ minutes. Two horizons is what fits a
pass alongside the census.

The census itself is taken at **one** horizon, 12, and that is a deliberate
choice rather than a shortcut: the census counts how many bars a motif group
has, which the label horizon barely moves, and 12 is the horizon Cove's pass-118
census used — so the "before" number here is comparable to that one rather than
merely similar.

Anyone extending this should pass `--horizon` repeatedly; the script takes as
many as you give it.

## Results — NOT MEASURED IN PASS 120, AND THAT IS THE HONEST STATE

**The 102-corpus run did not finish inside the pass budget.** It was launched at
minute 13 against an estimate of ~12 minutes built from a 3-corpus smoke, and it
was still in its census phase at minute 27. The estimate was wrong: the smoke
corpora were the first three alphabetically and shorter than the median, so the
per-corpus cost it implied was low. No number from this run is reported, because
there is no number — and a partial census quoted as a median over 102 corpora is
exactly the shape of error this file exists to avoid.

**What IS measured, and it is a smoke only — do not quote it as the result.**
Three base corpora at 3600s, horizon 12, test 250, train 350 vs 2000:

    arm       med train bars   med vocab   med supported   med covered
    short               350           47             4.0         39.1%
    long               2000           97            25.5         73.5%

    horizon 12   trades   net/trade    baseline    abstained corpora
    short UP          3    +6.2390%    +0.0990%    1 of 3
    short DOWN       20    -1.7097%    -0.8615%    1 of 3
    long  UP         88    +0.3763%    +0.0990%    0 of 3
    long  DOWN       40    -0.2322%    -0.8615%    0 of 3

At n=3 corpora this establishes nothing about edge. The short arm's UP row is
three trades. What it does show is that the machinery runs end to end and that
the support lever moves in the expected direction, which is why the script is
committed.

## What shipped, and what the next pass should do

`scripts/omen_l1_support_floor.py` is committed and complete (`7e54ba0`). It
takes the census, the UP/DOWN split with trade counts and between-corpus standard
errors, the abstention rate per horizon, the UNSPENT/ABSTAINED separation, and
the one-class-only FAIL verdict. Nothing in it needs writing again.

**The next pass runs it and nothing else**, budgeting for it properly:

    python -X utf8 scripts/omen_l1_support_floor.py --horizon 12 --test 250         --json-out data/brain_experiments/L1-SUPPORT-FLOOR-pass121.json

One horizon, not two, and expect roughly 10 minutes. Two practical notes for
whoever picks it up. First, **run it with `python -u`** or redirect through a
line-buffered pipe — stdout buffering meant this run produced zero bytes for
fourteen minutes and its progress was unreadable, which is most of why the
over-run was not caught early enough to shrink the arm. Second, the corpus count
will be **fewer than 102**: `min_bars` is `LOOKBACK_BARS + train_long + test +
horizon + 60` = 2490 at these settings, and only 92 of the 102 base 3600s corpora
hold 2000+ bars. Report the count the script prints, not 102.
