# Pass 110 — Gale — the resolved-prediction feeder for pools 15/16/19

**Item:** [c2f12cb0]. **Corpus:** `data/historical_ohlcv/base/0004_AERO-USDC.json`,
bars [18913, 21913), 3000 labelled samples, horizon 12.

## The defect

Pools 15 (`self_outcome`), 16 (`self_agreement`) and 19 (`self_error_run`) were
wired into `COLLECTIONS` and streamed to the node, and read **0.002
distinctness** — pass 109's measurement. That is not "nearly constant", it is
**one distinct frame over the whole corpus**: the `na` sentinel. The three pools
were bound, consolidated and queried while carrying exactly zero information.

The cause was not the frames. `omen_metacognition.self_frames` filters out any
prediction that is not settled — that filter is the guard against the
prediction_error loop that took recall from 100% to 30% on this substrate — and
**nothing anywhere constructed a settled prediction**. Every call passed an
empty history, so the guard filtered everything and the sentinel was the only
frame that could ever be emitted.

## What shipped

`trading/omen_resolved_history.py` — `ResolvedHistory`, one instance per
(symbol, horizon).

    record(bar, predicted, agreed=, asked=)   a prediction, UNSETTLED
    settle(resolve_index, actual)             the outcome that lands there
    as_of(bar)                                what a frame at `bar` may read

**Causality is structural, not conventional.** A prediction made at bar `j` over
horizon `h` resolves at `j + h`, and `as_of(i)` returns a row only when
`resolve_index <= i` **and** the row carries an outcome. Both conditions are
required and the second is not implied by the first: a walk that forgot to
settle a due row must not have "I forgot" read as "I predicted nothing".
`record` refuses a backwards bar and `settle` refuses to overwrite an outcome.

## The numbers

Driver: the majority-class rule over the settled rows each bar is allowed to
see — causal, non-oracle, and the same rule the scoreboard baselines against.
3000 predictions fed; `pending` at the end is exactly 12, the horizon, which is
what a correctly warm walk-forward holds in flight.

| pool | pass 109 | pass 110 | distinct frames | verdict |
|---|---|---|---|---|
| self_outcome   | 0.002 (1 frame) | **0.0077** | 1 → **23** | below the empty band |
| self_agreement | 0.002 (1 frame) | **0.0167** | 1 → **50** | below the empty band |
| self_error_run | 0.002 (1 frame) | **0.0063** | 1 → **19** | below the empty band |

Commonest frames, and they are the fact the brain could not previously know
about itself:

    457  15.2%  err run=m8to15 dir=murk
    443  14.8%  err run=m4to7  dir=murk
    704  23.5%  slf hit=q1 n=16plus last=miss

"I have been wrong 8 to 15 bars running, and every one of those calls was the
same label." That is the pass-109 defect stated from the inside — trough called
50 times into a window where 14.2% of bars rose.

## Query set: all three DILUTE. Train them, do not query them.

Every one sits below the 0.260 query floor, and **that is the correct outcome,
not a disappointing one**. These frames are deliberately coarse — bucketed
rates, run-length bands — so a healthy one is a few dozen shareable values, not
a few thousand unique ones. A stream this coarse votes for the label
*distribution* over everything it matches, which is what dilutes a query and
what helps a consolidation. This satisfies the item's third criterion on its
second branch, explicitly rather than by omission.

**A correction to my own instrument, because it nearly produced a wrong
verdict.** I first tested "constant" as a *ratio* (`<= 0.010`) and the script
printed "CONSTANT — carrying nothing" for streams holding 23 and 19 distinct
values. A constant is a **count** test — one value — and judging a deliberately
coarse pool on the ratio reports failure for a stream doing exactly its job.
The same conflation would wrongly cut an abstraction layer for having
abstracted well, which matters directly to the L1 work running beside this.

## Does the frame KNOW anything? Measured offline, and all three SEPARATE

A vocabulary is necessary and not sufficient. A pool can hold fifty distinct
values and every one of them be unrelated to what happens next, in which case
wiring it costs a consolidation and a query per sample for nothing. That is
answerable *before* any training run, and answering it first is the discipline
pass 108 went without.

Question: given the self-frame at bar `i`, how often is the prediction made at
bar `i` correct? Scored after the walk from the truth map, so nothing in the
walk could read it. Frames with n < 30 dropped. Base rate **25.8%**.

| pool | worst frame | best frame | spread |
|---|---|---|---|
| self_outcome   | 21.7% (n=189) | 37.1% (n=62) | **+15.4%** |
| self_agreement | 8.8% (n=34)   | 45.5% (n=33) | **+36.6%** |
| self_error_run | 8.6% (n=185)  | 34.8% (n=46) | **+26.1%** |

The single most useful row in this whole pass:

    err run=m16plus dir=climb   ->  8.6% correct, n=185
    base rate                        25.8%

**"I have been wrong sixteen or more bars running, and every one of those calls
was `climb`" predicts being wrong again at a third of the base rate.** That is
self-knowledge in the operator's sense: a brain that knows when it does not know
is worth more than a slightly more accurate one, because abstention is free.
Nothing in the topology could previously represent it.

`self_agreement` separating +36.6% is the agreement-over-confidence finding
reproduced from the inside — unanimity band q1 with a high hit rate runs 45.5%
against q6 at 8.8%.

### It replicates in an UP window and a DOWN window

One window is not evidence, and every brain number ever made on this repo scored
the same window. Same corpus, 3000 bars each, driver and horizon unchanged:

| window | drift | pool | worst | best | spread |
|---|---|---|---|---|---|
| UP (end 6000)    | +44.0% | self_outcome   | 10.5% (n=351) | 62.3% (n=77)  | **+51.8%** |
| UP               |        | self_agreement | 6.3% (n=63)   | 55.6% (n=36)  | **+49.2%** |
| UP               |        | self_error_run | 14.7% (n=75)  | 37.6% (n=173) | **+22.9%** |
| DOWN (end 18000) | -54.0% | self_outcome   | 12.0% (n=108) | 50.8% (n=132) | **+38.7%** |
| DOWN             |        | self_agreement | 12.1% (n=33)  | 46.4% (n=112) | **+34.3%** |
| DOWN             |        | self_error_run | 17.7% (n=282) | 43.4% (n=159) | **+25.7%** |

All three separate in both directions, and `err run=m16plus dir=climb` is the
**worst frame in the DOWN window too** (17.7% against a 28.7% base, n=282) —
the same frame, independently, in a different regime 12000 bars away. A long
error run predicting further error is not an artifact of one window.

**One honest deduction from these numbers.** `self_outcome`'s separator is
dominated by `last=hit` versus `last=miss` in both windows, and a majority-class
driver is right in runs, so part of that +51.8% is the driver's own
autocorrelation rather than knowledge. `self_error_run` is the more conservative
of the three at +22.9%/+25.7% and is the one to trust.

**The bound on this claim.** These rates are measured with the majority-class
driver, so the driver's own behaviour partly shapes which frames get populated.
What it establishes is that the frames are *related to correctness at all*,
which is what decides whether the wiring is worth a training run. It does not
establish how much the node's own predictions would gain.

## What is NOT measured, and it is the next step

**No held-out edge number.** The vocabulary of a self-frame is a function of the
*shape* of the prediction stream — how often it is right, how long it stays
wrong — and not of which predictor produced it, so the majority-class driver is
sufficient to answer "can these pools carry information at all" and is *not*
sufficient to answer "does the brain predict better with them". That needs the
node's own predictions in a walk-forward against a v4 19-pool identity, on one
fabric, in an UP and a DOWN window. Filed as the follow-up.

Nothing here should be quoted as an edge. It is a vocabulary measurement.

## Commands that prove each part

    python -X utf8 -m pytest tests/test_a_later_outcome_cannot_reach_an_earlier_frame.py -q
    python -X utf8 scripts/omen_self_distinctness.py \
        --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --bars 3000

The lookahead test fails against the obvious wrong implementation: filtering
`as_of` on the `resolved` flag alone makes bar 149 see 3 rows instead of 1,
because settling bar 200 leaks backwards into every earlier read.
