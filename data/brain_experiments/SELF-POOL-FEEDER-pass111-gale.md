# The self pools were trained as CONSTANTS, and that is why the query path read DEAD

Pass 111, Gale. Item [c2f12cb0].

> ## 2026-09-10, pass 112 — THE DILUTION VERDICT IN THIS REPORT IS UNDERPOWERED, NOT A NEGATIVE
>
> Added by Cove for item **[5ec44914]**, third criterion, so this file does not
> keep reading as settled. **What stands and what does not:**
>
> **STANDS — the constant-frame finding, which is what this report is named
> after.** `build_collections` was called without `history=`, every self frame
> was the `na` sentinel, and the pools trained as a constant. That is a code
> fact, it was proven with an A-vs-A control (0/60 → 45/60), and nothing here
> touches it.
>
> **DOES NOT STAND — the conclusion "the self pools DILUTE the query, train on
> them do not query them."** The four-cell table it rests on carries **19
> trades in total** on a 60-sample held-out window: UP with self −0.4939% on
> **7 trades** against without +0.3107% on **ONE trade**; DOWN with self
> −3.2611% on **2 trades** against without −1.9445% on **9 trades**. A
> per-trade mean on one trade has no standard error. At the dispersion this
> feed actually shows (per-trade absolute returns 2–3%), detecting a 1.0pp
> effect needs roughly **63 trades per arm at sd=2% and 141 at sd=3%**; even a
> generous 1.5pp effect at sd=2% needs ~28 per arm. The cells had 7, 1, 2 and
> 9 — **underpowered by 3× to 30×**, so the run cannot distinguish dilution
> from noise *in either direction*. This is **not** a claim that the pools
> help. It is that the run licenses neither verdict, and this report's own
> text already labels the 1-trade cell "unrankable" without applying that rule
> to the summary line.
>
> **ALSO NOT STANDS — "the metacognition pools" as the subject.**
> `self_agreement` returned **2 distinct values over 800 samples**, still
> effectively a constant, for the reason this report states correctly:
> agreement is measured across query sets and the sample builder runs before
> any node exists. So the arm tested **two working pools and one inert one**,
> and the inert one is the one carrying the only signal with a measurement
> behind it (99.4% unanimous vs 73.3% split).
>
> **Do not re-run the identical arm expecting a different answer — run a
> powered one.** Criteria and the power arithmetic are on [5ec44914].
>
> ## 2026-09-11, pass 116 — THE POWERED ARM WAS RUN. Do not stop reading here.
>
> Added by Gale after the operator flagged that a negative recorded from this
> file "will stop anyone looking again". The block above ends by asking for a
> powered arm; **that arm exists**, and a reader who stops at this file lands
> on a verdict that has since been measured and did not survive. It is in
> `data/brain_experiments/SELF-POOL-POWERED-pass113-cove.md`, commits
> `5c789ca` (DOWN) and `de68f7d` (UP), both windows, one fabric back-to-back.
>
> | powered arm | with self | without self | buy every bar |
> |---|---|---|---|
> | DOWN, LINK-WETH arbitrum, 200 bars, up-rate 15.0% | **+0.3734%** on n=11 | **−1.7172%** on n=54 | −2.2088% on n=200 |
> | UP, CRV-WETH arbitrum, 600 bars, up-rate 61.8% | **−0.2564%** on n=52 | **−0.0643%** on n=48 | −0.2016% on n=600 |
>
> **The sign in DOWN is the opposite of this report's verdict, and it reverses
> again in UP.** The DOWN cell's n=11 is under the item's own n=30 floor and
> that report declines to give it a per-trade verdict, exactly as this file
> should have declined on its 1-trade cell.
>
> **The settled statement, which is a negative about the VERDICT and not about
> the pools:** on the evidence now in hand the self pools have **no measurable
> directional effect**, and this file's dilution finding was **noise**. The
> pools themselves remain **NOT MEASURED** — `self_agreement`, the one pool
> carrying a signal with a measurement behind it (99.4% unanimous vs 73.3%
> split), measured **0.004 distinct** in every arm above, so it was a constant
> in all of them and the powered run reports itself as testing **two pools,
> not three**. Nothing yet has tested the metacognition idea the operator
> asked for. What would settle it is one thing: feed `self_agreement` a real
> node's per-query-set votes.
>
> **What still STANDS from this file is only what it is named after** — the
> self frames were trained as constants, proven by an A-vs-A control
> (0/60 → 45/60). That is a code fact and nothing above touches it.


## The finding, in one sentence

Pools 15/16/19 were bound, streamed and READ by a real 19-pool node, and they
moved nothing, because every sample-building loop in this repo called
`build_collections` **without `history=`** -- so every self frame in every
training set was the `na` sentinel and the three pools trained as a constant.

## Setup

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21926 bars, 1 pair |
| node | fresh `127.0.0.1:8091`, brain dir `brain-data-meta-p111`, `pool_count` 20, tick 0 at start |
| identity | `market_predictor_v4_meta.identity.toml` (19 pools; 15-19 `kind="Internal"`) |
| production | `:8090` untouched, uptime 87874s at start and still up |
| horizon | 12 bars |
| flag | `OMEN_META_COLLECTIONS=1` |

## 1. The self frames were ONE value. Measured, both ways, same 800 bars.

Window `[21114, 21914)`, 800 samples, back-to-back in one process.

| collection | FED (`history=`) distinct | per sample | UNFED (the old loop) distinct | the value it emitted |
|---|---|---|---|---|
| `self_outcome` | **24** | 0.0300 | **1** | `slf hit=na n=0 last=na` |
| `self_agreement` | **2** | 0.0025 | **1** | `agr unan=na rate=na` |
| `self_error_run` | **25** | 0.0312 | **1** | `err run=na dir=na` |
| `temporal_sequence` | 210 | 0.2625 | (not history-fed) | -- |
| `temporal_scale` | 198 | 0.2475 | (not history-fed) | -- |

A stream with one distinct value over 800 samples cannot move a query however
good the pool is. Nothing in the system said so: the node returned 200, the B
arm fired six streams per prediction, and the pass-110 verdict read
`QUERY PATH DEAD` with no error anywhere.

**`self_agreement` is still near-constant at 2 values, and I am not going to
dress that up.** Agreement is `agreed/asked` across query sets, and a sample
builder that runs before any node exists has no query sets to disagree. Pool
16 will carry approximately nothing until the recorded prediction is a NODE's
call with its per-query-set votes attached. On the dilution law it is a
candidate to train and never query.

`temporal_sequence` at 0.2625 sits just ABOVE the empty band's upper edge
(0.103-0.260) and `temporal_scale` at 0.2475 sits just inside it. Neither is
near the 0.30 identifier guard, so neither is the near-unique trap -- but 18
being inside the band is a reason to sweep it rather than assume it queries.

## 2. The query path went DEAD -> LIVE on a real node. 0/60 -> 45/60.

`scripts/omen_query_path_probe.py`, same node, same corpus, train
`[21241, 21841)` 600 samples taught 600/600 in ONE epoch, test
`[21853, 21913)` 60 samples. A = `temporal,geometry,cross`;
B = A + `self_outcome,self_agreement,self_error_run`.

| | pass 110 (unfed) | pass 111 (fed) |
|---|---|---|
| control A vs A | 0/60 | **0/60** |
| treatment A vs B | **0/60 -- DEAD** | **45/60 -- LIVE** |
| streams fired | A 3, B 6 | A 3, B 6 |
| verdicts | admitted 60/60 both arms | admitted 60/60 both arms |

The control at 0/60 is what makes the 45 mean anything: the node is
deterministic on a repeated query, so 45 moved predictions is the self pools
being read, not run-to-run variance. Both arms fired the same stream COUNTS
in both passes -- 3 and 6 -- which is exactly why this was invisible: the
plumbing was never broken, the payload was constant.

    OMEN_META_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8091 \
      python -X utf8 scripts/omen_query_path_probe.py \
      --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --horizon 12 \
      --train 600 --test 60 --query-a temporal,geometry,cross \
      --query-b temporal,geometry,cross,self_outcome,self_agreement,self_error_run

**Moving a prediction is not improving one.** 45 of 60 predictions CHANGED;
nothing here says they changed for the better. That is the next criterion and
it is not answered in this pass.

## 3. The fix

`scripts/omen_experiment.build_samples` now walks a `ResolvedHistory`
alongside the bars and hands `as_of(index)` to `build_collections`. Both
`omen_experiment` and `omen_query_path_probe` call that one function, so both
are fixed by it.

Ordering is settle-then-frame-then-record: a prediction whose horizon lands
exactly on this bar is a FACT by the time this bar is decided, so withholding
it would be needlessly blind -- the guard is against reading the OPEN call,
and `self_frames` drops unresolved rows on top of `as_of` doing it.

The recorded prediction is the causal majority of the settled rows the bar is
allowed to see. That is deliberately NOT the node's own call: sizing the
self-pool vocabulary and measuring the node's skill are different jobs, and
only the first belongs in a builder that runs before a node exists.

**Passing `history=` is a strict no-op with `OMEN_META_COLLECTIONS` off** --
`build_collections` reads it only behind that flag -- so every non-meta run in
this repo produces byte-identical frames to before the change. The flag stays
OFF by default because sending pool 15 to a v2/v3 node returns
`unknown input pool id` and `_consolidate` reports the whole sample as a MISS:
enabling against the wrong node does not degrade training, it silently stops
it.

## 4. The test

`tests/test_a_training_set_cannot_teach_the_self_pools_a_constant.py`, 4 tests.

The load-bearing one is the negative control: it rebuilds 200 frames through
the OLD call shape and asserts all three self frames are still exactly one
`na` value. Without it the main assertion could go green for an unrelated
reason. With it, a green result means the history was genuinely threaded
through. A third test spies on `ResolvedHistory.as_of` and fails if any row
handed to a frame is unresolved -- that is the prediction_error loop that took
recall from 100% to 30% here, checked at the seam instead of in a comment. A
fourth asserts the flag-off build never even emits the three keys.

    python -X utf8 -m pytest tests/test_a_training_set_cannot_teach_the_self_pools_a_constant.py -q
    -> 4 passed

## 5. Held-out edge with the self pools QUERIED: BELOW BASELINE in the UP window

Second fresh node `:8092`, brain dir `brain-data-meta-p111b`, `pool_count` 20,
tick 0 at start. ONE fabric, train window PINNED at `--train-end 21691`
(train `[21091, 21691)`, 546 balanced pairs, ONE epoch, 0 failed), then both
held-out windows scored against it -- the DOWN arm with `--skip-train`, so the
two numbers are back-to-back on the same fabric and are comparable to each
other.

Query set `temporal,geometry,cross,self_outcome,self_error_run`.
`self_agreement` was DROPPED from the query on the dilution law: it measures
0.004 distinct per sample, which is the empty band's floor, and section 1
above explains why it cannot do better from a sample builder.

Windows chosen so the fabric saw neither: UP `[21853, 21913)` (up-rate 53.3%,
mean forward +0.6838%) and DOWN `[21703, 21763)` (up-rate 11.7%, mean forward
-1.5636%).

### UP window, on the operator's 15:07 scoreboard

| rank | metric | with self pools queried | honest baseline | verdict |
|---|---|---|---|---|
| (a) | **per-trade net on trough omens** | **-0.4939%** (7 omens) | buy-every-bar **+0.0338%** | **WORSE by 0.53pp** |
| (b) | **trough precision** (paid the round trip) | **28.6%** of 7 | -- | 7 omens is unrankable |
| (d) | exact accuracy (DEMOTED) | 6.7% of 60 | majority class 40.0% | far below |

### DOWN window, same fabric, `--skip-train`

| rank | metric | with self pools queried | honest baseline | verdict |
|---|---|---|---|---|
| (a) | **per-trade net on trough omens** | **-3.2611%** (2 omens) | buy-every-bar **-2.2136%** | **WORSE by 1.05pp** |
| (b) | **trough precision** | **0.0%** of 2 | -- | 2 omens is unrankable |
| (d) | exact accuracy (DEMOTED) | 48.3% of 60 | majority class 60.0% | below |

### The verdict over BOTH windows

**Below baseline in both. No edge. Nothing here should be promoted.**

| | UP | DOWN |
|---|---|---|
| per-trade net, trough omens | -0.4939% | -3.2611% |
| buy-every-bar in the same window | +0.0338% | -2.2136% |
| difference | **-0.53pp** | **-1.05pp** |
| exact vs majority | 6.7% vs 40.0% | 48.3% vs 60.0% |

### 5b. THE ONE-VARIABLE CONTROL: the self pools DILUTE the query

Run in the same pass, on the SAME fabric, the SAME two windows, `--skip-train`
both times. The only difference is that `self_outcome` and `self_error_run`
are dropped from the query set. This is the arm the section below said was
missing, so the section below is now answered rather than left hanging.

| | UP with self | UP WITHOUT | DOWN with self | DOWN WITHOUT |
|---|---|---|---|---|
| per-trade net, trough omens | -0.4939% (7) | **+0.3107%** (1) | -3.2611% (2) | **-1.9445%** (9) |
| buy-every-bar, same window | +0.0338% | +0.0338% | -2.2136% | -2.2136% |
| vs buy-every-bar | **-0.53pp** | **+0.28pp** | **-1.05pp** | **+0.27pp** |
| exact (demoted) | 6.7% | **21.7%** | **48.3%** | 28.3% |
| majority baseline | 40.0% | 40.0% | 60.0% | 60.0% |

**Adding the two self pools to the QUERY made the money scoreboard worse in
BOTH windows** -- by 0.81pp in UP and 1.32pp in DOWN. That is the dilution
law, and it is exactly what their distinctness predicted: `self_outcome`
0.037 and `self_error_run` 0.035 both sit far below the 0.103 floor of the
empty band. **Train on them, do not query them.**

Exact accuracy disagrees with itself across the windows (worse in UP, better
in DOWN, below the majority baseline in all four cells), which is the
operator's 15:07 point made concrete: it is not the thing to optimise, and it
would have supported the opposite conclusion had it been ranked first.

**AND IT FALSIFIES WHAT I FLAGGED BELOW.** I noticed the DOWN arm called
`trough` twice where pass 109 called it 50 times, and wondered aloud whether
the self pools were suppressing a bad call. They are not. The control calls
trough 9 times in DOWN and 1 time in UP; the self arm calls it 2 and 7. The
effect has no consistent direction, so it is not the self pools -- it is
window-to-window noise on a handful of omens. Running the control cost about
two minutes and stopped a plausible story from entering the record as a
finding.

### The observation that prompted that control, kept for the record

Pass 109's DOWN-window defect was that the brain called `trough` (buy) **50
times** into a window where 14.2% of bars rose. This DOWN window is 11.7% up
and it called trough **twice**. That is the exact behaviour the item
hypothesised the self pools would produce -- a brain that notices it has been
wrong the same way for many bars stops making the call.

I flagged it as NOT evidence because there was no arm without the self pools,
and then I ran that arm (5b). It is not the self pools. Kept here so the next
reader sees the claim and its refutation together rather than the claim
alone.

**This is a negative result and I am reporting it as one.** Feeding the self
pools made the query path live; it did not make the predictions better in the
UP window. Per the standing instruction, an at-or-below-baseline result
reported honestly is a finished measurement.

One mechanical note that is NOT an excuse and should be checked before the
next arm: the fabric predicted `slide` 34 times into a test window whose true
mix was `slide: 1`. Train mix was slide-heavy (192 of 600) and the test
window is `murk`/`climb`-heavy. That is a regime mismatch between a pinned
train window and a held-out window 160 bars later, and it caps exact accuracy
independently of anything the self pools do.

Train recall was 99.5% (199/200) on the same fabric that scored 6.7%
held-out. That gap is the standing finding of this repo restated, not news.

## 6. Pools 17 and 18 cannot be QUERIED: no bucket has enough support to judge

The operator's guess (14:17) was that `temporal_sequence` and `temporal_scale`
would query well and 15/16/19 might dilute. Measured on the same two windows,
600 bars each, no node required:

| collection | buckets with n>=30, DOWN | coverage | buckets with n>=30, UP | coverage |
|---|---|---|---|---|
| `temporal_sequence` | **0** of 210 | 0% | **0** of 191 | 0% |
| `temporal_scale` | 1 of ~170 | 6% | **0** of 172 | 0% |
| `self_outcome` | 5 | 74% | 7 | 74% |
| `self_error_run` | 7 | 74% | 9 | 96% |
| `geometry` (reference) | 0 of 585 | 0% | 0 of 579 | 0% |

**The shape of the answer is the opposite of the guess.** 15 and 19 have the
support; 17 and 18 do not. A stream whose every value is nearly unique cannot
be judged and should not be queried -- that is the identifier trap in another
form, and it is consistent with their per-sample distinctness (0.26, 0.25)
sitting at the top of the empty band rather than below it.

**I am deliberately NOT quoting a label-skew number as evidence.** Jet's null
(15:25) shows that best-bucket-minus-base-rate, selected post hoc over many
buckets, clears any reasonable threshold on pure noise 96-100% of the time. I
have no null for the base rates here (13.2% DOWN, 15.3% UP), so coverage is
the finding and skew is unjudged. Building that null is a cheap next step and
would make this table say something stronger.

## 7. The self frames DO carry signal about trough -- the QUERY is what cannot use it

I said in section 6 that building the null was the cheap next step, so I built
it. Null model: trough assigned INDEPENDENTLY of the frame at this window's
own base rate, over the SAME bucket sizes, 4000 trials, best-bucket-minus-base
each trial -- the same statistic, generated from nothing.

| window | collection | buckets | observed skew | null median | null p95 | P(null >= observed) |
|---|---|---|---|---|---|---|
| DOWN | `self_outcome` | 5 | +0.183 | +0.046 | +0.101 | **0.001** |
| DOWN | `self_error_run` | 7 | +0.159 | +0.057 | +0.118 | **0.006** |
| UP | `self_outcome` | 7 | +0.251 | +0.061 | +0.141 | **0.000** |
| UP | `self_error_run` | 9 | +0.472 | +0.077 | +0.149 | **0.000** |

All four survive, in both directions of market.

**Why this does not contradict Jet's 15:25 null, which found the same pools
inside it.** Jet simulated 14 to 50 buckets; these are 5 to 9. Post-hoc
selection of a maximum gets easier the more buckets you select from, so the
null's median skew falls from Jet's +0.18-0.29 to +0.05-0.08 here. Same
statistic, different amount of selection. Jet's caution was right and so is
this: the way to make the statistic mean something was to cut the bucket count,
not to argue with the threshold.

**And it sharpens the negative rather than overturning it.** The self frames
carry real information about the label we actually trade, in an UP window and
a DOWN window -- and adding them to the fabric's QUERY still cost 0.81pp and
1.32pp per trade (5b). Those are compatible: the information is present in the
frames and the retrieval cannot exploit it as an extra stream. That is the
dilution law being a fact about the QUERY MECHANISM, not about whether a
stream is informative -- which is a sharper statement than this repo had
before, and it is the reason "train on everything, query the discriminating
few" is the right rule rather than a heuristic.

**The next experiment this argues for** is not another query-set sweep. It is
using the self frames OUTSIDE the query -- as an abstention gate on the answer
the fabric already gives, which is free, since a bucket reading 0.000 trough
over n>=30 in both windows is a bucket where the buy call should simply not be
placed.
