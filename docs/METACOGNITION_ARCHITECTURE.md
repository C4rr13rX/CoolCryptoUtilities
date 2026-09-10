# Metacognition needs DEPTH: pools predicting on pools, not five siblings

**The five flat pools stay. What is not acceptable is stopping there.**

The operator asked for metacognition and I shipped five flat `Internal` pools
(15-19) that each read raw market frames and each feed the outcome pool. Those
are fine and they keep their place: five independent pools carrying self-state
and temporal structure are worth having on their own terms, and their frames,
bucketing and guards are all sound.

What they are not is DEEP metacognition, and treating them as if they were --
shipping five siblings and calling the direction done -- is the actual error.
This document adds the depth they were missing. Nothing below removes them.

## What was actually asked for

> "By metacognition I don't mean just one pool. It would theoretically be pools
> predicting on pools predicting on pools and passing their output to each
> other and others in ways that tells us something if variables occur together
> and/or over time in a sequence that has meant something predictable."

Three claims, all structural:

1. **Depth.** A pool's *input* is another pool's *prediction*, stacked in
   layers.
2. **Lateral flow.** Pools pass output to each other, not only upward.
3. **Motifs.** What is being learned is co-occurrence *and* sequence — "these
   variables together, in this order, have meant something before."

And the premise underneath, which is correct and is the reason the
architecture has to change:

> "Prediction works in this model by the idea of 'I've seen it before', but
> abstraction context is multivariate. It's looking for motifs. Or it should
> be."

A binding fabric answers "I have seen this before". With one flat layer, "this"
is the raw conjunction of every sensory frame at one instant — a nearly unique
key. That is why distinctness governs everything here, why near-unique streams
maximise recall and destroy generalisation, and why the fabric can reproduce
98.7% of its training set while generalising at chance. **It has seen the
instant before. It has never seen the MOTIF before, because nothing names one.**

A motif is an abstraction over instants. To recognise one you need a layer
whose vocabulary is *already abstract* — and the only way to get that on this
substrate is to make one layer's output another layer's input.

## The substrate already supports this. Verified, not assumed.

- `crates/brain/src/brain.rs:3531` — `integrate(query_pool, target_pool)`.
  **Any pool can be a target.** Nothing privileges the Action pool except
  convention.
- `crates/brain/tests/critical_thinking.rs:76` —
  `chain_explore_walks_from_seed_to_target_pool` walks pool → pool through the
  grounded fact graph and asserts it reaches a *different* pool. Lateral flow
  is a tested property.
- `crates/brain/src/identity.rs:184` — `FeedbackLoopSpec { source_pool,
  target_pool, signal, gain, delay_ticks }`. Domain-neutral pool-to-pool
  edges, including a **delay**, which is how a temporal edge is expressed.
- `trading/omen_brain.py:991` — `_consolidate(streams, outcome_pool,
  outcome_frame)` already trains into an arbitrary pool.
- `trading/omen_brain.py:1025-1028` — **the two-stage chain already in
  production**: stage 1 trains market frames into `REGIME_POOL`, then stage 2
  reads that pool as an input. This is one layer of exactly the structure
  being described. It works. It has simply never been generalised past depth 2.

So this is not a substrate feature request. It is an architecture that the
runtime supports, the client half-implements, and nobody has built.

## Why the flat design fails, in one measurement

The existing chain is *also* the best evidence for how to do it wrong. Stage 1
predicts the regime and feeds its guess to stage 2. Measured: 4 distinct
values over 2725 samples, reproduced at 73.3%, at 0.98 confidence when wrong —
for a value that is a **deterministic function of the bars**.

The lesson is not "chaining fails". It is:

> **Never make a layer predict what a layer could compute. Chain layers to
> ABSTRACT, not to guess.**

A higher layer earns its place when its vocabulary is *smaller and more
meaningful* than its input's — when it turns a near-unique instant key into a
motif name that many instants share. If layer 2's output is as distinct as
layer 1's input, it has abstracted nothing and only added a lossy copy.

## The design

Four layers. Each layer's vocabulary is deliberately smaller than the layer
below it — that shrinkage *is* the abstraction, and it is the thing to measure.

```
  L0  SENSORY          pools 1-14, unchanged
      geometry, temporal, flow, volatility, cross, news, instrument,
      and the three relation pools

           |  many near-unique frames per instant
           v

  L1  CO-OCCURRENCE    "which variables are doing something together, NOW"
      target pools, trained FROM L0
      vocabulary: a few dozen motif names, not one per sample

           |  one motif name per instant
           v

  L2  SEQUENCE         "which motifs, in which ORDER, over the last N bars"
      trained FROM a window of L1 outputs
      this is where 'over time in a sequence' lives

           |  one sequence-motif name per instant
           v

  L3  SELF             "when I have seen this sequence-motif before,
                        was I right, and which way was I wrong"
      trained FROM (L2 output, my past prediction, the settled outcome)

           |
           v
      OUTCOME POOL 11  the omen
```

**Lateral edges**, which are the part a strict stack would miss: L1 motifs are
also inputs to L3, so "I am unreliable when THIS co-occurrence is present"
is expressible without routing through L2. `FeedbackLoopSpec.delay_ticks`
carries the temporal edges.

### What each layer's frame actually is

- **L1 co-occurrence.** Not a re-encoding of the bar. The *pattern of which
  L0 streams are simultaneously extreme*: e.g. `co vol=hi flow=hi geo=mid
  cross=lo`. Distinctness target ~0.02-0.10 — many instants must share a
  motif or there is no abstraction. Computed, never predicted.
- **L2 sequence.** The ordered path of the last N **L1 motif names**, not of
  prices. This is the layer that makes "a sequence that has meant something
  predictable" representable at all. My `temporal_sequence` pool encodes a
  path of raw price steps; that is the same idea applied one layer too low.
- **L3 self.** Keyed by L2's motif: *given this sequence-motif, what is my
  historical hit rate and error direction?* This is where pools 15/16/19
  belong — reading an abstraction, not raw frames.

### Where the five existing pools sit

They are kept, and they are ALSO the starting material for the upper layers.
Both things are true and neither cancels the other:

* **As independent pools they stay wired as they are.** A self-outcome frame
  read straight off raw market context is a legitimate signal and costs
  nothing to keep. Their frames and guards -- bucketing, the resolved-only
  feedback guard, carrying agreement and never confidence -- are sound and
  are reused verbatim.
* **As the seed of a hierarchy they get a second, deeper wiring.**
  `temporal_sequence` and `temporal_scale` are L2-shaped ideas currently
  reading L0 input; `self_outcome`, `self_agreement` and `self_error_run` are
  L3-shaped ideas currently keyed on raw context rather than on a motif. The
  upper layers read the SAME frames keyed differently, in parallel with the
  flat ones.

That parallel arrangement is deliberate: it makes the depth question
falsifiable. If the deep wiring adds nothing over the flat pools, the
measurement says so directly, because both are present in one fabric.

## What "accurate" means here, and it is NOT 5-class exact accuracy

The operator's framing, and it narrows the target usefully:

> "It's about capturing and understanding enough data to be profitable, which
> means we need to be accurate in specifically predicting when we can buy low
> and/or sell high."

**The labels are already right.** `trough` maps to `buy` and `crest` to `sell`
in `OMEN_ACTIONS`; `murk` exists so a move that cannot pay its own round trip
is not a signal. The vocabulary was built for exactly this question.

**The headline number is not.** Every report leads with `heldout_exact_accuracy`
-- 5-class exact -- and every arm is compared on it. That measures whether the
brain can label a bar, which is a harder and different question than whether it
can find a profitable entry. The two come apart in both directions:

* A run can improve exact accuracy by getting `murk` and `slide` right more
  often and place no better trades at all. Most of the label mass is not
  `trough`; in the pass-109 DOWN window it was `slide` 99, `murk` 44,
  `trough` 25, `crest` 10, `climb` 2. **Exact accuracy is dominated by labels
  that are never traded.**
* A run can find genuinely good entries and score badly on exact, because
  calling a `trough` a `climb` is an error that still buys into a rise.

So the scoreboard for this work is, in order:

1. **Per-trade net on `trough` omens, against buy-every-bar in the same
   window.** This already exists in section 4 of the experiment output and is
   simply not the number anyone quotes.
2. **Trough precision** -- of the bars called `trough`, what share actually
   paid the round trip. This is "accurate in specifically predicting when we
   can buy low", stated as a number.
3. **Total P/L against the majority-class MONEY rule**, per Gale's pass-109
   point: "always `slide`" = never buy = 0.0000, "always `climb`" = buy every
   bar. A five-class head that spreads its predictions was beaten by a
   constant on both.
4. Exact accuracy, reported but demoted -- useful as a sanity check that the
   fabric is learning anything, never as the thing being optimised.

**`crest` is currently unscored as an action.** `omen_experiment.py:552` says
"long-only, so a crest is an abstention, not a short", which is correct for
the live lane -- but it means the sell-high half of the operator's target is
measured nowhere. A `crest` that correctly predicts a fall is worth real money
as an EXIT on a held position even in a long-only system, and nothing reports
whether the brain can call one. That gap should be closed by scoring `crest`
precision against forward returns, not by shorting.

**What this changes about the layers.** A motif layer earns its place if it
improves trough precision and per-trade net, not if it nudges 5-class exact.
Report all four numbers per arm so the distinction is visible, and state which
one moved.

## What would falsify this

This is a bigger change than anything the loop has tried, so it needs a
falsification plan that does not depend on a final edge number.

1. **Abstraction is measurable.** Distinctness must DROP layer over layer:
   L0 ~0.4-0.96, L1 target <0.15, L2 <0.15, L3 <0.05. **A layer whose
   distinctness does not fall has abstracted nothing** and must be cut. This
   is a cheap, early, decisive test that needs no held-out edge.
2. **Each layer must be queryable.** `scripts/omen_query_path_probe.py` must
   show a changed prediction when a layer's frame is perturbed. A layer that
   is trained and never read is the pass-108 relation failure repeated.
3. **Depth must earn itself.** L0→outcome, then +L1, then +L2, then +L3,
   back-to-back on one fabric. The rig is deterministic (Gale, pass 109:
   identical to every figure on a rebuilt fabric), so small differences are
   real. **Any layer that does not move held-out accuracy in BOTH an up and a
   down window gets cut**, not kept for elegance.
4. **The baseline is the majority class as a MONEY rule.** Gale's pass-109
   point: "always predict `slide`" = never buy = 0.0000, and the brain scored
   -1.0696. "Always predict `climb`" = buy every bar = +6.6314 against the
   brain's +0.1182. A five-class head that spreads its predictions is being
   beaten by a constant. Beat the constant, on total P/L, in both windows.

## The traps this must not walk into

Every one of these is already paid for in this repo:

- **Never predict what you can compute.** Every layer frame is computed from
  settled facts. The stage-1 regime is the counter-example, at 0.98 confidence
  when wrong.
- **The feedback guard stays.** L3 reads only RESOLVED predictions. Feeding a
  live prediction back as its own input is the `prediction_error` loop that
  took recall from 100% to 30%.
- **One consolidation epoch.** Two entrenches a dominant attractor.
- **Byte-disjoint names at every layer.** Atoms are bytes: an L1 motif name
  must not be a substring of an L2 name, or the frequent swallows the rare.
- **Train broad, query narrow.** The dilution law does not stop applying
  because the streams are now abstract.
- **Depth is not free.** Each layer costs a consolidation and a query per
  sample. Training throughput already collapses with fabric size (18.6/s at
  250 pairs → 3.1/s at 2500). Measure the cost per layer and say whether the
  arm fits in a pass before launching it.

## Order of work

1. **L1 only.** Add the co-occurrence layer, prove distinctness drops, prove
   the query path fires it, measure L0 vs L0+L1 in both windows.
2. **L2 on top of L1** — the sequence layer, keyed on L1 motif names. Rewire
   `temporal_sequence` to read L1 rather than raw prices.
3. **L3 last** — rewire pools 15/16/19 to key on L2's motif.
4. Only then consider lateral edges and `delay_ticks`.

One layer per pass. A layer that does not pay gets cut and the report says so.
