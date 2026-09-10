# Pool topology design — pass 106, Cove

Written BEFORE code, as the operator asked. Nothing here claims an edge. It
establishes what the substrate can and cannot do, because the brief's first
named direction rests on a premise that is false, and a pass spent finding
that out by experiment would have measured only node variance.

## 0. The finding that changes the plan

**`PoolKind::Internal` is behaviourally inert.** Setting `kind = "Internal"`
on a pool changes nothing about how the engine treats it.

Proof, in `D:/Projects/W1z4rDV1510n`:

```
grep -c "PoolKind::Internal" crates/brain/src/brain.rs \
                             crates/brain/src/pool.rs \
                             crates/node/src/identity.rs
# -> 0, 0, 0
```

The variant is *declared* at `crates/brain/src/identity.rs:51` and matched
nowhere. The only behavioural match on `PoolKind` anywhere in the engine is:

```
crates/brain/src/brain.rs:7417:    if matches!(ps.kind, PoolKind::Action) {
```

Its own doc comment reads `Internal pool (binding, integration, future
composite layers)` — "future" is doing the work in that sentence. The
constructors confirm it: `identity.rs` ships `sensory_byte_passthrough` and
`action_byte_passthrough`, and no internal equivalent.

**The supporting citation points at the wrong thing.** The standing
instructions cite `pool.rs:683` — "internal learned-route frames re-stimulate
atoms grounded by other pools" — as evidence that Internal pools compose.
That comment sits inside `impl AtomEncoding for InstructionIntentEncoding`.
It describes the `instruction-intent` **prototype's** atom encoding, and the
word "internal" there is prose about internally-generated frames. It is not
about the `PoolKind` enum. Consequently `coding_debug.identity.toml`'s two
`kind = "Internal"` pools (`resolution` id 9, `repair_relation` id 11) are a
naming convention, not a working example of pool-to-pool binding.

### What follows from it

The fabric will **not** compose a relation between two pools on its own.
There is no learned-route machinery to switch on. Therefore:

> Pool-to-pool association must be **computed by the client and fed as its own
> frame**. The relation becomes a first-class bindable thing by being *sent*,
> not by being declared.

This is consistent with the traps list rather than in tension with it. The
trap already paid for is "never make the substrate guess what the caller can
compute" — the chained regime stream, reproduced at 73.3% at 0.98 confidence
when wrong, for a deterministic function of the bars. Computing the relation
caller-side is the *safe* side of that trap. What we must not do is feed a
relation and also expect the fabric to re-derive it.

### The levers that actually exist

1. **`prototype`** — exactly three are registered (`identity.rs:312-327`):
   `byte-passthrough`, `code-structure`, `instruction-intent`. An unregistered
   name is a hard build error (`unknown pool prototype '{0}'`). Anything
   market-shaped uses `byte-passthrough` today; a new encoding is a Rust
   change, not a TOML change.
2. **What the client sends** — `trading/omen_brain.py:165`, a 7-entry tuple:
   `Collection(name, prefix, pool_id)`. Pool ids are already env-overridable
   via `OMEN_POOL_*`, so a differently-configured node needs no code change,
   but the *set* of collections is hardcoded.
3. **Per-pool knobs** — `recent_atoms_window`, `max_concept_member_count`,
   `concept_emergence_threshold`, `decay_rate`, `prune_floor`. These are the
   ones that must never be left bare (bare defaults truncate frames to 19%
   recall).

## 1. Ground truth of the current topology

`market_predictor_v2.identity.toml` declares 11 pools. The client feeds 9.

| pool | id | kind | fed by client? |
|---|---|---|---|
| ohlcv_geometry | 1 | SensoryInput | yes — `geometry`/`geo` |
| temporal_returns | 2 | SensoryInput | yes — `temporal`/`tmp` |
| volume_flow | 3 | SensoryInput | yes — `flow`/`flw` |
| volatility_range | 4 | SensoryInput | yes — `volatility`/`vol` |
| market_regime | 5 | SensoryInput | yes — `REGIME_POOL`, chained stage-1 |
| cross_market | 6 | SensoryInput | yes — `cross`/`crs` |
| **news_entities** | **7** | SensoryInput | **NO — dead** |
| **news_state** | **8** | SensoryInput | **NO — dead** |
| forecast_horizon | 9 | SensoryInput | yes — `horizon`/`hzn` |
| instrument_context | 10 | SensoryInput | yes — `instrument`/`ins` |
| future_outcome | 11 | Action | yes — `OMEN_POOL`, decode target |

Two declared pools are never fed: no `Collection` maps to 7 or 8, and
`grep -i "news_entit|news_state|market_news" trading/omen_brain.py
scripts/omen_*.py` returns nothing. They are inert declarations. That is not
itself a bug — but any census that says "11 pools" is overcounting by two,
and the news crawl we already run is not reaching the brain at all.

So the honest description of the limitation is sharper than the brief's:
**seven flat sensory siblings that meet only at the Action pool**, plus one
chained regime stream, plus two dead pools.

## 2. The change proposed for the next pass — ONE change

Add **client-computed relation collections**. Each carries the relation
between two existing sensory families as its own prefixed frame.

Three relations, chosen because each is a normalisation the flat topology
provably cannot express (a flat sibling set can represent "return = x" and
"range = y" but never "x is large *for* y" — that conjunction only exists if
something writes it down):

| new collection | prefix | relation | why it cannot be expressed today |
|---|---|---|---|
| `rel_move_vs_vol` | `rmv` | temporal × volatility | is this move big *relative to its own recent range*? The single most standard normalisation in price prediction. |
| `rel_shape_vs_flow` | `rsf` | geometry × flow | is the shape *confirmed by volume*, or is it a thin-book artifact? |
| `rel_sym_vs_mkt` | `rsm` | cross × temporal | is this move idiosyncratic or market-wide? Directly separates alpha from beta. |

### Declared as `SensoryInput`, deliberately

Since `Internal` is inert, marking these `Internal` buys nothing. It also
carries a real risk: if any code path ever routes `observe` by kind, an
`Internal` pool would silently receive nothing — a silent-drop failure, the
worst kind to debug. `SensoryInput` is behaviourally identical today and safe
under both possibilities. The choice does not depend on the unknown.

Pool ids 12, 13, 14. Every knob set explicitly — windows sized to the
mid-tier sensory pools (32768 / 16 / 5 / 0.00002 / 0.001) so that the *only*
difference from v2 is the presence of the relation streams.

### How it gets measured, honestly

- Fresh brain dir, node on `:8091`. Never `:8090` — that is production,
  uptime 56185s when checked this pass.
- Back-to-back on one fabric: v2 collections vs v2+relations. Cross-session
  comparison is noise (89.2% and 93.6% on the same fabric 34 minutes apart).
- Held-out only, in an **up window and a down window**. Train recall is not
  a result.
- Both baselines reported: majority class, and buy-every-bar per-trade return.
- The relation streams must clear `MIN_QUERY_DISTINCTNESS` to be queried at
  all — `collection_distinctness` measures it from the corpus. A relation
  that buckets to a near-constant is diluting, and the dilution law says
  train on it but do not query it. **Check distinctness before concluding
  anything about accuracy.**
- Expected outcome is at or below baseline. That is the normal result here
  and it is a finished pass.

## 3. The other three directions, re-scoped against the engine

- **Metacognition.** Agreement is the strong signal (99.4% unanimous vs 73.3%
  split) and it is computed client-side today in `CONSENSUS_QUERIES`
  (`omen_brain.py:239`). Feeding it back as a pool is cheap. But note what
  the code comment already says honestly: unanimity is a *reproduction* gate,
  not an edge gate — held-out it was 33.3% unanimous against a 31.2% majority
  class. Do this one for abstention, not for accuracy, and do not expect the
  held-out number to move.
- **Temporal pools.** Straightforward as extra collections at several bar
  scales. Cheapest of the four, since it needs no new relation logic — just
  the same encoder over different windows. Bar-count time and wall-clock time
  differ in this feed and should be separate streams.
- **Chart-shape + mutations.** Best fit for a byte-atom substrate, and the
  most work: it needs a mutation generator (stretch / compress / invert /
  truncate / add noise) to teach "same shape" rather than instances. This is
  the one that most wants a new `prototype`, i.e. a Rust change. Schedule it
  last, and only after a relation stream has shown the plumbing works.

## 3.5 The topology was built and PROVEN TO LOAD this pass

`brains/market_predictor_v3_assoc.identity.toml` (14 pools = v2's 11 +
relations 12/13/14) and its deployment were written, and a node was brought
up on a **fresh** brain dir — `brain-data-assoc-p106`, not `brain-data-omen`
or `-omen2`, which are dirty from prior runs:

```
& "D:\Projects\W1z4rDV1510n\start_node.ps1" -Addr 127.0.0.1:8091 \
    -BrainDir "D:\Projects\W1z4rDV1510n\brain-data-assoc-p106" \
    -Identity "brains\market_predictor_v3_assoc.identity.toml" \
    -Deployment "brains\market_predictor_v3_assoc.deployment.toml"
```

`/health` returned `status OK, uptime_secs 0`. Production on `:8090` was
untouched (it was at uptime 56185s and stayed up).

Then the load was verified by consolidating one frame into each of three
pools — **with a negative control**, because "the node returned 200" is not
proof that a pool exists:

| probe | result |
|---|---|
| pool 1 `ohlcv_geometry` (existing, control) | `consolidated: True`, fired 22 |
| **pool 12 `rel_move_vs_vol` (NEW)** | **`consolidated: True`, fired 27** |
| pool 99 (does not exist, negative control) | `consolidated: False`, `unknown input pool id 99` |

The third row is what makes the second row mean something: the node **does**
validate pool ids and rejects unknown ones, so pool 12 firing 27 atoms is a
real load of the new topology and not a permissive accept-anything path.

**Two facts are now established rather than assumed:**

1. The 14-pool topology loads and is live.
2. A client can feed a newly declared pool directly — which is the mechanism
   all four of the operator's directions need, given that `PoolKind::Internal`
   will not compose anything for us.

## 3.6 The encoder shipped too, opt-in and proven end-to-end

`build_collections` now computes the three relation frames and
`RELATION_COLLECTIONS` adds pools 12/13/14, behind
**`OMEN_RELATION_COLLECTIONS=1`, default OFF**.

That default is load-bearing rather than timid. A v2 node declares 11 pools,
so sending it pool 12 returns `unknown input pool id 12` and `_consolidate`
reports the whole sample as a **miss**. Enabling the relations against the
wrong node does not weaken training — it silently stops it. The test is named
for that failure:
`tests/test_relation_collections_are_off_until_the_node_has_the_pools.py`.

Proven with the relations on, against the v3 node:

```
invariant holds; frames == COLLECTIONS == 10
consolidated: True | streams: 10
relation pools fired: {12: 26, 13: 28, 14: 27}
```

### What the three relations carry

- `rmv` — `z6`, `z24`, `rngv`. The move in units of its own noise:
  `ret / (vol24 * sqrt(n))`. `vol24` is a per-**step** stdev, so noise over
  n steps scales as `vol24*sqrt(n)`; fraction over fraction leaves this
  dimensionless.
- `rsf` — `dir`, `pos`, `impact`. Is the shape confirmed by who is trading
  it? `impact` is move per unit of relative volume: a wide range on thin
  volume is a book artifact, not a move.
- `rtn` — `t168`, `t24`, `exp`. Distance from the long baseline over that
  symbol's own noise.

Pool 14 is `rel_trend_vs_noise`, **not** `rel_sym_vs_mkt` as first drafted.
`build_collections` sees one symbol's bars, so a genuine cross-sectional
relation is not computable at that seam; shipping the cross-sectional name on
a symbol-vs-own-baseline quantity would have been a fake label on a real
stream. A true market-relative pool needs an aggregate passed in — a later
pass.

### A new bucketer, and an honest note on it

`_bucket_signed` maps a signed dimensionless z linearly onto 20 levels over
[-4, +4]. `_bucket_return` is log-spaced and calibrated for fractions with a
0.0001 floor, so the band a z-score lives in (0.5–3) lands in about four
adjacent levels. Measured: **7 levels vs 5** across that band. That is the
justification — a real but modest gain. It is *not* true that
`_bucket_return` fails to separate high from low volatility; on a 20x
volatility pair it separates them too (u24 vs u18). The test docstrings say
so rather than overclaiming.

### Two invariants this had to respect

1. `set(build_collections(...)) == {c.name for c in COLLECTIONS}` — asserted
   by the pre-existing `test_every_collection_has_its_own_byte_prefix`. The
   first draft computed the relations unconditionally and broke it; the gate
   caught it. One flag now gates the frame keys and the COLLECTIONS entries
   together, so what is built is exactly what is streamed.
2. No caller iterates the frames dict generically — every access is by key
   (checked across `trading/`, `scripts/`, `services/`), so the additive keys
   reach nothing that did not ask for them.

### MEASURED: all three relations are TRAIN-ONLY streams

The distinctness check, run on the real corpus rather than deferred —
13,219 samples over 4 files (AERO-USDC x3, cbBTC-USDC), horizon 12:

| collection | distinctness | queryable (>= 0.2)? |
|---|---|---|
| geometry | 0.510 | YES |
| temporal | 0.537 | YES |
| flow | 0.169 | no |
| cross | 0.121 | no |
| volatility | 0.086 | no |
| horizon | 0.000 | no |
| instrument | 0.000 | no |
| **rel_shape_flow** | **0.119** | **no — dilutes** |
| **rel_move_vol** | **0.089** | **no — dilutes** |
| **rel_trend_noise** | **0.030** | **no — dilutes** |

`discriminating_collections` returns `('geometry', 'temporal')` — unchanged
by the relations.

**Read this honestly: the relations do not join the query set.** By the
dilution law they are train-only streams. Adding them to
`OMEN_PREDICT_COLLECTIONS` would dilute a query that currently discriminates,
which is the mistake the law exists to prevent.

Two things stop this from being a verdict on the idea:

1. They are **not unusually bad** — they land in the same band as their own
   parent streams (`cross` 0.121, `volatility` 0.086). The relations inherit
   the coarseness of the families they relate. Only `geometry` and `temporal`
   clear 0.2 at all, and they always have.
2. `rel_trend_noise` at **0.030** is the outlier and is close to constant.
   Its `t168`/`t24` are z-scores over long baselines that move slowly, and
   its `exp` field duplicates one already in `volatility`. It is the first
   candidate to redesign or drop.

### FIXED, same pass: the relations are now queryable

The lever was bucket resolution, not more pools, and it was cheap enough to
measure and ship immediately. Sweeping `_bucket_signed`, same 13,219 samples:

| span/levels | rel_move_vol | rel_shape_flow | rel_trend_noise |
|---|---|---|---|
| 4.0/20 *(was)* | 0.089 | 0.119 | 0.030 |
| 2.0/20 | 0.161 | 0.119 | 0.072 |
| 1.0/40 | 0.306 | **0.208** | **0.268** |
| **2.0/40 *(shipped)*** | **0.314** | **0.208** | 0.165 |

**2.0/40 is now the default.** 1.0/40 lifts all three, but it saturates every
`|z| > 1` — throwing away exactly the large moves the relation exists to
flag. 2.0/40 keeps the tail out to `|z| = 2` and still clears the floor on
two of three. Choosing the config that scores worse on the metric, because
the metric is a proxy and the tail is the point, is the trade being made
here deliberately.

The number that moved, verified through the shipped code path with no
monkeypatching:

```
rel_move_vol     0.314  CLEARS 0.2
rel_shape_flow   0.208  CLEARS 0.2
rel_trend_noise  0.165  below floor (train-only)

query set: ('geometry','temporal')
        -> ('geometry','temporal','rel_move_vol','rel_shape_flow')
```

`discriminating_collections` now admits two of the three relations. The
association layer went from three dead train-only streams to two live query
streams in the same pass that built it.

### Confirmed on a wider corpus (self-audit)

The 13,219-sample measurement above drew on 4 files of which 3 were the same
symbol — effectively 2 symbols, which is thin for a claim about the feed.
Re-measured across **7 distinct symbols, 17,315 samples** (AERO-USDC,
cbBTC-USDC, CBBTC-USDC, EURC-USDC, JITOSOL-CBBTC, VELVET-USDC, VVV-WETH):

```
rel_move_vol     0.320  CLEARS
rel_shape_flow   0.211  CLEARS
rel_trend_noise  0.150  below floor
query set: ('geometry','temporal','rel_move_vol','rel_shape_flow')
```

The conclusion is unchanged and slightly stronger. Note the corpus holds
**229 files** — this is 14 of them, so it is a wider check, not an exhaustive
one. Note also that `AERO-USDC`/`cbBTC-USDC`/`CBBTC-USDC` show a
case-inconsistent symbol naming in the corpus filenames, which is worth a
look on its own.

`rel_trend_noise` stays below the floor either way. It is near-constant for
structural reasons — slow long-baseline z-scores, plus an `exp` field
duplicated from `volatility` — so it wants redesigning, not re-bucketing.
That is a named, scoped next change.

### Where the next pass starts

Not at the topology — at measurement. Run the relations against held-out data
in an **up window and a down window**, back-to-back on one fabric, both
baselines reported. Check `collection_distinctness` on `rmv`/`rsf`/`rtn`
first: a relation that buckets near-constant is a diluting stream, and the
dilution law says train on it but do not query it.

## 4. What was NOT done this pass, and why

No accuracy number was produced. With the pass budget spent establishing that
direction #1 does not work as briefed, producing a held-out number in an up
*and* a down window would have meant rushing it, and a rushed edge number in
this repo has historically been a fake one. The next pass starts at code with
the premise corrected, which is the point of writing it down first.

— Cove, pass 106
