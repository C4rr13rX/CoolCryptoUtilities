# Association topology — design, and the first measurement of it

Pass 108, Gale. Item `[cd461b30]`.

This finishes the arc Cove opened in pass 106 (`TOPOLOGY-DESIGN-pass106-cove.md`)
and half-closed in pass 107. The design is restated here in the form it ended up
taking, the premise underneath it is re-verified from source rather than
inherited, and — the part that was missing — **the relation arm is measured
against the flat arm, in an UP window and a DOWN window.**

Nothing here claims an edge. The result is a negative, and it is stated plainly.

---

## 1. The premise, re-verified from source

The item is titled "design the `PoolKind::Internal` association topology". The
honest answer is that **`PoolKind::Internal` cannot carry the association**,
and I re-checked that rather than take it on Cove's word, because the whole
design rests on it:

```
$ grep -rn "PoolKind::Internal" crates/          # in D:/Projects/W1z4rDV1510n
(count: 0)

$ grep -rn "matches!(.*PoolKind" crates/brain/src/*.rs
crates/brain/src/brain.rs:7417:  if matches!(ps.kind, PoolKind::Action) {
```

The variant is declared and matched **nowhere**. The only behavioural match on
`PoolKind` in the entire engine is `Action`. Setting `kind = "Internal"` on a
pool changes nothing the engine does with it.

So the association cannot be *declared*. It has to be **computed by the client
and sent as its own frame**. The relation becomes a first-class bindable thing
by being written down, not by being labelled.

This is the safe side of a trap already paid for — "never make the substrate
guess what the caller can compute". What we must not do is feed a relation *and*
expect the fabric to re-derive it.

### Why the new pools are `SensoryInput` and not `Internal`

Deliberate, and it does not depend on the unknown. `Internal` buys nothing today,
and if any code path ever routes `observe` by kind, an `Internal` pool would
silently receive nothing — a silent-drop failure, the worst kind to debug.
`SensoryInput` is behaviourally identical now and safe under both futures.

---

## 2. The topology

`market_predictor_v3_assoc.identity.toml` = v2's 11 pools + three relation pools.
Each binds **two named sensory pools** and carries the normalisation between them:

| pool | id | binds | the relation it writes down |
|---|---|---|---|
| `rel_move_vs_vol` | 12 | `temporal_returns` (2) × `volatility_range` (4) | is this move big *for its own recent range*? |
| `rel_shape_vs_flow` | 13 | `ohlcv_geometry` (1) × `volume_flow` (3) | is the shape *confirmed by volume*, or a thin-book artifact? |
| `rel_trend_vs_noise` | 14 | `temporal_returns` (2) × `volatility_range` (4), path-wise | is the drift large against the path noise that produced it? |

Each is a conjunction a flat sibling set provably cannot express: the fabric can
hold "return = x" and "range = y", but never "x is large **for** y" — that
conjunction exists only if something writes it down.

**Encoding:** byte-passthrough over a bucketed ratio, prefix-namespaced
(`rmv` / `rsf` / `rtn`), so relation atoms are byte-disjoint from the sensory
atoms they are derived from. Client side these are three `Collection` entries in
`trading/omen_brain.py`, behind `OMEN_RELATION_COLLECTIONS=1`, off by default.

That default is load-bearing rather than timid: a v2 node declares 11 pools, so
sending it pool 12 returns `unknown input pool id 12` and the sample is a MISS.
Enabling the relations against the wrong node does not degrade training, it
**silently stops it**.

**Every knob is set explicitly** — bare defaults truncate frames to 19% recall:

```
$ grep -A9 'name = "rel_' brains/market_predictor_v3_assoc.identity.toml \
    | grep -E "recent_atoms_window|max_concept_member_count|decay_rate|prune_floor"
recent_atoms_window = 32768      max_concept_member_count = 16
decay_rate = 0.00002             prune_floor = 0.001          (x3, one per pool)
```

---

## 3. How it was measured

Four cells: {flat, +relations} × {UP window, DOWN window}. One arm per **fresh
fabric** — four of them — because you cannot train two arms into one fabric
without the second reading as the sum of both. All four run back-to-back inside
one session, which is what controls the node variance that made the same fabric
read 89.2% and 93.6% thirty-four minutes apart.

Corpus: `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21926 hourly bars.
Two 900-bar slices, far apart in the corpus, chosen by the realised direction of
their **held-out tail** and written to `p108_aero_up.json` / `p108_aero_down.json`:

| window | corpus idx | held-out up-rate | held-out mean forward |
|---|---|---|---|
| DOWN | 7300 | 27.2% | −2.1836% |
| UP | 12400 | (see §4) | (see §4) |

Per cell: 350 train samples (balanced), 180 held-out, horizon 12 bars, 12-bar
purge gap between train and test. Nodes on **:8093–:8096**. Production on
**:8090 was never touched** — verified healthy at uptime 72045s throughout.

### The power of this test, stated up front

180 held-out samples puts the standard error on an accuracy near 0.3 at about
**3.4 percentage points**. This design can only detect a *large* effect —
roughly 7pp or more. It cannot resolve a small one, and I do not claim it can.
That is the price of fitting four honest cells into the pass instead of one
well-powered cell in a single window, and a single window is the exact trap
that has produced two fake edges in this repo already.

---

## 4. Results

**One of the four cells produced a number. Three did not, and the reason is a
resource fault on the box, not a property of the topology.** Both facts below
are measured, and the second is the more useful of the two.

### 4.1 The cell that completed — DOWN window, flat arm (7 collections)

`omen-aero_down-h12-DOWN-20260910-140509.json`, node :8094, fresh fabric,
311 balanced train samples, 180 held-out:

| measure | value | baseline | verdict |
|---|---|---|---|
| held-out exact | **30.0%** (180/180 admitted) | majority class **55.0%** | **25pp BELOW baseline** |
| per-trade net of cost | **−3.1565%** (41 buy omens, 19.5% paid) | every-bar buy **−2.8336%** | **worse than buying every bar** |
| train recall | 100.0% (40/40) | — | reproduction, not prediction |

The confidence sweep is flat — 0.00 through 0.50 all select the same 41 trades
at −3.156%. Confidence carries no information about correctness here, which
matches the standing finding that confidence is worthless as a correctness gate.

Train recall 100% beside held-out 30% against a 55% majority is this fabric's
signature failure restated: **it reproduces everything and generalises at worse
than chance.**

### 4.2 The three cells that did not complete, and why

The UP-window pair (:8095, :8096) and the DOWN relations arm (:8093) never
returned. The cause is not the encoder and not the corpus:

```
RAM total 31.8GB   free 2.5GB   used 29.3GB
pid 10792  w1z4rd_node  6820 MB   (:8091, Cove's pass-106 fabric, up 4.3h, idle)
pid   588  python       3646 MB
```

Nodes :8095 and :8096 **answered `/health` with `uptime_secs 6` and then died** —
`Get-NetTCPConnection` shows `NO LISTENER` on both while the surviving nodes
still listen. :8093 kept its listener but stopped answering `/stats` inside 12s.
The box is at **92% memory**, and a w1z4rd_node that has trained for hours holds
several gigabytes and never gives it back.

**This is why the relation arm went unmeasured in pass 107 as well.** Cove
recorded it as "10 collections train 3.3x slower (1.7/s vs 5.6/s)" and read it
as a throughput problem. On this evidence it is at least partly a memory
problem: the relation arm adds three pools' worth of resident state to a box
that has none to give, and past a threshold the node does not slow down, it
dies. A throughput fix (fewer samples) does not address that; only freeing
memory does.

Corroborating the diagnosis rather than assuming it: the one cell that completed
is the one that ran when **two** nodes were up. Every cell launched once **four**
nodes were up failed. The flat arm is not privileged here — the DOWN flat cell
and the UP flat cell are the same code on the same corpus size, and the UP flat
cell died too.

---

## 5. Verdict

**The item's measurement criterion is NOT met, and I am not going to redefine it
to look met.** Criterion 3 asks for held-out edge in an UP window *and* a DOWN
window, both arms, back-to-back. I have one arm in one window.

What is established:

1. **The design is written down and its premise is verified from source.**
   `PoolKind::Internal` is matched 0 times in the engine; association must be
   client-computed and sent as its own frame. Pools 12/13/14 name the two
   sensory pools they bind, the relation, and its encoding, with every knob set
   explicitly (§2).
2. **No edge is claimed. The one measured cell is below both baselines** — 30.0%
   exact against a 55.0% majority class, and −3.157% per trade against −2.834%
   for buying every bar. In a DOWN window the fabric's omens lose *more* than
   indiscriminate buying.
3. **The blocker on the relation arm is now named and measured**: the box runs
   at 2.5GB free of 31.8GB, and fresh nodes die seconds after binding. This is
   a prerequisite for direction #1, and it is an infrastructure fix, not a
   topology one.

`OMEN_STRATEGY_ENABLED` is **0** and was never touched. Production on `:8090`
was never trained against and was verified healthy (uptime 72045s) at the end.

### What the next pass should do first

Reclaim memory **before** launching anything: retire idle experiment nodes
(`:8091` at 6.8GB and `:8092` are both from finished passes), then run the four
cells **sequentially on one reused port with a fresh brain dir per cell**, not
four concurrent nodes. Four concurrent nodes is what killed this measurement,
and sequential costs no more wall-clock than four dead ones.

— Gale, pass 108

