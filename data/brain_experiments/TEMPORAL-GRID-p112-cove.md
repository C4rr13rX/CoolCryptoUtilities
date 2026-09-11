# Bar-count time is not wall-clock time, and the horizon atom crosses the two

**Pass 112 — Cove — item [c9880f94], acceptance criterion 1 and 2.**
**No node was started, no fabric trained, no brain directory written.
`OMEN_STRATEGY_ENABLED` stays 0.**

> **CRITERION 1 FIRED, AND IT SAYS STOP.** The item's first and cheapest
> criterion is "state whether the brain's training grid and the live feed's
> irregular tick spacing are being crossed anywhere without resampling … say so
> and stop". They are crossed. No temporal pool was added this pass, because a
> new pool measured through a seam that means two different things would
> measure neither.

## The command that proves every number here

```
python -X utf8 scripts/omen_temporal_census.py \
    --corpus data/brain_experiments/p108_aero_up.json \
    --corpus data/brain_experiments/p108_aero_down.json \
    --horizon 12 --samples 600 --live-hours 6 \
    --json-out data/brain_experiments/TEMPORAL-GRID-p112-cove.json
```

Exit code **2** — the census exits nonzero when the crossing exists, so it can
gate a pass rather than merely inform one. Tests:
`tests/test_a_bar_count_horizon_is_not_a_wall_clock_horizon.py`, 6 passed.

| | |
|---|---|
| corpora | `data/brain_experiments/p108_aero_up.json`, `p108_aero_down.json` |
| symbol / pairs | AERO-USDC, 1 pair each, 900 bars each |
| census window | newest 600 bars before the horizon purge, both corpora |
| live tape | `market_stream`, last 6.0h, 1696 ticks, 8 busiest symbols |
| horizon | 12 bars |

---

## RESULT 1 — the horizon atom is the same bytes for two questions 60x apart

`trading/omen_brain.py:598` builds the horizon frame as

```python
horizon = f"hzn h={int(horizon_bars)}"
```

`bar_seconds` is a field on the `Omen` answer (`omen_brain.py:823`) and it
**never enters any frame** — `grep -n "bar_seconds" trading/omen_brain.py`
returns six lines, all of them plumbing on the answer, none of them a frame.

Measured, not assumed:

| | training corpus | live path |
|---|---|---|
| cadence | **3600 s**, `uniform_share` **1.0000** over 899 h, max gap 3600 s | `OMEN_BAR_SECONDS` = **60** (`omen_reversion.py:46`) |
| horizon frame | `'hzn h=12'` | `'hzn h=12'` |
| what it means | **720 minutes ahead** | **12 minutes ahead** |

One atom, two questions, **60x apart in wall clock**. The fabric has no way to
tell them apart because the discriminating quantity was never written down.

### How far this reaches, stated at its real size

* It is **trained into every sample** — `horizon` is in `COLLECTIONS`, so every
  training pair binds it.
* It is **fired by one of four consensus members** —
  `CONSENSUS_QUERIES[3] = ("temporal","geometry","cross","horizon","instrument")`.
* It is **not** in `PREDICT_COLLECTIONS` (`("temporal","geometry","cross")`), so
  the primary answer does not fire it. This is consensus-member contamination,
  not a corrupted primary decode.
* **No held-out number in `data/brain_experiments/` is invalidated.** Every
  experiment trains and tests on ONE corpus at ONE cadence, so the atom means
  the same thing on both sides of the split. The crossing bites at the
  corpus→live seam and nowhere else. Saying otherwise would overstate it.

---

## RESULT 2 — the live bar list is not a grid, and the live path cannot fire

`omen_reversion.bars_from_samples` buckets ticks at 60 s and drops empty
buckets — correct for price integrity ("a forward-filled bar is a price that
never traded"), and it means the list it returns is indexed by **bar count**
while the gap between consecutive indices is whatever the tape did.

`market_stream`, last 6 h, bucketed at 60 s:

| symbol | ticks | bars | possible | filled | adjacent | max gap | median index step |
|---|---|---|---|---|---|---|---|
| ETH-USDT | 90 | 82 | 352 | 0.233 | 0.210 | 66 | 180 s |
| TIBBIR-USDC | 87 | 78 | 356 | 0.219 | 0.143 | 60 | 180 s |
| VIRTUAL-USDC | 87 | 85 | 351 | 0.242 | 0.274 | 58 | 180 s |
| DRB-USDC | 86 | 76 | 351 | 0.217 | 0.213 | 60 | 180 s |
| ETH-USDC | 85 | 76 | 346 | 0.220 | 0.240 | 60 | 180 s |
| SOL-USDC | 84 | 78 | 353 | 0.221 | 0.247 | 61 | 180 s |
| WETH-USDT | 83 | 76 | 354 | 0.215 | 0.227 | **94** | 180 s |
| AERO-USDC | 82 | 77 | 352 | 0.219 | 0.158 | 60 | 180 s |

**One step of the bar index is a median 180 seconds against a nominal 60** —
three times the declared bar, with a tail of 58 to 94 buckets in a single step.
Only 14.3–27.4% of consecutive list indices are genuinely adjacent in time.

And the consequence that is not about units at all:

| symbol | bars that form in the live window | bars required | fires? |
|---|---|---|---|
| ETH-USDT | 39.6 | 169 | **NO** |
| VIRTUAL-USDC | 41.2 | 169 | **NO** |
| AERO-USDC | 37.2 | 169 | **NO** |
| *(all 8)* | 36.5 – 41.2 | 169 | **NO** |

`omen_reversion` fetches `(LOOKBACK_BARS + 2) * BAR_SECONDS` = **170 minutes**
of ticks and refuses under `LOOKBACK_BARS + 1` = **169 closed bars**. At the
measured density, 37 form. **The omen strategy cannot produce an omen on this
tape for any symbol**, and that is true independently of
`OMEN_STRATEGY_ENABLED=0`. To reach 169 closed bars at 0.22 fill it would need
roughly 12.9 hours of ticks and it asks for 2.8.

A third instance of the same crossing, for the record:
`omen_reversion.py:273` and `:295` set the position's
`omen_horizon_sec = HORIZON_BARS * BAR_SECONDS` = 720 s = 12 minutes — nominal
bar seconds again, where the real index step is 180 s.

---

## RESULT 3 — `temporal` distinctness 1.0 is NOT an index. I checked.

The item's prior was that a stream unique on every sample is a counter or a
timestamp. **It is not.** Per-slot census over 600 samples:

| slot | UP distinct | UP distinctness | DOWN distinct | DOWN distinctness |
|---|---|---|---|---|
| r1 | 31 | 0.0517 | 32 | 0.0533 |
| r2 | 29 | 0.0483 | 30 | 0.0500 |
| r3 | 29 | 0.0483 | 33 | **0.0550** |
| r6 | 30 | 0.0500 | 30 | 0.0500 |
| r12 | 28 | 0.0467 | 31 | 0.0517 |
| r24 | 30 | 0.0500 | 30 | 0.0500 |
| r48 | 29 | 0.0483 | 30 | 0.0500 |
| r168 | 28 | 0.0467 | 31 | 0.0517 |
| flip | 12 | 0.0200 | 14 | 0.0233 |
| streak | 9 | 0.0150 | 11 | 0.0183 |
| acc | 30 | 0.0500 | 32 | 0.0533 |

Max slot distinctness **0.0517 UP / 0.0550 DOWN**. Nothing is an index. The
collection's 1.0 is the **conjunction of eleven honest slots** of roughly 30
values each: 30^11 possible keys against 600 samples, so every sample is unique
*by construction*.

That matters because the two readings have opposite fixes. A counter would be
contamination — delete it. An honest conjunction is a **topology** problem:
there is nothing to purge, and the fabric is handed a near-unique key it can
only memorise, never a reusable part it could bind. It is the same mechanism
the operator named for L1 at 19:07 — "it has seen the INSTANT, never the
MOTIF" — arriving here through eleven slots instead of five.

Collection distinctness for context (600 samples, UP / DOWN):
geometry 0.9783 / 0.9700 · **temporal 1.0000 / 1.0000** · flow 0.2333 / 0.2567
· volatility 0.1600 / 0.1417 · cross 0.4183 / 0.4317 · horizon 0.0017 / 0.0017
· instrument 0.0017 / 0.0017.

---

## What was NOT done, and why, rather than redefining the criteria

* **Criteria 3 and 4 (new pools, explicitly-set pool knobs) — not attempted.**
  Criterion 1 fired and instructs a stop. Adding a time-scale pool or a
  sequence pool while one bar index means 3600 s on one side of the seam and a
  measured 180 s on the other would produce a held-out number that could not be
  attributed.
* **Criterion 5 (back-to-back held-out measurement against the pass-107
  baseline) — not met.** There is no new arm to measure. This pass produced an
  audit and an instrument, not a topology change.
* Note that pools 17/18 (`temporal_sequence`, `temporal_scale`) already exist
  in `omen_brain.META_COLLECTIONS` from `6cadc58`, off behind
  `OMEN_META_COLLECTIONS=0`. Whoever turns them on inherits this seam, which is
  the reason to fix it first.

## The fix this points at, for whoever takes it

Put the cadence in the frame: `hzn h=12 s=3600` versus `hzn h=12 s=60`, so the
two questions are two atoms. It is small in bytes and large in blast radius —
`build_collections` gains a parameter, every caller changes, and **every
existing fabric's horizon atom is invalidated**, so it needs its own pass and
its own back-to-back measurement. Separately, `LOOKBACK_BARS = 168` at
`BAR_SECONDS = 60` is unsatisfiable on the measured tape; the honest options
are a longer fetch window, a coarser live bar, or a shorter lookback, and which
one is a decision about what the strategy IS.
