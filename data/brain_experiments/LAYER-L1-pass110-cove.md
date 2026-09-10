# L1 co-occurrence: the layer abstracts, and its buy-low lift does NOT survive out of sample

**Pass 110 — Cove — 2026-09-10**

## The one-line result

The L1 motif layer passes the falsification test the operator asked for
(distinctness falls hard, L0 0.544 → L1 0.014) and carries strong **in-sample**
buy-low information (2.69x trough lift, UP; 2.25x, DOWN). **That lift does not
generalise.** Held out, L1 is at or below buy-every-bar in both windows.

This is a negative result and it is a finished measurement. It was obtained
with **no node and no fabric**, before anyone spent a training run on a pool
that does not yet exist.

## Corpus, windows, cost

| | |
|---|---|
| corpus | `data/brain_experiments/p108_aero_up.json`, `p108_aero_down.json` |
| bars | 900 each, AERO-USDC, base chain |
| horizon | 12 bars |
| distinctness sample | 719 frame sets, bars [168, 887) |
| held-out fit window | bars [357, 707), 350 samples — **fit here only** |
| held-out test window | bars [719, 887), 168 samples, 12-bar purge between |
| round-trip cost | 0.6500% (`ROUND_TRIP_COST`) |

## 1. The falsification test — L1 ABSTRACTS

The rule in `trading/omen_layers.py`: a higher layer earns its place only when
its vocabulary is smaller than its input's. Measured:

| layer | UP | DOWN |
|---|---|---|
| L0 mean of the 5 input streams | 0.5441 | 0.5455 |
| **L1 co-occurrence** | **0.0139** | **0.0139** |
| L2 motif sequence (4 steps) | 0.2017 | 0.2559 |

L1 is a **10-motif vocabulary over 719 samples**, identical in size in both
windows. Distinctness falls by a factor of 39. The layer is not cut.

L2 at 0.20–0.26 sits just under the 0.30 identifier guard — it abstracts, but
it is close enough to the line that lengthening `MOTIF_SEQUENCE_STEPS` should
not be done without re-running this.

## 2. Why the dilution law's 0.20 floor is the WRONG test here

`MIN_QUERY_DISTINCTNESS` is 0.20 and L1 reads 0.0139, so the dilution law would
auto-cut L1 from any query set. **That inference is invalid for an abstraction
layer.** The law was measured on *sensory* streams, where low distinctness
means constant means uninformative. A motif layer is low-distinctness *by
design* — the shrinkage IS the abstraction. Applying the sensory floor cuts
every layer before it is measured.

The right test is whether the coarse frame's **label distribution is skewed**.
`label_skew` in the probe measures exactly that, and it is what produced the
in-sample numbers below.

## 3. In-sample: the motif carries the outcome, strongly

Trough (= buy-low) rate per motif, groups with n ≥ 20:

**UP window** — corpus trough rate 13.6%, groups cover 94% of samples

| n | share | trough | lift | top label | motif |
|---:|---:|---:|---:|---|---|
| 101 | 14.0% | 36.6% | **2.69x** | slide | `geo=mid tem=lo flo=mid vol=hi cro=hi` |
| 219 | 30.5% | 19.2% | 1.41x | slide | `geo=mid tem=lo flo=mid vol=hi cro=lo` |
| 234 | 32.5% | 3.4% | 0.25x | climb | `geo=mid tem=hi flo=mid vol=hi cro=hi` |
| 99 | 13.8% | 3.0% | 0.22x | climb | `geo=mid tem=hi flo=mid vol=hi cro=lo` |
| 22 | 3.1% | 0.0% | 0.00x | climb | `geo=mid tem=hi flo=hi vol=hi cro=hi` |

**DOWN window** — corpus trough rate 11.1%, groups cover 95% of samples

| n | share | trough | lift | top label | motif |
|---:|---:|---:|---:|---|---|
| 40 | 5.6% | 25.0% | **2.25x** | slide | `geo=mid tem=lo flo=lo vol=hi cro=lo` |
| 22 | 3.1% | 22.7% | 2.04x | slide | `geo=mid tem=lo flo=hi vol=hi cro=lo` |
| 378 | 52.6% | 15.6% | 1.40x | slide | `geo=mid tem=lo flo=mid vol=hi cro=lo` |
| 65 | 9.0% | 7.7% | 0.69x | murk | `geo=mid tem=lo flo=mid vol=hi cro=hi` |
| 95 | 13.2% | 0.0% | 0.00x | climb | `geo=mid tem=hi flo=mid vol=hi cro=lo` |
| 86 | 12.0% | 0.0% | 0.00x | crest | `geo=mid tem=hi flo=mid vol=hi cro=hi` |

The **sign agrees across both windows**: `tem=lo` motifs hold the troughs,
`tem=hi` motifs hold almost none (0.0–3.0% trough across 400+ samples in the
two windows combined).

## 4. Held out: the lift does NOT survive

Fit the motif → trough map on the train window, freeze it, apply it to the
held-out window. Called = bars whose train lift ≥ 1.3. Scored on the
operator's order: per-trade net first, trough precision second.

| | UP | DOWN |
|---|---:|---:|
| motifs called buyable | 1 | 2 |
| called trades | 12 (7.1% of window) | 109 (64.9%) |
| **(a) per-trade net, called** | **+3.2476%** | **−3.1416%** |
| buy-every-bar baseline | +3.8377% | −2.9288% |
| **EDGE** | **−0.5901%** | **−0.2128%** |
| **(b) trough precision, called** | **0.0%** | **17.4%** |
| window base trough rate | 7.1% | 14.3% |

**Both windows are at or below baseline.** Read honestly:

- The UP cell is **12 trades and cannot be ranked**. It is reported, not
  concluded from.
- The DOWN cell has real support (109 trades) and is the informative one.
  Trough precision does improve, 14.3% → 17.4% (+3.1pp), so the motif carries
  *something*. But it calls **64.9% of the window** — it is barely a filter —
  and its per-trade net is 0.21pp **worse** than buying every bar.

A 2.25–2.69x in-sample lift collapsing to a +3.1pp precision bump with a
negative net is the signature of a map fitted to the train window's regime
rather than to a repeatable relation.

## 4b. The SELL-HIGH half, measured here for the first time

`omen_experiment.py:552` opens a position only on a buy-low omen because the
live lane is long-only — so a crest is an abstention and **its accuracy is
scored nowhere**. That is half the labelled vocabulary going unjudged. A crest
that correctly calls a fall is worth money as an **exit on a held position**,
so it is scored against forward returns, not by shorting.

No cost is charged on a crest call: exiting a position you already hold does
not open a round trip, and billing one would be the "round trip billed twice
to one leg" shape the profit-logic audit exists to catch.

| | UP | DOWN |
|---|---:|---:|
| sellable motifs (train crest lift ≥ 1.3) | 2 | 2 |
| crest calls | 106 (63.1% of window) | 28 (16.7%) |
| **crest fall precision** | **23.6%** | **89.3%** |
| window base fall rate | 28.0% | 73.8% |
| delta | **−4.4pp** | **+15.5pp** |
| mean forward on called | +5.4421% | −3.1547% |
| mean forward, window | +4.4877% | −2.2788% |

**This is asymmetric and it therefore fails the standing both-windows rule.**
In the DOWN window the crest signal is genuinely good: +15.5pp precision, and
the bars it calls fall *harder* than the window average (−3.15% vs −2.28%),
which is exactly what an exit signal should do. In the UP window it is worse
than the base rate, and it calls 63.1% of bars, which is not a signal.

So: **no edge is claimed.** A rule that only works when the market is already
falling is the mirror image of the long-only rule that flatters itself in an
up window, and this repo has already paid for that error twice. What is worth
recording is that the sell-high half is *more* informative than the buy-low
half on the same motifs and the same windows — the first evidence here that
the exit side may be the better place to spend effort than the entry side.

## 5. A defect in the layer worth one look

Three of the five streams are **near-constant inside the motif**: `geo=mid` on
~100% of samples, `vol=hi` on ~100%, `flo=mid` on ~90%. So L1 is effectively a
`tem × cro` recode — a 4-value alphabet wearing a 5-stream name, and its lift
is plausibly a coarse mean-reversion signal rather than a co-occurrence one.
`_band_of` is saturating on three of its five inputs.

That is the first thing to fix if L1 is revisited, and it is testable without
a node: a band function that does not saturate should raise the motif count
above 10 and change the lift table. I did **not** edit
`trading/omen_layers.py` — the claim was held by another agent this pass.

## 6. What this does and does not settle

**Settles:** the L1 layer abstracts, so it is not cut on the distinctness rule;
and the sensory dilution floor must not be used to judge a layer.

**Settles:** L1's buy-low lift, on this corpus and these two windows, does not
generalise. A fabric cannot extract from a stream what the stream does not
contain out of sample, so an L0-vs-L0+L1 node arm on this corpus is unlikely
to pay for itself and should not be the next thing anyone runs.

**Does NOT settle:** whether L1 helps the *fabric* as a training-side stream
even though it fails as a standalone rule. Training binds every collection and
only the query set is subject to dilution — a stream can be worth binding and
not worth firing. That remains untested.

**Blocked, and named exactly:** the node arm cannot be run at all yet. The
highest pool id in `market_predictor_v4_meta.identity.toml` is **19**. L1 needs
a pool 20, which does not exist in any identity file. Sending pool 20 to the
current node returns `unknown input pool id 20` and `_consolidate` reports
every sample as a MISS — which does not degrade training, it **silently stops**
it. A new identity in `W1z4rDV1510n` is a prerequisite, not a detail.

## Reproduce

```
python -X utf8 scripts/omen_layer_probe.py \
  --corpus data/brain_experiments/p108_aero_up.json --horizon 12 \
  --heldout --json-out data/brain_experiments/layer-up.json

python -X utf8 scripts/omen_layer_probe.py \
  --corpus data/brain_experiments/p108_aero_down.json --horizon 12 \
  --heldout --json-out data/brain_experiments/layer-down.json
```

The probe exits **nonzero** when a layer fails to abstract, so it gates a pass
rather than merely informing it. Guard tests:

```
python -X utf8 -m pytest tests/test_a_motif_layer_that_abstracts_nothing_is_cut.py -q
```

---

## 7. ADDENDUM — the encoder was blind, and fixing it does not rescue the edge

Gale independently ran the falsification on AERO 0004 (3000 bars) and produced
the per-stream census that explains section 5's caveat: **`_band_of` never saw
a `q` token.** Geometry frames are all quantile buckets (`geo p24=q5 body=q17
uw=q0`), the band function counted only `u`/`d`/`r` prefixes, so every geometry
frame tied 0–0 and returned `mid` on **600 of 600 bars**. Volatility read `hi`
on 600/600 for a different reason (three `u` tokens every bar by
construction). A 5-slot motif had **2 live slots**, which is exactly what a
13-motif vocabulary predicts.

Fixed the `q` blindness only — one change, quantile 0–19 split into thirds
(`q<=6` lo, `q>=13` hi) — and re-measured back-to-back on the same corpus:

| | before | after (UP) | after (DOWN) |
|---|---:|---:|---:|
| L1 vocabulary | 10 motifs | **21** | **25** |
| L1 distinctness | 0.0139 | 0.0292 | 0.0348 |
| L2 distinctness | 0.2017 / 0.2559 | **0.4520** | **0.4159** |

**The encoder fix is real and the held-out edge still does not appear:**

| | UP | DOWN |
|---|---:|---:|
| per-trade net edge | **−0.9346%** (was −0.5901%) | **−0.2128%** (unchanged) |
| trough precision | 0.0% vs 7.1% base | 17.4% vs 14.3% base |
| crest fall precision | 44.4% on 9 calls | 94.1% on 17 calls |

Both crest cells are now **below the 20-call support floor**, so neither is
rankable — the sharper encoder made the sell-high signal more selective and
too small to judge at this corpus size. The DOWN crest number stays
interesting and stays unproven.

**A consequence that must not be missed: L2 now FAILS the identifier guard.**
At 0.4520 distinct per sample a 4-step motif path is approaching a
near-unique key — the trap that maximises train recall and destroys
generalisation, and the exact reason `SEQUENCE_STEPS` went from 8 to 5 in
`omen_metacognition`. The probe exits **nonzero** and says so. `MOTIF_SEQUENCE_STEPS`
must come down from 4 before anything trains on L2. I did not change it in
this pass — that is a second change and this pass already made one.

**Still not fixed, and named:** volatility saturates at `hi` on ~100% of bars
because its frame carries three `u` tokens every bar. Banding each stream
against *its own* distribution rather than a fixed letter rule is the real
repair, and it is Gale's suggestion, not mine.

### Reproduce the addendum

```
python -X utf8 scripts/omen_layer_probe.py \
  --corpus data/brain_experiments/p108_aero_up.json --horizon 12 --heldout
# exits 1: L2 DOES NOT ABSTRACT (0.4520 against 0.3000)
```

## 8. MOTIF_SEQUENCE_STEPS swept, and set to 3

The addendum left L2 failing its own guard. Swept the step count on both
windows, 719 samples each, **after** the encoder fix — which matters, because
the same sweep against the blind encoder read 0.177 at 4 steps and would have
justified keeping it:

| steps | L2 distinctness UP | DOWN | verdict |
|---:|---:|---:|---|
| 2 | 0.1266 | 0.1530 | comfortably under the guard |
| **3** | **0.2976** | **0.2962** | **passes, margin 0.8%** |
| 4 | 0.4520 | 0.4159 | FAILS the 0.30 identifier guard |

Set to **3**: it passes and carries more order than 2. **The margin is thin
and that is not a rounding detail** — a corpus with a richer L1 vocabulary
will push 3 over the line too, and the failure mode is the expensive one. The
probe now exits 0 on both windows.

Anyone taking this further should re-run the probe on their corpus and drop to
2 rather than argue with the number.
