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
