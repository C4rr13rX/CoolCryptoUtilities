# Held-out measurement — pass 107, Cove

**The result is AT-OR-BELOW BASELINE. No edge is claimed.** That is the normal
outcome here and this document says so plainly.

## What was measured

Corpus `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21926 bars, 3600s
cadence. Horizon 12 bars = **720 minutes**. Round trip cost 0.6500%, omen
threshold 0.9750%.

- train bars `[18501, 21501)` -> 3000 samples, balanced to 2717
- test  bars `[21513, 21913)` -> 400 samples (12-bar purge gap between them)
- node: `127.0.0.1:8091`, brain dir `brain-data-assoc-p106`, the 14-pool
  v3_assoc fabric brought up in pass 106. Production `:8090` untouched
  (uptime 69659s throughout).

Report JSON: `omen-AERO-USDC-h12-20260910-133426.json`.

## The numbers

| measure | value | baseline | verdict |
|---|---|---|---|
| **held-out exact** | **28.5%** (400/400 admitted) | majority class **30.2%** | **BELOW** |
| market up-rate | 47.2% | — | down window |
| train recall | 97.5% (195/200) | — | reproduction, not prediction |
| buy omens | 80 of 400 | — | |
| **net per trade** | **-0.2528%** | every-bar buy **-0.5995%** | both negative |
| buy hit rate | 42.5% | — | |

**Read the money row carefully.** The brain's buys lose 0.2528% per trade.
Buying every bar loses 0.5995%. The gap (+0.347%/trade) is *selection against
a falling tape*, measured in **one down window**, and it is not an edge claim:

1. Exact accuracy is **below** the majority-class baseline. A classifier that
   cannot beat "always say murk" is not predicting.
2. One window is never an edge here. The standing instruction names two fake
   numbers (+0.9067%, 78%) produced by exactly this shape — a long-only rule
   flattered by a single window. An up window is required and was not run.
3. Both arms are negative. "Loses less" is not "pays".

## Two defects the run exposed

### 1. The garbage control is failing

40 frames of pure random bytes were fed in. **17 of 40 (42.5%) came back
`actionable`**, at confidence min 0.579 / med 0.629 / max 0.671.

Real frames answer at confidence med 0.970, so confidence *does* separate
noise (0.63) from signal (0.97) — but the actionable gate does not use that
separation, and calling random bytes tradeable 42.5% of the time is a live
defect in the gate, not a curiosity.

### 2. Confidence is inert as a filter

The confidence sweep is perfectly flat — every threshold from 0.00 to 0.50
admits the identical 80 trades at -0.253%:

```
0.00->80t -0.253% | 0.05->80t -0.253% | 0.10->80t -0.253%
0.20->80t -0.253% | 0.30->80t -0.253% | 0.50->80t -0.253%
```

Minimum real confidence is 0.879, so no threshold at or below 0.5 can bind.
This corroborates the standing note that confidence is worthless as a
correctness gate, and it means a confidence floor cannot be used to buy back
the negative expectancy.

## SELF-CORRECTION: pass 106's relation streams were NOT in this run

**This arm fired 7 collections, not 10.** The measured query set was
`('geometry', 'temporal', 'cross')`. Distinctness over the 2717 training
frames:

```
temporal 1.000  geometry 0.961  cross 0.254
flow 0.104  volatility 0.061  horizon 0.000  instrument 0.000
```

`rel_move_vol`, `rel_shape_flow` and `rel_trend_noise` are **absent from the
table entirely** — not below the floor, not computed.

The cause is `OMEN_RELATION_COLLECTIONS`, the default-off flag shipped in
pass 106 (`trading/omen_brain.py:187`). With it unset, `COLLECTIONS` is the
7-tuple and `build_collections` returns 7 keys. Verified directly:

```
python -X utf8 -c "from trading.omen_brain import COLLECTIONS, build_collections; ..."
-> FRAME KEYS: ['cross','flow','geometry','horizon','instrument','temporal','volatility']
```

The flag is **correct and should stay off by default** — its docstring is
right that sending pool 12 to a v2 node silently converts every training
sample into a miss. The correction is to my pass-106 commit message
(`6706bc3`), which read *"the query set admits two of three"*. That was
measured through a direct call with the relations present. **By default the
query set admits none of three**, because the streams are not built at all.
Anyone re-running the experiment would see no relations and have no idea why.

So the distinctness numbers in `6706bc3` were real but **not reachable by the
experiment path**, and that commit overstated what had moved.

## What was deliberately NOT done

The with-relations arm was not run. Doing it required a fresh 8.7-minute
training pass with `OMEN_RELATION_COLLECTIONS=1`, and the pass clock did not
hold it.

**Retraining relations onto `brain-data-assoc-p106` was rejected rather than
skipped.** That fabric has now been taught 2717 samples of 7-collection
frames. Training 10-collection frames on top would leave one fabric holding
two different frame shapes — a dirty fabric, and the exact condition the
operator's brief names as invalidating a measurement. The clean baseline is
worth more than a rushed second arm.

## The next pass starts here

The baseline arm is measured and the fabric is clean. The comparison needs a
**second fresh brain dir**, not this one:

```powershell
& "D:\Projects\W1z4rDV1510n\start_node.ps1" -Addr 127.0.0.1:8092 `
    -BrainDir "D:\Projects\W1z4rDV1510n\brain-data-assoc-p107-rel" `
    -Identity "brains\market_predictor_v3_assoc.identity.toml" `
    -Deployment "brains\market_predictor_v3_assoc.deployment.toml"
```

```bash
OMEN_RELATION_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=127.0.0.1:8092 \
python -X utf8 scripts/omen_experiment.py \
  --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --train 3000 --test 400 --horizon 12 --seed 7
```

Same corpus, same `--seed 7`, same split, one variable changed. Compare
against the table above. Then repeat both arms on an **up** window before any
number is called an edge.

— Cove, pass 107
