# Which published per-trade numbers were unreadable — pass 424, Sage

Criterion 5 of item `e0ecbda3`, and it completes Cove's pass-123 note
(`READABILITY-AUDIT-pass123-cove.md`), which audited three layer-probe
artifacts and said so. This one covers **the newest published report of all
five harnesses**, and it resolves the one Cove marked UNDETERMINED with a
measurement rather than a base-rate substitution.

Every ceiling below is **measured on that report's own window**, node-free, by
`scripts.omen_experiment.window_label_ceiling` — `label_omen` is a forward
return plus a 60-bar min/max over stored bars, so no brain is involved and no
median is assumed. A machine-readable twin is written to
`data/brain_experiments/READABILITY-AUDIT-pass424-sage.json`, which is
**gitignored like every other `.json` in this directory** — the table below is
the published record, and the reproduction command at the foot regenerates any
row of it.

The rule: a per-trade net is quotable only when BOTH n's clear 30
(`READABLE_LABEL_FLOOR`) — the window's **label ceiling** (how many bars it
labels `trough`, the cap on correct buys at perfect recall) and the **call
count**. They have different fixes — bars vs recall — so the table names which
failed.

## The table

| harness | newest report | held-out bars | trough ceiling (measured) | calls | published per-trade | verdict |
|---|---|---|---|---|---|---|
| `omen_agreement_census` | `agreement-AERO-USDC-h12-UP-UP-20260910-192825.json` arm ALL | 100 | **4** (4.00%) | 3 | **+1.1236%** | **UNREADABLE — ceiling AND calls** |
| `omen_agreement_census` | same file, arm AGREE | 100 (22 admitted) | **4** | 0 | 0.0 | **NOT A NUMBER** — zero calls printed as a per-trade net |
| `omen_shape_mutations` | `p120-jet-arm-shapequeried-UP.json` | 400 | **44** (11.00%) | 85 | +0.5441% | **READABLE** — both n's clear 30 |
| `omen_shape_mutations` | `p120-jet-arm-shapequeried-DOWN.json` | 400 | **39** (9.75%) | 76 | -3.0897% | **READABLE** |
| `omen_layer_probe` | `layer-down-heldout.json` | 168 | **24** (14.29%) | 109 | -3.1416% | **UNREADABLE — ceiling** (Cove's UNDETERMINED, resolved) |
| `omen_scorable_windows` | `SCORABLE-WINDOWS-pass113-cove.json` | 60 | — | **max 29 across 102 rows** | 19 rows quoted a `called_net`, 83 quoted 0.0 on zero calls | **UNREADABLE — calls, all 102 rows** |
| `omen_query_path_probe` | `QUERY-PATH-selfpools-pass111-gale.json` | 60 | — | — | **none published** | nothing to refuse |

## The three things this table settles

**1. The census positive is three calls in a window that could hold four.**
`+1.1236%` is the only positive per-trade number the census ever published, and
it is one arm's three buys. The window's measured trough ceiling is **4** — not
the 14 the pass-114 median (14.46%) would have predicted for 100 bars. Assuming
the median would have overstated this window's ceiling by 3.5x, which is why
the guard measures instead. The two older census artifacts
(`...DOWN-DOWN-20260910-192725/192754.json`, both window `[21703, 21803]`) hold
**18** troughs in 100 bars and made 15 and 17 calls: also below the floor on
both counts, both negative (-1.1898%, -1.4747%). Same corpus, same 100 bars,
ceilings of 4 and 18 — the base rate is not a corpus constant, and a window's
ceiling cannot be inferred from its neighbours.

**2. `layer-down-heldout.json` is refused on the CEILING, and its corpus is
now identified.** The artifact records no `corpus` field, which is why Cove
could not settle it. It does record `train_window [357, 707]` and
`train_base_trough 0.111428…` = **39 of 350** troughs, and that is a
fingerprint. Measured over the candidates: `p108_aero_down.json` gives exactly
**39**, `p108_aero_up.json` gives 44, `0004_AERO-USDC.json` gives 54. The match
is unique. Its held-out window `[719, 887]` on that corpus holds **24** troughs
— below 30, so the number is refused on the window, not on recall. It made
**109** calls in 168 bars: it called a buy on 64.9% of bars in a window that
contains 24 correct ones, so `-3.1416%` is mostly the cost of calling
everything, not a measurement of a rule.

**3. Only the shape-mutation arms survive the guard — and what they say is
negative.** Both `p120-jet-arm-shapequeried-*` windows are 400 bars, above the
208-bar `MIN_READABLE_HELDOUT_BARS_AT_3600S`, with 44/39 troughs and 85/76
calls. Nothing in this item's machinery refuses them. Read them, then: on the
UP window the rule's `buy_net_per_trade` is **+0.5441%** while
`every_bar_net_per_trade` — buying every single bar — is **+1.5952%**. The
called subset earns about a third of what indiscriminate buying earned on the
same window. The one readable per-trade pair in the whole census shows the
selection making things worse, and it was published beside four unreadable
positives that made the family look better than it is.

## What is NOT claimed

Readable is not true. The two shape arms clear the floor; that says a number of
that size could have been produced by that window, not that it generalises. And
`heldout_default_bars` in the guard is a floor at perfect recall — a window that
clears it can still be refused on calls, which is exactly what happens to
`layer-up-heldout.json` (9 calls) and `p119-gale-l2max-up-heldout.json` (7
calls) in Cove's table.

## Proof

    python -X utf8 -m pytest tests/test_a_default_window_that_held_nine_troughs_cannot_quote_a_per_trade_net.py -q

Every ceiling in the table is reproducible in one command against the report's
own recorded `corpus` and `test_window`, e.g. the census row:

    python -X utf8 -c "import sys; sys.path.insert(0,'.'); \
      from pathlib import Path; \
      from scripts.omen_experiment import load_bars, window_label_ceiling; \
      print(window_label_ceiling(load_bars(Path('data/historical_ohlcv/base/0004_AERO-USDC.json')), 21813, 21913, 12))"
    # {'scored_bars': 100, 'trough': 4, 'crest': 11, 'trough_base_rate': 0.04, 'readable': False}
