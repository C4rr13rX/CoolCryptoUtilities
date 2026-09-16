# Which published per-trade numbers were unreadable — pass 123, Cove

PARTIAL AND SAID SO. This note covers the three harnesses whose guard shipped
this pass (`omen_agreement_census`, `omen_layer_probe`, `omen_scorable_windows`).
`omen_query_path_probe` and `omen_shape_mutations` are NOT yet wired and their
published reports are NOT audited here.

## The rule being applied

A per-trade net is quotable only when BOTH n's clear the 30 floor
(`trading.omen_scoreboard.READABLE_TRADES`, imported by
`scripts/omen_experiment.py` as `READABLE_LABEL_FLOOR`):

* **the window's label ceiling** — how many bars the held-out window actually
  labels `trough`. That is the ceiling on CORRECT buy calls at perfect recall.
* **the call count** — how many buys the rule actually made.

They have different fixes (bars vs recall), so the audit names which one failed.

## The artifacts

| report | held-out bars | calls | published per-trade | verdict | ceiling |
|---|---|---|---|---|---|
| `layer-up-heldout.json` | 168 | 9 | +2.9030% | **UNREADABLE — call count** | not recorded by the artifact |
| `p119-gale-l2max-up-heldout.json` | 168 | 7 | -0.2235% | **UNREADABLE — call count** | not recorded by the artifact |
| `layer-down-heldout.json` | 168 | 109 | -3.1416% | **UNDETERMINED** | not recorded by the artifact |

Two of the three are refused on the call count alone, and that needs no
re-measurement: 9 calls and 7 calls are below 30, so `+2.9030%` — the only
POSITIVE per-trade number among them — is one number wearing a percentage sign.

The third cannot be settled from the file. `layer-down-heldout.json` clears the
call floor at 109, and its label ceiling is **not in the artifact**: every
report in this table was written before `heldout_label_counts` existed, so the
count of troughs in those 168 bars was never recorded. Saying "168 bars at the
14.46% pass-114 median gives 24" would be quoting a different corpus's base
rate as this one's measurement, which is the exact substitution this whole item
exists to stop. It is marked UNDETERMINED until the arm is re-run under the
guard, which now writes the census.

What 168 bars DOES settle, without assuming any base rate: it is below the
`MIN_READABLE_HELDOUT_BARS_AT_3600S` = 208 bars that 30 troughs needs at the
median rate AT PERFECT RECALL. So the window was under-sized on its own terms
whatever its true trough count was.

## What changed in the code this pass

* `scripts/omen_agreement_census.py` — `--test` 200 → 208
  (`MIN_READABLE_HELDOUT_BARS_AT_3600S`, not a copied literal), `build_parser`
  split out so the default is testable without a node, `heldout_readability`
  spliced and `validate_report_readability` called at write time.
* `scripts/omen_layer_probe.py` — `--test` 180 → 208; `guard_edge_for_report`
  existed but was **never called**, and it referenced `heldout_readability`
  without importing it, so the first call would have been a `NameError`. Both
  fixed: the import is now explicit and the guard runs on both the L1 and the
  shipped-L2 arm before either JSON file is written. An `unspent` arm now
  carries its window's label census too, because an absent ceiling reads as
  "not applicable" when what is true is "this window held nine troughs either
  way".
* `scripts/omen_scorable_windows.py` — `--test` 60 → 208. Measured node-free on
  0004_AERO-USDC: 656 of 656 candidate 60-bar windows hold fewer than 30 trough
  labels, median 9. Each row's `called_net` is now guarded against **its own**
  window (the float survives as `called_net_raw`, which the pooled arithmetic
  reads), and the pooled UP/DOWN/BOTH cells carry a pooled ceiling that is
  additive across the windows pooled — 17 windows of 9 troughs genuinely can
  produce 153 correct calls, so the pooled cell is NOT refused on the same
  grounds as the per-corpus one, and the block records the geometry it pooled
  rather than hiding it.

## Proof

    python -X utf8 -m pytest tests/test_a_default_window_that_held_nine_troughs_cannot_quote_a_per_trade_net.py -q
    # 8 passed

The first two cases fail against the old behaviour: `score()` returned
`total / len(trades)` unconditionally, and the sweep wrote
`round(edge["called_net"], 6)` into every row.
