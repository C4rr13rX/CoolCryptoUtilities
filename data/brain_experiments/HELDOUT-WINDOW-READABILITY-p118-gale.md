# The 60-bar held-out window cannot carry a per-trade net, measured on the corpus itself

Pass 118, Gale, item [3366105d]. **Node-free** — nothing here queried a brain,
so none of it is subject to the run-to-run fabric variance that makes two
numbers measured 34 minutes apart incomparable.

## Corpus, windows and cost

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21,926 bars |
| cadence | 3600s, measured (`bar_seconds`) |
| horizon | 720 minutes = 12 bars of 3600s (the shipped default, asked in minutes) |
| round trip | 0.6500% |
| omen threshold | 0.9750% (`OMEN_COST_MULTIPLE` 1.5) |
| labeller | `trading.omen_brain.label_omen`, unchanged |
| windows | every candidate held-out window `--list-windows` offers, stepping `--test // 2` back from the end of the corpus |

## The number

`--test 60` and `--test 400`, same corpus, same horizon, same threshold,
same labeller. The `trough` column is the count of trough labels the window
**contains** — the ceiling on buy omens at *perfect* recall, before the brain
is asked anything.

| held-out window | candidates | readable (>= 30 trough labels) | unreadable | trough labels min / median / max |
|---|---|---|---|---|
| **60 bars** | 656 | **0** | **656** | 0 / **9** / 25 |
| **400 bars** | 97 | **97** | **0** | 32 / **60** / 84 |

Not one 60-bar window in this corpus — not one of 656 — contains 30 trough
labels. The median holds **9**, which lands on the 8.7 predicted from pass
114's 14.46% median base rate across 150 eligible 3600s corpora. The worst
holds **zero**: window `[21703, 21763)` is a textbook DOWN window at 11.7%
up-rate and −1.5636% mean forward, and it has **no buy-low label in it at
all**. Its regime column reads perfectly usable. Its ceiling is nothing.

At the raised default of 400 bars every candidate clears the floor, with a
median of 60 trough labels — twice the floor, which is the margin that lets a
corpus whose trough rate is a third of the median still be readable.

## What this says about the numbers already in this directory

Pass 111's four arms reported `buy_omens` of 1, 9, 7 and 2 out of 60 held-out
bars. The median 60-bar window here contains 9 trough labels. **Every one of
those four counts sits inside the ceiling its own window imposed**, so not one
of them was evidence about the brain — a brain with perfect recall and a brain
with none produce counts in the same range when the window holds 9 labels. The
UP-base cell's `buy_net_per_trade` +0.0031% at `buy_hit_rate` 1.0, the only
positive cell in that table and therefore the one that got quoted, is **one
trade**.

Lowering the omen threshold to manufacture labels is the trap, and it is
already measured: `OMEN_COST_MULTIPLE` 0.5 raises the trough rate to 18.57%,
which is 1.28x the labels for a label that no longer means the forward move
paid its own round trip. The window is what gets raised. The threshold is not.

## What shipped

`scripts/omen_experiment.py`:

* **The ceiling is measured on the window and written into the report.**
  `heldout_readability()` counts the trough and crest labels the held-out
  window actually contains and emits `heldout_trough_labels`,
  `heldout_trough_base_rate`, `heldout_crest_labels`,
  `heldout_crest_base_rate`, `heldout_label_counts` and
  `heldout_window_bars`. Measured on *this* window, never assumed from the
  pass-114 median: that median is a fact about 3600s corpora, and a 600s
  corpus at the same wall-clock horizon labels a different share of its bars.
* **A cell that cannot be read says so instead of showing a percentage.**
  `buy_net_per_trade` and `sell_net_per_trade` are the string `UNREADABLE`
  whenever the window's label ceiling or the call count is below 30; the raw
  value moves to `net_per_trade_raw`, kept for reproducibility, named so it is
  not quoted by accident. The two reasons are recorded separately because they
  send you to different fixes — too few **labels** means the window is too
  short and the fix is bars; too few **calls** in a window with plenty of
  labels means recall is the problem and the fix is the brain.
* **The sell cell exists and says it was never scored.** This harness is
  long-only, so the crest cell carries its label ceiling and the explicit
  reason `NOT SCORED`. An omitted cell reads "not applicable"; the truth is
  "never measured".
* **The floor is imported, not restated.** `READABLE_LABEL_FLOOR` is
  `trading.omen_scoreboard.READABLE_TRADES`, so the label floor and the trade
  floor cannot drift apart in a later edit.
* **The default is 400 bars and it is named in the report.**
  `heldout_default_bars` and `heldout_window_was_default` travel with every
  run, because a default that lives only in an argparse line cannot be
  compared against the window a report actually used.
  `MIN_READABLE_HELDOUT_BARS_AT_3600S` = 208 records the arithmetic:
  30 / 0.1446 = 207.5 bars at perfect recall.
* **A write-time guard, the sibling of `validate_report_horizon`.**
  `validate_report_readability()` refuses to write a report that is missing
  the ceiling fields, that omits the sell cell, or that holds a bare float
  under a guarded key while its own cell is unreadable. Splice order in
  `main` is what puts the guarded value there today, and splice order is
  exactly what a later edit reorders without noticing.
* **`--list-windows` shows the ceiling next to the regime.** That census is
  where a window size is chosen, and until now it showed a window's DIRECTION
  and nothing about whether it could carry a number, so every candidate looked
  equally usable. It now prints the trough count, the base rate and a
  readable yes/NO per candidate, and a footer naming how many candidates
  cannot carry a readable net.

## Commands that prove it

```
# the two censuses in the table above (node-free, ~40s each)
python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
    --train 2000 --test 60  --list-windows
python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
    --train 2000 --test 400 --list-windows

# the behaviour, including the pass-111 regression cell
python -X utf8 -m pytest tests/test_a_sixty_bar_window_cannot_carry_a_per_trade_net.py -q
```

## What is NOT proved here

No report was written end to end, because there is no omen node on :8091 this
pass and standing instructions forbid training against production's fabric on
:8090. The report *fields* are proved by the unit tests over
`heldout_readability`, and the report *write* is proved by
`validate_report_readability`, which runs at write time beside
`validate_report_horizon` and refuses the old shape. The first node-backed run
after this will produce the first report carrying a ceiling.

## What to try next

The same floor applies to the other harnesses and they are further from it
than `omen_experiment` was: `omen_scorable_windows.py` defaults `--test` to
**60**, `omen_query_path_probe.py` and `omen_shape_mutations.py` to **120**,
`omen_layer_probe.py` to **180**, `omen_agreement_census.py` to **200**. At
this corpus's median that is 9, 17, 26 and 29 trough labels — all four below
the floor, and the 200 is the one that will look closest to fine while still
being unreadable.
