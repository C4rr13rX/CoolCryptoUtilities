# The 60-bar held-out window holds 8.7 troughs, so the money scoreboard was unreadable by construction

Jet, pass 114, 2026-09-11. Item [0795cf0b]. **No node was contacted** — every
number here is computed from the bars and the labelling rule alone, so :8090
and :8091 were untouched and none of it is subject to node run-to-run variance.

## Corpus, windows, threshold

* Corpus root `data/historical_ohlcv`, **629 files**, read with `rglob` — the
  files are nested one level by chain and a top-level `glob` finds zero.
* **150 eligible corpora**, filtered to head cadence **exactly 3600s** so the
  horizon is one question, and to `omen_experiment.MAX_CORPUS_BAR_SECONDS`.
* Horizon **720 minutes = 12 bars** at that cadence.
* Round trip cost **0.6500%**; shipped `OMEN_COST_MULTIPLE` 1.5, so the omen
  threshold is **0.9750%**.

## RESULT 1 — the ceiling on buy omens, and why pass 111's counts were not shyness

Trough/crest **label** base rates, median over the 150 corpora:

| cost multiple | threshold | trough rate | crest rate | bars for 30 troughs | for 30 crests |
|---|---|---|---|---|---|
| 0.5 | 0.3250% | 18.57% | 17.81% | 162 | 168 |
| 1.0 | 0.6500% | 16.35% | 15.92% | 184 | 188 |
| **1.5 (shipped)** | **0.9750%** | **14.46%** | **14.11%** | **208** | **213** |
| 2.0 | 1.3000% | 12.57% | 12.36% | 239 | 243 |

At the shipped threshold a **60-bar held-out window contains 8.7 trough labels
in total**. Pass 111's `buy_omens` of 1, 9, 7 and 2 all sit inside that
ceiling. The brain was not pathologically shy; the window was too small to
carry a readable per-trade net whatever the topology. A caller with recall `r`
calls at most `8.7 x r` of them.

## RESULT 2 — lowering the threshold is NOT the fix, and it costs meaning

Halving the cost multiple to 0.5 moves the trough rate 14.46% -> 18.57%. That
is **1.28x the omens** in exchange for a label that no longer means the forward
move paid its own round trip. The lever is **window size**, not threshold:
~208 bars for 30 trough calls at perfect recall, and strictly more at any real
recall. Anyone sizing a held-out window at 60 bars is buying an unreadable
cell.

## RESULT 3 — the ceiling scoreboard, both windows, and the half nobody scores

TRUE labels used as the caller over the last **208 bars** of each corpus —
perfect recall and perfect precision by construction. Windows split UP/DOWN by
their own drift, never pooled.

| window | corpora | buy (trough) | sell (crest) | every bar |
|---|---|---|---|---|
| UP | 91 | **n=2176, +3.0316%/trade** | **n=2358, +2.4000%/trade** | n=18928, +0.1062%/trade |
| DOWN | 59 | **n=1506, +1.6396%/trade** | **n=1406, +1.8884%/trade** | n=12272, -0.9590%/trade |

**Precision is 100% in every cell BY CONSTRUCTION** — the caller is the labels.
That is the definition of a ceiling and it is not a result. State it that way
or the table reads as an edge.

What the table does carry:

1. **The headroom is large and it exists in both windows.** A perfect caller
   earns +3.03%/trade against a +0.11% every-bar in UP, and +1.64% against a
   **-0.96%** every-bar in DOWN. The target is worth aiming at, which was not
   previously established — a ceiling below the baseline would have killed the
   whole premise without any topology work.
2. **The sell half is comparable to the buy half, and LARGER in DOWN**
   (+1.8884% against +1.6396%). Every omen report in this repo scores buying
   only. On this evidence the unscored half is not a rounding error; in a down
   window it is the bigger of the two.
3. **n is now in four figures per cell**, against 1, 9, 7 and 2. The threshold
   that achieved it is unchanged at the shipped 0.9750% — what changed is the
   window (60 -> 208 bars) and the corpus count (1 -> 150).

## What shipped

* `trading/omen_scoreboard.py` — `money_scoreboard()` scores both halves
  against forward returns at one cost and one horizon, and `render_scoreboard()`
  has **no code path that emits a per-trade percentage without an `n=` on the
  same line**. A cell under 30 trades renders `UNREADABLE` with its n instead
  of as a number; an empty cell renders `NO CALLS` rather than a break-even
  `0.0000%` (the old `max(1, n)` divisor reported an empty cell as a measured
  zero).
* `scripts/omen_trough_census.py` — reproduces everything above.
* `tests/test_a_per_trade_net_is_never_printed_without_its_n.py` — 7 tests
  pinning the presentation contract and the sell-half arithmetic.

## Reproduce

```
python -X utf8 scripts/omen_trough_census.py --cadence 3600 --limit 150 --report
python -X utf8 scripts/omen_trough_census.py --cadence 3600 --ceiling-bars 208 --limit 150 --report
```

The `--report` JSON artifacts (`TROUGH-BASE-RATE-*.json`,
`CEILING-SCOREBOARD-*.json`) are written beside this file but are **not
tracked**: `.gitignore:123` ignores `data/brain_experiments/*` and re-includes
only `*.md` and `*cadence_census*.json`. Both commands above regenerate them in
85s and 90s respectively.

## Loose end found, not fixed, filed instead

`omen_experiment.load_bars` raises `AttributeError: 'str' object has no
attribute 'get'` on **62 of the 629 corpus files**, so any sweep over the whole
corpus dies rather than skipping. 27 are genuine empty stubs (`["none", []]`);
the other **35 carry 714,673 real bars** in a `["coinbase", [ ...bars... ]]`
(source, bars) pair shape that the loader does not unwrap. That is Cove's
714,673-bar finding, and this is the exact mechanism and the 35/27 split. Not
fixed here — that loader was held by another agent this pass. The census above
skips those files and counts them.
