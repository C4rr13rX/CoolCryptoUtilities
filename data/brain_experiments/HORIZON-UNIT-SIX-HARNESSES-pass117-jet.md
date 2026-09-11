# Six omen harnesses asked in bars; asked in minutes, the label audit's headline doubles

Pass 117, Jet. Item [0c022bd0]. Commit `60e624f`.

## The defect

Pass 114 taught `scripts/omen_experiment.py` to take a horizon in MINUTES and
convert it with each corpus's own measured cadence. It is ONE of nine harnesses
that write to `data/brain_experiments/`. The other eight still took a bar COUNT
and still defaulted to 12, so the fix covered one report in nine and every other
report remained uncomparable with every other.

Cadence census over `data/historical_ohlcv`, files with 2000+ bars, measured
this pass: **73s to 12036s**. 449 files at 3600s, 36 at 300s, 19 at 600s, and a
long tail. `--horizon 12` is a 15-minute question on one of these files and a
40-hour question on another, and both reports wrote `h12`.

## The measurement

`scripts/omen_label_audit.py` over two REAL corpora at two cadences —
`data/historical_ohlcv/arbitrum/0001_WETH-USDT.json` (300s, 25919 bars) and
`data/historical_ohlcv/arbitrum/0000_ARB-WBTC.json` (3600s, 26270 bars). Same
files, same `--cost`, same take/stop, back to back.

| | `--horizon 12` (bars) | `720` minutes (per-corpus) |
|---|---|---|
| bar count, 300s file | 12 (= 60 min) | 144 (= 720 min) |
| bar count, 3600s file | 12 (= 720 min) | 12 (= 720 min) |
| bars walked | 52163 | 52031 |
| **murk bars the path actually WON** | **17.17%** | **40.00%** |
| endpoint buy-labelled bars | 10359 | 15016 |
| every-bar-long win rate | 30.4% | 44.1% |
| endpoint oracle, per trade | +2.4868% | +2.4624% |
| path oracle, per trade | +0.3250% | +0.3250% |
| path − endpoint | −2.1618% | −2.1374% |

MISSED TRADES — the share of bars the endpoint label refuses that actually
completed a winning round trip — is this harness's headline, and it **more than
doubled**, 17.17% → 40.00%, purely by asking both corpora the same wall-clock
question. Every missed-trades figure in a prior multi-corpus label audit is a
pooled artifact of two different horizons under one name.

The two *oracle per-trade* numbers barely move, and that is worth saying
plainly: the direction of this harness's conclusion (the path target's ceiling
is below the endpoint target's, by about 2.1pp/trade) survives the fix. The
defect corrupted the counts, not that comparison.

## What changed

All six of `omen_agreement_census`, `omen_label_audit`, `omen_self_distinctness`,
`omen_query_path_probe`, `omen_shape_mutations`, `omen_temporal_census`:

* take `--horizon-minutes`, default 720, converted per corpus via
  `omen_experiment.resolve_horizon` using that corpus's measured cadence;
* `--horizon` still takes bars as an explicit override, and the report records
  the minutes it worked out to, so every command recorded in
  `data/attempts-revenir.md` still runs and reproduces bar-for-bar — 720 minutes
  of 3600s bars IS the old 12;
* every report carries `horizon_minutes`, `horizon_bars` and `bar_seconds` and
  is passed through `validate_report_horizon` BEFORE the write;
* `omen_self_distinctness` wrote no report at all and now writes one behind
  `--report`.

Two shared helpers in `omen_experiment`: `add_horizon_args` (one flag pair, one
help text, nine harnesses) and `settle_horizon`. `horizon_request` freezes the
caller's asked UNIT on first call, because `settle_horizon` rewrites
`args.horizon` to bars and a sweep's second corpus would otherwise have read
that bar count, concluded the caller asked in bars, and applied the first file's
horizon at a different cadence — the same defect one loop iteration later.
`omen_label_audit` and `omen_temporal_census` both depend on that freeze.

## Proof

    python -X utf8 -m pytest tests/test_every_omen_harness_asks_the_same_wall_clock_question.py -q
    -> 26 passed

Restoring the bars-only `add_argument("--horizon", type=int, default=12)` on all
six turns **16 of the 26 red**, including
`test_the_same_minutes_resolve_to_different_bars_at_different_cadences` on all
six and the report-triple check on all four that write a report without a node.

70 passed across every other test that imports `omen_experiment` or one of the
six. `scripts/pass_gate.py --check` = 723 passed, 0 failed.

## What is NOT done

`scripts/omen_l2_scheme_probe.py:232` and `scripts/omen_layer_probe.py:331`
still default to 12 bars. They were excluded from this item by decision (another
agent holds them), not by oversight, so the count is 7 of 9 fixed. Anyone
quoting an `h12` from either of those two still cannot compare it across
corpora.

No held-out edge number was measured this pass — this is a comparability fix to
the instruments, not a brain result.
