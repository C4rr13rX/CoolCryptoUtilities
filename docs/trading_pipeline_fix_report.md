# Trading Pipeline Fix Report (Checklist)

Updated: 2026-02-07 03:40:22 -05:00

This is a persistent, check-off-able audit + remediation plan for getting the trading pipeline to reliably progress:
`training/lab -> ghost/paper -> (optional) live` with realistic profitability checks.

Notes:
- “Live” execution remains **explicitly gated**. Nothing here should silently execute real swaps/bridges.
- Each item includes a starting-point file reference so it’s easy to re-locate the implementation.

## How To Validate

- Unit tests: `pytest -q`
- Trading-focused tests: `pytest -q tests -k trading`

## Checklist (Highest Impact First)

- [x] **Out-of-sample evaluation (time holdout) for training + promotion gates**
  - Problem: candidate training and evaluation use the same samples (in-sample), inflating scores and breaking readiness gates.
  - References: `trading/pipeline.py:685`, `trading/pipeline.py:705`, `trading/pipeline.py:2148`

- [x] **Confusion-matrix evaluation must align to the evaluated subset**
  - Problem: confusion matrices are built using `data_loader.last_sample_meta()` for the full dataset, but evaluation may run on a subset (once holdout exists), causing misalignment.
  - References: `trading/pipeline.py:2359` (`_build_confusion_report`)

- [x] **Ghost/Paper PnL must be cost-realistic and consistent with simulation balances**
  - Problem: `profit = (price - entry_price) * size` ignores modeled fees/slippage; balances don’t reflect costs either.
  - References: `trading/bot.py:2415`, `trading/bot.py:2066`

- [x] **Fill/slippage feedback loop must actually record fills**
  - Problem: `TradingBot.record_fill()` exists but is never called, so execution bias stays untrained and scheduler can’t adapt.
  - References: `trading/bot.py:3393`, `db.py:664`

- [x] **BusScheduler spread/depth filter must not be dead code**
  - Problem: spread filter uses `hasattr(self, "_bias")` but `_bias` doesn’t exist, so `implied_spread` is always 0.
  - References: `trading/scheduler.py:336`

- [x] **Bus transition actions must actually execute (gated)**
  - Problem: pipeline computes `bus_swap_actions`, bot stores them, but nothing consumes/execut
    es them; stage transitions can stall or be misleading.
  - References: `trading/pipeline.py:3434`, `trading/bot.py:1031`, `production.py:409`

- [x] **Live execution wiring (still gated)**
  - Fixed: enter/exit now execute real swaps when `ENABLE_LIVE_TRADING=1` and execution is explicitly enabled (set `EXECUTE_LIVE_TRADES=1` or `LIVE_TRADES_DRY_RUN=0`).
  - Fixed: partial exits keep the remaining position instead of deleting it.
  - References: `trading/bot.py:2274`, `trading/bot.py:440`, `services/swap_service.py:1`, `services/bridge_service.py:1`
