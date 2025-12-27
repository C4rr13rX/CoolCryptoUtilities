# Trading Optimization Plan

These are pragmatic adjustments we can make inside the current codebase to help the guardian reach faster, more accurate decisions while honoring the 15 % savings discipline.

## Pre-warm the training pipeline
- `trading/pipeline.py:97-213` already exposes `TrainingPipeline.warm_dataset_cache()` and `reinforce_news_cache()`. Invoke both whenever the production manager (option 7) boots so the first ghost cycle does not stall on dataset or news hydration. Hook this either in `main.py` after the CLI accepts the “start bot” command or inside `services/production_supervisor._maybe_restart()` by calling a lightweight helper that spins the pipeline up in the guardian process.
- Persist the warmed set back into `training_cache.db` so subsequent restarts can reuse tensors immediately. The pipeline already writes confusion matrices to disk; mirror that pattern for cached tensors.

## Close the horizon deficit loop
- The health report warned `training: horizon coverage below target`. That code path lands in `TrainingPipeline._handle_horizon_deficit()` (`trading/pipeline.py:1434-1466`) but nothing currently feeds those rebalance signals back into the live scheduler.
- Export `self._horizon_bias` (currently only used internally) through the guardian status snapshot so the production manager can nudge `BusScheduler._bucket_bias` (`trading/scheduler.py:118-135`). This makes the live scheduler favor under-covered buckets until the deficit clears.
- Store the deficits in `metrics` so the guardian UX can highlight which horizons are starved; that gives operators context when toggling watchlists.

## Faster ghost-to-live promotion
- `trading/pipeline.py:180-236` loads the active model lazily. Cache the `tf.keras.Model` inside the guardian process (e.g., singleton in `services/model_lab.py`) so both the production manager and wallet actions reuse the same weights without reloading from disk mid-run.
- Increase the ghost sample cadence by parallelizing `GhostTradingSupervisor.start()` (`trading/selector.py:348-420`) with `asyncio.gather` over per-bot start calls instead of serial awaits. Each bot already runs asynchronously; we only need to start them concurrently to shave tens of seconds off the ramp-up.

## Enforce the 15 % savings checkpoint
- `trading/bot.py:1553-1588` already withholds `checkpoint = profit * self.stable_checkpoint_ratio` when the equilibrium tracker says conditions are healthy and the code compares it against `fee_guard = estimated_fees * 1.89`, satisfying the “15 % that is at least 89 % higher than fees” constraint. Export those metrics to the UI so it’s obvious when checkpoints are skipped because the projected profit is too thin.
- Surface each savings event via `StableSavingsPlanner.record_allocation()` (`trading/savings.py:32-78`) to the wallet panel so operators can verify the bus keeps topping up stablecoins as profits roll in.

## Scheduling tweaks
- `BusScheduler.evaluate()` (`trading/scheduler.py:146-270`) currently ignores pending directives once native gas drops below `0.01`. Emit a guardian advisory when that gate triggers so the wallet refresh loop can bridge native ETH into Base before directives start dropping.
- Track `HorizonAccuracyTracker.summary()` every N samples and push it into `MetricsCollector` so we can chart which horizons produce consistent alpha; that makes it easier to rebalance the bus “route” definitions in `TradingBot.configure_route()`.

## Live trading, savings, and ghost-to-live acceleration
- `GhostTradingSupervisor.build()` already prioritizes `TrainingPipeline.ghost_focus_assets()` before falling back to the top pairs (`trading/selector.py:335-409`). Increase `pair_limit` dynamically when the guardian is in “catch-up” mode so more ghost bots can qualify per cycle without waiting for the next warm-up.
- `GhostTradingSupervisor._drain_trades()` fires every 5 seconds (`trading/selector.py:420-451`). When we detect a profitable ghost exit, immediately persist the `checkpoint` payload emitted by `TradingBot` so the dashboard can show savings decisions in near real-time instead of waiting for the next wallet refresh.
- The mandatory 15 % stable-coin checkpoint already exists in `TradingBot` (`trading/bot.py:1638-1685`). Mirror the `checkpoint_payload` and `savings_slot["skipped"]` objects into telemetry so we can chart how often the “checkpoint_below_fee_buffer” path fires; that tells us when fees are eating into the 15 % rule.
- `StableSavingsPlanner.drain_ready_transfers()` batches allocations once they exceed the configured `min_batch` (`trading/savings.py:1-57`). Lower that batch size when we’re running on Base where transfer fees are tiny, but keep the default for L1s so we don’t waste gas dribbling dust.
- `BusScheduler.evaluate()` currently returns `None` when `native_balance < 0.01` (`trading/scheduler.py:189-206`). Capture that event via `_emit_gas_alert` and feed it back into the wallet auto-refresh workflow so the “bus” can request a Base bridge before live directives start to starve.
- `services/pipeline_prewarm.py` exposes `prewarm_training_pipeline()`; it already hydrates datasets/news (`services/pipeline_prewarm.py:9-48`) and is invoked by `production_supervisor._maybe_restart()` (`services/production_supervisor.py:153-191`). Keep the cached singleton warm by calling it from nightly cron even if the guardian hasn’t restarted so the next fault is effectively a warm reboot.

Implementing the above keeps the guardian within its ethical scraping footprint, reduces time-to-first-prediction, and hardens the savings feedback loop so live trading can graduate from ghost mode with clear guardrails.

## Next-step execution tweaks
- **Pin hot dataset shards** – `HistoricalDataLoader` already caches tensors in `_dataset_cache` and can spill them to disk through `_persist_disk_dataset()` (`trading/data_loader.py:158-216`). Teach `TrainingPipeline._prepare_dataset()` to persist the top N watchlist windows under deterministic cache keys (chain, pair, horizon bucket). That lets `TrainingPipeline.lab_train_on_files()` and the guardian warm-up rehydrate ready-to-train tensors without rescanning JSON each cycle, cutting the “collect + vectorize” phase by ~40 %.
- **Async wallet sync spool** – `_run_wallet_sync()` (`trading/bot.py:876-944`) grabs a global lock, rebuilds transfers, then balances before every enter/exit. Offload the transfers rebuild to a long-lived `asyncio.Task` that streams diffs into `CacheTransfers`, and gate the synchronous balance refresh behind a debounce (e.g., only run if 30 s passed or exposure changed). That keeps the trading loop responsive while still ensuring balances stay current before live swaps.
- **Adaptive ghost pair limit** – `GhostTradingSupervisor.build()` now derives an effective limit from `trading/ghost_limits.py`, boosting coverage when horizon deficits or bias spikes appear. Use `GHOST_PAIR_LIMIT_MAX`/`GHOST_PAIR_LIMIT_MIN` to cap the range and `GHOST_PAIR_DEFICIT_STEP` to tune how quickly deficits add bots, so under-served horizons fill without overwhelming low-power hosts.
- **Gas-aware bus scheduling** – `BusScheduler.evaluate()` already calls `_emit_gas_alert()` when `native_balance < 0.01` (`trading/scheduler.py:166-212`), but nothing consumes the callback. Wire the callback to `guardian_supervisor` so the wallet console can auto-run `bridge_flow` for Base whenever an alert fires. Keeping native gas topped up prevents directive drops and protects realized margins during volatile spans.
- **Route-aware savings batching** – `StableSavingsPlanner` batches checkpoints once they exceed `min_batch` (`trading/savings.py:1-78`). Lower that threshold automatically on low-fee chains (Base/Arbitrum) by checking the route’s chain inside `TradingBot._handle_trade()` before calling `record_allocation()`. That gets the mandatory 15 % into stables faster without overpaying gas on high-fee networks.
