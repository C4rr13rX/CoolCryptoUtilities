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

Implementing the above keeps the guardian within its ethical scraping footprint, reduces time-to-first-prediction, and hardens the savings feedback loop so live trading can graduate from ghost mode with clear guardrails.
