CoolCryptoUtilities – Targeted Improvements
===========================================

The following upgrades align with the health report requirements (accuracy across 5 minute–6 month windows, efficient CPU execution, ethical news coverage, and iterative fix/test loops). Each improvement lists its goal and validation plan.

1. Real-time symbol normalization
   - Align synthetic symbols such as `WETH-USDbC` with exchange-supported pairs so websocket feeds unlock for Binance/Coinbase.
   - Validation: unit test for `_split_symbol` covering USDbC; run `pytest tests/test_data_stream.py`.

2. Websocket fallback guard
   - Throttle “URL unavailable” spam and ensure REST consensus polling keeps streaming so downstream metrics stay populated when no websocket exists.
   - Validation: unit test for log throttle helper; manual verification via `pytest tests/test_data_stream.py`.

3. TensorFlow CPU runtime tuning
   - Force CPU-only execution, reduce worker threads, and silence CUDA init failures to keep the pipeline stable on an i5 + 32 GB host.
   - Validation: run an import smoke test via `pytest tests/test_brain_modules.py -k runtime --maxfail=1` (ensures TensorFlow loads with new settings).

4. System profile detection
   - Detect hardware limits (CPU count, RAM) and expose a shared profile so training/data loaders scale gracefully.
   - Validation: new unit test for `services.system_profile.detect_system_profile`.

5. Hardware-aware data loader
   - Honor the system profile by capping `max_files`, `max_samples_per_file`, and cache sizes, keeping RAM usage bounded without losing fidelity.
   - Validation: extend `tests/test_data_loader.py::test_expand_limits_invalidate_cache` to cover profile application.

6. Horizon analytics recorder
   - Persist per-horizon coverage/MAE stats (5 m → 6 m) every training iteration so the optimizer & scheduler know which windows are reliable.
   - Validation: add assertions in `tests/test_scheduler.py` ensuring summaries exist (or unit test for the helper in `TrainingPipeline`).

7. Opportunity-biased scheduler
   - Surface buy-low/sell-high signals to the bus scheduler so directives favor those windows when equilibrium confidence is high.
   - Validation: new test in `tests/test_scheduler.py` exercising the bias hook.

8. Decision-layer opportunity boost
   - When the brain flags an opportunity, adjust direction probability/margins and annotate the queued trade for both ghost/live modes, ensuring 15 % savings once equilibrium holds.
   - Validation: extend `tests/test_opportunity.py` (or add a focused test) to check the new bias path.

9. Ethical news source catalog
   - Load an external JSON catalog of free/ethical RSS feeds, merge with defaults, and allow ISO-date windows for harvesting so arbitrary historical spans can enrich training.
   - Validation: extend `tests/test_news_ingestor.py` to cover config loading + `harvest_window`.

10. Training concurrency guard
    - Serialize `train_candidate` via a lock so background refinement no longer throws “bad parameter / API misuse,” keeping fix/test loops deterministic.
    - Validation: add a lightweight test that simulates concurrent `train_candidate` calls via mocking, or run `pytest tests/test_lab_preview.py` to ensure orchestration still passes.

11. Live-source market fallback
    - When REST endpoints cool down, reuse the most recent on-chain/live sample so ghost trading stays fed and warnings de-escalate.
    - Validation: run `pytest tests/test_data_stream.py`.

12. REST-only websocket toggle
    - Add `MARKET_WEBSOCKET_DISABLED=1` (alias `MARKET_WS_DISABLED`) to skip websocket sessions when endpoints are blocked, keeping REST consensus flowing without repeated DNS retries.
    - Auto-switch to REST-only after `REST_ONLY_WS_FAILURES` websocket failures, waiting `REST_ONLY_RETRY_SEC` before retrying live connections.
    - Validation: run `pytest tests/test_data_stream.py -k ws_disabled`.

13. Early stopping for CPU training
   - Add `TRAIN_EARLY_STOP` (plus patience/min-delta tuning) so candidate training halts once loss plateaus, keeping the i5 CPU budget under control without sacrificing calibration.
   - Validation: run `pytest tests/test_lab_preview.py`.

14. DNS outage domain guard
   - Add `NETWORK_OUTAGE_BLOCK_REST_SAME_DOMAIN=1` to optionally pause REST polling when a websocket DNS outage affects all REST hosts on the same domain.
   - Validation: run `pytest tests/test_data_stream.py -k same_domain`.

15. Offline-only market mode
   - Set `MARKET_FORCE_OFFLINE=1` (aliases `MARKET_OFFLINE_ONLY`/`NETWORK_FORCE_OFFLINE`) to skip live websocket/REST calls and stream cached snapshots when running in restricted or DNS-impaired environments.
   - Validation: run `pytest tests/test_data_stream.py -k force_offline_skips_network`.
