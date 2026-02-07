# Free-Only API Replacement Strategy (Research-Based)

Goal: Replace every paid or externally dependent API with free, public-good, or self-hosted alternatives. Prioritize Layer 1 data, public datasets, and locally run services. Any third-party dependency must be demonstrably free, open, and available for the public good with no paid-tier lock-in.

## Definition of "Free"
- No paid tiers or usage-based billing.
- Open-source software or public-good data sources.
- Self-hosted services preferred when possible.
- Public APIs allowed only if they are free, stable, and not gating critical features behind paid plans.

## Phase 0: Inventory and Classification (Blocking Step)
Produce a dependency matrix of every external call and paid service.

Inventory checklist:
- Code scan for outbound HTTP requests and SDKs.
- Identify: domain, endpoint, auth type, cost, data category, and criticality.
- Classify by function:
  - Market data
  - On-chain data
  - News/sentiment
  - AI inference
  - Storage/search
  - Notifications
  - Identity/auth
  - Maps/geo (if any)

Deliverable: `docs/free_api_dependency_matrix.md` (source, cost, replacement target, risk).

## Phase 1: Replacement Architecture (Free-Only)
Build a free-only abstraction layer so providers can be swapped without touching trading logic.

Core principles:
- Provider interfaces: `MarketDataProvider`, `ChainDataProvider`, `NewsProvider`, `AiProvider`.
- Runtime selection via config.
- Always keep a "local first" provider as primary.

### Market Data (Price/Volume/Orderbook)
Primary (preferred):
- On-chain data derived from self-hosted nodes and indexers.
Secondary (free public):
- Exchange public REST/WebSocket endpoints (Binance, Coinbase, Kraken, etc).
Risk:
- Public endpoints can rate-limit or degrade; use them as fallback only.

Implementation:
- Run local chain nodes for supported networks.
- Add local price index derived from DEX pools and on-chain swaps.
- Cache everything in Postgres/ClickHouse to avoid re-querying.

### On-Chain Data
Preferred stack:
- Self-hosted nodes (Ethereum: Nethermind/Erigon/Geth; Bitcoin: bitcoind).
- Indexer: Blockscout or The Graph (self-hosted) for queries.
- Optional: Substreams (self-hosted) for streaming.

Benefits:
- Full control, no API limits, supports deterministic audit trails.

### Storage-Efficient Chain Strategy (5-Year Window)
If you only need the last ~5 years:
- Use pruned or snap-synced nodes with a bounded history window where supported.
- Prefer nodes with configurable pruning and snapshot import.
- Backfill only required blocks (by height/time) for analytics; avoid full history.
- Cache normalized chain events in Postgres/ClickHouse and drop raw block data once indexed.
- Use incremental daily backfills + rolling window retention.
- For Bitcoin-like chains: assume full history is large; keep pruned node + store last 5 years in local DB.

Tradeoffs:
- Pruning reduces disk but may limit historical queries; mitigate by persisting needed features in local DB.
- Use cold storage for raw data if required, but do not keep it online.

### News / Sentiment / Research
Free sources:
- RSS feeds (major outlets + crypto blogs).
- Public datasets: GDELT, CommonCrawl, Wikipedia dumps.
- Self-hosted scraping with rate limits + robots rules.

Constraints:
- Respect source ToS.
- Use cached summaries and keep original URLs for attribution.

### AI Models / Inference
Replace any paid AI inference with open-source local inference:
- Text: Llama, Mistral, Qwen via vLLM or llama.cpp.
- Embeddings: BGE or E5 via local embedding server.
- Vision: Qwen-VL / LLaVA locally.
- PDF/OCR: PyMuPDF + Tesseract.

### Storage / Search
Free stack:
- Postgres for relational + Timescale/Crunchy if needed.
- ClickHouse for high-volume time-series.
- Redis for caching.
- OpenSearch or Meilisearch for full text search.

### Notifications
Free options:
- Email via self-hosted SMTP (Postfix) or local sendmail.
- Webhooks to local consumers.
Constraint:
- SMS is rarely free at scale. If required, use user-provided gateways.

### Auth / Users
Local Django auth only. No external auth providers.

## Phase 2: Research Plan (Evidence-Based Decisions)
For each category, gather evidence on:
- Data accuracy vs paid source.
- Latency under load.
- Coverage (symbols, markets).
- Rate-limit resilience.
- Failure modes and fallback quality.

Methods:
- Side-by-side comparisons for at least 2 weeks.
- Spot checks for outliers or missing data.
- Record results in a `reports/free_api_eval/` dataset.

## Phase 3: Replacement Execution Plan
1. Build adapters for each provider category.
2. Switch core modules to call adapter interfaces only.
3. Introduce local-first providers.
4. Add fallback providers only for redundancy.
5. Remove paid dependencies entirely.

## Phase 4: Validation and Hardening
Validation checklist:
- Cold start (no paid services running).
- Stress test on data ingestion.
- Simulated outages for every provider.
- Price/volume accuracy checks vs reference.
- On-chain indexer resilience.

## Risks and Mitigations
- Reliability risk: Use local caching and persistent queues.
- Data gaps: Cross-source consensus and synthetic fallback.
- Legal risk: Respect source usage policies and robots rules.
- Maintenance cost: Plan for disk/compute budgets and ops scripts.

## Implementation Artifacts (Suggested)
- `docs/free_api_dependency_matrix.md`
- `docs/free_api_replacement_strategy.md` (this doc)
- `scripts/audit_external_calls.py` (to auto-scan)
- `configs/providers.yaml` (provider selection)

## Acceptance Criteria
- No paid API keys required for core functionality.
- All external calls are either self-hosted or public-good sources.
- System continues to function if all paid services are removed.
