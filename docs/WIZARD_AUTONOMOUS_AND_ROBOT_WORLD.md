# W1z4rD + C0d3rV2 24/7 Autonomous Lifecycle, Polite Elastic PC-Allowance, Multimodal Architecture & Living Eye

## Overview

The **Wizard Area** (`http://localhost:8001/wizard-chat`) in `CoolCryptoUtilities` is driven exclusively by **W1z4rD Brain** (`http://127.0.0.1:8090` + Senior SE brain on `http://127.0.0.1:18096`) and **C0d3rV2** (`tools/c0d3rV2` + `tools/wizard_learning`), with **zero external LLM/agent dependencies**.

Whenever W1z4rD is not actively serving a live user conversation, `tools/wizard_learning/autonomous_lifecycle.py` continuously runs five self-improving phases 24/7 at `BELOW_NORMAL_PRIORITY_CLASS`:

1. **`SELF_EDUCATING`**:
   - Resolves queued `/hypothesis/queue` items using verified `AnchoredMemory`, empirically validated mechanics/physics curriculum (`EMPIRICAL_CURRICULUM`), and the local 72.2M-terminal Senior SE W1z4rD brain (`:18096`).
   - Synthesizes parametric first-principles robotics systems via `DeductiveReasoner` + Kùzu/SymPy dimensional closure (`KuzuWorkbench`) and ingests open-access peer-reviewed mechanics literature (`NCBIResearchClient`).
2. **`TRAINING_SELF` (Including Streaming Multimodal + Affect/Identity Hebbian Co-Firing)**:
   - Writes deterministic lossless bindings + symbolic `Eem` relation induction via `POST /brain/pretrain_binding` (`POOL_TEXT=1 -> POOL_ACTION=4`), reinforces `/multi_pool/train_pair`, persists conversational foundations in `AnchoredMemoryManager` (with cross-process `st_mtime` hot-reload), and binds equations into Kùzu.
   - Streams cloud-verified, zero-local-bloat cross-modal episodes via `tools/wizard_learning/multimodal_feeder.py` (`PoliteMultimodalFeeder`) across **text (`POOL_TEXT=1`), visual primitives (`POOL_IMAGE=2`), spoken acoustic motifs (`POOL_AUDIO=3`), and social/action grounding (`POOL_ACTION=4`)** so neurons that wire together fire together without embellishing media or bloating neuron count.
   - Seeds and reinforces `social_affect` and `ocular_identity` neural pools via `tools/wizard_learning/neural_eye_dynamics.py` (`train_slice_of_affect_identity_corpus`) from curated affect/identity episodes and `crossmodal_episodes.jsonl`.
3. **`TESTING_TRAINING`**:
   - Actively probes W1z4rD recall on conversational foundations (`Hello`, `Who are you?`) and newly trained physics/engineering items across `AnchoredMemory -> /brain/chat -> /chat`, immediately self-healing any miss.
4. **`CONTROLLING_ROBOT_WORLD`**:
   - Drives the BrandDozer **Multiscale Robot World** (`C:\Users\Adam\Desktop\Multiscale Robot World`, project `dff079ccfa97462a8bd03228e749c287`) multi-agent W1z4rD robot controllers (`wizard_prime` + replicated `wizard_agent_02..N`), verifying raw-material refining (`26` ores/alloys), scratch robot construction (`56` real-world parts), RCL robot programming, in-world robot replication, and continuous Mark $N \to N+1$ physical body self-upgrading, then feeds newly discovered physical design laws back into W1z4rD brain training.
5. **`SLEEPING_SETTLING`**:
   - Monitors baseline-relative `w1z4rd_node` working-set growth (`SettlingController`, cleanly isolated from Waitress's Python RSS) and episode budgets.
   - Marks `wizard_cognitive_state.json` as `SLEEPING_SETTLING` (`can_chat: False`, recording `settling_holder_pid`), tunes `/brain/tier_orchestrator/params` to the current PC allowance tier, executes `/brain/sleep` (`120s` timeout) -> `/brain/consolidate` (`120s` timeout) -> throttled `/brain/checkpoint` & `/neuro/checkpoint` (`180s` timeout, max once per `8 min`) -> Windows `EmptyWorkingSet` trim, verifies post-sleep recall, resets `sleep_pressure` to `0.0`, and unlocks chat in a `finally:` block.

---

## Top-Anchored Eye-Centric Layout, Event-Driven Biological Eyelids, Neural Sentiment & On-The-Fly Eye Self-Redesign

1. **Top-Anchored Eye-Centric Sanctum (`WizardChatView.vue` + `App.vue`)**:
   - Across mobile, desktop, and fullscreen, the **Living Eye** (`WizardEye.vue` + `WizardMetricRing.vue`) is anchored near the top of the viewport (`justify-content: flex-start`) as the primary visual focal point.
   - The **Reply Area** (`.reply-area-dock`) sits right on/near-under the Eye (`margin-top: calc(... * -0.37)` below the iris/pupil without obscuring them), and the **Message Box** (`.input-bar`) sits directly underneath the Reply Area. Diagnostics and Studio drawers open as floating overlays so nothing pushes or crowds the Eye.
2. **Event-Driven Biological Eyelid States (`cognitive_state.py` + `WizardEye.vue` `threejs-mutable-eye-v2`)**:
   - Uses a full 3D shaded upper/lower eyelid dermis + closed palpebral seam + subsurface eyeball dome shader (`edgeDist <= 0` renders anatomical closed eyelids rather than discarding fragments):
     - **`offline` (`nodeOnline === false` or `mode === 'OFFLINE'`)**: Eyelids **completely shut (`openness = 0.0`) with zero movement** (frozen shader clock, zero gaze/particle motion).
     - **`settling_sleep` (`SLEEPING_SETTLING` / `CHECKPOINT_LOCKED` for `< 3600s`)**: Eyelids **completely shut (`openness = 0.0`, never open)** with slow subsurface globe drift and periodic eyelid squeezes (`uLidSqueeze`).
     - **`rem_sleep` (`SLEEPING_SETTLING` / `CHECKPOINT_LOCKED` for `>= 3600s`, `is_rem_sleep === true`)**: Eyelids **completely shut (`openness = 0.0`, never open)** in phasic **REM state** (rapid darting subsurface ocular bursts beneath the shut lid, micro-squeeze twitches, and subtle dream bioluminescence).
     - **`waking` (`< 6.0s` after transitioning from offline/settling -> awake)**: **Sleepy eye waking up** (`~5.6s` groggy multi-step curve: initial wake squeeze -> narrow slit crack -> heavy sleepy blink -> half-lidded rise with pupillary light-reflex constriction -> alert openness).
3. **Emergent Neural-Pool Conversational Sentiment (`tools/wizard_learning/neural_eye_dynamics.py`)**:
   - Conversational sentiments (`valence`, `arousal`, `warmth`, `focus`, `wonder`, `caution`) are **never** fabricated from regex/keyword lists on user text.
   - `probe_neural_conversation_sentiment()` queries W1z4rD's `social_affect` neural pool (`/multi_pool/ask`), activated 5-pool brain concepts, and live substrate neuromodulators (`DA`, `NE`, `ACh`, `5-HT` from `/brain`). When no `social_affect` neurons fire, `pool_grounded` is `False` (`dominant_sentiment = "neutral"`), modulated honestly by substrate neuromodulator baselines alone.
4. **On-The-Fly Identity & Experience Eye Self-Redesign (`tools/wizard_learning/self_design.py` + `neural_eye_dynamics.py`)**:
   - Expands declarative ocular morphology (`EYE_BOUNDS`) to 16 bounded parameters: `hue`, `secondary_hue`, `heterochromia`, `openness`, `pupil_aspect` (circular `1.0` to slit `0.28`), `pupil_scale`, `iris_radius`, `limbal_ring`, `striae_detail`, `collarette_glow`, `sclera_tint`, `cornea_gloss`, `spiral_tightness`, `spiral_arms`, `nebula_strength`, and `particle_speed`.
   - `evaluate_on_the_fly_eye_redesign()` probes W1z4rD's `ocular_identity` neural pool + `brain.identity.toml` + experiential maturity (self-education cycles & robotics Mark generation). When neural identity activations diverge meaningfully (`peak_delta >= 0.14` or `mean_delta >= 0.06`, or operator-requested via `kind="autonomous-redesign"`), W1z4rD smoothly evolves its eye design, increments `identity_generation`, persists it to `LearningStore`, and morphs the live WebGL shader in real time without page reload.

---

## Polite Elastic PC-Allowance ("Nice" Multi-Strategy Resource Scaling)

Both W1z4rD and Multiscale Robot World share the host PC with multiple concurrent crypto trading & evolution strategies (`main.py --action start_production`, `market_evolution_service.py`, `run_waitress.py`, `monitoring_guardian`, `ContinuousRefinement`). Neither system may starve other strategies OR be killed when PC RAM tightens:

- **Polite Reserve + Under-Cap Cushion (`compute_elastic_pc_allowance`)**:
  - Leaves `max(8.0 GB, 26% of total RAM)` (`8.26 GB` on a `31.78 GB` PC) untouched for concurrent strategies, OS, and desktop apps, and stays at `88%` (`12%` cushion) of the remaining headroom above that floor.
  - Dynamically scales across four tiers without stopping execution (`EXPANDED`, `BALANCED`, `POLITE_UNDER_CAP`, `COMPACT_SHRINK`).
- **Rust TierOrchestrator & Build Guard (`D:\Projects\W1z4rDV1510n`)**:
  - `supervisor.toml`, `scripts/w1z4rd_supervisor.py`, and `start_node.ps1` set `W1Z4RD_TIER_MIN_SYS_AVAIL_MB = 8192` so `w1z4rd_node.exe` automatically pages cold neurons to `.wbrain` on SSD when free RAM drops below `8 GB` without ever terminating the node.

---

## Cloud-Synced Multimodal Corpora & Minimal-Neuron Canonical Decomposition

All curated audio and visual media live in Google Cloud Storage (`gs://w1z4rdv1510n-multimodal-corpus`) using `D:\Projects\W1z4rDV1510n\scripts\multimodal_cloud_sync.py`: every synthesized audio clip and downloaded image is **uploaded -> verified remotely (size + MD5 + SHA-256) -> deleted locally (`unlink`) immediately** (`0` leftover local temp media files).

1. **English Speech Synthesis Corpus (`scripts/curate_english_speech_cloud_sync.py`)**: `170` English utterances (`14.52 min`) synced to GCS and decomposed into non-embellished `W1ZA1-mel20-hnm-25ms` compact acoustic motifs (`165.5x` motif compression).
2. **Concrete Noun $\leftrightarrow$ Image $\leftrightarrow$ Spoken Audio Dictionary (`scripts/curate_noun_visual_audio_dictionary.py`)**: `125` license-clean (`Public Domain`, `CC0`, `CC-BY`) concrete nouns (`375/375` cloud-verified blobs) co-fired across 4 pools (`100%` `text->social`, `96.7%` `image->social` recall).
3. **5-Modality Meta-Stream Pool Erection & 4-Course Obstacle Suite (`scripts/run_multimodal_obstacle_course.py`)**: Passed `4/4` courses (`100%`) at `58.79 MiB` total RSS (`8/8` dynamically erected meta-stream pools, `98.55%` neuron savings).

---

## Millisecond Conversational, Symbolic-Physics & C0d3rV2 Agent Architecture (`4 ms – 65 ms`)

W1z4rD converses, solves symbolic math/physics/robotics, learns live Q&A corrections, and executes C0d3rV2 Agent Mode actions in milliseconds — **including while background synaptic settling (`SLEEPING_SETTLING` / REM) or `.wbrain` checkpointing is active**:

1. **Sub-15ms Local-First Conversational Cascade (`tools/wizard_learning/fast_conversational_engine.py` + `web/wizard_chat/views.py`)**:
   - **Tier 0 (`~0.2 ms`)**: Durable `AnchoredMemoryManager` + hot synaptic pool mirror (`_HOT_CHAT_CACHE`, synced continuously with W1z4rD's dedicated `chat_in -> chat_out` multi-pool).
   - **Tier 1 (`~0.5 ms`)**: Live substrate & cognitive introspection (`_try_live_introspection` reading `3.0s`-cached `SettlingController` metrics for real-time node RSS, PC allowance tier, sleep pressure, and Robot World Mark generation).
   - **Tier 2 (`~0.3 – 8 ms`)**: Pure-Python exact `fractions.Fraction` AST polynomial algebra & symbolic calculus (`solve`, `derivative of`, `integrate` — avoiding Windows Defender's multi-second cold file scan of `sympy`'s 2,400 submodules), NIST/NASA/ASTM `EMPIRICAL_CURRICULUM` laws, and `DeductiveFirstPrinciplesReasoner.deduce_robot_system()`.
   - **Tier 3 (`~1 ms`)**: AST-verified Python synthesis (`synthesize_verified_code` validated via `ast.parse`).
   - **Tier 4 (`~2 – 45 ms`)**: Dedicated `chat_in -> chat_out` (isolated from legacy `in`'s 54-concept cap) and `/query/integrated` on `127.0.0.1:8090`, guarded by a `50 ms` socket timeout and `5.0s` busy circuit-breaker (`_CHAT_POOL_BUSY_UNTIL`) when `:8090` is settling/checkpointing; ungrounded prompts enqueue asynchronously via `enqueue_hypothesis_async` without stalling the HTTP response.
2. **Single-Roundtrip Neural Eye Dynamics (`tools/wizard_learning/neural_eye_dynamics.py`)**:
   - `probe_unified_eye_dynamics()` issues at most **one** `40 ms`-bounded `/multi_pool/ask` against `social_cue` (`social_affect` + `ocular_identity` simultaneously) and reads in-memory cached neuromodulators (`0` blocking `/brain` mutex calls on the chat or agent critical path), backed by a `5.0s` settling circuit-breaker (`_SOCIAL_BUSY_UNTIL`) and `_HOT_SOCIAL_CACHE`.
3. **Settling-Tolerant Chat & Fast C0d3rV2 Agent Mode (`web/wizard_chat/agent_runner.py` + `agent_views.py`)**:
   - Only `mode == "OFFLINE"` blocks chat; during `SLEEPING_SETTLING` or `CHECKPOINT_LOCKED`, `WizardChatMessageView` preserves `cognitive_state` (so the Living Eye faithfully renders `settling_sleep` / `rem_sleep`) while serving conversational, analytical, and coding turns in milliseconds.
   - `_fast_c0d3r_direct_agent_turn()` executes workspace file reads (`~21 ms`), directory listings (`~28 ms`), `(mtime_ns, content)`-cached priority workspace searches (`~388 ms` cold / `<15 ms` warm, excluding `data/` and `storage/`), audited shell commands (`~23 ms`), and in-process/sandboxed AST-verified Python synthesis + live verification (`~107 ms`) without stalling in 64-step JSON orchestrator loops.
4. **Instant-Bind Lazy-WSGI Waitress & SQLite WAL Concurrency (`web/run_waitress.py` + `D:\Projects\W1z4rDV1510n\supervisor.toml`)**:
   - `run_waitress.py` enables `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` on `web/db.sqlite3` (preventing in-process `EdgeForge` `StrategyVenture` writes from exclusive-locking HTTP session/setting reads), sets `sys.setswitchinterval(0.0005)` (`0.5 ms` GIL quantum) across `48` threads, binds `:8001` in `<200 ms` via a lazy-WSGI dispatcher while `coolcrypto_dashboard.wsgi` warms in a background thread, and deduplicates overlapping watchdog launches via `_already_healthy()` + `<35s` booting-sibling yield.
5. **Silent Runtime Context Envelope, C0d3rV2 Context Tools & Massive Saved English Corpus (`1,202 / 1,202` Verified)**:
   - Every `/api/wizard-chat/message/` and `/api/wizard-chat/agent/` turn receives a `<0.1 ms` structured runtime context envelope (`tools/wizard_learning/runtime_context.py`: local 12h/24h time, date, weekday, timezone, UTC offset, cached host location, OS/workspace, and caller-supplied `context`/`data`), surfaced **strictly when asked or relevant** (verified with negative-control zero-leakage tests).
   - Upgraded **C0d3rV2** (`tools/c0d3rV2/context_tools.py`) with 4 registered tools: `ContextClockTool` (`context_clock`), `StructuredDataTool` (`data_transform`: inline JSON/CSV extraction, filtering, sorting, aggregation, JSON/CSV/Markdown-table formatting), `FastWebLookupTool` (`web_lookup`: DuckDuckGo Instant Answer + Wikipedia REST with automatic W1z4rD anchoring & multi-pool training), and `PolyglotCodeTool` (`code_synth`: Python, Rust, TypeScript, JavaScript, SQL, Go, Bash, PowerShell, HTML/CSS).
   - Compiled and permanently saved `1,198` English training/evaluation items under `data/wizard_learning/massive_context_corpus/` (`corpus_manifest.json` + 6 JSONL files, indexed in `<0.4 ms` by `ParaphraseCorpusIndex` in `tools/wizard_learning/context_and_data_engine.py`) and verified **`1,202 / 1,202 (100.00%)`** live HTTP accuracy in **`16.64s`** total (`p50 = 6.43 ms`, `p95 = 68.30 ms`, `p99 = 154.81 ms`, `max = 354.98 ms`) via `tools/wizard_learning/run_massive_ms_battery.py` (saved to `data/wizard_learning/massive_context_corpus/battery_report.json`).
6. **EdgeForge Isolated OS Worker Daemon (`web/edgeforge/worker_daemon.py`)**:
   - Moved `EdgeForge`'s 3 multi-chain OHLCV GA workers out of `Waitress`'s Python GIL into a detached `BELOW_NORMAL_PRIORITY_CLASS` OS subprocess (`web/edgeforge/worker_daemon.py`, publishing heartbeats every `5s` to `runtime/edgeforge_worker_state.json`), wired into `web/edgeforge/apps.py` and `web/edgeforge/views.py`. Eliminated periodic `10–25s` GIL stalls on `:8001` (dropping `1,202`-request `p50` from `79.99 ms` to `6.43 ms`, `p99` from `11,445 ms` to `154.81 ms`, and `max` from `25,458 ms` to `354.98 ms`) while `EdgeForge` continues running uninterrupted across Waitress reloads.

---

## Durable Operational Decisions & Known Limits

1. **RAM Measurement Isolation & Zero-Delta Sleep Guard**: `SettlingController._measure_wizard_node_ram()` measures only `w1z4rd_node.exe` (`<0.1ms` cached-PID fast path with a `30s` negative-scan cache when `w1z4rd_brain_server` is absent). `has_unsettled_activity` (`ram_delta_mb >= 25.0` or `episodes_since_settle >= 2`) prevents external RAM usage by concurrent crypto strategies from re-triggering zero-delta sleep loops.
2. **Stale-Lock Auto-Recovery with REM Live-Holder Preservation**: `CognitiveStateManager.read()` auto-recovers crashed `SLEEPING_SETTLING` / `CHECKPOINT_LOCKED` locks older than `240s` *only when `settling_holder_pid` is no longer alive*, allowing legitimate multi-hour consolidation cycles (`>= 3600s`) to enter and sustain `rem_sleep` while still recovering instantly from process crashes.
3. **Backward-Compatible `.wbrain` Brain & Pool Metadata Loaders (`crates/brain/src/wbrain_metadata.rs` & `pool.rs`)**: Both `deserialize_wbrain_brain_metadata` (`WbrainBrainMetadata::config.binding_pool_config`) and `deserialize_wbrain_pool_metadata` (`WbrainPoolMetadata::config`) try the current layout first and fall back to `LegacyPoolConfigWithoutFanout` (predating `PoolConfig::max_atom_fanout: usize` in positional `bincode`). Verified against both committed fixtures (`binding_posting_generation_compat` `2/2`, `atom_fanout_overflow_spill` `6/6`) and the live `87.55 GB` `brain.wbrain` (`live_87gb_wbrain_if_present_restores_cleanly` `1/1` in `2.16s`, `missing=[]`).
4. **Waitress Thread-Pool Protection & Known Limits (`web/wizard_chat/views.py` & `run_waitress.py`)**: All local node URLs normalize `localhost` to `127.0.0.1` (eliminating Windows `2.0s` IPv6 `::1` SYN retries). During deep `.wbrain` disk checkpoints on `D:\`, `:8090` write-lock routes (`/brain/pretrain_binding`, `/multi_pool/train_pair`, `/media/train_sequence`) queue asynchronously in background threads while the hot synaptic mirror serves user chat in `<25 ms`; novel ungrounded associations finish binding into `.wbrain` once the active checkpoint/settling pass releases the lock.
