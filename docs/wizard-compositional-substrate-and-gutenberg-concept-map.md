# W1z4rD Compositional Coding Substrate & Project Gutenberg Concept-Map Architecture

## 1. Root-Cause Resolution (Eliminating Template Shortcuts & Raw Prose Leaks)

Previously, imperative code/model prompts in `http://localhost:8001/wizard-chat` failed for four structural reasons:

1. **Speech-Act Conflation in Online Lookup (`tools/wizard_learning/context_and_data_engine.py`)**:
   - `resolve_online_lookup_if_needed` matched noun phrases on prompts like `"Create a python class for a bike."` and `"Create a model of a bike in Python."`, returning raw Wikipedia encyclopedia prose (*Decorator pattern* / *Bicycle and motorcycle dynamics*) and appending those mismatched pairs into `licensed_web_training_material.jsonl`.
   - **Fix**: Contaminated records were purged, and `is_code_or_model_creation_intent(prompt)` now enforces a strict speech-act boundary so imperative code/class/model/simulation requests **never** return raw encyclopedia prose.
2. **Hardcoded `if/elif` Canned Snippet Table (`tools/wizard_learning/fast_conversational_engine.py`)**:
   - `synthesize_verified_code` previously matched ~7 benchmark keywords (`lru`, `prime`, `fib`, `palindrome`, etc.) and returned `None` on real-world domain entities (`bike`, `barometer`, `submarine`, `trebuchet`).
   - **Fix**: Replaced with the **Rust-`:8095`-First + Open-Ontology Compositional Domain Coder** (`tools/wizard_learning/compositional_domain_coder.py`).
3. **Agent Runner Dead-End (`web/wizard_chat/agent_runner.py`)**:
   - `_fast_c0d3r_direct_agent_turn` previously swallowed unhandled turns instead of returning `None` to let `C0d3rV2`'s full `ProcessFlow` + `Orchestrator` execute multi-step workspace workflows.
   - **Fix**: `_fast_c0d3r_direct_agent_turn` now delegates single-turn verified synthesis cleanly with `allow_web_lookup` and returns `None` for multi-step orchestration so `C0d3rV2` `Orchestrator` runs unhindered.
4. **Dedicated Rust Neural Coding & Concept Substrate (`127.0.0.1:8095`)**:
   - `127.0.0.1:8090` (`w1z4rd_node.exe`) runs `market_small.identity.toml` (`ohlcv` / `news` / `outcome` trading pools).
   - `127.0.0.1:8095` (`w1z4rd_brain_server.exe`, `W1Z4RD_NODE_BRAIN_DIR=D:\Projects\W1z4rDV1510n\runtime\brains\wizard-live-chat-coding-8095\brain`) runs `coding_debug.identity.toml` backed by `.wbrain` + `brain.wal`.
   - Uses salience-weighted sensory keys (`{lang}::{slug}::{slug}::{slug}`) bound across `pool_id=1 -> pool_id=4` (avoiding `pool_id=12` `InstructionIntentEncoding` threshold collapse) plus modality verification so dictionary `concept::` nodes and executable `python::` / `rust::` / `typescript::` artifacts coexist cleanly without cross-entity or cross-modality aliasing.

---

## 2. Project Gutenberg Public-Domain Concept-to-Concept Graph ("What Counts the Most")

Compiled via `tools/wizard_learning/build_gutenberg_concept_map.py` and served in `< 0.1 ms` by `tools/wizard_learning/gutenberg_concept_map.py`:

- **Primary Public-Domain Sources** (Project Gutenberg headers/footers stripped per trademark guidelines):
  1. **Project Gutenberg `#29765`** — *Webster's Unabridged Dictionary (1913)* (`28,956,350` bytes, `73,610` clean non-obsolete entries parsed).
  2. **Project Gutenberg `#22`** — *Roget's Thesaurus of English Words and Phrases* (`1,494,120` bytes, `53,926` conceptual taxonomy mappings).
- **Dependency Centrality Selection (`25,000` Core Concepts in `8.09 MB`)**:
  - Every candidate concept is scored by **Graph In-Degree Centrality** across all `73,610` definitions (`3x` weight as an `is_a` hypernym, `2x` as a `has_part` constituent, `1x` as a `used_for` functional verb), plus structural edge richness, Roget taxonomic breadth, and STEM/physical domain weight (`Mech.`, `Phys.`, `Opt.`, `Elec.`, `Chem.`, `Hydraul.`, etc.).
  - Produces `25,000` high-leverage concept nodes and `77,843` explicit relational edges in `data/wizard_learning/gutenberg_concept_map/gutenberg_concept_graph.jsonl` (`8.09 MB`), keeping disk usage ultra-compact while pretraining the **Top `2,000` Foundational Anchor Concepts** directly into Rust `:8095` (`brain.wbrain` checkpointed).

---

## 3. Genuine End-to-End Execution & Consolidation Loop

For any imperative modeling/code request (`"Create a python class for a bike."`, `"Create a model of a sextant in Python."`, `"Write a Rust struct for a submarine."`, etc.):

1. **Direct Rust `:8095` Neural Recall First**: Queries `POST http://127.0.0.1:8095/brain/chat`. When grounded (`outside_grounding == False`) with a verified code artifact in the target language, returns directly from the Rust neural fabric in milliseconds.
2. **Offline-First Gutenberg Concept Graph + Open-Licensed Wikidata/Wikipedia Mechanics**:
   - Queries `GutenbergConceptMap` (`< 0.1 ms`, zero network I/O) for `is_a`, `has_part`, `used_for`, `roget_classes`, and `domain`.
   - When `allow_web_lookup=True`, merges live Wikidata (`CC0 1.0` `P279`/`P527`/`P366`) and Wikipedia (`CC BY-SA 4.0`) edges.
3. **Compositional Synthesis + Isolated Subprocess Verification**:
   - Maps discovered constituents (`ComponentState` preconditions where faulting a required part raises `RuntimeError`) and physical/logical traits (`kinematic_motion`, `geared_drivetrain`, `braking_system`, `steering_heading`, `electrical_power`, `thermal_fluid`, `optical_focus`, `calibrated_instrument`, `vertical_level`, `discrete_inventory`) into typed code.
   - Parses AST (`ast.parse`) and executes Python models in an isolated `python -I -c` subprocess (`exit 0`).
4. **Hebbian Consolidation into Rust `:8095`**:
   - Trains verified `(sensory_key, code)` bindings via `/brain/pretrain_bindings` + `/flush` (`brain.wal`) + `/brain/checkpoint` (`brain.wbrain`), so subsequent queries — even after a hard process kill and cold restart — recall directly from the Rust neural substrate.
