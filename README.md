# CoolCryptoUtilities

**Heads up:** Organism, Code Graph, and U53R xR080T are awkward and still in active development.

A Django-first crypto trading dashboard with a full pipeline behind it: data ingest, training, ghost trading, live readiness gates, wallet tooling, and ops consoles. The **Django website in `web/` is the main entrypoint**.

## What’s included
- **Django web UI** for wallet, pipeline status, guardian, telemetry, settings, labs
- **Trading pipeline** with ghost → live promotion gates and risk guardrails
- **Guardian supervisor** for scheduled monitoring and automation
- **Secure settings vault** (encrypted per-user secrets)
- **DataLab + Model Lab** utilities for datasets, indexers, and training runs
- **Wallet tools** for balances, transfers, NFTs, swaps, and (optional) bridging
- **BrandDozer delivery** with selectable AI provider (Codex CLI or c0d3r via AWS Bedrock)

## c0d3r (Bedrock-backed senior engineer + researcher)
c0d3r is the CLI/agent layer that turns natural-language objectives into rigorous, test-verified engineering work. It is designed to behave like a senior software engineer and research scientist: it plans, gathers evidence, applies constraints, runs experiments, executes tooling, and iterates until results are correct. It is built around a **closed-loop, evidence-first control system** that is explicit about what it knows, what it tested, and what it changed.

### Why it matters
c0d3r is meant to accelerate scientific and engineering progress by:
- **Reducing iteration cost**: fast, verified cycles from hypothesis → implementation → test → correction.
- **Improving reproducibility**: structured plans, constraints, and recorded evidence.
- **Making systems safer**: explicit validation, test gating, and auditability of changes.
- **Amplifying discovery**: research-first workflows that can integrate domain papers, datasets, and constraints into real code and experiments.

### Capabilities (high-level scope)
- **Evidence-driven planning**: formal plans with verification checkpoints and requirement checklists.
- **Scientific-method workflows**: hypothesis, experiment, measurement, and iteration with traceable evidence.
- **Research integration**: focused web research + source tracking to inform design and implementation.
- **Local execution loop**: command execution, file editing, and test runs that must succeed before advancing.
- **Microtests + full tests**: per-file micro-experiments (foo data) plus full suite verification.
- **Formal constraints**: math grounding and constraint propagation for rigorous tasks.
- **Dynamic strategy**: capability hooks that adapt the pipeline based on failures or user constraints.
- **Strict filesystem safety**: project-root enforcement, path normalization, and mutation auditing.

### Architecture summary (engineering view)
c0d3r implements a multi-stage control loop:
1) **Context synthesis**: repo scan + environment probe + memory summary.
2) **Constraint grounding**: math/physics framing when tasks are analytical.
3) **Research + evidence**: targeted queries with sources → decision notes.
4) **Plan → execute → verify**: strict state machine; tests gate progress.
5) **Microtests**: class/function-level sanity experiments before proceeding.
6) **Mutation audit**: prevents “no-op” results; forces real changes.
7) **Adaptive routing**: feedback-driven reroutes (research/diagnostics).

It is deliberately designed to **avoid heuristic-only completion**: every critical step is either verified by execution or rerouted to research/diagnostics until it can be verified.

### CLI quick start
```powershell
c0d3r "inspect this repo and summarize issues."
```

Set the provider and model:
```powershell
setx AWS_PROFILE FountainServer
setx AWS_DEFAULT_REGION us-east-1
setx C0D3R_MODEL global.anthropic.claude-sonnet-4-20250514-v1:0
```

### How c0d3r can advance science & engineering
- **Biosciences**: rapid modeling + reproducible pipelines for field and lab data.
- **Computational science**: constraint-driven derivation → tested simulation code.
- **Systems engineering**: auditable automation with verifiable correctness.
- **Public-good research**: faster iteration on tools for climate, health, and education.

### What’s next (roadmap)
To become an even more dynamic scientist/engineer, c0d3r is expanding:
- **Capability graph expansion** (more modular behaviors with verified execution hooks)
- **Stronger multi-model consensus** for high-stakes correctness
- **Richer domain toolchains** (scientific simulation, formal methods, data validation)
- **Matrix-backed research memory** for cross-disciplinary derivations
- **Safer long-running experiments** with automated rollback and provenance tracking

## Requirements
- Python 3.12 (3.11 usually works)
- Node.js 18+ (for frontend build)
- npm

## Quick start (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python web\manage.py migrate
.\.venv\Scripts\python web\manage.py createsuperuser
.\.venv\Scripts\python web\manage.py runserver
```

## Quick start (macOS / Linux)
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python web/manage.py migrate
python web/manage.py createsuperuser
python web/manage.py runserver
```

Open the site at `http://127.0.0.1:8000/` and log in.

## Frontend build
The UI is built in `web/frontend/`.
```bash
cd web/frontend
npm install
npm run build
```

## Developer reseat command
`reseat` runs the full reset + boot flow (frontend build, migrations, supervisors, etc). Useful during development.
```bash
python web/manage.py reseat
```
Common options:
- `--guardian-off` to disable Guardian auto-start
- `--production-off` to disable Production Manager auto-start
- `--noinstall` to skip npm install/build
- `--no-runserver` to skip launching the dev server

## Pipeline setup (UI)
Go to **Settings → Pipeline Wizard** to enter only the missing keys. This is the fastest way to unlock the full pipeline.

## BrandDozer AI providers
BrandDozer can use Codex (default) or the Bedrock-based c0d3r.
- CLI/GUI: `--session-provider codex|c0d3r`
- Env default: `BRANDDOZER_SESSION_PROVIDER=codex`
- Bedrock optional: `C0D3R_MODEL`, `C0D3R_INFERENCE_PROFILE`, `AWS_PROFILE`, `AWS_DEFAULT_REGION`

## Notes
- This repo uses a secure vault for secrets. Avoid committing keys to `.env`.
- The pipeline can run in ghost-only mode without enabling live trading.

---
If you want a slim install profile (no heavy ML libs) or a server deployment guide, ask and I’ll add it.
