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
c0d3r is a CLI/agent layer that turns natural-language objectives into disciplined engineering work by running a closed-loop workflow: it clarifies intent, plans, executes tools, inspects evidence, and iterates until the result matches the goal. It is built to behave like a senior software engineer and research scientist: it reasons about constraints, runs experiments, collects outputs, and records what it did so the work is explainable and repeatable.

### Why it matters
c0d3r is meant to accelerate scientific and engineering progress by:
- **Reducing iteration cost**: faster cycles from hypothesis → implementation → measurement → correction.
- **Improving reproducibility**: structured execution traces, consistent outputs, and captured evidence.
- **Increasing safety**: policy-driven limits on what changes are allowed, with validation and review points.
- **Amplifying discovery**: research-first workflows that can incorporate sources, datasets, and constraints into real code and experiments.

### Capabilities (high-level scope)
- **Evidence-driven planning**: creates actionable plans with checkpoints and explicit acceptance criteria.
- **Scientific-method workflow**: supports hypothesis, experiment design, measurement, and iteration with logged results.
- **Research integration**: can perform targeted web research and track sources to justify decisions (when enabled/required).
- **Local execution loop**: generates and runs terminal commands, streams output, summarizes results, and adapts next steps.
- **Verification hooks**: can run micro-checks (small sanity tests) and full checks (lint/build/test) before declaring success, depending on repo context and configuration.
- **Constraint grounding**: can frame tasks in formal constraints (math/physics/requirements) to reduce ambiguity and guide decisions.
- **Adaptive strategy**: routes into diagnostics, additional research, or alternate tooling based on failures and observed outputs.
- **Audit-friendly logging**: captures command transcripts, exit codes, and relevant output slices (e.g., last N lines) for traceability.
- **Filesystem safety policies**: supports guardrails such as project-root-only mutation and path normalization when strict mode is configured.

### Turing Test + Memory Verification
This repo includes a **modernized Turing-style rubric** and a repeatable obstacle harness that probes STM/LTM recall, multi-turn consistency, and tool-use behavior.
- Rubric: `runtime/c0d3r/turing_rubric.md` and `runtime/c0d3r/turing_rubric.json`
- Harness: `runtime/c0d3r/obstacle_course.ps1`
- Logs: `runtime/c0d3r/obstacle_logs/` and `runtime/c0d3r/turing_eval.json`

Run the harness (Windows):
```powershell
powershell -ExecutionPolicy Bypass -File runtime\c0d3r\obstacle_course.ps1
```
Optional retries:
- `C0D3R_TURING_MAX_ATTEMPTS=3`

### Graph Store (Second DB)
The equation matrix is stored in Django’s primary DB **and** mirrored into an embedded graph database for efficient traversal and query.
- Default graph engine: **Kùzu** (MIT-licensed, embedded, open-source)
- Storage path: `storage/graph/kuzu` (override with `GRAPH_DB_DIR`)
- Sync control: `C0D3R_GRAPH_SYNC_ON_WRITE=1` (disable with `0`)

### Architecture summary (engineering view)
c0d3r implements a multi-stage control loop:
1) **Context synthesis**: scans repo structure and probes the environment (OS, toolchain, runtime).
2) **Constraint framing**: translates the objective into constraints, risks, and measurable outcomes.
3) **Plan and actions**: proposes a plan and produces structured, machine-parseable action outputs (JSON) suitable for execution.
4) **Execute and observe**: runs commands with streaming output and captures head/tail plus full logs.
5) **Evaluate and adjust**: compares observed results to acceptance criteria; retries with bounded attempts when needed.
6) **Verification and close-out**: runs checks appropriate to the change (from lightweight sanity checks to full test suites), then summarizes what changed and what evidence supports correctness.

It is designed to avoid “guess-only” completion: critical steps are grounded in observable outputs (command results, tests, logs) and rerouted into diagnostics or research when evidence is insufficient.

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
- **Biosciences**: reproducible pipelines for field/lab data, rapid prototyping of analyses, and auditable transformations.
- **Computational science**: constraint-driven derivation → tested simulation code with logged assumptions and measurements.
- **Systems engineering**: tool-driven automation with traceable outcomes and policy-controlled risk.
- **Public-good research**: faster iteration on tools for climate, health, education, and accessibility, with captured evidence for review and reuse.

## Requirements
- Python 3.8+ (3.11+ recommended)
- Node.js 18+ (for frontend build)
- npm

## One-command quickstart (all OS)
These scripts install deps, use SQLite by default, add repo CLIs to PATH, migrate, and start the server.
- Windows PowerShell: `scripts\quickstart.ps1`
- Windows CMD: `scripts\quickstart.cmd`
- macOS / Linux: `bash scripts/quickstart.sh`
Legacy/older OS fallback: if full deps fail, the scripts fall back to `requirements_legacy.txt` and then a minimal Django set to get the web UI running.

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
