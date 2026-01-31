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
