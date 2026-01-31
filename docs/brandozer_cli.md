# Brandozer CLI

`brandozer` is a small helper for running the BrandDozer delivery runner from the terminal. It assumes your working directory is the project root unless you pass `--root`.

## Setup
- Make sure the repo `bin` directory is on your `PATH` (e.g., `export PATH="$PWD/bin:$PATH"` from the project root).
- The CLI loads Django via `coolcrypto_dashboard.settings`, so run it from an environment where dependencies are installed and database settings are available.

## Usage
```
brandozer --help
brandozer start "Ship the new dashboard UX"          # creates/reuses a project rooted at CWD
brandozer start "Ghost-trade upgrade" --run-id <uuid> # restart or reuse a specific delivery run id
brandozer start "Solo plan loop" --team-mode solo --session-provider codex --codex-model gpt-5.2-codex --codex-reasoning medium
brandozer start "Solo plan loop" --team-mode solo --session-provider c0d3r --c0d3r-model anthropic.claude-3-7-sonnet-20250219-v1:0
brandozer runs --limit 5                              # list recent runs
brandozer status <run-id>                             # show run state and phase
brandozer tail <run-id> --lines 120                   # tail the latest orchestrator log
brandozer stop <run-id>                               # request a stop/cancel on a run
brandozer projects                                    # list known BrandDozer projects
```

- `--root` lets you target a different project path.
- `--no-acceptance` on `start` will skip the acceptance gating step for the run.
- Logs are tailed from the most recent orchestrator session; if none are present, the CLI falls back to the latest session on that run.
- `--session-provider` selects the engine: `codex` (CLI) or `c0d3r` (Bedrock).
- `--team-mode solo` runs a single session that maintains a plan file under `runtime/branddozer/solo_plans/<run-id>/plan.json`.
