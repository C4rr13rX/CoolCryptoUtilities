#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_django(settings_module: str) -> None:
    os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
    os.environ.setdefault("BRANDDOZER_FAST_BASELINE", "1")
    os.environ.setdefault("DJANGO_DB_VENDOR", "sqlite")
    os.environ.setdefault("DJANGO_PREFER_SQLITE_FALLBACK", "1")
    default_sqlite = PROJECT_ROOT / "runtime" / "branddozer" / "branddozer.db"
    os.environ.setdefault("DJANGO_SQLITE_PATH", str(default_sqlite.resolve()))
    try:
        default_sqlite.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    from services.env_loader import EnvLoader

    EnvLoader.load()
    import django

    django.setup()
    _ensure_sqlite_ready()


def _build_prompt(run_id: Optional[str]) -> str:
    target = f"Run: {run_id}" if run_id else "Run: (none specified; select the most relevant active run)"
    return (
        "You are the BrandDozer Auto-Upgrader. Act directly via the repo to improve the delivery system itself "
        "and the active trading pipeline run. Make safe, incremental changes without asking for permission.\n\n"
        f"{target}\n"
        "- Inspect BrandDozer (delivery orchestrator, gates, UX audit, backlog flow) and the trading pipeline state.\n"
        "- Fix blockers keeping baseline/gates from completing; scope/skip heavy scans if they stall; restart the run if needed after applying fixes.\n"
        "- Ensure UX snapshots + README are emitted for UI gates.\n"
        "- Strengthen trading pipeline goals: ghost-trade validation, safe promotion to live with minimal capital, risk/loss prevention, sparse-wallet handling, bus scheduling swaps.\n"
        "- Keep guardian/production stable; do not touch monitoring_guardian/ or tools/codex_session.py.\n"
        "- Summarize changes, tests run, and any restarts you triggered.\n"
        "- Keep working in a fix/test loop until no obvious blockers remain.\n"
    )


def _run_once(run_id: Optional[str], workdir: Path) -> str:
    from tools.codex_session import CodexSession, codex_default_settings

    session = CodexSession(
        session_name="branddozer-auto-upgrader",
        transcript_dir=Path("runtime/branddozer/auto_upgrade_transcripts"),
        read_timeout_s=None,
        workdir=str(workdir),
        **codex_default_settings(),
    )
    prompt = _build_prompt(run_id)
    return session.send(prompt, stream=False)


def _ensure_sqlite_ready() -> None:
    """
    Best-effort initialization for local sqlite runs so BrandDozer can persist runs
    even when Postgres is unavailable. No-ops if migrations are already applied.
    """
    try:
        from django.core.management import call_command
        from django.db import connection
    except Exception:
        return
    try:
        tables = set(connection.introspection.table_names())
    except Exception:
        tables = set()
    brand_tables = {t for t in tables if t.startswith("branddozer_")}
    if brand_tables and "django_migrations" in tables:
        return
    try:
        call_command("migrate", interactive=False, verbosity=0)
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous BrandDozer auto-upgrade loop using CodexSession.")
    parser.add_argument("--run-id", help="Active delivery run id to target (optional).")
    parser.add_argument("--settings", default="coolcrypto_dashboard.settings", help="Django settings module.")
    parser.add_argument("--interval", type=int, default=1200, help="Seconds between upgrade passes.")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit.")
    args = parser.parse_args()

    _load_django(args.settings)
    workdir = PROJECT_ROOT
    out_dir = Path("runtime/branddozer/auto_upgrade")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.once:
        output = _run_once(args.run_id, workdir)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"auto_upgrade_{ts}.log"
        path.write_text(output, encoding="utf-8")
        print(f"Wrote auto-upgrade output to {path}")
        return

    print(f"Starting auto-upgrade loop (interval {args.interval}s)...")
    while True:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output = _run_once(args.run_id, workdir)
        path = out_dir / f"auto_upgrade_{ts}.log"
        path.write_text(output, encoding="utf-8")
        print(f"[{ts}] Wrote auto-upgrade output to {path}")
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
