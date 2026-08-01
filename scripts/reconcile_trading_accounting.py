#!/usr/bin/env python3
"""Quarantine pre-v2 ghost accounting and start an auditable outcome epoch."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import get_db  # noqa: E402
from services.trading_accounting import ACCOUNTING_VERSION  # noqa: E402


def build_audit(db) -> dict:
    state = db.load_state() or {}
    ghost = state.get("ghost_trading") if isinstance(state, dict) else {}
    ghost = ghost if isinstance(ghost, dict) else {}
    return {
        "created_at": time.time(),
        "reason": (
            "Pre-v2 aggregates combined non-USD quote prices, proposed exits, "
            "non-unique trade IDs, and concurrent state writers."
        ),
        "legacy": {
            "stable_bank": ghost.get("stable_bank"),
            "total_profit": ghost.get("total_profit"),
            "realized_profit": ghost.get("realized_profit"),
            "total_trades": ghost.get("total_trades"),
            "wins": ghost.get("wins"),
            "session_id": ghost.get("session_id"),
            "position_count": len(ghost.get("positions") or {}),
            "accounting_version": ghost.get("accounting_version"),
        },
        "verified": {
            "ghost": db.trade_outcome_summary("ghost"),
            "live": db.trade_outcome_summary("live"),
        },
    }


def _quarantine_file(path: Path, quarantine_root: Path, stamp: int) -> str | None:
    if not path.exists():
        return None
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"{path.stem}.{stamp}{path.suffix}"
    path.replace(target)
    return str(target.relative_to(ROOT))


def apply_reconciliation(db, audit: dict) -> dict:
    stamp = int(time.time())
    quarantine_root = ROOT / "runtime" / "quarantine" / "trading-accounting"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    report_path = quarantine_root / f"legacy-accounting.{stamp}.json"
    report_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    moved = []
    for path in (
        ROOT / "data" / "strategy_ledger.json",
        ROOT / "data" / "reports" / "live_readiness.json",
    ):
        target = _quarantine_file(path, quarantine_root, stamp)
        if target:
            moved.append(target)

    state = db.load_state() or {}
    if not isinstance(state, dict):
        state = {}
    previous = state.get("ghost_trading") if isinstance(state.get("ghost_trading"), dict) else {}
    next_session = max(1, int(previous.get("session_id") or 0) + 1)
    state["ghost_trading"] = {
        "accounting_version": ACCOUNTING_VERSION,
        "accounting_epoch": stamp,
        "legacy_quarantine": {
            "report": str(report_path.relative_to(ROOT)),
            "created_at": audit["created_at"],
            "reason": audit["reason"],
            "moved_files": moved,
        },
        "stable_bank": 0.0,
        "total_profit": 0.0,
        "realized_profit": 0.0,
        "total_trades": 0,
        "wins": 0,
        "positions": {},
        "routes": {},
        "session_id": next_session,
        "active_exposure": {},
        "auto_execute_approved": False,
    }
    db.save_state(state)
    return {
        "applied": True,
        "accounting_version": ACCOUNTING_VERSION,
        "accounting_epoch": stamp,
        "session_id": next_session,
        "report": str(report_path.relative_to(ROOT)),
        "moved_files": moved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply the quarantine/reset after printing the audit.")
    args = parser.parse_args()
    db = get_db()
    audit = build_audit(db)
    result = {"audit": audit, "applied": False}
    if args.apply:
        result["reconciliation"] = apply_reconciliation(db, audit)
        result["applied"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
