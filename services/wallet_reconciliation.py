"""One freshness-aware wallet snapshot for trading and dashboard consumers."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict

from services.wallet_state import load_wallet_state

_refresh_lock = threading.Lock()
_refresh_last_attempt = 0.0


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        try:
            return float(text)
        except (TypeError, ValueError):
            return 0.0


def reconciled_wallet_snapshot(wallet_alias: str = "guardian") -> Dict[str, Any]:
    """Return current wallet facts, or explicitly mark an old cache untrusted.

    The on-chain refresh writes ``storage/wallet_state/state.json`` only after
    completing its balance pass.  Consumers therefore see the last complete
    snapshot, never a mixture of partially updated database rows.
    """
    snapshot = load_wallet_state() or {}
    updated_at = snapshot.get("updated_at")
    updated_epoch = _epoch(updated_at)
    age_seconds = max(0.0, time.time() - updated_epoch) if updated_epoch else None
    max_age = max(5.0, float(os.getenv("WALLET_SNAPSHOT_MAX_AGE_SEC", "180") or 180.0))
    fresh = bool(updated_epoch and age_seconds is not None and age_seconds <= max_age)
    raw_rows = snapshot.get("balances") if isinstance(snapshot.get("balances"), list) else []
    balances = []
    cached_total = 0.0
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        try:
            usd = float(raw.get("usd", raw.get("usd_amount", 0.0)) or 0.0)
        except (TypeError, ValueError):
            usd = 0.0
        try:
            quantity = float(raw.get("quantity") or 0.0)
        except (TypeError, ValueError):
            quantity = 0.0
        if quantity <= 0.0:
            continue
        cached_total += usd
        balances.append({
            "wallet": snapshot.get("wallet") or wallet_alias,
            "chain": str(raw.get("chain") or "").lower(),
            "token": raw.get("token"),
            "symbol": raw.get("symbol"),
            "quantity": raw.get("quantity"),
            "usd_amount": usd,
            "updated_at": raw.get("updated_at") or updated_at,
        })
    return {
        "wallet_alias": wallet_alias,
        "wallet": snapshot.get("wallet") or wallet_alias,
        "updated_at": updated_at,
        "updated_epoch": updated_epoch or None,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age,
        "fresh": fresh,
        "status": "current" if fresh else "stale_refresh_required",
        "total_usd": cached_total if fresh else None,
        "cached_total_usd": cached_total,
        "balances": balances,
    }


def request_wallet_refresh(*, minimum_interval_seconds: float = 45.0) -> bool:
    """Start one non-blocking on-chain reconciliation when the cache is old."""
    global _refresh_last_attempt
    now = time.time()
    with _refresh_lock:
        if now - _refresh_last_attempt < max(5.0, minimum_interval_seconds):
            return False
        _refresh_last_attempt = now

    def worker() -> None:
        try:
            from router_wallet import UltraSwapBridge
            from services.wallet_state import capture_wallet_state

            bridge = UltraSwapBridge()
            # Heartbeat reconciliation is balance-only. Transfer backfills and
            # NFT metadata are independent, much slower jobs and must not make
            # a wallet total remain stale for minutes.
            capture_wallet_state(
                bridge=bridge,
                refresh_transfers=False,
                refresh_nfts=False,
            )
        except Exception:
            # Staleness remains explicit in the public snapshot. The next
            # heartbeat retries after the cooldown rather than publishing a
            # fabricated total.
            pass

    threading.Thread(target=worker, daemon=True, name="wallet-reconciliation").start()
    return True
