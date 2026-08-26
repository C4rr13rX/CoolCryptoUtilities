"""One freshness-aware wallet snapshot for trading and dashboard consumers."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

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


def _prefer_fresher_db_balances(
    snapshot: Dict[str, Any],
    wallet_address: Optional[str] = None,
    *,
    _rows_for_test: Optional[list] = None,
) -> Dict[str, Any]:
    """Rebuild the snapshot from the balances table when it is fresher.

    There are two stores holding the same fact: the ``balances`` table (backed
    by the ``core.Balance`` model) and ``storage/wallet_state/state.json``.
    Only the refresh worker writes the JSON, so a direct balance write --
    a forced rescan, a swap settling, anything that touches the model without
    completing a full wallet pass -- left the two disagreeing with no way to
    reconcile them.

    Observed 2026-08-26: the table held a correct $14.17 on base (8.378 USDC
    plus $5.79 of ETH) while the JSON still reported $6.39 from two hours
    earlier, and the pipeline reads the JSON. It concluded there was no
    capital and refused to trade against funds that were sitting right there.

    Taking the newer of the two makes the model the source of truth without
    giving up the JSON's one real advantage: it is written atomically after a
    complete pass, so it is preferred whenever it is actually current.
    """
    if _rows_for_test is not None:
        rows = [dict(row) for row in _rows_for_test]
    else:
        try:
            from db import get_db

            # sqlite3.Row supports indexing but not .get(); normalise to dicts
            # so this reads the same whether the driver changes or not.
            rows = [
                dict(row)
                for row in (
                    get_db().fetch_balances_flat(
                        wallet=wallet_address, include_zero=False
                    )
                    or []
                )
            ]
        except Exception:
            return snapshot
    if not rows:
        return snapshot

    newest_db = 0.0
    balances: list[Dict[str, Any]] = []
    total_usd = 0.0
    for row in rows:
        try:
            quantity = float(row.get("quantity") or 0.0)
        except (TypeError, ValueError):
            continue
        if quantity <= 0.0:
            continue
        try:
            usd = float(row.get("usd_amount") or 0.0)
        except (TypeError, ValueError):
            usd = 0.0
        # `fetch_balances_flat` does not select the raw `ts` column, so date
        # the row from `updated_at` -- using ts here silently produced 0 for
        # every row, which made the table look infinitely old and handed the
        # decision back to the stale JSON every time.
        newest_db = max(newest_db, _epoch(row.get("updated_at")))
        total_usd += usd
        balances.append({
            "wallet": row.get("wallet"),
            "chain": str(row.get("chain") or "").lower(),
            "token": row.get("token"),
            "symbol": row.get("symbol"),
            "quantity": row.get("quantity"),
            "usd": usd,
            "usd_amount": usd,
            "updated_at": row.get("updated_at"),
        })
    if not balances:
        return snapshot

    json_epoch = _epoch(snapshot.get("updated_at"))
    if json_epoch >= newest_db:
        return snapshot          # the JSON pass is at least as recent

    merged = dict(snapshot)
    merged["balances"] = balances
    merged["updated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest_db)
    )
    totals = dict(merged.get("totals") or {})
    totals["usd"] = round(total_usd, 2)
    merged["totals"] = totals
    merged["source"] = "balances_table"
    return merged


def reconciled_wallet_snapshot(wallet_alias: str = "guardian") -> Dict[str, Any]:
    """Return current wallet facts, or explicitly mark an old cache untrusted.

    The on-chain refresh writes ``storage/wallet_state/state.json`` only after
    completing its balance pass.  Consumers therefore see the last complete
    snapshot, never a mixture of partially updated database rows.
    """
    snapshot = load_wallet_state() or {}
    # The balances table keys rows by wallet ADDRESS, and also holds rows
    # under the literal alias 'guardian' from older writes. Summing both
    # double-counts the same account -- it reported $65.60 for a wallet
    # holding $14.17 -- so scope strictly to the resolved address.
    snapshot = _prefer_fresher_db_balances(snapshot, snapshot.get("wallet"))
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
        except Exception as exc:
            # Staleness remains explicit in the public snapshot. The next
            # heartbeat retries after the cooldown rather than publishing a
            # fabricated total.
            #
            # But say WHY. Swallowing this silently meant a wallet could sit
            # unrefreshed for days with the only symptom being a generic
            # "wallet_snapshot_stale" reason, and no way to tell a transient
            # RPC blip from a hard misconfiguration (a missing
            # MNEMONIC/PRIVATE_KEY raises here and would never self-heal).
            try:
                from services.logging_utils import log_message

                log_message(
                    "wallet-reconciliation",
                    f"balance refresh failed: {type(exc).__name__}: {exc}",
                    severity="warning",
                )
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True, name="wallet-reconciliation").start()
    return True
