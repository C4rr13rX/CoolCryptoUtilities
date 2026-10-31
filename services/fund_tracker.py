"""services/fund_tracker.py
Cross-process unified fund reservation and outbound transfer tracker.

Tracks:
1. Outbound transfers to external wallets, cold vaults, or cross-chain bridges.
2. In-flight trade directives, swap reservations, and balance locks.
3. Real-time spendable balance deductions for BusScheduler and PortfolioState
   to guarantee trading routines never attempt to swap funds that are locked,
   transferred, or committed elsewhere.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.atomic_json import file_lock, read_json, write_json

DEFAULT_FUND_TRACKER_PATH = Path("data/fund_reservations.json")


class FundTracker:
    """Manages real-time reservations and outbound fund movements."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_FUND_TRACKER_PATH
        self._mem_lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_mtime: float = 0.0

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "updated_at": 0.0,
            "reservations": {},
            "outbound_transfers": [],
        }

    def _load_state_unlocked(self) -> Dict[str, Any]:
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else 0.0
            if self._cache is not None and mtime == self._cache_mtime:
                return self._cache
        except OSError:
            pass

        data, ok = read_json(self.path, default=self._empty_state())
        if not ok or not isinstance(data, dict):
            data = self._empty_state()

        if "reservations" not in data or not isinstance(data["reservations"], dict):
            data["reservations"] = {}
        if "outbound_transfers" not in data or not isinstance(data["outbound_transfers"], list):
            data["outbound_transfers"] = []

        self._cache = data
        try:
            self._cache_mtime = self.path.stat().st_mtime if self.path.exists() else 0.0
        except OSError:
            self._cache_mtime = 0.0
        return data

    def _save_state_unlocked(self, state: Dict[str, Any]) -> bool:
        state["updated_at"] = time.time()
        ok = write_json(self.path, state)
        if ok:
            self._cache = state
            try:
                self._cache_mtime = self.path.stat().st_mtime
            except OSError:
                self._cache_mtime = time.time()
        return ok

    # ------------------------------------------------------------------
    # Token matching & normalization
    # ------------------------------------------------------------------

    def normalize_token(self, token: str, chain: str = "ethereum") -> str:
        s = (token or "").strip()
        if not s:
            return ""
        if s.startswith("0x") and len(s) == 42:
            return s.lower()
        su = s.upper()
        if su in {"ETH", "WETH", "NATIVE"} and chain.lower() in {"ethereum", "arbitrum", "optimism", "base"}:
            return "ETH"
        if su in {"MATIC", "POL", "WMATIC"} and chain.lower() == "polygon":
            return "MATIC"
        return su

    def tokens_match(self, t1: str, t2: str, chain: str = "ethereum") -> bool:
        n1 = self.normalize_token(t1, chain)
        n2 = self.normalize_token(t2, chain)
        if not n1 or not n2:
            return False
        if n1 == n2:
            return True

        # Try mapping symbols to addresses via core_tokens_for_chain
        try:
            from services.token_catalog import core_tokens_for_chain
            token_map = core_tokens_for_chain(chain.lower())
            addr1 = token_map.get(n1, "").lower() if not n1.startswith("0x") else n1
            addr2 = token_map.get(n2, "").lower() if not n2.startswith("0x") else n2
            if addr1 and addr2 and addr1 == addr2:
                return True
        except Exception:
            pass

        return False

    # ------------------------------------------------------------------
    # Outbound transfers
    # ------------------------------------------------------------------

    def record_outbound_transfer(
        self,
        *,
        chain: str,
        token: str,
        amount: float,
        destination: str,
        reason: str,
        tx_hash: Optional[str] = None,
        source: str = "transfer",
        ttl_seconds: float = 3600.0,
    ) -> Dict[str, Any]:
        """Record an outbound transfer (manual send, bridge out, vault sweep).

        Deducts from available quantity until balance cache refreshes on-chain.
        """
        amount_val = max(0.0, float(amount))
        ch = chain.lower().strip()
        tok = self.normalize_token(token, ch)
        now = time.time()

        transfer_id = f"xfer_{uuid.uuid4().hex[:12]}"
        record = {
            "id": transfer_id,
            "chain": ch,
            "token": tok,
            "token_raw": str(token).strip(),
            "amount": amount_val,
            "destination": str(destination).strip(),
            "reason": str(reason).strip(),
            "source": str(source).strip(),
            "tx_hash": str(tx_hash).strip() if tx_hash else "",
            "timestamp": now,
            "expires_at": now + max(60.0, float(ttl_seconds)),
            "status": "active",
        }

        with self._mem_lock:
            with file_lock(self.path):
                state = self._load_state_unlocked()
                # Deduplicate by tx_hash if provided
                if tx_hash:
                    for existing in state.get("outbound_transfers", []):
                        if existing.get("tx_hash") and existing.get("tx_hash").lower() == tx_hash.lower():
                            return existing
                state.setdefault("outbound_transfers", []).append(record)
                # Keep last 200 transfers to avoid unbounded file growth
                if len(state["outbound_transfers"]) > 200:
                    state["outbound_transfers"] = state["outbound_transfers"][-200:]
                self._save_state_unlocked(state)

        return record

    # ------------------------------------------------------------------
    # In-flight trade reservations
    # ------------------------------------------------------------------

    def reserve_funds(
        self,
        *,
        chain: str,
        token: str,
        amount: float,
        reason: str,
        reservation_id: Optional[str] = None,
        ttl_seconds: float = 180.0,
    ) -> Dict[str, Any]:
        """Lock funds for an in-flight directive or pending swap."""
        amount_val = max(0.0, float(amount))
        ch = chain.lower().strip()
        tok = self.normalize_token(token, ch)
        now = time.time()
        res_id = reservation_id or f"res_{uuid.uuid4().hex[:12]}"

        record = {
            "id": res_id,
            "chain": ch,
            "token": tok,
            "token_raw": str(token).strip(),
            "amount": amount_val,
            "reason": str(reason).strip(),
            "timestamp": now,
            "expires_at": now + max(15.0, float(ttl_seconds)),
            "status": "active",
        }

        with self._mem_lock:
            with file_lock(self.path):
                state = self._load_state_unlocked()
                state.setdefault("reservations", {})[res_id] = record
                self._save_state_unlocked(state)

        return record

    def release_reservation(self, reservation_id: str) -> bool:
        """Release a reservation upon fill, cancellation, or error."""
        if not reservation_id:
            return False
        with self._mem_lock:
            with file_lock(self.path):
                state = self._load_state_unlocked()
                reservations = state.get("reservations", {})
                if reservation_id in reservations:
                    reservations.pop(reservation_id, None)
                    self._save_state_unlocked(state)
                    return True
        return False

    def record_trade_execution(
        self,
        *,
        chain: str,
        sell_token: str,
        sell_amount: float,
        buy_token: str,
        tx_hash: Optional[str] = None,
        reservation_id: Optional[str] = None,
        source: str = "swap_execution",
    ) -> Dict[str, Any]:
        """Record trade execution, releasing any reservation and holding sell deduction."""
        if reservation_id:
            self.release_reservation(reservation_id)

        return self.record_outbound_transfer(
            chain=chain,
            token=sell_token,
            amount=sell_amount,
            destination=f"swap:{buy_token}",
            reason=f"swap_outbound:{buy_token}",
            tx_hash=tx_hash,
            source=source,
            ttl_seconds=1200.0,  # 20 minutes max until cache refresh
        )

    # ------------------------------------------------------------------
    # Query locked amounts & spendable balances
    # ------------------------------------------------------------------

    def get_locked_amount(
        self, chain: str, token: str, current_cache_ts: Optional[float] = None
    ) -> float:
        """Calculate total locked/pending amount for (chain, token).

        If current_cache_ts is provided:
        Transfers that completed before cache_ts - 5.0 are already factored into
        the cached balance, so they are not double-deducted.
        """
        ch = chain.lower().strip()
        now = time.time()
        total_locked = 0.0

        with self._mem_lock:
            state = self._load_state_unlocked()

        # 1. In-flight reservations
        reservations = state.get("reservations", {})
        for res_id, r in list(reservations.items()):
            if r.get("status") != "active":
                continue
            if r.get("expires_at", 0.0) < now:
                continue
            if r.get("chain") == ch and self.tokens_match(r.get("token", ""), token, ch):
                total_locked += float(r.get("amount", 0.0))

        # 2. Outbound transfers
        outbound = state.get("outbound_transfers", [])
        for xfer in outbound:
            if xfer.get("status") != "active":
                continue
            if xfer.get("expires_at", 0.0) < now:
                continue
            # If cache refreshed AFTER transfer occurred, on-chain cache already reflects deduction
            xfer_ts = float(xfer.get("timestamp", 0.0))
            if current_cache_ts and current_cache_ts > (xfer_ts + 5.0):
                continue
            if xfer.get("chain") == ch and self.tokens_match(xfer.get("token", ""), token, ch):
                total_locked += float(xfer.get("amount", 0.0))

        return max(0.0, round(total_locked, 6))

    def reconcile_with_cache(self, chain: str, cache_ts: float) -> int:
        """Mark transfers on `chain` older than cache_ts - 5.0 as reconciled."""
        ch = chain.lower().strip()
        reconciled_count = 0
        now = time.time()

        with self._mem_lock:
            with file_lock(self.path):
                state = self._load_state_unlocked()
                # Clean up expired reservations
                reservations = state.get("reservations", {})
                expired_res = [k for k, v in reservations.items() if v.get("expires_at", 0.0) < now]
                for k in expired_res:
                    reservations.pop(k, None)

                # Reconcile transfers
                for xfer in state.get("outbound_transfers", []):
                    if xfer.get("chain") == ch and xfer.get("status") == "active":
                        xfer_ts = float(xfer.get("timestamp", 0.0))
                        if cache_ts > (xfer_ts + 5.0):
                            xfer["status"] = "reconciled"
                            xfer["reconciled_at"] = cache_ts
                            reconciled_count += 1
                        elif xfer.get("expires_at", 0.0) < now:
                            xfer["status"] = "expired"

                self._save_state_unlocked(state)

        return reconciled_count

    def get_available_quantity(
        self, chain: str, token: str, raw_balance: float, current_cache_ts: Optional[float] = None
    ) -> float:
        """Deduct locked / transferred funds from raw balance."""
        locked = self.get_locked_amount(chain, token, current_cache_ts=current_cache_ts)
        return max(0.0, float(raw_balance) - locked)

    def get_summary(self, chain: Optional[str] = None) -> Dict[str, Any]:
        """Return diagnostic breakdown of active fund locks and transfers."""
        now = time.time()
        ch = chain.lower().strip() if chain else None

        with self._mem_lock:
            state = self._load_state_unlocked()

        active_res = []
        for r in state.get("reservations", {}).values():
            if r.get("status") == "active" and r.get("expires_at", 0.0) >= now:
                if not ch or r.get("chain") == ch:
                    active_res.append(r)

        active_transfers = []
        recent_transfers = []
        for x in reversed(state.get("outbound_transfers", [])):
            if not ch or x.get("chain") == ch:
                if x.get("status") == "active" and x.get("expires_at", 0.0) >= now:
                    active_transfers.append(x)
                if len(recent_transfers) < 20:
                    recent_transfers.append(x)

        # Locked sums per token
        locked_by_token: Dict[str, float] = {}
        for r in active_res:
            key = f"{r.get('chain')}:{r.get('token')}"
            locked_by_token[key] = locked_by_token.get(key, 0.0) + float(r.get("amount", 0.0))
        for x in active_transfers:
            key = f"{x.get('chain')}:{x.get('token')}"
            locked_by_token[key] = locked_by_token.get(key, 0.0) + float(x.get("amount", 0.0))

        return {
            "chain_filter": ch,
            "active_reservations_count": len(active_res),
            "active_transfers_count": len(active_transfers),
            "locked_totals": {k: round(v, 6) for k, v in locked_by_token.items()},
            "active_reservations": active_res,
            "active_transfers": active_transfers,
            "recent_transfers": recent_transfers,
        }

    def clear_all(self) -> None:
        """Clear all reservations and transfers (testing / administrative reset)."""
        with self._mem_lock:
            with file_lock(self.path):
                self._save_state_unlocked(self._empty_state())


_fund_tracker_instance: Optional[FundTracker] = None


def get_fund_tracker(path: Optional[Path] = None) -> FundTracker:
    """Singleton getter for the global FundTracker."""
    global _fund_tracker_instance
    if _fund_tracker_instance is None or path is not None:
        _fund_tracker_instance = FundTracker(path=path)
    return _fund_tracker_instance
