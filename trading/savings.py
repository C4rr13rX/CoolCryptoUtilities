from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SavingsEvent:
    token: str
    amount: float
    mode: str
    reason: str
    equilibrium_score: float
    trade_id: str
    timestamp: float
    chain: str = ""
    min_batch: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class StableSavingsPlanner:
    def __init__(self, *, min_batch: float = 50.0) -> None:
        self.min_batch = max(1.0, float(min_batch))
        self._buckets: Dict[Tuple[str, str, str], float] = defaultdict(float)
        self._pending_meta: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    def record_allocation(
        self,
        *,
        amount: float,
        token: str,
        mode: str,
        equilibrium_score: float,
        trade_id: str,
        reason: str = "profit-share",
        chain: Optional[str] = None,
        min_batch_override: Optional[float] = None,
    ) -> SavingsEvent:
        token_u = token.upper()
        chain_key = (chain or "global").lower()
        key = (mode, token_u, chain_key)
        self._buckets[key] += max(0.0, float(amount))
        min_batch = self.min_batch
        existing_meta = self._pending_meta.get(key) or {}
        if "min_batch" in existing_meta:
            try:
                min_batch = float(existing_meta["min_batch"])
            except (TypeError, ValueError):
                min_batch = self.min_batch
        if min_batch_override is not None:
            try:
                min_batch = float(min_batch_override)
            except (TypeError, ValueError):
                min_batch = self.min_batch
        min_batch = max(1.0, min(float(min_batch), self.min_batch))
        self._pending_meta[key] = {
            "equilibrium_score": float(equilibrium_score),
            "updated": time.time(),
            "min_batch": min_batch,
        }
        event = SavingsEvent(
            token=token_u,
            amount=float(amount),
            mode=mode,
            reason=reason,
            equilibrium_score=float(equilibrium_score),
            trade_id=trade_id,
            timestamp=time.time(),
            chain=chain_key,
            min_batch=min_batch,
        )
        return event

    def drain_ready_transfers(self) -> List[SavingsEvent]:
        transfers: List[SavingsEvent] = []
        now = time.time()
        for key, total in list(self._buckets.items()):
            meta = self._pending_meta.get(key) or {}
            min_batch = meta.get("min_batch", self.min_batch)
            try:
                min_batch_val = float(min_batch)
            except (TypeError, ValueError):
                min_batch_val = self.min_batch
            min_batch_val = max(1.0, min(min_batch_val, self.min_batch))
            if total < min_batch_val:
                continue
            mode, token, chain = key
            meta = self._pending_meta.get(key, {})
            transfers.append(
                SavingsEvent(
                    token=token,
                    amount=total,
                    mode=mode,
                    reason="scheduled-transfer",
                    equilibrium_score=float(meta.get("equilibrium_score", 0.0)),
                    trade_id=f"savings-{mode}-{token}-{int(now)}",
                    timestamp=now,
                    chain=chain,
                    min_batch=min_batch_val,
                )
            )
            self._buckets[key] = 0.0
        return transfers
