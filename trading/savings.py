from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


@dataclass
class SavingsEvent:
    token: str
    amount: float
    mode: str
    reason: str
    equilibrium_score: float
    trade_id: str
    timestamp: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class StableSavingsPlanner:
    def __init__(self, *, min_batch: float = 50.0) -> None:
        self.min_batch = max(1.0, float(min_batch))
        self._buckets: Dict[Tuple[str, str], float] = defaultdict(float)
        self._pending_meta: Dict[Tuple[str, str], Dict[str, float]] = {}

    def record_allocation(
        self,
        *,
        amount: float,
        token: str,
        mode: str,
        equilibrium_score: float,
        trade_id: str,
        reason: str = "profit-share",
    ) -> SavingsEvent:
        token_u = token.upper()
        key = (mode, token_u)
        self._buckets[key] += max(0.0, float(amount))
        self._pending_meta[key] = {
            "equilibrium_score": float(equilibrium_score),
            "updated": time.time(),
        }
        event = SavingsEvent(
            token=token_u,
            amount=float(amount),
            mode=mode,
            reason=reason,
            equilibrium_score=float(equilibrium_score),
            trade_id=trade_id,
            timestamp=time.time(),
        )
        return event

    def drain_ready_transfers(self) -> List[SavingsEvent]:
        transfers: List[SavingsEvent] = []
        now = time.time()
        for key, total in list(self._buckets.items()):
            if total < self.min_batch:
                continue
            mode, token = key
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
                )
            )
            self._buckets[key] = 0.0
        return transfers
