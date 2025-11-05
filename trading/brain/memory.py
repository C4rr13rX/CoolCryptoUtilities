from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MemoryEntry:
    vector: np.ndarray
    profit: float
    metadata: Dict[str, float]


class PatternMemory:
    """
    Feedforward memory bank storing feature fingerprints for profitable trades.

    Similar vectors can be matched quickly to bias new decisions. The implementation
    uses cosine similarity and keeps memory compact by evicting low-utility entries.
    """

    def __init__(self, dim: int, max_entries: int = 512) -> None:
        self.dim = dim
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []

    def add(self, vector: np.ndarray, profit: float, **meta: float) -> None:
        if vector.shape[0] != self.dim:
            return
        entry = MemoryEntry(vector=vector.astype(np.float32), profit=float(profit), metadata=dict(meta))
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: abs(e.profit), reverse=True)
            self.entries = self.entries[: self.max_entries]

    def match(self, vector: np.ndarray, top_k: int = 3) -> Optional[Tuple[float, Dict[str, float]]]:
        if not self.entries or vector.shape[0] != self.dim:
            return None
        vec = vector.astype(np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0.0:
            return None
        similarities: List[Tuple[float, MemoryEntry]] = []
        for entry in self.entries:
            dot = float(np.dot(entry.vector, vec))
            denom = float(np.linalg.norm(entry.vector) * vec_norm)
            if denom == 0.0:
                continue
            similarities.append((dot / denom, entry))
        if not similarities:
            return None
        similarities.sort(key=lambda x: x[0], reverse=True)
        top = similarities[:top_k]
        weighted_profit = sum(sim * entry.profit for sim, entry in top)
        sum_weights = sum(abs(sim) for sim, _ in top)
        if sum_weights == 0.0:
            return None
        score = weighted_profit / sum_weights
        meta = {}
        for sim, entry in top:
            for k, v in entry.metadata.items():
                meta[k] = meta.get(k, 0.0) + sim * v
        return score, meta
