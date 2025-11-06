from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class Node:
    key: str
    kind: str  # "asset", "news", "volatility", "portfolio"
    value: float = 0.0
    last_update: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class NeuroGraph:
    """
    Lightweight dependency graph that stores weighted relationships between nodes.

    The edges behave like synapses: repeated co-activation increases weight,
    while inactivity decays it. The graph can be queried for influence maps that
    downstream components (swarm, scheduler) consume to bias decisions.
    """

    def __init__(self, decay: float = 0.995, learning_rate: float = 0.05) -> None:
        self.decay = float(decay)
        self.learning_rate = float(learning_rate)
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self._lock = threading.Lock()

    def upsert_node(self, key: str, kind: str, value: float, ts: float, **meta: float) -> None:
        with self._lock:
            node = self._nodes.get(key)
            if node is None:
                node = Node(key=key, kind=kind, value=value, last_update=ts, metadata=dict(meta))
                self._nodes[key] = node
            else:
                node.value = value
                node.last_update = ts
                node.metadata.update(meta)

    def reinforce(self, source: str, target: str, ts: float, strength: float) -> None:
        if source == target:
            return
        key = (source, target)
        with self._lock:
            prev = self._edges[key]
            updated = prev * self.decay + self.learning_rate * strength
            self._edges[key] = float(max(-1.0, min(1.0, updated)))

    def decay_all(self) -> None:
        with self._lock:
            for edge, weight in list(self._edges.items()):
                new_weight = weight * self.decay
                if abs(new_weight) < 1e-4:
                    self._edges.pop(edge, None)
                else:
                    self._edges[edge] = new_weight

    def influence_map(self, node_key: str, max_depth: int = 2) -> Dict[str, float]:
        """
        Return the accumulated influence of neighbours up to `max_depth`.
        """
        with self._lock:
            if node_key not in self._nodes:
                return {}
            visited = {node_key}
            frontier = [(node_key, 1.0)]
            influence: Dict[str, float] = defaultdict(float)
            for _ in range(max_depth):
                next_frontier = []
                for current, weight in frontier:
                    for (src, dst), edge_weight in self._edges.items():
                        if src != current or dst in visited:
                            continue
                        propagated = weight * edge_weight
                        if abs(propagated) < 1e-5:
                            continue
                        influence[dst] += propagated
                        visited.add(dst)
                        next_frontier.append((dst, propagated))
                frontier = next_frontier
                if not frontier:
                    break
            return dict(influence)

    def connected_pairs(self, kind: Optional[str] = None) -> Iterable[Tuple[str, str, float]]:
        with self._lock:
            for (src, dst), weight in self._edges.items():
                if kind and (self._nodes.get(src, Node("", "", 0, 0)).kind != kind):
                    continue
                yield src, dst, weight

    def strength(self, source: str, target: str) -> float:
        return float(self._edges.get((source, target), 0.0))

    def boost(self, source: str, target: str, impulse: float) -> None:
        """
        External stimulation: directly increase an edge weight (used by news shock router).
        """
        key = (source, target)
        with self._lock:
            self._edges[key] = float(max(-1.0, min(1.0, self._edges[key] + impulse)))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        Capture the current adjacency list for diagnostics.
        """
        graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        with self._lock:
            for (src, dst), weight in self._edges.items():
                graph[src][dst] = weight
        return dict(graph)

    def nodes_snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        Lightweight view of node metadata used by observability tooling.
        """
        with self._lock:
            return {
                key: {
                    "kind": node.kind,
                    "value": node.value,
                    "last_update": node.last_update,
                    "metadata": dict(node.metadata),
                }
                for key, node in self._nodes.items()
            }

    def confidence_adjustment(self, symbol: str) -> float:
        """
        Compute a confidence multiplier for a symbol based on graph saturation.
        """
        influence = self.influence_map(symbol, max_depth=1)
        if not influence:
            return 1.0
        total = sum(abs(w) for w in influence.values())
        # map to [0.8, 1.2]
        return float(0.8 + 0.4 * math.tanh(total))
