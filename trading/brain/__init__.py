"""
High-level brain package that hosts the self-refining control logic.

The design is intentionally modular so components can evolve independently:

* neuro_graph.py  – dynamic dependency graph of assets/news/portfolio nodes
* swarm.py        – multi-resolution ensemble that produces directional votes
* event_engine.py – reflex/feedback wiring for fast risk responses
* memory.py       – pattern memory bank and feedforward boosts
* scenario.py     – scenario reactor (optimistic / pessimistic / neutral)
* arb_cell.py     – specialised volatility arbitrage cell (e.g. ETH↔stablecoin)

Each module exposes lightweight APIs that the trading bot can orchestrate.
The implementations below favour CPU-friendly numpy-based operations so the
system remains viable on a modest i5 + 32 GB RAM environment.
"""

from .neuro_graph import NeuroGraph
from .swarm import MultiResolutionSwarm
from .event_engine import EventEngine
from .memory import PatternMemory
from .scenario import ScenarioReactor
from .arb_cell import VolatilityArbCell

__all__ = [
    "NeuroGraph",
    "MultiResolutionSwarm",
    "EventEngine",
    "PatternMemory",
    "ScenarioReactor",
    "VolatilityArbCell",
]
