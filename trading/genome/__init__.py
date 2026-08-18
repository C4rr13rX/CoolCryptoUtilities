"""Live scoring for W1z4rDV1510n market-evolution genomes.

The GA evolves genomes against an offline feature set (futures basis, funding
rates, news sentiment, cross-sectional breadth) that a live RouteState price
window does not carry. This package builds those same features from live data
so an evolved champion can emit ghost trades and earn a real ledger record.

Design rule: the feature builders are IMPORTED from the GA repo rather than
reimplemented. A reimplementation would drift, and a genome scored on features
that differ from its training distribution is not the genome that was
measured -- it would produce a ghost record for a model nobody validated.
"""
from trading.genome.features import (
    GENOME_REPO_AVAILABLE,
    LiveFeatureBuilder,
    build_live_features,
)
from trading.genome.champion import ChampionGenome, load_champion

__all__ = [
    "GENOME_REPO_AVAILABLE",
    "LiveFeatureBuilder",
    "build_live_features",
    "ChampionGenome",
    "load_champion",
]
