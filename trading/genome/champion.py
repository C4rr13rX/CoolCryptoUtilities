"""Load the current market-evolution champion and score live features.

Reads ``runtime/market-evolution/champion.json`` from the GA repo, so the
strategy always ghost-trades whichever genome the GA currently favours --
no copy to keep in sync.

Scoring reuses the GA's own model reconstruction. A genome is a feature
list plus learner hyperparameters; the fitted model lives only inside an
evaluation run, so live scoring refits it on the genome's training window.
When that is not possible the champion reports unavailable rather than
guessing -- an unvalidated signal is worse than no signal.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_GENOME_REPO = Path(os.getenv("W1Z4RD_REPO_ROOT", r"D:\Projects\W1z4rDV1510n"))
_CHAMPION_PATH = _GENOME_REPO / "runtime" / "market-evolution" / "champion.json"

# The champion file is rewritten every generation; re-reading it on every
# evaluation would hammer the disk, so cache briefly.
_CACHE_SECONDS = float(os.getenv("GENOME_CHAMPION_CACHE_SECONDS", "60"))


@dataclass
class ChampionGenome:
    """A GA champion, as far as the live pipeline needs to know it."""

    genome_id: str
    features: List[str] = field(default_factory=list)
    learner_kind: str = ""
    confidence_quantile: float = 0.25
    profit_factor: float = 0.0
    coverage: float = 0.0
    evaluated_folds: int = 0
    expectancy: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def strategy_id(self) -> str:
        """Ledger identity.

        Deliberately includes the genome id: a new champion must earn its
        OWN ghost record rather than inherit the incumbent's graduation.
        """
        return f"genome_{self.genome_id[:12]}"

    def missing_features(self, available: Dict[str, float]) -> List[str]:
        return [name for name in self.features if name not in available]

    def is_scorable(self, available: Dict[str, float]) -> bool:
        """Every feature must be present.

        continuous_features defaults absent keys to 0.0, which silently
        shifts the model off its training distribution, so require the full
        vector instead of tolerating gaps.
        """
        return bool(self.features) and not self.missing_features(available)


_cache: Dict[str, Any] = {"loaded_at": 0.0, "champion": None, "mtime": 0.0}


def load_champion(path: Optional[Path] = None) -> Optional[ChampionGenome]:
    """Return the current champion, or None when unavailable."""
    target = Path(path) if path else _CHAMPION_PATH
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return None

    now = time.time()
    if (_cache["champion"] is not None
            and _cache["mtime"] == mtime
            and now - _cache["loaded_at"] < _CACHE_SECONDS):
        return _cache["champion"]

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    result = payload.get("result") or {}
    summary = result.get("summary") or {}
    champion = ChampionGenome(
        genome_id=str(payload.get("genome_id") or ""),
        features=list(payload.get("features") or []),
        learner_kind=str(payload.get("learner_kind") or ""),
        confidence_quantile=float(payload.get("confidence_quantile") or 0.25),
        profit_factor=float(summary.get("min_profit_factor") or 0.0),
        coverage=float(summary.get("min_coverage") or 0.0),
        evaluated_folds=int(result.get("evaluated_folds") or 0),
        expectancy=float(summary.get("min_expectancy") or 0.0),
        raw=payload,
    )
    if not champion.genome_id or not champion.features:
        return None

    _cache.update({"loaded_at": now, "champion": champion, "mtime": mtime})
    return champion


def champion_meets_objective(champion: ChampionGenome) -> bool:
    """Is this genome good enough to be worth ghost-trading at all?

    Mirrors the GA's own objective: real profit, measured on the full
    walk-forward rather than a lucky fold.
    """
    objective = float(os.getenv("GENOME_MIN_PROFIT_FACTOR", "1.10"))
    min_folds = int(os.getenv("GENOME_MIN_FOLDS", "3"))
    return (champion.profit_factor >= objective
            and champion.evaluated_folds >= min_folds
            and champion.expectancy > 0.0)
