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

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_GENOME_REPO = Path(os.getenv("W1Z4RD_REPO_ROOT", r"D:\Projects\W1z4rDV1510n"))
_CHAMPION_PATH = _GENOME_REPO / "runtime" / "market-evolution" / "champion.json"

# The champion file is rewritten every generation and re-parsing it on every
# evaluation is the cost worth avoiding, so the parsed genome is cached against
# a hash of the file's bytes. There is deliberately no time-to-live: a TTL here
# only ever delayed a handover the bot is supposed to see immediately, and it
# could not detect the change anyway (see load_champion).
#
# GENOME_CHAMPION_CACHE_SECONDS is read only to warn that it no longer does
# anything, rather than letting a set value imply staleness that is not there.
if os.getenv("GENOME_CHAMPION_CACHE_SECONDS"):
    print(
        "[genome] GENOME_CHAMPION_CACHE_SECONDS is ignored; the champion cache "
        "is invalidated by content, not by elapsed time",
        file=sys.stderr,
    )


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


_cache: Dict[str, Any] = {"loaded_at": 0.0, "champion": None, "key": None}


def load_champion(path: Optional[Path] = None) -> Optional[ChampionGenome]:
    """Return the current champion, or None when unavailable."""
    target = Path(path) if path else _CHAMPION_PATH
    # Validated on the file's CONTENT, not its timestamp.
    #
    # This was an mtime check, and mtime cannot see the change that matters
    # here. A champion is rewritten in place every few generations, and two
    # genomes serialise to the same length, so a handover is a same-path,
    # same-size rewrite -- and on Windows st_mtime is coarse enough that it
    # lands inside a single tick. Every field the cache compared was therefore
    # identical across a real swap, and it kept serving the OLD genome for up
    # to 60 seconds. That attributes one genome's trades to another's ledger,
    # which is precisely the corruption this module exists to prevent. The
    # cached entry also ignored WHICH path was asked for, so a second file
    # could be answered with the first one's champion.
    #
    # Hashing costs one read of a few hundred bytes; what the cache actually
    # saves is the parse and the dataclass construction, and those are still
    # skipped on a hit.
    try:
        raw = target.read_bytes()
    except OSError:
        return None
    key = (str(target), hashlib.sha256(raw).hexdigest())
    if _cache["champion"] is not None and _cache["key"] == key:
        return _cache["champion"]

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
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

    _cache.update({"loaded_at": time.time(), "champion": champion, "key": key})
    return champion


def champion_meets_objective(champion: ChampionGenome) -> bool:
    """Is this genome good enough to be worth ghost-trading at all?

    Mirrors the GA's own objective: real profit, measured on the full
    walk-forward rather than a lucky fold.
    """
    # Prefer the GA's own declared objective so the two cannot drift apart.
    # A live gate looser than the GA would trade genomes the search does not
    # endorse; a stricter one would silently ignore champions it promoted.
    # The env var still wins, for deliberate divergence.
    default = 1.05
    try:  # pragma: no cover - depends on the sibling GA checkout
        from scripts.market_evolution_service import (  # type: ignore
            OBJECTIVE_PROFIT_FACTOR,
        )
        default = float(OBJECTIVE_PROFIT_FACTOR)
    except Exception:
        pass
    objective = float(os.getenv("GENOME_MIN_PROFIT_FACTOR", "") or default)
    min_folds = int(os.getenv("GENOME_MIN_FOLDS", "3"))
    return (champion.profit_factor >= objective
            and champion.evaluated_folds >= min_folds
            and champion.expectancy > 0.0)
