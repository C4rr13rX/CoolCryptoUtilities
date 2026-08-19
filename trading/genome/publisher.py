"""Publish live genome signals for the whole asset universe once per tick.

Genome features are cross-sectional -- market breadth is a median across
every asset at one timestamp -- so they cannot be built per-pair inside a
strategy's evaluate(). This computes them once for the universe and hands the
strategy a plain dict to look up, exactly as the swarm consensus signal works.

Scoring reuses the GA's own model. A genome is a feature list plus learner
hyperparameters; the fitted estimator lives only inside an evaluation run and
is never persisted, so the publisher refits it once per champion on the same
dataset the GA measured, then caches it. Refitting on the GA's data rather
than reimplementing the fit is what keeps the live signal the same model that
earned the profit factor.

When the model cannot be produced the asset is reported unscorable and the
strategy abstains. A defaulted feature or an invented direction would be a
confident signal from a model nobody validated -- the exact failure this
whole path exists to avoid.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from trading.genome.champion import (
    ChampionGenome,
    champion_meets_objective,
    load_champion,
)
from trading.genome.features import GENOME_REPO_AVAILABLE, LiveFeatureBuilder

_GENOME_REPO = Path(os.getenv("W1Z4RD_REPO_ROOT", r"D:\Projects\W1z4rDV1510n"))



def _env_float(name: str, default: float) -> float:
    """Read a float knob, falling back when unset or malformed."""
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default

class GenomeSignalPublisher:
    """Build one signal per asset for the current champion."""

    def __init__(self, reference_asset: str = "BTC") -> None:
        self._builder = LiveFeatureBuilder(reference_asset)
        self._last_built = 0.0
        self._cached: Dict[str, Dict[str, Any]] = {}
        self._interval = float(os.getenv("GENOME_SIGNAL_INTERVAL_SECONDS", "300"))
        # Which genome the cached signals belong to, so a champion swap
        # invalidates them immediately instead of trading the previous
        # genome's directions until the interval lapses.
        self._cached_genome_id = ""
        # Fitting is expensive, so hold the model per genome id.
        self._model: Any = None
        self._model_genome_id = ""
        self._model_failed_for = ""

    def available(self) -> bool:
        return GENOME_REPO_AVAILABLE

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _fitted_model(self, champion: ChampionGenome) -> Any:
        """Refit the champion on the GA dataset, cached per genome.

        Returns None when the model cannot be produced; callers must then
        mark assets unscorable rather than guessing a direction.
        """
        if self._model is not None and self._model_genome_id == champion.genome_id:
            return self._model
        if self._model_failed_for == champion.genome_id:
            return None

        # Drop the previous champion's model BEFORE attempting a refit. If the
        # refit fails we must not be left holding a model fitted for a
        # different genome: a later call for that old genome would return it
        # without re-validating, and a champion swap is exactly when this
        # happens. Losing the cache costs one refit; keeping a stale model
        # costs trades scored by the wrong genome.
        self._model = None
        self._model_genome_id = ""

        try:  # pragma: no cover - depends on the sibling GA checkout
            from scripts.market_evolution_service import (  # type: ignore
                fit_live_surrogate,
            )
        except Exception:
            self._model_failed_for = champion.genome_id
            return None

        # A fitted model is worth persisting: producing one loads the whole GA
        # dataset, which is the single heaviest thing this process does.
        cached = self._load_cached_model(champion.genome_id)
        if cached is not None:
            self._model, self._model_genome_id = cached, champion.genome_id
            return cached

        try:
            model = fit_live_surrogate(champion.raw)
        except MemoryError:
            # Resource exhaustion is transient and says nothing about the
            # genome. Latching it in _model_failed_for would mark a perfectly
            # good champion permanently unscorable for the life of the
            # process: measured 2026-08-19 with 0.65 GB free, every one of 33
            # assets went unscorable and stayed that way even after memory
            # was released. Leave the failure unlatched so the next refresh
            # retries.
            return None
        except Exception:
            model = None
        if model is None:
            self._model_failed_for = champion.genome_id
            return None
        self._store_cached_model(champion.genome_id, model)
        self._model, self._model_genome_id = model, champion.genome_id
        return model


    # ------------------------------------------------------------------
    # On-disk model cache
    # ------------------------------------------------------------------
    def _model_cache_path(self, genome_id: str) -> Path:
        root = Path(os.getenv("GENOME_MODEL_CACHE_DIR",
                              str(Path("data") / "genome-models")))
        return root / f"{genome_id}.joblib"

    def _load_cached_model(self, genome_id: str) -> Any:
        """Return a previously fitted model, or None.

        A corrupt or unreadable cache is treated as absent rather than fatal:
        the cost is one refit.
        """
        path = self._model_cache_path(genome_id)
        if not path.exists():
            return None
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            return None

    def _store_cached_model(self, genome_id: str, model: Any) -> None:
        """Persist a fitted model so the next process need not refit.

        Failure here is not an error -- the model is already usable in memory.
        """
        try:
            import joblib

            path = self._model_cache_path(genome_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
        except Exception:
            pass

    def _score(self, champion: ChampionGenome, model: Any,
               values: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Direction, confidence and edge from the genome's own surface."""
        try:  # pragma: no cover - depends on the sibling GA checkout
            from scripts.market_evolution_service import (  # type: ignore
                feature_vector,
                genome_from_dict,
            )
        except Exception:
            return None

        try:
            genome = genome_from_dict(champion.raw)
            vector = np.asarray(
                [feature_vector({"features": values}, genome)], dtype=np.float32
            )
            if not np.all(np.isfinite(vector)):
                return None
            probability = float(model.probability(vector)[0])
            confidence = float(model.selection_confidence(vector)[0])
            direction = int(model.predict(vector)[0])
        except Exception:
            return None

        if not all(map(np.isfinite, (probability, confidence))):
            return None

        # The genome's confidence quantile is its own abstention threshold:
        # below it the GA would not have acted either.
        if confidence < float(champion.confidence_quantile or 0.0):
            direction = 0

        # Asymmetric selectivity: the two sides do not carry the same edge.
        #
        # Measured on the GA's own walk-forward folds (179,603 rows, 33 assets,
        # 4 folds, net of the same 25bps round trip), splitting the confident
        # tail by side:
        #
        #     side          meanPF   minPF    expectancy
        #     SHORT/exit    1.7974   0.7843   0.00837
        #     BOTH          1.4574   0.7506   0.00560
        #     LONG/enter    1.4372   0.7137   0.00499
        #
        # The exit side won 3 of 4 folds outright (1.91 vs 1.04 and 2.02 vs
        # 1.21, losing only in the fold-3 rally) and its advantage survived a
        # cost sweep to 60bps -- 2.4x the real charge -- and four random seeds.
        #
        # This is spot trading, so direction<0 is an exit rather than a true
        # short: the edge is knowing when to SELL. Demanding a stronger signal
        # to buy than to sell concentrates risk where the evidence is, without
        # inventing a short the venue cannot express.
        entry_floor = _env_float("GENOME_ENTRY_CONFIDENCE_FLOOR", 0.0)
        if direction > 0 and entry_floor > 0.0 and confidence < entry_floor:
            direction = 0

        # Expected edge is anchored on the measured expectancy rather than a
        # fresh guess, so live sizing matches what was actually validated.
        expected = abs(float(champion.expectancy or 0.0)) * max(1.0, confidence * 2.0)
        return {
            "direction": direction,
            "confidence": max(0.0, min(1.0, confidence)),
            "direction_prob": max(0.0, min(1.0, probability)),
            "expected_return": expected,
        }

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def build(
        self,
        bars_by_asset: Dict[str, Sequence[Dict[str, Any]]],
        *,
        force: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Return asset -> signal dict, cached between intervals."""
        now = time.time()
        champion = load_champion()
        if champion is None:
            self._cached, self._last_built, self._cached_genome_id = {}, now, ""
            return {}

        # A champion swap must take effect at once. Serving cached signals
        # for up to GENOME_SIGNAL_INTERVAL_SECONDS after a handover would
        # trade the OLD genome's directions under the new genome's ledger id,
        # corrupting the ghost record that decides live promotion.
        fresh = (self._cached
                 and self._cached_genome_id == champion.genome_id
                 and (now - self._last_built) < self._interval)
        if not force and fresh:
            return self._cached

        meets = champion_meets_objective(champion)
        features = self._builder.build(bars_by_asset)
        model = self._fitted_model(champion) if (features and meets) else None

        signals: Dict[str, Dict[str, Any]] = {}
        for asset, values in features.items():
            scorable = champion.is_scorable(values)
            signal: Dict[str, Any] = {
                "genome_id": champion.genome_id,
                "profit_factor": champion.profit_factor,
                # The genome's validated abstention band, so consumers gate on
                # what the search qualified instead of a separate constant.
                "confidence_quantile": float(champion.confidence_quantile or 0.0),
                "coverage": champion.coverage,
                "evaluated_folds": champion.evaluated_folds,
                "meets_objective": meets,
                "scorable": False,
                "direction": 0,
                "confidence": 0.0,
                "direction_prob": 0.5,
                "expected_return": 0.0,
            }
            if scorable and meets and model is not None:
                scored = self._score(champion, model, values)
                if scored is not None:
                    signal.update(scored)
                    signal["scorable"] = True
            signals[asset] = signal

        self._cached, self._last_built = signals, now
        self._cached_genome_id = champion.genome_id
        return signals
