"""Build market-evolution genome features from live market data.

The genome was fitted on features produced by
``scripts/market_signal_audit.continuous_features`` in the W1z4rDV1510n repo.
Those functions are imported directly rather than reimplemented: a
reimplementation drifts, and a genome scored on a different feature
distribution than it was trained on is a model nobody validated.

Live bars MUST carry ``buy_volume`` and ``sell_volume`` alongside OHLCV:
continuous_features derives order-flow imbalance from them and raises
KeyError without them. Plain OHLCV is not sufficient.

If the GA repo is not importable the builder degrades to unavailable rather
than guessing feature values -- a wrong feature vector produces confident
nonsense, which is worse than abstaining.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# The GA repo is a sibling checkout; allow an env override for other layouts.
_GENOME_REPO = Path(
    os.getenv("W1Z4RD_REPO_ROOT", r"D:\Projects\W1z4rDV1510n")
)

GENOME_REPO_AVAILABLE = False
_continuous_features = None
_attach_market_breadth = None

if _GENOME_REPO.is_dir():
    if str(_GENOME_REPO) not in sys.path:
        sys.path.insert(0, str(_GENOME_REPO))
    try:  # pragma: no cover - depends on sibling checkout
        from scripts.market_signal_audit import (  # type: ignore
            attach_market_breadth as _attach_market_breadth,
            continuous_features as _continuous_features,
        )
        GENOME_REPO_AVAILABLE = True
    except Exception:
        GENOME_REPO_AVAILABLE = False


# 168 hourly bars of history are required: continuous_features indexes
# bars[index - 167:index + 1] for its rolling windows.
REQUIRED_HISTORY_BARS = 168


class LiveFeatureBuilder:
    """Turn live OHLCV bars into the genome's expected feature dict.

    ``bars_by_asset`` maps asset -> chronological list of hourly bars, each
    ``{"timestamp", "open", "high", "low", "close", "volume"}``. Every asset
    must be present for the same timestamp, because market breadth features
    (market_median_r6, market_breadth_r1 ...) are cross-sectional: they are
    the median across the universe at one instant. Scoring a single asset in
    isolation silently produces breadth values of its own return, which is
    not what the genome learned.
    """

    def __init__(self, reference_asset: str = "BTC") -> None:
        self.reference_asset = reference_asset.upper()

    def available(self) -> bool:
        return GENOME_REPO_AVAILABLE

    def build(
        self,
        bars_by_asset: Dict[str, Sequence[Dict[str, float]]],
        *,
        supplemental: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return asset -> feature dict for the most recent shared bar.

        Returns an empty dict when the GA repo is unavailable or history is
        too short. Abstaining is deliberate: a partial feature vector scores
        as confident nonsense.
        """
        if not GENOME_REPO_AVAILABLE:
            return {}

        reference = list(bars_by_asset.get(self.reference_asset) or [])
        if len(reference) < REQUIRED_HISTORY_BARS:
            return {}
        reference_times = [float(bar["timestamp"]) for bar in reference]

        rows: List[Dict[str, Any]] = []
        for asset, bars in bars_by_asset.items():
            series = list(bars)
            if len(series) < REQUIRED_HISTORY_BARS:
                continue
            index = len(series) - 1
            try:
                features = _continuous_features(
                    series, index, reference, reference_times,
                    (supplemental or {}).get(asset),
                )
            except Exception:
                # One bad asset must not poison the whole cross-section.
                continue
            rows.append({
                "asset": asset,
                "timestamp": float(series[index]["timestamp"]),
                "features": features,
            })

        if not rows:
            return {}

        # Cross-sectional breadth needs every asset at the same timestamp.
        try:
            _attach_market_breadth(rows)
        except Exception:
            return {}

        return {str(row["asset"]): row["features"] for row in rows}


def build_live_features(
    bars_by_asset: Dict[str, Sequence[Dict[str, float]]],
    *,
    reference_asset: str = "BTC",
    supplemental: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper around :class:`LiveFeatureBuilder`."""
    return LiveFeatureBuilder(reference_asset).build(
        bars_by_asset, supplemental=supplemental
    )
