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
_attach_news_features = None
_add_derived_features = None
_attach_causal_normalization = None

_TRADING_DATA = os.getenv(
    "GENOME_TRADING_DATA_ROOT",
    os.path.join("D:", os.sep, "Projects", "CoolCryptoUtilities", "data"),
)

# The news archive the GA fitted against. Same file, so live sentiment
# features match the training distribution.
_NEWS_PATH = Path(os.getenv(
    "GENOME_NEWS_PATH",
    os.path.join(_TRADING_DATA, "news", "historical_deduplicated.json"),
))

if _GENOME_REPO.is_dir():
    if str(_GENOME_REPO) not in sys.path:
        sys.path.insert(0, str(_GENOME_REPO))
    try:  # pragma: no cover - depends on sibling checkout
        from scripts.market_signal_audit import (  # type: ignore
            attach_market_breadth as _attach_market_breadth,
            continuous_features as _continuous_features,
        )
        from scripts.market_evolution_service import (  # type: ignore
            add_derived_features as _add_derived_features,
            attach_causal_normalization as _attach_causal_normalization,
            attach_news_features as _attach_news_features,
        )
        GENOME_REPO_AVAILABLE = True
    except Exception:
        GENOME_REPO_AVAILABLE = False


# 168 hourly bars of history are required: continuous_features indexes
# bars[index - 167:index + 1] for its rolling windows.
REQUIRED_HISTORY_BARS = 168


def normalize_bar(bar: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Coerce a pipeline OHLCV row into the shape the GA features expect.

    The trading store writes buy_volume/sell_volume/net_volume but no total
    ``volume``; continuous_features requires ``volume`` and raises KeyError
    without it. Total volume is buy + sell -- net_volume is the SIGNED
    imbalance and must not be substituted, or every volume-derived feature
    (volume_ratio168, flow_imbalance ...) silently inverts on sell-heavy
    bars.

    Returns None when a bar is unusable, so one malformed row cannot poison
    an asset's whole series.
    """
    if bar.get("buy_volume") is None or bar.get("sell_volume") is None:
        # Order flow is REQUIRED, never defaulted. Zeros here would make
        # flow_imbalance and friends constant, which is not the distribution
        # the genome was fitted on -- a confident signal from a model nobody
        # validated. Refusing is the safe answer.
        return None
    try:
        buy = float(bar["buy_volume"])
        sell = float(bar["sell_volume"])
        volume = bar.get("volume")
        total = float(volume) if volume is not None else buy + sell
        out = {
            "timestamp": float(bar["timestamp"]),
            "open": float(bar["open"]), "high": float(bar["high"]),
            "low": float(bar["low"]), "close": float(bar["close"]),
            "volume": total, "buy_volume": buy, "sell_volume": sell,
        }
    except (KeyError, TypeError, ValueError):
        return None
    if out["close"] <= 0 or out["timestamp"] <= 0:
        return None
    return out


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

        reference = [b for b in (normalize_bar(bar)
                                 for bar in (bars_by_asset.get(self.reference_asset) or []))
                     if b]
        if len(reference) < REQUIRED_HISTORY_BARS:
            return {}
        reference_times = [float(bar["timestamp"]) for bar in reference]

        rows: List[Dict[str, Any]] = []
        for asset, bars in bars_by_asset.items():
            series = [b for b in (normalize_bar(bar) for bar in bars) if b]
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

        # Reproduce load_dataset()'s enrichment chain in the SAME order.
        # Order matters: derived features read breadth/news outputs, and the
        # causal z-scores normalise whatever exists by then. Running these out
        # of order silently yields different values than the genome trained on.
        try:
            _attach_market_breadth(rows)
            _attach_news_features(rows, _NEWS_PATH if _NEWS_PATH.is_file() else None)
            _add_derived_features(rows)
            _attach_causal_normalization(rows)
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
