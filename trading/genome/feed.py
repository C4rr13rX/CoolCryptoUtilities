"""Publish champion genome signals to the strategy layer on their own cadence.

The genome's features are cross-sectional: market breadth is a median across
the whole universe at one timestamp, so they cannot be built per-pair inside a
strategy's evaluate(). ``GenomeChampionStrategy`` therefore reads a prepared
dict out of ``ctx.extras["genome_signals"]`` -- and until now nothing wrote
that key, so the strategy abstained on every tick and the champion never
earned a ghost record.

Two properties of the build shape this module:

  * it needs the whole universe with >= 168 hourly bars per asset, which is
    the historical corpus the GA itself trained on, not the bot's per-symbol
    price deques; and
  * it costs ~27 seconds, which is far too slow to sit in a per-symbol tick.

So the refresh runs on a timer in a background thread and the tick only ever
reads the last published dict. A stale-but-real signal set is what the ghost
ledger needs; blocking the trading loop for half a minute is not.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading.genome.publisher import GenomeSignalPublisher

#: The corpus the GA evolves against. Reusing it is deliberate: the live
#: signal must come from the same distribution the profit factor was measured
#: on, or the ghost record does not test the genome that was validated.
MANIFEST_PATH = Path(os.getenv(
    "GENOME_MARKET_MANIFEST",
    r"D:\Projects\W1z4rDV1510n\runtime\benchmarks\market-corpus-manifest.json",
))

#: LiveFeatureBuilder needs 168 bars; keep a margin so rolling windows at the
#: tail are fully populated rather than silently truncated.
BARS_KEPT = 260
MINIMUM_BARS = 168


def _rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        return payload.get("bars") or payload.get("rows") or []
    return payload if isinstance(payload, list) else []


def load_universe_bars(manifest_path: Path = MANIFEST_PATH) -> Dict[str, List[Dict[str, Any]]]:
    """Read the selected assets' recent bars, skipping anything unusable.

    A short or missing series is dropped rather than padded: the genome was
    fitted on complete windows, and a padded one scores as confident nonsense.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    bars: Dict[str, List[Dict[str, Any]]] = {}
    for record in manifest.get("selected") or []:
        if str(record.get("selected")).lower() not in ("true", "1"):
            continue
        path = Path(str(record.get("path") or ""))
        if not path.exists():
            continue
        try:
            series = _rows(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if len(series) >= MINIMUM_BARS:
            bars[str(record.get("base_asset"))] = series[-BARS_KEPT:]
    return bars


class GenomeSignalFeed:
    """Keep ``scheduler.external_signals["genome_signals"]`` current."""

    def __init__(self, scheduler: Any, *, interval_seconds: Optional[float] = None) -> None:
        self._scheduler = scheduler
        self._publisher = GenomeSignalPublisher()
        self._interval = float(
            interval_seconds
            if interval_seconds is not None
            else os.getenv("GENOME_FEED_INTERVAL_SECONDS", "900")
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_published = 0.0
        self.last_error = ""

    def available(self) -> bool:
        return self._publisher.available()

    def refresh_once(self) -> int:
        """Rebuild and publish. Returns how many signals were published.

        Publishing an empty dict is meaningful -- it tells the strategy the
        champion is not scorable right now, which is a reason to abstain
        rather than to keep trading a stale view.
        """
        try:
            bars = load_universe_bars()
            signals = self._publisher.build(bars, force=True) if bars else {}
        except Exception as error:  # pragma: no cover - defensive
            self.last_error = repr(error)
            print(f"[genome-feed] refresh failed: {error!r}", flush=True)
            return 0
        self._scheduler.external_signals["genome_signals"] = signals
        self.last_published = time.time()
        self.last_error = ""
        scorable = sum(1 for value in signals.values() if value.get("scorable"))
        # Say something on every refresh. A feed that publishes nothing and a
        # feed that never started look identical in the logs otherwise, and
        # that ambiguity is exactly what hid this path being disconnected.
        print(f"[genome-feed] published={len(signals)} scorable={scorable} "
              f"assets={len(bars)}", flush=True)
        return len(signals)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._interval)

    def start(self) -> bool:
        """Begin refreshing in the background. False when unavailable."""
        if not self.available() or self._thread is not None:
            return False
        self._thread = threading.Thread(
            target=self._run, name="genome-signal-feed", daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

#: One feed per process. The selector builds a TradingBot per pair (up to
#: GHOST_PAIR_LIMIT of them) and every bot shares the same BusScheduler-owned
#: external_signals, so a feed per bot would run the same ~27s universe-wide
#: build dozens of times over and publish identical results. The signals are
#: cross-sectional over the whole universe; there is exactly one of them.
_SHARED_FEED: Optional["GenomeSignalFeed"] = None
_SHARED_LOCK = threading.Lock()


def ensure_feed(scheduler: Any) -> Optional["GenomeSignalFeed"]:
    """Return the process-wide feed, starting it on first call.

    Returns None when the GA repo is not importable, which is the same
    condition under which the strategy abstains anyway.
    """
    global _SHARED_FEED
    with _SHARED_LOCK:
        if _SHARED_FEED is not None:
            # Later bots share the running feed; point it at whichever
            # scheduler asked most recently so a rebuilt scheduler still
            # receives publications.
            _SHARED_FEED._scheduler = scheduler
            return _SHARED_FEED
        feed = GenomeSignalFeed(scheduler)
        if not feed.start():
            return None
        _SHARED_FEED = feed
        return feed

